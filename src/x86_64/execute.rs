//! One-byte-opcode dispatch and execution: one `match` over the opcode byte.
//!
//! Prefixes (including REX) never reach [`Cpu::dispatch`] — the step loop
//! consumes them and records their effect. Handlers pick the operand width
//! from the resolved [`OpSize`]; instructions whose 64-bit-mode default is 64
//! bits (stack operations, near branches) promote it first. Encodings that
//! 64-bit mode removed raise #UD.
//!
//! Returned cycle counts are nominal relative weights (see [`Cpu::cycles`]).

use super::OpSize::{O16, O32, O64};
use super::modrm::Operand;
use super::registers::{RFlags, cr0, reg};
use super::{Bus, Cpu, Exception, Exec};

/// The eight ALU operations selected by bits 5–3 of the opcode (or the `reg`
/// field of group 80–83), in encoding order. `CMP` (index 7) is handled by
/// the caller not writing back.
macro_rules! alu_table {
    ($($f:ident),+) => { [$(Cpu::$f),+] };
}
pub(crate) const ALU8: [fn(&mut Cpu, u8, u8) -> u8; 8] =
    alu_table!(add8, or8, adc8, sbb8, and8, sub8, xor8, sub8);
pub(crate) const ALU16: [fn(&mut Cpu, u16, u16) -> u16; 8] =
    alu_table!(add16, or16, adc16, sbb16, and16, sub16, xor16, sub16);
pub(crate) const ALU32: [fn(&mut Cpu, u32, u32) -> u32; 8] =
    alu_table!(add32, or32, adc32, sbb32, and32, sub32, xor32, sub32);
pub(crate) const ALU64: [fn(&mut Cpu, u64, u64) -> u64; 8] =
    alu_table!(add64, or64, adc64, sbb64, and64, sub64, xor64, sub64);

impl Cpu {
    /// Execute the instruction whose (non-prefix) opcode byte is `opcode`.
    /// Returns the cycles consumed.
    pub(crate) fn dispatch<B: Bus>(&mut self, bus: &mut B, opcode: u8) -> Exec<u32> {
        match opcode {
            // --- ALU: ADD OR ADC SBB AND SUB XOR CMP, six forms each --------
            0x00 | 0x08 | 0x10 | 0x18 | 0x20 | 0x28 | 0x30 | 0x38 => {
                let f = ALU8[(opcode >> 3) as usize];
                self.op_rm_r8(bus, f, opcode != 0x38)
            }
            0x01 | 0x09 | 0x11 | 0x19 | 0x21 | 0x29 | 0x31 | 0x39 => {
                self.op_rm_r(bus, (opcode >> 3) as usize, opcode != 0x39)
            }
            0x02 | 0x0A | 0x12 | 0x1A | 0x22 | 0x2A | 0x32 | 0x3A => {
                let f = ALU8[(opcode >> 3) as usize];
                self.op_r_rm8(bus, f, opcode != 0x3A)
            }
            0x03 | 0x0B | 0x13 | 0x1B | 0x23 | 0x2B | 0x33 | 0x3B => {
                self.op_r_rm(bus, (opcode >> 3) as usize, opcode != 0x3B)
            }
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 | 0x3C => {
                let f = ALU8[(opcode >> 3) as usize];
                let b = self.fetch8(bus)?;
                let r = f(self, self.gpr8(0), b);
                if opcode != 0x3C {
                    self.set_gpr8(0, r);
                }
                Ok(2)
            }
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D => {
                let i = (opcode >> 3) as usize;
                let b = self.fetch_imm(bus)?;
                let a = self.acc();
                let r = self.alu(i, a, b);
                if opcode != 0x3D {
                    self.set_acc(r);
                }
                Ok(2)
            }

            // Immediate group: 80/82 = r/m8,imm8; 81 = r/m,imm; 83 = r/m,imm8 (sign-extended)
            0x80 | 0x82 => {
                if opcode == 0x82 && self.m64 {
                    return Err(Exception::ud());
                }
                self.imm_len = 1;
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && m.sub() != 7)?;
                let a = self.read_op8(bus, op)?;
                let b = self.fetch8(bus)?;
                let r = ALU8[m.sub() as usize](self, a, b);
                if m.sub() != 7 {
                    self.write_op8(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            0x81 | 0x83 => {
                self.imm_len = if opcode == 0x83 {
                    1
                } else if self.osize == O16 {
                    2
                } else {
                    4
                };
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && m.sub() != 7)?;
                let a = self.read_op(bus, op)?;
                let b = if opcode == 0x81 {
                    self.fetch_imm(bus)?
                } else {
                    self.fetch8(bus)? as i8 as i64 as u64
                };
                let r = self.alu(m.sub() as usize, a, b);
                if m.sub() != 7 {
                    self.write_op(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }

            // --- Stack: PUSH/POP segment registers (legacy only) --------------
            0x06 | 0x0E | 0x16 | 0x1E => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                let v = self.regs.seg_sel(opcode >> 3) as u64;
                self.push(bus, v)?;
                Ok(2)
            }
            0x07 | 0x17 | 0x1F => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                let v = self.pop_sreg(bus)?;
                let idx = opcode >> 3;
                self.load_seg(bus, idx, v)?;
                if idx == reg::SS {
                    self.inhibit_interrupts = true;
                }
                Ok(7)
            }

            // --- Two-byte escape ----------------------------------------------
            0x0F => {
                let op2 = self.fetch8(bus)?;
                self.dispatch_0f(bus, op2)
            }

            // --- BCD adjustments (legacy only) ---------------------------------
            0x27 => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                self.daa();
                Ok(4)
            }
            0x2F => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                self.das();
                Ok(4)
            }
            0x37 => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                self.aaa();
                Ok(4)
            }
            0x3F => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                self.aas();
                Ok(4)
            }

            // --- INC/DEC r (legacy; REX prefixes in 64-bit mode) ---------------
            0x40..=0x47 => {
                let i = opcode & 7;
                match self.osize {
                    O16 => {
                        let r = self.inc16(self.regs.reg16(i));
                        self.regs.set_reg16(i, r);
                    }
                    _ => {
                        let r = self.inc32(self.regs.reg32(i));
                        self.regs.set_reg32(i, r);
                    }
                }
                Ok(2)
            }
            0x48..=0x4F => {
                let i = opcode & 7;
                match self.osize {
                    O16 => {
                        let r = self.dec16(self.regs.reg16(i));
                        self.regs.set_reg16(i, r);
                    }
                    _ => {
                        let r = self.dec32(self.regs.reg32(i));
                        self.regs.set_reg32(i, r);
                    }
                }
                Ok(2)
            }

            // --- PUSH/POP r --------------------------------------------------------
            0x50..=0x57 => {
                self.osize = self.stack_osize();
                let v = self.regs.reg64((opcode & 7) | self.rex_b() << 3);
                self.push(bus, v)?;
                Ok(2)
            }
            0x58..=0x5F => {
                self.osize = self.stack_osize();
                let i = (opcode & 7) | self.rex_b() << 3;
                let v = self.pop(bus)?;
                match self.osize {
                    O16 => self.regs.set_reg16(i, v as u16),
                    O32 => self.regs.set_reg32(i, v as u32),
                    O64 => self.regs.set_reg64(i, v),
                }
                Ok(4)
            }

            // --- PUSHA/POPA (legacy only) -------------------------------------
            // Both walk the frame step by step: a fault mid-way keeps the
            // memory/register effects of completed steps, but the stack
            // pointer reverts so the instruction can restart.
            0x60 => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                self.commit_on_fault = true;
                let esp0 = self.regs.gpr[reg::RSP as usize];
                let wrap = self.stack_wrap();
                let width = if self.osize == O32 { 4u64 } else { 2 };
                let bottom = self.stack_ptr().wrapping_sub(8 * width) & wrap;
                for i in (0..8u8).rev() {
                    let v = if i == reg::RSP {
                        esp0
                    } else {
                        self.regs.reg64(i)
                    };
                    let slot = bottom.wrapping_add((7 - i) as u64 * width) & wrap;
                    if self.osize == O32 {
                        self.write32(bus, reg::SS, slot, v as u32)?;
                    } else {
                        self.write16(bus, reg::SS, slot, v as u16)?;
                    }
                }
                if self.regs.seg[reg::SS as usize].db() {
                    self.regs.gpr[reg::RSP as usize] = bottom;
                } else {
                    self.regs.set_reg16(reg::RSP, bottom as u16);
                }
                Ok(18)
            }
            0x61 => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                self.commit_on_fault = true;
                let wrap = self.stack_wrap();
                let mut sp = self.stack_ptr();
                for i in (0..8u8).rev() {
                    let v = if self.osize == O32 {
                        let v = self.read32(bus, reg::SS, sp)? as u64;
                        sp = sp.wrapping_add(4) & wrap;
                        v
                    } else {
                        let v = self.read16(bus, reg::SS, sp)? as u64;
                        sp = sp.wrapping_add(2) & wrap;
                        v
                    };
                    if i != reg::RSP {
                        if self.osize == O32 {
                            self.regs.set_reg32(i, v as u32);
                        } else {
                            self.regs.set_reg16(i, v as u16);
                        }
                    }
                }
                if self.regs.seg[reg::SS as usize].db() {
                    self.regs.gpr[reg::RSP as usize] = sp;
                } else {
                    self.regs.set_reg16(reg::RSP, sp as u16);
                }
                Ok(24)
            }

            // --- BOUND (legacy) / MOVSXD-ARPL ----------------------------------
            0x62 => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                let (m, op) = self.modrm(bus)?;
                let Operand::Mem { seg, off } = op else {
                    return Err(Exception::ud());
                };
                if self.osize == O32 {
                    let idx = self.regs.reg32(m.reg()) as i32;
                    let lo = self.read32(bus, seg, off)? as i32;
                    let hi = self.read32(bus, seg, off.wrapping_add(4) & self.data_wrap())? as i32;
                    if idx < lo || idx > hi {
                        return Err(Exception::br());
                    }
                } else {
                    let idx = self.regs.reg16(m.reg()) as i16;
                    let lo = self.read16(bus, seg, off)? as i16;
                    let hi = self.read16(bus, seg, off.wrapping_add(2) & self.data_wrap())? as i16;
                    if idx < lo || idx > hi {
                        return Err(Exception::br());
                    }
                }
                Ok(10)
            }
            0x63 => {
                if self.m64 {
                    // MOVSXD r, r/m32 (r/m16 with a 16-bit operand size).
                    let (m, op) = self.modrm(bus)?;
                    match self.osize {
                        O64 => {
                            let v = self.read_op32(bus, op)? as i32 as i64 as u64;
                            self.regs.set_reg64(m.reg(), v);
                        }
                        O32 => {
                            let v = self.read_op32(bus, op)?;
                            self.regs.set_reg32(m.reg(), v);
                        }
                        O16 => {
                            let v = self.read_op16(bus, op)?;
                            self.regs.set_reg16(m.reg(), v);
                        }
                    }
                    return Ok(2);
                }
                // ARPL is protected-mode only.
                if !self.protected_mode() {
                    return Err(Exception::ud());
                }
                let (m, op) = self.modrm(bus)?;
                let dst = self.read_op16(bus, op)?;
                let src = self.regs.reg16(m.reg());
                if dst & 3 < src & 3 {
                    self.regs.rflags.insert(RFlags::ZF);
                    self.write_op16(bus, op, (dst & !3) | (src & 3))?;
                } else {
                    self.regs.rflags.remove(RFlags::ZF);
                }
                Ok(20)
            }

            // --- PUSH imm / IMUL imm ------------------------------------------------
            0x68 => {
                self.osize = self.stack_osize();
                let v = self.fetch_imm(bus)?;
                self.push(bus, v)?;
                Ok(2)
            }
            0x6A => {
                self.osize = self.stack_osize();
                let v = self.fetch8(bus)? as i8 as i64 as u64;
                self.push(bus, v)?;
                Ok(2)
            }
            0x69 | 0x6B => {
                self.imm_len = if opcode == 0x6B {
                    1
                } else if self.osize == O16 {
                    2
                } else {
                    4
                };
                let (m, op) = self.modrm(bus)?;
                let a = self.read_op(bus, op)?;
                let b = if opcode == 0x69 {
                    self.fetch_imm(bus)?
                } else {
                    self.fetch8(bus)? as i8 as i64 as u64
                };
                match self.osize {
                    O16 => {
                        let r = self.imul_trunc16(a as u16, b as u16);
                        self.regs.set_reg16(m.reg(), r);
                    }
                    O32 => {
                        let r = self.imul_trunc32(a as u32, b as u32);
                        self.regs.set_reg32(m.reg(), r);
                    }
                    O64 => {
                        let r = self.imul_trunc64(a, b);
                        self.regs.set_reg64(m.reg(), r);
                    }
                }
                Ok(if op.is_mem() { 22 } else { 20 })
            }

            // --- INS / OUTS -----------------------------------------------------------
            0x6C => self.string_op(bus, Cpu::ins8, false),
            0x6D => {
                if self.osize == O16 {
                    self.string_op(bus, Cpu::ins16, false)
                } else {
                    self.string_op(bus, Cpu::ins32, false)
                }
            }
            0x6E => self.string_op(bus, Cpu::outs8, false),
            0x6F => {
                if self.osize == O16 {
                    self.string_op(bus, Cpu::outs16, false)
                } else {
                    self.string_op(bus, Cpu::outs32, false)
                }
            }

            // --- Conditional jumps -------------------------------------------------
            0x70..=0x7F => {
                self.osize = self.branch_osize();
                let rel = self.fetch8(bus)? as i8 as i64;
                if self.cond(opcode & 0xF) {
                    self.jump_rel(rel)?;
                    Ok(7)
                } else {
                    Ok(3)
                }
            }

            // --- TEST / XCHG ----------------------------------------------------------
            0x84 => self.op_rm_r8(bus, Cpu::and8, false),
            0x85 => {
                let i = 4; // AND slot: TEST sets flags like AND
                self.op_rm_r(bus, i, false)
            }
            0x86 => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem())?;
                let a = self.read_op8(bus, op)?;
                let b = self.gpr8(m.reg());
                self.write_op8(bus, op, b)?;
                self.set_gpr8(m.reg(), a);
                Ok(if op.is_mem() { 5 } else { 3 })
            }
            0x87 => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem())?;
                let a = self.read_op(bus, op)?;
                let b = self.regs.reg64(m.reg());
                match self.osize {
                    O16 => {
                        self.write_op16(bus, op, b as u16)?;
                        self.regs.set_reg16(m.reg(), a as u16);
                    }
                    O32 => {
                        self.write_op32(bus, op, b as u32)?;
                        self.regs.set_reg32(m.reg(), a as u32);
                    }
                    O64 => {
                        self.write_op64(bus, op, b)?;
                        self.regs.set_reg64(m.reg(), a);
                    }
                }
                Ok(if op.is_mem() { 5 } else { 3 })
            }
            0x90..=0x97 => {
                // XCHG eAX, r (90 without REX.B = NOP; F3 90 = PAUSE).
                let i = (opcode & 7) | self.rex_b() << 3;
                if opcode == 0x90 && self.rex_b() == 0 {
                    return Ok(1);
                }
                match self.osize {
                    O16 => {
                        let t = self.regs.reg16(0);
                        let v = self.regs.reg16(i);
                        self.regs.set_reg16(0, v);
                        self.regs.set_reg16(i, t);
                    }
                    O32 => {
                        let t = self.regs.reg32(0);
                        let v = self.regs.reg32(i);
                        self.regs.set_reg32(0, v);
                        self.regs.set_reg32(i, t);
                    }
                    O64 => {
                        self.regs.gpr.swap(0, (i & 15) as usize);
                    }
                }
                Ok(3)
            }
            0xA8 => {
                let b = self.fetch8(bus)?;
                let a = self.gpr8(0);
                self.and8(a, b);
                Ok(2)
            }
            0xA9 => {
                let b = self.fetch_imm(bus)?;
                let a = self.acc();
                self.alu(4, a, b);
                Ok(2)
            }

            // --- MOV -----------------------------------------------------------------
            0x88 => {
                let (m, op) = self.modrm(bus)?;
                let v = self.gpr8(m.reg());
                self.write_op8(bus, op, v)?;
                Ok(2)
            }
            0x89 => {
                let (m, op) = self.modrm(bus)?;
                match self.osize {
                    O16 => {
                        let v = self.regs.reg16(m.reg());
                        self.write_op16(bus, op, v)?;
                    }
                    O32 => {
                        let v = self.regs.reg32(m.reg());
                        self.write_op32(bus, op, v)?;
                    }
                    O64 => {
                        let v = self.regs.reg64(m.reg());
                        self.write_op64(bus, op, v)?;
                    }
                }
                Ok(2)
            }
            0x8A => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op8(bus, op)?;
                self.set_gpr8(m.reg(), v);
                Ok(if op.is_mem() { 4 } else { 2 })
            }
            0x8B => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op(bus, op)?;
                self.write_reg_osize(m.reg(), v);
                Ok(if op.is_mem() { 4 } else { 2 })
            }
            0x8C => {
                // MOV r/m, Sreg (register destination zero-extends to the
                // operand size; memory always gets 16 bits).
                let (m, op) = self.modrm(bus)?;
                if m.sub() > 5 {
                    return Err(Exception::ud());
                }
                let v = self.regs.seg_sel(m.sub());
                match op {
                    Operand::Reg(i) => self.write_reg_osize(i, v as u64),
                    Operand::Mem { seg, off } => self.write16(bus, seg, off, v)?,
                }
                Ok(2)
            }
            0x8D => {
                let (m, op) = self.modrm(bus)?;
                let Operand::Mem { off, .. } = op else {
                    return Err(Exception::ud());
                };
                self.write_reg_osize(m.reg(), off);
                Ok(2)
            }
            0x8E => {
                let (m, op) = self.modrm(bus)?;
                if m.sub() == reg::CS || m.sub() > 5 {
                    return Err(Exception::ud());
                }
                let v = self.read_op16(bus, op)?;
                self.load_seg(bus, m.sub(), v)?;
                if m.sub() == reg::SS {
                    self.inhibit_interrupts = true;
                }
                Ok(if op.is_mem() { 5 } else { 2 })
            }
            0x8F => {
                // POP r/m (/0). The value pops before the EA resolves, so an
                // EA that uses (R/E)SP sees the post-pop value — hence pop
                // first, then decode.
                self.osize = self.stack_osize();
                let v = self.pop(bus)?;
                let (m, op) = self.modrm(bus)?;
                if m.sub() != 0 {
                    return Err(Exception::ud());
                }
                self.write_op(bus, op, v)?;
                Ok(if op.is_mem() { 5 } else { 4 })
            }
            0xA0 => {
                let off = self.fetch_moffs(bus)?;
                let seg = self.seg_or(reg::DS);
                let v = self.read8(bus, seg, off)?;
                self.set_gpr8(0, v);
                Ok(4)
            }
            0xA1 => {
                let off = self.fetch_moffs(bus)?;
                let seg = self.seg_or(reg::DS);
                let v = match self.osize {
                    O16 => self.read16(bus, seg, off)? as u64,
                    O32 => self.read32(bus, seg, off)? as u64,
                    O64 => self.read64(bus, seg, off)?,
                };
                self.set_acc(v);
                Ok(4)
            }
            0xA2 => {
                let off = self.fetch_moffs(bus)?;
                let seg = self.seg_or(reg::DS);
                let v = self.gpr8(0);
                self.write8(bus, seg, off, v)?;
                Ok(2)
            }
            0xA3 => {
                let off = self.fetch_moffs(bus)?;
                let seg = self.seg_or(reg::DS);
                match self.osize {
                    O16 => {
                        let v = self.regs.reg16(0);
                        self.write16(bus, seg, off, v)?;
                    }
                    O32 => {
                        let v = self.regs.reg32(0);
                        self.write32(bus, seg, off, v)?;
                    }
                    O64 => {
                        let v = self.regs.gpr[0];
                        self.write64(bus, seg, off, v)?;
                    }
                }
                Ok(2)
            }
            0xB0..=0xB7 => {
                let v = self.fetch8(bus)?;
                self.set_gpr8((opcode & 7) | self.rex_b() << 3, v);
                Ok(2)
            }
            0xB8..=0xBF => {
                let i = (opcode & 7) | self.rex_b() << 3;
                match self.osize {
                    O16 => {
                        let v = self.fetch16(bus)?;
                        self.regs.set_reg16(i, v);
                    }
                    O32 => {
                        let v = self.fetch32(bus)?;
                        self.regs.set_reg32(i, v);
                    }
                    O64 => {
                        // The one true imm64: MOV r64, imm64.
                        let v = self.fetch64(bus)?;
                        self.regs.set_reg64(i, v);
                    }
                }
                Ok(2)
            }
            0xC6 => {
                self.imm_len = 1;
                let (m, op) = self.modrm(bus)?;
                if m.sub() != 0 {
                    return Err(Exception::ud());
                }
                let v = self.fetch8(bus)?;
                self.write_op8(bus, op, v)?;
                Ok(2)
            }
            0xC7 => {
                self.imm_len = if self.osize == O16 { 2 } else { 4 };
                let (m, op) = self.modrm(bus)?;
                if m.sub() != 0 {
                    return Err(Exception::ud());
                }
                let v = self.fetch_imm(bus)?;
                self.write_op(bus, op, v)?;
                Ok(2)
            }

            // --- Wide pointer loads: LES / LDS (legacy only) --------------------
            0xC4 | 0xC5 => {
                if self.m64 {
                    // These bytes are VEX prefixes on real silicon; with no
                    // AVX support the whole space is #UD.
                    return Err(Exception::ud());
                }
                let (m, op) = self.modrm(bus)?;
                let Operand::Mem { seg, off } = op else {
                    return Err(Exception::ud());
                };
                let sreg = if opcode == 0xC4 { reg::ES } else { reg::DS };
                self.load_far_pointer(bus, sreg, m.reg(), seg, off)?;
                Ok(7)
            }

            // --- Conversions / flag transfers ---------------------------------------
            0x98 => {
                match self.osize {
                    O16 => {
                        let v = self.gpr8(0) as i8 as i16 as u16;
                        self.regs.set_reg16(0, v);
                    }
                    O32 => {
                        let v = self.regs.reg16(0) as i16 as i32 as u32;
                        self.regs.set_reg32(0, v);
                    }
                    O64 => {
                        self.regs.gpr[0] = self.regs.reg32(0) as i32 as i64 as u64;
                    }
                }
                Ok(3)
            }
            0x99 => {
                match self.osize {
                    O16 => {
                        let v = if self.regs.reg16(0) & 0x8000 != 0 {
                            0xFFFF
                        } else {
                            0
                        };
                        self.regs.set_reg16(2, v);
                    }
                    O32 => {
                        let v = if self.regs.reg32(0) & 0x8000_0000 != 0 {
                            !0u32
                        } else {
                            0
                        };
                        self.regs.set_reg32(2, v);
                    }
                    O64 => {
                        self.regs.gpr[2] = if self.regs.gpr[0] >> 63 != 0 { !0 } else { 0 };
                    }
                }
                Ok(2)
            }
            0x9E => {
                // SAHF: SF ZF AF PF CF from AH.
                let ah = (self.regs.reg8(4, false) as u32) & RFlags::STATUS & 0xFF;
                let keep = self.regs.rflags.bits() & !(RFlags::STATUS & 0xFF);
                self.regs.rflags = RFlags::from_bits_truncate(keep | ah);
                Ok(3)
            }
            0x9F => {
                // LAHF: bit 1 reads as 1, bits 3/5 as 0.
                let lo = (self.regs.rflags.bits() as u8 & 0xD5) | 0x02;
                self.regs.set_reg8(4, false, lo);
                Ok(2)
            }

            // --- PUSHF / POPF ----------------------------------------------------------
            0x9C => {
                if self.regs.rflags.contains(RFlags::VM) && self.regs.rflags.iopl() < 3 {
                    return Err(Exception::gp(0));
                }
                self.osize = self.stack_osize();
                match self.osize {
                    O16 => {
                        let v = self.regs.rflags.image16();
                        self.push16(bus, v)?;
                    }
                    O32 => {
                        let v = self.regs.rflags.image() as u32;
                        self.push32(bus, v)?;
                    }
                    O64 => {
                        let v = self.regs.rflags.image();
                        self.push64(bus, v)?;
                    }
                }
                Ok(4)
            }
            0x9D => {
                if self.regs.rflags.contains(RFlags::VM) && self.regs.rflags.iopl() < 3 {
                    return Err(Exception::gp(0));
                }
                self.osize = self.stack_osize();
                let v = self.pop(bus)?;
                let mask = self.popf_mask();
                self.regs.rflags.load(v, mask);
                Ok(5)
            }

            // --- String operations --------------------------------------------------
            0xA4 => self.string_op(bus, Cpu::movs8, false),
            0xA5 => match self.osize {
                O16 => self.string_op(bus, Cpu::movs16, false),
                O32 => self.string_op(bus, Cpu::movs32, false),
                O64 => self.string_op(bus, Cpu::movs64, false),
            },
            0xA6 => self.string_op(bus, Cpu::cmps8, true),
            0xA7 => match self.osize {
                O16 => self.string_op(bus, Cpu::cmps16, true),
                O32 => self.string_op(bus, Cpu::cmps32, true),
                O64 => self.string_op(bus, Cpu::cmps64, true),
            },
            0xAA => self.string_op(bus, Cpu::stos8, false),
            0xAB => match self.osize {
                O16 => self.string_op(bus, Cpu::stos16, false),
                O32 => self.string_op(bus, Cpu::stos32, false),
                O64 => self.string_op(bus, Cpu::stos64, false),
            },
            0xAC => self.string_op(bus, Cpu::lods8, false),
            0xAD => match self.osize {
                O16 => self.string_op(bus, Cpu::lods16, false),
                O32 => self.string_op(bus, Cpu::lods32, false),
                O64 => self.string_op(bus, Cpu::lods64, false),
            },
            0xAE => self.string_op(bus, Cpu::scas8, true),
            0xAF => match self.osize {
                O16 => self.string_op(bus, Cpu::scas16, true),
                O32 => self.string_op(bus, Cpu::scas32, true),
                O64 => self.string_op(bus, Cpu::scas64, true),
            },

            // --- Shift/rotate groups ---------------------------------------------------
            0xC0 => {
                self.imm_len = 1;
                self.shift_grp8(bus, None)
            }
            0xC1 => {
                self.imm_len = 1;
                self.shift_grp(bus, None)
            }
            0xD0 => self.shift_grp8(bus, Some(1)),
            0xD1 => self.shift_grp(bus, Some(1)),
            0xD2 => {
                let c = self.regs.reg8(1, false) as u32;
                self.shift_grp8(bus, Some(c))
            }
            0xD3 => {
                let c = self.regs.reg8(1, false) as u32;
                self.shift_grp(bus, Some(c))
            }

            // --- RET near ---------------------------------------------------------------
            0xC3 => {
                self.osize = self.branch_osize();
                let ip = self.pop(bus)?;
                self.set_ip(ip)?;
                Ok(10)
            }
            0xC2 => {
                self.osize = self.branch_osize();
                let n = self.fetch16(bus)?;
                let ip = self.pop(bus)?;
                self.set_ip(ip)?;
                self.adjust_sp(n as i64);
                Ok(10)
            }

            // --- ENTER / LEAVE ------------------------------------------------------------
            0xC8 => {
                self.osize = self.stack_osize();
                let alloc = self.fetch16(bus)? as u64;
                let level = (self.fetch8(bus)? & 0x1F) as u32;
                self.enter(bus, alloc, level)?;
                Ok(10 + 4 * level)
            }
            0xC9 => {
                // (R/E)SP <- (R/E)BP per mode/SS.B, then pop (R/E)BP.
                self.osize = self.stack_osize();
                if self.m64 {
                    self.regs.gpr[reg::RSP as usize] = self.regs.gpr[reg::RBP as usize];
                } else if self.regs.seg[reg::SS as usize].db() {
                    self.regs.gpr[reg::RSP as usize] = self.regs.reg32(reg::RBP) as u64;
                } else {
                    let bp = self.regs.reg16(reg::RBP);
                    self.regs.set_reg16(reg::RSP, bp);
                }
                let v = self.pop(bus)?;
                match self.osize {
                    O16 => self.regs.set_reg16(reg::RBP, v as u16),
                    O32 => self.regs.set_reg32(reg::RBP, v as u32),
                    O64 => self.regs.gpr[reg::RBP as usize] = v,
                }
                Ok(4)
            }

            // --- RET far ---------------------------------------------------------------------
            0xCB => self.retf(bus, 0),
            0xCA => {
                let n = self.fetch16(bus)?;
                self.retf(bus, n as u64)
            }

            // --- Software interrupts -----------------------------------------------------------
            0xCC => {
                self.software_int(bus, 3)?;
                Ok(33)
            }
            0xCD => {
                let v = self.fetch8(bus)?;
                // In V86 mode, INT n is IOPL-sensitive.
                if self.regs.rflags.contains(RFlags::VM) && self.regs.rflags.iopl() < 3 {
                    return Err(Exception::gp(0));
                }
                self.software_int(bus, v)?;
                Ok(37)
            }
            0xCE => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                if self.regs.rflags.contains(RFlags::OF) {
                    self.software_int(bus, 4)?;
                    Ok(35)
                } else {
                    Ok(3)
                }
            }
            0xF1 => {
                // ICEBP (undocumented): trap through vector 1.
                self.software_int(bus, 1)?;
                Ok(33)
            }

            // --- IRET ---------------------------------------------------------------------------
            0xCF => self.iret(bus),

            // --- AAM / AAD / SALC / XLAT ----------------------------------------------------------
            0xD4 => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                let base = self.fetch8(bus)?;
                if !self.aam(base) {
                    return Err(Exception::de());
                }
                Ok(17)
            }
            0xD5 => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                let base = self.fetch8(bus)?;
                self.aad(base);
                Ok(19)
            }
            0xD6 => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                // Undocumented SALC: AL = CF ? FF : 00.
                let v = if self.regs.rflags.contains(RFlags::CF) {
                    0xFF
                } else {
                    0
                };
                self.set_gpr8(0, v);
                Ok(3)
            }
            0xD7 => {
                let seg = self.seg_or(reg::DS);
                let bx = match self.asize {
                    O16 => self.regs.reg16(3) as u64,
                    O32 => self.regs.reg32(3) as u64,
                    O64 => self.regs.gpr[3],
                };
                let off = bx.wrapping_add(self.gpr8(0) as u64) & self.data_wrap();
                let v = self.read8(bus, seg, off)?;
                self.set_gpr8(0, v);
                Ok(5)
            }

            // --- ESC (x87 opcodes; no coprocessor state exists) ------------------------------------
            0xD8..=0xDF => {
                if self.regs.cr0 & (cr0::EM | cr0::TS) != 0 {
                    return Err(Exception::nm());
                }
                let _ = self.modrm(bus)?;
                Ok(2)
            }

            // --- Loops / IN / OUT ----------------------------------------------------------------------
            0xE0..=0xE2 => {
                self.osize = self.branch_osize();
                let rel = self.fetch8(bus)? as i8 as i64;
                let c = self.count_reg().wrapping_sub(1);
                self.set_count_reg(c);
                let go = c != 0
                    && match opcode {
                        0xE0 => !self.regs.rflags.contains(RFlags::ZF),
                        0xE1 => self.regs.rflags.contains(RFlags::ZF),
                        _ => true,
                    };
                if go {
                    self.jump_rel(rel)?;
                }
                Ok(11)
            }
            0xE3 => {
                self.osize = self.branch_osize();
                let rel = self.fetch8(bus)? as i8 as i64;
                if self.count_reg() == 0 {
                    self.jump_rel(rel)?;
                    Ok(9)
                } else {
                    Ok(5)
                }
            }
            0xE4 => {
                let p = self.fetch8(bus)? as u16;
                self.io_check(bus, p, 1)?;
                let v = bus.io_read(p);
                self.set_gpr8(0, v);
                Ok(12)
            }
            0xE5 => {
                let p = self.fetch8(bus)? as u16;
                self.in_acc(bus, p)?;
                Ok(12)
            }
            0xE6 => {
                let p = self.fetch8(bus)? as u16;
                self.io_check(bus, p, 1)?;
                bus.io_write(p, self.gpr8(0));
                Ok(10)
            }
            0xE7 => {
                let p = self.fetch8(bus)? as u16;
                self.out_acc(bus, p)?;
                Ok(10)
            }
            0xEC => {
                let p = self.regs.reg16(2);
                self.io_check(bus, p, 1)?;
                let v = bus.io_read(p);
                self.set_gpr8(0, v);
                Ok(13)
            }
            0xED => {
                let p = self.regs.reg16(2);
                self.in_acc(bus, p)?;
                Ok(13)
            }
            0xEE => {
                let p = self.regs.reg16(2);
                self.io_check(bus, p, 1)?;
                bus.io_write(p, self.gpr8(0));
                Ok(11)
            }
            0xEF => {
                let p = self.regs.reg16(2);
                self.out_acc(bus, p)?;
                Ok(11)
            }

            // --- CALL / JMP -------------------------------------------------------------------------------
            0x9A => {
                // CALL far ptr16:16/32 (legacy only).
                if self.m64 {
                    return Err(Exception::ud());
                }
                let off = self.fetch_imm(bus)?;
                let sel = self.fetch16(bus)?;
                self.call_far(bus, sel, off)?;
                Ok(17)
            }
            0xE8 => {
                self.osize = self.branch_osize();
                let rel = self.fetch_rel(bus)?;
                let ret = self.regs.rip;
                self.push(bus, ret)?;
                self.jump_rel(rel)?;
                Ok(7)
            }
            0xE9 => {
                self.osize = self.branch_osize();
                let rel = self.fetch_rel(bus)?;
                self.jump_rel(rel)?;
                Ok(7)
            }
            0xEA => {
                if self.m64 {
                    return Err(Exception::ud());
                }
                let off = self.fetch_imm(bus)?;
                let sel = self.fetch16(bus)?;
                self.jump_far(bus, sel, off)?;
                Ok(12)
            }
            0xEB => {
                self.osize = self.branch_osize();
                let rel = self.fetch8(bus)? as i8 as i64;
                self.jump_rel(rel)?;
                Ok(7)
            }

            // --- Processor control ------------------------------------------------------------------------
            0x9B => {
                // WAIT: #NM when both TS and MP are set.
                if self.regs.cr0 & cr0::TS != 0 && self.regs.cr0 & cr0::MP != 0 {
                    return Err(Exception::nm());
                }
                Ok(6)
            }
            0xF4 => {
                if self.cpl() != 0 {
                    return Err(Exception::gp(0));
                }
                self.halted = true;
                Ok(5)
            }
            0xF5 => {
                self.regs.rflags.toggle(RFlags::CF);
                Ok(2)
            }
            0xF8 => {
                self.regs.rflags.remove(RFlags::CF);
                Ok(2)
            }
            0xF9 => {
                self.regs.rflags.insert(RFlags::CF);
                Ok(2)
            }
            0xFA => {
                if self.iopl_sensitive_blocked() {
                    return Err(Exception::gp(0));
                }
                self.regs.rflags.remove(RFlags::IF);
                Ok(3)
            }
            0xFB => {
                if self.iopl_sensitive_blocked() {
                    return Err(Exception::gp(0));
                }
                self.regs.rflags.insert(RFlags::IF);
                self.inhibit_interrupts = true;
                Ok(3)
            }
            0xFC => {
                self.regs.rflags.remove(RFlags::DF);
                Ok(2)
            }
            0xFD => {
                self.regs.rflags.insert(RFlags::DF);
                Ok(2)
            }

            // --- Group F6/F7: TEST NOT NEG MUL IMUL DIV IDIV --------------------------------------------------
            0xF6 => {
                let (m, mut op) = self.modrm(bus)?;
                // TEST's imm8 follows; a RIP-relative EA must account for it.
                if m.sub() <= 1 {
                    self.fixup_rip_rel(&mut op, 1);
                }
                self.lock_check(op.is_mem() && matches!(m.sub(), 2 | 3))?;
                let v = self.read_op8(bus, op)?;
                match m.sub() {
                    0 | 1 => {
                        let b = self.fetch8(bus)?;
                        self.and8(v, b);
                        Ok(if op.is_mem() { 5 } else { 2 })
                    }
                    2 => {
                        self.write_op8(bus, op, !v)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    3 => {
                        let r = self.neg8(v);
                        self.write_op8(bus, op, r)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    4 => {
                        self.mul8(v);
                        Ok(12)
                    }
                    5 => {
                        self.imul8(v);
                        Ok(12)
                    }
                    6 => {
                        if !self.div8(v) {
                            return Err(Exception::de());
                        }
                        Ok(17)
                    }
                    _ => {
                        if !self.idiv8(v) {
                            return Err(Exception::de());
                        }
                        Ok(22)
                    }
                }
            }
            0xF7 => {
                let (m, mut op) = self.modrm(bus)?;
                if m.sub() <= 1 {
                    let n = if self.osize == O16 { 2 } else { 4 };
                    self.fixup_rip_rel(&mut op, n);
                }
                self.lock_check(op.is_mem() && matches!(m.sub(), 2 | 3))?;
                let v = self.read_op(bus, op)?;
                match m.sub() {
                    0 | 1 => {
                        let b = self.fetch_imm(bus)?;
                        self.alu(4, v, b);
                        Ok(if op.is_mem() { 5 } else { 2 })
                    }
                    2 => {
                        self.write_op(bus, op, !v)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    3 => {
                        let r = match self.osize {
                            O16 => self.neg16(v as u16) as u64,
                            O32 => self.neg32(v as u32) as u64,
                            O64 => self.neg64(v),
                        };
                        self.write_op(bus, op, r)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    4 => {
                        match self.osize {
                            O16 => self.mul16(v as u16),
                            O32 => self.mul32(v as u32),
                            O64 => self.mul64(v),
                        }
                        Ok(20)
                    }
                    5 => {
                        match self.osize {
                            O16 => self.imul16(v as u16),
                            O32 => self.imul32(v as u32),
                            O64 => self.imul64(v),
                        }
                        Ok(20)
                    }
                    6 => {
                        let ok = match self.osize {
                            O16 => self.div16(v as u16),
                            O32 => self.div32(v as u32),
                            O64 => self.div64(v),
                        };
                        if !ok {
                            return Err(Exception::de());
                        }
                        Ok(30)
                    }
                    _ => {
                        let ok = match self.osize {
                            O16 => self.idiv16(v as u16),
                            O32 => self.idiv32(v as u32),
                            O64 => self.idiv64(v),
                        };
                        if !ok {
                            return Err(Exception::de());
                        }
                        Ok(35)
                    }
                }
            }

            // --- Group FE: INC/DEC r/m8 ------------------------------------------------------------------------
            0xFE => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && m.sub() < 2)?;
                match m.sub() {
                    0 | 1 => {
                        let v = self.read_op8(bus, op)?;
                        let r = if m.sub() == 0 {
                            self.inc8(v)
                        } else {
                            self.dec8(v)
                        };
                        self.write_op8(bus, op, r)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    _ => Err(Exception::ud()),
                }
            }

            // --- Group FF: INC DEC CALL CALL-far JMP JMP-far PUSH ------------------------------------------------
            0xFF => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && m.sub() < 2)?;
                match m.sub() {
                    0 | 1 => {
                        let v = self.read_op(bus, op)?;
                        let r = match (self.osize, m.sub()) {
                            (O16, 0) => self.inc16(v as u16) as u64,
                            (O16, _) => self.dec16(v as u16) as u64,
                            (O32, 0) => self.inc32(v as u32) as u64,
                            (O32, _) => self.dec32(v as u32) as u64,
                            (O64, 0) => self.inc64(v),
                            (O64, _) => self.dec64(v),
                        };
                        self.write_op(bus, op, r)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    2 => {
                        self.osize = self.branch_osize();
                        let v = self.read_op(bus, op)?;
                        let ret = self.regs.rip;
                        self.push(bus, ret)?;
                        self.set_ip(v)?;
                        Ok(if op.is_mem() { 10 } else { 7 })
                    }
                    3 => {
                        let Operand::Mem { seg, off } = op else {
                            return Err(Exception::ud());
                        };
                        let (sel, dst) = self.read_far_pointer(bus, seg, off)?;
                        self.call_far(bus, sel, dst)?;
                        Ok(22)
                    }
                    4 => {
                        self.osize = self.branch_osize();
                        let v = self.read_op(bus, op)?;
                        self.set_ip(v)?;
                        Ok(if op.is_mem() { 10 } else { 7 })
                    }
                    5 => {
                        let Operand::Mem { seg, off } = op else {
                            return Err(Exception::ud());
                        };
                        let (sel, dst) = self.read_far_pointer(bus, seg, off)?;
                        self.jump_far(bus, sel, dst)?;
                        Ok(17)
                    }
                    6 => {
                        // PUSH r/m: the operand reads at the *stack* size in
                        // 64-bit mode (no 32-bit encoding exists).
                        self.osize = self.stack_osize();
                        let v = self.read_op(bus, op)?;
                        self.push(bus, v)?;
                        Ok(if op.is_mem() { 5 } else { 2 })
                    }
                    _ => Err(Exception::ud()),
                }
            }

            // Prefixes are consumed by the step loop.
            _ => unreachable!("prefix byte {opcode:#04X} reached dispatch"),
        }
    }

    // --- Dispatch helpers -------------------------------------------------------

    /// The accumulator at the current operand size (zero-extended).
    #[inline]
    pub(crate) fn acc(&self) -> u64 {
        match self.osize {
            O16 => self.regs.reg16(0) as u64,
            O32 => self.regs.reg32(0) as u64,
            O64 => self.regs.gpr[0],
        }
    }

    /// Write the accumulator at the current operand size.
    #[inline]
    pub(crate) fn set_acc(&mut self, v: u64) {
        match self.osize {
            O16 => self.regs.set_reg16(0, v as u16),
            O32 => self.regs.set_reg32(0, v as u32),
            O64 => self.regs.gpr[0] = v,
        }
    }

    /// Write register `i` at the current operand size.
    #[inline]
    pub(crate) fn write_reg_osize(&mut self, i: u8, v: u64) {
        match self.osize {
            O16 => self.regs.set_reg16(i, v as u16),
            O32 => self.regs.set_reg32(i, v as u32),
            O64 => self.regs.set_reg64(i, v),
        }
    }

    /// Run ALU operation `i` (see [`ALU64`]) at the current operand size on
    /// zero-extended operands, returning the zero-extended result.
    #[inline]
    pub(crate) fn alu(&mut self, i: usize, a: u64, b: u64) -> u64 {
        match self.osize {
            O16 => ALU16[i](self, a as u16, b as u16) as u64,
            O32 => ALU32[i](self, a as u32, b as u32) as u64,
            O64 => ALU64[i](self, a, b),
        }
    }

    /// Compensate a RIP-relative memory operand for `n` immediate bytes when
    /// the immediate's presence was not knowable before ModRM decode
    /// (`F6`/`F7 TEST`): the dispatcher left `imm_len` at 0, so the EA came
    /// out `n` bytes short.
    fn fixup_rip_rel(&mut self, op: &mut Operand, n: u64) {
        if self.used_rip_rel
            && let Operand::Mem { off, .. } = op
        {
            *off = off.wrapping_add(n);
        }
    }

    /// Evaluate condition code `n` (the low nibble of a Jcc/SETcc opcode).
    pub(crate) fn cond(&self, n: u8) -> bool {
        let f = self.regs.rflags;
        let r = match n >> 1 {
            0 => f.contains(RFlags::OF),
            1 => f.contains(RFlags::CF),
            2 => f.contains(RFlags::ZF),
            3 => f.contains(RFlags::CF) || f.contains(RFlags::ZF),
            4 => f.contains(RFlags::SF),
            5 => f.contains(RFlags::PF),
            6 => f.contains(RFlags::SF) != f.contains(RFlags::OF),
            _ => f.contains(RFlags::ZF) || (f.contains(RFlags::SF) != f.contains(RFlags::OF)),
        };
        r != (n & 1 != 0)
    }

    /// Fetch a near-branch displacement: rel32 (sign-extended) at 32/64-bit
    /// operand size, rel16 at 16-bit.
    #[inline]
    pub(crate) fn fetch_rel<B: Bus>(&mut self, bus: &mut B) -> Exec<i64> {
        if self.osize == O16 {
            Ok(self.fetch16(bus)? as i16 as i64)
        } else {
            Ok(self.fetch32(bus)? as i32 as i64)
        }
    }

    /// Relative jump: add `rel` to RIP, truncating per operand size.
    /// Illegal targets fault at the transfer, before it commits.
    #[inline]
    pub(crate) fn jump_rel(&mut self, rel: i64) -> Exec<()> {
        let ip = self.regs.rip.wrapping_add(rel as u64);
        self.set_ip(ip)
    }

    /// Absolute near jump target (RET/JMP/CALL indirect), truncated per
    /// operand size and checked against the CS limit (legacy) or
    /// canonicality (64-bit mode).
    #[inline]
    pub(crate) fn set_ip(&mut self, ip: u64) -> Exec<()> {
        let ip = match self.osize {
            O16 => ip & 0xFFFF,
            O32 => ip & 0xFFFF_FFFF,
            O64 => ip,
        };
        if self.m64 {
            if !Self::canonical(ip) {
                return Err(Exception::gp(0));
            }
        } else if ip > self.regs.seg[reg::CS as usize].limit as u64 {
            return Err(Exception::gp(0));
        }
        self.regs.rip = ip;
        Ok(())
    }

    /// Fetch a `moffs` direct address (width = address size).
    #[inline]
    fn fetch_moffs<B: Bus>(&mut self, bus: &mut B) -> Exec<u64> {
        match self.asize {
            O16 => Ok(self.fetch16(bus)? as u64),
            O32 => Ok(self.fetch32(bus)? as u64),
            O64 => self.fetch64(bus),
        }
    }

    /// The set of RFLAGS bits `POPF`/`IRET` may modify in the current mode.
    pub(crate) fn popf_mask(&self) -> u32 {
        let mut mask = 0x0000_7FD5u32; // all defined 16-bit flags incl. IOPL/NT
        if self.osize != O16 {
            // VM is never writable this way; AC and ID need a wide operand.
            mask |= (RFlags::RF | RFlags::AC | RFlags::ID).bits();
        }
        if self.protected_mode() {
            let cpl = self.cpl();
            if cpl > 0 {
                mask &= !RFlags::IOPL.bits();
                if cpl as u32 > (self.regs.rflags.bits() >> 12) & 3 {
                    mask &= !RFlags::IF.bits();
                }
            }
        } else if self.regs.rflags.contains(RFlags::VM) {
            // The caller already required IOPL == 3; IOPL itself is fixed.
            mask &= !RFlags::IOPL.bits();
        }
        mask
    }

    /// CLI/STI legality: IOPL-sensitive in protected and V86 modes.
    fn iopl_sensitive_blocked(&self) -> bool {
        if self.regs.cr0 & cr0::PE == 0 {
            return false;
        }
        self.cpl() > self.regs.rflags.iopl()
    }

    /// `op r/m, r` at the current operand size (ALU table index `i`).
    #[inline(always)]
    fn op_rm_r<B: Bus>(&mut self, bus: &mut B, i: usize, wb: bool) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        self.lock_check(wb && op.is_mem())?;
        let a = self.read_op(bus, op)?;
        let b = self.regs.reg64(m.reg());
        let r = self.alu(i, a, b);
        if wb {
            self.write_op(bus, op, r)?;
        }
        Ok(if op.is_mem() { 7 } else { 2 })
    }

    /// `op r, r/m` at the current operand size — destination is the register.
    #[inline(always)]
    fn op_r_rm<B: Bus>(&mut self, bus: &mut B, i: usize, wb: bool) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let a = match self.osize {
            O16 => self.regs.reg16(m.reg()) as u64,
            O32 => self.regs.reg32(m.reg()) as u64,
            O64 => self.regs.reg64(m.reg()),
        };
        let b = self.read_op(bus, op)?;
        let r = self.alu(i, a, b);
        if wb {
            self.write_reg_osize(m.reg(), r);
        }
        Ok(if op.is_mem() { 6 } else { 2 })
    }

    /// `op r/m8, r8` — destination is the r/m operand. `wb == false` for CMP/TEST.
    #[inline(always)]
    fn op_rm_r8<B: Bus>(
        &mut self,
        bus: &mut B,
        f: fn(&mut Cpu, u8, u8) -> u8,
        wb: bool,
    ) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        self.lock_check(wb && op.is_mem())?;
        let a = self.read_op8(bus, op)?;
        let b = self.gpr8(m.reg());
        let r = f(self, a, b);
        if wb {
            self.write_op8(bus, op, r)?;
        }
        Ok(if op.is_mem() { 7 } else { 2 })
    }

    /// `op r8, r/m8` — destination is the register operand.
    #[inline(always)]
    fn op_r_rm8<B: Bus>(
        &mut self,
        bus: &mut B,
        f: fn(&mut Cpu, u8, u8) -> u8,
        wb: bool,
    ) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let a = self.gpr8(m.reg());
        let b = self.read_op8(bus, op)?;
        let r = f(self, a, b);
        if wb {
            self.set_gpr8(m.reg(), r);
        }
        Ok(if op.is_mem() { 6 } else { 2 })
    }

    /// Shift/rotate group on a byte operand. `count` is `Some` for the
    /// 1-bit/CL forms (D0/D2); `None` means an imm8 count follows ModRM (C0).
    fn shift_grp8<B: Bus>(&mut self, bus: &mut B, count: Option<u32>) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let v = self.read_op8(bus, op)?;
        let n = match count {
            Some(n) => n,
            None => self.fetch8(bus)? as u32,
        };
        let r = self.shift_dispatch8(m.sub(), v, n);
        self.write_op8(bus, op, r)?;
        Ok(if op.is_mem() { 7 } else { 3 })
    }

    fn shift_dispatch8(&mut self, sub: u8, v: u8, n: u32) -> u8 {
        match sub {
            0 => self.rol8(v, n),
            1 => self.ror8(v, n),
            2 => self.rcl8(v, n),
            3 => self.rcr8(v, n),
            4 => self.shl8(v, n),
            5 => self.shr8(v, n),
            6 => self.shl8(v, n), // /6 aliases SHL
            _ => self.sar8(v, n),
        }
    }

    /// Shift/rotate group at the current operand size (see [`Cpu::shift_grp8`]).
    fn shift_grp<B: Bus>(&mut self, bus: &mut B, count: Option<u32>) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let v = self.read_op(bus, op)?;
        let n = match count {
            Some(n) => n,
            None => self.fetch8(bus)? as u32,
        };
        let r = self.shift_dispatch(m.sub(), v, n);
        self.write_op(bus, op, r)?;
        Ok(if op.is_mem() { 7 } else { 3 })
    }

    fn shift_dispatch(&mut self, sub: u8, v: u64, n: u32) -> u64 {
        match self.osize {
            O16 => {
                let v = v as u16;
                (match sub {
                    0 => self.rol16(v, n),
                    1 => self.ror16(v, n),
                    2 => self.rcl16(v, n),
                    3 => self.rcr16(v, n),
                    4 => self.shl16(v, n),
                    5 => self.shr16(v, n),
                    6 => self.shl16(v, n),
                    _ => self.sar16(v, n),
                }) as u64
            }
            O32 => {
                let v = v as u32;
                (match sub {
                    0 => self.rol32(v, n),
                    1 => self.ror32(v, n),
                    2 => self.rcl32(v, n),
                    3 => self.rcr32(v, n),
                    4 => self.shl32(v, n),
                    5 => self.shr32(v, n),
                    6 => self.shl32(v, n),
                    _ => self.sar32(v, n),
                }) as u64
            }
            O64 => match sub {
                0 => self.rol64(v, n),
                1 => self.ror64(v, n),
                2 => self.rcl64(v, n),
                3 => self.rcr64(v, n),
                4 => self.shl64(v, n),
                5 => self.shr64(v, n),
                6 => self.shl64(v, n),
                _ => self.sar64(v, n),
            },
        }
    }

    /// Read an `offset:selector` far pointer from memory (offset width per
    /// operand size — `m16:16`, `m16:32` or `m16:64`).
    pub(crate) fn read_far_pointer<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u64,
    ) -> Exec<(u16, u64)> {
        let wrap = self.data_wrap();
        match self.osize {
            O16 => {
                let dst = self.read16(bus, seg, off)? as u64;
                let sel = self.read16(bus, seg, off.wrapping_add(2) & wrap)?;
                Ok((sel, dst))
            }
            O32 => {
                let dst = self.read32(bus, seg, off)? as u64;
                let sel = self.read16(bus, seg, off.wrapping_add(4) & wrap)?;
                Ok((sel, dst))
            }
            O64 => {
                let dst = self.read64(bus, seg, off)?;
                let sel = self.read16(bus, seg, off.wrapping_add(8) & wrap)?;
                Ok((sel, dst))
            }
        }
    }

    /// `LES/LDS/LFS/LGS/LSS r, m`: load the register then the segment.
    pub(crate) fn load_far_pointer<B: Bus>(
        &mut self,
        bus: &mut B,
        sreg: u8,
        r: u8,
        seg: u8,
        off: u64,
    ) -> Exec<()> {
        let (sel, dst) = self.read_far_pointer(bus, seg, off)?;
        self.load_seg(bus, sreg, sel)?;
        self.write_reg_osize(r, dst);
        if sreg == reg::SS {
            self.inhibit_interrupts = true;
        }
        Ok(())
    }

    /// ENTER: build a stack frame with `level` nested frame pointers.
    fn enter<B: Bus>(&mut self, bus: &mut B, alloc: u64, level: u32) -> Exec<()> {
        let bp = self.regs.gpr[reg::RBP as usize];
        self.push(bus, bp)?;
        let frame = self.stack_ptr();
        if level > 0 {
            for _ in 1..level {
                // Walk the display chain: BP steps down one slot per level.
                match self.osize {
                    O16 => {
                        let nbp = self.regs.reg16(reg::RBP).wrapping_sub(2);
                        self.regs.set_reg16(reg::RBP, nbp);
                        let v = self.read16(bus, reg::SS, nbp as u64)?;
                        self.push16(bus, v)?;
                    }
                    O32 => {
                        let nbp = self.regs.reg32(reg::RBP).wrapping_sub(4);
                        self.regs.set_reg32(reg::RBP, nbp);
                        let v = self.read32(bus, reg::SS, nbp as u64)?;
                        self.push32(bus, v)?;
                    }
                    O64 => {
                        let nbp = self.regs.gpr[reg::RBP as usize].wrapping_sub(8);
                        self.regs.gpr[reg::RBP as usize] = nbp;
                        let v = self.read64(bus, reg::SS, nbp)?;
                        self.push64(bus, v)?;
                    }
                }
            }
            self.push(bus, frame)?;
        }
        match self.osize {
            O16 => self.regs.set_reg16(reg::RBP, frame as u16),
            O32 => self.regs.set_reg32(reg::RBP, frame as u32),
            O64 => self.regs.gpr[reg::RBP as usize] = frame,
        }
        self.adjust_sp(-(alloc as i64));
        Ok(())
    }

    // --- Port I/O (wide access composed from the byte-wide bus) -------------------

    /// IN eAX/AX/EAX, port at the operand size (64-bit forms do not exist;
    /// REX.W is ignored, as on hardware).
    fn in_acc<B: Bus>(&mut self, bus: &mut B, p: u16) -> Exec<()> {
        if self.osize == O16 {
            self.io_check(bus, p, 2)?;
            let v = self.io_read16(bus, p);
            self.regs.set_reg16(0, v);
        } else {
            self.io_check(bus, p, 4)?;
            let v = self.io_read32(bus, p);
            self.regs.set_reg32(0, v);
        }
        Ok(())
    }

    /// OUT port, eAX at the operand size.
    fn out_acc<B: Bus>(&mut self, bus: &mut B, p: u16) -> Exec<()> {
        if self.osize == O16 {
            self.io_check(bus, p, 2)?;
            let v = self.regs.reg16(0);
            self.io_write16(bus, p, v);
        } else {
            self.io_check(bus, p, 4)?;
            let v = self.regs.reg32(0);
            self.io_write32(bus, p, v);
        }
        Ok(())
    }

    pub(crate) fn io_read16<B: Bus>(&mut self, bus: &mut B, port: u16) -> u16 {
        bus.io_read(port) as u16 | (bus.io_read(port.wrapping_add(1)) as u16) << 8
    }

    pub(crate) fn io_write16<B: Bus>(&mut self, bus: &mut B, port: u16, v: u16) {
        bus.io_write(port, v as u8);
        bus.io_write(port.wrapping_add(1), (v >> 8) as u8);
    }

    pub(crate) fn io_read32<B: Bus>(&mut self, bus: &mut B, port: u16) -> u32 {
        self.io_read16(bus, port) as u32 | (self.io_read16(bus, port.wrapping_add(2)) as u32) << 16
    }

    pub(crate) fn io_write32<B: Bus>(&mut self, bus: &mut B, port: u16, v: u32) {
        self.io_write16(bus, port, v as u16);
        self.io_write16(bus, port.wrapping_add(2), (v >> 16) as u16);
    }

    // --- String operations ----------------------------------------------------------

    /// (R/E)CX per address size.
    #[inline]
    fn count_reg(&self) -> u64 {
        match self.asize {
            O16 => self.regs.gpr[1] & 0xFFFF,
            O32 => self.regs.gpr[1] & 0xFFFF_FFFF,
            O64 => self.regs.gpr[1],
        }
    }

    #[inline]
    fn set_count_reg(&mut self, v: u64) {
        match self.asize {
            O16 => self.regs.set_reg16(1, v as u16),
            O32 => self.regs.set_reg32(1, v as u32),
            O64 => self.regs.gpr[1] = v,
        }
    }

    /// (R/E)SI / (R/E)DI per address size.
    #[inline]
    fn index_reg(&self, i: u8) -> u64 {
        let v = self.regs.gpr[i as usize];
        match self.asize {
            O16 => v & 0xFFFF,
            O32 => v & 0xFFFF_FFFF,
            O64 => v,
        }
    }

    #[inline]
    fn set_index_reg(&mut self, i: u8, v: u64) {
        match self.asize {
            O16 => self.regs.set_reg16(i, v as u16),
            O32 => self.regs.set_reg32(i, v as u32),
            O64 => self.regs.gpr[i as usize] = v,
        }
    }

    /// ±element-size depending on DF.
    #[inline]
    fn delta(&self, n: u64) -> u64 {
        if self.regs.rflags.contains(RFlags::DF) {
            n.wrapping_neg()
        } else {
            n
        }
    }

    /// Advance (R/E)SI by ±n.
    #[inline]
    fn step_si(&mut self, n: u64) {
        let d = self.delta(n);
        let v = self.index_reg(reg::RSI).wrapping_add(d);
        self.set_index_reg(reg::RSI, v);
    }

    /// Advance (R/E)DI by ±n.
    #[inline]
    fn step_di(&mut self, n: u64) {
        let d = self.delta(n);
        let v = self.index_reg(reg::RDI).wrapping_add(d);
        self.set_index_reg(reg::RDI, v);
    }

    /// Run one string primitive, honoring an active REP/REPE/REPNE prefix.
    /// `cmp` marks CMPS/SCAS, whose repetition also terminates on the ZF
    /// condition. Faults mid-iteration keep completed progress (RIP rewinds
    /// to the instruction, prefixes included, so it restarts).
    fn string_op<B: Bus>(
        &mut self,
        bus: &mut B,
        one: fn(&mut Cpu, &mut B) -> Exec<()>,
        cmp: bool,
    ) -> Exec<u32> {
        self.commit_on_fault = true;
        let Some(cont_on_zf) = self.rep else {
            one(self, bus)?;
            return Ok(7);
        };
        let mut cycles = 5;
        while self.count_reg() != 0 {
            one(self, bus)?;
            self.set_count_reg(self.count_reg().wrapping_sub(1));
            cycles += 4;
            if cmp && self.regs.rflags.contains(RFlags::ZF) != cont_on_zf {
                break;
            }
            // Hardware recognizes interrupts between iterations; a single
            // REP can run for 2^64 of them, so yield by rewinding RIP to the
            // prefix. The next `step()` takes the interrupt, and the
            // handler's IRET resumes the REP with the already-decremented
            // count and index registers.
            if self.count_reg() != 0 && self.interrupt_pending() {
                self.regs.rip = self.start_rip;
                break;
            }
        }
        Ok(cycles)
    }
}

/// Generate the per-width string primitives (one iteration each; the REP
/// loop lives in [`Cpu::string_op`]).
macro_rules! string_ops {
    ($t:ty, $n:literal, $movs:ident, $cmps:ident, $stos:ident, $lods:ident, $scas:ident,
     $read:ident, $write:ident, $sub:ident, $set_acc:ident) => {
        impl Cpu {
            fn $movs<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let seg = self.seg_or(reg::DS);
                let si = self.index_reg(reg::RSI);
                let v = self.$read(bus, seg, si)?;
                let di = self.index_reg(reg::RDI);
                self.$write(bus, reg::ES, di, v)?;
                self.step_si($n);
                self.step_di($n);
                Ok(())
            }

            fn $cmps<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let seg = self.seg_or(reg::DS);
                let a = self.$read(bus, seg, self.index_reg(reg::RSI))?;
                let b = self.$read(bus, reg::ES, self.index_reg(reg::RDI))?;
                self.$sub(a, b);
                self.step_si($n);
                self.step_di($n);
                Ok(())
            }

            fn $stos<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let v = self.regs.gpr[0] as $t;
                let di = self.index_reg(reg::RDI);
                self.$write(bus, reg::ES, di, v)?;
                self.step_di($n);
                Ok(())
            }

            fn $lods<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let seg = self.seg_or(reg::DS);
                let v = self.$read(bus, seg, self.index_reg(reg::RSI))?;
                self.regs.$set_acc(0, v);
                self.step_si($n);
                Ok(())
            }

            fn $scas<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let b = self.$read(bus, reg::ES, self.index_reg(reg::RDI))?;
                let a = self.regs.gpr[0] as $t;
                self.$sub(a, b);
                self.step_di($n);
                Ok(())
            }
        }
    };
}

string_ops!(
    u16, 2, movs16, cmps16, stos16, lods16, scas16, read16, write16, sub16, set_reg16
);
string_ops!(
    u32, 4, movs32, cmps32, stos32, lods32, scas32, read32, write32, sub32, set_reg32
);
string_ops!(
    u64, 8, movs64, cmps64, stos64, lods64, scas64, read64, write64, sub64, set_reg64
);

impl Cpu {
    // The 8-bit string primitives are hand-written: the accumulator write of
    // LODSB honors no REX high-byte rule (it is always AL), so the generic
    // macro shape fits, but the 8-bit register accessor takes the extra
    // `rex` argument.

    fn movs8<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let seg = self.seg_or(reg::DS);
        let si = self.index_reg(reg::RSI);
        let v = self.read8(bus, seg, si)?;
        let di = self.index_reg(reg::RDI);
        self.write8(bus, reg::ES, di, v)?;
        self.step_si(1);
        self.step_di(1);
        Ok(())
    }

    fn cmps8<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let seg = self.seg_or(reg::DS);
        let a = self.read8(bus, seg, self.index_reg(reg::RSI))?;
        let b = self.read8(bus, reg::ES, self.index_reg(reg::RDI))?;
        self.sub8(a, b);
        self.step_si(1);
        self.step_di(1);
        Ok(())
    }

    fn stos8<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let v = self.regs.gpr[0] as u8;
        let di = self.index_reg(reg::RDI);
        self.write8(bus, reg::ES, di, v)?;
        self.step_di(1);
        Ok(())
    }

    fn lods8<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let seg = self.seg_or(reg::DS);
        let v = self.read8(bus, seg, self.index_reg(reg::RSI))?;
        self.regs.set_reg8(0, false, v);
        self.step_si(1);
        Ok(())
    }

    fn scas8<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let b = self.read8(bus, reg::ES, self.index_reg(reg::RDI))?;
        let a = self.regs.gpr[0] as u8;
        self.sub8(a, b);
        self.step_di(1);
        Ok(())
    }

    /// INS always stores to ES:(R/E)DI — no override.
    fn ins8<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let port = self.regs.reg16(2);
        self.io_check(bus, port, 1)?;
        let v = bus.io_read(port);
        let di = self.index_reg(reg::RDI);
        self.write8(bus, reg::ES, di, v)?;
        self.step_di(1);
        Ok(())
    }

    fn ins16<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let port = self.regs.reg16(2);
        self.io_check(bus, port, 2)?;
        let v = self.io_read16(bus, port);
        let di = self.index_reg(reg::RDI);
        self.write16(bus, reg::ES, di, v)?;
        self.step_di(2);
        Ok(())
    }

    fn ins32<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let port = self.regs.reg16(2);
        self.io_check(bus, port, 4)?;
        let v = self.io_read32(bus, port);
        let di = self.index_reg(reg::RDI);
        self.write32(bus, reg::ES, di, v)?;
        self.step_di(4);
        Ok(())
    }

    fn outs8<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let port = self.regs.reg16(2);
        self.io_check(bus, port, 1)?;
        let seg = self.seg_or(reg::DS);
        let v = self.read8(bus, seg, self.index_reg(reg::RSI))?;
        bus.io_write(port, v);
        self.step_si(1);
        Ok(())
    }

    fn outs16<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let port = self.regs.reg16(2);
        self.io_check(bus, port, 2)?;
        let seg = self.seg_or(reg::DS);
        let v = self.read16(bus, seg, self.index_reg(reg::RSI))?;
        self.io_write16(bus, port, v);
        self.step_si(2);
        Ok(())
    }

    fn outs32<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
        let port = self.regs.reg16(2);
        self.io_check(bus, port, 4)?;
        let seg = self.seg_or(reg::DS);
        let v = self.read32(bus, seg, self.index_reg(reg::RSI))?;
        self.io_write32(bus, port, v);
        self.step_si(4);
        Ok(())
    }
}

impl Cpu {
    /// Push at the *operand* size (far-call frames, which do not get the
    /// stack-size promotion — a 64-bit push needs REX.W).
    pub(crate) fn push_op<B: Bus>(&mut self, bus: &mut B, v: u64) -> Exec<()> {
        match self.osize {
            O16 => self.push16(bus, v as u16),
            O32 => self.push32(bus, v as u32),
            O64 => self.push64(bus, v),
        }
    }

    /// Pop at the *operand* size (RETF/IRET frames).
    pub(crate) fn pop_op<B: Bus>(&mut self, bus: &mut B) -> Exec<u64> {
        match self.osize {
            O16 => Ok(self.pop16(bus)? as u64),
            O32 => Ok(self.pop32(bus)? as u64),
            O64 => self.pop64(bus),
        }
    }
}
