//! One-byte-opcode dispatch and execution: one `match` over the opcode byte.
//!
//! Prefixes never reach [`Cpu::dispatch`] — the step loop consumes them and
//! records their effect (`seg_override`, `rep`, `lock`, operand/address
//! size). Unlike the 8086, the 386 faults on undefined encodings (#UD), and
//! every handler is fallible: memory operands go through segment limit
//! checks and paging.
//!
//! Returned cycle counts are approximate documented 386 base timings; the
//! prefetch queue and dynamic bus sizing are not modeled.

use super::modrm::Operand;
use super::registers::{EFlags, cr0, reg};
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

impl Cpu {
    /// Execute the instruction whose (non-prefix) opcode byte is `opcode`.
    /// Returns the cycles consumed.
    pub(crate) fn dispatch<B: Bus>(&mut self, bus: &mut B, opcode: u8) -> Exec<u32> {
        self.stat(|s| s.opcode_hist[opcode as usize] += 1);
        match opcode {
            // --- ALU: ADD OR ADC SBB AND SUB XOR CMP, six forms each --------
            0x00 | 0x08 | 0x10 | 0x18 | 0x20 | 0x28 | 0x30 | 0x38 => {
                let f = ALU8[(opcode >> 3) as usize];
                self.op_rm_r8(bus, f, opcode != 0x38)
            }
            0x01 | 0x09 | 0x11 | 0x19 | 0x21 | 0x29 | 0x31 | 0x39 => {
                let wb = opcode != 0x39;
                if self.osize32 {
                    let f = ALU32[(opcode >> 3) as usize];
                    self.op_rm_r32(bus, f, wb)
                } else {
                    let f = ALU16[(opcode >> 3) as usize];
                    self.op_rm_r16(bus, f, wb)
                }
            }
            0x02 | 0x0A | 0x12 | 0x1A | 0x22 | 0x2A | 0x32 | 0x3A => {
                let f = ALU8[(opcode >> 3) as usize];
                self.op_r_rm8(bus, f, opcode != 0x3A)
            }
            0x03 | 0x0B | 0x13 | 0x1B | 0x23 | 0x2B | 0x33 | 0x3B => {
                let wb = opcode != 0x3B;
                if self.osize32 {
                    let f = ALU32[(opcode >> 3) as usize];
                    self.op_r_rm32(bus, f, wb)
                } else {
                    let f = ALU16[(opcode >> 3) as usize];
                    self.op_r_rm16(bus, f, wb)
                }
            }
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 | 0x3C => {
                let f = ALU8[(opcode >> 3) as usize];
                let b = self.fetch8(bus)?;
                let r = f(self, self.regs.reg8(0), b);
                if opcode != 0x3C {
                    self.regs.set_reg8(0, r);
                }
                Ok(2)
            }
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D => {
                let i = (opcode >> 3) as usize;
                if self.osize32 {
                    let b = self.fetch32(bus)?;
                    let r = ALU32[i](self, self.regs.gpr[0], b);
                    if opcode != 0x3D {
                        self.regs.gpr[0] = r;
                    }
                } else {
                    let b = self.fetch16(bus)?;
                    let r = ALU16[i](self, self.regs.reg16(0), b);
                    if opcode != 0x3D {
                        self.regs.set_reg16(0, r);
                    }
                }
                Ok(2)
            }

            // Immediate group: 80/82 = r/m8,imm8; 81 = r/m,imm; 83 = r/m,imm8 (sign-extended)
            0x80 | 0x82 => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && m.reg() != 7)?;
                let a = self.read_op8(bus, op)?;
                let b = self.fetch8(bus)?;
                let r = ALU8[m.reg() as usize](self, a, b);
                if m.reg() != 7 {
                    self.write_op8(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            0x81 | 0x83 => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && m.reg() != 7)?;
                let a = self.read_op(bus, op)?;
                let b = if opcode == 0x81 {
                    self.fetch_imm(bus)?
                } else {
                    self.fetch8(bus)? as i8 as i32 as u32
                };
                let r = if self.osize32 {
                    ALU32[m.reg() as usize](self, a, b)
                } else {
                    ALU16[m.reg() as usize](self, a as u16, b as u16) as u32
                };
                if m.reg() != 7 {
                    self.write_op(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }

            // --- Stack: PUSH/POP segment registers ---------------------------
            0x06 | 0x0E | 0x16 | 0x1E => {
                let v = self.regs.seg_sel(opcode >> 3) as u32;
                self.push(bus, v)?;
                Ok(2)
            }
            0x07 | 0x17 | 0x1F => {
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

            // --- BCD adjustments ------------------------------------------------
            0x27 => {
                self.daa();
                Ok(4)
            }
            0x2F => {
                self.das();
                Ok(4)
            }
            0x37 => {
                self.aaa();
                Ok(4)
            }
            0x3F => {
                self.aas();
                Ok(4)
            }

            // --- INC/DEC r --------------------------------------------------------
            0x40..=0x47 => {
                let i = opcode & 7;
                if self.osize32 {
                    let r = self.inc32(self.regs.reg32(i));
                    self.regs.set_reg32(i, r);
                } else {
                    let r = self.inc16(self.regs.reg16(i));
                    self.regs.set_reg16(i, r);
                }
                Ok(2)
            }
            0x48..=0x4F => {
                let i = opcode & 7;
                if self.osize32 {
                    let r = self.dec32(self.regs.reg32(i));
                    self.regs.set_reg32(i, r);
                } else {
                    let r = self.dec16(self.regs.reg16(i));
                    self.regs.set_reg16(i, r);
                }
                Ok(2)
            }

            // --- PUSH/POP r --------------------------------------------------------
            0x50..=0x57 => {
                // Unlike the 8086, PUSH (E)SP pushes the pre-decrement value.
                let v = self.regs.reg32(opcode & 7);
                self.push(bus, v)?;
                Ok(2)
            }
            0x58..=0x5F => {
                let v = self.pop(bus)?;
                if self.osize32 {
                    self.regs.set_reg32(opcode & 7, v);
                } else {
                    self.regs.set_reg16(opcode & 7, v as u16);
                }
                Ok(4)
            }

            // --- PUSHA/POPA --------------------------------------------------------
            // Both walk the frame step by step: a fault mid-way keeps the
            // memory/register effects of completed steps, but the stack
            // pointer reverts so the instruction can restart (386 behavior).
            0x60 => {
                // The 386 writes the frame bottom-up: EDI lands first at
                // SP-32 (SP-16 for 16-bit), ending with EAX just below the
                // old SP. A fault mid-way keeps the completed writes and
                // leaves SP at its original value (visible when a slot
                // straddles the stack limit).
                self.commit_on_fault = true;
                let esp0 = self.regs.gpr[reg::ESP as usize];
                let wrap = self.stack_wrap();
                let width = if self.osize32 { 4u32 } else { 2 };
                let bottom = self.stack_ptr().wrapping_sub(8 * width) & wrap;
                for i in (0..8u8).rev() {
                    let v = if i == reg::ESP {
                        esp0
                    } else {
                        self.regs.reg32(i)
                    };
                    let slot = bottom.wrapping_add((7 - i) as u32 * width) & wrap;
                    if self.osize32 {
                        self.write32w(bus, reg::SS, slot, v, wrap)?;
                    } else {
                        self.write16w(bus, reg::SS, slot, v as u16, wrap)?;
                    }
                }
                if self.regs.seg[reg::SS as usize].db() {
                    self.regs.gpr[reg::ESP as usize] = bottom;
                } else {
                    self.regs.set_reg16(reg::ESP, bottom as u16);
                }
                Ok(18)
            }
            0x61 => {
                // POPA/POPAD walk an internal pointer and write the stack
                // pointer once at the end. POPAD really loads the "skipped"
                // ESP slot too - with a 16-bit stack only the low word is
                // then overwritten, so the slot's high word lands in ESP
                // (386 behavior asserted by the hardware test suite).
                self.commit_on_fault = true;
                let wrap = self.stack_wrap();
                let mut sp = self.stack_ptr();
                for i in (0..8u8).rev() {
                    let v = if self.osize32 {
                        let v = self.read32w(bus, reg::SS, sp, wrap)?;
                        sp = sp.wrapping_add(4) & wrap;
                        v
                    } else {
                        let v = self.read16w(bus, reg::SS, sp, wrap)? as u32;
                        sp = sp.wrapping_add(2) & wrap;
                        v
                    };
                    if self.osize32 {
                        self.regs.set_reg32(i, v);
                    } else if i != reg::ESP {
                        self.regs.set_reg16(i, v as u16);
                    }
                }
                if self.regs.seg[reg::SS as usize].db() {
                    self.regs.gpr[reg::ESP as usize] = sp;
                } else {
                    self.regs.set_reg16(reg::ESP, sp as u16);
                }
                Ok(24)
            }

            // --- BOUND / ARPL ------------------------------------------------------
            0x62 => {
                let (m, op) = self.modrm(bus)?;
                let Operand::Mem { seg, off } = op else {
                    return Err(Exception::ud());
                };
                if self.osize32 {
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
                // ARPL is protected-mode only.
                if !self.protected_mode() {
                    return Err(Exception::ud());
                }
                let (m, op) = self.modrm(bus)?;
                let dst = self.read_op16(bus, op)?;
                let src = self.regs.reg16(m.reg());
                if dst & 3 < src & 3 {
                    self.regs.eflags.insert(EFlags::ZF);
                    self.write_op16(bus, op, (dst & !3) | (src & 3))?;
                } else {
                    self.regs.eflags.remove(EFlags::ZF);
                }
                Ok(20)
            }

            // --- PUSH imm / IMUL imm ------------------------------------------------
            0x68 => {
                let v = self.fetch_imm(bus)?;
                self.push(bus, v)?;
                Ok(2)
            }
            0x6A => {
                let v = self.fetch8(bus)? as i8 as i32 as u32;
                self.push(bus, v)?;
                Ok(2)
            }
            0x69 | 0x6B => {
                let (m, op) = self.modrm(bus)?;
                let a = self.read_op(bus, op)?;
                let b = if opcode == 0x69 {
                    self.fetch_imm(bus)?
                } else {
                    self.fetch8(bus)? as i8 as i32 as u32
                };
                if self.osize32 {
                    let r = self.imul_trunc32(a, b);
                    self.regs.set_reg32(m.reg(), r);
                } else {
                    let r = self.imul_trunc16(a as u16, b as u16);
                    self.regs.set_reg16(m.reg(), r);
                }
                Ok(if op.is_mem() { 22 } else { 20 })
            }

            // --- INS / OUTS -----------------------------------------------------------
            0x6C => self.string_op(bus, Cpu::ins8, false),
            0x6D => {
                if self.osize32 {
                    self.string_op(bus, Cpu::ins32, false)
                } else {
                    self.string_op(bus, Cpu::ins16, false)
                }
            }
            0x6E => self.string_op(bus, Cpu::outs8, false),
            0x6F => {
                if self.osize32 {
                    self.string_op(bus, Cpu::outs32, false)
                } else {
                    self.string_op(bus, Cpu::outs16, false)
                }
            }

            // --- Conditional jumps -------------------------------------------------
            0x70..=0x7F => {
                let rel = self.fetch8(bus)? as i8 as i32;
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
                if self.osize32 {
                    self.op_rm_r32(bus, Cpu::and32, false)
                } else {
                    self.op_rm_r16(bus, Cpu::and16, false)
                }
            }
            0x86 => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem())?;
                let a = self.read_op8(bus, op)?;
                let b = self.regs.reg8(m.reg());
                self.write_op8(bus, op, b)?;
                self.regs.set_reg8(m.reg(), a);
                Ok(if op.is_mem() { 5 } else { 3 })
            }
            0x87 => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem())?;
                let a = self.read_op(bus, op)?;
                let b = self.regs.reg32(m.reg());
                if self.osize32 {
                    self.write_op32(bus, op, b)?;
                    self.regs.set_reg32(m.reg(), a);
                } else {
                    self.write_op16(bus, op, b as u16)?;
                    self.regs.set_reg16(m.reg(), a as u16);
                }
                Ok(if op.is_mem() { 5 } else { 3 })
            }
            0x90..=0x97 => {
                // XCHG eAX, r (90 = NOP).
                let i = opcode & 7;
                if self.osize32 {
                    let t = self.regs.gpr[0];
                    self.regs.gpr[0] = self.regs.reg32(i);
                    self.regs.set_reg32(i, t);
                } else {
                    let t = self.regs.reg16(0);
                    self.regs.set_reg16(0, self.regs.reg16(i));
                    self.regs.set_reg16(i, t);
                }
                Ok(3)
            }
            0xA8 => {
                let b = self.fetch8(bus)?;
                let a = self.regs.reg8(0);
                self.and8(a, b);
                Ok(2)
            }
            0xA9 => {
                if self.osize32 {
                    let b = self.fetch32(bus)?;
                    let a = self.regs.gpr[0];
                    self.and32(a, b);
                } else {
                    let b = self.fetch16(bus)?;
                    let a = self.regs.reg16(0);
                    self.and16(a, b);
                }
                Ok(2)
            }

            // --- MOV -----------------------------------------------------------------
            0x88 => {
                let (m, op) = self.modrm(bus)?;
                let v = self.regs.reg8(m.reg());
                self.write_op8(bus, op, v)?;
                Ok(2)
            }
            0x89 => {
                let (m, op) = self.modrm(bus)?;
                if self.osize32 {
                    let v = self.regs.reg32(m.reg());
                    self.write_op32(bus, op, v)?;
                } else {
                    let v = self.regs.reg16(m.reg());
                    self.write_op16(bus, op, v)?;
                }
                Ok(2)
            }
            0x8A => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op8(bus, op)?;
                self.regs.set_reg8(m.reg(), v);
                Ok(if op.is_mem() { 4 } else { 2 })
            }
            0x8B => {
                let (m, op) = self.modrm(bus)?;
                if self.osize32 {
                    let v = self.read_op32(bus, op)?;
                    self.regs.set_reg32(m.reg(), v);
                } else {
                    let v = self.read_op16(bus, op)?;
                    self.regs.set_reg16(m.reg(), v);
                }
                Ok(if op.is_mem() { 4 } else { 2 })
            }
            0x8C => {
                // MOV r/m16, Sreg (register destination zero-extends to the
                // operand size; memory always gets 16 bits).
                let (m, op) = self.modrm(bus)?;
                if m.reg() > 5 {
                    return Err(Exception::ud());
                }
                let v = self.regs.seg_sel(m.reg());
                match op {
                    Operand::Reg(i) if self.osize32 => self.regs.set_reg32(i, v as u32),
                    Operand::Reg(i) => self.regs.set_reg16(i, v),
                    Operand::Mem { seg, off } => self.write16(bus, seg, off, v)?,
                }
                Ok(2)
            }
            0x8D => {
                let (m, op) = self.modrm(bus)?;
                let Operand::Mem { off, .. } = op else {
                    return Err(Exception::ud());
                };
                if self.osize32 {
                    self.regs.set_reg32(m.reg(), off);
                } else {
                    self.regs.set_reg16(m.reg(), off as u16);
                }
                Ok(2)
            }
            0x8E => {
                let (m, op) = self.modrm(bus)?;
                if m.reg() == reg::CS || m.reg() > 5 {
                    return Err(Exception::ud());
                }
                let v = self.read_op16(bus, op)?;
                self.load_seg(bus, m.reg(), v)?;
                if m.reg() == reg::SS {
                    self.inhibit_interrupts = true;
                }
                Ok(if op.is_mem() { 5 } else { 2 })
            }
            0x8F => {
                // POP r/m (/0). The value pops before the EA resolves, so an
                // EA that uses (E)SP sees the post-pop value — hence pop
                // first, then decode.
                let v = self.pop(bus)?;
                let (m, op) = self.modrm(bus)?;
                if m.reg() != 0 {
                    return Err(Exception::ud());
                }
                self.write_op(bus, op, v)?;
                Ok(if op.is_mem() { 5 } else { 4 })
            }
            0xA0 => {
                let off = self.fetch_moffs(bus)?;
                let seg = self.seg_or(reg::DS);
                let v = self.read8(bus, seg, off)?;
                self.regs.set_reg8(0, v);
                Ok(4)
            }
            0xA1 => {
                let off = self.fetch_moffs(bus)?;
                let seg = self.seg_or(reg::DS);
                if self.osize32 {
                    self.regs.gpr[0] = self.read32(bus, seg, off)?;
                } else {
                    let v = self.read16(bus, seg, off)?;
                    self.regs.set_reg16(0, v);
                }
                Ok(4)
            }
            0xA2 => {
                let off = self.fetch_moffs(bus)?;
                let seg = self.seg_or(reg::DS);
                let v = self.regs.reg8(0);
                self.write8(bus, seg, off, v)?;
                Ok(2)
            }
            0xA3 => {
                let off = self.fetch_moffs(bus)?;
                let seg = self.seg_or(reg::DS);
                if self.osize32 {
                    let v = self.regs.gpr[0];
                    self.write32(bus, seg, off, v)?;
                } else {
                    let v = self.regs.reg16(0);
                    self.write16(bus, seg, off, v)?;
                }
                Ok(2)
            }
            0xB0..=0xB7 => {
                let v = self.fetch8(bus)?;
                self.regs.set_reg8(opcode & 7, v);
                Ok(2)
            }
            0xB8..=0xBF => {
                if self.osize32 {
                    let v = self.fetch32(bus)?;
                    self.regs.set_reg32(opcode & 7, v);
                } else {
                    let v = self.fetch16(bus)?;
                    self.regs.set_reg16(opcode & 7, v);
                }
                Ok(2)
            }
            0xC6 => {
                let (m, op) = self.modrm(bus)?;
                if m.reg() != 0 {
                    return Err(Exception::ud());
                }
                let v = self.fetch8(bus)?;
                self.write_op8(bus, op, v)?;
                Ok(2)
            }
            0xC7 => {
                let (m, op) = self.modrm(bus)?;
                if m.reg() != 0 {
                    return Err(Exception::ud());
                }
                let v = self.fetch_imm(bus)?;
                self.write_op(bus, op, v)?;
                Ok(2)
            }

            // --- Wide pointer loads: LES / LDS ------------------------------------
            0xC4 | 0xC5 => {
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
                if self.osize32 {
                    self.regs.gpr[0] = self.regs.reg16(0) as i16 as i32 as u32;
                } else {
                    let v = self.regs.reg8(0) as i8 as i16 as u16;
                    self.regs.set_reg16(0, v);
                }
                Ok(3)
            }
            0x99 => {
                if self.osize32 {
                    self.regs.gpr[2] = if self.regs.gpr[0] & 0x8000_0000 != 0 {
                        !0
                    } else {
                        0
                    };
                } else {
                    let v = if self.regs.reg16(0) & 0x8000 != 0 {
                        0xFFFF
                    } else {
                        0
                    };
                    self.regs.set_reg16(2, v);
                }
                Ok(2)
            }
            0x9E => {
                // SAHF: SF ZF AF PF CF from AH.
                let ah = (self.regs.reg8(4) as u32) & EFlags::STATUS & 0xFF;
                let keep = self.regs.eflags.bits() & !(EFlags::STATUS & 0xFF);
                self.regs.eflags = EFlags::from_bits_truncate(keep | ah);
                Ok(3)
            }
            0x9F => {
                // LAHF: bit 1 reads as 1, bits 3/5 as 0.
                let lo = (self.regs.eflags.bits() as u8 & 0xD5) | 0x02;
                self.regs.set_reg8(4, lo);
                Ok(2)
            }

            // --- PUSHF / POPF ----------------------------------------------------------
            0x9C => {
                if self.regs.eflags.contains(EFlags::VM) && self.regs.eflags.iopl() < 3 {
                    return Err(Exception::gp(0));
                }
                if self.osize32 {
                    let v = self.regs.eflags.image32();
                    self.push32(bus, v)?;
                } else {
                    let v = self.regs.eflags.image16();
                    self.push16(bus, v)?;
                }
                Ok(4)
            }
            0x9D => {
                if self.regs.eflags.contains(EFlags::VM) && self.regs.eflags.iopl() < 3 {
                    return Err(Exception::gp(0));
                }
                let v = self.pop(bus)?;
                let mask = self.popf_mask();
                self.regs.eflags.load(v, mask);
                Ok(5)
            }

            // --- String operations --------------------------------------------------
            0xA4 => self.string_op(bus, Cpu::movs8, false),
            0xA5 => {
                if self.osize32 {
                    self.string_op(bus, Cpu::movs32, false)
                } else {
                    self.string_op(bus, Cpu::movs16, false)
                }
            }
            0xA6 => self.string_op(bus, Cpu::cmps8, true),
            0xA7 => {
                if self.osize32 {
                    self.string_op(bus, Cpu::cmps32, true)
                } else {
                    self.string_op(bus, Cpu::cmps16, true)
                }
            }
            0xAA => self.string_op(bus, Cpu::stos8, false),
            0xAB => {
                if self.osize32 {
                    self.string_op(bus, Cpu::stos32, false)
                } else {
                    self.string_op(bus, Cpu::stos16, false)
                }
            }
            0xAC => self.string_op(bus, Cpu::lods8, false),
            0xAD => {
                if self.osize32 {
                    self.string_op(bus, Cpu::lods32, false)
                } else {
                    self.string_op(bus, Cpu::lods16, false)
                }
            }
            0xAE => self.string_op(bus, Cpu::scas8, true),
            0xAF => {
                if self.osize32 {
                    self.string_op(bus, Cpu::scas32, true)
                } else {
                    self.string_op(bus, Cpu::scas16, true)
                }
            }

            // --- Shift/rotate groups ---------------------------------------------------
            0xC0 => self.shift_grp8(bus, None),
            0xC1 => self.shift_grp(bus, None),
            0xD0 => self.shift_grp8(bus, Some(1)),
            0xD1 => self.shift_grp(bus, Some(1)),
            0xD2 => {
                let c = self.regs.reg8(1) as u32;
                self.shift_grp8(bus, Some(c))
            }
            0xD3 => {
                let c = self.regs.reg8(1) as u32;
                self.shift_grp(bus, Some(c))
            }

            // --- RET near ---------------------------------------------------------------
            0xC3 => {
                let ip = self.pop(bus)?;
                self.set_ip(ip)?;
                Ok(10)
            }
            0xC2 => {
                let n = self.fetch16(bus)?;
                let ip = self.pop(bus)?;
                self.set_ip(ip)?;
                self.adjust_sp(n as i32);
                Ok(10)
            }

            // --- ENTER / LEAVE ------------------------------------------------------------
            0xC8 => {
                let alloc = self.fetch16(bus)? as u32;
                let level = (self.fetch8(bus)? & 0x1F) as u32;
                self.enter(bus, alloc, level)?;
                Ok(10 + 4 * level)
            }
            0xC9 => {
                // (E)SP <- (E)BP per SS.B, then pop (E)BP per operand size.
                if self.regs.seg[reg::SS as usize].db() {
                    self.regs.gpr[reg::ESP as usize] = self.regs.gpr[reg::EBP as usize];
                } else {
                    let bp = self.regs.reg16(reg::EBP);
                    self.regs.set_reg16(reg::ESP, bp);
                }
                let v = self.pop(bus)?;
                if self.osize32 {
                    self.regs.gpr[reg::EBP as usize] = v;
                } else {
                    self.regs.set_reg16(reg::EBP, v as u16);
                }
                Ok(4)
            }

            // --- RET far ---------------------------------------------------------------------
            0xCB => self.retf(bus, 0),
            0xCA => {
                let n = self.fetch16(bus)?;
                self.retf(bus, n as u32)
            }

            // --- Software interrupts -----------------------------------------------------------
            0xCC => {
                self.software_int(bus, 3)?;
                Ok(33)
            }
            0xCD => {
                let v = self.fetch8(bus)?;
                // In V86 mode, INT n is IOPL-sensitive.
                if self.regs.eflags.contains(EFlags::VM) && self.regs.eflags.iopl() < 3 {
                    return Err(Exception::gp(0));
                }
                self.software_int(bus, v)?;
                Ok(37)
            }
            0xCE => {
                if self.regs.eflags.contains(EFlags::OF) {
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
                let base = self.fetch8(bus)?;
                if !self.aam(base) {
                    return Err(Exception::de());
                }
                Ok(17)
            }
            0xD5 => {
                let base = self.fetch8(bus)?;
                self.aad(base);
                Ok(19)
            }
            0xD6 => {
                // Undocumented SALC: AL = CF ? FF : 00.
                let v = if self.regs.eflags.contains(EFlags::CF) {
                    0xFF
                } else {
                    0
                };
                self.regs.set_reg8(0, v);
                Ok(3)
            }
            0xD7 => {
                let seg = self.seg_or(reg::DS);
                let bx = if self.asize32 {
                    self.regs.gpr[3]
                } else {
                    self.regs.reg16(3) as u32
                };
                let off = bx.wrapping_add(self.regs.reg8(0) as u32);
                let off = if self.asize32 { off } else { off & 0xFFFF };
                let v = self.read8(bus, seg, off)?;
                self.regs.set_reg8(0, v);
                Ok(5)
            }

            // --- ESC (x87 opcodes; no coprocessor attached) -----------------------------------------
            0xD8..=0xDF => {
                if self.regs.cr0 & (cr0::EM | cr0::TS) != 0 {
                    return Err(Exception::nm());
                }
                let _ = self.modrm(bus)?;
                Ok(2)
            }

            // --- Loops / IN / OUT ----------------------------------------------------------------------
            0xE0..=0xE2 => {
                let rel = self.fetch8(bus)? as i8 as i32;
                let c = self.count_reg().wrapping_sub(1);
                self.set_count_reg(c);
                let go = c != 0
                    && match opcode {
                        0xE0 => !self.regs.eflags.contains(EFlags::ZF),
                        0xE1 => self.regs.eflags.contains(EFlags::ZF),
                        _ => true,
                    };
                if go {
                    self.jump_rel(rel)?;
                    Ok(11)
                } else {
                    Ok(11)
                }
            }
            0xE3 => {
                let rel = self.fetch8(bus)? as i8 as i32;
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
                self.regs.set_reg8(0, v);
                Ok(12)
            }
            0xE5 => {
                let p = self.fetch8(bus)? as u16;
                if self.osize32 {
                    self.io_check(bus, p, 4)?;
                    self.regs.gpr[0] = self.io_read32(bus, p);
                } else {
                    self.io_check(bus, p, 2)?;
                    let v = self.io_read16(bus, p);
                    self.regs.set_reg16(0, v);
                }
                Ok(12)
            }
            0xE6 => {
                let p = self.fetch8(bus)? as u16;
                self.io_check(bus, p, 1)?;
                bus.io_write(p, self.regs.reg8(0));
                Ok(10)
            }
            0xE7 => {
                let p = self.fetch8(bus)? as u16;
                if self.osize32 {
                    self.io_check(bus, p, 4)?;
                    let v = self.regs.gpr[0];
                    self.io_write32(bus, p, v);
                } else {
                    self.io_check(bus, p, 2)?;
                    let v = self.regs.reg16(0);
                    self.io_write16(bus, p, v);
                }
                Ok(10)
            }
            0xEC => {
                let p = self.regs.reg16(2);
                self.io_check(bus, p, 1)?;
                let v = bus.io_read(p);
                self.regs.set_reg8(0, v);
                Ok(13)
            }
            0xED => {
                let p = self.regs.reg16(2);
                if self.osize32 {
                    self.io_check(bus, p, 4)?;
                    self.regs.gpr[0] = self.io_read32(bus, p);
                } else {
                    self.io_check(bus, p, 2)?;
                    let v = self.io_read16(bus, p);
                    self.regs.set_reg16(0, v);
                }
                Ok(13)
            }
            0xEE => {
                let p = self.regs.reg16(2);
                self.io_check(bus, p, 1)?;
                bus.io_write(p, self.regs.reg8(0));
                Ok(11)
            }
            0xEF => {
                let p = self.regs.reg16(2);
                if self.osize32 {
                    self.io_check(bus, p, 4)?;
                    let v = self.regs.gpr[0];
                    self.io_write32(bus, p, v);
                } else {
                    self.io_check(bus, p, 2)?;
                    let v = self.regs.reg16(0);
                    self.io_write16(bus, p, v);
                }
                Ok(11)
            }

            // --- CALL / JMP -------------------------------------------------------------------------------
            0x9A => {
                // CALL far ptr16:16/32.
                let off = self.fetch_imm(bus)?;
                let sel = self.fetch16(bus)?;
                self.call_far(bus, sel, off)?;
                Ok(17)
            }
            0xE8 => {
                let rel = if self.osize32 {
                    self.fetch32(bus)? as i32
                } else {
                    self.fetch16(bus)? as i16 as i32
                };
                let ret = self.regs.eip;
                self.push(bus, ret)?;
                self.jump_rel(rel)?;
                Ok(7)
            }
            0xE9 => {
                let rel = if self.osize32 {
                    self.fetch32(bus)? as i32
                } else {
                    self.fetch16(bus)? as i16 as i32
                };
                self.jump_rel(rel)?;
                Ok(7)
            }
            0xEA => {
                let off = self.fetch_imm(bus)?;
                let sel = self.fetch16(bus)?;
                self.jump_far(bus, sel, off)?;
                Ok(12)
            }
            0xEB => {
                let rel = self.fetch8(bus)? as i8 as i32;
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
                self.regs.eflags.toggle(EFlags::CF);
                Ok(2)
            }
            0xF8 => {
                self.regs.eflags.remove(EFlags::CF);
                Ok(2)
            }
            0xF9 => {
                self.regs.eflags.insert(EFlags::CF);
                Ok(2)
            }
            0xFA => {
                if self.iopl_sensitive_blocked() {
                    return Err(Exception::gp(0));
                }
                self.regs.eflags.remove(EFlags::IF);
                Ok(3)
            }
            0xFB => {
                if self.iopl_sensitive_blocked() {
                    return Err(Exception::gp(0));
                }
                self.regs.eflags.insert(EFlags::IF);
                self.inhibit_interrupts = true;
                Ok(3)
            }
            0xFC => {
                self.regs.eflags.remove(EFlags::DF);
                Ok(2)
            }
            0xFD => {
                self.regs.eflags.insert(EFlags::DF);
                Ok(2)
            }

            // --- Group F6/F7: TEST NOT NEG MUL IMUL DIV IDIV --------------------------------------------------
            0xF6 => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && matches!(m.reg(), 2 | 3))?;
                let v = self.read_op8(bus, op)?;
                match m.reg() {
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
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && matches!(m.reg(), 2 | 3))?;
                let v = self.read_op(bus, op)?;
                match m.reg() {
                    0 | 1 => {
                        let b = self.fetch_imm(bus)?;
                        if self.osize32 {
                            self.and32(v, b);
                        } else {
                            self.and16(v as u16, b as u16);
                        }
                        Ok(if op.is_mem() { 5 } else { 2 })
                    }
                    2 => {
                        self.write_op(bus, op, !v)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    3 => {
                        let r = if self.osize32 {
                            self.neg32(v)
                        } else {
                            self.neg16(v as u16) as u32
                        };
                        self.write_op(bus, op, r)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    4 => {
                        if self.osize32 {
                            self.mul32(v)
                        } else {
                            self.mul16(v as u16)
                        }
                        Ok(if self.osize32 { 38 } else { 25 })
                    }
                    5 => {
                        if self.osize32 {
                            self.imul32(v)
                        } else {
                            self.imul16(v as u16)
                        }
                        Ok(if self.osize32 { 38 } else { 25 })
                    }
                    6 => {
                        let ok = if self.osize32 {
                            self.div32(v)
                        } else {
                            self.div16(v as u16)
                        };
                        if !ok {
                            return Err(Exception::de());
                        }
                        Ok(if self.osize32 { 38 } else { 27 })
                    }
                    _ => {
                        let ok = if self.osize32 {
                            self.idiv32(v)
                        } else {
                            self.idiv16(v as u16)
                        };
                        if !ok {
                            return Err(Exception::de());
                        }
                        Ok(if self.osize32 { 43 } else { 30 })
                    }
                }
            }

            // --- Group FE: INC/DEC r/m8 ------------------------------------------------------------------------
            0xFE => {
                let (m, op) = self.modrm(bus)?;
                self.lock_check(op.is_mem() && m.reg() < 2)?;
                match m.reg() {
                    0 | 1 => {
                        let v = self.read_op8(bus, op)?;
                        let r = if m.reg() == 0 {
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
                self.lock_check(op.is_mem() && m.reg() < 2)?;
                match m.reg() {
                    0 | 1 => {
                        let v = self.read_op(bus, op)?;
                        let r = if self.osize32 {
                            if m.reg() == 0 {
                                self.inc32(v)
                            } else {
                                self.dec32(v)
                            }
                        } else if m.reg() == 0 {
                            self.inc16(v as u16) as u32
                        } else {
                            self.dec16(v as u16) as u32
                        };
                        self.write_op(bus, op, r)?;
                        Ok(if op.is_mem() { 6 } else { 2 })
                    }
                    2 => {
                        let v = self.read_op(bus, op)?;
                        let ret = self.regs.eip;
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

    /// Evaluate condition code `n` (the low nibble of a Jcc/SETcc opcode).
    pub(crate) fn cond(&self, n: u8) -> bool {
        let f = self.regs.eflags;
        let r = match n >> 1 {
            0 => f.contains(EFlags::OF),
            1 => f.contains(EFlags::CF),
            2 => f.contains(EFlags::ZF),
            3 => f.contains(EFlags::CF) || f.contains(EFlags::ZF),
            4 => f.contains(EFlags::SF),
            5 => f.contains(EFlags::PF),
            6 => f.contains(EFlags::SF) != f.contains(EFlags::OF),
            _ => f.contains(EFlags::ZF) || (f.contains(EFlags::SF) != f.contains(EFlags::OF)),
        };
        r != (n & 1 != 0)
    }

    /// Relative jump: add `rel` to EIP, truncating to 16 bits when the
    /// operand size is 16-bit. Targets beyond the CS limit fault at the
    /// transfer, before it commits.
    #[inline]
    pub(crate) fn jump_rel(&mut self, rel: i32) -> Exec<()> {
        let ip = self.regs.eip.wrapping_add(rel as u32);
        self.set_ip(ip)
    }

    /// Absolute near jump target (RET/JMP/CALL indirect), truncated per
    /// operand size and checked against the CS limit.
    #[inline]
    pub(crate) fn set_ip(&mut self, ip: u32) -> Exec<()> {
        let ip = if self.osize32 { ip } else { ip & 0xFFFF };
        if ip > self.regs.seg[reg::CS as usize].limit {
            return Err(Exception::gp(0));
        }
        self.regs.eip = ip;
        Ok(())
    }

    /// Fetch a `moffs` direct address (width = address size).
    #[inline]
    fn fetch_moffs<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        if self.asize32 {
            self.fetch32(bus)
        } else {
            Ok(self.fetch16(bus)? as u32)
        }
    }

    /// The set of EFLAGS bits `POPF`/`IRET` may modify in the current mode.
    pub(crate) fn popf_mask(&self) -> u32 {
        let mut mask = 0x0000_7FD5u32; // all defined 16-bit flags incl. IOPL/NT
        if self.osize32 {
            mask |= EFlags::RF.bits(); // VM is never writable this way
        }
        if self.protected_mode() {
            let cpl = self.cpl();
            if cpl > 0 {
                mask &= !EFlags::IOPL.bits();
                if cpl as u32 > (self.regs.eflags.bits() >> 12) & 3 {
                    mask &= !EFlags::IF.bits();
                }
            }
        } else if self.regs.eflags.contains(EFlags::VM) {
            // The caller already required IOPL == 3; IOPL itself is fixed.
            mask &= !EFlags::IOPL.bits();
        }
        mask
    }

    /// CLI/STI legality: IOPL-sensitive in protected and V86 modes.
    fn iopl_sensitive_blocked(&self) -> bool {
        if self.regs.cr0 & cr0::PE == 0 {
            return false;
        }
        self.cpl() > self.regs.eflags.iopl()
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
        let b = self.regs.reg8(m.reg());
        let r = f(self, a, b);
        if wb {
            self.write_op8(bus, op, r)?;
        }
        Ok(if op.is_mem() { 7 } else { 2 })
    }

    #[inline(always)]
    fn op_rm_r16<B: Bus>(
        &mut self,
        bus: &mut B,
        f: fn(&mut Cpu, u16, u16) -> u16,
        wb: bool,
    ) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        self.lock_check(wb && op.is_mem())?;
        let a = self.read_op16(bus, op)?;
        let b = self.regs.reg16(m.reg());
        let r = f(self, a, b);
        if wb {
            self.write_op16(bus, op, r)?;
        }
        Ok(if op.is_mem() { 7 } else { 2 })
    }

    #[inline(always)]
    fn op_rm_r32<B: Bus>(
        &mut self,
        bus: &mut B,
        f: fn(&mut Cpu, u32, u32) -> u32,
        wb: bool,
    ) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        self.lock_check(wb && op.is_mem())?;
        let a = self.read_op32(bus, op)?;
        let b = self.regs.reg32(m.reg());
        let r = f(self, a, b);
        if wb {
            self.write_op32(bus, op, r)?;
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
        let a = self.regs.reg8(m.reg());
        let b = self.read_op8(bus, op)?;
        let r = f(self, a, b);
        if wb {
            self.regs.set_reg8(m.reg(), r);
        }
        Ok(if op.is_mem() { 6 } else { 2 })
    }

    #[inline(always)]
    fn op_r_rm16<B: Bus>(
        &mut self,
        bus: &mut B,
        f: fn(&mut Cpu, u16, u16) -> u16,
        wb: bool,
    ) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let a = self.regs.reg16(m.reg());
        let b = self.read_op16(bus, op)?;
        let r = f(self, a, b);
        if wb {
            self.regs.set_reg16(m.reg(), r);
        }
        Ok(if op.is_mem() { 6 } else { 2 })
    }

    #[inline(always)]
    fn op_r_rm32<B: Bus>(
        &mut self,
        bus: &mut B,
        f: fn(&mut Cpu, u32, u32) -> u32,
        wb: bool,
    ) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let a = self.regs.reg32(m.reg());
        let b = self.read_op32(bus, op)?;
        let r = f(self, a, b);
        if wb {
            self.regs.set_reg32(m.reg(), r);
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
        let r = self.shift_dispatch8(m.reg(), v, n);
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
            6 => self.shl8(v, n), // /6 aliases SHL on the 386
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
        let r = self.shift_dispatch(m.reg(), v, n);
        self.write_op(bus, op, r)?;
        Ok(if op.is_mem() { 7 } else { 3 })
    }

    fn shift_dispatch(&mut self, sub: u8, v: u32, n: u32) -> u32 {
        if self.osize32 {
            match sub {
                0 => self.rol32(v, n),
                1 => self.ror32(v, n),
                2 => self.rcl32(v, n),
                3 => self.rcr32(v, n),
                4 => self.shl32(v, n),
                5 => self.shr32(v, n),
                6 => self.shl32(v, n),
                _ => self.sar32(v, n),
            }
        } else {
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
            }) as u32
        }
    }

    /// Read an `offset:selector` far pointer from memory (offset width per
    /// operand size).
    pub(crate) fn read_far_pointer<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
    ) -> Exec<(u16, u32)> {
        let wrap = self.data_wrap();
        if self.osize32 {
            let dst = self.read32(bus, seg, off)?;
            let sel = self.read16(bus, seg, off.wrapping_add(4) & wrap)?;
            Ok((sel, dst))
        } else {
            let dst = self.read16(bus, seg, off)? as u32;
            let sel = self.read16(bus, seg, off.wrapping_add(2) & wrap)?;
            Ok((sel, dst))
        }
    }

    /// `LES/LDS/LFS/LGS/LSS r, m`: load the register then the segment.
    pub(crate) fn load_far_pointer<B: Bus>(
        &mut self,
        bus: &mut B,
        sreg: u8,
        r: u8,
        seg: u8,
        off: u32,
    ) -> Exec<()> {
        let (sel, dst) = self.read_far_pointer(bus, seg, off)?;
        self.load_seg(bus, sreg, sel)?;
        if self.osize32 {
            self.regs.set_reg32(r, dst);
        } else {
            self.regs.set_reg16(r, dst as u16);
        }
        if sreg == reg::SS {
            self.inhibit_interrupts = true;
        }
        Ok(())
    }

    /// ENTER: build a stack frame with `level` nested frame pointers.
    fn enter<B: Bus>(&mut self, bus: &mut B, alloc: u32, level: u32) -> Exec<()> {
        let bp = self.regs.gpr[reg::EBP as usize];
        self.push(bus, bp)?;
        let frame = self.stack_ptr();
        if level > 0 {
            for _ in 1..level {
                // Walk the display chain: BP steps down one slot per level.
                let wrap = self.stack_wrap();
                if self.osize32 {
                    let nbp = self.regs.gpr[reg::EBP as usize].wrapping_sub(4);
                    self.regs.gpr[reg::EBP as usize] = nbp;
                    let v = self.read32w(bus, reg::SS, nbp & wrap, wrap)?;
                    self.push32(bus, v)?;
                } else {
                    let nbp = self.regs.reg16(reg::EBP).wrapping_sub(2);
                    self.regs.set_reg16(reg::EBP, nbp);
                    let v = self.read16w(bus, reg::SS, nbp as u32, wrap)?;
                    self.push16(bus, v)?;
                }
            }
            self.push(bus, frame)?;
        }
        if self.osize32 {
            self.regs.gpr[reg::EBP as usize] = frame;
        } else {
            self.regs.set_reg16(reg::EBP, frame as u16);
        }
        self.adjust_sp(-(alloc as i32));
        Ok(())
    }

    // --- Port I/O (wide access composed from the byte-wide bus) -------------------

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

    /// (E)CX per address size.
    #[inline]
    fn count_reg(&self) -> u32 {
        if self.asize32 {
            self.regs.gpr[1]
        } else {
            self.regs.gpr[1] & 0xFFFF
        }
    }

    #[inline]
    fn set_count_reg(&mut self, v: u32) {
        if self.asize32 {
            self.regs.gpr[1] = v;
        } else {
            self.regs.set_reg16(1, v as u16);
        }
    }

    /// (E)SI / (E)DI per address size.
    #[inline]
    fn index_reg(&self, i: u8) -> u32 {
        let v = self.regs.gpr[i as usize];
        if self.asize32 { v } else { v & 0xFFFF }
    }

    #[inline]
    fn set_index_reg(&mut self, i: u8, v: u32) {
        if self.asize32 {
            self.regs.gpr[i as usize] = v;
        } else {
            self.regs.set_reg16(i, v as u16);
        }
    }

    /// ±element-size depending on DF.
    #[inline]
    fn delta(&self, n: u32) -> u32 {
        if self.regs.eflags.contains(EFlags::DF) {
            n.wrapping_neg()
        } else {
            n
        }
    }

    /// Advance (E)SI by ±n.
    #[inline]
    fn step_si(&mut self, n: u32) {
        let d = self.delta(n);
        let v = self.index_reg(reg::ESI).wrapping_add(d);
        self.set_index_reg(reg::ESI, v);
    }

    /// Advance (E)DI by ±n.
    #[inline]
    fn step_di(&mut self, n: u32) {
        let d = self.delta(n);
        let v = self.index_reg(reg::EDI).wrapping_add(d);
        self.set_index_reg(reg::EDI, v);
    }

    /// Run one string primitive, honoring an active REP/REPE/REPNE prefix.
    /// `cmp` marks CMPS/SCAS, whose repetition also terminates on the ZF
    /// condition. Faults mid-iteration keep completed progress (EIP rewinds
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
            if cmp && self.regs.eflags.contains(EFlags::ZF) != cont_on_zf {
                break;
            }
            // Hardware recognizes interrupts between iterations; with a
            // 32-bit address size a single REP can run for 2^32 of them, so
            // yield by rewinding EIP to the prefix. The next `step()` takes
            // the interrupt, and the handler's IRET resumes the REP with the
            // already-decremented (E)CX and index registers.
            if self.count_reg() != 0 && self.interrupt_pending() {
                self.regs.eip = self.start_eip;
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
     $ins:ident, $outs:ident,
     $read:ident, $write:ident, $sub:ident, $set_acc:ident, $io_read:ident, $io_write:ident) => {
        impl Cpu {
            fn $movs<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let seg = self.seg_or(reg::DS);
                let si = self.index_reg(reg::ESI);
                let v = self.$read(bus, seg, si)?;
                let di = self.index_reg(reg::EDI);
                self.$write(bus, reg::ES, di, v)?;
                self.step_si($n);
                self.step_di($n);
                Ok(())
            }

            fn $cmps<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let seg = self.seg_or(reg::DS);
                let a = self.$read(bus, seg, self.index_reg(reg::ESI))?;
                let b = self.$read(bus, reg::ES, self.index_reg(reg::EDI))?;
                self.$sub(a, b);
                self.step_si($n);
                self.step_di($n);
                Ok(())
            }

            fn $stos<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let v = self.regs.gpr[0] as $t;
                let di = self.index_reg(reg::EDI);
                self.$write(bus, reg::ES, di, v)?;
                self.step_di($n);
                Ok(())
            }

            fn $lods<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let seg = self.seg_or(reg::DS);
                let v = self.$read(bus, seg, self.index_reg(reg::ESI))?;
                self.regs.$set_acc(0, v);
                self.step_si($n);
                Ok(())
            }

            fn $scas<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let b = self.$read(bus, reg::ES, self.index_reg(reg::EDI))?;
                let a = self.regs.gpr[0] as $t;
                self.$sub(a, b);
                self.step_di($n);
                Ok(())
            }

            /// INS always stores to ES:(E)DI — no override.
            fn $ins<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let port = self.regs.reg16(2);
                self.io_check(bus, port, $n)?;
                let v = self.$io_read(bus, port);
                let di = self.index_reg(reg::EDI);
                self.$write(bus, reg::ES, di, v)?;
                self.step_di($n);
                Ok(())
            }

            fn $outs<B: Bus>(&mut self, bus: &mut B) -> Exec<()> {
                let port = self.regs.reg16(2);
                self.io_check(bus, port, $n)?;
                let seg = self.seg_or(reg::DS);
                let v = self.$read(bus, seg, self.index_reg(reg::ESI))?;
                self.$io_write(bus, port, v);
                self.step_si($n);
                Ok(())
            }
        }
    };
}

string_ops!(
    u8, 1, movs8, cmps8, stos8, lods8, scas8, ins8, outs8, read8, write8, sub8, set_reg8, io_read8,
    io_write8
);
string_ops!(
    u16, 2, movs16, cmps16, stos16, lods16, scas16, ins16, outs16, read16, write16, sub16,
    set_reg16, io_read16, io_write16
);
string_ops!(
    u32, 4, movs32, cmps32, stos32, lods32, scas32, ins32, outs32, read32, write32, sub32,
    set_reg32, io_read32, io_write32
);

impl Cpu {
    /// Byte port read (uniform shape for the string-op macro).
    fn io_read8<B: Bus>(&mut self, bus: &mut B, port: u16) -> u8 {
        bus.io_read(port)
    }

    /// Byte port write.
    fn io_write8<B: Bus>(&mut self, bus: &mut B, port: u16, v: u8) {
        bus.io_write(port, v);
    }
}
