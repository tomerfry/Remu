//! Two-byte (`0F xx`) opcode dispatch: system instructions (control
//! registers, descriptor tables, MSRs, SYSCALL/SYSRET, SWAPGS, INVLPG),
//! CPUID, bit operations, MOVZX/MOVSX, CMOVcc, SETcc, SHLD/SHRD, the atomic
//! primitives (CMPXCHG, CMPXCHG8B/16B, XADD) and BSWAP. SSE encodings raise
//! #UD — no SIMD state exists, and CPUID says so.

use super::OpSize::{O16, O32, O64};
use super::modrm::{ModRm, Operand};
use super::registers::{RFlags, cr0, cr4, efer, reg};
use super::{Bus, Cpu, Exception, Exec};

impl Cpu {
    pub(crate) fn dispatch_0f<B: Bus>(&mut self, bus: &mut B, opcode: u8) -> Exec<u32> {
        // Decode-time LOCK legality: the bit-set group, the atomic
        // primitives, and group 9 (CMPXCHG8B/16B).
        if self.lock
            && !matches!(
                opcode,
                0xAB | 0xB3 | 0xBB | 0xBA | 0xB0 | 0xB1 | 0xC0 | 0xC1 | 0xC7
            )
        {
            return Err(Exception::ud());
        }
        match opcode {
            0x00 => self.group6(bus),
            0x01 => self.group7(bus),
            0x02 => self.lar_lsl(bus, false),
            0x03 => self.lar_lsl(bus, true),
            0x05 => self.syscall(),
            0x06 => {
                // CLTS.
                self.require_ring0()?;
                self.regs.cr0 &= !cr0::TS;
                Ok(5)
            }
            0x07 => self.sysret(),
            0x08 | 0x09 => {
                // INVD/WBINVD: no cache is modeled.
                self.require_ring0()?;
                Ok(5)
            }
            0x0B => Err(Exception::ud()), // UD2
            0x0D | 0x18..=0x1F => {
                // Multi-byte NOP and the prefetch/hint group: consume the
                // ModRM operand, do nothing.
                let _ = self.modrm(bus)?;
                Ok(1)
            }

            // --- MOV to/from control and debug registers ----------------------
            // The ModRM mod field is ignored: the operand is always a GPR and
            // no displacement is fetched. REX.R extends the register field,
            // reaching CR8.
            0x20 => {
                self.require_ring0()?;
                let m = self.modrm_reg_only(bus)?;
                let v = match m.reg() {
                    0 => self.regs.cr0,
                    2 => self.regs.cr2,
                    3 => self.regs.cr3,
                    4 => self.regs.cr4,
                    8 => self.regs.cr8,
                    _ => return Err(Exception::ud()),
                };
                self.write_gpr_mode(m.rm(), v);
                Ok(6)
            }
            0x22 => {
                self.require_ring0()?;
                let m = self.modrm_reg_only(bus)?;
                let v = self.read_gpr_mode(m.rm());
                self.write_cr(m.reg(), v)?;
                Ok(10)
            }
            0x21 => {
                self.require_ring0()?;
                let m = self.modrm_reg_only(bus)?;
                let n = match m.sub() {
                    4 => 6,
                    5 => 7,
                    n => n,
                } as usize;
                let v = self.regs.dr[n];
                self.write_gpr_mode(m.rm(), v);
                Ok(14)
            }
            0x23 => {
                self.require_ring0()?;
                let m = self.modrm_reg_only(bus)?;
                let n = match m.sub() {
                    4 => 6,
                    5 => 7,
                    n => n,
                } as usize;
                self.regs.dr[n] = self.read_gpr_mode(m.rm());
                Ok(16)
            }

            // --- TSC / MSR / PMC access -----------------------------------------
            0x30 => {
                // WRMSR.
                self.require_ring0()?;
                let index = self.regs.reg32(reg::RCX);
                let v = (self.regs.reg32(reg::RDX) as u64) << 32 | self.regs.reg32(reg::RAX) as u64;
                self.wrmsr(index, v)?;
                Ok(30)
            }
            0x31 => {
                // RDTSC (CR4.TSD makes it privileged).
                if self.regs.cr4 & cr4::TSD != 0 && self.cpl() != 0 {
                    return Err(Exception::gp(0));
                }
                let t = self.cycles;
                self.regs.set_reg32(reg::RAX, t as u32);
                self.regs.set_reg32(reg::RDX, (t >> 32) as u32);
                Ok(5)
            }
            0x32 => {
                // RDMSR.
                self.require_ring0()?;
                let index = self.regs.reg32(reg::RCX);
                let v = self.rdmsr(index)?;
                self.regs.set_reg32(reg::RAX, v as u32);
                self.regs.set_reg32(reg::RDX, (v >> 32) as u32);
                Ok(30)
            }
            0x33 => {
                // RDPMC: no performance counters are modeled — reads as 0.
                if self.regs.cr4 & cr4::PCE == 0 && self.cpl() != 0 {
                    return Err(Exception::gp(0));
                }
                self.regs.set_reg32(reg::RAX, 0);
                self.regs.set_reg32(reg::RDX, 0);
                Ok(5)
            }

            // --- CMOVcc ----------------------------------------------------------
            0x40..=0x4F => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op(bus, op)?;
                // The destination is written even on a false condition (it
                // keeps its value) — in 64-bit mode that still zeroes the
                // upper half of a 32-bit destination.
                let taken = self.cond(opcode & 0xF);
                let cur = match self.osize {
                    O16 => self.regs.reg16(m.reg()) as u64,
                    O32 => self.regs.reg32(m.reg()) as u64,
                    O64 => self.regs.reg64(m.reg()),
                };
                self.write_reg_osize(m.reg(), if taken { v } else { cur });
                Ok(if op.is_mem() { 5 } else { 4 })
            }

            // --- Long conditional jumps ----------------------------------------
            0x80..=0x8F => {
                self.osize = self.branch_osize();
                let rel = self.fetch_rel(bus)?;
                if self.cond(opcode & 0xF) {
                    self.jump_rel(rel)?;
                    Ok(7)
                } else {
                    Ok(3)
                }
            }

            // --- SETcc -----------------------------------------------------------
            0x90..=0x9F => {
                let (_, op) = self.modrm(bus)?;
                let v = self.cond(opcode & 0xF) as u8;
                self.write_op8(bus, op, v)?;
                Ok(4)
            }

            // --- PUSH/POP FS/GS (legal in 64-bit mode, unlike the others) --------
            0xA0 | 0xA8 => {
                self.osize = self.stack_osize();
                let idx = if opcode == 0xA0 { reg::FS } else { reg::GS };
                let v = self.regs.seg_sel(idx) as u64;
                self.push(bus, v)?;
                Ok(2)
            }
            0xA1 | 0xA9 => {
                self.osize = self.stack_osize();
                let idx = if opcode == 0xA1 { reg::FS } else { reg::GS };
                let v = self.pop_sreg(bus)?;
                self.load_seg(bus, idx, v)?;
                Ok(7)
            }

            // --- CPUID -------------------------------------------------------------
            0xA2 => {
                self.cpuid();
                Ok(14)
            }

            // --- Bit tests -------------------------------------------------------------
            0xA3 => self.bt_reg_index(bus, 0), // BT
            0xAB => self.bt_reg_index(bus, 1), // BTS
            0xB3 => self.bt_reg_index(bus, 2), // BTR
            0xBB => self.bt_reg_index(bus, 3), // BTC
            0xBA => {
                // Group 8: BT/BTS/BTR/BTC r/m, imm8 (/4../7).
                self.imm_len = 1;
                let (m, op) = self.modrm(bus)?;
                if m.sub() < 4 {
                    return Err(Exception::ud());
                }
                let imm = self.fetch8(bus)? as u64;
                let bit = imm
                    & match self.osize {
                        O16 => 15,
                        O32 => 31,
                        O64 => 63,
                    };
                self.bt_apply(bus, op, m.sub() - 4, bit)
            }

            // --- SHLD / SHRD -----------------------------------------------------------
            0xA4 => {
                self.imm_len = 1;
                self.shxd(bus, false, None)
            }
            0xA5 => {
                let c = self.regs.reg8(1, false) as u32;
                self.shxd(bus, false, Some(c))
            }
            0xAC => {
                self.imm_len = 1;
                self.shxd(bus, true, None)
            }
            0xAD => {
                let c = self.regs.reg8(1, false) as u32;
                self.shxd(bus, true, Some(c))
            }

            // --- IMUL r, r/m --------------------------------------------------------------
            0xAF => {
                let (m, op) = self.modrm(bus)?;
                let src = self.read_op(bus, op)?;
                match self.osize {
                    O16 => {
                        let a = self.regs.reg16(m.reg());
                        let r = self.imul_trunc16(a, src as u16);
                        self.regs.set_reg16(m.reg(), r);
                    }
                    O32 => {
                        let a = self.regs.reg32(m.reg());
                        let r = self.imul_trunc32(a, src as u32);
                        self.regs.set_reg32(m.reg(), r);
                    }
                    O64 => {
                        let a = self.regs.reg64(m.reg());
                        let r = self.imul_trunc64(a, src);
                        self.regs.set_reg64(m.reg(), r);
                    }
                }
                Ok(if op.is_mem() { 22 } else { 20 })
            }

            // --- CMPXCHG / XADD ------------------------------------------------------------
            0xB0 | 0xB1 => self.cmpxchg(bus, opcode == 0xB0),
            0xC0 | 0xC1 => self.xadd(bus, opcode == 0xC0),

            // --- LSS / LFS / LGS --------------------------------------------------------------
            0xB2 | 0xB4 | 0xB5 => {
                let (m, op) = self.modrm(bus)?;
                let Operand::Mem { seg, off } = op else {
                    return Err(Exception::ud());
                };
                let sreg = match opcode {
                    0xB2 => reg::SS,
                    0xB4 => reg::FS,
                    _ => reg::GS,
                };
                self.load_far_pointer(bus, sreg, m.reg(), seg, off)?;
                Ok(7)
            }

            // --- MOVZX / MOVSX -------------------------------------------------------------------
            0xB6 => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op8(bus, op)? as u64;
                self.write_reg_osize(m.reg(), v);
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            0xB7 => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op16(bus, op)? as u64;
                self.write_reg_osize(m.reg(), v);
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            0xBE => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op8(bus, op)? as i8 as i64 as u64;
                self.write_reg_osize(m.reg(), v);
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            0xBF => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op16(bus, op)? as i16 as i64 as u64;
                self.write_reg_osize(m.reg(), v);
                Ok(if op.is_mem() { 6 } else { 3 })
            }

            // --- BSF / BSR -----------------------------------------------------------------------------
            // Hardware leaves the destination unchanged and most flags
            // undefined for a zero source; this core fixes them
            // deterministically: ZF reports the zero source, all other
            // affected flags clear.
            0xBC | 0xBD => {
                let (m, op) = self.modrm(bus)?;
                let src = self.read_op(bus, op)?;
                let f = &mut self.regs.rflags;
                f.remove(RFlags::CF | RFlags::OF | RFlags::AF | RFlags::SF | RFlags::PF);
                if src == 0 {
                    f.insert(RFlags::ZF);
                } else {
                    f.remove(RFlags::ZF);
                    let idx = if opcode == 0xBC {
                        src.trailing_zeros() as u64
                    } else {
                        63 - src.leading_zeros() as u64
                    };
                    self.write_reg_osize(m.reg(), idx);
                }
                Ok(11)
            }

            // --- Group 9: CMPXCHG8B / CMPXCHG16B ----------------------------------------------------------
            0xC7 => self.group9(bus),

            // --- BSWAP -------------------------------------------------------------------------------------
            0xC8..=0xCF => {
                let i = (opcode & 7) | self.rex_b() << 3;
                match self.osize {
                    O64 => {
                        let v = self.regs.reg64(i);
                        self.regs.set_reg64(i, v.swap_bytes());
                    }
                    O32 => {
                        let v = self.regs.reg32(i);
                        self.regs.set_reg32(i, v.swap_bytes());
                    }
                    // 16-bit BSWAP is architecturally undefined; common
                    // silicon zeroes the register, which this core adopts.
                    O16 => self.regs.set_reg16(i, 0),
                }
                Ok(6)
            }

            _ => Err(Exception::ud()),
        }
    }

    // --- Helpers ----------------------------------------------------------------

    /// Fetch a ModRM byte whose `mod` field is ignored and whose `rm` names
    /// a GPR (the MOV-to/from-CR/DR forms). No displacement follows.
    fn modrm_reg_only<B: Bus>(&mut self, bus: &mut B) -> Exec<ModRm> {
        Ok(ModRm {
            byte: self.fetch8(bus)?,
            r: self.rex_r() << 3,
            b: self.rex_b() << 3,
        })
    }

    /// Read a GPR at the mode's control-register width (64-bit in long
    /// mode, 32-bit zero-extended in legacy modes).
    fn read_gpr_mode(&self, i: u8) -> u64 {
        if self.m64 {
            self.regs.reg64(i)
        } else {
            self.regs.reg32(i) as u64
        }
    }

    /// Write a GPR at the mode's control-register width.
    fn write_gpr_mode(&mut self, i: u8, v: u64) {
        if self.m64 {
            self.regs.set_reg64(i, v);
        } else {
            self.regs.set_reg32(i, v as u32);
        }
    }

    /// MOV to control register `n`, with the transition rules.
    fn write_cr(&mut self, n: u8, v: u64) -> Exec<()> {
        match n {
            0 => {
                // Reserved-high bits and PG-without-PE reject.
                if v >> 32 != 0 || (v & cr0::PG != 0 && v & cr0::PE == 0) {
                    return Err(Exception::gp(0));
                }
                let old = self.regs.cr0;
                // Long-mode activation: enabling paging with EFER.LME
                // demands PAE; success sets LMA. Disabling paging leaves
                // long mode.
                if v & cr0::PG != 0 && old & cr0::PG == 0 && self.regs.msr.efer & efer::LME != 0 {
                    if self.regs.cr4 & cr4::PAE == 0 {
                        return Err(Exception::gp(0));
                    }
                    self.regs.msr.efer |= efer::LMA;
                }
                if v & cr0::PG == 0 && old & cr0::PG != 0 {
                    self.regs.msr.efer &= !efer::LMA;
                }
                self.regs.cr0 = v | cr0::ET;
                self.flush_tlb();
            }
            2 => self.regs.cr2 = v,
            3 => {
                // Bits above MAXPHYADDR are reserved.
                if v & !0x000F_FFFF_FFFF_FFFF != 0 {
                    return Err(Exception::gp(0));
                }
                self.regs.cr3 = v;
                self.flush_tlb();
            }
            4 => {
                if v & !cr4::SUPPORTED != 0 {
                    return Err(Exception::gp(0));
                }
                // PAE cannot be cleared while long mode is active.
                if v & cr4::PAE == 0 && self.regs.msr.efer & efer::LMA != 0 {
                    return Err(Exception::gp(0));
                }
                self.regs.cr4 = v;
                self.flush_tlb();
            }
            8 => {
                if v & !0xF != 0 {
                    return Err(Exception::gp(0));
                }
                self.regs.cr8 = v;
            }
            _ => return Err(Exception::ud()),
        }
        Ok(())
    }

    /// CPUID: an honest description of this core — a 64-bit integer machine
    /// with no x87/SSE state. Keeping SSE bits clear steers feature-probing
    /// guests onto integer code paths.
    fn cpuid(&mut self) {
        let leaf = self.regs.reg32(reg::RAX);
        let (a, b, c, d) = match leaf {
            // Vendor string "GenuineIntel", max basic leaf = 1.
            0 => (1, 0x756e_6547, 0x6c65_746e, 0x4965_6e69),
            // Family 6 / model 15 / stepping 1.
            // EDX: FPU-absent but flags below advertise the integer set:
            // TSC MSR PAE CX8 SEP? no — TSC(4) MSR(5) PAE(6) CX8(8) CMOV(15)
            // PSE(3) PGE(13) PAT(16). ECX: CX16(13).
            1 => (0x0000_06F1, 0, 1 << 13, 0x0001_A178),
            // Max extended leaf.
            0x8000_0000 => (0x8000_0008, 0, 0, 0),
            // ECX: LAHF in 64-bit (0). EDX: SYSCALL(11) NX(20) LM(29).
            0x8000_0001 => (0, 0, 1, (1 << 11) | (1 << 20) | (1 << 29)),
            // Brand string: "Remu virtual x86-64 CPU".
            0x8000_0002 => (
                u32::from_le_bytes(*b"Remu"),
                u32::from_le_bytes(*b" vir"),
                u32::from_le_bytes(*b"tual"),
                u32::from_le_bytes(*b" x86"),
            ),
            0x8000_0003 => (
                u32::from_le_bytes(*b"-64 "),
                u32::from_le_bytes(*b"CPU\0"),
                0,
                0,
            ),
            0x8000_0004 => (0, 0, 0, 0),
            // Address sizes: 52 physical, 48 virtual.
            0x8000_0008 => (0x0000_3034, 0, 0, 0),
            _ => (0, 0, 0, 0),
        };
        self.regs.set_reg32(reg::RAX, a);
        self.regs.set_reg32(reg::RBX, b);
        self.regs.set_reg32(reg::RCX, c);
        self.regs.set_reg32(reg::RDX, d);
    }

    /// #GP(0) unless we are at ring 0 (real mode qualifies; V86 never does).
    fn require_ring0(&self) -> Exec<()> {
        if self.regs.cr0 & cr0::PE != 0 && self.cpl() != 0 {
            Err(Exception::gp(0))
        } else {
            Ok(())
        }
    }

    // --- CMPXCHG / XADD -----------------------------------------------------------

    /// CMPXCHG r/m, reg: compare the accumulator with the destination
    /// (setting flags as `CMP`); if equal, store `reg` into the destination,
    /// otherwise load the destination into the accumulator.
    fn cmpxchg<B: Bus>(&mut self, bus: &mut B, byte: bool) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        self.lock_check(op.is_mem())?;
        if byte {
            let dest = self.read_op8(bus, op)?;
            let acc = self.gpr8(0);
            self.sub8(acc, dest); // flags only (CMP)
            if acc == dest {
                let src = self.gpr8(m.reg());
                self.write_op8(bus, op, src)?;
            } else {
                self.set_gpr8(0, dest);
            }
            return Ok(if op.is_mem() { 7 } else { 6 });
        }
        let dest = self.read_op(bus, op)?;
        let acc = self.acc();
        self.alu(5, acc, dest); // SUB slot: flags as CMP
        if acc == dest {
            let src = match self.osize {
                O16 => self.regs.reg16(m.reg()) as u64,
                O32 => self.regs.reg32(m.reg()) as u64,
                O64 => self.regs.reg64(m.reg()),
            };
            self.write_op(bus, op, src)?;
        } else {
            self.set_acc(dest);
        }
        Ok(if op.is_mem() { 7 } else { 6 })
    }

    /// XADD r/m, reg: `temp = dest + reg; reg = dest; dest = temp` (flags as
    /// `ADD`).
    fn xadd<B: Bus>(&mut self, bus: &mut B, byte: bool) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        self.lock_check(op.is_mem())?;
        if byte {
            let dest = self.read_op8(bus, op)?;
            let src = self.gpr8(m.reg());
            let sum = self.add8(dest, src);
            self.set_gpr8(m.reg(), dest);
            self.write_op8(bus, op, sum)?;
            return Ok(if op.is_mem() { 7 } else { 6 });
        }
        let dest = self.read_op(bus, op)?;
        let src = match self.osize {
            O16 => self.regs.reg16(m.reg()) as u64,
            O32 => self.regs.reg32(m.reg()) as u64,
            O64 => self.regs.reg64(m.reg()),
        };
        let sum = self.alu(0, dest, src); // ADD slot
        self.write_reg_osize(m.reg(), dest);
        self.write_op(bus, op, sum)?;
        Ok(if op.is_mem() { 7 } else { 6 })
    }

    /// Group 9: CMPXCHG8B (CMPXCHG16B with REX.W) — /1, memory only.
    fn group9<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        if m.sub() != 1 {
            return Err(Exception::ud());
        }
        let Operand::Mem { seg, off } = op else {
            return Err(Exception::ud());
        };
        self.lock_check(true)?;
        if self.osize == O64 {
            // CMPXCHG16B demands 16-byte alignment.
            if off & 0xF != 0 {
                return Err(Exception::gp(0));
            }
            let lo = self.read64(bus, seg, off)?;
            let hi = self.read64(bus, seg, off.wrapping_add(8))?;
            if lo == self.regs.gpr[reg::RAX as usize] && hi == self.regs.gpr[reg::RDX as usize] {
                self.write64(bus, seg, off, self.regs.gpr[reg::RBX as usize])?;
                self.write64(
                    bus,
                    seg,
                    off.wrapping_add(8),
                    self.regs.gpr[reg::RCX as usize],
                )?;
                self.regs.rflags.insert(RFlags::ZF);
            } else {
                self.regs.gpr[reg::RAX as usize] = lo;
                self.regs.gpr[reg::RDX as usize] = hi;
                self.regs.rflags.remove(RFlags::ZF);
            }
        } else {
            let wrap = self.data_wrap();
            let lo = self.read32(bus, seg, off)?;
            let hi = self.read32(bus, seg, off.wrapping_add(4) & wrap)?;
            let cur = (hi as u64) << 32 | lo as u64;
            let acc = (self.regs.reg32(reg::RDX) as u64) << 32 | self.regs.reg32(reg::RAX) as u64;
            if cur == acc {
                let nlo = self.regs.reg32(reg::RBX);
                let nhi = self.regs.reg32(reg::RCX);
                self.write32(bus, seg, off, nlo)?;
                self.write32(bus, seg, off.wrapping_add(4) & wrap, nhi)?;
                self.regs.rflags.insert(RFlags::ZF);
            } else {
                self.regs.set_reg32(reg::RAX, cur as u32);
                self.regs.set_reg32(reg::RDX, (cur >> 32) as u32);
                self.regs.rflags.remove(RFlags::ZF);
            }
        }
        Ok(10)
    }

    // --- Group 6: SLDT STR LLDT LTR VERR VERW (protected mode only) -----------

    fn group6<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        if !self.protected_mode() {
            return Err(Exception::ud());
        }
        let (m, op) = self.modrm(bus)?;
        match m.sub() {
            0 => {
                // SLDT: register destination zero-extends per operand size.
                let v = self.regs.ldtr.sel;
                match op {
                    Operand::Reg(i) => self.write_reg_osize(i, v as u64),
                    Operand::Mem { seg, off } => self.write16(bus, seg, off, v)?,
                }
                Ok(2)
            }
            1 => {
                let v = self.regs.tr.sel;
                match op {
                    Operand::Reg(i) => self.write_reg_osize(i, v as u64),
                    Operand::Mem { seg, off } => self.write16(bus, seg, off, v)?,
                }
                Ok(2)
            }
            2 => {
                // LLDT.
                self.require_ring0()?;
                let sel = self.read_op16(bus, op)?;
                self.lldt(bus, sel)?;
                Ok(20)
            }
            3 => {
                // LTR.
                self.require_ring0()?;
                let sel = self.read_op16(bus, op)?;
                self.ltr(bus, sel)?;
                Ok(23)
            }
            4 | 5 => {
                // VERR/VERW: ZF = selector is readable/writable from here.
                let sel = self.read_op16(bus, op)?;
                let ok = self.verify_seg(bus, sel, m.sub() == 5)?;
                self.regs.rflags.set(RFlags::ZF, ok);
                Ok(10)
            }
            _ => Err(Exception::ud()),
        }
    }

    fn lldt<B: Bus>(&mut self, bus: &mut B, sel: u16) -> Exec<()> {
        if sel & 0xFFFC == 0 {
            // Null LDT: valid, but unusable.
            self.regs.ldtr.sel = sel;
            self.regs.ldtr.base = 0;
            self.regs.ldtr.limit = 0;
            self.regs.ldtr.attrs = 0;
            return Ok(());
        }
        if sel & 4 != 0 {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        let d = self.read_descriptor(bus, sel)?;
        if d.attrs & 0x1F != 0x02 {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::np(sel & 0xFFFC));
        }
        self.regs.ldtr.sel = sel;
        self.regs.ldtr.base = d.base;
        self.regs.ldtr.limit = d.limit;
        self.regs.ldtr.attrs = d.attrs;
        Ok(())
    }

    fn ltr<B: Bus>(&mut self, bus: &mut B, sel: u16) -> Exec<()> {
        if sel & 0xFFFC == 0 || sel & 4 != 0 {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        let d = self.read_descriptor(bus, sel)?;
        // Available TSS: long mode only defines the 64-bit type (9); legacy
        // also accepts a 16-bit TSS (1).
        let ok = if self.long_mode() {
            d.attrs & 0x1F == 0x09
        } else {
            d.attrs & 0x1F == 0x01 || d.attrs & 0x1F == 0x09
        };
        if !ok {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::np(sel & 0xFFFC));
        }
        self.mark_tss_busy(bus, sel)?;
        self.regs.tr.sel = sel;
        self.regs.tr.base = d.base;
        self.regs.tr.limit = d.limit;
        self.regs.tr.attrs = d.attrs | 0x02; // busy
        Ok(())
    }

    /// Set the busy bit in a TSS descriptor (GDT write, as on hardware).
    fn mark_tss_busy<B: Bus>(&mut self, bus: &mut B, sel: u16) -> Exec<()> {
        let addr = self.regs.gdtr.base.wrapping_add((sel & 0xFFF8) as u64);
        let b = self.sys_read8(bus, addr.wrapping_add(5))?;
        self.sys_write8(bus, addr.wrapping_add(5), b | 0x02)
    }

    /// VERR/VERW predicate; never faults on bad selectors, just reports.
    fn verify_seg<B: Bus>(&mut self, bus: &mut B, sel: u16, write: bool) -> Exec<bool> {
        if sel & 0xFFFC == 0 {
            return Ok(false);
        }
        let Ok(d) = self.read_descriptor(bus, sel) else {
            return Ok(false);
        };
        if !d.present() {
            return Ok(false);
        }
        let attrs = d.attrs;
        let is_code = attrs & 0x18 == 0x18;
        let conforming = attrs & 0x1C == 0x1C;
        if attrs & 0x10 == 0 {
            return Ok(false); // system segment
        }
        if !conforming {
            let rpl = (sel & 3) as u8;
            if d.dpl() < self.cpl() || d.dpl() < rpl {
                return Ok(false);
            }
        }
        Ok(if write {
            !is_code && attrs & 0x02 != 0
        } else {
            !is_code || attrs & 0x02 != 0
        })
    }

    // --- Group 7: SGDT SIDT LGDT LIDT SMSW LMSW INVLPG SWAPGS RDTSCP ---------------

    fn group7<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        match m.sub() {
            0 | 1 => {
                // SGDT/SIDT (memory only): limit word + full base.
                let Operand::Mem { seg, off } = op else {
                    return Err(Exception::ud());
                };
                let t = if m.sub() == 0 {
                    self.regs.gdtr
                } else {
                    self.regs.idtr
                };
                self.write16(bus, seg, off, t.limit)?;
                if self.long_mode() {
                    self.write64(bus, seg, off.wrapping_add(2), t.base)?;
                } else {
                    self.write32(bus, seg, off.wrapping_add(2), t.base as u32)?;
                }
                Ok(9)
            }
            2 | 3 => {
                // LGDT/LIDT (memory only, ring 0): base is 8 bytes in long
                // mode; a 16-bit operand size in legacy modes loads only 24
                // base bits.
                self.require_ring0()?;
                let Operand::Mem { seg, off } = op else {
                    return Err(Exception::ud());
                };
                let limit = self.read16(bus, seg, off)?;
                let base = if self.long_mode() {
                    let b = self.read64(bus, seg, off.wrapping_add(2))?;
                    if !Self::canonical(b) {
                        return Err(Exception::gp(0));
                    }
                    b
                } else {
                    let mut b = self.read32(bus, seg, off.wrapping_add(2))?;
                    if self.osize == O16 {
                        b &= 0x00FF_FFFF;
                    }
                    b as u64
                };
                if m.sub() == 2 {
                    self.regs.gdtr = super::DescTable { base, limit };
                } else {
                    self.regs.idtr = super::DescTable { base, limit };
                }
                Ok(11)
            }
            4 => {
                // SMSW: low CR0 word (more into a wide register).
                match op {
                    Operand::Reg(i) => self.write_reg_osize(i, self.regs.cr0),
                    Operand::Mem { seg, off } => {
                        let v = self.regs.cr0 as u16;
                        self.write16(bus, seg, off, v)?;
                    }
                }
                Ok(2)
            }
            6 => {
                // LMSW: loads MP/EM/TS and can set (never clear) PE.
                self.require_ring0()?;
                let v = self.read_op16(bus, op)? as u64;
                let pe = (self.regs.cr0 | v) & cr0::PE;
                self.regs.cr0 = (self.regs.cr0 & !(cr0::MP | cr0::EM | cr0::TS | cr0::PE))
                    | (v & (cr0::MP | cr0::EM | cr0::TS))
                    | pe;
                self.flush_tlb();
                Ok(10)
            }
            7 => match op {
                Operand::Mem { seg, off } => {
                    // INVLPG: drop the TLB entry for the operand's page. The
                    // address is not access-checked.
                    self.require_ring0()?;
                    let base = if self.m64 {
                        if seg == reg::FS || seg == reg::GS {
                            self.regs.seg[seg as usize].base
                        } else {
                            0
                        }
                    } else {
                        self.regs.seg[seg as usize].base
                    };
                    self.invlpg(base.wrapping_add(off));
                    Ok(12)
                }
                Operand::Reg(rm) => match rm & 7 {
                    0 => self.swapgs(),
                    1 => {
                        // RDTSCP.
                        if self.regs.cr4 & cr4::TSD != 0 && self.cpl() != 0 {
                            return Err(Exception::gp(0));
                        }
                        let t = self.cycles;
                        self.regs.set_reg32(reg::RAX, t as u32);
                        self.regs.set_reg32(reg::RDX, (t >> 32) as u32);
                        let aux = self.regs.msr.tsc_aux;
                        self.regs.set_reg32(reg::RCX, aux);
                        Ok(5)
                    }
                    _ => Err(Exception::ud()),
                },
            },
            _ => Err(Exception::ud()),
        }
    }

    // --- LAR / LSL -------------------------------------------------------------------

    fn lar_lsl<B: Bus>(&mut self, bus: &mut B, lsl: bool) -> Exec<u32> {
        if !self.protected_mode() {
            return Err(Exception::ud());
        }
        let (m, op) = self.modrm(bus)?;
        let sel = self.read_op16(bus, op)?;
        let result = 'check: {
            if sel & 0xFFFC == 0 {
                break 'check None;
            }
            let Ok(d) = self.read_descriptor(bus, sel) else {
                break 'check None;
            };
            if !d.present() {
                break 'check None;
            }
            let is_code_data = d.attrs & 0x10 != 0;
            let styp = (d.attrs & 0x0F) as u8;
            // Valid system types; long mode drops the 16-bit ones and gates
            // without limits.
            let type_ok = if is_code_data {
                true
            } else if self.long_mode() {
                if lsl {
                    matches!(styp, 2 | 9 | 11)
                } else {
                    matches!(styp, 2 | 9 | 11 | 12)
                }
            } else if lsl {
                matches!(styp, 1 | 2 | 3 | 9 | 11)
            } else {
                matches!(styp, 1 | 2 | 3 | 4 | 5 | 9 | 11 | 12)
            };
            if !type_ok {
                break 'check None;
            }
            let conforming = d.attrs & 0x1C == 0x1C;
            if !(is_code_data && conforming) {
                let rpl = (sel & 3) as u8;
                if d.dpl() < self.cpl() || d.dpl() < rpl {
                    break 'check None;
                }
            }
            Some(d)
        };

        match result {
            Some(d) => {
                self.regs.rflags.insert(RFlags::ZF);
                let v = if lsl {
                    d.limit as u64
                } else {
                    // Access byte (and flags nibble) shifted to its
                    // descriptor position.
                    (d.attrs as u64 & 0xFF) << 8 | (d.attrs as u64 & 0x0F00) << 12
                };
                self.write_reg_osize(m.reg(), v);
            }
            None => self.regs.rflags.remove(RFlags::ZF),
        }
        Ok(11)
    }

    // --- Bit tests ---------------------------------------------------------------------

    /// BT/BTS/BTR/BTC with a register bit index: memory operands address the
    /// element containing the (signed) bit offset.
    fn bt_reg_index<B: Bus>(&mut self, bus: &mut B, sub: u8) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let idx = match self.osize {
            O16 => self.regs.reg16(m.reg()) as i16 as i64,
            O32 => self.regs.reg32(m.reg()) as i32 as i64,
            O64 => self.regs.reg64(m.reg()) as i64,
        };
        match op {
            Operand::Reg(_) => {
                let bit = idx as u64
                    & match self.osize {
                        O16 => 15,
                        O32 => 31,
                        O64 => 63,
                    };
                self.bt_apply(bus, op, sub, bit)
            }
            Operand::Mem { seg, off } => {
                // Signed displacement to the containing element; the final
                // offset wraps at the address size.
                let (disp, bit) = match self.osize {
                    O16 => (((idx >> 4) as u64).wrapping_mul(2), idx as u64 & 15),
                    O32 => (((idx >> 5) as u64).wrapping_mul(4), idx as u64 & 31),
                    O64 => (((idx >> 6) as u64).wrapping_mul(8), idx as u64 & 63),
                };
                let off = off.wrapping_add(disp) & self.data_wrap();
                let op = Operand::Mem { seg, off };
                self.bt_apply(bus, op, sub, bit)
            }
        }
    }

    /// Test bit `bit` of the operand into CF and apply `sub`
    /// (0 BT, 1 BTS, 2 BTR, 3 BTC). Only CF is architecturally defined by
    /// the result; the other arithmetic flags are left untouched here.
    fn bt_apply<B: Bus>(&mut self, bus: &mut B, op: Operand, sub: u8, bit: u64) -> Exec<u32> {
        self.lock_check(op.is_mem() && sub != 0)?;
        let v = self.read_op(bus, op)?;
        self.regs.rflags.set(RFlags::CF, v >> bit & 1 != 0);
        if sub != 0 {
            let mask = 1u64 << bit;
            let r = match sub {
                1 => v | mask,
                2 => v & !mask,
                _ => v ^ mask,
            };
            self.write_op(bus, op, r)?;
        }
        Ok(if op.is_mem() { 8 } else { 4 })
    }

    // --- SHLD / SHRD ---------------------------------------------------------------------

    /// Double-precision shift; `count` is `None` for the imm8 forms (fetched
    /// after ModRM). The count masks to 5 bits (6 for a 64-bit operand). A
    /// 16-bit operand shifted by more than 16 is architecturally undefined;
    /// this core computes it from a `dst:src` window (the source refeeds),
    /// a deterministic choice.
    fn shxd<B: Bus>(&mut self, bus: &mut B, right: bool, count: Option<u32>) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let dst = self.read_op(bus, op)?;
        let src = match self.osize {
            O16 => self.regs.reg16(m.reg()) as u64,
            O32 => self.regs.reg32(m.reg()) as u64,
            O64 => self.regs.reg64(m.reg()),
        };
        let n = match count {
            Some(c) => c,
            None => self.fetch8(bus)? as u32,
        } & if self.osize == O64 { 63 } else { 31 };
        if n == 0 {
            return Ok(if op.is_mem() { 7 } else { 3 });
        }

        let (bits, sign, mask) = match self.osize {
            O16 => (16u32, 1u64 << 15, 0xFFFF),
            O32 => (32, 1 << 31, 0xFFFF_FFFF),
            O64 => (64, 1u64 << 63, u64::MAX),
        };

        // Two 128-bit windows cover every case: the count is < bits for 32-
        // and 64-bit operands (its mask guarantees it), and only a 16-bit
        // operand can exceed its width, which the window handles by feeding
        // the source in again.
        let (r, cf, prev) = if right {
            // SHRD: window = src:dst, shift right; CF is the last bit out of
            // the low end.
            let win = (src as u128) << bits | dst as u128;
            let r = ((win >> n) as u64) & mask;
            let cf = win >> (n - 1) & 1 != 0;
            let prev = ((win >> (n - 1)) as u64) & mask;
            (r, cf, prev)
        } else {
            // SHLD: window = dst:src in a 2*bits field; shift left, keep the
            // top `bits`. CF is dst's bit (bits - n) — the last bit out.
            let win = (dst as u128) << bits | src as u128;
            let r = ((win << n) >> bits) as u64 & mask;
            let cf = if n <= bits {
                dst >> (bits - n) & 1 != 0
            } else {
                src >> (2 * bits - n) & 1 != 0
            };
            (r, cf, 0)
        };

        let f = &mut self.regs.rflags;
        f.set(RFlags::CF, cf);
        if right {
            // OF: the sign bit changed in the last 1-bit step.
            f.set(RFlags::OF, (r ^ prev) & sign != 0);
        } else {
            f.set(RFlags::OF, (r & sign != 0) != cf);
        }
        f.remove(RFlags::AF);
        match self.osize {
            O16 => f.set_szp16(r as u16),
            O32 => f.set_szp32(r as u32),
            O64 => f.set_szp64(r),
        }
        self.write_op(bus, op, r)?;
        Ok(if op.is_mem() { 7 } else { 3 })
    }
}
