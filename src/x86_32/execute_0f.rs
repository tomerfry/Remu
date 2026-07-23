//! Two-byte (`0F xx`) opcode dispatch: the 386 system instructions, bit
//! operations, MOVZX/MOVSX, SETcc, SHLD/SHRD and long conditional jumps.
//! Encodings the 386 does not define raise #UD.

use super::modrm::{ModRm, Operand};
use super::registers::{EFlags, cr0, reg};
use super::{Bus, Cpu, Exception, Exec};

impl Cpu {
    pub(crate) fn dispatch_0f<B: Bus>(&mut self, bus: &mut B, opcode: u8) -> Exec<u32> {
        self.stat(|s| s.opcode_0f_hist[opcode as usize] += 1);
        // Decode-time LOCK legality: only BTS/BTR/BTC (and group 8) can lock —
        // plus CMPXCHG/XADD, the atomic primitives, when extensions are on.
        let lock_atomic = self.extensions && matches!(opcode, 0xB0 | 0xB1 | 0xC0 | 0xC1);
        if self.lock && !lock_atomic && !matches!(opcode, 0xAB | 0xB3 | 0xBB | 0xBA) {
            return Err(Exception::ud());
        }
        match opcode {
            0x00 => self.group6(bus),
            0x01 => self.group7(bus),
            0x02 => self.lar_lsl(bus, false),
            0x03 => self.lar_lsl(bus, true),
            0x06 => {
                // CLTS.
                self.require_ring0()?;
                self.prepare_cold_write(); // cr0
                self.regs.cr0 &= !cr0::TS;
                Ok(5)
            }

            // --- MOV to/from control, debug and test registers ---------------
            // The ModRM mod field is ignored: the operand is always a GPR and
            // no displacement is fetched.
            0x20 => {
                self.require_ring0()?;
                let m = ModRm(self.fetch8(bus)?);
                let v = match m.reg() {
                    0 => self.regs.cr0,
                    2 => self.regs.cr2,
                    3 => self.regs.cr3,
                    _ => return Err(Exception::ud()),
                };
                self.regs.set_reg32(m.rm(), v);
                Ok(6)
            }
            0x22 => {
                self.require_ring0()?;
                self.prepare_cold_write(); // cr0/cr3
                let m = ModRm(self.fetch8(bus)?);
                let v = self.regs.reg32(m.rm());
                match m.reg() {
                    0 => {
                        // Enabling paging requires protection.
                        if v & cr0::PG != 0 && v & cr0::PE == 0 {
                            return Err(Exception::gp(0));
                        }
                        self.regs.cr0 = v;
                        self.flush_tlb();
                    }
                    2 => self.regs.cr2 = v,
                    3 => {
                        self.regs.cr3 = v;
                        self.flush_tlb();
                    }
                    _ => return Err(Exception::ud()),
                }
                Ok(10)
            }
            0x21 => {
                self.require_ring0()?;
                let m = ModRm(self.fetch8(bus)?);
                let n = match m.reg() {
                    4 => 6,
                    5 => 7,
                    n => n,
                } as usize;
                self.regs.set_reg32(m.rm(), self.regs.dr[n]);
                Ok(14)
            }
            0x23 => {
                self.require_ring0()?;
                self.prepare_cold_write(); // dr
                let m = ModRm(self.fetch8(bus)?);
                let n = match m.reg() {
                    4 => 6,
                    5 => 7,
                    n => n,
                } as usize;
                self.regs.dr[n] = self.regs.reg32(m.rm());
                Ok(16)
            }
            0x24 => {
                self.require_ring0()?;
                let m = ModRm(self.fetch8(bus)?);
                let v = match m.reg() {
                    6 => self.regs.tr6,
                    7 => self.regs.tr7,
                    _ => return Err(Exception::ud()),
                };
                self.regs.set_reg32(m.rm(), v);
                Ok(12)
            }
            0x26 => {
                self.require_ring0()?;
                self.prepare_cold_write(); // tr6/tr7
                let m = ModRm(self.fetch8(bus)?);
                let v = self.regs.reg32(m.rm());
                match m.reg() {
                    6 => self.regs.tr6 = v,
                    7 => self.regs.tr7 = v,
                    _ => return Err(Exception::ud()),
                }
                Ok(12)
            }

            // --- Long conditional jumps ----------------------------------------
            0x80..=0x8F => {
                let rel = if self.osize32 {
                    self.fetch32(bus)? as i32
                } else {
                    self.fetch16(bus)? as i16 as i32
                };
                let n = opcode & 0xF;
                let taken = self.cond(n);
                #[cfg(feature = "symbolic")]
                if self.sym_active() {
                    self.sym_branch(n, taken);
                }
                if taken {
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

            // --- PUSH/POP FS/GS ----------------------------------------------------
            0xA0 | 0xA8 => {
                let idx = if opcode == 0xA0 { reg::FS } else { reg::GS };
                let v = self.regs.seg_sel(idx) as u32;
                self.push(bus, v)?;
                Ok(2)
            }
            0xA1 | 0xA9 => {
                let idx = if opcode == 0xA1 { reg::FS } else { reg::GS };
                let v = self.pop_sreg(bus)?;
                self.load_seg(bus, idx, v)?;
                Ok(7)
            }

            // --- Bit tests -------------------------------------------------------------
            0xA3 => self.bt_reg_index(bus, 0), // BT
            0xAB => self.bt_reg_index(bus, 1), // BTS
            0xB3 => self.bt_reg_index(bus, 2), // BTR
            0xBB => self.bt_reg_index(bus, 3), // BTC
            0xBA => {
                // Group 8: BT/BTS/BTR/BTC r/m, imm8 (/4../7).
                let (m, op) = self.modrm(bus)?;
                if m.reg() < 4 {
                    return Err(Exception::ud());
                }
                let imm = self.fetch8(bus)? as u32;
                self.bt_apply(
                    bus,
                    op,
                    m.reg() - 4,
                    imm & if self.osize32 { 31 } else { 15 },
                )
            }

            // --- SHLD / SHRD -----------------------------------------------------------
            0xA4 => self.shxd(bus, false, None),
            0xA5 => {
                let c = self.regs.reg8(1) as u32;
                self.shxd(bus, false, Some(c))
            }
            0xAC => self.shxd(bus, true, None),
            0xAD => {
                let c = self.regs.reg8(1) as u32;
                self.shxd(bus, true, Some(c))
            }

            // --- IMUL r, r/m --------------------------------------------------------------
            0xAF => {
                let (m, op) = self.modrm(bus)?;
                let src = self.read_op(bus, op)?;
                if self.osize32 {
                    let a = self.regs.reg32(m.reg());
                    let r = self.imul_trunc32(a, src);
                    self.regs.set_reg32(m.reg(), r);
                } else {
                    let a = self.regs.reg16(m.reg());
                    let r = self.imul_trunc16(a, src as u16);
                    self.regs.set_reg16(m.reg(), r);
                }
                Ok(if op.is_mem() { 22 } else { 20 })
            }

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
                let v = self.read_op8(bus, op)? as u32;
                if self.osize32 {
                    self.regs.set_reg32(m.reg(), v);
                } else {
                    self.regs.set_reg16(m.reg(), v as u16);
                }
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            0xB7 => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op16(bus, op)? as u32;
                if self.osize32 {
                    self.regs.set_reg32(m.reg(), v);
                } else {
                    self.regs.set_reg16(m.reg(), v as u16);
                }
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            0xBE => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op8(bus, op)? as i8 as i32 as u32;
                if self.osize32 {
                    self.regs.set_reg32(m.reg(), v);
                } else {
                    self.regs.set_reg16(m.reg(), v as u16);
                }
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            0xBF => {
                let (m, op) = self.modrm(bus)?;
                let v = self.read_op16(bus, op)? as i16 as i32 as u32;
                if self.osize32 {
                    self.regs.set_reg32(m.reg(), v);
                } else {
                    self.regs.set_reg16(m.reg(), v as u16);
                }
                Ok(if op.is_mem() { 6 } else { 3 })
            }

            // --- BSF / BSR -----------------------------------------------------------------------------
            // The "undefined" flags are fully determined on the 386 (and
            // asserted by the test suite); the formulas below are derived
            // from hardware captures. A source of 0 leaves the destination
            // unchanged and sets flags as if the result were 0.
            0xBC | 0xBD => {
                let (m, op) = self.modrm(bus)?;
                let src = self.read_op(bus, op)?;
                let src = if self.osize32 { src } else { src & 0xFFFF };
                let w = if self.osize32 { 32u32 } else { 16 };
                let sign = 1u32 << (w - 1);
                if src == 0 {
                    let f = &mut self.regs.eflags;
                    f.remove(EFlags::CF | EFlags::OF | EFlags::AF | EFlags::SF);
                    f.insert(EFlags::ZF | EFlags::PF);
                    return Ok(11);
                }
                let neg = src.wrapping_neg() & (sign | (sign - 1));
                let pf_neg = (neg as u8).count_ones().is_multiple_of(2);
                if opcode == 0xBC {
                    let idx = src.trailing_zeros();
                    let f = &mut self.regs.eflags;
                    f.remove(EFlags::ZF);
                    if idx == 0 {
                        // No scan steps ran: the flags carry the signature
                        // of the internal negate.
                        f.set(EFlags::AF, true);
                        f.set(EFlags::CF, src >> 1 & 1 != 0);
                        f.set(EFlags::OF, src & sign != 0);
                        f.set(EFlags::SF, src & sign == 0);
                        f.set(EFlags::PF, pf_neg);
                    } else {
                        f.remove(EFlags::CF | EFlags::OF | EFlags::AF | EFlags::SF);
                        f.set(EFlags::PF, idx.count_ones().is_multiple_of(2));
                    }
                    if self.osize32 {
                        self.regs.set_reg32(m.reg(), idx);
                    } else {
                        self.regs.set_reg16(m.reg(), idx as u16);
                    }
                } else {
                    let idx = 31 - src.leading_zeros();
                    let sh = w - 1 - idx;
                    let a = src << sh; // the found bit aligned to the MSB
                    let f = &mut self.regs.eflags;
                    f.remove(EFlags::ZF);
                    f.set(EFlags::SF, neg & sign != 0);
                    f.set(EFlags::CF, a & (sign >> 1) != 0);
                    f.set(
                        EFlags::OF,
                        idx == 0 || ((a >> (w - 2)) ^ (a >> (w - 3))) & 1 != 0,
                    );
                    f.set(EFlags::PF, pf_neg);
                    f.set(EFlags::AF, src & 0xF != 0);
                    if self.osize32 {
                        self.regs.set_reg32(m.reg(), idx);
                    } else {
                        self.regs.set_reg16(m.reg(), idx as u16);
                    }
                }
                Ok(11)
            }

            // --- Post-386 extensions (opt-in via `Cpu::extensions`) -----------
            // Real 32-bit binaries and glibc use these; a strict 386 raises
            // #UD, so they are gated to preserve conformance.
            0x18..=0x1F if self.extensions => {
                // Long NOP (0F 1F) and the prefetch/hint group (0F 18..0F 1E):
                // consume the ModRM operand, do nothing.
                let _ = self.modrm(bus)?;
                Ok(3)
            }
            0x31 if self.extensions => {
                // RDTSC: EDX:EAX = timestamp counter (our cycle count).
                self.regs.set_reg32(reg::EAX, self.cycles as u32);
                self.regs.set_reg32(reg::EDX, (self.cycles >> 32) as u32);
                Ok(5)
            }
            0x40..=0x4F if self.extensions => self.cmovcc(bus, opcode & 0x0F),
            0xA2 if self.extensions => {
                self.cpuid();
                Ok(14)
            }
            0xB0 | 0xB1 if self.extensions => self.cmpxchg(bus, opcode == 0xB0),
            0xC0 | 0xC1 if self.extensions => self.xadd(bus, opcode == 0xC0),
            0xC8..=0xCF if self.extensions => {
                // BSWAP r32: reverse byte order (16-bit form is undefined).
                let i = opcode & 0x07;
                let v = self.regs.reg32(i);
                self.regs.set_reg32(i, v.swap_bytes());
                Ok(6)
            }

            _ => Err(Exception::ud()),
        }
    }

    // --- Post-386 extension helpers ------------------------------------------

    /// CMOVcc: read the source (always, per hardware) and move it to the
    /// destination register only when the condition holds.
    fn cmovcc<B: Bus>(&mut self, bus: &mut B, cc: u8) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let v = self.read_op(bus, op)?;
        if self.cond(cc) {
            if self.osize32 {
                self.regs.set_reg32(m.reg(), v);
            } else {
                self.regs.set_reg16(m.reg(), v as u16);
            }
        }
        Ok(if op.is_mem() { 5 } else { 4 })
    }

    /// CMPXCHG r/m, reg: compare the accumulator with the destination (setting
    /// flags as `CMP`); if equal, store `reg` into the destination, otherwise
    /// load the destination into the accumulator.
    fn cmpxchg<B: Bus>(&mut self, bus: &mut B, byte: bool) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        self.lock_check(op.is_mem())?;
        if byte {
            let dest = self.read_op8(bus, op)?;
            let acc = self.regs.reg8(reg::EAX);
            self.sub8(acc, dest); // flags only (CMP)
            if acc == dest {
                let src = self.regs.reg8(m.reg());
                self.write_op8(bus, op, src)?;
            } else {
                self.regs.set_reg8(reg::EAX, dest);
            }
        } else if self.osize32 {
            let dest = self.read_op32(bus, op)?;
            let acc = self.regs.reg32(reg::EAX);
            self.sub32(acc, dest);
            if acc == dest {
                let src = self.regs.reg32(m.reg());
                self.write_op32(bus, op, src)?;
            } else {
                self.regs.set_reg32(reg::EAX, dest);
            }
        } else {
            let dest = self.read_op16(bus, op)?;
            let acc = self.regs.reg16(reg::EAX);
            self.sub16(acc, dest);
            if acc == dest {
                let src = self.regs.reg16(m.reg());
                self.write_op16(bus, op, src)?;
            } else {
                self.regs.set_reg16(reg::EAX, dest);
            }
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
            let src = self.regs.reg8(m.reg());
            let sum = self.add8(dest, src);
            self.regs.set_reg8(m.reg(), dest);
            self.write_op8(bus, op, sum)?;
        } else if self.osize32 {
            let dest = self.read_op32(bus, op)?;
            let src = self.regs.reg32(m.reg());
            let sum = self.add32(dest, src);
            self.regs.set_reg32(m.reg(), dest);
            self.write_op32(bus, op, sum)?;
        } else {
            let dest = self.read_op16(bus, op)?;
            let src = self.regs.reg16(m.reg());
            let sum = self.add16(dest, src);
            self.regs.set_reg16(m.reg(), dest);
            self.write_op16(bus, op, sum)?;
        }
        Ok(if op.is_mem() { 7 } else { 6 })
    }

    /// CPUID: a deliberately minimal descriptor. Advertising no SSE/MMX/CX8
    /// keeps glibc's IFUNC resolvers on the generic i386/i686 code paths,
    /// shrinking the opcode surface we must support. FPU is advertised so
    /// glibc does not bail, but note the core has no x87 arithmetic yet.
    fn cpuid(&mut self) {
        let leaf = self.regs.reg32(reg::EAX);
        let (a, b, c, d) = match leaf {
            // Vendor string "GenuineIntel", max basic leaf = 1.
            0 => (1, 0x756e_6547, 0x6c65_746e, 0x4965_6e69),
            // Family 5 / model 1 / stepping 1; features: FPU | TSC | CMOV.
            1 => (0x0000_0511, 0, 0, 0x0000_8011),
            _ => (0, 0, 0, 0),
        };
        self.regs.set_reg32(reg::EAX, a);
        self.regs.set_reg32(reg::EBX, b);
        self.regs.set_reg32(reg::ECX, c);
        self.regs.set_reg32(reg::EDX, d);
    }

    /// #GP(0) unless we are at ring 0 (real mode qualifies; V86 never does).
    fn require_ring0(&self) -> Exec<()> {
        if self.regs.cr0 & cr0::PE != 0 && self.cpl() != 0 {
            Err(Exception::gp(0))
        } else {
            Ok(())
        }
    }

    // --- Group 6: SLDT STR LLDT LTR VERR VERW (protected mode only) -----------

    fn group6<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        if !self.protected_mode() {
            return Err(Exception::ud());
        }
        let (m, op) = self.modrm(bus)?;
        match m.reg() {
            0 => {
                // SLDT: register destination zero-extends per operand size.
                let v = self.regs.ldtr.sel;
                match op {
                    Operand::Reg(i) if self.osize32 => self.regs.set_reg32(i, v as u32),
                    Operand::Reg(i) => self.regs.set_reg16(i, v),
                    Operand::Mem { seg, off } => self.write16(bus, seg, off, v)?,
                }
                Ok(2)
            }
            1 => {
                let v = self.regs.tr.sel;
                match op {
                    Operand::Reg(i) if self.osize32 => self.regs.set_reg32(i, v as u32),
                    Operand::Reg(i) => self.regs.set_reg16(i, v),
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
                let ok = self.verify_seg(bus, sel, m.reg() == 5)?;
                self.regs.eflags.set(EFlags::ZF, ok);
                Ok(10)
            }
            _ => Err(Exception::ud()),
        }
    }

    fn lldt<B: Bus>(&mut self, bus: &mut B, sel: u16) -> Exec<()> {
        self.prepare_cold_write(); // ldtr
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
        if d.attrs & 0x80 == 0 {
            return Err(Exception::np(sel & 0xFFFC));
        }
        self.regs.ldtr.sel = sel;
        self.regs.ldtr.base = d.base;
        self.regs.ldtr.limit = d.limit;
        self.regs.ldtr.attrs = d.attrs;
        Ok(())
    }

    fn ltr<B: Bus>(&mut self, bus: &mut B, sel: u16) -> Exec<()> {
        self.prepare_cold_write(); // tr
        if sel & 0xFFFC == 0 || sel & 4 != 0 {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        let d = self.read_descriptor(bus, sel)?;
        // Available TSS (16- or 32-bit).
        if d.attrs & 0x1F != 0x01 && d.attrs & 0x1F != 0x09 {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if d.attrs & 0x80 == 0 {
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
        let addr = self.regs.gdtr.base.wrapping_add((sel & 0xFFF8) as u32);
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

    // --- Group 7: SGDT SIDT LGDT LIDT SMSW LMSW ----------------------------------

    fn group7<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        match m.reg() {
            0 | 1 => {
                // SGDT/SIDT (memory only): limit word + full 32-bit base.
                let Operand::Mem { seg, off } = op else {
                    return Err(Exception::ud());
                };
                let t = if m.reg() == 0 {
                    self.regs.gdtr
                } else {
                    self.regs.idtr
                };
                self.write16(bus, seg, off, t.limit)?;
                self.write32(bus, seg, off.wrapping_add(2), t.base)?;
                Ok(9)
            }
            2 | 3 => {
                // LGDT/LIDT (memory only, ring 0): a 16-bit operand size
                // loads only 24 base bits.
                self.require_ring0()?;
                let Operand::Mem { seg, off } = op else {
                    return Err(Exception::ud());
                };
                let limit = self.read16(bus, seg, off)?;
                let mut base = self.read32(bus, seg, off.wrapping_add(2))?;
                if !self.osize32 {
                    base &= 0x00FF_FFFF;
                }
                self.prepare_cold_write(); // gdtr/idtr
                if m.reg() == 2 {
                    self.regs.gdtr = super::DescTable { base, limit };
                } else {
                    self.regs.idtr = super::DescTable { base, limit };
                }
                Ok(11)
            }
            4 => {
                // SMSW: low CR0 word (full CR0 into a 32-bit register).
                match op {
                    Operand::Reg(i) if self.osize32 => self.regs.set_reg32(i, self.regs.cr0),
                    Operand::Reg(i) => self.regs.set_reg16(i, self.regs.cr0 as u16),
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
                let v = self.read_op16(bus, op)? as u32;
                self.prepare_cold_write(); // cr0
                let pe = (self.regs.cr0 | v) & cr0::PE;
                self.regs.cr0 = (self.regs.cr0 & !(cr0::MP | cr0::EM | cr0::TS | cr0::PE))
                    | (v & (cr0::MP | cr0::EM | cr0::TS))
                    | pe;
                self.flush_tlb();
                Ok(10)
            }
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
            // Valid system types: TSS/LDT/gates for LAR; things with limits for LSL.
            let type_ok = if is_code_data {
                true
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
                self.regs.eflags.insert(EFlags::ZF);
                if lsl {
                    let v = d.limit;
                    if self.osize32 {
                        self.regs.set_reg32(m.reg(), v);
                    } else {
                        self.regs.set_reg16(m.reg(), v as u16);
                    }
                } else {
                    // Access byte (and flags nibble for 32-bit) shifted to
                    // its descriptor position.
                    let v = (d.attrs as u32 & 0xFF) << 8 | (d.attrs as u32 & 0x0F00) << 12;
                    if self.osize32 {
                        self.regs.set_reg32(m.reg(), v);
                    } else {
                        self.regs.set_reg16(m.reg(), v as u16);
                    }
                }
            }
            None => self.regs.eflags.remove(EFlags::ZF),
        }
        Ok(11)
    }

    // --- Bit tests ---------------------------------------------------------------------

    /// BT/BTS/BTR/BTC with a register bit index: memory operands address the
    /// word/dword containing the (signed) bit offset.
    fn bt_reg_index<B: Bus>(&mut self, bus: &mut B, sub: u8) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let idx = if self.osize32 {
            self.regs.reg32(m.reg())
        } else {
            self.regs.reg16(m.reg()) as i16 as i32 as u32
        };
        match op {
            Operand::Reg(_) => {
                let bit = idx & if self.osize32 { 31 } else { 15 };
                self.bt_apply(bus, op, sub, bit)
            }
            Operand::Mem { seg, off } => {
                // Signed displacement to the containing element; the final
                // offset wraps at the address size (16-bit EAs stay in 64K).
                let (disp, bit) = if self.osize32 {
                    (((idx as i32) >> 5).wrapping_mul(4) as u32, idx & 31)
                } else {
                    (((idx as i32) >> 4).wrapping_mul(2) as u32, idx & 15)
                };
                let off = off.wrapping_add(disp) & self.data_wrap();
                let op = Operand::Mem { seg, off };
                self.bt_apply(bus, op, sub, bit)
            }
        }
    }

    /// Test bit `bit` of the operand into CF and apply `sub`
    /// (0 BT, 1 BTS, 2 BTR, 3 BTC).
    ///
    /// The 386 routes the operand through its rotator (`temp = v ror bit`),
    /// which is architecturally visible: CF is bit 0 of the rotation and OF
    /// gets the ROR top-two-bits rule; the other flags are untouched.
    fn bt_apply<B: Bus>(&mut self, bus: &mut B, op: Operand, sub: u8, bit: u32) -> Exec<u32> {
        self.lock_check(op.is_mem() && sub != 0)?;
        let v = self.read_op(bus, op)?;
        let t = if self.osize32 {
            v.rotate_right(bit)
        } else {
            (v as u16).rotate_right(bit) as u32
        };
        let sign = if self.osize32 { 0x8000_0000 } else { 0x8000 };
        let f = &mut self.regs.eflags;
        f.set(EFlags::CF, t & 1 != 0);
        f.set(EFlags::OF, (t ^ (t << 1)) & sign != 0);
        if sub != 0 {
            let mask = 1u32 << bit;
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
    /// after ModRM).
    ///
    /// Hardware-exact 386 semantics (asserted by the test suite): counts
    /// past a 16-bit operand keep pulling bits from the *source* (the
    /// shifter input is dst:src:src); AF is always set; OF is CF^MSB(result)
    /// for left shifts and "MSB changed in the last 1-bit step" for right
    /// shifts; SZP follow the result.
    fn shxd<B: Bus>(&mut self, bus: &mut B, right: bool, count: Option<u32>) -> Exec<u32> {
        let (m, op) = self.modrm(bus)?;
        let dst = self.read_op(bus, op)?;
        let src = if self.osize32 {
            self.regs.reg32(m.reg())
        } else {
            self.regs.reg16(m.reg()) as u32
        };
        let n = match count {
            Some(c) => c,
            None => self.fetch8(bus)? as u32,
        } & 31;
        if n == 0 {
            return Ok(if op.is_mem() { 7 } else { 3 });
        }

        let r = if self.osize32 {
            let (r, cf) = if right {
                let step = |k: u32| -> u32 {
                    if k == 0 {
                        dst
                    } else {
                        (dst >> k) | (src << (32 - k))
                    }
                };
                (step(n), dst >> (n - 1) & 1 != 0)
            } else {
                ((dst << n) | (src >> (32 - n)), dst >> (32 - n) & 1 != 0)
            };
            let f = &mut self.regs.eflags;
            f.set(EFlags::CF, cf);
            if right {
                let prev = if n == 1 {
                    dst
                } else {
                    (dst >> (n - 1)) | (src << (33 - n))
                };
                f.set(EFlags::OF, (r ^ prev) & 0x8000_0000 != 0);
            } else {
                f.set(EFlags::OF, (r & 0x8000_0000 != 0) != cf);
            }
            f.insert(EFlags::AF);
            f.set_szp32(r);
            r
        } else {
            let (dst, src) = (dst as u16, src as u16);
            let step = |k: u32| -> u16 {
                match k {
                    0 => dst,
                    1..=16 => (((dst as u32) >> k) as u16) | ((src as u32) << (16 - k)) as u16,
                    _ => (src >> (k - 16)) | ((src as u32) << (32 - k)) as u16,
                }
            };
            let lstep = |k: u32| -> (u16, bool) {
                if k <= 16 {
                    let r = ((dst as u32) << k | (src as u32) >> (16 - k).min(31)) as u16;
                    let r = if k == 16 { src } else { r };
                    (r, (dst as u32) >> (16 - k) & 1 != 0)
                } else {
                    (
                        ((src as u32) << (k - 16) | (src as u32) >> (32 - k)) as u16,
                        src >> (32 - k) & 1 != 0,
                    )
                }
            };
            let (r, cf) = if right {
                (
                    step(n),
                    if n <= 16 {
                        dst >> (n - 1) & 1 != 0
                    } else {
                        src >> (n - 17) & 1 != 0
                    },
                )
            } else {
                lstep(n)
            };
            let f = &mut self.regs.eflags;
            f.set(EFlags::CF, cf);
            if right {
                let prev = step(n - 1);
                f.set(EFlags::OF, (r ^ prev) & 0x8000 != 0);
            } else {
                f.set(EFlags::OF, (r & 0x8000 != 0) != cf);
            }
            f.insert(EFlags::AF);
            f.set_szp16(r);
            r as u32
        };
        self.write_op(bus, op, r)?;
        Ok(if op.is_mem() { 7 } else { 3 })
    }
}
