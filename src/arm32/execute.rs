//! The executor: one dispatch over [`DecodedInsn`] runs both ARM and Thumb
//! code, cached or freshly decoded.
//!
//! Cycle figures are the ARM7TDMI datasheet's nominal counts with S, N and
//! I cycles collapsed to 1 — rough relative weights in the spirit of the
//! other cores, not bus-accurate timings.

use super::alu::{add_with_carry, mul_cycles, shift_imm, shift_reg, sub_with_carry};
use super::decode::{
    BR_LINK, DecodedInsn, Op, PCA, TBL_SUFFIX, alu_op, blk, dp, hw, mul, sr, xfer,
};
use super::registers::psr;
use super::{Bus, Cpu, Exception, Exec, HostTrap};

impl Cpu {
    /// Evaluate condition field `cond` against the CPSR (15 = never, the
    /// ARM7TDMI's reading of the NV space).
    #[inline]
    pub(crate) fn cond_pass(&self, cond: u8) -> bool {
        let p = self.regs.cpsr;
        match cond {
            0x0 => p.z(),
            0x1 => !p.z(),
            0x2 => p.c(),
            0x3 => !p.c(),
            0x4 => p.n(),
            0x5 => !p.n(),
            0x6 => p.v(),
            0x7 => !p.v(),
            0x8 => p.c() && !p.z(),
            0x9 => !p.c() || p.z(),
            0xA => p.n() == p.v(),
            0xB => p.n() != p.v(),
            0xC => !p.z() && p.n() == p.v(),
            0xD => p.z() || p.n() != p.v(),
            0xE => true,
            _ => false,
        }
    }

    /// Read register `i` as an operand: R15 yields the pipeline value
    /// (`pc + 8` in ARM state, `pc + 4` in Thumb), the Thumb [`PCA`]
    /// sentinel the word-aligned variant.
    #[inline]
    pub(crate) fn reg_op(&self, i: u8) -> u32 {
        match i {
            15 => self.pc_op(),
            PCA => self.pc_op() & !3,
            _ => self.regs.gpr[i as usize],
        }
    }

    /// Read register `i` under the register-specified-shift quirk: R15
    /// reads as `pc + 12` (the extra internal cycle advances the prefetch).
    #[inline]
    fn reg_op12(&self, i: u8) -> u32 {
        if i == 15 {
            self.start_pc.wrapping_add(12)
        } else {
            self.regs.gpr[i as usize]
        }
    }

    /// The R15 operand value: address of the current instruction plus the
    /// two-stage prefetch depth.
    #[inline]
    pub(crate) fn pc_op(&self) -> u32 {
        self.start_pc.wrapping_add(if self.thumb { 4 } else { 8 })
    }

    /// The value STR/STM store for R15 (`pc + 12` on the ARM7TDMI).
    #[inline]
    fn pc_store(&self) -> u32 {
        self.start_pc.wrapping_add(if self.thumb { 6 } else { 12 })
    }

    /// Write register `i`, branching (with state-appropriate alignment)
    /// when it is the PC.
    #[inline]
    fn set_reg(&mut self, i: u8, v: u32) {
        if i == 15 {
            self.branch_to(v);
        } else {
            self.regs.gpr[i as usize] = v;
        }
    }

    /// Load the PC with `target`. In ARM state the raw value is stored —
    /// the ARM7TDMI keeps whatever was written and its prefetch unit
    /// aligns the fetch address instead (`exec_one` does the same); in
    /// Thumb state the pipeline reload drops bit 0. Both suite-pinned.
    #[inline]
    pub(crate) fn branch_to(&mut self, target: u32) {
        self.regs.gpr[15] = if self.regs.cpsr.t() {
            target & !1
        } else {
            target
        };
    }

    /// Restore the CPSR from the current mode's SPSR (the `S`-bit PC-write
    /// forms), switching register banks with it. The SPSR image is copied
    /// raw; in User/System — which have no SPSR — nothing happens and the
    /// caller's plain flag write (if any) stands (suite-pinned).
    fn restore_cpsr(&mut self) {
        let Some(bits) = self.regs.spsr_banked() else {
            return;
        };
        let next = super::registers::Psr::from_bits(bits);
        self.regs.set_mode(next.mode());
        self.regs.cpsr = next;
    }

    /// Execute one decoded instruction. The PC has already advanced past
    /// it; `self.start_pc`/`self.thumb` describe it. Returns the cycles
    /// consumed, or the exception to deliver.
    pub(crate) fn exec_decoded<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        if !self.cond_pass(d.cond) {
            return Ok(1);
        }
        match d.op {
            Op::DpImm | Op::DpShImm | Op::DpShReg => self.exec_dp(d),
            Op::Mul => Ok(self.exec_mul(d)),
            Op::Mull => Ok(self.exec_mull(d)),
            Op::Mrs => {
                let v = if d.aux & sr::SPSR != 0 {
                    self.regs.spsr_bits()
                } else {
                    self.regs.cpsr.bits()
                };
                if d.rd == 15 {
                    // MRS never reloads the pipeline: the written value
                    // slips one fetch slot, so execution continues at
                    // `value - 4` (suite-pinned).
                    self.regs.gpr[15] = v.wrapping_sub(4);
                } else {
                    self.regs.gpr[d.rd as usize] = v;
                }
                Ok(1)
            }
            Op::MsrReg => Ok(self.exec_msr(self.reg_op(d.rm), d.aux)),
            Op::MsrImm => Ok(self.exec_msr(d.imm, d.aux)),
            Op::BranchImm => {
                let target = self.pc_op().wrapping_add(d.imm);
                if d.aux & BR_LINK != 0 {
                    self.regs.gpr[14] = self.start_pc.wrapping_add(4);
                }
                self.branch_to(target);
                Ok(3)
            }
            Op::Bx => {
                let v = self.reg_op(d.rm);
                self.regs.cpsr.set(psr::T, v & 1 != 0);
                // Unlike other PC writes, BX consumes bit 0 (it became T).
                self.branch_to(v & !1);
                Ok(3)
            }
            Op::Ldr => self.exec_ldr(bus, d),
            Op::Str => self.exec_str(bus, d),
            Op::LdrMisc => self.exec_ldr_misc(bus, d),
            Op::StrMisc => self.exec_str_misc(bus, d),
            Op::Ldm => self.exec_ldm(bus, d),
            Op::Stm => self.exec_stm(bus, d),
            Op::Swp => {
                // The swap's internal cycle advances the prefetch before
                // the operands are read: R15 reads +12 (suite-pinned).
                let addr = self.reg_op12(d.rn);
                let src = self.reg_op12(d.rm);
                let old = if d.aux & 1 != 0 {
                    let v = bus.read(addr) as u32;
                    self.write8(bus, addr, src as u8);
                    v
                } else {
                    let v = self.read32_rot(bus, addr);
                    self.write32(bus, addr & !3, src);
                    v
                };
                self.set_reg(d.rd, old);
                Ok(4)
            }
            Op::Swi => {
                if self.trap_swi {
                    self.set_host_trap(HostTrap::Syscall);
                } else {
                    self.raise(Exception::Swi);
                }
                Ok(3)
            }
            Op::Undef => Err(Exception::Undefined),
            Op::ThumbBl => {
                if d.aux & TBL_SUFFIX == 0 {
                    // Prefix: LR = pc + 4 + sext(off11) << 12.
                    let hi = (((d.imm << 21) as i32) >> 9) as u32;
                    self.regs.gpr[14] = self.pc_op().wrapping_add(hi);
                    Ok(1)
                } else {
                    // Suffix: branch to LR + off11*2, LR = return | 1.
                    let ret = self.regs.gpr[15] | 1;
                    let target = self.regs.gpr[14].wrapping_add(d.imm << 1);
                    self.regs.gpr[14] = ret;
                    self.branch_to(target);
                    Ok(3)
                }
            }
        }
    }

    // --- Data processing --------------------------------------------------------

    fn exec_dp(&mut self, d: &DecodedInsn) -> Exec<u32> {
        let carry_in = self.regs.cpsr.c();
        // Operand 2 and the shifter carry-out.
        let shreg = d.op == Op::DpShReg;
        let (op2, sc) = match d.op {
            Op::DpImm => (
                d.imm,
                if d.aux & dp::ROT != 0 {
                    d.imm >> 31 != 0
                } else {
                    carry_in
                },
            ),
            Op::DpShImm => shift_imm(
                (d.aux & dp::TY) >> dp::TY_SHIFT,
                d.rs,
                self.reg_op(d.rm),
                carry_in,
            ),
            // The shift amount is read *before* the internal cycle
            // advances the prefetch, so Rs = 15 yields `pc + 8` while the
            // Rm/Rn operands read `pc + 12` (suite-pinned).
            _ => shift_reg(
                (d.aux & dp::TY) >> dp::TY_SHIFT,
                self.reg_op(d.rs) as u8,
                self.reg_op12(d.rm),
                carry_in,
            ),
        };
        let rn = if shreg {
            self.reg_op12(d.rn)
        } else {
            self.reg_op(d.rn)
        };

        let mut c = sc;
        let mut v = self.regs.cpsr.v();
        let mut arith = |r: (u32, bool, bool)| {
            c = r.1;
            v = r.2;
            r.0
        };
        let opn = d.aux & dp::OP;
        let (result, writes_rd) = match opn {
            alu_op::AND => (rn & op2, true),
            alu_op::EOR => (rn ^ op2, true),
            alu_op::SUB => (arith(sub_with_carry(rn, op2, true)), true),
            alu_op::RSB => (arith(sub_with_carry(op2, rn, true)), true),
            alu_op::ADD => (arith(add_with_carry(rn, op2, false)), true),
            alu_op::ADC => (arith(add_with_carry(rn, op2, carry_in)), true),
            alu_op::SBC => (arith(sub_with_carry(rn, op2, carry_in)), true),
            alu_op::RSC => (arith(sub_with_carry(op2, rn, carry_in)), true),
            alu_op::TST => (rn & op2, false),
            alu_op::TEQ => (rn ^ op2, false),
            alu_op::CMP => (arith(sub_with_carry(rn, op2, true)), false),
            alu_op::CMN => (arith(add_with_carry(rn, op2, false)), false),
            alu_op::ORR => (rn | op2, true),
            alu_op::MOV => (op2, true),
            alu_op::BIC => (rn & !op2, true),
            _ => (!op2, true),
        };

        let s = d.aux & dp::S != 0;
        let mut cycles = 1 + shreg as u32;
        let restores = d.rd == 15 && s && self.regs.cpsr.mode().has_spsr();
        if s && !(restores && writes_rd) {
            // The plain flag write — including every Rd = 15 form in
            // User/System, which have no SPSR to restore (suite-pinned).
            self.regs.cpsr.set_nz(result);
            self.regs.cpsr.set(psr::C, c);
            if !matches!(
                opn,
                alu_op::AND
                    | alu_op::EOR
                    | alu_op::TST
                    | alu_op::TEQ
                    | alu_op::ORR
                    | alu_op::MOV
                    | alu_op::BIC
                    | alu_op::MVN
            ) {
                self.regs.cpsr.set(psr::V, v);
            }
        }
        if restores {
            // The mode-return forms (MOVS PC, LR / SUBS PC, LR, #4) and
            // the test-op "TSTP" family: CPSR = SPSR, overwriting any
            // flags computed above.
            self.restore_cpsr();
        }
        if writes_rd {
            if d.rd == 15 {
                self.branch_to(result); // in the restored state
                cycles += 2;
            } else {
                self.regs.gpr[d.rd as usize] = result;
            }
        }
        Ok(cycles)
    }

    // --- Multiplies --------------------------------------------------------------

    // The multiplies burn internal cycles before reading their operands,
    // so R15 reads as `pc + 12` in all of them (suite-pinned) — hence
    // `reg_op12` throughout.

    fn exec_mul(&mut self, d: &DecodedInsn) -> u32 {
        let rs = self.reg_op12(d.rs);
        let mut r = self.reg_op12(d.rm).wrapping_mul(rs);
        let mut cycles = 1 + mul_cycles(rs, true);
        if d.aux & mul::ACC != 0 {
            r = r.wrapping_add(self.reg_op12(d.rn));
            cycles += 1;
        }
        self.set_reg(d.rd, r);
        if d.aux & mul::S != 0 {
            // C is architecturally *meaningless* after MULS on the
            // ARM7TDMI; this core leaves it unchanged (deterministic pick).
            self.regs.cpsr.set_nz(r);
        }
        cycles
    }

    fn exec_mull(&mut self, d: &DecodedInsn) -> u32 {
        let rs = self.reg_op12(d.rs);
        let rm = self.reg_op12(d.rm);
        let signed = d.aux & mul::SIGNED != 0;
        let mut r = if signed {
            (rm as i32 as i64).wrapping_mul(rs as i32 as i64) as u64
        } else {
            (rm as u64).wrapping_mul(rs as u64)
        };
        let mut cycles = 2 + mul_cycles(rs, signed);
        if d.aux & mul::ACC != 0 {
            let acc = (self.reg_op12(d.rd) as u64) << 32 | self.reg_op12(d.rn) as u64;
            r = r.wrapping_add(acc);
            cycles += 1;
        }
        self.set_reg(d.rn, r as u32); // RdLo
        self.set_reg(d.rd, (r >> 32) as u32); // RdHi
        if d.aux & mul::S != 0 {
            // C and V are meaningless on hardware; left unchanged.
            self.regs.cpsr.set(psr::N, r >> 63 != 0);
            self.regs.cpsr.set(psr::Z, r == 0);
        }
        cycles
    }

    // --- PSR transfer --------------------------------------------------------------

    /// MSR: write `value` into the CPSR or SPSR under the c/x/s/f field
    /// mask in `aux`. The bits land raw (reserved middle bits, reserved
    /// mode encodings and even the T bit included), with one adjustment:
    /// a CPSR control-field write forces mode bit 4 on, as the ARM7TDMI
    /// does. User mode can only touch the flag field.
    fn exec_msr(&mut self, value: u32, aux: u8) -> u32 {
        let fields = aux >> sr::FIELD_SHIFT;
        let mut mask = 0u32;
        for (bit, part) in [
            (1u8, 0x0000_00FFu32),
            (2, 0x0000_FF00),
            (4, 0x00FF_0000),
            (8, 0xFF00_0000),
        ] {
            if fields & bit != 0 {
                mask |= part;
            }
        }
        if aux & sr::SPSR != 0 {
            self.regs.set_spsr(value, mask);
        } else {
            if !self.regs.cpsr.privileged() {
                mask &= 0xFF00_0000;
            }
            let mut raw = (self.regs.cpsr.bits() & !mask) | (value & mask);
            if mask & 0xFF != 0 {
                raw |= 0x10; // mode bit 4 is hardwired on CPSR writes
            }
            let next = super::registers::Psr::from_bits(raw);
            self.regs.set_mode(next.mode());
            self.regs.cpsr = next;
        }
        1
    }

    // --- Single data transfers --------------------------------------------------------

    /// Base, offset-applied address, and write-back value for the single
    /// transfers ([`xfer`] aux layout).
    fn xfer_addr(&mut self, d: &DecodedInsn) -> (u32, u32) {
        let base = self.reg_op(d.rn);
        let off = if d.aux & xfer::REG != 0 {
            shift_imm(
                d.aux >> xfer::TY_SHIFT,
                d.rs,
                self.reg_op(d.rm),
                self.regs.cpsr.c(),
            )
            .0
        } else {
            d.imm
        };
        let indexed = if d.aux & xfer::UP != 0 {
            base.wrapping_add(off)
        } else {
            base.wrapping_sub(off)
        };
        let addr = if d.aux & xfer::PRE != 0 {
            indexed
        } else {
            base
        };
        (addr, indexed)
    }

    /// Base write-back for the single transfers. A PC base reads `pc + 8`
    /// for the address, but the written-back value is computed off
    /// `pc + 12` (the write happens one internal cycle later, when the
    /// prefetch has advanced) — suite-pinned.
    #[inline]
    fn xfer_writeback(&mut self, rn: u8, wb: u32) {
        if rn == 15 {
            self.regs.gpr[15] = wb.wrapping_add(4);
        } else {
            self.regs.gpr[rn as usize] = wb;
        }
    }

    fn exec_ldr<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        let (addr, wb) = self.xfer_addr(d);
        let v = if d.aux & xfer::BYTE != 0 {
            bus.read(addr) as u32
        } else {
            self.read32_rot(bus, addr)
        };
        // Write-back first: when Rd == Rn the loaded value wins.
        if d.aux & xfer::WB != 0 && d.rn != PCA {
            self.xfer_writeback(d.rn, wb);
        }
        if d.rd == 15 {
            self.branch_to(v);
            Ok(5)
        } else {
            self.regs.gpr[d.rd as usize] = v;
            Ok(3)
        }
    }

    fn exec_str<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        let (addr, wb) = self.xfer_addr(d);
        let v = if d.rd == 15 {
            self.pc_store()
        } else {
            self.regs.gpr[d.rd as usize]
        };
        if d.aux & xfer::BYTE != 0 {
            self.write8(bus, addr, v as u8);
        } else {
            self.write32(bus, addr & !3, v);
        }
        if d.aux & xfer::WB != 0 && d.rn != PCA {
            self.xfer_writeback(d.rn, wb);
        }
        Ok(2)
    }

    // --- Halfword and signed transfers ---------------------------------------------

    /// Address arithmetic for the halfword/signed forms ([`hw`] aux
    /// layout).
    fn hw_addr(&mut self, d: &DecodedInsn) -> (u32, u32) {
        let base = self.reg_op(d.rn);
        let off = if d.aux & hw::REG != 0 {
            self.reg_op(d.rm)
        } else {
            d.imm
        };
        let indexed = if d.aux & hw::UP != 0 {
            base.wrapping_add(off)
        } else {
            base.wrapping_sub(off)
        };
        let addr = if d.aux & hw::PRE != 0 { indexed } else { base };
        (addr, indexed)
    }

    fn exec_ldr_misc<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        let (addr, wb) = self.hw_addr(d);
        let v = match d.aux & hw::SH {
            hw::H => self.read16_rot(bus, addr),
            hw::SB => bus.read(addr) as i8 as i32 as u32,
            _ => {
                // LDRSH from an odd address degrades to LDRSB on the
                // ARM7TDMI (the rotated high byte is the sign-extended
                // one).
                if addr & 1 != 0 {
                    bus.read(addr) as i8 as i32 as u32
                } else {
                    bus.read16(addr) as i16 as i32 as u32
                }
            }
        };
        if d.aux & hw::WB != 0 && d.rn != PCA {
            self.xfer_writeback(d.rn, wb);
        }
        if d.rd == 15 {
            self.branch_to(v);
            Ok(5)
        } else {
            self.regs.gpr[d.rd as usize] = v;
            Ok(3)
        }
    }

    fn exec_str_misc<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        let (addr, wb) = self.hw_addr(d);
        let v = if d.rd == 15 {
            self.pc_store()
        } else {
            self.regs.gpr[d.rd as usize]
        };
        self.write16(bus, addr & !1, v as u16);
        if d.aux & hw::WB != 0 && d.rn != PCA {
            self.xfer_writeback(d.rn, wb);
        }
        Ok(2)
    }

    // --- Block transfers --------------------------------------------------------------

    /// Start address and write-back value shared by LDM/STM: the transfer
    /// always walks ascending addresses, lowest register first.
    fn blk_addr(&self, d: &DecodedInsn) -> (u32, u32) {
        let base = self.reg_op(d.rn);
        let bytes = if d.aux & blk::EMPTY != 0 {
            0x40
        } else {
            4 * (d.imm as u16).count_ones()
        };
        let (start, wb) = if d.aux & blk::UP != 0 {
            let s = if d.aux & blk::PRE != 0 {
                base.wrapping_add(4)
            } else {
                base
            };
            (s, base.wrapping_add(bytes))
        } else {
            let down = base.wrapping_sub(bytes);
            let s = if d.aux & blk::PRE != 0 {
                down
            } else {
                down.wrapping_add(4)
            };
            (s, down)
        };
        (start & !3, wb)
    }

    fn exec_ldm<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        let (mut addr, wb) = self.blk_addr(d);
        let list = d.imm as u16;
        let pc_in_list = list & 1 << 15 != 0;
        // S with PC in the list restores the CPSR; S without transfers the
        // user bank.
        let user_bank = d.aux & blk::S != 0 && !pc_in_list;
        let mut pc_val = None;
        for i in 0..16 {
            if list & 1 << i == 0 {
                continue;
            }
            let v = bus.read32(addr);
            addr = addr.wrapping_add(4);
            if i == 15 {
                pc_val = Some(v);
            } else if user_bank {
                self.regs.set_user_reg(i, v);
            } else {
                self.regs.gpr[i] = v;
            }
        }
        // Write-back is suppressed when the base was in the list (the
        // loaded value wins on the ARM7TDMI). A user-bank transfer also
        // writes the base back through the *user* bank — the base was
        // read banked, but the write-back lands where the transfer's
        // register selection points (suite-pinned).
        if d.aux & blk::WB != 0 && list & 1 << d.rn == 0 {
            if user_bank {
                self.regs.set_user_reg(d.rn as usize, wb);
            } else {
                self.regs.gpr[d.rn as usize] = wb;
            }
        }
        let n = list.count_ones();
        let mut cycles = n + 2;
        if let Some(v) = pc_val {
            if d.aux & blk::S != 0 {
                self.restore_cpsr();
            }
            if d.aux & blk::EMPTY != 0 {
                // The empty-list quirk loads the PC raw — even the Thumb
                // bit-0 drop is skipped (suite-pinned).
                self.regs.gpr[15] = v;
            } else {
                self.branch_to(v);
            }
            cycles += 2;
        }
        Ok(cycles)
    }

    fn exec_stm<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        let (mut addr, wb) = self.blk_addr(d);
        let list = d.imm as u16;
        let user_bank = d.aux & blk::S != 0;
        let lowest = list.trailing_zeros() as usize;
        for i in 0..16 {
            if list & 1 << i == 0 {
                continue;
            }
            let v = if i == d.rn as usize && d.aux & blk::WB != 0 && i != lowest {
                // Base in the list: the first slot stores the original
                // base, later slots the written-back value (this outranks
                // the R15 store rule when the base *is* R15).
                wb
            } else if i == 15 {
                self.pc_store()
            } else if user_bank {
                self.regs.user_reg(i)
            } else {
                self.regs.gpr[i]
            };
            self.write32(bus, addr, v);
            addr = addr.wrapping_add(4);
        }
        if d.aux & blk::WB != 0 {
            if user_bank {
                self.regs.set_user_reg(d.rn as usize, wb);
            } else {
                self.regs.gpr[d.rn as usize] = wb;
            }
        }
        Ok(list.count_ones() + 1)
    }
}
