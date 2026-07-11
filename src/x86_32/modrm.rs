//! ModRM/SIB decoding and effective-address calculation for both address
//! sizes.
//!
//! Every ModRM-form instruction calls [`Cpu::modrm`] once, right after its
//! opcode byte: it fetches the ModRM byte (plus SIB and displacement) and
//! resolves the memory operand to a `segment:offset` pair, honoring override
//! prefixes and the BP/SP→SS defaults. Handlers then read/write through the
//! returned [`Operand`].

use super::registers::reg;
use super::{Bus, Cpu, Exec};

/// A raw ModRM byte with accessors for its three fields.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModRm(pub u8);

impl ModRm {
    /// The `mod` field (bits 7–6).
    #[inline]
    pub fn md(self) -> u8 {
        self.0 >> 6
    }

    /// The `reg` field (bits 5–3): a register index or a group sub-opcode.
    #[inline]
    pub fn reg(self) -> u8 {
        (self.0 >> 3) & 7
    }

    /// The `rm` field (bits 2–0).
    #[inline]
    pub fn rm(self) -> u8 {
        self.0 & 7
    }
}

/// A resolved instruction operand: either a register index (interpreted as
/// 8/16/32-bit by the handler) or a memory location as an unresolved
/// `segment-register:offset` pair (resolution happens per access, so limit
/// checks and paging apply naturally).
#[derive(Debug, Clone, Copy)]
pub(crate) enum Operand {
    Reg(u8),
    Mem { seg: u8, off: u32 },
}

impl Operand {
    /// Whether this operand lives in memory (used to pick reg-vs-mem timings
    /// and to validate LOCK prefixes).
    #[inline]
    pub fn is_mem(self) -> bool {
        matches!(self, Operand::Mem { .. })
    }
}

impl Cpu {
    /// Fetch and decode a ModRM byte (plus SIB/displacement), resolving the
    /// `rm` operand for the current address size.
    pub(crate) fn modrm<B: Bus>(&mut self, bus: &mut B) -> Exec<(ModRm, Operand)> {
        self.stat(|s| s.modrm_calls += 1);
        let m = ModRm(self.fetch8(bus)?);
        if m.md() == 3 {
            return Ok((m, Operand::Reg(m.rm())));
        }
        let op = if self.asize32 {
            self.ea32(bus, m)?
        } else {
            self.ea16(bus, m)?
        };
        Ok((m, op))
    }

    /// 16-bit effective address (identical table to the 8086).
    fn ea16<B: Bus>(&mut self, bus: &mut B, m: ModRm) -> Exec<Operand> {
        // Direct address: mod == 00, rm == 110 is a bare disp16 (no BP).
        if m.md() == 0 && m.rm() == 6 {
            let off = self.fetch16(bus)? as u32;
            return Ok(Operand::Mem {
                seg: self.seg_or(reg::DS),
                off,
            });
        }

        let disp = match m.md() {
            0 => 0,
            1 => self.fetch8(bus)? as i8 as u16,
            _ => self.fetch16(bus)?,
        };

        let r = &self.regs;
        let (base, seg) = match m.rm() {
            0 => (r.reg16(reg::EBX).wrapping_add(r.reg16(reg::ESI)), reg::DS),
            1 => (r.reg16(reg::EBX).wrapping_add(r.reg16(reg::EDI)), reg::DS),
            2 => (r.reg16(reg::EBP).wrapping_add(r.reg16(reg::ESI)), reg::SS),
            3 => (r.reg16(reg::EBP).wrapping_add(r.reg16(reg::EDI)), reg::SS),
            4 => (r.reg16(reg::ESI), reg::DS),
            5 => (r.reg16(reg::EDI), reg::DS),
            6 => (r.reg16(reg::EBP), reg::SS),
            _ => (r.reg16(reg::EBX), reg::DS),
        };
        Ok(Operand::Mem {
            seg: self.seg_or(seg),
            off: base.wrapping_add(disp) as u32,
        })
    }

    /// 32-bit effective address, with the SIB byte.
    fn ea32<B: Bus>(&mut self, bus: &mut B, m: ModRm) -> Exec<Operand> {
        // Direct address: mod == 00, rm == 101 is a bare disp32 (no EBP).
        if m.md() == 0 && m.rm() == 5 {
            let off = self.fetch32(bus)?;
            return Ok(Operand::Mem {
                seg: self.seg_or(reg::DS),
                off,
            });
        }

        let (base, seg) = if m.rm() == 4 {
            self.sib(bus, m)?
        } else {
            let seg = if m.rm() == reg::EBP { reg::SS } else { reg::DS };
            (self.regs.reg32(m.rm()), seg)
        };

        let disp = match m.md() {
            0 => 0,
            1 => self.fetch8(bus)? as i8 as u32,
            _ => self.fetch32(bus)?,
        };
        Ok(Operand::Mem {
            seg: self.seg_or(seg),
            off: base.wrapping_add(disp),
        })
    }

    /// Decode the SIB byte: `(base + (index << scale), default segment)`.
    ///
    /// `index == 100` encodes "no index"; the 386 (unlike every later x86)
    /// then applies a non-zero scale to the *base* register — three
    /// officially blank rows of the SIB table that real silicon executes
    /// this way, asserted by the SingleStepTests 386 suite.
    fn sib<B: Bus>(&mut self, bus: &mut B, m: ModRm) -> Exec<(u32, u8)> {
        let sib = self.fetch8(bus)?;
        let (scale, index, base_reg) = (sib >> 6, (sib >> 3) & 7, sib & 7);

        // mod == 00, base == 101: disp32 replaces the base register (and is
        // never scaled, even by the quirk below).
        let (mut base, seg, scalable) = if base_reg == 5 && m.md() == 0 {
            (self.fetch32(bus)?, reg::DS, false)
        } else {
            let seg = if base_reg == reg::ESP || base_reg == reg::EBP {
                reg::SS
            } else {
                reg::DS
            };
            (self.regs.reg32(base_reg), seg, true)
        };

        if index != 4 {
            base = base.wrapping_add(self.regs.reg32(index) << scale);
        } else if scalable {
            base <<= scale; // the 386 scaled-base quirk (no-op for scale 0)
        }
        Ok((base, seg))
    }

    // --- Operand access -------------------------------------------------------

    /// Read an operand as a byte.
    #[inline]
    pub(crate) fn read_op8<B: Bus>(&mut self, bus: &mut B, op: Operand) -> Exec<u8> {
        match op {
            Operand::Reg(i) => Ok(self.regs.reg8(i)),
            Operand::Mem { seg, off } => self.read8(bus, seg, off),
        }
    }

    /// Write an operand as a byte.
    #[inline]
    pub(crate) fn write_op8<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u8) -> Exec<()> {
        match op {
            Operand::Reg(i) => {
                self.regs.set_reg8(i, v);
                Ok(())
            }
            Operand::Mem { seg, off } => self.write8(bus, seg, off, v),
        }
    }

    /// Read an operand as a word.
    #[inline]
    pub(crate) fn read_op16<B: Bus>(&mut self, bus: &mut B, op: Operand) -> Exec<u16> {
        match op {
            Operand::Reg(i) => Ok(self.regs.reg16(i)),
            Operand::Mem { seg, off } => self.read16(bus, seg, off),
        }
    }

    /// Write an operand as a word.
    #[inline]
    pub(crate) fn write_op16<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u16) -> Exec<()> {
        match op {
            Operand::Reg(i) => {
                self.regs.set_reg16(i, v);
                Ok(())
            }
            Operand::Mem { seg, off } => self.write16(bus, seg, off, v),
        }
    }

    /// Read an operand as a double-word.
    #[inline]
    pub(crate) fn read_op32<B: Bus>(&mut self, bus: &mut B, op: Operand) -> Exec<u32> {
        match op {
            Operand::Reg(i) => Ok(self.regs.reg32(i)),
            Operand::Mem { seg, off } => self.read32(bus, seg, off),
        }
    }

    /// Write an operand as a double-word.
    #[inline]
    pub(crate) fn write_op32<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u32) -> Exec<()> {
        match op {
            Operand::Reg(i) => {
                self.regs.set_reg32(i, v);
                Ok(())
            }
            Operand::Mem { seg, off } => self.write32(bus, seg, off, v),
        }
    }

    /// Read an operand at the current operand size (zero-extended).
    #[inline]
    pub(crate) fn read_op<B: Bus>(&mut self, bus: &mut B, op: Operand) -> Exec<u32> {
        if self.osize32 {
            self.read_op32(bus, op)
        } else {
            Ok(self.read_op16(bus, op)? as u32)
        }
    }

    /// Write an operand at the current operand size.
    #[inline]
    pub(crate) fn write_op<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u32) -> Exec<()> {
        if self.osize32 {
            self.write_op32(bus, op, v)
        } else {
            self.write_op16(bus, op, v as u16)
        }
    }
}
