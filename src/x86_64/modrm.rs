//! ModRM/SIB decoding and effective-address calculation for all three
//! address sizes, including the REX register extensions and RIP-relative
//! addressing.
//!
//! Every ModRM-form instruction calls [`Cpu::modrm`] once, right after its
//! opcode byte: it fetches the ModRM byte (plus SIB and displacement) and
//! resolves the memory operand to a `segment:offset` pair, honoring override
//! prefixes and the BP/SP→SS defaults. Handlers then read/write through the
//! returned [`Operand`].

use super::OpSize::{O16, O32};
use super::registers::reg;
use super::{Bus, Cpu, Exec};

/// A raw ModRM byte with accessors for its three fields. The `reg` and `rm`
/// accessors already include the REX extension bit where it applies.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModRm {
    pub(crate) byte: u8,
    /// REX.R << 3 (extends `reg`).
    pub(crate) r: u8,
    /// REX.B << 3 (extends a register `rm`; memory forms fold REX.B into the
    /// EA instead).
    pub(crate) b: u8,
}

impl ModRm {
    /// The `mod` field (bits 7–6).
    #[inline]
    pub fn md(self) -> u8 {
        self.byte >> 6
    }

    /// The `reg` field (bits 5–3) extended by REX.R: a register index or a
    /// group sub-opcode (groups ignore the extension by masking with 7).
    #[inline]
    pub fn reg(self) -> u8 {
        ((self.byte >> 3) & 7) | self.r
    }

    /// The `reg` field without the REX extension (group sub-opcodes).
    #[inline]
    pub fn sub(self) -> u8 {
        (self.byte >> 3) & 7
    }

    /// The `rm` field (bits 2–0) extended by REX.B (register operands only).
    #[inline]
    pub fn rm(self) -> u8 {
        (self.byte & 7) | self.b
    }
}

/// A resolved instruction operand: either a register index (interpreted as
/// 8/16/32/64-bit by the handler) or a memory location as an unresolved
/// `segment-register:offset` pair (resolution happens per access, so
/// canonical/limit checks and paging apply naturally).
#[derive(Debug, Clone, Copy)]
pub(crate) enum Operand {
    Reg(u8),
    Mem { seg: u8, off: u64 },
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
        let m = ModRm {
            byte: self.fetch8(bus)?,
            r: self.rex_r() << 3,
            b: self.rex_b() << 3,
        };
        if m.md() == 3 {
            return Ok((m, Operand::Reg(m.rm())));
        }
        let op = match self.asize {
            O16 => self.ea16(bus, m)?,
            _ => self.ea_wide(bus, m)?,
        };
        Ok((m, op))
    }

    /// 16-bit effective address (legacy modes only; 64-bit mode cannot
    /// encode it).
    fn ea16<B: Bus>(&mut self, bus: &mut B, m: ModRm) -> Exec<Operand> {
        // Direct address: mod == 00, rm == 110 is a bare disp16 (no BP).
        if m.md() == 0 && m.byte & 7 == 6 {
            let off = self.fetch16(bus)? as u64;
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
        let (base, seg) = match m.byte & 7 {
            0 => (r.reg16(reg::RBX).wrapping_add(r.reg16(reg::RSI)), reg::DS),
            1 => (r.reg16(reg::RBX).wrapping_add(r.reg16(reg::RDI)), reg::DS),
            2 => (r.reg16(reg::RBP).wrapping_add(r.reg16(reg::RSI)), reg::SS),
            3 => (r.reg16(reg::RBP).wrapping_add(r.reg16(reg::RDI)), reg::SS),
            4 => (r.reg16(reg::RSI), reg::DS),
            5 => (r.reg16(reg::RDI), reg::DS),
            6 => (r.reg16(reg::RBP), reg::SS),
            _ => (r.reg16(reg::RBX), reg::DS),
        };
        Ok(Operand::Mem {
            seg: self.seg_or(seg),
            off: base.wrapping_add(disp) as u64,
        })
    }

    /// 32/64-bit effective address, with the SIB byte and (in 64-bit mode)
    /// RIP-relative addressing. The computed offset is truncated to 32 bits
    /// under a 32-bit address size.
    fn ea_wide<B: Bus>(&mut self, bus: &mut B, m: ModRm) -> Exec<Operand> {
        // mod == 00, rm == 101 (ignoring REX.B): disp32 in legacy modes,
        // RIP-relative disp32 in 64-bit mode.
        if m.md() == 0 && m.byte & 7 == 5 {
            let disp = self.fetch32(bus)? as i32 as i64 as u64;
            let off = if self.m64 {
                // Relative to the *next* instruction: past the displacement
                // (where RIP points now) plus any trailing immediate, whose
                // length the dispatcher recorded in `imm_len` before decode.
                self.used_rip_rel = true;
                self.regs
                    .rip
                    .wrapping_add(self.imm_len as u64)
                    .wrapping_add(disp)
            } else {
                disp
            };
            return Ok(Operand::Mem {
                seg: self.seg_or(reg::DS),
                off: self.trunc_ea(off),
            });
        }

        let (base, seg) = if m.byte & 7 == 4 {
            self.sib(bus, m)?
        } else {
            let rm = (m.byte & 7) | m.b;
            let seg = if m.byte & 7 == reg::RBP {
                reg::SS
            } else {
                reg::DS
            };
            (self.ea_reg(rm), seg)
        };

        let disp = match m.md() {
            0 => 0,
            1 => self.fetch8(bus)? as i8 as i64 as u64,
            _ => self.fetch32(bus)? as i32 as i64 as u64,
        };
        Ok(Operand::Mem {
            seg: self.seg_or(seg),
            off: self.trunc_ea(base.wrapping_add(disp)),
        })
    }

    /// A register value for effective-address math, truncated per address
    /// size.
    #[inline]
    fn ea_reg(&self, i: u8) -> u64 {
        let v = self.regs.reg64(i);
        if self.asize == O32 {
            v as u32 as u64
        } else {
            v
        }
    }

    /// Truncate a computed effective address to the address size.
    #[inline]
    fn trunc_ea(&self, off: u64) -> u64 {
        if self.asize == O32 {
            off as u32 as u64
        } else {
            off
        }
    }

    /// Decode the SIB byte: `(base + (index << scale), default segment)`.
    ///
    /// `index == 100` with REX.X clear encodes "no index" (RSP can never be
    /// an index); with REX.X set it selects R12. `base == 101` with mod == 00
    /// replaces the base with a disp32 regardless of REX.B.
    fn sib<B: Bus>(&mut self, bus: &mut B, m: ModRm) -> Exec<(u64, u8)> {
        let sib = self.fetch8(bus)?;
        let (scale, index3, base3) = (sib >> 6, (sib >> 3) & 7, sib & 7);
        let index = index3 | (self.rex_x() << 3);
        let base_reg = base3 | m.b;

        let (mut base, seg) = if base3 == 5 && m.md() == 0 {
            (self.fetch32(bus)? as i32 as i64 as u64, reg::DS)
        } else {
            let seg = if base3 == reg::RSP || base3 == reg::RBP {
                reg::SS
            } else {
                reg::DS
            };
            (self.ea_reg(base_reg), seg)
        };

        if index != 4 {
            base = base.wrapping_add(self.ea_reg(index) << scale);
        }
        Ok((base, seg))
    }

    // --- Operand access -------------------------------------------------------

    /// Read an operand as a byte.
    #[inline]
    pub(crate) fn read_op8<B: Bus>(&mut self, bus: &mut B, op: Operand) -> Exec<u8> {
        match op {
            Operand::Reg(i) => Ok(self.gpr8(i)),
            Operand::Mem { seg, off } => self.read8(bus, seg, off),
        }
    }

    /// Write an operand as a byte.
    #[inline]
    pub(crate) fn write_op8<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u8) -> Exec<()> {
        match op {
            Operand::Reg(i) => {
                self.set_gpr8(i, v);
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

    /// Write an operand as a double-word (register destinations zero their
    /// upper half, per the 64-bit rule).
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

    /// Read an operand as a quad-word.
    #[inline]
    pub(crate) fn read_op64<B: Bus>(&mut self, bus: &mut B, op: Operand) -> Exec<u64> {
        match op {
            Operand::Reg(i) => Ok(self.regs.reg64(i)),
            Operand::Mem { seg, off } => self.read64(bus, seg, off),
        }
    }

    /// Write an operand as a quad-word.
    #[inline]
    pub(crate) fn write_op64<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u64) -> Exec<()> {
        match op {
            Operand::Reg(i) => {
                self.regs.set_reg64(i, v);
                Ok(())
            }
            Operand::Mem { seg, off } => self.write64(bus, seg, off, v),
        }
    }

    /// Read an operand at the current operand size (zero-extended).
    #[inline]
    pub(crate) fn read_op<B: Bus>(&mut self, bus: &mut B, op: Operand) -> Exec<u64> {
        match self.osize {
            O16 => Ok(self.read_op16(bus, op)? as u64),
            O32 => Ok(self.read_op32(bus, op)? as u64),
            _ => self.read_op64(bus, op),
        }
    }

    /// Write an operand at the current operand size.
    #[inline]
    pub(crate) fn write_op<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u64) -> Exec<()> {
        match self.osize {
            O16 => self.write_op16(bus, op, v as u16),
            O32 => self.write_op32(bus, op, v as u32),
            _ => self.write_op64(bus, op, v),
        }
    }
}
