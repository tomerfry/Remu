//! ModRM decoding and effective-address calculation.
//!
//! Every ModRM-form instruction calls [`Cpu::modrm`] once, right after its
//! opcode byte: it fetches the ModRM byte plus any displacement and resolves
//! the memory operand to a concrete `segment:offset` pair (honoring override
//! prefixes and the BP→SS default). Handlers then read/write through the
//! returned [`Operand`], so each operation is implemented exactly once
//! regardless of addressing form.

use super::registers::reg;
use super::{Bus, Cpu};

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
/// 8- or 16-bit by the handler) or a memory location with its segment already
/// resolved to a value.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Operand {
    Reg(u8),
    Mem { seg: u16, off: u16 },
}

impl Operand {
    /// Whether this operand lives in memory (used to pick reg-vs-mem timings).
    #[inline]
    pub fn is_mem(self) -> bool {
        matches!(self, Operand::Mem { .. })
    }
}

impl Cpu {
    /// Fetch and decode a ModRM byte (and its displacement), resolving the
    /// `rm` operand. Also records the documented effective-address penalty in
    /// `self.ea_cycles` (0 for register operands).
    pub(crate) fn modrm<B: Bus>(&mut self, bus: &mut B) -> (ModRm, Operand) {
        let m = ModRm(self.fetch_byte(bus));
        if m.md() == 3 {
            self.ea_cycles = 0;
            return (m, Operand::Reg(m.rm()));
        }

        // Direct address: mod == 00, rm == 110 is a bare disp16 (no BP).
        if m.md() == 0 && m.rm() == 6 {
            let off = self.fetch_word(bus);
            self.ea_cycles = 6;
            return (
                m,
                Operand::Mem {
                    seg: self.seg_or(reg::DS),
                    off,
                },
            );
        }

        let disp = match m.md() {
            0 => 0,
            1 => self.fetch_byte(bus) as i8 as u16, // sign-extended
            _ => self.fetch_word(bus),
        };

        let r = &self.regs;
        // (base+index value, default segment, EA cycles for the no-disp form)
        let (base, seg, ea) = match m.rm() {
            0 => (r.bx.wrapping_add(r.si), reg::DS, 7),
            1 => (r.bx.wrapping_add(r.di), reg::DS, 8),
            2 => (r.bp.wrapping_add(r.si), reg::SS, 8),
            3 => (r.bp.wrapping_add(r.di), reg::SS, 7),
            4 => (r.si, reg::DS, 5),
            5 => (r.di, reg::DS, 5),
            6 => (r.bp, reg::SS, 5),
            _ => (r.bx, reg::DS, 5),
        };
        self.ea_cycles = if m.md() == 0 { ea } else { ea + 4 };
        (
            m,
            Operand::Mem {
                seg: self.seg_or(seg),
                off: base.wrapping_add(disp),
            },
        )
    }

    /// Read an operand as a byte.
    #[inline]
    pub(crate) fn read_op8<B: Bus>(&mut self, bus: &mut B, op: Operand) -> u8 {
        match op {
            Operand::Reg(i) => self.regs.reg8(i),
            Operand::Mem { seg, off } => self.read8(bus, seg, off),
        }
    }

    /// Write an operand as a byte.
    #[inline]
    pub(crate) fn write_op8<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u8) {
        match op {
            Operand::Reg(i) => self.regs.set_reg8(i, v),
            Operand::Mem { seg, off } => self.write8(bus, seg, off, v),
        }
    }

    /// Read an operand as a word.
    #[inline]
    pub(crate) fn read_op16<B: Bus>(&mut self, bus: &mut B, op: Operand) -> u16 {
        match op {
            Operand::Reg(i) => self.regs.reg16(i),
            Operand::Mem { seg, off } => self.read16(bus, seg, off),
        }
    }

    /// Write an operand as a word.
    #[inline]
    pub(crate) fn write_op16<B: Bus>(&mut self, bus: &mut B, op: Operand, v: u16) {
        match op {
            Operand::Reg(i) => self.regs.set_reg16(i, v),
            Operand::Mem { seg, off } => self.write16(bus, seg, off, v),
        }
    }
}
