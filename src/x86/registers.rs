//! 8086 register file and the FLAGS register.

use bitflags::bitflags;

bitflags! {
    /// The 8086 FLAGS register.
    ///
    /// Only the nine architectural flags exist as storage. The reserved bits
    /// have fixed values on the 8086 when FLAGS is read as a word (`PUSHF`,
    /// interrupts): bit 1 and bits 12–15 read as 1, bits 3 and 5 read as 0.
    /// [`Flags::to_word`]/[`Flags::from_word`] centralize that quirk.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Flags: u16 {
        /// Carry.
        const CF = 0x0001;
        /// Parity (of the low byte of the result).
        const PF = 0x0004;
        /// Auxiliary carry (out of bit 3, for BCD).
        const AF = 0x0010;
        /// Zero.
        const ZF = 0x0040;
        /// Sign.
        const SF = 0x0080;
        /// Trap (single-step interrupt after each instruction).
        const TF = 0x0100;
        /// Interrupt enable.
        const IF = 0x0200;
        /// Direction (string operations decrement when set).
        const DF = 0x0400;
        /// Overflow.
        const OF = 0x0800;
    }
}

impl Flags {
    /// FLAGS as pushed on the stack: reserved bits 1 and 12–15 forced to 1.
    #[inline]
    pub fn to_word(self) -> u16 {
        self.bits() | 0xF002
    }

    /// Load FLAGS from a word (`POPF`, `IRET`), discarding reserved bits.
    #[inline]
    pub fn from_word(w: u16) -> Self {
        Flags::from_bits_truncate(w)
    }

    /// Set `SF`, `ZF` and `PF` from an 8-bit result — the common tail of most
    /// flag-setting instructions.
    #[inline]
    pub fn set_szp8(&mut self, v: u8) {
        self.set(Flags::SF, v & 0x80 != 0);
        self.set(Flags::ZF, v == 0);
        self.set(Flags::PF, v.count_ones().is_multiple_of(2));
    }

    /// Set `SF`, `ZF` and `PF` from a 16-bit result. `PF` reflects only the
    /// low byte, as on real hardware.
    #[inline]
    pub fn set_szp16(&mut self, v: u16) {
        self.set(Flags::SF, v & 0x8000 != 0);
        self.set(Flags::ZF, v == 0);
        self.set(Flags::PF, (v as u8).count_ones().is_multiple_of(2));
    }
}

/// Register indices as encoded in instructions (ModRM `reg`/`rm` fields and
/// the low bits of one-byte-form opcodes).
///
/// 16-bit order: AX CX DX BX SP BP SI DI. 8-bit order: AL CL DL BL AH CH DH BH.
/// Segment order: ES CS SS DS.
pub mod reg {
    pub const AX: u8 = 0;
    pub const CX: u8 = 1;
    pub const DX: u8 = 2;
    pub const BX: u8 = 3;
    pub const SP: u8 = 4;
    pub const BP: u8 = 5;
    pub const SI: u8 = 6;
    pub const DI: u8 = 7;
    pub const ES: u8 = 0;
    pub const CS: u8 = 1;
    pub const SS: u8 = 2;
    pub const DS: u8 = 3;
}

/// The 8086 register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registers {
    /// General registers. The first four are also addressable as 8-bit halves
    /// (`AL`/`AH` etc.) via [`Registers::reg8`].
    pub ax: u16,
    pub bx: u16,
    pub cx: u16,
    pub dx: u16,
    /// Stack pointer (offset into the segment at `SS`).
    pub sp: u16,
    /// Base pointer (defaults to the `SS` segment in addressing).
    pub bp: u16,
    /// Source index.
    pub si: u16,
    /// Destination index.
    pub di: u16,
    /// Segment registers.
    pub es: u16,
    pub cs: u16,
    pub ss: u16,
    pub ds: u16,
    /// Instruction pointer (offset into the segment at `CS`).
    pub ip: u16,
    /// Processor flags.
    pub flags: Flags,
}

impl Registers {
    /// Power-on register state: execution starts at `FFFF:0000`, flags clear.
    pub fn new() -> Self {
        Registers {
            ax: 0,
            bx: 0,
            cx: 0,
            dx: 0,
            sp: 0,
            bp: 0,
            si: 0,
            di: 0,
            es: 0,
            cs: 0xFFFF,
            ss: 0,
            ds: 0,
            ip: 0,
            flags: Flags::empty(),
        }
    }

    /// Read a 16-bit register by its instruction encoding (AX CX DX BX SP BP SI DI).
    #[inline]
    pub fn reg16(&self, i: u8) -> u16 {
        match i & 7 {
            0 => self.ax,
            1 => self.cx,
            2 => self.dx,
            3 => self.bx,
            4 => self.sp,
            5 => self.bp,
            6 => self.si,
            _ => self.di,
        }
    }

    /// Write a 16-bit register by its instruction encoding.
    #[inline]
    pub fn set_reg16(&mut self, i: u8, v: u16) {
        match i & 7 {
            0 => self.ax = v,
            1 => self.cx = v,
            2 => self.dx = v,
            3 => self.bx = v,
            4 => self.sp = v,
            5 => self.bp = v,
            6 => self.si = v,
            _ => self.di = v,
        }
    }

    /// Read an 8-bit register by its instruction encoding (AL CL DL BL AH CH DH BH).
    #[inline]
    pub fn reg8(&self, i: u8) -> u8 {
        let w = self.reg16(i & 3);
        if i & 4 != 0 { (w >> 8) as u8 } else { w as u8 }
    }

    /// Write an 8-bit register by its instruction encoding.
    #[inline]
    pub fn set_reg8(&mut self, i: u8, v: u8) {
        let r = i & 3;
        let w = self.reg16(r);
        let w = if i & 4 != 0 {
            (w & 0x00FF) | ((v as u16) << 8)
        } else {
            (w & 0xFF00) | v as u16
        };
        self.set_reg16(r, w);
    }

    /// Read a segment register by its instruction encoding (ES CS SS DS).
    #[inline]
    pub fn seg(&self, i: u8) -> u16 {
        match i & 3 {
            0 => self.es,
            1 => self.cs,
            2 => self.ss,
            _ => self.ds,
        }
    }

    /// Write a segment register by its instruction encoding.
    #[inline]
    pub fn set_seg(&mut self, i: u8, v: u16) {
        match i & 3 {
            0 => self.es = v,
            1 => self.cs = v,
            2 => self.ss = v,
            _ => self.ds = v,
        }
    }
}

impl Default for Registers {
    fn default() -> Self {
        Registers::new()
    }
}
