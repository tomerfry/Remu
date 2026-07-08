//! CPU register file and the processor status flags.

use bitflags::bitflags;

bitflags! {
    /// The processor status register `P` (flags `NV-BDIZC`).
    ///
    /// Note that the physical 6502 `P` register has only six real flip-flops.
    /// Bit 4 (`B`) and bit 5 (`U`) do not exist as storage — they only take
    /// meaning when `P` is pushed to the stack. The push/pull helpers on
    /// [`Cpu`](crate::cpu::Cpu) centralize that quirk; in the in-memory
    /// representation we keep `U` set and `B` clear.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Status: u8 {
        /// Carry.
        const C = 0b0000_0001;
        /// Zero.
        const Z = 0b0000_0010;
        /// Interrupt disable.
        const I = 0b0000_0100;
        /// Decimal mode.
        const D = 0b0000_1000;
        /// Break (not a real register bit; meaningful only when pushed).
        const B = 0b0001_0000;
        /// Unused (not a real register bit; reads as 1).
        const U = 0b0010_0000;
        /// Overflow.
        const V = 0b0100_0000;
        /// Negative.
        const N = 0b1000_0000;
    }
}

impl Status {
    /// Set the `Z` and `N` flags to reflect `value` — the single most common
    /// flag update, performed by nearly every instruction that produces a result.
    #[inline]
    pub fn set_zn(&mut self, value: u8) {
        self.set(Status::Z, value == 0);
        self.set(Status::N, value & 0x80 != 0);
    }
}

/// The 6502 register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registers {
    /// Accumulator.
    pub a: u8,
    /// Index register X.
    pub x: u8,
    /// Index register Y.
    pub y: u8,
    /// Stack pointer (low byte; the stack lives at `$0100..=$01FF`).
    pub sp: u8,
    /// Program counter.
    pub pc: u16,
    /// Processor status flags.
    pub p: Status,
}

impl Registers {
    /// Power-on register state: `I` and `U` set, stack pointer at `$FD`.
    pub fn new() -> Self {
        Registers {
            a: 0,
            x: 0,
            y: 0,
            sp: 0xFD,
            pc: 0,
            p: Status::I | Status::U,
        }
    }
}

impl Default for Registers {
    fn default() -> Self {
        Registers::new()
    }
}
