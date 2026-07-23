//! ARM32 register file: sixteen 32-bit registers with per-mode banking
//! (r8–r14), the CPSR/SPSR program status registers, and the seven ARMv4
//! processor modes.
//!
//! `gpr` always holds the *current mode's* view; switching modes via
//! [`Registers::set_mode`] (or an exception entry) spills r8–r14 into the
//! old mode's bank and loads the new mode's. At an instruction boundary
//! `gpr[15]` is the address of the next instruction to execute — the
//! pipeline's +8/+4 that ARM code observes when reading R15 is applied by
//! the executor, not stored here.

/// Register indices by their conventional names.
pub mod reg {
    /// Stack pointer (r13).
    pub const SP: usize = 13;
    /// Link register (r14).
    pub const LR: usize = 14;
    /// Program counter (r15).
    pub const PC: usize = 15;
}

/// The seven ARMv4 processor modes, by their CPSR mode-field encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    /// User — unprivileged; the only mode without an SPSR (with [`Mode::Sys`]).
    Usr = 0x10,
    /// Fast interrupt — banks r8–r14.
    Fiq = 0x11,
    /// Interrupt.
    Irq = 0x12,
    /// Supervisor — reset and SWI land here.
    Svc = 0x13,
    /// Abort — prefetch/data aborts land here.
    Abt = 0x17,
    /// Undefined — the undefined-instruction trap lands here.
    Und = 0x1B,
    /// System — privileged, but shares the user register bank (ARMv4).
    Sys = 0x1F,
}

impl Mode {
    /// Decode a CPSR mode field. Returns `None` for the 25 reserved
    /// encodings (entering one is UNPREDICTABLE on hardware; [`Psr`]
    /// writers keep the previous mode instead).
    pub fn from_bits(bits: u32) -> Option<Mode> {
        Some(match bits & 0x1F {
            0x10 => Mode::Usr,
            0x11 => Mode::Fiq,
            0x12 => Mode::Irq,
            0x13 => Mode::Svc,
            0x17 => Mode::Abt,
            0x1B => Mode::Und,
            0x1F => Mode::Sys,
            _ => return None,
        })
    }

    /// Whether this mode has a banked SPSR (all but User/System).
    #[inline]
    pub fn has_spsr(self) -> bool {
        !matches!(self, Mode::Usr | Mode::Sys)
    }

    /// Whether this mode may write the CPSR control bits (all but User).
    #[inline]
    pub fn privileged(self) -> bool {
        self != Mode::Usr
    }

    /// Whether two modes share their entire register bank (User and
    /// System do; every other pair is distinct at r13–r14 or, for FIQ,
    /// r8–r14).
    #[inline]
    fn same_bank(a: Mode, b: Mode) -> bool {
        a == b || (!a.has_spsr() && !b.has_spsr())
    }

    /// Index into the SPSR bank / the r13–r14 banks for exception modes.
    #[inline]
    fn bank(self) -> Option<usize> {
        Some(match self {
            Mode::Fiq => 0,
            Mode::Svc => 1,
            Mode::Abt => 2,
            Mode::Irq => 3,
            Mode::Und => 4,
            Mode::Usr | Mode::Sys => return None,
        })
    }
}

/// A program status register (CPSR or SPSR).
///
/// All 32 bits are stored raw, exactly as the SingleStepTests ARM7TDMI
/// suite pins: MSR can deposit reserved middle bits and even reserved
/// *mode* encodings, and MRS reads them back verbatim. A reserved mode
/// encoding selects the User/System register bank (and no SPSR) until a
/// valid mode is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Psr(u32);

/// [`Psr`] flag bits.
pub mod psr {
    /// Negative.
    pub const N: u32 = 1 << 31;
    /// Zero.
    pub const Z: u32 = 1 << 30;
    /// Carry (for subtraction: NOT borrow).
    pub const C: u32 = 1 << 29;
    /// Signed overflow.
    pub const V: u32 = 1 << 28;
    /// IRQ disable.
    pub const I: u32 = 1 << 7;
    /// FIQ disable.
    pub const F: u32 = 1 << 6;
    /// Thumb state.
    pub const T: u32 = 1 << 5;
    /// The mode field.
    pub const MODE: u32 = 0x1F;
}

impl Psr {
    /// A PSR from raw bits.
    pub fn from_bits(bits: u32) -> Psr {
        Psr(bits)
    }

    /// The raw 32-bit image.
    #[inline]
    pub fn bits(self) -> u32 {
        self.0
    }

    /// Overwrite the bits selected by `mask` with those from `value`,
    /// keeping the rest. Raw — no validation.
    #[inline]
    pub fn write(&mut self, value: u32, mask: u32) {
        self.0 = (self.0 & !mask) | (value & mask);
    }

    /// The raw mode field (bits 4:0), which may be a reserved encoding.
    #[inline]
    pub fn mode_bits(self) -> u32 {
        self.0 & psr::MODE
    }

    /// The register-bank mode: the decoded mode field, with reserved
    /// encodings selecting the User/System bank (matching the ARM7TDMI's
    /// bank decoder as pinned by the suite).
    #[inline]
    pub fn mode(self) -> Mode {
        Mode::from_bits(self.0).unwrap_or(Mode::Usr)
    }

    /// Whether the CPSR grants privilege: everything but User mode
    /// proper. (Reserved encodings behave as privileged.)
    #[inline]
    pub fn privileged(self) -> bool {
        self.mode_bits() != Mode::Usr as u32
    }

    /// Replace the mode field.
    #[inline]
    pub fn set_mode(&mut self, m: Mode) {
        self.0 = (self.0 & !psr::MODE) | m as u32;
    }

    /// Negative flag.
    #[inline]
    pub fn n(self) -> bool {
        self.0 & psr::N != 0
    }

    /// Zero flag.
    #[inline]
    pub fn z(self) -> bool {
        self.0 & psr::Z != 0
    }

    /// Carry flag.
    #[inline]
    pub fn c(self) -> bool {
        self.0 & psr::C != 0
    }

    /// Overflow flag.
    #[inline]
    pub fn v(self) -> bool {
        self.0 & psr::V != 0
    }

    /// IRQ-disable bit.
    #[inline]
    pub fn i(self) -> bool {
        self.0 & psr::I != 0
    }

    /// FIQ-disable bit.
    #[inline]
    pub fn f(self) -> bool {
        self.0 & psr::F != 0
    }

    /// Thumb-state bit.
    #[inline]
    pub fn t(self) -> bool {
        self.0 & psr::T != 0
    }

    /// Set or clear a single flag bit.
    #[inline]
    pub fn set(&mut self, bit: u32, on: bool) {
        if on {
            self.0 |= bit;
        } else {
            self.0 &= !bit;
        }
    }

    /// Set N and Z from a result.
    #[inline]
    pub fn set_nz(&mut self, v: u32) {
        self.0 = (self.0 & !(psr::N | psr::Z)) | (v & psr::N) | if v == 0 { psr::Z } else { 0 };
    }
}

/// The ARM32 register file.
///
/// `gpr` is the current mode's window; the six `*_bank` arrays hold the
/// *inactive* copies of the banked registers (r8–r14 for FIQ vs everyone
/// else, r13–r14 per exception mode). Cross-mode accessors
/// ([`Registers::user_reg`], [`Registers::reg_of`]) see through the
/// banking, so tests and OS layers never juggle the raw banks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registers {
    /// r0–r15 as seen by the current mode. `gpr[15]` is the address of the
    /// next instruction to execute (no pipeline offset).
    pub gpr: [u32; 16],
    /// The current program status register.
    pub cpsr: Psr,
    /// Banked SPSRs (FIQ, SVC, ABT, IRQ, UND).
    spsr: [Psr; 5],
    /// r8–r14 of the user/system bank, while an exception mode is current.
    usr_bank: [u32; 7],
    /// r8–r14 of the FIQ bank, while another mode is current.
    fiq_bank: [u32; 7],
    /// r13–r14 of SVC, ABT, IRQ, UND (indices per [`Mode::bank`] minus the
    /// FIQ slot).
    exc_bank: [[u32; 2]; 4],
}

impl Registers {
    /// Power-on state: Supervisor mode, ARM state, IRQ+FIQ disabled, PC at
    /// the reset vector.
    pub fn new() -> Self {
        Registers {
            gpr: [0; 16],
            cpsr: Psr::from_bits(Mode::Svc as u32 | psr::I | psr::F),
            spsr: [Psr::from_bits(Mode::Svc as u32); 5],
            usr_bank: [0; 7],
            fiq_bank: [0; 7],
            exc_bank: [[0; 2]; 4],
        }
    }

    /// Switch to `new` mode: spill the current mode's banked registers and
    /// load `new`'s, then update the CPSR mode field. A no-op when `new`
    /// is the current mode. User↔System share a bank, so that switch moves
    /// nothing.
    pub fn set_mode(&mut self, new: Mode) {
        let old = self.cpsr.mode();
        if old == new {
            self.cpsr.set_mode(new);
            return;
        }
        // Spill the current window into `old`'s storage.
        match old {
            Mode::Fiq => self.fiq_bank.copy_from_slice(&self.gpr[8..15]),
            m => {
                self.usr_bank[..5].copy_from_slice(&self.gpr[8..13]);
                match m.bank() {
                    Some(b) => self.exc_bank[b - 1] = [self.gpr[13], self.gpr[14]],
                    None => self.usr_bank[5..].copy_from_slice(&self.gpr[13..15]),
                }
            }
        }
        // Load `new`'s storage into the window.
        match new {
            Mode::Fiq => self.gpr[8..15].copy_from_slice(&self.fiq_bank),
            m => {
                self.gpr[8..13].copy_from_slice(&self.usr_bank[..5]);
                let [sp, lr] = match m.bank() {
                    Some(b) => self.exc_bank[b - 1],
                    None => [self.usr_bank[5], self.usr_bank[6]],
                };
                self.gpr[13] = sp;
                self.gpr[14] = lr;
            }
        }
        self.cpsr.set_mode(new);
    }

    /// The current mode's SPSR. User/System have none; reading it is
    /// UNPREDICTABLE on hardware — this core returns the CPSR.
    #[inline]
    pub fn spsr(&self) -> Psr {
        match self.cpsr.mode().bank() {
            Some(b) => self.spsr[b],
            None => self.cpsr,
        }
    }

    /// The raw SPSR image MRS reads: the banked SPSR in exception modes,
    /// the CPSR itself in User/System (which have no SPSR) — suite-pinned.
    #[inline]
    pub fn spsr_bits(&self) -> u32 {
        match self.cpsr.mode().bank() {
            Some(b) => self.spsr[b].bits(),
            None => self.cpsr.bits(),
        }
    }

    /// The current mode's banked SPSR image, or `None` in User/System
    /// (the `S`-bit PC-write forms restore nothing there — suite-pinned).
    #[inline]
    pub(crate) fn spsr_banked(&self) -> Option<u32> {
        self.cpsr.mode().bank().map(|b| self.spsr[b].bits())
    }

    /// Write the current mode's SPSR under `mask`. UNPREDICTABLE in
    /// User/System on hardware — this core discards the write.
    #[inline]
    pub fn set_spsr(&mut self, value: u32, mask: u32) {
        if let Some(b) = self.cpsr.mode().bank() {
            self.spsr[b].write(value, mask);
        }
    }

    /// Replace the current mode's SPSR wholesale (exception entry).
    #[inline]
    pub(crate) fn spsr_store(&mut self, p: Psr) {
        if let Some(b) = self.cpsr.mode().bank() {
            self.spsr[b] = p;
        }
    }

    /// Read `mode`'s SPSR regardless of the current mode (User/System map
    /// to the CPSR, as in [`Registers::spsr`]).
    pub fn spsr_of(&self, mode: Mode) -> Psr {
        match mode.bank() {
            Some(b) => self.spsr[b],
            None => self.cpsr,
        }
    }

    /// Write `mode`'s SPSR regardless of the current mode (no-op for
    /// User/System).
    pub fn set_spsr_of(&mut self, mode: Mode, p: Psr) {
        if let Some(b) = mode.bank() {
            self.spsr[b] = p;
        }
    }

    /// Read register `i` of the *user* bank, regardless of the current
    /// mode (the LDM/STM S-bit transfer set).
    #[inline]
    pub fn user_reg(&self, i: usize) -> u32 {
        let cur = self.cpsr.mode();
        match i {
            0..=7 | 15 => self.gpr[i],
            8..=12 if cur != Mode::Fiq => self.gpr[i],
            8..=12 => self.usr_bank[i - 8],
            _ if !cur.has_spsr() => self.gpr[i],
            _ => self.usr_bank[i - 8],
        }
    }

    /// Write register `i` of the *user* bank, regardless of the current
    /// mode.
    #[inline]
    pub fn set_user_reg(&mut self, i: usize, v: u32) {
        let cur = self.cpsr.mode();
        match i {
            0..=7 | 15 => self.gpr[i] = v,
            8..=12 if cur != Mode::Fiq => self.gpr[i] = v,
            8..=12 => self.usr_bank[i - 8] = v,
            _ if !cur.has_spsr() => self.gpr[i] = v,
            _ => self.usr_bank[i - 8] = v,
        }
    }

    /// Write register `i` as seen from `mode`, regardless of the current
    /// mode (the setter half of [`Registers::reg_of`]).
    pub fn set_reg_of(&mut self, mode: Mode, i: usize, v: u32) {
        let cur = self.cpsr.mode();
        if Mode::same_bank(cur, mode) || i < 8 || i == 15 {
            self.gpr[i] = v;
            return;
        }
        let fiq_banked = (8..=14).contains(&i) && (cur == Mode::Fiq) != (mode == Mode::Fiq);
        let exc_banked = (13..=14).contains(&i);
        if !fiq_banked && !exc_banked {
            self.gpr[i] = v;
            return;
        }
        match mode {
            Mode::Fiq => self.fiq_bank[i - 8] = v,
            m => {
                if i < 13 {
                    // Only reachable when `cur` is FIQ: r8–r12 of everyone else.
                    self.usr_bank[i - 8] = v;
                } else {
                    match m.bank() {
                        Some(b) => self.exc_bank[b - 1][i - 13] = v,
                        None => self.usr_bank[i - 8] = v,
                    }
                }
            }
        }
    }

    /// Read register `i` as seen from `mode`, regardless of the current
    /// mode (for tests and OS-emulation layers).
    pub fn reg_of(&self, mode: Mode, i: usize) -> u32 {
        let cur = self.cpsr.mode();
        if Mode::same_bank(cur, mode) || i < 8 || i == 15 {
            return self.gpr[i];
        }
        let fiq_banked = (8..=14).contains(&i) && (cur == Mode::Fiq) != (mode == Mode::Fiq);
        let exc_banked = (13..=14).contains(&i);
        if !fiq_banked && !exc_banked {
            return self.gpr[i];
        }
        match mode {
            Mode::Fiq => self.fiq_bank[i - 8],
            m => {
                if i < 13 {
                    // Only reachable when `cur` is FIQ: r8–r12 of everyone else.
                    self.usr_bank[i - 8]
                } else {
                    match m.bank() {
                        Some(b) => self.exc_bank[b - 1][i - 13],
                        None => self.usr_bank[i - 8],
                    }
                }
            }
        }
    }
}

impl Default for Registers {
    fn default() -> Self {
        Registers::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psr_stores_raw_bits() {
        let p = Psr::from_bits(0xFFFF_FFFF);
        assert_eq!(p.bits(), 0xFFFF_FFFF, "all 32 bits are storage");
        assert_eq!(p.mode(), Mode::Sys); // 0b11111
    }

    #[test]
    fn psr_reserved_mode_selects_user_bank() {
        let mut p = Psr::from_bits(Mode::Irq as u32);
        p.write(0x1A, psr::MODE); // reserved encoding, stored raw
        assert_eq!(p.mode_bits(), 0x1A);
        assert_eq!(p.mode(), Mode::Usr, "reserved modes use the user bank");
        assert!(p.privileged(), "reserved modes are not User proper");
    }

    #[test]
    fn mode_switch_banks_r13_r14() {
        let mut r = Registers::new(); // SVC
        r.gpr[13] = 0x1000;
        r.gpr[14] = 0x2000;
        r.gpr[7] = 77;
        r.set_mode(Mode::Irq);
        r.gpr[13] = 0x3000;
        assert_eq!(r.gpr[7], 77, "low registers are shared");
        r.set_mode(Mode::Svc);
        assert_eq!(r.gpr[13], 0x1000);
        assert_eq!(r.gpr[14], 0x2000);
        r.set_mode(Mode::Irq);
        assert_eq!(r.gpr[13], 0x3000);
    }

    #[test]
    fn fiq_banks_r8_to_r12() {
        let mut r = Registers::new();
        r.set_mode(Mode::Usr);
        for i in 8..15 {
            r.gpr[i] = i as u32;
        }
        r.set_mode(Mode::Fiq);
        for i in 8..15 {
            r.gpr[i] = 100 + i as u32;
            assert_eq!(r.user_reg(i), i as u32, "user bank visible from FIQ");
            assert_eq!(r.reg_of(Mode::Usr, i), i as u32);
        }
        r.set_mode(Mode::Usr);
        for i in 8..15 {
            assert_eq!(r.gpr[i], i as u32);
            assert_eq!(r.reg_of(Mode::Fiq, i), 100 + i as u32);
        }
    }

    #[test]
    fn usr_sys_share_bank() {
        let mut r = Registers::new();
        r.set_mode(Mode::Usr);
        r.gpr[13] = 0xAAAA;
        r.set_mode(Mode::Sys);
        assert_eq!(r.gpr[13], 0xAAAA);
        r.set_mode(Mode::Irq);
        assert_eq!(r.reg_of(Mode::Sys, 13), 0xAAAA);
        assert_eq!(r.user_reg(13), 0xAAAA);
    }

    #[test]
    fn set_user_reg_writes_through_banking() {
        let mut r = Registers::new(); // SVC
        r.set_user_reg(13, 0xBEEF);
        assert_eq!(r.reg_of(Mode::Usr, 13), 0xBEEF);
        assert_ne!(r.gpr[13], 0xBEEF);
        r.set_user_reg(3, 3);
        assert_eq!(r.gpr[3], 3, "low registers write straight through");
    }
}
