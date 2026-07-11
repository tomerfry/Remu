//! Microcode-accurate 8086 division: the CORD core loop and its signed
//! co-routines PREIDIV/POSTIDIV/NEGATE.
//!
//! Division on the 8086 is a microcode loop of 1-bit ALU steps, and the flags
//! it leaves behind — including the "undefined" ones — are fully determined by
//! those internal operations. This matters beyond curiosity: on a divide
//! fault, INT 0 pushes FLAGS to the stack, so the exact undefined-flag state
//! is architecturally visible (and asserted by the SingleStepTests suite).
//!
//! This is a translation of the reverse-engineered microcode (Ken Shirriff's
//! 8086 die analysis; Daniel Balsom's MartyPC implementation, MIT licensed),
//! minus the per-cycle bookkeeping. `AAM` reuses CORD as on real silicon.

use super::Cpu;
use super::registers::Flags;

macro_rules! div_width {
    ($t:ty, $sign:literal, $szp:ident,
     $sub:ident, $neg:ident, $rcl:ident,
     $cord:ident, $negate:ident, $prediv:ident, $postdiv:ident) => {
        /// `a - b` → (result, borrow, overflow, aux-carry) without touching CPU flags.
        #[inline]
        fn $sub(a: $t, b: $t) -> ($t, bool, bool, bool) {
            let r = a.wrapping_sub(b);
            (
                r,
                b > a,
                (a ^ b) & (a ^ r) & $sign != 0,
                (a ^ b ^ r) & 0x10 != 0,
            )
        }

        /// Two's-complement negate → (result, carry-out as for `0 - v`).
        #[inline]
        fn $neg(v: $t) -> ($t, bool) {
            ((0 as $t).wrapping_sub(v), v != 0)
        }

        /// 1-bit rotate-left-through-carry → (result, carry-out).
        #[inline]
        fn $rcl(v: $t, carry: bool) -> ($t, bool) {
            ((v << 1) | carry as $t, v & $sign != 0)
        }

        impl Cpu {
            /// CORD: the unsigned division core. `tmpa:tmpc` is the dividend, `tmpb`
            /// the divisor. Returns `(tmpc, tmpa, carry)` where the quotient is the
            /// *complement* of `tmpc` and `tmpa` is the remainder, or `Err(())` on a
            /// divide fault (flags already reflect the initial comparison).
            fn $cord(
                &mut self,
                mut tmpa: $t,
                tmpb: $t,
                mut tmpc: $t,
            ) -> Result<($t, $t, bool), ()> {
                // 188: SUBT tmpa | 189: F — fault unless the divisor exceeds the
                // dividend's high half (catches both div-by-zero and overflow).
                let (sigma, mut carry, of, af) = $sub(tmpa, tmpb);
                let f = &mut self.regs.flags;
                f.set(Flags::AF, af);
                f.set(Flags::OF, of);
                f.set(Flags::CF, carry);
                f.$szp(sigma);
                if !carry {
                    return Err(());
                }

                for _ in 0..<$t>::BITS {
                    // 18c/18d: rotate the dividend left through carry.
                    (tmpc, carry) = $rcl(tmpc, carry);
                    (tmpa, carry) = $rcl(tmpa, carry);
                    if carry {
                        // 195: RCY | 196: SUBT (no flag update)
                        carry = false;
                        tmpa = tmpa.wrapping_sub(tmpb);
                    } else {
                        // 18f: trial subtraction, updating the real flags.
                        let (sigma, cy, of, af) = $sub(tmpa, tmpb);
                        let f = &mut self.regs.flags;
                        f.set(Flags::AF, af);
                        f.set(Flags::OF, of);
                        f.set(Flags::CF, cy);
                        f.$szp(sigma);
                        carry = cy;
                        if !cy {
                            tmpa = sigma; // 196
                        }
                    }
                }

                // 192/194: final quotient-bit rotate; the carry out is the signed
                // overflow check consumed by POSTIDIV.
                (tmpc, carry) = $rcl(tmpc, carry);
                let (_, cy) = $rcl(tmpc, carry);
                carry = cy;
                self.regs.flags.set(Flags::CF, carry);
                Ok((tmpc, tmpa, carry))
            }

            /// NEGATE: flips the dividend (unless `skip`) and the divisor to positive,
            /// tracking the result sign in `neg`. Leaves CF = original divisor sign.
            fn $negate(
                &mut self,
                mut tmpa: $t,
                mut tmpb: $t,
                mut tmpc: $t,
                mut neg: bool,
                skip: bool,
            ) -> ($t, $t, $t, bool) {
                if !skip {
                    // 1b6..1ba: negate the double-width dividend tmpa:tmpc.
                    let (c, cy) = $neg(tmpc);
                    tmpc = c;
                    tmpa = if cy { !tmpa } else { $neg(tmpa).0 };
                    neg = !neg;
                }
                // 1bb..1be: CF = divisor sign; negate the divisor if negative.
                let carry = tmpb & $sign != 0;
                self.regs.flags.set(Flags::CF, carry);
                if carry {
                    tmpb = $neg(tmpb).0;
                    neg = !neg;
                }
                (tmpa, tmpb, tmpc, neg)
            }

            /// PREIDIV: entry gate for signed division — negate the dividend only if
            /// it is negative, always fix up the divisor.
            fn $prediv(&mut self, tmpa: $t, tmpb: $t, tmpc: $t, neg: bool) -> ($t, $t, $t, bool) {
                let negative = tmpa & $sign != 0;
                self.$negate(tmpa, tmpb, tmpc, neg, !negative)
            }

            /// POSTIDIV: signed fix-up after CORD. `tmpa` is the raw remainder,
            /// `tmpb` the *original* dividend high half, `tmpc` the complemented
            /// quotient, `carry` CORD's range check. Returns (remainder, quotient).
            fn $postdiv(
                &mut self,
                tmpa: $t,
                tmpb: $t,
                tmpc: $t,
                carry: bool,
                neg: bool,
            ) -> Result<($t, $t), ()> {
                // 1c4: quotient out of signed range → fault.
                if !carry {
                    return Err(());
                }
                // 1c5..1c8: the remainder takes the sign of the dividend.
                let tmpa = if tmpb & $sign != 0 {
                    $neg(tmpa).0
                } else {
                    tmpa
                };
                // 1c9..1cb: un-complement the quotient; a REP prefix (F1) negates it
                // instead — the famous 8086 REP IDIV quirk.
                let sigma = if neg { tmpc.wrapping_add(1) } else { !tmpc };
                // 1cc: CCOF
                self.regs.flags.remove(Flags::CF | Flags::OF);
                Ok((tmpa, sigma))
            }
        }
    };
}

div_width!(
    u8, 0x80, set_szp8, sub8_fl, neg8_v, rcl8_v, cord8, negate8, prediv8, postdiv8
);
div_width!(
    u16, 0x8000, set_szp16, sub16_fl, neg16_v, rcl16_v, cord16, negate16, prediv16, postdiv16
);

impl Cpu {
    /// `DIV`/`IDIV r/m8`: divide AX by `divisor` into AL (quotient) and AH
    /// (remainder). Returns `false` on a divide fault, with the flags exactly
    /// as the microcode leaves them (the caller raises INT 0).
    #[must_use]
    pub(crate) fn div8(&mut self, divisor: u8, signed: bool) -> bool {
        let dividend = self.regs.ax;
        let mut tmpa = (dividend >> 8) as u8;
        let mut tmpc = dividend as u8;
        let mut tmpb = divisor;
        let mut negate = signed && self.rep.is_some();
        if signed {
            (tmpa, tmpb, tmpc, negate) = self.prediv8(tmpa, tmpb, tmpc, negate);
        }
        let Ok((c, a, carry)) = self.cord8(tmpa, tmpb, tmpc) else {
            return false;
        };
        let (mut rem, tmpc) = (a, c);
        let mut quot = !tmpc;
        if signed {
            match self.postdiv8(rem, (dividend >> 8) as u8, tmpc, carry, negate) {
                Ok((r, q)) => (rem, quot) = (r, q),
                Err(()) => return false,
            }
        }
        self.regs.ax = (rem as u16) << 8 | quot as u16;
        true
    }

    /// `DIV`/`IDIV r/m16`: divide DX:AX by `divisor` into AX/DX.
    #[must_use]
    pub(crate) fn div16(&mut self, divisor: u16, signed: bool) -> bool {
        let (mut tmpa, mut tmpc) = (self.regs.dx, self.regs.ax);
        let dividend_hi = tmpa;
        let mut tmpb = divisor;
        let mut negate = signed && self.rep.is_some();
        if signed {
            (tmpa, tmpb, tmpc, negate) = self.prediv16(tmpa, tmpb, tmpc, negate);
        }
        let Ok((c, a, carry)) = self.cord16(tmpa, tmpb, tmpc) else {
            return false;
        };
        let (mut rem, tmpc) = (a, c);
        let mut quot = !tmpc;
        if signed {
            match self.postdiv16(rem, dividend_hi, tmpc, carry, negate) {
                Ok((r, q)) => (rem, quot) = (r, q),
                Err(()) => return false,
            }
        }
        self.regs.ax = quot;
        self.regs.dx = rem;
        true
    }

    /// `AAM base` (D4 ib): AL → AH:AL via the CORD divider, as on real
    /// silicon. Returns `false` on base 0 (divide fault) with the observed
    /// hardware flag state.
    #[must_use]
    pub(crate) fn aam(&mut self, base: u8) -> bool {
        let al = self.regs.ax as u8;
        match self.cord8(0, base, al) {
            Ok((quotient, remainder, _)) => {
                self.regs.ax = ((!quotient) as u16) << 8 | remainder as u16;
                self.regs.flags.set_szp8(remainder);
                self.regs.flags.remove(Flags::CF | Flags::AF | Flags::OF);
                true
            }
            Err(()) => {
                self.regs.flags.set_szp8(0);
                self.regs.flags.remove(Flags::CF | Flags::AF | Flags::OF);
                false
            }
        }
    }
}
