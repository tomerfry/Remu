//! Flag-exact ALU helpers: the barrel shifter and the add/subtract
//! carry/overflow rules shared by the ARM and Thumb executors.

/// Barrel-shifter operation, as encoded in bits 6:5 of a data-processing
/// instruction.
pub(crate) const LSL: u8 = 0;
pub(crate) const LSR: u8 = 1;
pub(crate) const ASR: u8 = 2;
pub(crate) const ROR: u8 = 3;

/// Shift `value` by an *immediate* amount, returning the result and the
/// shifter carry-out. Encodes the amount-0 special cases: `LSL #0` passes
/// the value through with the old carry, `LSR #0`/`ASR #0` mean a full
/// 32-bit shift, and `ROR #0` is RRX (rotate right through carry).
#[inline]
pub(crate) fn shift_imm(ty: u8, amount: u8, value: u32, carry_in: bool) -> (u32, bool) {
    match (ty, amount) {
        (LSL, 0) => (value, carry_in),
        (LSL, n) => (value << n, value & (1 << (32 - n)) != 0),
        (LSR, 0) => (0, value >> 31 != 0),
        (LSR, n) => (value >> n, value & (1 << (n - 1)) != 0),
        (ASR, 0) => (((value as i32) >> 31) as u32, value >> 31 != 0),
        (ASR, n) => (((value as i32) >> n) as u32, value & (1 << (n - 1)) != 0),
        (ROR, 0) => (value >> 1 | (carry_in as u32) << 31, value & 1 != 0),
        (_, n) => (value.rotate_right(n as u32), value & (1 << (n - 1)) != 0),
    }
}

/// Shift `value` by a *register-specified* amount (the low byte of Rs),
/// returning the result and the shifter carry-out. Amount 0 passes the
/// value through with the old carry; amounts of 32 and beyond take the
/// architected saturating forms (no special encodings here — the byte is
/// used as-is, as on hardware).
#[inline]
pub(crate) fn shift_reg(ty: u8, amount: u8, value: u32, carry_in: bool) -> (u32, bool) {
    if amount == 0 {
        return (value, carry_in);
    }
    let n = amount as u32;
    match ty {
        LSL => match n {
            1..=31 => (value << n, value & (1 << (32 - n)) != 0),
            32 => (0, value & 1 != 0),
            _ => (0, false),
        },
        LSR => match n {
            1..=31 => (value >> n, value & (1 << (n - 1)) != 0),
            32 => (0, value >> 31 != 0),
            _ => (0, false),
        },
        ASR => match n {
            1..=31 => (((value as i32) >> n) as u32, value & (1 << (n - 1)) != 0),
            _ => (((value as i32) >> 31) as u32, value >> 31 != 0),
        },
        _ => {
            // ROR by n: only the low five bits matter once n > 0; a
            // multiple of 32 leaves the value and copies bit 31 to carry.
            let r = n & 31;
            if r == 0 {
                (value, value >> 31 != 0)
            } else {
                (value.rotate_right(r), value & (1 << (r - 1)) != 0)
            }
        }
    }
}

/// `a + b + carry` with the ARM carry/overflow rules: returns
/// `(result, carry_out, overflow)`.
#[inline]
pub(crate) fn add_with_carry(a: u32, b: u32, carry: bool) -> (u32, bool, bool) {
    let (r1, c1) = a.overflowing_add(b);
    let (r, c2) = r1.overflowing_add(carry as u32);
    let v = ((a ^ r) & (b ^ r)) >> 31 != 0;
    (r, c1 | c2, v)
}

/// `a - b - !carry` (SBC; SUB passes `carry = true`) with the ARM borrow
/// convention: carry-out is *set* when no borrow occurred.
#[inline]
pub(crate) fn sub_with_carry(a: u32, b: u32, carry: bool) -> (u32, bool, bool) {
    add_with_carry(a, !b, carry)
}

/// Booth's-algorithm early-termination cycle count for MUL/MLA/MULL: the
/// multiplier array consumes `Rs` eight bits per cycle and stops once the
/// remainder is all-zeros (or, for signed variants, all-ones).
#[inline]
pub(crate) fn mul_cycles(rs: u32, signed: bool) -> u32 {
    for m in 1..4 {
        let rest = rs >> (8 * m);
        let ones = u32::MAX >> (8 * m);
        if rest == 0 || (signed && rest == ones) {
            return m;
        }
    }
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_imm_special_encodings() {
        // LSR #0 encodes LSR #32.
        assert_eq!(shift_imm(LSR, 0, 0x8000_0001, false), (0, true));
        // ASR #0 encodes ASR #32.
        assert_eq!(shift_imm(ASR, 0, 0x8000_0000, false), (0xFFFF_FFFF, true));
        assert_eq!(shift_imm(ASR, 0, 0x7FFF_FFFF, true), (0, false));
        // ROR #0 encodes RRX.
        assert_eq!(shift_imm(ROR, 0, 0x0000_0003, false), (1, true));
        assert_eq!(shift_imm(ROR, 0, 0x0000_0002, true), (0x8000_0001, false));
        // LSL #0 keeps the carry.
        assert_eq!(shift_imm(LSL, 0, 5, true), (5, true));
    }

    #[test]
    fn shift_reg_wide_amounts() {
        assert_eq!(shift_reg(LSL, 32, 1, false), (0, true));
        assert_eq!(shift_reg(LSL, 33, 1, true), (0, false));
        assert_eq!(shift_reg(LSR, 32, 0x8000_0000, false), (0, true));
        assert_eq!(shift_reg(ASR, 40, 0x8000_0000, false), (0xFFFF_FFFF, true));
        // ROR by a multiple of 32 keeps the value, carry = bit 31.
        assert_eq!(shift_reg(ROR, 32, 0x8000_0001, false), (0x8000_0001, true));
        assert_eq!(shift_reg(ROR, 33, 0x8000_0001, false), (0xC000_0000, true));
        // Amount 0 (an Rs whose low byte is zero) changes nothing.
        assert_eq!(shift_reg(ROR, 0, 7, true), (7, true));
    }

    #[test]
    fn adc_sbc_flags() {
        assert_eq!(add_with_carry(u32::MAX, 1, false), (0, true, false));
        assert_eq!(
            add_with_carry(0x7FFF_FFFF, 1, false),
            (0x8000_0000, false, true)
        );
        // SUB 0 - 1: borrow (carry clear), no overflow.
        assert_eq!(sub_with_carry(0, 1, true), (u32::MAX, false, false));
        // SUB min - 1: overflow.
        assert_eq!(
            sub_with_carry(0x8000_0000, 1, true),
            (0x7FFF_FFFF, true, true)
        );
    }

    #[test]
    fn mul_early_termination() {
        assert_eq!(mul_cycles(0x0000_00FF, false), 1);
        assert_eq!(mul_cycles(0x0000_FF00, false), 2);
        assert_eq!(mul_cycles(0x00FF_0000, false), 3);
        assert_eq!(mul_cycles(0xFF00_0000, false), 4);
        assert_eq!(
            mul_cycles(0xFFFF_FFFF, true),
            1,
            "all-ones terminates signed"
        );
        assert_eq!(mul_cycles(0xFFFF_FFFF, false), 4);
    }
}
