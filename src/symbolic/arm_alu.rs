//! Symbolic counterpart to `arm32::alu` — the NZCV world.
//!
//! ARM's ALU differs from x86's: `add_with_carry`/`sub_with_carry` return
//! `(result, carry_out, overflow)`, N and Z come from the 32-bit result, and
//! the C flag of a *logical* op is the **barrel-shifter carry**, not an adder
//! carry. Subtraction is `add_with_carry(a, !b, cin)`, so C is the *not-borrow*.
//! These builders mirror the concrete ARM ALU and are pinned by the
//! differential test below. Everything is 32-bit.

use super::expr::{BinOp, BoolExpr, CmpOp, Expr, UnOp};

/// ARM registers are 32-bit.
pub const W: u16 = 32;

/// Barrel-shifter operation (bits 6:5 of a data-processing instruction).
pub const LSL: u8 = 0;
pub const LSR: u8 = 1;
pub const ASR: u8 = 2;
pub const ROR: u8 = 3;

#[inline]
fn cw(k: u64) -> Expr {
    Expr::constant(W, k)
}

/// `(result, carry_out, overflow)` for `a + b + cin`.
pub fn add_with_carry(a: &Expr, b: &Expr, cin: &BoolExpr) -> (Expr, BoolExpr, BoolExpr) {
    let c = Expr::ite(cin.clone(), cw(1), cw(0));
    let r = Expr::bin(BinOp::Add, Expr::bin(BinOp::Add, a.clone(), b.clone()), c);
    // Width-safe carry-out: a + b + cin overflowed ⇔ r < a, or r == a with cin.
    let carry = BoolExpr::or(
        BoolExpr::cmp(CmpOp::Ult, r.clone(), a.clone()),
        BoolExpr::and(BoolExpr::cmp(CmpOp::Eq, r.clone(), a.clone()), cin.clone()),
    );
    // Signed overflow: operands agree in sign but differ from the result.
    let ar = Expr::bin(BinOp::Xor, a.clone(), r.clone());
    let br = Expr::bin(BinOp::Xor, b.clone(), r.clone());
    let v = BoolExpr::bit_of(Expr::bin(BinOp::And, ar, br), W - 1);
    (r, carry, v)
}

/// `(result, carry_out, overflow)` for `a - b - !cin` — ARM's `a + !b + cin`.
pub fn sub_with_carry(a: &Expr, b: &Expr, cin: &BoolExpr) -> (Expr, BoolExpr, BoolExpr) {
    add_with_carry(a, &Expr::un(UnOp::Not, b.clone()), cin)
}

/// N (bit 31) and Z (== 0) of a result.
pub fn nz(r: &Expr) -> (BoolExpr, BoolExpr) {
    (
        BoolExpr::bit_of(r.clone(), W - 1),
        BoolExpr::cmp(CmpOp::Eq, r.clone(), cw(0)),
    )
}

/// Immediate barrel shift: `(result, shifter_carry)`. `amount` is 0..31,
/// concrete. Encodes the amount-0 specials (LSL#0 passthrough, LSR#0/ASR#0 =
/// shift-by-32, ROR#0 = RRX).
pub fn shift_imm(ty: u8, amount: u8, value: &Expr, cin: &BoolExpr) -> (Expr, BoolExpr) {
    let n = amount as u64;
    let bit = |i: u16| BoolExpr::bit_of(value.clone(), i);
    match (ty, amount) {
        (LSL, 0) => (value.clone(), cin.clone()),
        (LSL, _) => (Expr::bin(BinOp::Shl, value.clone(), cw(n)), bit(W - amount as u16)),
        (LSR, 0) => (cw(0), bit(W - 1)),
        (LSR, _) => (Expr::bin(BinOp::Lshr, value.clone(), cw(n)), bit(amount as u16 - 1)),
        (ASR, 0) => (Expr::bin(BinOp::Ashr, value.clone(), cw((W - 1) as u64)), bit(W - 1)),
        (ASR, _) => (Expr::bin(BinOp::Ashr, value.clone(), cw(n)), bit(amount as u16 - 1)),
        (ROR, 0) => {
            // RRX: result = (value >> 1) | (cin << 31), carry = bit 0.
            let r = Expr::bin(
                BinOp::Or,
                Expr::bin(BinOp::Lshr, value.clone(), cw(1)),
                Expr::ite(cin.clone(), cw(0x8000_0000), cw(0)),
            );
            (r, bit(0))
        }
        (_, _) => {
            // ROR n: rotate right.
            let r = Expr::bin(
                BinOp::Or,
                Expr::bin(BinOp::Lshr, value.clone(), cw(n)),
                Expr::bin(BinOp::Shl, value.clone(), cw(W as u64 - n)),
            );
            (r, bit(amount as u16 - 1))
        }
    }
}

#[cfg(test)]
mod tests {
    //! Differential test: symbolic == concrete `arm32::alu` for the
    //! add/sub carry+overflow rules and the immediate barrel shifter, over
    //! random inputs.

    use super::*;
    use crate::arm32::alu as concrete;
    use crate::symbolic::expr::Model;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    #[test]
    fn add_sub_match_concrete() {
        let mut rng = Rng(0x51ce_1234_9876_abcd);
        let model = Model::new();
        for _ in 0..20000 {
            let a = rng.next() as u32;
            let b = rng.next() as u32;
            let cin = rng.next() & 1 == 1;
            let ec = BoolExpr::constant(cin);
            let ea = Expr::constant(W, a as u64);
            let eb = Expr::constant(W, b as u64);

            let (r, c, v) = concrete::add_with_carry(a, b, cin);
            let (sr, sc, sv) = add_with_carry(&ea, &eb, &ec);
            assert_eq!(sr.eval(&model), r as u64, "add result a={a:x} b={b:x} cin={cin}");
            assert_eq!(sc.eval(&model), c, "add carry a={a:x} b={b:x} cin={cin}");
            assert_eq!(sv.eval(&model), v, "add ovf a={a:x} b={b:x} cin={cin}");

            let (r, c, v) = concrete::sub_with_carry(a, b, cin);
            let (sr, sc, sv) = sub_with_carry(&ea, &eb, &ec);
            assert_eq!(sr.eval(&model), r as u64, "sub result a={a:x} b={b:x} cin={cin}");
            assert_eq!(sc.eval(&model), c, "sub carry a={a:x} b={b:x} cin={cin}");
            assert_eq!(sv.eval(&model), v, "sub ovf a={a:x} b={b:x} cin={cin}");
        }
    }

    #[test]
    fn shift_imm_matches_concrete() {
        let mut rng = Rng(0xa11c_e5e5_0f0f_1122);
        let model = Model::new();
        for _ in 0..8000 {
            let v = rng.next() as u32;
            let cin = rng.next() & 1 == 1;
            let ev = Expr::constant(W, v as u64);
            let ec = BoolExpr::constant(cin);
            for ty in 0..4u8 {
                for amount in 0..32u8 {
                    let (r, c) = concrete::shift_imm(ty, amount, v, cin);
                    let (sr, sc) = shift_imm(ty, amount, &ev, &ec);
                    assert_eq!(sr.eval(&model), r as u64, "shift ty={ty} n={amount} v={v:x}");
                    assert_eq!(sc.eval(&model), c, "shift carry ty={ty} n={amount} v={v:x} cin={cin}");
                }
            }
        }
    }
}
