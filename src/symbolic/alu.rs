//! Symbolic counterpart to `x86_32::alu`.
//!
//! Given symbolic operands and an operation, these builders produce the result
//! expression plus the flags the operation defines — mirroring the concrete
//! 386 ALU bit-for-bit. The eight core ALU ops are keyed by the SAME index as
//! `x86_32::execute::{ALU8,ALU16,ALU32}` (ADD OR ADC SBB AND SUB XOR CMP), so
//! the concolic overlay can dispatch the concrete and symbolic sides
//! identically.
//!
//! Because `Expr` carries its width at runtime, each op needs a single
//! width-generic builder here (versus the concrete side's per-width macro
//! expansion). Flags are returned as a [`FlagDefs`] where `None` means "left
//! unchanged" — matching the concrete ALU, which writes only a subset (e.g.
//! `INC` preserves `CF`). Correctness is pinned by the differential test at the
//! bottom, which checks every op × width against the concrete methods.
//!
//! Not yet modeled (deferred to a later milestone; concolic concretizes them
//! from the live run in the meantime): `RCL`/`RCR` (rotate-through-carry loops),
//! multiply/divide, and the BCD adjusts.

use super::expr::{BinOp, BoolExpr, CmpOp, Expr, Width};

/// The six 386 status flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flag {
    Cf,
    Pf,
    Af,
    Zf,
    Sf,
    Of,
}

/// The flags an operation defines. `None` = unchanged by this op.
#[derive(Debug, Clone, Default)]
pub struct FlagDefs {
    pub cf: Option<BoolExpr>,
    pub pf: Option<BoolExpr>,
    pub af: Option<BoolExpr>,
    pub zf: Option<BoolExpr>,
    pub sf: Option<BoolExpr>,
    pub of: Option<BoolExpr>,
}

/// The value `1 << (width - 1)` — the sign bit position's mask.
#[inline]
fn sign_val(w: Width) -> u64 {
    1u64 << (w - 1)
}

/// Even parity of the low 8 bits of `r` (the 386 `PF`): `NOT(xor of bits 0..8)`.
fn parity8(r: &Expr) -> BoolExpr {
    let mut acc = BoolExpr::bit_of(r.clone(), 0);
    for i in 1..8 {
        acc = BoolExpr::xor(acc, BoolExpr::bit_of(r.clone(), i));
    }
    BoolExpr::not(acc)
}

/// `SF`, `ZF`, `PF` from a `w`-bit result — the shared tail of the arithmetic
/// and logic ops. Returned as a partial [`FlagDefs`] (the other flags `None`)
/// so callers can fill the rest with struct-update syntax.
fn szp(r: &Expr, w: Width) -> FlagDefs {
    FlagDefs {
        sf: Some(BoolExpr::bit_of(r.clone(), w - 1)),
        zf: Some(BoolExpr::cmp(CmpOp::Eq, r.clone(), Expr::constant(w, 0))),
        pf: Some(parity8(r)),
        ..FlagDefs::default()
    }
}

#[inline]
fn xor(a: &Expr, b: &Expr) -> Expr {
    Expr::bin(BinOp::Xor, a.clone(), b.clone())
}

/// `AF = bit 3 carry = (a ^ b ^ r) & 0x10 != 0`.
fn af_carry(a: &Expr, b: &Expr, r: &Expr) -> BoolExpr {
    BoolExpr::bit_of(xor(&xor(a, b), r), 4)
}

/// `OF` for addition: `(a ^ r) & (b ^ r) & sign != 0`.
fn of_add(a: &Expr, b: &Expr, r: &Expr, w: Width) -> BoolExpr {
    let t = Expr::bin(BinOp::And, xor(a, r), xor(b, r));
    BoolExpr::bit_of(t, w - 1)
}

/// `OF` for subtraction: `(a ^ b) & (a ^ r) & sign != 0`.
fn of_sub(a: &Expr, b: &Expr, r: &Expr, w: Width) -> BoolExpr {
    let t = Expr::bin(BinOp::And, xor(a, b), xor(a, r));
    BoolExpr::bit_of(t, w - 1)
}

/// `ADD`.
pub fn add(a: &Expr, b: &Expr, w: Width) -> (Expr, FlagDefs) {
    let r = Expr::bin(BinOp::Add, a.clone(), b.clone());
    let f = FlagDefs {
        cf: Some(BoolExpr::cmp(CmpOp::Ult, r.clone(), a.clone())), // wrap ⇒ r < a
        af: Some(af_carry(a, b, &r)),
        of: Some(of_add(a, b, &r, w)),
        ..szp(&r, w)
    };
    (r, f)
}

/// `ADC` (add with the symbolic carry-in `cin`).
pub fn adc(a: &Expr, b: &Expr, cin: &BoolExpr, w: Width) -> (Expr, FlagDefs) {
    let c = Expr::ite(cin.clone(), Expr::constant(w, 1), Expr::constant(w, 0));
    let r = Expr::bin(BinOp::Add, Expr::bin(BinOp::Add, a.clone(), b.clone()), c);
    // CF = carry out of the top bit, computed in width+1 with no wrap.
    let w1 = w + 1;
    let ea = Expr::zext(w1, a.clone());
    let eb = Expr::zext(w1, b.clone());
    let ec = Expr::ite(cin.clone(), Expr::constant(w1, 1), Expr::constant(w1, 0));
    let sum = Expr::bin(BinOp::Add, Expr::bin(BinOp::Add, ea, eb), ec);
    let f = FlagDefs {
        cf: Some(BoolExpr::bit_of(sum, w)),
        af: Some(af_carry(a, b, &r)),
        of: Some(of_add(a, b, &r, w)),
        ..szp(&r, w)
    };
    (r, f)
}

/// `SUB` (and `CMP`, which shares its flags and discards the result).
pub fn sub(a: &Expr, b: &Expr, w: Width) -> (Expr, FlagDefs) {
    let r = Expr::bin(BinOp::Sub, a.clone(), b.clone());
    let f = FlagDefs {
        cf: Some(BoolExpr::cmp(CmpOp::Ult, a.clone(), b.clone())), // borrow ⇒ b > a
        af: Some(af_carry(a, b, &r)),
        of: Some(of_sub(a, b, &r, w)),
        ..szp(&r, w)
    };
    (r, f)
}

/// `SBB` (subtract with the symbolic borrow-in `cin`).
pub fn sbb(a: &Expr, b: &Expr, cin: &BoolExpr, w: Width) -> (Expr, FlagDefs) {
    let c = Expr::ite(cin.clone(), Expr::constant(w, 1), Expr::constant(w, 0));
    let r = Expr::bin(BinOp::Sub, Expr::bin(BinOp::Sub, a.clone(), b.clone()), c);
    // CF = (b + c) > a, computed in width+1.
    let w1 = w + 1;
    let ea = Expr::zext(w1, a.clone());
    let eb = Expr::zext(w1, b.clone());
    let ec = Expr::ite(cin.clone(), Expr::constant(w1, 1), Expr::constant(w1, 0));
    let bc = Expr::bin(BinOp::Add, eb, ec);
    let f = FlagDefs {
        cf: Some(BoolExpr::cmp(CmpOp::Ult, ea, bc)),
        af: Some(af_carry(a, b, &r)),
        of: Some(of_sub(a, b, &r, w)),
        ..szp(&r, w)
    };
    (r, f)
}

/// `AND`/`OR`/`XOR`: clear `CF`/`OF`/`AF`, set `SZP` from the result.
fn logic(op: BinOp, a: &Expr, b: &Expr, w: Width) -> (Expr, FlagDefs) {
    let r = Expr::bin(op, a.clone(), b.clone());
    let f = FlagDefs {
        cf: Some(BoolExpr::constant(false)),
        of: Some(BoolExpr::constant(false)),
        af: Some(BoolExpr::constant(false)),
        ..szp(&r, w)
    };
    (r, f)
}

/// Dispatch the eight core ALU ops by their encoding index (bits 5–3 of the
/// opcode), matching `x86_32::execute::ALU*`. `cin` supplies the carry/borrow
/// for `ADC`/`SBB` and is ignored otherwise.
pub fn alu(idx: usize, a: &Expr, b: &Expr, cin: &BoolExpr, w: Width) -> (Expr, FlagDefs) {
    match idx {
        0 => add(a, b, w),
        1 => logic(BinOp::Or, a, b, w),
        2 => adc(a, b, cin, w),
        3 => sbb(a, b, cin, w),
        4 => logic(BinOp::And, a, b, w),
        5 | 7 => sub(a, b, w), // SUB and CMP
        6 => logic(BinOp::Xor, a, b, w),
        _ => unreachable!("alu index {idx} out of range"),
    }
}

/// `INC`: add 1 but preserve `CF`.
pub fn inc(a: &Expr, w: Width) -> (Expr, FlagDefs) {
    let r = Expr::bin(BinOp::Add, a.clone(), Expr::constant(w, 1));
    let low = Expr::bin(BinOp::And, a.clone(), Expr::constant(w, 0xF));
    let f = FlagDefs {
        af: Some(BoolExpr::cmp(CmpOp::Eq, low, Expr::constant(w, 0xF))),
        of: Some(BoolExpr::cmp(CmpOp::Eq, r.clone(), Expr::constant(w, sign_val(w)))),
        ..szp(&r, w)
    };
    (r, f)
}

/// `DEC`: subtract 1 but preserve `CF`.
pub fn dec(a: &Expr, w: Width) -> (Expr, FlagDefs) {
    let r = Expr::bin(BinOp::Sub, a.clone(), Expr::constant(w, 1));
    let low = Expr::bin(BinOp::And, a.clone(), Expr::constant(w, 0xF));
    let f = FlagDefs {
        af: Some(BoolExpr::cmp(CmpOp::Eq, low, Expr::constant(w, 0))),
        of: Some(BoolExpr::cmp(CmpOp::Eq, a.clone(), Expr::constant(w, sign_val(w)))),
        ..szp(&r, w)
    };
    (r, f)
}

/// `NEG`.
pub fn neg(a: &Expr, w: Width) -> (Expr, FlagDefs) {
    let r = Expr::bin(BinOp::Sub, Expr::constant(w, 0), a.clone());
    let low = Expr::bin(BinOp::And, a.clone(), Expr::constant(w, 0xF));
    let f = FlagDefs {
        cf: Some(BoolExpr::cmp(CmpOp::Ne, a.clone(), Expr::constant(w, 0))),
        af: Some(BoolExpr::cmp(CmpOp::Ne, low, Expr::constant(w, 0))),
        of: Some(BoolExpr::cmp(CmpOp::Eq, a.clone(), Expr::constant(w, sign_val(w)))),
        ..szp(&r, w)
    };
    (r, f)
}

/// The five shift/rotate operations (excluding the carry-through `RCL`/`RCR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    Shl,
    Shr,
    Sar,
    Rol,
    Ror,
}

/// A shift/rotate with a **concrete** count (the common case: `imm8` or a
/// concrete `CL`). A symbolic count is concretized by the overlay before
/// calling here. Count 0 (after the 5-bit mask) leaves all flags unchanged,
/// matching the 386.
pub fn shift(kind: ShiftKind, v: &Expr, count: u32, w: Width) -> (Expr, FlagDefs) {
    let n = count & 31;
    if n == 0 {
        return (v.clone(), FlagDefs::default());
    }
    let wu = w as u32;
    let cw = |k: u64| Expr::constant(w, k);
    match kind {
        ShiftKind::Shl => {
            // Model the 386's 64-bit barrel shifter, then truncate.
            let wide = Expr::bin(BinOp::Shl, Expr::zext(64, v.clone()), Expr::constant(64, n as u64));
            let r = Expr::extract(w - 1, 0, wide.clone());
            let cf = if n.is_multiple_of(wu) {
                BoolExpr::bit_of(v.clone(), 0)
            } else {
                BoolExpr::bit_of(wide, w)
            };
            let f = FlagDefs {
                of: Some(BoolExpr::xor(BoolExpr::bit_of(r.clone(), w - 1), cf.clone())),
                cf: Some(cf),
                af: Some(BoolExpr::constant(false)),
                ..szp(&r, w)
            };
            (r, f)
        }
        ShiftKind::Shr => {
            let r = Expr::bin(BinOp::Lshr, v.clone(), cw(n as u64));
            let cf = if n.is_multiple_of(wu) {
                BoolExpr::bit_of(v.clone(), w - 1)
            } else {
                let shifted = Expr::bin(BinOp::Lshr, Expr::zext(64, v.clone()), Expr::constant(64, (n - 1) as u64));
                BoolExpr::bit_of(shifted, 0)
            };
            // OF = bit (w-1) of (v >> (n-1)): the pre-last-step MSB.
            let of_src = Expr::bin(BinOp::Lshr, Expr::zext(64, v.clone()), Expr::constant(64, (n - 1) as u64));
            let f = FlagDefs {
                cf: Some(cf),
                of: Some(BoolExpr::bit_of(of_src, w - 1)),
                af: Some(BoolExpr::constant(false)),
                ..szp(&r, w)
            };
            (r, f)
        }
        ShiftKind::Sar => {
            let r = Expr::bin(BinOp::Ashr, v.clone(), cw(n as u64));
            // CF = bit (n-1) of the sign-extended value.
            let sx = Expr::sext(64, v.clone());
            let cf = BoolExpr::bit_of(Expr::bin(BinOp::Ashr, sx, Expr::constant(64, (n - 1) as u64)), 0);
            let f = FlagDefs {
                cf: Some(cf),
                of: Some(BoolExpr::constant(false)),
                af: Some(BoolExpr::constant(false)),
                ..szp(&r, w)
            };
            (r, f)
        }
        ShiftKind::Rol => {
            let m = n % wu;
            let r = if m == 0 {
                v.clone()
            } else {
                Expr::bin(
                    BinOp::Or,
                    Expr::bin(BinOp::Shl, v.clone(), cw(m as u64)),
                    Expr::bin(BinOp::Lshr, v.clone(), cw((wu - m) as u64)),
                )
            };
            let cf = BoolExpr::bit_of(r.clone(), 0);
            // ROL leaves SF/ZF/PF/AF untouched.
            let f = FlagDefs {
                of: Some(BoolExpr::xor(BoolExpr::bit_of(r.clone(), w - 1), cf.clone())),
                cf: Some(cf),
                ..FlagDefs::default()
            };
            (r, f)
        }
        ShiftKind::Ror => {
            let m = n % wu;
            let r = if m == 0 {
                v.clone()
            } else {
                Expr::bin(
                    BinOp::Or,
                    Expr::bin(BinOp::Lshr, v.clone(), cw(m as u64)),
                    Expr::bin(BinOp::Shl, v.clone(), cw((wu - m) as u64)),
                )
            };
            // OF = bit (w-1) of (r ^ (r << 1)); ROR leaves SF/ZF/PF/AF untouched.
            let t = Expr::bin(BinOp::Xor, r.clone(), Expr::bin(BinOp::Shl, r.clone(), cw(1)));
            let f = FlagDefs {
                cf: Some(BoolExpr::bit_of(r.clone(), w - 1)),
                of: Some(BoolExpr::bit_of(t, w - 1)),
                ..FlagDefs::default()
            };
            (r, f)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Differential test: for every op × width, run the concrete 386 ALU and
    //! the symbolic builder on the same random inputs, evaluate the symbolic
    //! result and flags, and assert they agree exactly. This is M0's exit
    //! criterion — it pins the symbolic ALU to the concrete reference.

    use super::*;
    use crate::symbolic::expr::Model;
    use crate::x86_32::Cpu;
    use crate::x86_32::registers::EFlags;

    /// Deterministic xorshift PRNG (reproducible test runs).
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

    /// The six status flags in [`Flag`] order: Cf, Pf, Af, Zf, Sf, Of.
    fn concrete_flags(f: EFlags) -> [bool; 6] {
        [
            f.contains(EFlags::CF),
            f.contains(EFlags::PF),
            f.contains(EFlags::AF),
            f.contains(EFlags::ZF),
            f.contains(EFlags::SF),
            f.contains(EFlags::OF),
        ]
    }

    /// Materialize `defs` to concrete flags under `model`, falling back to the
    /// prior value where the op leaves a flag unchanged.
    fn sym_flags(defs: &FlagDefs, prior: [bool; 6], model: &Model) -> [bool; 6] {
        let m = |o: &Option<BoolExpr>, p: bool| o.as_ref().map(|e| e.eval(model)).unwrap_or(p);
        [
            m(&defs.cf, prior[0]),
            m(&defs.pf, prior[1]),
            m(&defs.af, prior[2]),
            m(&defs.zf, prior[3]),
            m(&defs.sf, prior[4]),
            m(&defs.of, prior[5]),
        ]
    }

    fn concrete_alu8(cpu: &mut Cpu, idx: usize, a: u8, b: u8) -> u8 {
        match idx {
            0 => cpu.add8(a, b),
            1 => cpu.or8(a, b),
            2 => cpu.adc8(a, b),
            3 => cpu.sbb8(a, b),
            4 => cpu.and8(a, b),
            5 => cpu.sub8(a, b),
            6 => cpu.xor8(a, b),
            7 => cpu.sub8(a, b),
            _ => unreachable!(),
        }
    }
    fn concrete_alu16(cpu: &mut Cpu, idx: usize, a: u16, b: u16) -> u16 {
        match idx {
            0 => cpu.add16(a, b),
            1 => cpu.or16(a, b),
            2 => cpu.adc16(a, b),
            3 => cpu.sbb16(a, b),
            4 => cpu.and16(a, b),
            5 => cpu.sub16(a, b),
            6 => cpu.xor16(a, b),
            7 => cpu.sub16(a, b),
            _ => unreachable!(),
        }
    }
    fn concrete_alu32(cpu: &mut Cpu, idx: usize, a: u32, b: u32) -> u32 {
        match idx {
            0 => cpu.add32(a, b),
            1 => cpu.or32(a, b),
            2 => cpu.adc32(a, b),
            3 => cpu.sbb32(a, b),
            4 => cpu.and32(a, b),
            5 => cpu.sub32(a, b),
            6 => cpu.xor32(a, b),
            7 => cpu.sub32(a, b),
            _ => unreachable!(),
        }
    }

    fn seed_flags(rng: &mut Rng, cin: bool) -> EFlags {
        let mut f = EFlags::from_bits_truncate(rng.next() as u32);
        f.set(EFlags::CF, cin);
        f
    }

    #[test]
    fn alu_ops_match_concrete() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let model = Model::new();
        for idx in 0..8usize {
            for _ in 0..3000 {
                let cin = rng.next() & 1 == 1;
                let (a, b) = (rng.next(), rng.next());

                // 8-bit
                let mut cpu = Cpu::new();
                cpu.regs.eflags = seed_flags(&mut rng, cin);
                let prior = concrete_flags(cpu.regs.eflags);
                let cres = concrete_alu8(&mut cpu, idx, a as u8, b as u8) as u64;
                let cflags = concrete_flags(cpu.regs.eflags);
                let (r, d) = alu(
                    idx,
                    &Expr::constant(8, a & 0xFF),
                    &Expr::constant(8, b & 0xFF),
                    &BoolExpr::constant(cin),
                    8,
                );
                assert_eq!(r.eval(&model), cres, "op {idx} w8 a={a:x} b={b:x}");
                assert_eq!(sym_flags(&d, prior, &model), cflags, "flags op {idx} w8 a={a:x} b={b:x}");

                // 16-bit
                let mut cpu = Cpu::new();
                cpu.regs.eflags = seed_flags(&mut rng, cin);
                let prior = concrete_flags(cpu.regs.eflags);
                let cres = concrete_alu16(&mut cpu, idx, a as u16, b as u16) as u64;
                let cflags = concrete_flags(cpu.regs.eflags);
                let (r, d) = alu(
                    idx,
                    &Expr::constant(16, a & 0xFFFF),
                    &Expr::constant(16, b & 0xFFFF),
                    &BoolExpr::constant(cin),
                    16,
                );
                assert_eq!(r.eval(&model), cres, "op {idx} w16");
                assert_eq!(sym_flags(&d, prior, &model), cflags, "flags op {idx} w16");

                // 32-bit
                let mut cpu = Cpu::new();
                cpu.regs.eflags = seed_flags(&mut rng, cin);
                let prior = concrete_flags(cpu.regs.eflags);
                let cres = concrete_alu32(&mut cpu, idx, a as u32, b as u32) as u64;
                let cflags = concrete_flags(cpu.regs.eflags);
                let (r, d) = alu(
                    idx,
                    &Expr::constant(32, a & 0xFFFF_FFFF),
                    &Expr::constant(32, b & 0xFFFF_FFFF),
                    &BoolExpr::constant(cin),
                    32,
                );
                assert_eq!(r.eval(&model), cres, "op {idx} w32");
                assert_eq!(sym_flags(&d, prior, &model), cflags, "flags op {idx} w32");
            }
        }
    }

    #[test]
    fn inc_dec_neg_match_concrete() {
        let mut rng = Rng(0xdead_beef_0bad_f00d);
        let model = Model::new();
        for _ in 0..5000 {
            let a = rng.next();
            for op in 0..3 {
                // 8-bit
                let mut cpu = Cpu::new();
                cpu.regs.eflags = EFlags::from_bits_truncate(rng.next() as u32);
                let prior = concrete_flags(cpu.regs.eflags);
                let (cres, d) = match op {
                    0 => (cpu.inc8(a as u8) as u64, inc(&Expr::constant(8, a & 0xFF), 8)),
                    1 => (cpu.dec8(a as u8) as u64, dec(&Expr::constant(8, a & 0xFF), 8)),
                    _ => (cpu.neg8(a as u8) as u64, neg(&Expr::constant(8, a & 0xFF), 8)),
                };
                let cflags = concrete_flags(cpu.regs.eflags);
                assert_eq!(d.0.eval(&model), cres, "unary {op} w8 a={a:x}");
                assert_eq!(sym_flags(&d.1, prior, &model), cflags, "unary {op} flags w8 a={a:x}");

                // 32-bit
                let mut cpu = Cpu::new();
                cpu.regs.eflags = EFlags::from_bits_truncate(rng.next() as u32);
                let prior = concrete_flags(cpu.regs.eflags);
                let (cres, d) = match op {
                    0 => (cpu.inc32(a as u32) as u64, inc(&Expr::constant(32, a & 0xFFFF_FFFF), 32)),
                    1 => (cpu.dec32(a as u32) as u64, dec(&Expr::constant(32, a & 0xFFFF_FFFF), 32)),
                    _ => (cpu.neg32(a as u32) as u64, neg(&Expr::constant(32, a & 0xFFFF_FFFF), 32)),
                };
                let cflags = concrete_flags(cpu.regs.eflags);
                assert_eq!(d.0.eval(&model), cres, "unary {op} w32");
                assert_eq!(sym_flags(&d.1, prior, &model), cflags, "unary {op} flags w32");
            }
        }
    }

    #[test]
    fn shifts_match_concrete() {
        let mut rng = Rng(0x0102_0304_0506_0708);
        let model = Model::new();
        let kinds = [
            ShiftKind::Shl,
            ShiftKind::Shr,
            ShiftKind::Sar,
            ShiftKind::Rol,
            ShiftKind::Ror,
        ];
        for &k in &kinds {
            for _ in 0..4000 {
                let v = rng.next();
                let count = (rng.next() % 40) as u32; // exercise masking and n > width

                // 8-bit
                let mut cpu = Cpu::new();
                cpu.regs.eflags = EFlags::from_bits_truncate(rng.next() as u32);
                let prior = concrete_flags(cpu.regs.eflags);
                let cres = concrete_shift8(&mut cpu, k, v as u8, count) as u64;
                let cflags = concrete_flags(cpu.regs.eflags);
                let (r, d) = shift(k, &Expr::constant(8, v & 0xFF), count, 8);
                assert_eq!(r.eval(&model), cres, "shift {k:?} w8 v={v:x} n={count}");
                assert_eq!(sym_flags(&d, prior, &model), cflags, "shift {k:?} flags w8 v={v:x} n={count}");

                // 16-bit
                let mut cpu = Cpu::new();
                cpu.regs.eflags = EFlags::from_bits_truncate(rng.next() as u32);
                let prior = concrete_flags(cpu.regs.eflags);
                let cres = concrete_shift16(&mut cpu, k, v as u16, count) as u64;
                let cflags = concrete_flags(cpu.regs.eflags);
                let (r, d) = shift(k, &Expr::constant(16, v & 0xFFFF), count, 16);
                assert_eq!(r.eval(&model), cres, "shift {k:?} w16 v={v:x} n={count}");
                assert_eq!(sym_flags(&d, prior, &model), cflags, "shift {k:?} flags w16 v={v:x} n={count}");

                // 32-bit
                let mut cpu = Cpu::new();
                cpu.regs.eflags = EFlags::from_bits_truncate(rng.next() as u32);
                let prior = concrete_flags(cpu.regs.eflags);
                let cres = concrete_shift32(&mut cpu, k, v as u32, count) as u64;
                let cflags = concrete_flags(cpu.regs.eflags);
                let (r, d) = shift(k, &Expr::constant(32, v & 0xFFFF_FFFF), count, 32);
                assert_eq!(r.eval(&model), cres, "shift {k:?} w32 v={v:x} n={count}");
                assert_eq!(sym_flags(&d, prior, &model), cflags, "shift {k:?} flags w32 v={v:x} n={count}");
            }
        }
    }

    fn concrete_shift8(cpu: &mut Cpu, k: ShiftKind, v: u8, n: u32) -> u8 {
        match k {
            ShiftKind::Shl => cpu.shl8(v, n),
            ShiftKind::Shr => cpu.shr8(v, n),
            ShiftKind::Sar => cpu.sar8(v, n),
            ShiftKind::Rol => cpu.rol8(v, n),
            ShiftKind::Ror => cpu.ror8(v, n),
        }
    }
    fn concrete_shift16(cpu: &mut Cpu, k: ShiftKind, v: u16, n: u32) -> u16 {
        match k {
            ShiftKind::Shl => cpu.shl16(v, n),
            ShiftKind::Shr => cpu.shr16(v, n),
            ShiftKind::Sar => cpu.sar16(v, n),
            ShiftKind::Rol => cpu.rol16(v, n),
            ShiftKind::Ror => cpu.ror16(v, n),
        }
    }
    fn concrete_shift32(cpu: &mut Cpu, k: ShiftKind, v: u32, n: u32) -> u32 {
        match k {
            ShiftKind::Shl => cpu.shl32(v, n),
            ShiftKind::Shr => cpu.shr32(v, n),
            ShiftKind::Sar => cpu.sar32(v, n),
            ShiftKind::Rol => cpu.rol32(v, n),
            ShiftKind::Ror => cpu.ror32(v, n),
        }
    }
}
