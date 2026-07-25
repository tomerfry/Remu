//! Bitvector expression AST — SMT-LIB QF_BV, in Rust.
//!
//! Every value is a width-typed bitvector `Expr`; boolean facts (flags, path
//! constraints, comparisons) are a separate `BoolExpr`. Both are reference
//! counted so cloning a node is cheap and structurally-shared subtrees cost
//! nothing extra. The handles are `Arc`, not `Rc`, so that a `Cpu` carrying an
//! overlay stays `Send + Sync` — `#[pyclass]` requires it, and the atomics are
//! noise next to the solver.
//!
//! Constructors **constant-fold on construction**: if every operand is a
//! `Const`, the node collapses to a single `Const` immediately. This is what
//! keeps the concolic overlay sparse — any subtree that touches no symbolic
//! input reduces to a plain integer, so tainting one byte never symbolizes the
//! rest of the machine.
//!
//! (Hash-consing of symbolic nodes is a later optimization; correctness does
//! not need it and M0 exercises only constant leaves.)

use std::collections::HashMap;
use std::sync::Arc;

/// Width of a bitvector, in bits. The 386 uses 1/8/16/32; wider temporaries
/// (9/17/33/64) appear when a construction needs headroom, e.g. the carry-out
/// bit of an `ADC`.
pub type Width = u16;

/// A symbolic input variable, identified by a small integer. The engine keeps
/// the mapping from `SymId` to a name and seed value in the symbolic state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymId(pub u32);

/// Binary bitvector operators. Both operands and the result share one width
/// (the shift amount is also `width`-wide, matching SMT-LIB's `bvshl` family).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    /// Logical left shift (`bvshl`): shifting by ≥ width yields 0.
    Shl,
    /// Logical right shift (`bvlshr`).
    Lshr,
    /// Arithmetic right shift (`bvashr`).
    Ashr,
    Udiv,
    Sdiv,
    Urem,
    Srem,
}

/// Unary bitvector operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    /// Two's-complement negation.
    Neg,
    /// Bitwise complement.
    Not,
}

/// Comparison operators producing a `BoolExpr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
}

/// The shape of a bitvector node.
#[derive(Debug)]
pub enum Kind {
    /// A concrete value, already masked to the node's width.
    Const(u64),
    /// A symbolic input.
    Symbol(SymId),
    Bin(BinOp, Expr, Expr),
    Un(UnOp, Expr),
    /// Bit slice `[hi:lo]` inclusive; result width is `hi - lo + 1`.
    Extract { hi: Width, lo: Width, e: Expr },
    /// Concatenation with the first operand as the high part.
    Concat(Expr, Expr),
    /// Zero- or sign-extension to a wider width.
    Extend { signed: bool, to: Width, e: Expr },
    /// If-then-else selecting between two equal-width values.
    Ite(BoolExpr, Expr, Expr),
}

/// An interned bitvector node: its width plus its shape.
#[derive(Debug)]
pub struct ExprNode {
    pub width: Width,
    pub kind: Kind,
}

/// A reference-counted handle to a bitvector expression. Cloning is cheap.
#[derive(Debug, Clone)]
pub struct Expr(Arc<ExprNode>);

/// The shape of a boolean node.
#[derive(Debug)]
pub enum BoolKind {
    Const(bool),
    Cmp(CmpOp, Expr, Expr),
    /// Extract a single bit `i` of a bitvector as a boolean.
    BitOf(Expr, Width),
    And(BoolExpr, BoolExpr),
    Or(BoolExpr, BoolExpr),
    Xor(BoolExpr, BoolExpr),
    Not(BoolExpr),
}

/// A reference-counted handle to a boolean expression.
#[derive(Debug, Clone)]
pub struct BoolExpr(Arc<BoolKind>);

/// An assignment of concrete values to symbolic inputs, used by [`Expr::eval`]
/// / [`BoolExpr::eval`] to evaluate an expression to a concrete result.
pub type Model = HashMap<SymId, u64>;

/// The all-ones mask for a `width`-bit value.
#[inline]
pub(crate) fn mask(width: Width) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Interpret the low `width` bits of `val` as a two's-complement signed number,
/// sign-extended into a full `i64`.
#[inline]
pub(crate) fn sign_extend(val: u64, width: Width) -> i64 {
    if width == 0 || width >= 64 {
        return val as i64;
    }
    let shift = 64 - width;
    ((val << shift) as i64) >> shift
}

/// Concrete evaluation of a binary op on `width`-bit values, result masked.
fn eval_bin(op: BinOp, a: u64, b: u64, width: Width) -> u64 {
    let m = mask(width);
    let (a, b) = (a & m, b & m);
    let r = match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => {
            if b >= width as u64 {
                0
            } else {
                a << b
            }
        }
        BinOp::Lshr => {
            if b >= width as u64 {
                0
            } else {
                a >> b
            }
        }
        BinOp::Ashr => {
            let s = sign_extend(a, width);
            if b >= width as u64 {
                (s >> 63) as u64
            } else {
                (s >> b) as u64
            }
        }
        // Division by zero follows SMT-LIB: bvudiv → all ones, bvurem → a.
        BinOp::Udiv => {
            if b == 0 {
                m
            } else {
                a / b
            }
        }
        BinOp::Urem => {
            if b == 0 {
                a
            } else {
                a % b
            }
        }
        BinOp::Sdiv => {
            let (x, y) = (sign_extend(a, width), sign_extend(b, width));
            if y == 0 {
                if x < 0 { 1 } else { m }
            } else {
                x.wrapping_div(y) as u64
            }
        }
        BinOp::Srem => {
            let (x, y) = (sign_extend(a, width), sign_extend(b, width));
            if y == 0 {
                a
            } else {
                x.wrapping_rem(y) as u64
            }
        }
    };
    r & m
}

/// Concrete evaluation of a comparison on `width`-bit values.
fn eval_cmp(op: CmpOp, a: u64, b: u64, width: Width) -> bool {
    let m = mask(width);
    let (au, bu) = (a & m, b & m);
    let (as_, bs) = (sign_extend(a, width), sign_extend(b, width));
    match op {
        CmpOp::Eq => au == bu,
        CmpOp::Ne => au != bu,
        CmpOp::Ult => au < bu,
        CmpOp::Ule => au <= bu,
        CmpOp::Ugt => au > bu,
        CmpOp::Uge => au >= bu,
        CmpOp::Slt => as_ < bs,
        CmpOp::Sle => as_ <= bs,
        CmpOp::Sgt => as_ > bs,
        CmpOp::Sge => as_ >= bs,
    }
}

impl Expr {
    fn node(width: Width, kind: Kind) -> Expr {
        Expr(Arc::new(ExprNode { width, kind }))
    }

    /// A concrete constant of the given width (`val` is masked to `width`).
    pub fn constant(width: Width, val: u64) -> Expr {
        Expr::node(width, Kind::Const(val & mask(width)))
    }

    /// A symbolic input variable.
    pub fn symbol(id: SymId, width: Width) -> Expr {
        Expr::node(width, Kind::Symbol(id))
    }

    #[inline]
    pub fn width(&self) -> Width {
        self.0.width
    }

    #[inline]
    pub fn kind(&self) -> &Kind {
        &self.0.kind
    }

    /// The constant value, if this node is a `Const`.
    #[inline]
    pub fn as_const(&self) -> Option<u64> {
        match self.0.kind {
            Kind::Const(v) => Some(v),
            _ => None,
        }
    }

    /// Build a binary node, folding when both operands are constant and
    /// applying a few cheap algebraic identities.
    pub fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
        debug_assert_eq!(a.width(), b.width(), "bin operand width mismatch");
        let w = a.width();
        if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
            return Expr::constant(w, eval_bin(op, x, y, w));
        }
        // Identities that keep flag expressions from ballooning.
        match op {
            BinOp::Add | BinOp::Or | BinOp::Xor | BinOp::Sub if b.as_const() == Some(0) => {
                return a;
            }
            BinOp::Shl | BinOp::Lshr | BinOp::Ashr if b.as_const() == Some(0) => return a,
            BinOp::And if b.as_const() == Some(mask(w)) => return a,
            BinOp::And if b.as_const() == Some(0) => return Expr::constant(w, 0),
            BinOp::Or if b.as_const() == Some(mask(w)) => return Expr::constant(w, mask(w)),
            BinOp::Mul if b.as_const() == Some(1) => return a,
            BinOp::Xor | BinOp::Sub if Expr::ptr_eq(&a, &b) => return Expr::constant(w, 0),
            _ => {}
        }
        Expr::node(w, Kind::Bin(op, a, b))
    }

    /// Build a unary node, folding constants.
    pub fn un(op: UnOp, e: Expr) -> Expr {
        let w = e.width();
        if let Some(v) = e.as_const() {
            let r = match op {
                UnOp::Neg => 0u64.wrapping_sub(v),
                UnOp::Not => !v,
            };
            return Expr::constant(w, r);
        }
        Expr::node(w, Kind::Un(op, e))
    }

    /// Bit slice `[hi:lo]` inclusive.
    pub fn extract(hi: Width, lo: Width, e: Expr) -> Expr {
        debug_assert!(hi >= lo && hi < e.width(), "extract out of range");
        let w = hi - lo + 1;
        if lo == 0 && w == e.width() {
            return e;
        }
        if let Some(v) = e.as_const() {
            return Expr::constant(w, (v >> lo) & mask(w));
        }
        Expr::node(w, Kind::Extract { hi, lo, e })
    }

    /// Concatenate with `hi` as the high-order part.
    pub fn concat(hi: Expr, lo: Expr) -> Expr {
        let w = hi.width() + lo.width();
        if let (Some(h), Some(l)) = (hi.as_const(), lo.as_const()) {
            return Expr::constant(w, (h << lo.width()) | l);
        }
        Expr::node(w, Kind::Concat(hi, lo))
    }

    /// Zero- or sign-extend to width `to` (`to >= self.width()`).
    pub fn extend(signed: bool, to: Width, e: Expr) -> Expr {
        debug_assert!(to >= e.width(), "extend narrows");
        if to == e.width() {
            return e;
        }
        if let Some(v) = e.as_const() {
            let val = if signed {
                sign_extend(v, e.width()) as u64 & mask(to)
            } else {
                v & mask(e.width())
            };
            return Expr::constant(to, val);
        }
        Expr::node(to, Kind::Extend { signed, to, e })
    }

    /// Zero-extend convenience.
    pub fn zext(to: Width, e: Expr) -> Expr {
        Expr::extend(false, to, e)
    }

    /// Sign-extend convenience.
    pub fn sext(to: Width, e: Expr) -> Expr {
        Expr::extend(true, to, e)
    }

    /// If-then-else, folding when the condition is a known boolean.
    pub fn ite(c: BoolExpr, t: Expr, e: Expr) -> Expr {
        debug_assert_eq!(t.width(), e.width(), "ite branch width mismatch");
        if let Some(b) = c.as_const() {
            return if b { t } else { e };
        }
        Expr::node(t.width(), Kind::Ite(c, t, e))
    }

    fn ptr_eq(a: &Expr, b: &Expr) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    /// Evaluate to a concrete value under `model` (unbound symbols read as 0).
    pub fn eval(&self, model: &Model) -> u64 {
        match self.kind() {
            Kind::Const(v) => *v,
            Kind::Symbol(id) => model.get(id).copied().unwrap_or(0) & mask(self.width()),
            Kind::Bin(op, a, b) => eval_bin(*op, a.eval(model), b.eval(model), self.width()),
            Kind::Un(op, e) => {
                let v = e.eval(model);
                let r = match op {
                    UnOp::Neg => 0u64.wrapping_sub(v),
                    UnOp::Not => !v,
                };
                r & mask(self.width())
            }
            Kind::Extract { hi: _, lo, e } => (e.eval(model) >> lo) & mask(self.width()),
            Kind::Concat(h, l) => (h.eval(model) << l.width()) | l.eval(model),
            Kind::Extend { signed, to, e } => {
                let v = e.eval(model);
                if *signed {
                    sign_extend(v, e.width()) as u64 & mask(*to)
                } else {
                    v & mask(e.width())
                }
            }
            Kind::Ite(c, t, e) => {
                if c.eval(model) {
                    t.eval(model)
                } else {
                    e.eval(model)
                }
            }
        }
    }
}

impl BoolExpr {
    fn node(kind: BoolKind) -> BoolExpr {
        BoolExpr(Arc::new(kind))
    }

    pub fn constant(b: bool) -> BoolExpr {
        BoolExpr::node(BoolKind::Const(b))
    }

    #[inline]
    pub fn as_const(&self) -> Option<bool> {
        match *self.0 {
            BoolKind::Const(b) => Some(b),
            _ => None,
        }
    }

    #[inline]
    pub fn kind(&self) -> &BoolKind {
        &self.0
    }

    /// A comparison, folding when both operands are constant.
    pub fn cmp(op: CmpOp, a: Expr, b: Expr) -> BoolExpr {
        debug_assert_eq!(a.width(), b.width(), "cmp operand width mismatch");
        if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
            return BoolExpr::constant(eval_cmp(op, x, y, a.width()));
        }
        BoolExpr::node(BoolKind::Cmp(op, a, b))
    }

    /// Extract bit `i` of a bitvector as a boolean.
    pub fn bit_of(e: Expr, i: Width) -> BoolExpr {
        if let Some(v) = e.as_const() {
            return BoolExpr::constant((v >> i) & 1 == 1);
        }
        BoolExpr::node(BoolKind::BitOf(e, i))
    }

    pub fn and(a: BoolExpr, b: BoolExpr) -> BoolExpr {
        match (a.as_const(), b.as_const()) {
            (Some(false), _) | (_, Some(false)) => BoolExpr::constant(false),
            (Some(true), _) => b,
            (_, Some(true)) => a,
            _ => BoolExpr::node(BoolKind::And(a, b)),
        }
    }

    pub fn or(a: BoolExpr, b: BoolExpr) -> BoolExpr {
        match (a.as_const(), b.as_const()) {
            (Some(true), _) | (_, Some(true)) => BoolExpr::constant(true),
            (Some(false), _) => b,
            (_, Some(false)) => a,
            _ => BoolExpr::node(BoolKind::Or(a, b)),
        }
    }

    pub fn xor(a: BoolExpr, b: BoolExpr) -> BoolExpr {
        match (a.as_const(), b.as_const()) {
            (Some(x), Some(y)) => BoolExpr::constant(x ^ y),
            (Some(false), _) => b,
            (_, Some(false)) => a,
            (Some(true), _) => BoolExpr::not(b),
            (_, Some(true)) => BoolExpr::not(a),
            _ => BoolExpr::node(BoolKind::Xor(a, b)),
        }
    }

    // Named `not` to match the QF_BV vocabulary (`and`/`or`/`xor`/`not`); it is
    // an associated constructor, not the `std::ops::Not` operator.
    #[allow(clippy::should_implement_trait)]
    pub fn not(a: BoolExpr) -> BoolExpr {
        match a.as_const() {
            Some(b) => BoolExpr::constant(!b),
            None => BoolExpr::node(BoolKind::Not(a)),
        }
    }

    /// Evaluate to a concrete boolean under `model`.
    pub fn eval(&self, model: &Model) -> bool {
        match &*self.0 {
            BoolKind::Const(b) => *b,
            BoolKind::Cmp(op, a, b) => eval_cmp(*op, a.eval(model), b.eval(model), a.width()),
            BoolKind::BitOf(e, i) => (e.eval(model) >> i) & 1 == 1,
            BoolKind::And(a, b) => a.eval(model) && b.eval(model),
            BoolKind::Or(a, b) => a.eval(model) || b.eval(model),
            BoolKind::Xor(a, b) => a.eval(model) ^ b.eval(model),
            BoolKind::Not(a) => !a.eval(model),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_folds_to_single_node() {
        let a = Expr::constant(32, 5);
        let b = Expr::constant(32, 7);
        let r = Expr::bin(BinOp::Add, a, b);
        assert_eq!(r.as_const(), Some(12));
    }

    #[test]
    fn symbolic_add_does_not_fold_but_evals() {
        let x = Expr::symbol(SymId(0), 8);
        let r = Expr::bin(BinOp::Add, x, Expr::constant(8, 3));
        assert!(r.as_const().is_none());
        let mut model = Model::new();
        model.insert(SymId(0), 0xFE);
        assert_eq!(r.eval(&model), 0x01); // (0xFE + 3) & 0xFF
    }

    #[test]
    fn extract_concat_roundtrip() {
        let x = Expr::symbol(SymId(0), 32);
        let lo = Expr::extract(15, 0, x.clone());
        let hi = Expr::extract(31, 16, x.clone());
        let joined = Expr::concat(hi, lo);
        let mut model = Model::new();
        model.insert(SymId(0), 0xDEAD_BEEF);
        assert_eq!(joined.eval(&model), 0xDEAD_BEEF);
    }

    #[test]
    fn signed_shift_and_extend() {
        // -2 as i8 arithmetic-shifted right by 1 is -1.
        let v = Expr::constant(8, 0xFE);
        let r = Expr::bin(BinOp::Ashr, v, Expr::constant(8, 1));
        assert_eq!(r.as_const(), Some(0xFF));
        // sign-extend 0xFF (i8 = -1) to 32 bits.
        let s = Expr::sext(32, Expr::constant(8, 0xFF));
        assert_eq!(s.as_const(), Some(0xFFFF_FFFF));
    }
}
