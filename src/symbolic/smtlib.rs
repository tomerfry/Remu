//! SMT-LIB 2 (QF_BV) serialization of the bitvector AST.
//!
//! Dependency-free: turns [`Expr`]/[`BoolExpr`] and a set of path constraints
//! into a solver-agnostic script. A user with no solver wired in can still take
//! this text and run it through z3/cvc5/bitwuzla by hand; the `symbolic-solver`
//! feature drives a solver with it automatically.
//!
//! Input symbols are named `x!<id>`.
//!
//! ## Why this is a DAG walk, not a tree walk
//!
//! The AST is a DAG: flag expressions reference their operands, and an `ADC`
//! chain feeds each carry-out back into the next carry-in, so one node is
//! commonly reachable by many paths. Printing it as a tree duplicates every
//! shared subterm, which is exponential in the chain length — a measured
//! ~4.2x per chained `ADC`, so twelve of them reached a gigabyte of text. A
//! recursive walk also overflowed the stack (and killed the process, with no
//! catchable error) at a few thousand chained operations.
//!
//! So each distinct node is emitted exactly once as a `define-fun` binding and
//! referred to afterwards by name, and the traversal uses an explicit stack.
//! Output is then linear in the number of distinct nodes and the depth of the
//! expression is irrelevant. Constants and symbols stay inline — naming them
//! would cost more than it saves.

use std::collections::HashMap;

use super::expr::{BinOp, BoolExpr, BoolKind, CmpOp, Expr, Kind, UnOp};

/// Accumulates `define-fun` bindings for a set of formulas, sharing every
/// repeated subterm across all of them.
#[derive(Debug, Default)]
pub struct Smt {
    defs: String,
    expr_ids: HashMap<usize, u32>,
    bool_ids: HashMap<usize, u32>,
    next: u32,
}

/// One entry of the explicit traversal stack.
enum Node {
    E(Expr),
    B(BoolExpr),
}

impl Smt {
    pub fn new() -> Self {
        Smt::default()
    }

    /// The `define-fun` block for everything defined so far. Emit it before the
    /// assertions that reference it.
    pub fn defs(&self) -> &str {
        &self.defs
    }

    /// Define `b` and all of its subterms, and return the token that refers to
    /// it (a binding name, or an inline literal for a constant).
    pub fn bool_term(&mut self, b: &BoolExpr) -> String {
        self.define_all(Node::B(b.clone()));
        self.bool_ref(b)
    }

    /// Post-order over the DAG with an explicit stack, defining each node once.
    fn define_all(&mut self, root: Node) {
        let mut stack = vec![(root, false)];
        while let Some((n, expanded)) = stack.pop() {
            if expanded {
                self.define(&n);
                continue;
            }
            if self.is_defined(&n) {
                continue;
            }
            let children = children_of(&n);
            stack.push((n, true));
            for c in children {
                if !self.is_defined(&c) {
                    stack.push((c, false));
                }
            }
        }
    }

    /// Whether this node needs no binding — already defined, or an inline leaf.
    fn is_defined(&self, n: &Node) -> bool {
        match n {
            Node::E(e) => is_inline_expr(e) || self.expr_ids.contains_key(&e.ptr_key()),
            Node::B(b) => b.as_const().is_some() || self.bool_ids.contains_key(&b.ptr_key()),
        }
    }

    /// Emit the binding for a node whose children are all defined already.
    fn define(&mut self, n: &Node) {
        if self.is_defined(n) {
            return; // reached twice via a diamond
        }
        let id = self.next;
        self.next += 1;
        match n {
            Node::E(e) => {
                let body = self.expr_body(e);
                self.defs.push_str(&format!(
                    "(define-fun e{} () (_ BitVec {}) {})\n",
                    id,
                    e.width(),
                    body
                ));
                self.expr_ids.insert(e.ptr_key(), id);
            }
            Node::B(b) => {
                let body = self.bool_body(b);
                self.defs
                    .push_str(&format!("(define-fun b{id} () Bool {body})\n"));
                self.bool_ids.insert(b.ptr_key(), id);
            }
        }
    }

    /// How to refer to `e`: an inline literal, or its binding name.
    fn expr_ref(&self, e: &Expr) -> String {
        match e.kind() {
            Kind::Const(v) => format!("(_ bv{} {})", v, e.width()),
            Kind::Symbol(id) => format!("x!{}", id.0),
            _ => format!("e{}", self.expr_ids[&e.ptr_key()]),
        }
    }

    fn bool_ref(&self, b: &BoolExpr) -> String {
        match b.as_const() {
            Some(true) => "true".to_string(),
            Some(false) => "false".to_string(),
            None => format!("b{}", self.bool_ids[&b.ptr_key()]),
        }
    }

    fn expr_body(&self, e: &Expr) -> String {
        match e.kind() {
            // Inline leaves never get a binding, so they never get here.
            Kind::Const(_) | Kind::Symbol(_) => self.expr_ref(e),
            Kind::Bin(op, a, b) => {
                let name = match op {
                    BinOp::Add => "bvadd",
                    BinOp::Sub => "bvsub",
                    BinOp::Mul => "bvmul",
                    BinOp::And => "bvand",
                    BinOp::Or => "bvor",
                    BinOp::Xor => "bvxor",
                    BinOp::Shl => "bvshl",
                    BinOp::Lshr => "bvlshr",
                    BinOp::Ashr => "bvashr",
                    BinOp::Udiv => "bvudiv",
                    BinOp::Sdiv => "bvsdiv",
                    BinOp::Urem => "bvurem",
                    BinOp::Srem => "bvsrem",
                };
                format!("({} {} {})", name, self.expr_ref(a), self.expr_ref(b))
            }
            Kind::Un(op, a) => {
                let name = match op {
                    UnOp::Neg => "bvneg",
                    UnOp::Not => "bvnot",
                };
                format!("({} {})", name, self.expr_ref(a))
            }
            Kind::Extract { hi, lo, e } => {
                format!("((_ extract {} {}) {})", hi, lo, self.expr_ref(e))
            }
            Kind::Concat(h, l) => {
                format!("(concat {} {})", self.expr_ref(h), self.expr_ref(l))
            }
            Kind::Extend { signed, to, e } => {
                let kw = if *signed { "sign_extend" } else { "zero_extend" };
                format!("((_ {} {}) {})", kw, to - e.width(), self.expr_ref(e))
            }
            Kind::Ite(c, t, f) => format!(
                "(ite {} {} {})",
                self.bool_ref(c),
                self.expr_ref(t),
                self.expr_ref(f)
            ),
        }
    }

    fn bool_body(&self, b: &BoolExpr) -> String {
        match b.kind() {
            BoolKind::Const(_) => self.bool_ref(b),
            BoolKind::Cmp(op, a, b) => {
                let (x, y) = (self.expr_ref(a), self.expr_ref(b));
                match op {
                    CmpOp::Eq => format!("(= {x} {y})"),
                    CmpOp::Ne => format!("(not (= {x} {y}))"),
                    CmpOp::Ult => format!("(bvult {x} {y})"),
                    CmpOp::Ule => format!("(bvule {x} {y})"),
                    CmpOp::Ugt => format!("(bvugt {x} {y})"),
                    CmpOp::Uge => format!("(bvuge {x} {y})"),
                    CmpOp::Slt => format!("(bvslt {x} {y})"),
                    CmpOp::Sle => format!("(bvsle {x} {y})"),
                    CmpOp::Sgt => format!("(bvsgt {x} {y})"),
                    CmpOp::Sge => format!("(bvsge {x} {y})"),
                }
            }
            BoolKind::BitOf(e, i) => {
                format!("(= ((_ extract {} {}) {}) #b1)", i, i, self.expr_ref(e))
            }
            BoolKind::And(x, y) => format!("(and {} {})", self.bool_ref(x), self.bool_ref(y)),
            BoolKind::Or(x, y) => format!("(or {} {})", self.bool_ref(x), self.bool_ref(y)),
            BoolKind::Xor(x, y) => format!("(xor {} {})", self.bool_ref(x), self.bool_ref(y)),
            BoolKind::Not(x) => format!("(not {})", self.bool_ref(x)),
        }
    }
}

/// Constants and symbols print inline rather than getting a binding.
fn is_inline_expr(e: &Expr) -> bool {
    matches!(e.kind(), Kind::Const(_) | Kind::Symbol(_))
}

fn children_of(n: &Node) -> Vec<Node> {
    match n {
        Node::E(e) => match e.kind() {
            Kind::Const(_) | Kind::Symbol(_) => vec![],
            Kind::Bin(_, a, b) => vec![Node::E(a.clone()), Node::E(b.clone())],
            Kind::Un(_, a) => vec![Node::E(a.clone())],
            Kind::Extract { e, .. } => vec![Node::E(e.clone())],
            Kind::Concat(h, l) => vec![Node::E(h.clone()), Node::E(l.clone())],
            Kind::Extend { e, .. } => vec![Node::E(e.clone())],
            Kind::Ite(c, t, f) => {
                vec![Node::B(c.clone()), Node::E(t.clone()), Node::E(f.clone())]
            }
        },
        Node::B(b) => match b.kind() {
            BoolKind::Const(_) => vec![],
            BoolKind::Cmp(_, a, b) => vec![Node::E(a.clone()), Node::E(b.clone())],
            BoolKind::BitOf(e, _) => vec![Node::E(e.clone())],
            BoolKind::And(x, y) | BoolKind::Or(x, y) | BoolKind::Xor(x, y) => {
                vec![Node::B(x.clone()), Node::B(y.clone())]
            }
            BoolKind::Not(x) => vec![Node::B(x.clone())],
        },
    }
}
