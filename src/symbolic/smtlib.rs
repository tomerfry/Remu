//! SMT-LIB 2 (QF_BV) serialization of the bitvector AST.
//!
//! Dependency-free: turns [`Expr`]/[`BoolExpr`] and a set of path constraints
//! into a solver-agnostic script. A user with no solver wired in can still take
//! this text and run it through z3/cvc5/bitwuzla by hand; the `symbolic-solver`
//! feature drives a solver with it automatically.
//!
//! Input symbols are named `x!<id>`. Emission is a straight recursive walk;
//! shared DAG nodes are re-emitted (fine for the modest constraints concolic
//! execution produces — a `let`/`define-fun` sharing pass is a future
//! optimization).

use super::expr::{BinOp, BoolExpr, BoolKind, CmpOp, Expr, Kind, UnOp};

/// Append the SMT-LIB term for a bitvector expression.
pub fn write_expr(e: &Expr, out: &mut String) {
    match e.kind() {
        Kind::Const(v) => {
            out.push_str("(_ bv");
            out.push_str(&v.to_string());
            out.push(' ');
            out.push_str(&e.width().to_string());
            out.push(')');
        }
        Kind::Symbol(id) => {
            out.push_str("x!");
            out.push_str(&id.0.to_string());
        }
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
            write_list2(name, a, b, out);
        }
        Kind::Un(op, a) => {
            let name = match op {
                UnOp::Neg => "bvneg",
                UnOp::Not => "bvnot",
            };
            out.push('(');
            out.push_str(name);
            out.push(' ');
            write_expr(a, out);
            out.push(')');
        }
        Kind::Extract { hi, lo, e } => {
            out.push_str("((_ extract ");
            out.push_str(&hi.to_string());
            out.push(' ');
            out.push_str(&lo.to_string());
            out.push_str(") ");
            write_expr(e, out);
            out.push(')');
        }
        Kind::Concat(h, l) => write_list2("concat", h, l, out),
        Kind::Extend { signed, to, e } => {
            let by = to - e.width();
            out.push_str(if *signed { "((_ sign_extend " } else { "((_ zero_extend " });
            out.push_str(&by.to_string());
            out.push_str(") ");
            write_expr(e, out);
            out.push(')');
        }
        Kind::Ite(c, t, f) => {
            out.push_str("(ite ");
            write_bool(c, out);
            out.push(' ');
            write_expr(t, out);
            out.push(' ');
            write_expr(f, out);
            out.push(')');
        }
    }
}

/// Append the SMT-LIB formula for a boolean expression.
pub fn write_bool(b: &BoolExpr, out: &mut String) {
    match b.kind() {
        BoolKind::Const(v) => out.push_str(if *v { "true" } else { "false" }),
        BoolKind::Cmp(op, a, b) => match op {
            CmpOp::Eq => write_list2("=", a, b, out),
            CmpOp::Ne => {
                out.push_str("(not ");
                write_list2("=", a, b, out);
                out.push(')');
            }
            CmpOp::Ult => write_list2("bvult", a, b, out),
            CmpOp::Ule => write_list2("bvule", a, b, out),
            CmpOp::Ugt => write_list2("bvugt", a, b, out),
            CmpOp::Uge => write_list2("bvuge", a, b, out),
            CmpOp::Slt => write_list2("bvslt", a, b, out),
            CmpOp::Sle => write_list2("bvsle", a, b, out),
            CmpOp::Sgt => write_list2("bvsgt", a, b, out),
            CmpOp::Sge => write_list2("bvsge", a, b, out),
        },
        BoolKind::BitOf(e, i) => {
            out.push_str("(= ((_ extract ");
            out.push_str(&i.to_string());
            out.push(' ');
            out.push_str(&i.to_string());
            out.push_str(") ");
            write_expr(e, out);
            out.push_str(") #b1)");
        }
        BoolKind::And(a, b) => write_bool2("and", a, b, out),
        BoolKind::Or(a, b) => write_bool2("or", a, b, out),
        BoolKind::Xor(a, b) => write_bool2("xor", a, b, out),
        BoolKind::Not(a) => {
            out.push_str("(not ");
            write_bool(a, out);
            out.push(')');
        }
    }
}

fn write_list2(name: &str, a: &Expr, b: &Expr, out: &mut String) {
    out.push('(');
    out.push_str(name);
    out.push(' ');
    write_expr(a, out);
    out.push(' ');
    write_expr(b, out);
    out.push(')');
}

fn write_bool2(name: &str, a: &BoolExpr, b: &BoolExpr, out: &mut String) {
    out.push('(');
    out.push_str(name);
    out.push(' ');
    write_bool(a, out);
    out.push(' ');
    write_bool(b, out);
    out.push(')');
}
