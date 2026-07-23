//! Symbolic (concolic) execution overlay for the 80386 core.
//!
//! The concrete interpreter stays the single source of truth and stays fast; a
//! sparse symbolic *shadow* rides alongside it, tracking bitvector expressions
//! only for the registers, flags and memory bytes derived from a symbolic
//! input. Control flow is always driven by the concrete run (so `CS:EIP`, the
//! icache and paging never see symbolic data); path constraints are collected
//! at branches whose deciding flags are symbolic, and a solver negates them to
//! discover inputs that reach new paths.
//!
//! The whole module is gated behind the `symbolic` feature, so a default build
//! is byte-for-byte the current core.
//!
//! ## Milestones
//! - M0: [`expr`] — the bitvector AST — and [`alu`] — the symbolic ALU,
//!   differentially tested against the concrete 386 ALU.
//! - M1: the sparse [`state`] overlay ([`SymEngine`]) plus the instrumentation
//!   seams in the 386 core (see `x86_32::symbolic`), collecting path
//!   constraints.
//! - **M2 (this):** [`smtlib`] export of the constraints (dependency-free) and,
//!   behind `symbolic-solver`, a [`solver`] that drives z3/cvc5 to solve for an
//!   input that flips a branch.

pub mod alu;
pub mod expr;
pub mod smtlib;
#[cfg(feature = "symbolic-solver")]
pub mod solver;
pub mod state;

pub use expr::{BoolExpr, Expr, Model, SymId, Width};
pub use state::{Place, SymEngine};
#[cfg(feature = "symbolic-solver")]
pub use solver::Solver;
