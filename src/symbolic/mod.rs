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
//! - **M0 (this):** [`expr`] — the bitvector AST — and [`alu`] — the symbolic
//!   ALU, differentially tested against the concrete 386 ALU. Standalone; the
//!   core is untouched.
//! - M1: sparse [`state`] overlay + instrumentation seams + constraint
//!   collection (no solver).
//! - M2: SMT-LIB export + `easy-smt` solver + negation-driven exploration.

pub mod alu;
pub mod expr;
