//! The 80386 ↔ symbolic-overlay glue (feature `symbolic`).
//!
//! The arch-neutral engine lives in [`crate::symbolic`]; this module is the
//! thin per-core binding: it resolves the core's [`Operand`] into a [`Place`]
//! (a `{slot, lo, width}` register slice or a linear address), gathers the
//! concrete values the engine needs, and forwards to the [`SymEngine`] hanging
//! off [`Cpu::sym`]. The instrumentation call sites live in `execute.rs`/
//! `mod.rs`, each `#[cfg(feature = "symbolic")]`-gated so a default build
//! carries none of this.
//!
//! While the overlay is active the core is forced onto its reference fused
//! path (see `exec_one`), so only the fused handlers here need seams.

use super::modrm::Operand;
use super::registers::EFlags;
use super::Cpu;
use crate::symbolic::{BoolExpr, Model, Place, SymEngine, SymId, UnaryOp, Width};

/// Resolve a register operand index + width into a `{slot, lo, width}` slice,
/// mapping the 386 `AL CL DL BL AH CH DH BH` 8-bit convention.
fn reg_place(idx: u8, width: Width) -> Place {
    let (slot, lo) = match width {
        8 => (idx & 3, if idx & 4 != 0 { 8 } else { 0 }),
        _ => (idx & 7, 0),
    };
    Place::reg(slot, lo, width)
}

impl Cpu {
    // --- Public API -------------------------------------------------------------

    /// Enable the concolic overlay (idempotent).
    pub fn sym_init(&mut self) {
        self.sym
            .get_or_insert_with(|| Box::new(SymEngine::new(8, 32)))
            .enable();
    }

    /// Mark a 32-bit GPR (encoding slot 0..7) symbolic, seeded with its current
    /// concrete value.
    pub fn sym_symbolize_reg32(&mut self, slot: u8, name: &str) -> SymId {
        let concrete = self.regs.gpr[(slot & 7) as usize] as u64;
        self.sym
            .as_deref_mut()
            .expect("call sym_init first")
            .symbolize_reg(slot & 7, 0, 32, name.to_string(), concrete)
    }

    /// Mark `width/8` memory bytes at linear address `lin` symbolic, seeded with
    /// `concrete` (the caller supplies the current bytes, little-endian).
    pub fn sym_symbolize_mem(&mut self, lin: u32, width: Width, name: &str, concrete: u64) -> SymId {
        self.sym
            .as_deref_mut()
            .expect("call sym_init first")
            .symbolize_mem(lin as u64, width, name.to_string(), concrete)
    }

    /// The path constraints collected so far (empty if the overlay is off).
    pub fn sym_constraints(&self) -> &[BoolExpr] {
        match self.sym.as_deref() {
            Some(e) => e.constraints(),
            None => &[],
        }
    }

    /// An SMT-LIB 2 (QF_BV) script for the current path — all constraints
    /// asserted. Solver-agnostic; run it through any SMT solver.
    pub fn sym_smtlib(&self) -> String {
        self.sym.as_deref().map(|e| e.smtlib(None)).unwrap_or_default()
    }

    /// An SMT-LIB 2 script that flips branch `i`: a satisfying model takes the
    /// *other* side of that branch (the solve-for-input query).
    pub fn sym_smtlib_flip(&self, i: usize) -> String {
        self.sym.as_deref().map(|e| e.smtlib(Some(i))).unwrap_or_default()
    }

    /// The engine, for inspection (inputs, seed, constraints).
    pub fn sym_engine(&self) -> Option<&SymEngine> {
        self.sym.as_deref()
    }

    /// A copy of the concolic seed model (input symbol → concrete value).
    pub fn sym_seed(&self) -> Model {
        self.sym.as_deref().map(|e| e.seed().clone()).unwrap_or_default()
    }

    /// Whether an SMT solver binary is launchable (see `REMU_SMT_SOLVER`).
    #[cfg(feature = "symbolic-solver")]
    pub fn sym_solver_available() -> bool {
        crate::symbolic::Solver::new().available()
    }

    /// Solve the current path's constraints, returning a model keyed by input
    /// name. `None` on unsat or solver failure.
    #[cfg(feature = "symbolic-solver")]
    pub fn sym_solve(&self) -> Option<std::collections::HashMap<String, u64>> {
        self.sym_solve_impl(None)
    }

    /// Solve for an input that flips branch `i` (reaches the other path).
    #[cfg(feature = "symbolic-solver")]
    pub fn sym_solve_flip(&self, i: usize) -> Option<std::collections::HashMap<String, u64>> {
        self.sym_solve_impl(Some(i))
    }

    #[cfg(feature = "symbolic-solver")]
    fn sym_solve_impl(&self, flip: Option<usize>) -> Option<std::collections::HashMap<String, u64>> {
        let eng = self.sym.as_deref()?;
        let raw = crate::symbolic::Solver::new().solve(&eng.smtlib(flip))?;
        let mut out = std::collections::HashMap::new();
        for (k, v) in raw {
            if let Some(id) = k.strip_prefix("x!").and_then(|d| d.parse::<u32>().ok())
                && let Some(name) = eng.input_name(id)
            {
                out.insert(name.to_string(), v);
            }
        }
        Some(out)
    }

    /// Check the golden concolic invariant: every live register/flag shadow
    /// evaluates (under the seed) to the current concrete value.
    pub fn sym_check_invariant(&self) -> bool {
        let Some(eng) = self.sym.as_deref() else {
            return true;
        };
        let f = self.regs.eflags;
        let flags = [
            f.contains(EFlags::CF),
            f.contains(EFlags::PF),
            f.contains(EFlags::AF),
            f.contains(EFlags::ZF),
            f.contains(EFlags::SF),
            f.contains(EFlags::OF),
        ];
        eng.check(|i| self.regs.gpr[i] as u64, flags)
    }

    // --- Hook helpers (called from the fused handlers) --------------------------

    /// Whether symbolic tracking is on. Cheap; the call sites are also gated.
    #[inline]
    pub(crate) fn sym_active(&self) -> bool {
        self.sym.as_deref().is_some_and(|e| e.enabled)
    }

    fn place_of(&self, op: Operand, width: Width) -> Place {
        match op {
            Operand::Reg(i) => reg_place(i, width),
            Operand::Mem { seg, off } => {
                let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
                Place::mem(lin as u64, width)
            }
        }
    }

    /// The concrete full-register value backing a register place (0 otherwise).
    fn reg_concrete(&self, place: Place) -> u64 {
        match place {
            Place::Reg { slot, .. } => self.regs.gpr[slot as usize] as u64,
            _ => 0,
        }
    }

    /// Core ALU forward: the concrete op already ran, so this only updates the
    /// shadow and flags.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu(&mut self, idx: usize, width: Width, a: Place, b: Place, dst: Option<Place>, av: u64, bv: u64, rv: u64) {
        let cin = self.regs.eflags.contains(EFlags::CF);
        let concrete_reg = dst.map_or(0, |d| self.reg_concrete(d));
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.alu(idx, width, a, b, dst, av, bv, rv, concrete_reg, cin);
        }
    }

    /// `op r/m, r` (destination is the r/m operand).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu_rm_r(&mut self, idx: usize, width: Width, op: Operand, reg: u8, wb: bool, av: u64, bv: u64, rv: u64) {
        let ap = self.place_of(op, width);
        let bp = reg_place(reg, width);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, width, ap, bp, dst, av, bv, rv);
    }

    /// `op r, r/m` (destination is the register operand).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu_r_rm(&mut self, idx: usize, width: Width, op: Operand, reg: u8, wb: bool, av: u64, bv: u64, rv: u64) {
        let ap = reg_place(reg, width);
        let bp = self.place_of(op, width);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, width, ap, bp, dst, av, bv, rv);
    }

    /// `op AL/AX/EAX, imm`.
    pub(crate) fn sym_alu_acc_imm(&mut self, idx: usize, width: Width, wb: bool, av: u64, bv: u64, rv: u64) {
        let ap = reg_place(0, width);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, width, ap, Place::Imm, dst, av, bv, rv);
    }

    /// Group-1 `op r/m, imm` (op index is the ModRM `reg` field; index 7 = CMP,
    /// no writeback).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu_grp1(&mut self, idx: usize, width: Width, op: Operand, wb: bool, av: u64, bv: u64, rv: u64) {
        let ap = self.place_of(op, width);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, width, ap, Place::Imm, dst, av, bv, rv);
    }

    /// `INC`/`DEC`/`NEG` on a register.
    pub(crate) fn sym_unary(&mut self, op: UnaryOp, reg: u8, width: Width, av: u64, rv: u64) {
        let place = reg_place(reg, width);
        let concrete_reg = self.reg_concrete(place);
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.unary(op, place, width, av, rv, concrete_reg);
        }
    }

    /// Drop a register's shadow (after a concrete `MOV reg, imm`).
    pub(crate) fn sym_concretize_reg(&mut self, reg: u8, width: Width) {
        let Place::Reg { slot, .. } = reg_place(reg, width) else {
            return;
        };
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.concretize_reg(slot);
        }
    }

    /// Shift/rotate group forward (`sub` = ModRM `reg` field, `n` = concrete count).
    pub(crate) fn sym_shift(&mut self, sub: u8, op: Operand, width: Width, n: u32, vv: u64, rv: u64) {
        let place = self.place_of(op, width);
        let concrete_reg = self.reg_concrete(place);
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.shift(sub, place, width, n, vv, rv, concrete_reg);
        }
    }

    /// `MOV dst, src` forward.
    fn sym_mov(&mut self, dst: Place, src: Place, width: Width, val: u64) {
        let concrete_reg = self.reg_concrete(dst);
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.mov(dst, src, width, val, concrete_reg);
        }
    }

    /// `MOV r/m, r` (88/89).
    pub(crate) fn sym_mov_rm_r(&mut self, width: Width, op: Operand, reg: u8, val: u64) {
        let dst = self.place_of(op, width);
        self.sym_mov(dst, reg_place(reg, width), width, val);
    }

    /// `MOV r, r/m` (8A/8B).
    pub(crate) fn sym_mov_r_rm(&mut self, width: Width, op: Operand, reg: u8, val: u64) {
        let src = self.place_of(op, width);
        self.sym_mov(reg_place(reg, width), src, width, val);
    }

    /// Conditional branch: record a path constraint when the condition is
    /// symbolic. `n` is the low nibble of the Jcc opcode; `taken` the concrete
    /// decision.
    pub(crate) fn sym_branch(&mut self, n: u8, taken: bool) {
        let f = self.regs.eflags;
        let bits = [
            f.contains(EFlags::CF),
            f.contains(EFlags::PF),
            f.contains(EFlags::AF),
            f.contains(EFlags::ZF),
            f.contains(EFlags::SF),
            f.contains(EFlags::OF),
        ];
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.branch(n, taken, bits);
        }
    }
}
