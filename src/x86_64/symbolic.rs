//! The x86-64 ↔ symbolic-overlay glue (feature `symbolic`).
//!
//! Mirrors `x86_32::symbolic`, reusing the arch-neutral [`SymEngine`] with the
//! x86-64 register model: 16 registers of 64 bits, REX-dependent 8-bit encoding,
//! and the flat/FS-GS linear-address rule in long mode. Covers the common wide
//! (16/32/64-bit) ALU path, MOV, immediate-MOV concretization, and conditional
//! branches; 8-bit ALU, INC/DEC and shifts stay concrete for now (self-heal
//! keeps that sound).

use super::modrm::Operand;
use super::registers::{reg, RFlags};
use super::{Cpu, O16, O32, O64};
use crate::symbolic::{BoolExpr, Model, Place, SymEngine, SymId, Width};

impl Cpu {
    // --- Public API -------------------------------------------------------------

    /// Enable the concolic overlay (idempotent).
    pub fn sym_init(&mut self) {
        self.sym
            .get_or_insert_with(|| Box::new(SymEngine::new(16, 64, 6)))
            .enable();
    }

    /// Mark a 64-bit GPR (encoding slot 0..15) symbolic, seeded with its current
    /// value.
    pub fn sym_symbolize_reg64(&mut self, slot: u8, name: &str) -> SymId {
        let concrete = self.regs.gpr[(slot & 15) as usize];
        self.sym
            .as_deref_mut()
            .expect("call sym_init first")
            .symbolize_reg(slot & 15, 0, 64, name.to_string(), concrete)
    }

    /// Mark `width/8` memory bytes at linear address `lin` symbolic, seeded with
    /// `concrete` (little-endian).
    pub fn sym_symbolize_mem(&mut self, lin: u64, width: Width, name: &str, concrete: u64) -> SymId {
        self.sym
            .as_deref_mut()
            .expect("call sym_init first")
            .symbolize_mem(lin, width, name.to_string(), concrete)
    }

    /// The path constraints collected so far.
    pub fn sym_constraints(&self) -> &[BoolExpr] {
        match self.sym.as_deref() {
            Some(e) => e.constraints(),
            None => &[],
        }
    }

    /// SMT-LIB 2 (QF_BV) script for the current path.
    pub fn sym_smtlib(&self) -> String {
        self.sym.as_deref().map(|e| e.smtlib(None)).unwrap_or_default()
    }

    /// SMT-LIB 2 script that flips branch `i`.
    pub fn sym_smtlib_flip(&self, i: usize) -> String {
        self.sym.as_deref().map(|e| e.smtlib(Some(i))).unwrap_or_default()
    }

    pub fn sym_engine(&self) -> Option<&SymEngine> {
        self.sym.as_deref()
    }

    pub fn sym_seed(&self) -> Model {
        self.sym.as_deref().map(|e| e.seed().clone()).unwrap_or_default()
    }

    /// Check the golden concolic invariant.
    pub fn sym_check_invariant(&self) -> bool {
        let Some(eng) = self.sym.as_deref() else {
            return true;
        };
        let f = self.regs.rflags;
        let flags = [
            f.contains(RFlags::CF),
            f.contains(RFlags::PF),
            f.contains(RFlags::AF),
            f.contains(RFlags::ZF),
            f.contains(RFlags::SF),
            f.contains(RFlags::OF),
        ];
        eng.check(|i| self.regs.gpr[i], flags)
    }

    #[cfg(feature = "symbolic-solver")]
    pub fn sym_solver_available() -> bool {
        crate::symbolic::Solver::new().available()
    }

    #[cfg(feature = "symbolic-solver")]
    pub fn sym_solve(&self) -> Option<std::collections::HashMap<String, u64>> {
        self.sym_solve_impl(None)
    }

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

    // --- Hook helpers -----------------------------------------------------------

    #[inline]
    pub(crate) fn sym_active(&self) -> bool {
        self.sym.as_deref().is_some_and(|e| e.enabled)
    }

    /// Current operand-size width in bits.
    pub(crate) fn sym_width(&self) -> Width {
        match self.osize {
            O16 => 16,
            O32 => 32,
            O64 => 64,
        }
    }

    /// Resolve a register operand into a `{slot, lo, width}` slice. 8-bit uses
    /// the REX-dependent encoding (legacy AH/CH/DH/BH vs SPL/BPL/… + R8B..).
    fn reg_place(&self, idx: u8, width: Width) -> Place {
        let (slot, lo) = if width == 8 {
            if self.rex.is_none() && idx & 0xC == 4 {
                (idx & 3, 8) // legacy high byte
            } else {
                (idx & 15, 0)
            }
        } else {
            (idx & 15, 0)
        };
        Place::reg(slot, lo, width)
    }

    /// Long-mode/legacy linear address of a memory operand.
    fn lin_of(&self, seg: u8, off: u64) -> u64 {
        if self.m64 {
            let base = if seg == reg::FS || seg == reg::GS {
                self.regs.seg[seg as usize].base
            } else {
                0
            };
            base.wrapping_add(off)
        } else {
            (self.regs.seg[seg as usize].base as u32).wrapping_add(off as u32) as u64
        }
    }

    fn place_of(&self, op: Operand, width: Width) -> Place {
        match op {
            Operand::Reg(i) => self.reg_place(i, width),
            Operand::Mem { seg, off } => Place::mem(self.lin_of(seg, off), width),
        }
    }

    fn reg_concrete(&self, place: Place) -> u64 {
        match place {
            Place::Reg { slot, .. } => self.regs.gpr[slot as usize],
            _ => 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu(&mut self, idx: usize, width: Width, a: Place, b: Place, dst: Option<Place>, av: u64, bv: u64, rv: u64) {
        let cin = self.regs.rflags.contains(RFlags::CF);
        let concrete_reg = dst.map_or(0, |d| self.reg_concrete(d));
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.alu(idx, width, a, b, dst, av, bv, rv, concrete_reg, cin);
        }
    }

    /// `op r/m, r` (wide; destination is the r/m operand).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu_rm_r(&mut self, idx: usize, op: Operand, reg: u8, wb: bool, av: u64, bv: u64, rv: u64) {
        let w = self.sym_width();
        let ap = self.place_of(op, w);
        let bp = self.reg_place(reg, w);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, w, ap, bp, dst, av, bv, rv);
    }

    /// `op r, r/m` (wide; destination is the register operand).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu_r_rm(&mut self, idx: usize, op: Operand, reg: u8, wb: bool, av: u64, bv: u64, rv: u64) {
        let w = self.sym_width();
        let ap = self.reg_place(reg, w);
        let bp = self.place_of(op, w);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, w, ap, bp, dst, av, bv, rv);
    }

    /// `op acc, imm` (wide).
    pub(crate) fn sym_alu_acc_imm(&mut self, idx: usize, wb: bool, av: u64, bv: u64, rv: u64) {
        let w = self.sym_width();
        let ap = self.reg_place(0, w);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, w, ap, Place::Imm, dst, av, bv, rv);
    }

    /// Group-1 `op r/m, imm` (wide; `sub` 7 = CMP, no writeback).
    pub(crate) fn sym_alu_grp1(&mut self, sub: u8, op: Operand, av: u64, bv: u64, rv: u64) {
        let w = self.sym_width();
        let ap = self.place_of(op, w);
        let dst = if sub != 7 { Some(ap) } else { None };
        self.sym_alu(sub as usize, w, ap, Place::Imm, dst, av, bv, rv);
    }

    /// Drop a register's shadow (after a concrete `MOV reg, imm`).
    pub(crate) fn sym_concretize_reg(&mut self, reg: u8, width: Width) {
        let Place::Reg { slot, .. } = self.reg_place(reg, width) else {
            return;
        };
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.concretize_reg(slot);
        }
    }

    fn sym_mov(&mut self, dst: Place, src: Place, width: Width, val: u64) {
        let concrete_reg = self.reg_concrete(dst);
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.mov(dst, src, width, val, concrete_reg);
        }
    }

    /// `MOV r/m, r` (88 = 8-bit, 89 = wide).
    pub(crate) fn sym_mov_rm_r(&mut self, width: Width, op: Operand, reg: u8, val: u64) {
        let dst = self.place_of(op, width);
        let src = self.reg_place(reg, width);
        self.sym_mov(dst, src, width, val);
    }

    /// `MOV r, r/m` (8A = 8-bit, 8B = wide).
    pub(crate) fn sym_mov_r_rm(&mut self, width: Width, op: Operand, reg: u8, val: u64) {
        let dst = self.reg_place(reg, width);
        let src = self.place_of(op, width);
        self.sym_mov(dst, src, width, val);
    }

    /// Conditional branch: record a path constraint when the condition is symbolic.
    pub(crate) fn sym_branch(&mut self, n: u8, taken: bool) {
        let f = self.regs.rflags;
        let bits = [
            f.contains(RFlags::CF),
            f.contains(RFlags::PF),
            f.contains(RFlags::AF),
            f.contains(RFlags::ZF),
            f.contains(RFlags::SF),
            f.contains(RFlags::OF),
        ];
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.branch(n, taken, bits);
        }
    }
}
