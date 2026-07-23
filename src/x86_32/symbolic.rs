//! The 80386 ↔ symbolic-overlay glue (feature `symbolic`).
//!
//! The arch-neutral engine lives in [`crate::symbolic`]; this module is the
//! thin per-core binding: it converts the core's [`Operand`] into a
//! [`Place`], gathers the concrete values the engine needs, and forwards to the
//! [`SymEngine`] hanging off [`Cpu::sym`]. The instrumentation call sites live
//! in `execute.rs`/`mod.rs`, each `#[cfg(feature = "symbolic")]`-gated so a
//! default build carries none of this.
//!
//! While the overlay is active the core is forced onto the fused decode path
//! (see `exec_one`), so only the fused handlers here need seams — the icache
//! and decoded path never see symbolic data.

use super::modrm::Operand;
use super::registers::EFlags;
use super::Cpu;
use crate::symbolic::{BoolExpr, Model, Place, SymEngine, SymId, Width};

/// The GPR dword slot a register operand occupies (8-bit maps AH..BH → EAX..EBX).
#[inline]
fn reg_base(idx: u8, width: Width) -> usize {
    if width == 8 {
        (idx & 3) as usize
    } else {
        (idx & 7) as usize
    }
}

impl Cpu {
    // --- Public API -------------------------------------------------------------

    /// Enable the concolic overlay (idempotent). Marks the core to run its
    /// reference fused path and track symbolic dataflow.
    pub fn sym_init(&mut self) {
        self.sym
            .get_or_insert_with(|| Box::new(SymEngine::new()))
            .enable();
    }

    /// Mark a 32-bit GPR (encoding slot 0..7) symbolic, seeded with its current
    /// concrete value. Returns the fresh input symbol id.
    pub fn sym_symbolize_reg32(&mut self, slot: u8, name: &str) -> SymId {
        let concrete = self.regs.gpr[(slot & 7) as usize];
        self.sym
            .as_deref_mut()
            .expect("call sym_init first")
            .symbolize_reg(slot, 32, name.to_string(), concrete)
    }

    /// Mark `width/8` memory bytes at linear address `lin` symbolic, seeded with
    /// `concrete` (the caller supplies the current bytes, little-endian).
    pub fn sym_symbolize_mem(&mut self, lin: u32, width: Width, name: &str, concrete: u64) -> SymId {
        self.sym
            .as_deref_mut()
            .expect("call sym_init first")
            .symbolize_mem(lin, width, name.to_string(), concrete)
    }

    /// The path constraints collected so far (empty if the overlay is off).
    pub fn sym_constraints(&self) -> &[BoolExpr] {
        match self.sym.as_deref() {
            Some(e) => e.constraints(),
            None => &[],
        }
    }

    /// The engine, for inspection (inputs, seed, constraints).
    pub fn sym_engine(&self) -> Option<&SymEngine> {
        self.sym.as_deref()
    }

    /// A copy of the concolic seed model (input symbol → concrete value).
    pub fn sym_seed(&self) -> Model {
        self.sym.as_deref().map(|e| e.seed().clone()).unwrap_or_default()
    }

    /// Check the golden concolic invariant: every live register/flag shadow
    /// evaluates (under the seed) to the current concrete value. `true` when
    /// the overlay is off.
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
        eng.check(&self.regs.gpr, flags)
    }

    // --- Hook helpers (called from the fused handlers) --------------------------

    /// Whether symbolic tracking is on. Cheap; the call sites are also gated.
    #[inline]
    pub(crate) fn sym_active(&self) -> bool {
        self.sym.as_deref().is_some_and(|e| e.enabled)
    }

    fn place_of(&self, op: Operand, width: Width) -> Place {
        match op {
            Operand::Reg(i) => Place::reg(i, width),
            Operand::Mem { seg, off } => {
                let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
                Place::mem(lin, width)
            }
        }
    }

    /// Core ALU forward: the concrete op already ran, so this only updates the
    /// shadow and flags.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu(
        &mut self,
        idx: usize,
        width: Width,
        a: Place,
        b: Place,
        dst: Option<Place>,
        av: u64,
        bv: u64,
        rv: u64,
    ) {
        let cin = self.regs.eflags.contains(EFlags::CF);
        let dst_dword = match dst {
            Some(Place::Reg { idx, width }) => self.regs.gpr[reg_base(idx, width)],
            _ => 0,
        };
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.alu(idx, width, a, b, dst, av, bv, rv, dst_dword, cin);
        }
    }

    /// `op r/m, r` (destination is the r/m operand).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu_rm_r(&mut self, idx: usize, width: Width, op: Operand, reg: u8, wb: bool, av: u64, bv: u64, rv: u64) {
        let ap = self.place_of(op, width);
        let bp = Place::reg(reg, width);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, width, ap, bp, dst, av, bv, rv);
    }

    /// `op r, r/m` (destination is the register operand).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_alu_r_rm(&mut self, idx: usize, width: Width, op: Operand, reg: u8, wb: bool, av: u64, bv: u64, rv: u64) {
        let ap = Place::reg(reg, width);
        let bp = self.place_of(op, width);
        let dst = if wb { Some(ap) } else { None };
        self.sym_alu(idx, width, ap, bp, dst, av, bv, rv);
    }

    /// `op AL/AX/EAX, imm`.
    pub(crate) fn sym_alu_acc_imm(&mut self, idx: usize, width: Width, wb: bool, av: u64, bv: u64, rv: u64) {
        let ap = Place::reg(0, width);
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

    /// `MOV dst, src` forward.
    pub(crate) fn sym_mov(&mut self, dst: Place, src: Place, width: Width, val: u64) {
        let dst_dword = match dst {
            Place::Reg { idx, width } => self.regs.gpr[reg_base(idx, width)],
            _ => 0,
        };
        if let Some(eng) = self.sym.as_deref_mut() {
            eng.mov(dst, src, width, val, dst_dword);
        }
    }

    /// `MOV r/m, r` (88/89).
    pub(crate) fn sym_mov_rm_r(&mut self, width: Width, op: Operand, reg: u8, val: u64) {
        let dst = self.place_of(op, width);
        self.sym_mov(dst, Place::reg(reg, width), width, val);
    }

    /// `MOV r, r/m` (8A/8B).
    pub(crate) fn sym_mov_r_rm(&mut self, width: Width, op: Operand, reg: u8, val: u64) {
        let src = self.place_of(op, width);
        self.sym_mov(Place::reg(reg, width), src, width, val);
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
