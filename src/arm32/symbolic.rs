//! The ARM32 ↔ symbolic-overlay glue (feature `symbolic`).
//!
//! ARM differs from x86 enough that the engine's x86-shaped hooks don't apply:
//! it has 4 flags (N Z C V), a `(result, carry, overflow)` ALU where the C of a
//! logical op is the barrel-shifter carry, and **every instruction is
//! predicated**. So this glue drives the arch-neutral [`SymEngine`] through its
//! generic primitives (`read_place`/`write_place`/`flag`/`set_flag`/
//! `record_branch`) plus the [`arm_alu`] NZCV builders.
//!
//! Modeled: immediate data-processing (the 16 ALU ops) and the per-instruction
//! condition fork. Register-operand DP forms (`DpShImm`/`DpShReg`), LDR/STR and
//! r15-as-PC writes concretize for now — self-heal keeps that sound.
//!
//! [`arm_alu`]: crate::symbolic::arm_alu

use super::decode::{alu_op, dp, DecodedInsn, Op};
use super::Cpu;
use crate::symbolic::arm_alu;
use crate::symbolic::{BoolExpr, Expr, Model, Place, SymEngine, SymId};

/// ARM flag slot indices (the engine's flag Vec is sized 4 for this core).
const N: usize = 0;
const Z: usize = 1;
const C: usize = 2;
const V: usize = 3;

impl Cpu {
    // --- Public API -------------------------------------------------------------

    /// Enable the concolic overlay (idempotent).
    pub fn sym_init(&mut self) {
        self.sym
            .get_or_insert_with(|| Box::new(SymEngine::new(16, 32, 4)))
            .enable();
    }

    /// Mark register `r` (0..15) symbolic, seeded with its current value.
    pub fn sym_symbolize_reg(&mut self, r: u8, name: &str) -> SymId {
        let concrete = self.regs.gpr[(r & 15) as usize] as u64;
        self.sym
            .as_deref_mut()
            .expect("call sym_init first")
            .symbolize_reg(r & 15, 0, 32, name.to_string(), concrete)
    }

    /// Mark `width/8` memory bytes at `addr` symbolic, seeded with `concrete`.
    pub fn sym_symbolize_mem(&mut self, addr: u32, width: crate::symbolic::Width, name: &str, concrete: u64) -> SymId {
        self.sym
            .as_deref_mut()
            .expect("call sym_init first")
            .symbolize_mem(addr as u64, width, name.to_string(), concrete)
    }

    pub fn sym_constraints(&self) -> &[BoolExpr] {
        match self.sym.as_deref() {
            Some(e) => e.constraints(),
            None => &[],
        }
    }

    pub fn sym_smtlib(&self) -> String {
        self.sym.as_deref().map(|e| e.smtlib(None)).unwrap_or_default()
    }

    pub fn sym_smtlib_flip(&self, i: usize) -> String {
        self.sym.as_deref().map(|e| e.smtlib(Some(i))).unwrap_or_default()
    }

    pub fn sym_engine(&self) -> Option<&SymEngine> {
        self.sym.as_deref()
    }

    pub fn sym_seed(&self) -> Model {
        self.sym.as_deref().map(|e| e.seed().clone()).unwrap_or_default()
    }

    /// Check the golden concolic invariant (registers + NZCV flags).
    pub fn sym_check_invariant(&self) -> bool {
        let Some(eng) = self.sym.as_deref() else {
            return true;
        };
        let p = self.regs.cpsr;
        let flags = [p.n(), p.z(), p.c(), p.v(), false, false];
        eng.check(|i| self.regs.gpr[i] as u64, flags)
    }

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

    // --- Hooks ------------------------------------------------------------------

    #[inline]
    pub(crate) fn sym_active(&self) -> bool {
        self.sym.as_deref().is_some_and(|e| e.enabled)
    }

    /// Per-instruction predication fork: record a constraint when the condition
    /// depends on symbolic flags. `cond` is the 4-bit ARM condition; `pass` the
    /// concrete decision. AL/NV (>= 0xE) never fork and aren't passed here.
    pub(crate) fn sym_cond(&mut self, cond: u8, pass: bool) {
        let p = self.regs.cpsr;
        let (bn, bz, bc, bv) = (p.n(), p.z(), p.c(), p.v());
        let Some(eng) = self.sym.as_deref_mut() else {
            return;
        };
        let n = eng.flag(N, bn);
        let z = eng.flag(Z, bz);
        let c = eng.flag(C, bc);
        let v = eng.flag(V, bv);
        let not = BoolExpr::not;
        let pred = match cond {
            0x0 => z,                                                   // EQ
            0x1 => not(z),                                             // NE
            0x2 => c,                                                  // CS/HS
            0x3 => not(c),                                             // CC/LO
            0x4 => n,                                                  // MI
            0x5 => not(n),                                             // PL
            0x6 => v,                                                  // VS
            0x7 => not(v),                                             // VC
            0x8 => BoolExpr::and(c, not(z)),                          // HI
            0x9 => BoolExpr::or(not(c), z),                           // LS
            0xA => not(BoolExpr::xor(n, v)),                          // GE (N == V)
            0xB => BoolExpr::xor(n, v),                               // LT (N != V)
            0xC => BoolExpr::and(not(z), not(BoolExpr::xor(n, v))),   // GT
            _ => BoolExpr::or(z, BoolExpr::xor(n, v)),                // LE (0xD)
        };
        eng.record_branch(pred, pass);
    }

    /// Symbolic side of a data-processing instruction. Immediate (`DpImm`) and
    /// immediate-shifted-register (`DpShImm`) forms are modeled exactly;
    /// register-shift (`DpShReg`) and r15 destinations concretize. The concrete
    /// op has already run. Concrete values (`cin_c`..`v_c`) are the pre-op
    /// carry-in, `Rn`, `op2`, shifter-carry, result, and arithmetic carry/ovf.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sym_dp(&mut self, d: &DecodedInsn, cin_c: bool, rn_c: u32, op2_c: u32, sc_c: bool, result_c: u32, _c_c: bool, _v_c: bool) {
        let opn = d.aux & dp::OP;
        let s = d.aux & dp::S != 0;
        let writes_rd = !matches!(opn, alu_op::TST | alu_op::TEQ | alu_op::CMP | alu_op::CMN);
        let rd = d.rd;
        let rn = d.rn;
        let rm = d.rm;
        let ty = (d.aux & dp::TY) >> dp::TY_SHIFT;
        let concrete_rd = self.regs.gpr[rd as usize] as u64;
        // Rm's raw value (before the barrel shift) for the DpShImm read fallback.
        let rm_c = if d.op == Op::DpShImm { self.reg_op(rm) } else { 0 };

        let Some(eng) = self.sym.as_deref_mut() else {
            return;
        };

        // Register-shift forms and r15 destinations aren't modeled: drop the
        // affected shadows (self-heal keeps reads sound).
        if d.op == Op::DpShReg || (writes_rd && rd == 15) {
            if writes_rd && rd != 15 {
                eng.concretize(Place::reg(rd, 0, 32));
            }
            if s {
                eng.clear_flags();
            }
            return;
        }

        let cin = eng.flag(C, cin_c);
        let rn_e = eng.read_place(Place::reg(rn, 0, 32), 32, rn_c as u64);
        // Operand 2 and its shifter carry: a concrete rotated immediate, or the
        // symbolic Rm barrel-shifted by the immediate amount.
        let (op2, sc) = match d.op {
            Op::DpImm => (
                Expr::constant(32, op2_c as u64),
                if d.aux & dp::ROT != 0 {
                    BoolExpr::constant(sc_c)
                } else {
                    cin.clone()
                },
            ),
            _ => {
                let rm_e = eng.read_place(Place::reg(rm, 0, 32), 32, rm_c as u64);
                arm_alu::shift_imm(ty, d.rs, &rm_e, &cin)
            }
        };

        let t = BoolExpr::constant(true);
        let f = BoolExpr::constant(false);
        let and = |a: &Expr, b: &Expr| Expr::bin(crate::symbolic::expr::BinOp::And, a.clone(), b.clone());
        let (result, new_c, new_v): (Expr, BoolExpr, Option<BoolExpr>) = match opn {
            alu_op::AND => (and(&rn_e, &op2), sc, None),
            alu_op::EOR => (Expr::bin(crate::symbolic::expr::BinOp::Xor, rn_e.clone(), op2.clone()), sc, None),
            alu_op::SUB => wrap(arm_alu::sub_with_carry(&rn_e, &op2, &t)),
            alu_op::RSB => wrap(arm_alu::sub_with_carry(&op2, &rn_e, &t)),
            alu_op::ADD => wrap(arm_alu::add_with_carry(&rn_e, &op2, &f)),
            alu_op::ADC => wrap(arm_alu::add_with_carry(&rn_e, &op2, &cin)),
            alu_op::SBC => wrap(arm_alu::sub_with_carry(&rn_e, &op2, &cin)),
            alu_op::RSC => wrap(arm_alu::sub_with_carry(&op2, &rn_e, &cin)),
            alu_op::TST => (and(&rn_e, &op2), sc, None),
            alu_op::TEQ => (Expr::bin(crate::symbolic::expr::BinOp::Xor, rn_e.clone(), op2.clone()), sc, None),
            alu_op::CMP => wrap(arm_alu::sub_with_carry(&rn_e, &op2, &t)),
            alu_op::CMN => wrap(arm_alu::add_with_carry(&rn_e, &op2, &f)),
            alu_op::ORR => (Expr::bin(crate::symbolic::expr::BinOp::Or, rn_e.clone(), op2.clone()), sc, None),
            alu_op::MOV => (op2.clone(), sc, None),
            alu_op::BIC => (and(&rn_e, &Expr::un(crate::symbolic::expr::UnOp::Not, op2.clone())), sc, None),
            _ => (Expr::un(crate::symbolic::expr::UnOp::Not, op2.clone()), sc, None), // MVN
        };
        debug_assert_eq!(result.eval(eng.seed()), result_c as u64, "symbolic DP diverged");

        if s {
            let (n, z) = arm_alu::nz(&result);
            eng.set_flag(N, Some(n));
            eng.set_flag(Z, Some(z));
            eng.set_flag(C, Some(new_c));
            if let Some(v) = new_v {
                eng.set_flag(V, Some(v));
            }
        }
        if writes_rd {
            eng.write_place(Place::reg(rd, 0, 32), result, concrete_rd);
        }
    }
}

/// Reshape a `(result, carry, overflow)` tuple for the DP match.
fn wrap((r, c, v): (Expr, BoolExpr, BoolExpr)) -> (Expr, BoolExpr, Option<BoolExpr>) {
    (r, c, Some(v))
}
