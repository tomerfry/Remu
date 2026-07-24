//! The concolic overlay state — a sparse symbolic shadow of the machine.
//!
//! [`SymEngine`] holds a symbolic expression for each register/flag/memory byte
//! that is *tainted* (derived from a symbolic input); everything absent is
//! concrete and read from the real CPU. It is arch-neutral: it speaks only in
//! [`Place`]s (a fully-resolved register slice, a linear address + width, or an
//! immediate) and bitvector [`Expr`]s, so the per-core glue binds it without
//! leaking core types in here. The glue does the architecture-specific work of
//! turning a register operand into a `{slot, lo, width}` slice (e.g. the x86
//! `AH` high-byte convention, or x86-64 REX 8-bit registers); the engine just
//! stores one `reg_bits`-wide expression per register slot.
//!
//! **Soundness under partial coverage.** The overlay only models a subset of
//! instructions; anything else runs concretely and may change a register the
//! shadow still describes. Reads therefore *self-heal*: if a shadow no longer
//! evaluates (under the seed) to the concrete value the caller observed, it is
//! dropped and the concrete value is used. This keeps the invariant
//! `shadow.eval(seed) == concrete` true wherever a symbolic value is actually
//! consumed. The concolic seed is always available, so this check is cheap.

use std::collections::HashMap;

use super::alu::{self, FlagDefs};
use super::expr::{mask, BoolExpr, Expr, Model, SymId, Width};

/// x86 status-flag slot indices (the order [`FlagDefs`] and the x86 hooks use):
/// carry, parity, aux, zero, sign, overflow. Other cores index [`SymEngine::flags`]
/// with their own convention (e.g. ARM's N/Z/C/V) — the store is just a slice.
const CF: usize = 0;
const PF: usize = 1;
const AF: usize = 2;
const ZF: usize = 3;
const SF: usize = 4;
const OF: usize = 5;

/// A flag-setting unary op (`INC`/`DEC`/`NEG`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Inc,
    Dec,
    Neg,
}

/// Where an operand value lives, so a hook can fetch or store its shadow.
///
/// A register operand is a fully-resolved slice of a register: `slot` selects
/// the register, `lo` the low bit within it, `width` the operand size. The glue
/// resolves architecture-specific encodings (x86 `AH`, x86-64 REX bytes) into
/// this form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    Reg { slot: u8, lo: Width, width: Width },
    Mem { lin: u64, width: Width },
    /// An immediate — always concrete.
    Imm,
}

impl Place {
    pub fn reg(slot: u8, lo: Width, width: Width) -> Place {
        Place::Reg { slot, lo, width }
    }
    pub fn mem(lin: u64, width: Width) -> Place {
        Place::Mem { lin, width }
    }
}

/// A declared symbolic input: its variable id, a human name, and its width.
#[derive(Debug, Clone)]
pub struct Input {
    pub id: SymId,
    pub name: String,
    pub width: Width,
}

/// The sparse symbolic shadow of the CPU.
#[derive(Debug, Clone)]
pub struct SymEngine {
    /// When `false`, the core ignores the overlay entirely (fast path).
    pub enabled: bool,
    /// Full-register width in bits (32 for the 386, 64 for x86-64).
    reg_bits: Width,
    /// Per-register `reg_bits`-wide symbolic value; `None` = concrete.
    regs: Vec<Option<Expr>>,
    /// Per-status-flag symbolic value; `None` = concrete (read real flags).
    /// Sized per architecture (6 for x86, 4 for ARM NZCV); indexed by the
    /// core's own flag convention.
    flags: Vec<Option<BoolExpr>>,
    /// Symbolic memory, one 8-bit expr per linear address; absent = concrete.
    mem: HashMap<u64, Expr>,
    /// Concrete seed value for each input symbol (the concolic witness).
    seed: Model,
    /// Declared inputs, in creation order.
    inputs: Vec<Input>,
    /// Path constraints collected at symbolic branches.
    constraints: Vec<BoolExpr>,
    next_id: u32,
}

impl SymEngine {
    /// A fresh engine for a core with `num_regs` registers of `reg_bits` bits
    /// and `num_flags` status flags.
    pub fn new(num_regs: usize, reg_bits: Width, num_flags: usize) -> Self {
        SymEngine {
            enabled: false,
            reg_bits,
            regs: vec![None; num_regs],
            flags: vec![None; num_flags],
            mem: HashMap::new(),
            seed: Model::new(),
            inputs: Vec::new(),
            constraints: Vec::new(),
            next_id: 0,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn constraints(&self) -> &[BoolExpr] {
        &self.constraints
    }

    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// The concolic seed (input symbol → concrete value).
    pub fn seed(&self) -> &Model {
        &self.seed
    }

    /// The golden concolic invariant: every live register/flag shadow, when
    /// evaluated under the seed, equals the concrete value. `reg_val(slot)`
    /// returns the concrete full-register value. Used by tests and as a fuzzing
    /// oracle.
    pub fn check(&self, reg_val: impl Fn(usize) -> u64, flags: [bool; 6]) -> bool {
        for (i, slot) in self.regs.iter().enumerate() {
            if let Some(e) = slot
                && e.eval(&self.seed) != (reg_val(i) & mask(self.reg_bits))
            {
                return false;
            }
        }
        for (i, slot) in self.flags.iter().enumerate() {
            if let Some(e) = slot
                && e.eval(&self.seed) != flags[i]
            {
                return false;
            }
        }
        true
    }

    /// The friendly name of an input symbol by its numeric id.
    pub fn input_name(&self, id: u32) -> Option<&str> {
        self.inputs.iter().find(|i| i.id.0 == id).map(|i| i.name.as_str())
    }

    /// An SMT-LIB 2 (QF_BV) script for the current path. With `flip = Some(i)`,
    /// the constraints up to and including branch `i` are asserted with `i`
    /// negated (and later constraints dropped) — i.e. solve for an input that
    /// takes the *other* side of branch `i`. With `None`, all constraints are
    /// asserted (the current path).
    pub fn smtlib(&self, flip: Option<usize>) -> String {
        use super::smtlib::write_bool;
        let mut s = String::from("(set-logic QF_BV)\n");
        for inp in &self.inputs {
            s.push_str(&format!("(declare-const x!{} (_ BitVec {}))\n", inp.id.0, inp.width));
        }
        let end = flip.map_or(self.constraints.len(), |i| i + 1);
        for (k, c) in self.constraints[..end].iter().enumerate() {
            s.push_str("(assert ");
            if Some(k) == flip {
                s.push_str("(not ");
                write_bool(c, &mut s);
                s.push(')');
            } else {
                write_bool(c, &mut s);
            }
            s.push_str(")\n");
        }
        s.push_str("(check-sat)\n");
        if !self.inputs.is_empty() {
            s.push_str("(get-value (");
            for (k, inp) in self.inputs.iter().enumerate() {
                if k > 0 {
                    s.push(' ');
                }
                s.push_str(&format!("x!{}", inp.id.0));
            }
            s.push_str("))\n");
        }
        s
    }

    /// Allocate a fresh input symbol seeded with `seed_val`.
    fn fresh(&mut self, name: String, width: Width, seed_val: u64) -> SymId {
        let id = SymId(self.next_id);
        self.next_id += 1;
        self.seed.insert(id, seed_val & mask(width));
        self.inputs.push(Input { id, name, width });
        id
    }

    /// Mark a register slice `[lo +: width]` of register `slot` symbolic,
    /// seeding it from the current concrete register value. Preserves the other
    /// bits of the register (unlike an instruction write).
    pub fn symbolize_reg(&mut self, slot: u8, lo: Width, width: Width, name: String, concrete_reg: u64) -> SymId {
        let seed_val = (concrete_reg >> lo) & mask(width);
        let id = self.fresh(name, width, seed_val);
        let sym = Expr::symbol(id, width);
        let spliced = self.splice(slot, lo, width, sym, concrete_reg, false);
        self.set_slot(slot, spliced);
        id
    }

    /// Mark `width/8` memory bytes at `lin` symbolic, seeding from their
    /// current concrete value.
    pub fn symbolize_mem(&mut self, lin: u64, width: Width, name: String, concrete: u64) -> SymId {
        let id = self.fresh(name, width, concrete);
        let sym = Expr::symbol(id, width);
        self.store_mem(lin, width, sym);
        id
    }

    // --- Register shadow --------------------------------------------------------

    fn set_slot(&mut self, slot: u8, value: Expr) {
        self.regs[slot as usize] = if value.as_const().is_some() { None } else { Some(value) };
    }

    /// Read a register slice, self-healing a stale shadow against `concrete`.
    fn read_reg(&mut self, slot: u8, lo: Width, width: Width, concrete: u64) -> Expr {
        let Some(regv) = self.regs[slot as usize].clone() else {
            return Expr::constant(width, concrete);
        };
        let opnd = if width == self.reg_bits {
            regv
        } else {
            Expr::extract(lo + width - 1, lo, regv)
        };
        if opnd.eval(&self.seed) != (concrete & mask(width)) {
            self.regs[slot as usize] = None; // stale — an unmodeled write changed it
            return Expr::constant(width, concrete);
        }
        opnd
    }

    /// Instruction write of a register slice (applies the x86-64 32-bit
    /// zero-extend rule).
    fn write_reg(&mut self, slot: u8, lo: Width, width: Width, value: Expr, concrete_reg: u64) {
        let spliced = self.splice(slot, lo, width, value, concrete_reg, true);
        self.set_slot(slot, spliced);
    }

    /// Build the new full-register value with `value` placed at `[lo +: width]`.
    /// A full-width write replaces the register; a 32-bit write to a 64-bit
    /// register zero-extends (only when `instr_write`); everything else
    /// preserves the surrounding bits.
    fn splice(&self, slot: u8, lo: Width, width: Width, value: Expr, concrete_reg: u64, instr_write: bool) -> Expr {
        if width == self.reg_bits {
            return value;
        }
        if instr_write && self.reg_bits == 64 && width == 32 && lo == 0 {
            return Expr::zext(64, value);
        }
        let cur = self.regs[slot as usize]
            .clone()
            .unwrap_or_else(|| Expr::constant(self.reg_bits, concrete_reg));
        let hi_start = lo + width;
        let with_low = if lo > 0 {
            Expr::concat(value, Expr::extract(lo - 1, 0, cur.clone()))
        } else {
            value
        };
        if hi_start < self.reg_bits {
            Expr::concat(Expr::extract(self.reg_bits - 1, hi_start, cur), with_low)
        } else {
            with_low
        }
    }

    fn reg_symbolic(&self, slot: u8) -> bool {
        self.regs[slot as usize].is_some()
    }

    // --- Memory shadow ----------------------------------------------------------

    fn read_mem(&mut self, lin: u64, width: Width, concrete: u64) -> Expr {
        let nbytes = (width / 8) as u64;
        if (0..nbytes).all(|k| !self.mem.contains_key(&lin.wrapping_add(k))) {
            return Expr::constant(width, concrete);
        }
        let byte = |k: u64| {
            self.mem
                .get(&lin.wrapping_add(k))
                .cloned()
                .unwrap_or_else(|| Expr::constant(8, (concrete >> (8 * k)) & 0xFF))
        };
        let mut e = byte(nbytes - 1);
        for k in (0..nbytes - 1).rev() {
            e = Expr::concat(e, byte(k));
        }
        if e.eval(&self.seed) != (concrete & mask(width)) {
            for k in 0..nbytes {
                self.mem.remove(&lin.wrapping_add(k));
            }
            return Expr::constant(width, concrete);
        }
        e
    }

    fn store_mem(&mut self, lin: u64, width: Width, value: Expr) {
        let nbytes = (width / 8) as u64;
        for k in 0..nbytes {
            let byte = Expr::extract((8 * k + 7) as Width, (8 * k) as Width, value.clone());
            let a = lin.wrapping_add(k);
            if byte.as_const().is_some() {
                self.mem.remove(&a);
            } else {
                self.mem.insert(a, byte);
            }
        }
    }

    fn mem_symbolic(&self, lin: u64, width: Width) -> bool {
        (0..(width / 8) as u64).any(|k| self.mem.contains_key(&lin.wrapping_add(k)))
    }

    // --- Places -----------------------------------------------------------------

    /// Whether any bit of `place` currently carries a symbolic value.
    pub fn place_symbolic(&self, place: Place) -> bool {
        match place {
            Place::Reg { slot, .. } => self.reg_symbolic(slot),
            Place::Mem { lin, width } => self.mem_symbolic(lin, width),
            Place::Imm => false,
        }
    }

    /// Read the symbolic value of `place` at `width` bits, self-healing a stale
    /// shadow against the observed `concrete` value.
    pub fn read_place(&mut self, place: Place, width: Width, concrete: u64) -> Expr {
        match place {
            Place::Reg { slot, lo, .. } => self.read_reg(slot, lo, width, concrete),
            Place::Mem { lin, .. } => self.read_mem(lin, width, concrete),
            Place::Imm => Expr::constant(width, concrete),
        }
    }

    /// Store a symbolic value into `place`. `concrete_reg` is the destination
    /// register's concrete value (for sub-register splices; ignored for memory).
    pub fn write_place(&mut self, place: Place, value: Expr, concrete_reg: u64) {
        match place {
            Place::Reg { slot, lo, width } => self.write_reg(slot, lo, width, value, concrete_reg),
            Place::Mem { lin, width } => self.store_mem(lin, width, value),
            Place::Imm => {}
        }
    }

    /// Drop the shadow of `place` (make it concrete).
    pub fn concretize(&mut self, place: Place) {
        match place {
            Place::Reg { slot, .. } => self.regs[slot as usize] = None,
            Place::Mem { lin, width } => {
                for k in 0..(width / 8) as u64 {
                    self.mem.remove(&lin.wrapping_add(k));
                }
            }
            Place::Imm => {}
        }
    }

    /// Drop a register's shadow (e.g. after a `MOV reg, imm`).
    pub fn concretize_reg(&mut self, slot: u8) {
        self.regs[slot as usize] = None;
    }

    // --- Flags ------------------------------------------------------------------

    /// Read status flag `i` as a boolean expression, self-healing a stale
    /// shadow against `concrete_bit`.
    pub fn flag(&mut self, i: usize, concrete_bit: bool) -> BoolExpr {
        match self.flags[i].clone() {
            Some(e) => {
                if e.eval(&self.seed) != concrete_bit {
                    self.flags[i] = None; // stale
                    BoolExpr::constant(concrete_bit)
                } else {
                    e
                }
            }
            None => BoolExpr::constant(concrete_bit),
        }
    }

    /// Set status flag `i`; `None` leaves it unchanged, a folded (concrete)
    /// expression clears the shadow.
    pub fn set_flag(&mut self, i: usize, d: Option<BoolExpr>) {
        if let Some(e) = d {
            self.flags[i] = if e.as_const().is_some() { None } else { Some(e) };
        }
    }

    /// Clear every flag shadow (they became concrete).
    pub fn clear_flags(&mut self) {
        for f in self.flags.iter_mut() {
            *f = None;
        }
    }

    /// Record a branch/predication constraint from an already-built predicate
    /// (`pred == taken`). No-op when the predicate folded to a constant.
    pub fn record_branch(&mut self, pred: BoolExpr, taken: bool) {
        if pred.as_const().is_some() {
            return;
        }
        debug_assert_eq!(pred.eval(&self.seed), taken, "branch predicate diverged from concrete");
        self.constraints.push(if taken { pred } else { BoolExpr::not(pred) });
    }

    fn set_flags(&mut self, d: FlagDefs) {
        self.set_flag(CF, d.cf);
        self.set_flag(PF, d.pf);
        self.set_flag(AF, d.af);
        self.set_flag(ZF, d.zf);
        self.set_flag(SF, d.sf);
        self.set_flag(OF, d.of);
    }

    // --- Hooks (called by the per-core glue) ------------------------------------

    /// Symbolic side of an 8-op ALU instruction (see `alu::alu`). The concrete
    /// side already ran; `concrete_reg` is the destination register's concrete
    /// value (for splicing a sub-register result); `cin_bit` is the concrete CF.
    #[allow(clippy::too_many_arguments)]
    pub fn alu(
        &mut self,
        idx: usize,
        width: Width,
        a: Place,
        b: Place,
        dst: Option<Place>,
        av: u64,
        bv: u64,
        rv: u64,
        concrete_reg: u64,
        cin_bit: bool,
    ) {
        let needs_carry = idx == 2 || idx == 3; // ADC / SBB
        let any_sym =
            self.place_symbolic(a) || self.place_symbolic(b) || (needs_carry && self.flags[CF].is_some());
        if !any_sym {
            if let Some(d) = dst {
                self.concretize(d);
            }
            self.clear_flags();
            return;
        }
        let ea = self.read_place(a, width, av);
        let eb = self.read_place(b, width, bv);
        let cin = self.flag(CF, cin_bit);
        let (res, defs) = alu::alu(idx, &ea, &eb, &cin, width);
        debug_assert_eq!(res.eval(&self.seed), rv & mask(width), "symbolic ALU diverged from concrete");
        self.set_flags(defs);
        if let Some(d) = dst {
            self.write_place(d, res, concrete_reg);
        }
    }

    /// Symbolic side of a flag-setting unary op (`INC`/`DEC`/`NEG`). Concrete
    /// operands fold, so this handles both cases (`INC`/`DEC` leave `CF`).
    pub fn unary(&mut self, op: UnaryOp, place: Place, width: Width, av: u64, rv: u64, concrete_reg: u64) {
        let e = self.read_place(place, width, av);
        let (res, defs) = match op {
            UnaryOp::Inc => alu::inc(&e, width),
            UnaryOp::Dec => alu::dec(&e, width),
            UnaryOp::Neg => alu::neg(&e, width),
        };
        debug_assert_eq!(res.eval(&self.seed), rv & mask(width), "symbolic unary diverged");
        self.set_flags(defs);
        self.write_place(place, res, concrete_reg);
    }

    /// Symbolic side of a shift/rotate. `sub` is the group-2 sub-op
    /// (0=ROL 1=ROR 2=RCL 3=RCR 4/6=SHL 5=SHR 7=SAR); the count `n` is taken
    /// **concretely** (a symbolic `CL` is concretized to its seed value).
    /// `RCL`/`RCR` aren't modeled and concretize the destination. Uses the
    /// 386-style shift semantics — callers whose core differs (x86-64) should
    /// concretize instead.
    #[allow(clippy::too_many_arguments)]
    pub fn shift(&mut self, sub: u8, place: Place, width: Width, n: u32, vv: u64, rv: u64, concrete_reg: u64) {
        let kind = match sub {
            0 => alu::ShiftKind::Rol,
            1 => alu::ShiftKind::Ror,
            4 | 6 => alu::ShiftKind::Shl,
            5 => alu::ShiftKind::Shr,
            7 => alu::ShiftKind::Sar,
            _ => {
                self.concretize(place);
                self.flags[CF] = None;
                self.flags[OF] = None;
                return;
            }
        };
        let e = self.read_place(place, width, vv);
        let (res, defs) = alu::shift(kind, &e, n, width);
        debug_assert_eq!(res.eval(&self.seed), rv & mask(width), "symbolic shift diverged");
        self.set_flags(defs);
        self.write_place(place, res, concrete_reg);
    }

    /// Symbolic side of a `MOV dst, src`: propagate the source shadow, or
    /// concretize `dst` when the source is concrete.
    pub fn mov(&mut self, dst: Place, src: Place, width: Width, val: u64, concrete_reg: u64) {
        if !self.place_symbolic(src) {
            self.concretize(dst);
            return;
        }
        let e = self.read_place(src, width, val);
        self.write_place(dst, e, concrete_reg);
    }

    /// Symbolic side of a conditional branch. Records the path constraint when
    /// the deciding flags are symbolic; `concrete_bits` are the six live flag
    /// bits (CF PF AF ZF SF OF) and `taken` the concrete decision.
    pub fn branch(&mut self, n: u8, taken: bool, concrete_bits: [bool; 6]) {
        let cf = self.flag(CF, concrete_bits[CF]);
        let pf = self.flag(PF, concrete_bits[PF]);
        let zf = self.flag(ZF, concrete_bits[ZF]);
        let sf = self.flag(SF, concrete_bits[SF]);
        let of = self.flag(OF, concrete_bits[OF]);
        let base = match n >> 1 {
            0 => of,
            1 => cf,
            2 => zf,
            3 => BoolExpr::or(cf, zf),
            4 => sf,
            5 => pf,
            6 => BoolExpr::xor(sf, of),
            _ => BoolExpr::or(zf, BoolExpr::xor(sf, of)),
        };
        let pred = if n & 1 != 0 { BoolExpr::not(base) } else { base };
        self.record_branch(pred, taken);
    }
}
