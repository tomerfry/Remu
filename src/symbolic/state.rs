//! The concolic overlay state — a sparse symbolic shadow of the machine.
//!
//! [`SymEngine`] holds a symbolic expression for each register/flag/memory byte
//! that is *tainted* (derived from a symbolic input); everything absent is
//! concrete and read from the real CPU. It is arch-neutral: it speaks only in
//! [`Place`]s (a register index + width, a linear address + width, or an
//! immediate) and bitvector [`Expr`]s, so the per-core glue can bind it without
//! leaking core types in here.
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

/// Status-flag slot indices (the order [`FlagDefs`] and [`SymEngine::flags`]
/// use): carry, parity, aux, zero, sign, overflow.
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
/// Register indices follow the x86 instruction encoding; 8-bit uses the
/// `AL CL DL BL AH CH DH BH` convention (index bit 2 selects the high byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    Reg { idx: u8, width: Width },
    Mem { lin: u32, width: Width },
    /// An immediate — always concrete.
    Imm,
}

impl Place {
    pub fn reg(idx: u8, width: Width) -> Place {
        Place::Reg { idx, width }
    }
    pub fn mem(lin: u32, width: Width) -> Place {
        Place::Mem { lin, width }
    }

    /// The GPR dword slot a register operand lives in (8-bit maps AH..BH back
    /// onto EAX..EBX).
    fn base(idx: u8, width: Width) -> usize {
        if width == 8 {
            (idx & 3) as usize
        } else {
            (idx & 7) as usize
        }
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
    /// Per-GPR 32-bit symbolic value; `None` = concrete.
    regs: [Option<Expr>; 8],
    /// Per-status-flag symbolic value; `None` = concrete (read real EFLAGS).
    flags: [Option<BoolExpr>; 6],
    /// Symbolic memory, one 8-bit expr per linear address; absent = concrete.
    mem: HashMap<u32, Expr>,
    /// Concrete seed value for each input symbol (the concolic witness).
    seed: Model,
    /// Declared inputs, in creation order.
    inputs: Vec<Input>,
    /// Path constraints collected at symbolic branches.
    constraints: Vec<BoolExpr>,
    next_id: u32,
}

impl Default for SymEngine {
    fn default() -> Self {
        SymEngine::new()
    }
}

impl SymEngine {
    pub fn new() -> Self {
        SymEngine {
            enabled: false,
            regs: std::array::from_fn(|_| None),
            flags: std::array::from_fn(|_| None),
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
    /// evaluated under the seed, equals the concrete value the caller passes.
    /// Used by tests and as a fuzzing oracle.
    pub fn check(&self, gpr: &[u32; 8], flags: [bool; 6]) -> bool {
        for (i, slot) in self.regs.iter().enumerate() {
            if let Some(e) = slot
                && e.eval(&self.seed) != gpr[i] as u64
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

    /// Mark a register (whole dword, or a sub-register) symbolic, seeding it
    /// with its current concrete dword value. Returns the fresh symbol id.
    pub fn symbolize_reg(&mut self, idx: u8, width: Width, name: String, concrete_dword: u32) -> SymId {
        let base = Place::base(idx, width);
        let seed_val = sub_value(concrete_dword, idx, width);
        let id = self.fresh(name, width, seed_val as u64);
        let sym = Expr::symbol(id, width);
        let dword = self.compose_reg(base, idx, width, sym, concrete_dword);
        self.regs[base] = Some(dword);
        id
    }

    /// Mark `width/8` memory bytes at `lin` symbolic, seeding from their
    /// current concrete value. Returns the fresh symbol id.
    pub fn symbolize_mem(&mut self, lin: u32, width: Width, name: String, concrete: u64) -> SymId {
        let id = self.fresh(name, width, concrete);
        let sym = Expr::symbol(id, width);
        self.store_mem(lin, width, sym);
        id
    }

    // --- Register shadow --------------------------------------------------------

    /// Build the operand-width symbolic value of a register, self-healing a
    /// stale shadow against the observed `concrete` value.
    fn read_reg(&mut self, idx: u8, width: Width, concrete: u64) -> Expr {
        let base = Place::base(idx, width);
        let Some(dword) = self.regs[base].clone() else {
            return Expr::constant(width, concrete);
        };
        let opnd = match width {
            32 => dword,
            16 => Expr::extract(15, 0, dword),
            8 if idx & 4 != 0 => Expr::extract(15, 8, dword),
            8 => Expr::extract(7, 0, dword),
            _ => unreachable!("bad register width {width}"),
        };
        if opnd.eval(&self.seed) != (concrete & mask(width)) {
            self.regs[base] = None; // stale — an unmodeled write changed it
            return Expr::constant(width, concrete);
        }
        opnd
    }

    /// Overwrite a register's shadow with `value` (operand-width), preserving
    /// the untouched high bits. A fully-concrete result clears the shadow.
    fn write_reg(&mut self, idx: u8, width: Width, value: Expr, concrete_dword: u32) {
        let base = Place::base(idx, width);
        let dword = self.compose_reg(base, idx, width, value, concrete_dword);
        self.regs[base] = if dword.as_const().is_some() { None } else { Some(dword) };
    }

    /// Splice an operand-width `value` into the register's current dword shadow
    /// (or the concrete dword if there is none).
    fn compose_reg(&self, base: usize, idx: u8, width: Width, value: Expr, concrete_dword: u32) -> Expr {
        let cur = || {
            self.regs[base]
                .clone()
                .unwrap_or_else(|| Expr::constant(32, concrete_dword as u64))
        };
        match width {
            32 => value,
            16 => Expr::concat(Expr::extract(31, 16, cur()), value),
            8 if idx & 4 != 0 => {
                let c = cur();
                Expr::concat(Expr::extract(31, 16, c.clone()), Expr::concat(value, Expr::extract(7, 0, c)))
            }
            8 => Expr::concat(Expr::extract(31, 8, cur()), value),
            _ => unreachable!("bad register width {width}"),
        }
    }

    fn reg_symbolic(&self, idx: u8, width: Width) -> bool {
        self.regs[Place::base(idx, width)].is_some()
    }

    // --- Memory shadow ----------------------------------------------------------

    fn read_mem(&mut self, lin: u32, width: Width, concrete: u64) -> Expr {
        let nbytes = (width / 8) as u32;
        if (0..nbytes).all(|k| !self.mem.contains_key(&lin.wrapping_add(k))) {
            return Expr::constant(width, concrete);
        }
        // Little-endian: byte 0 is the low byte. Compose high → low.
        let byte = |k: u32| {
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

    fn store_mem(&mut self, lin: u32, width: Width, value: Expr) {
        let nbytes = (width / 8) as u32;
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

    fn mem_symbolic(&self, lin: u32, width: Width) -> bool {
        (0..(width / 8) as u32).any(|k| self.mem.contains_key(&lin.wrapping_add(k)))
    }

    // --- Places -----------------------------------------------------------------

    fn place_symbolic(&self, place: Place) -> bool {
        match place {
            Place::Reg { idx, width } => self.reg_symbolic(idx, width),
            Place::Mem { lin, width } => self.mem_symbolic(lin, width),
            Place::Imm => false,
        }
    }

    fn read_place(&mut self, place: Place, width: Width, concrete: u64) -> Expr {
        match place {
            Place::Reg { idx, .. } => self.read_reg(idx, width, concrete),
            Place::Mem { lin, .. } => self.read_mem(lin, width, concrete),
            Place::Imm => Expr::constant(width, concrete),
        }
    }

    fn write_place(&mut self, place: Place, value: Expr, concrete_dword: u32) {
        match place {
            Place::Reg { idx, width } => self.write_reg(idx, width, value, concrete_dword),
            Place::Mem { lin, width } => self.store_mem(lin, width, value),
            Place::Imm => {}
        }
    }

    fn concretize(&mut self, place: Place) {
        match place {
            Place::Reg { idx, width } => self.regs[Place::base(idx, width)] = None,
            Place::Mem { lin, width } => {
                for k in 0..(width / 8) as u32 {
                    self.mem.remove(&lin.wrapping_add(k));
                }
            }
            Place::Imm => {}
        }
    }

    // --- Flags ------------------------------------------------------------------

    fn flag_bool(&mut self, i: usize, concrete_bit: bool) -> BoolExpr {
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

    fn put_flag(&mut self, i: usize, d: Option<BoolExpr>) {
        if let Some(e) = d {
            // A folded (concrete) flag clears the shadow so reads see EFLAGS.
            self.flags[i] = if e.as_const().is_some() { None } else { Some(e) };
        }
    }

    fn set_flags(&mut self, d: FlagDefs) {
        self.put_flag(CF, d.cf);
        self.put_flag(PF, d.pf);
        self.put_flag(AF, d.af);
        self.put_flag(ZF, d.zf);
        self.put_flag(SF, d.sf);
        self.put_flag(OF, d.of);
    }

    // --- Hooks (called by the per-core glue) ------------------------------------

    /// Symbolic side of an 8-op ALU instruction. The concrete side has already
    /// run (setting EFLAGS and, if `dst`, the destination), so this only builds
    /// the shadow. `dst_dword` is the concrete dword of a register destination
    /// (for splicing a sub-register result); `cin_bit` is the concrete CF.
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
        dst_dword: u32,
        cin_bit: bool,
    ) {
        let needs_carry = idx == 2 || idx == 3; // ADC / SBB
        let any_sym =
            self.place_symbolic(a) || self.place_symbolic(b) || (needs_carry && self.flags[CF].is_some());
        if !any_sym {
            // Fully concrete: the concrete op is authoritative. Concretize the
            // destination and the flags this op defines (all six for the table).
            if let Some(d) = dst {
                self.concretize(d);
            }
            self.flags = std::array::from_fn(|_| None);
            return;
        }
        let ea = self.read_place(a, width, av);
        let eb = self.read_place(b, width, bv);
        let cin = self.flag_bool(CF, cin_bit);
        let (res, defs) = alu::alu(idx, &ea, &eb, &cin, width);
        debug_assert_eq!(res.eval(&self.seed), rv & mask(width), "symbolic ALU diverged from concrete");
        self.set_flags(defs);
        if let Some(d) = dst {
            self.write_place(d, res, dst_dword);
        }
    }

    /// Symbolic side of a flag-setting unary op (`INC`/`DEC`/`NEG`). Concrete
    /// operands fold, so this handles both the tainted and concrete cases
    /// (`INC`/`DEC` correctly leave `CF` untouched).
    pub fn unary(&mut self, op: UnaryOp, place: Place, width: Width, av: u64, rv: u64, dst_dword: u32) {
        let e = self.read_place(place, width, av);
        let (res, defs) = match op {
            UnaryOp::Inc => alu::inc(&e, width),
            UnaryOp::Dec => alu::dec(&e, width),
            UnaryOp::Neg => alu::neg(&e, width),
        };
        debug_assert_eq!(res.eval(&self.seed), rv & mask(width), "symbolic unary diverged");
        self.set_flags(defs);
        self.write_place(place, res, dst_dword);
    }

    /// Drop a register's shadow (e.g. after a `MOV reg, imm` writes a concrete
    /// value over it).
    pub fn concretize_reg(&mut self, idx: u8, width: Width) {
        self.concretize(Place::reg(idx, width));
    }

    /// Symbolic side of a shift/rotate. `sub` is the group-2 sub-op
    /// (0=ROL 1=ROR 2=RCL 3=RCR 4/6=SHL 5=SHR 7=SAR); the count `n` is taken
    /// **concretely** from the run (a symbolic `CL` is concretized to its seed
    /// value — a standard concolic simplification). `RCL`/`RCR` aren't modeled,
    /// so they concretize the destination.
    #[allow(clippy::too_many_arguments)]
    pub fn shift(&mut self, sub: u8, place: Place, width: Width, n: u32, vv: u64, rv: u64, dst_dword: u32) {
        let kind = match sub {
            0 => alu::ShiftKind::Rol,
            1 => alu::ShiftKind::Ror,
            4 | 6 => alu::ShiftKind::Shl,
            5 => alu::ShiftKind::Shr,
            7 => alu::ShiftKind::Sar,
            _ => {
                // RCL/RCR: not modeled — drop the shadow, clear the flags they
                // define (CF/OF), which are now concrete.
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
        self.write_place(place, res, dst_dword);
    }

    /// Symbolic side of a `MOV dst, src`: propagate the source shadow, or
    /// concretize `dst` when the source is concrete. `width` is the move width;
    /// `val` the concrete moved value; `dst_dword` the destination register's
    /// concrete dword (for sub-register splices).
    pub fn mov(&mut self, dst: Place, src: Place, width: Width, val: u64, dst_dword: u32) {
        if !self.place_symbolic(src) {
            self.concretize(dst);
            return;
        }
        let e = self.read_place(src, width, val);
        self.write_place(dst, e, dst_dword);
    }

    /// Symbolic side of a conditional branch. Records the path constraint when
    /// the deciding flags are symbolic; `concrete_bits` are the six live EFLAGS
    /// bits (CF PF AF ZF SF OF) and `taken` the concrete decision.
    pub fn branch(&mut self, n: u8, taken: bool, concrete_bits: [bool; 6]) {
        let cf = self.flag_bool(CF, concrete_bits[CF]);
        let pf = self.flag_bool(PF, concrete_bits[PF]);
        let zf = self.flag_bool(ZF, concrete_bits[ZF]);
        let sf = self.flag_bool(SF, concrete_bits[SF]);
        let of = self.flag_bool(OF, concrete_bits[OF]);
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
        if pred.as_const().is_some() {
            return; // condition fully concrete — not a symbolic branch
        }
        debug_assert_eq!(pred.eval(&self.seed), taken, "branch predicate diverged from concrete");
        self.constraints.push(if taken { pred } else { BoolExpr::not(pred) });
    }
}

/// Extract the operand-width value of a sub-register from its dword.
fn sub_value(dword: u32, idx: u8, width: Width) -> u32 {
    match width {
        32 => dword,
        16 => dword & 0xFFFF,
        8 if idx & 4 != 0 => (dword >> 8) & 0xFF,
        8 => dword & 0xFF,
        _ => unreachable!("bad register width {width}"),
    }
}
