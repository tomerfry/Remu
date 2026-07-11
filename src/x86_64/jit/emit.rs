//! Stage-A translator and x86-64 code emitter.
//!
//! Decode-ahead reuses the interpreter's `try_decode`, so the translatable
//! set is a subset of the decoded-instruction cache's hot set. Emission maps
//! each guest instruction to the equivalent host instruction (same ISA);
//! guest status flags ride the host EFLAGS and are materialized into
//! `regs.rflags` only at block exits.

use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynasmApi, DynasmLabelApi, dynasm};

use super::super::OpSize::{O32, O64};
use super::super::decode::{Decoded, DecodedInsn, Op, OpKind};
use super::super::icache::STAMP_SLOTS;
use super::super::{Bus, Cpu};
use super::{
    BlockMeta, JIT_SLOTS, OFF_CYCLES, OFF_GPR, OFF_INVAL, OFF_RFLAGS, OFF_RIP, OFF_STAMPS, Slot,
};

/// Longest block the translator will build.
const MAX_BLOCK_INSNS: usize = 64;

// Per-instruction cycle weights, mirroring `exec_decoded` exactly so JIT'd
// blocks keep `Cpu::cycles` in lockstep with the interpreter.
const CYC_MOV: u32 = 2;
const CYC_INCDEC: u32 = 2;
const CYC_JCC_TAKEN: u32 = 7;
const CYC_JCC_NOT: u32 = 3;
const CYC_JMP: u32 = 7;

/// Guest reg at `[rbp + gpr_off(i)]`.
#[inline]
fn gpr_off(i: u8) -> i32 {
    OFF_GPR + 8 * (i as i32)
}

/// A translated non-terminator body operation.
enum Body {
    /// `MOV reg, imm` (full-register value already resolved).
    StoreImm { reg: u8, val: u64, o64: bool },
    /// `INC`/`DEC reg` at O32/O64.
    IncDec { reg: u8, dec: bool, o64: bool },
}

/// How a block ends.
enum Term {
    /// Conditional branch: guest condition `cc` to `taken`, else `fallthrough`.
    Jcc {
        cc: u8,
        taken: u64,
        fallthrough: u64,
    },
    /// Unconditional relative jump.
    Jmp { target: u64 },
    /// No branch — the block simply runs out and continues at `rip`.
    Fall { rip: u64 },
}

/// A fully-analyzed block ready to emit.
struct Plan {
    body: Vec<Body>,
    term: Term,
    /// The taken/target address equals the block start (internal loop).
    self_loop: bool,
    /// Any body op writes flags (materialize at exits when set).
    flags_dirty: bool,
    /// Defined-flags mask of the last flag-writer (materialize scope).
    defmask: u32,
    ninsns: u32,
    /// Cycle weight of the body ops (terminator added per exit path).
    body_cycles: u32,
    start_rip: u64,
}

impl Cpu {
    /// Decode-ahead one instruction at the current RIP without executing it.
    /// Mirrors `exec_one`'s decode-state reset; returns `None` for a fault,
    /// LOCK/REP, or a non-hot (fused-only) opcode.
    fn jit_decode_one<B: Bus>(&mut self, bus: &mut B) -> Option<DecodedInsn> {
        self.ilen = 0;
        self.seg_override = None;
        self.rep = None;
        self.lock = false;
        self.lock_ok = false;
        self.rex = None;
        self.prefix66 = false;
        self.commit_on_fault = false;
        self.cold_saved = false;
        self.supervisor_override = false;
        self.imm_len = 0;
        self.used_rip_rel = false;
        let (opcode, npfx) = self.scan_prefixes(bus).ok()?;
        if self.lock || self.rep.is_some() {
            return None;
        }
        let mut d = DecodedInsn::default();
        match self.try_decode(bus, opcode, npfx, &mut d) {
            Ok(Decoded::Hot) => Some(d),
            _ => None,
        }
    }

    /// Classify a decoded instruction as a stage-A translatable body op.
    fn stage_a_body(d: &DecodedInsn) -> Option<Body> {
        let o64 = match d.osize() {
            O64 => true,
            O32 => false,
            _ => return None, // O16 not translated in stage A
        };
        match d.op {
            Op::MovRegImmW => Some(Body::StoreImm {
                reg: d.reg,
                val: if o64 { d.imm } else { d.imm as u32 as u64 },
                o64,
            }),
            Op::MovRmImmW if d.kind() == OpKind::Reg => Some(Body::StoreImm {
                reg: d.base,
                val: if o64 { d.imm } else { d.imm as u32 as u64 },
                o64,
            }),
            Op::IncDecRmW if d.kind() == OpKind::Reg => Some(Body::IncDec {
                reg: d.base,
                dec: d.aux != 0,
                o64,
            }),
            _ => None,
        }
    }

    /// Translate the block at the current RIP; returns whether one was
    /// installed. On failure the guest key is remembered as cold.
    pub(super) fn jit_translate<B: Bus>(&mut self, bus: &mut B, key: u64, phys: u64) -> bool {
        let start_rip = self.regs.rip;
        let Some(plan) = self.jit_plan(bus, phys, start_rip) else {
            self.regs.rip = start_rip;
            self.jit.cold.insert(key);
            return false;
        };
        self.regs.rip = start_rip; // decode-ahead advanced it

        let version = self.icache.clock;
        let stamp_slot = (phys >> 12) as usize & (STAMP_SLOTS - 1);
        let entry = emit_block(&mut self.jit.asm, &plan, version, stamp_slot);
        self.jit.asm.commit().expect("commit JIT block");

        let idx = self.jit.blocks.len() as u32;
        self.jit.blocks.push(BlockMeta {
            key,
            version,
            stamp_slot,
            entry,
            start_rip,
            ninsns: plan.ninsns,
        });
        self.jit.table[phys as usize & (JIT_SLOTS - 1)] = Slot { key, version, idx };
        true
    }

    /// Decode-ahead + analysis: build a [`Plan`] or `None` if nothing
    /// translatable starts here.
    fn jit_plan<B: Bus>(&mut self, bus: &mut B, phys: u64, start_rip: u64) -> Option<Plan> {
        let mut body = Vec::new();
        let mut flags_dirty = false;
        let mut defmask = 0u32;
        let mut total_len = 0u64;
        let mut ninsns = 0u32;
        let mut body_cycles = 0u32;
        let term;

        loop {
            if body.len() >= MAX_BLOCK_INSNS {
                term = Term::Fall {
                    rip: start_rip + total_len,
                };
                break;
            }
            // Never cross the physical page (the icache/SMC single-page rule).
            if (phys & 0xFFF) + total_len >= 0x1000 {
                term = Term::Fall {
                    rip: start_rip + total_len,
                };
                break;
            }
            let insn_start = self.regs.rip;
            let Some(d) = self.jit_decode_one(bus) else {
                self.regs.rip = insn_start;
                term = Term::Fall { rip: insn_start };
                break;
            };
            if (phys & 0xFFF) + total_len + d.len as u64 > 0x1000 {
                self.regs.rip = insn_start;
                term = Term::Fall { rip: insn_start };
                break;
            }
            let end = insn_start + d.len as u64;

            // Terminators.
            if d.op == Op::Jcc || d.op == Op::JmpRel {
                // A flag consumer with no in-block producer can't run on host
                // flags — end the block before it.
                if d.op == Op::Jcc && !flags_dirty {
                    self.regs.rip = insn_start;
                    term = Term::Fall { rip: insn_start };
                    break;
                }
                ninsns += 1;
                let target = end.wrapping_add(d.imm as i64 as u64);
                term = if d.op == Op::JmpRel {
                    Term::Jmp { target }
                } else {
                    Term::Jcc {
                        cc: d.aux & 0xF,
                        taken: target,
                        fallthrough: end,
                    }
                };
                break;
            }

            // Non-terminator body op.
            let Some(op) = Self::stage_a_body(&d) else {
                self.regs.rip = insn_start;
                term = Term::Fall { rip: insn_start };
                break;
            };
            body_cycles += match op {
                Body::StoreImm { .. } => CYC_MOV,
                Body::IncDec { .. } => {
                    flags_dirty = true;
                    // INC/DEC define SF ZF AF PF OF (not CF).
                    defmask = 0x8D4;
                    CYC_INCDEC
                }
            };
            body.push(op);
            total_len += d.len as u64;
            ninsns += 1;
        }

        if ninsns == 0 {
            return None;
        }

        let self_loop = match term {
            Term::Jcc { taken, .. } | Term::Jmp { target: taken } => taken == start_rip,
            Term::Fall { .. } => false,
        };
        Some(Plan {
            body,
            term,
            self_loop,
            flags_dirty,
            defmask,
            ninsns,
            body_cycles,
            start_rip,
        })
    }
}

/// Emit one guest condition as a host `Jcc` to `lbl` (same encoding family).
macro_rules! jcc_to {
    ($ops:ident, $cc:expr, $lbl:expr) => {
        match $cc & 0xF {
            0x0 => dynasm!($ops ; .arch x64 ; jo =>$lbl),
            0x1 => dynasm!($ops ; .arch x64 ; jno =>$lbl),
            0x2 => dynasm!($ops ; .arch x64 ; jb =>$lbl),
            0x3 => dynasm!($ops ; .arch x64 ; jae =>$lbl),
            0x4 => dynasm!($ops ; .arch x64 ; jz =>$lbl),
            0x5 => dynasm!($ops ; .arch x64 ; jnz =>$lbl),
            0x6 => dynasm!($ops ; .arch x64 ; jbe =>$lbl),
            0x7 => dynasm!($ops ; .arch x64 ; ja =>$lbl),
            0x8 => dynasm!($ops ; .arch x64 ; js =>$lbl),
            0x9 => dynasm!($ops ; .arch x64 ; jns =>$lbl),
            0xA => dynasm!($ops ; .arch x64 ; jp =>$lbl),
            0xB => dynasm!($ops ; .arch x64 ; jnp =>$lbl),
            0xC => dynasm!($ops ; .arch x64 ; jl =>$lbl),
            0xD => dynasm!($ops ; .arch x64 ; jge =>$lbl),
            0xE => dynasm!($ops ; .arch x64 ; jle =>$lbl),
            _ => dynasm!($ops ; .arch x64 ; jg =>$lbl),
        }
    };
}

/// Emit a body op.
fn emit_body(ops: &mut Assembler, b: &Body) {
    match *b {
        Body::StoreImm { reg, val, o64 } => {
            let off = gpr_off(reg);
            if o64 {
                dynasm!(ops ; .arch x64 ; mov rax, QWORD val as i64 ; mov [rbp + off], rax);
            } else {
                // O32 zero-extends bits 63:32 — store the full zero-extended
                // qword. `mov eax, imm` zero-extends into rax.
                dynasm!(ops ; .arch x64 ; mov eax, DWORD val as i32 ; mov [rbp + off], rax);
            }
        }
        Body::IncDec { reg, dec, o64 } => {
            let off = gpr_off(reg);
            if o64 {
                if dec {
                    dynasm!(ops ; .arch x64 ; dec QWORD [rbp + off]);
                } else {
                    dynasm!(ops ; .arch x64 ; inc QWORD [rbp + off]);
                }
            } else {
                // O32: operate on the dword (sets flags), then zero the upper
                // dword to mirror the guest's 32-bit zero-extension. The zeroing
                // `mov` does not disturb the flags the DEC/INC just set.
                if dec {
                    dynasm!(ops ; .arch x64 ; dec DWORD [rbp + off]);
                } else {
                    dynasm!(ops ; .arch x64 ; inc DWORD [rbp + off]);
                }
                dynasm!(ops ; .arch x64 ; mov DWORD [rbp + off + 4], 0);
            }
        }
    }
}

/// Materialize the live host status flags into `regs.rflags`, touching only
/// the bits in `defmask` (so e.g. INC/DEC leave the guest CF intact).
fn emit_materialize(ops: &mut Assembler, defmask: u32) {
    let low = (defmask & 0xD5) as i32; // CF PF AF ZF SF representable via LAHF
    let of = (defmask & 0x800) as i32; // OF bit, if defined
    let clear = !(defmask & 0x8D5) as i32;
    dynasm!(ops ; .arch x64
        ; lahf                       // AH = SF ZF x AF x PF x CF
        ; seto al                    // AL = OF
        ; movzx ecx, al
        ; shl ecx, 11                // OF << 11
        ; and ecx, DWORD of          // keep OF only if defined
        ; movzx eax, ah
        ; and eax, DWORD low         // keep defined low-byte flags
        ; or eax, ecx
        ; mov ecx, [rbp + OFF_RFLAGS]
        ; and ecx, DWORD clear       // clear the defined bits
        ; or ecx, eax
        ; mov [rbp + OFF_RFLAGS], ecx
    );
}

/// Cycles retired on an exit path: an accumulator register (self-loop) or a
/// compile-time constant (straight block).
enum Cyc {
    /// The self-loop accumulator (`r11`), already holding all retired cycles.
    R11,
    /// A constant total for a straight block.
    Const(u32),
}

/// Emit an exit: materialize flags (if dirty), flush cycles, set RIP, load the
/// retired instruction count, and return.
fn emit_exit(
    ops: &mut Assembler,
    flags_dirty: bool,
    defmask: u32,
    rip: u64,
    retired_r8: bool,
    ninsns: u32,
    cyc: Cyc,
) {
    // Materialize first: it reads the live host flags before the cycle add
    // clobbers them.
    if flags_dirty {
        emit_materialize(ops, defmask);
    }
    match cyc {
        Cyc::R11 => dynasm!(ops ; .arch x64 ; add [rbp + OFF_CYCLES], r11),
        Cyc::Const(c) => dynasm!(ops ; .arch x64 ; add QWORD [rbp + OFF_CYCLES], DWORD c as i32),
    }
    dynasm!(ops ; .arch x64
        ; mov r9, QWORD rip as i64
        ; mov [rbp + OFF_RIP], r9
    );
    if retired_r8 {
        dynasm!(ops ; .arch x64 ; mov rax, r8);
    } else {
        dynasm!(ops ; .arch x64 ; mov eax, DWORD ninsns as i32);
    }
    dynasm!(ops ; .arch x64 ; pop rbp ; ret);
}

/// Emit a whole block and return its entry offset. Calling convention
/// (win64): `fn(cpu: *mut Cpu, budget: u64) -> retired: u64`.
fn emit_block(ops: &mut Assembler, plan: &Plan, version: u64, stamp_slot: usize) -> AssemblyOffset {
    let stale = ops.new_dynamic_label();
    let stamp_off = (stamp_slot * 8) as i32;
    let entry = ops.offset();

    // Prologue: pin rbp, revalidate against the SMC stamp and inval clock.
    dynasm!(ops ; .arch x64
        ; push rbp
        ; mov rbp, rcx
        ; mov rax, QWORD version as i64
        ; mov r10, [rbp + OFF_STAMPS]         // stamps box data pointer
        ; cmp rax, [r10 + stamp_off]
        ; jb =>stale
        ; cmp rax, [rbp + OFF_INVAL]
        ; jb =>stale
    );

    let ninsns = plan.ninsns;

    let bc = plan.body_cycles;
    if plan.self_loop {
        let body_top = ops.new_dynamic_label();
        // retired = 0; cycles = 0; itercap = budget / ninsns (budget >= ninsns
        // is guaranteed by the dispatcher, so itercap >= 1).
        dynasm!(ops ; .arch x64
            ; xor r8d, r8d
            ; xor r11d, r11d
            ; mov rax, rdx
            ; xor edx, edx
            ; mov r9d, DWORD ninsns as i32
            ; div r9
            ; mov rcx, rax
            ; =>body_top
        );
        for b in &plan.body {
            emit_body(ops, b);
        }
        match plan.term {
            Term::Jcc {
                cc, fallthrough, ..
            } => {
                let exit_ft = ops.new_dynamic_label();
                let cyc_iter = (bc + CYC_JCC_TAKEN) as i32;
                let cyc_final = (bc + CYC_JCC_NOT) as i32;
                // Not-taken (inverse condition) exits the loop.
                jcc_to!(ops, cc ^ 1, exit_ft);
                dynasm!(ops ; .arch x64
                    ; lea r8, [r8 + ninsns as i32]
                    ; lea r11, [r11 + cyc_iter]
                    ; loop =>body_top
                );
                // Budget exhausted after a taken branch: RIP = block start.
                emit_exit(
                    ops,
                    plan.flags_dirty,
                    plan.defmask,
                    plan.start_rip,
                    true,
                    ninsns,
                    Cyc::R11,
                );
                dynasm!(ops ; .arch x64
                    ; =>exit_ft
                    ; lea r8, [r8 + ninsns as i32]
                    ; lea r11, [r11 + cyc_final]
                );
                emit_exit(
                    ops,
                    plan.flags_dirty,
                    plan.defmask,
                    fallthrough,
                    true,
                    ninsns,
                    Cyc::R11,
                );
            }
            Term::Jmp { .. } => {
                let cyc_iter = (bc + CYC_JMP) as i32;
                dynasm!(ops ; .arch x64
                    ; lea r8, [r8 + ninsns as i32]
                    ; lea r11, [r11 + cyc_iter]
                    ; loop =>body_top
                );
                emit_exit(
                    ops,
                    plan.flags_dirty,
                    plan.defmask,
                    plan.start_rip,
                    true,
                    ninsns,
                    Cyc::R11,
                );
            }
            Term::Fall { .. } => unreachable!("self_loop implies a branch terminator"),
        }
    } else {
        for b in &plan.body {
            emit_body(ops, b);
        }
        match plan.term {
            Term::Jcc {
                cc,
                taken,
                fallthrough,
            } => {
                let exit_taken = ops.new_dynamic_label();
                jcc_to!(ops, cc, exit_taken);
                let c = Cyc::Const(bc + CYC_JCC_NOT);
                emit_exit(
                    ops,
                    plan.flags_dirty,
                    plan.defmask,
                    fallthrough,
                    false,
                    ninsns,
                    c,
                );
                dynasm!(ops ; .arch x64 ; =>exit_taken);
                let c = Cyc::Const(bc + CYC_JCC_TAKEN);
                emit_exit(ops, plan.flags_dirty, plan.defmask, taken, false, ninsns, c);
            }
            Term::Jmp { target } => {
                let c = Cyc::Const(bc + CYC_JMP);
                emit_exit(
                    ops,
                    plan.flags_dirty,
                    plan.defmask,
                    target,
                    false,
                    ninsns,
                    c,
                );
            }
            Term::Fall { rip } => {
                let c = Cyc::Const(bc);
                emit_exit(ops, plan.flags_dirty, plan.defmask, rip, false, ninsns, c);
            }
        }
    }

    // Stale prologue exit: nothing ran, retired = 0.
    dynasm!(ops ; .arch x64
        ; =>stale
        ; xor eax, eax
        ; pop rbp
        ; ret
    );

    entry
}
