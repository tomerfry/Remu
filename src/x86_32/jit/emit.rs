//! Translator and x86-64 code emitter for the 80386 core's template JIT.
//!
//! Decode-ahead reuses the interpreter's `try_decode`, so the translatable
//! set is a subset of the decoded-instruction cache's hot set. Each 32-bit
//! guest instruction maps to the equivalent 32-bit host instruction (the host
//! is x86-64, a superset of the 386 integer ISA); guest status flags ride the
//! host EFLAGS and are materialized into `regs.eflags` only at block exits.
//!
//! ## Flag exactness
//!
//! Host and guest agree on most status-flag results, but not all: `AND`/`OR`/
//! `XOR` leave AF undefined on the host (the interpreter clears it), multi-bit
//! shifts/rotates leave OF undefined (the interpreter computes a value), and
//! two-operand `IMUL` leaves SF/ZF/PF undefined on the host (the interpreter
//! computes them). A forward flag-state machine tracks, per flag, whether the
//! host EFLAGS currently holds the exact value, a known constant, or garbage.
//! A block may only end where no flag is garbage, and a flag consumer (`Jcc`,
//! `ADC`…) may only run where the flags it reads are exact; otherwise the
//! block is cut short and the tail is interpreted.

use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynasmApi, DynasmLabelApi, dynasm};

use super::super::decode::{Decoded, DecodedInsn, Op, OpKind};
use super::super::icache::STAMP_SLOTS;
use super::super::{Bus, Cpu};
use super::{BlockMeta, JIT_SLOTS, OFF_CYCLES, OFF_EFLAGS, OFF_EIP, OFF_INVAL, OFF_STAMPS, Slot};

/// Longest block the translator will build.
const MAX_BLOCK_INSNS: usize = 64;

// EFLAGS status-bit positions.
const CF: u32 = 0x001;
const PF: u32 = 0x004;
const AF: u32 = 0x010;
const ZF: u32 = 0x040;
const SF: u32 = 0x080;
const OF: u32 = 0x800;
const ALL: u32 = CF | PF | AF | ZF | SF | OF; // 0x8D5

// Per-instruction cycle weights, mirroring `exec_decoded`'s register-form
// counts exactly so JIT'd blocks keep `Cpu::cycles` in lockstep with the
// interpreter. (Only register operands are translated, so the memory-form
// weights never apply.)
const CYC_MOV: u32 = 2;
const CYC_INCDEC: u32 = 2;
const CYC_ALU: u32 = 2;
const CYC_SHIFT: u32 = 3;
const CYC_IMUL: u32 = 20;
const CYC_JCC_TAKEN: u32 = 7;
const CYC_JCC_NOT: u32 = 3;
const CYC_JMP: u32 = 7;

/// Guest reg at `[rbp + gpr_off(i)]` (register file is `[u32; 8]`, stride 4).
#[inline]
fn gpr_off(i: u8) -> i32 {
    super::OFF_GPR + 4 * (i as i32)
}

/// A translated non-terminator body operation. Every operand is 32-bit.
#[derive(Clone, Copy)]
enum Body {
    /// `MOV reg, imm32`.
    StoreImm { reg: u8, val: u32 },
    /// `INC`/`DEC reg`.
    IncDec { reg: u8, dec: bool },
    /// `dst = dst OP src` (register/register ALU); `wb` false means TEST/CMP.
    AluRR { dst: u8, src: u8, aluop: u8, wb: bool },
    /// `dst = dst OP imm32`.
    AluImm { dst: u8, aluop: u8, imm: i32, wb: bool },
    /// Shift/rotate by an immediate count.
    Shift { reg: u8, shop: u8, count: u8 },
    /// `dst = dst * src` (two-operand IMUL).
    Imul { dst: u8, src: u8 },
}

/// Which status flags an operation defines exactly (host == interpreter),
/// forces to a constant, or leaves as host garbage; plus which it reads.
#[derive(Clone, Copy, Default)]
struct FlagEffect {
    exact: u32,
    const0: u32,
    const1: u32,
    garbage: u32,
    reads: u32,
}

impl FlagEffect {
    fn written(&self) -> u32 {
        self.exact | self.const0 | self.const1 | self.garbage
    }
}

/// How a block ends.
#[derive(Clone, Copy)]
enum Term {
    Jcc {
        cc: u8,
        taken: u32,
        fallthrough: u32,
    },
    Jmp {
        target: u32,
    },
    Fall {
        eip: u32,
    },
}

/// A fully-analyzed block ready to emit.
struct Plan {
    body: Vec<Body>,
    term: Term,
    self_loop: bool,
    /// Exit flag materialization: which bits come from the host EFLAGS, and
    /// which are forced to 0/1. Bits in none of these are left unchanged.
    mat_from_host: u32,
    mat_const0: u32,
    mat_const1: u32,
    ninsns: u32,
    body_cycles: u32,
    start_eip: u32,
}

impl Plan {
    fn flags_dirty(&self) -> bool {
        (self.mat_from_host | self.mat_const0 | self.mat_const1) != 0
    }
}

/// One decode-ahead candidate instruction.
struct Cand {
    body: Option<Body>,
    term: Option<Term>,
    effect: FlagEffect,
    cyc: u32,
    end_eip: u32,
}

/// Flags read by a `Jcc` condition code.
fn jcc_reads(cc: u8) -> u32 {
    match cc >> 1 {
        0 => OF,
        1 => CF,
        2 => ZF,
        3 => CF | ZF,
        4 => SF,
        5 => PF,
        6 => SF | OF,
        _ => ZF | SF | OF,
    }
}

/// Flag effect of an ALU op selector (0..8: ADD OR ADC SBB AND SUB XOR CMP).
fn alu_effect(aluop: u8) -> FlagEffect {
    match aluop {
        0 | 5 | 7 => FlagEffect {
            exact: ALL,
            ..Default::default()
        }, // ADD SUB CMP
        2 | 3 => FlagEffect {
            exact: ALL,
            reads: CF,
            ..Default::default()
        }, // ADC SBB
        1 | 4 | 6 => FlagEffect {
            // OR AND XOR (and TEST, encoded as AND): CF/OF = 0, SF/ZF/PF
            // exact, AF cleared.
            exact: CF | OF | SF | ZF | PF,
            const0: AF,
            ..Default::default()
        },
        _ => FlagEffect::default(),
    }
}

impl Cpu {
    /// Decode-ahead one instruction at the current EIP without executing it.
    fn jit_decode_one<B: Bus>(&mut self, bus: &mut B) -> Option<DecodedInsn> {
        self.ilen = 0;
        self.seg_override = None;
        self.rep = None;
        self.lock = false;
        self.lock_ok = false;
        self.commit_on_fault = false;
        self.cold_saved = false;
        self.supervisor_override = false;
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

    /// Classify a decoded instruction as a translatable body op with its flag
    /// effect and cycle weight (terminators are handled separately).
    ///
    /// Only 32-bit operand size is translated (16-bit forms write partial
    /// registers, which the register-slot model does not preserve) and only
    /// register operands (memory operands are a later stage).
    fn classify_body(d: &DecodedInsn) -> Option<(Body, FlagEffect, u32)> {
        if !d.osize32() {
            return None;
        }
        let reg = d.kind() == OpKind::Reg;
        match d.op {
            Op::MovRegImmW => Some((
                Body::StoreImm {
                    reg: d.reg,
                    val: d.imm,
                },
                FlagEffect::default(),
                CYC_MOV,
            )),
            Op::MovRmImmW if reg => Some((
                Body::StoreImm {
                    reg: d.base,
                    val: d.imm,
                },
                FlagEffect::default(),
                CYC_MOV,
            )),
            // Single-byte INC/DEC reg (40–4F).
            Op::IncReg => Some((
                Body::IncDec {
                    reg: d.reg,
                    dec: false,
                },
                FlagEffect {
                    exact: SF | ZF | AF | PF | OF,
                    ..Default::default()
                },
                CYC_INCDEC,
            )),
            Op::DecReg => Some((
                Body::IncDec {
                    reg: d.reg,
                    dec: true,
                },
                FlagEffect {
                    exact: SF | ZF | AF | PF | OF,
                    ..Default::default()
                },
                CYC_INCDEC,
            )),
            // Group FF /0 /1 INC/DEC r/m, register form.
            Op::IncDecRmW if reg => Some((
                Body::IncDec {
                    reg: d.base,
                    dec: d.aux != 0,
                },
                FlagEffect {
                    exact: SF | ZF | AF | PF | OF,
                    ..Default::default()
                },
                CYC_INCDEC,
            )),
            // ALU r/m,r reg-form: dst = rm, src = reg.
            Op::AluRmRW if reg => Some((
                Body::AluRR {
                    dst: d.base,
                    src: d.reg,
                    aluop: d.aux,
                    wb: d.wb(),
                },
                alu_effect(d.aux),
                CYC_ALU,
            )),
            // ALU r,r/m reg-form: dst = reg, src = rm.
            Op::AluRRmW if reg => Some((
                Body::AluRR {
                    dst: d.reg,
                    src: d.base,
                    aluop: d.aux,
                    wb: d.wb(),
                },
                alu_effect(d.aux),
                CYC_ALU,
            )),
            Op::AluAccImmW => Some((
                Body::AluImm {
                    dst: 0,
                    aluop: d.aux,
                    imm: d.imm as i32,
                    wb: d.wb(),
                },
                alu_effect(d.aux),
                CYC_ALU,
            )),
            Op::AluGrpImmW if reg => Some((
                Body::AluImm {
                    dst: d.base,
                    aluop: d.aux,
                    imm: d.imm as i32,
                    wb: d.wb(),
                },
                alu_effect(d.aux),
                CYC_ALU,
            )),
            Op::ShiftRmW if reg && d.aux & 0x80 == 0 => {
                let shop = d.aux & 7;
                if shop == 2 || shop == 3 {
                    return None; // RCL/RCR (carry-chained) not translated yet
                }
                let count = d.imm & 0x1F; // 386 masks the count to 5 bits
                let eff = shift_effect(shop, count);
                Some((
                    Body::Shift {
                        reg: d.base,
                        shop,
                        count: count as u8,
                    },
                    eff,
                    CYC_SHIFT,
                ))
            }
            Op::ImulRRmW if reg => Some((
                Body::Imul {
                    dst: d.reg,
                    src: d.base,
                },
                FlagEffect {
                    exact: CF | OF,
                    const0: AF,
                    garbage: SF | ZF | PF,
                    ..Default::default()
                },
                CYC_IMUL,
            )),
            _ => None,
        }
    }

    /// Translate the block at the current EIP; returns whether one was
    /// installed. On failure the guest key is remembered as cold.
    pub(super) fn jit_translate<B: Bus>(&mut self, bus: &mut B, key: u64, phys: u32) -> bool {
        let start_eip = self.regs.eip;
        let plan = self.jit_plan(bus, phys, start_eip);
        self.regs.eip = start_eip; // decode-ahead advanced it
        let Some(plan) = plan else {
            self.jit_mark_cold(key, phys);
            return false;
        };

        let version = self.icache.clock;
        let stamp_slot = (phys >> 12) as usize & (STAMP_SLOTS - 1);
        let entry = emit_block(&mut self.jit.asm, &plan, version, stamp_slot);
        // A commit failure (e.g. an unencodable relocation) must not crash the
        // emulator: discard the whole cache and interpret this head instead.
        if self.jit.asm.commit().is_err() {
            self.jit.flush();
            self.jit_mark_cold(key, phys);
            return false;
        }

        let idx = self.jit.blocks.len() as u32;
        self.jit.blocks.push(BlockMeta {
            key,
            version,
            stamp_slot,
            entry,
            start_eip,
            ninsns: plan.ninsns,
        });
        self.jit.table[phys as usize & (JIT_SLOTS - 1)] = Slot { key, version, idx };
        true
    }

    /// Decode-ahead + two-pass flag analysis into a [`Plan`].
    fn jit_plan<B: Bus>(&mut self, bus: &mut B, phys: u32, start_eip: u32) -> Option<Plan> {
        // Pass 1: decode-ahead candidates until a terminator, page end, an
        // untranslatable instruction, or the length cap.
        let mut cands: Vec<Cand> = Vec::new();
        let mut total_len = 0u32;
        loop {
            if cands.len() >= MAX_BLOCK_INSNS || (phys & 0xFFF) + total_len >= 0x1000 {
                break;
            }
            let insn_start = self.regs.eip;
            let Some(d) = self.jit_decode_one(bus) else {
                self.regs.eip = insn_start;
                break;
            };
            // Any legacy prefix costs the interpreter one cycle per byte; the
            // emitted block does not model that, so refuse prefixed forms to
            // keep cycles exact. (In a 32-bit segment the hot set is
            // prefix-free anyway.)
            if d.npfx() != 0 || (phys & 0xFFF) + total_len + d.len as u32 > 0x1000 {
                self.regs.eip = insn_start;
                break;
            }
            let end = insn_start.wrapping_add(d.len as u32);
            if d.op == Op::Jcc || d.op == Op::JmpRel {
                let target = end.wrapping_add(d.imm);
                let (term, reads) = if d.op == Op::JmpRel {
                    (Term::Jmp { target }, 0)
                } else {
                    (
                        Term::Jcc {
                            cc: d.aux & 0xF,
                            taken: target,
                            fallthrough: end,
                        },
                        jcc_reads(d.aux & 0xF),
                    )
                };
                let cyc = if d.op == Op::JmpRel { CYC_JMP } else { 0 };
                cands.push(Cand {
                    body: None,
                    term: Some(term),
                    effect: FlagEffect {
                        reads,
                        ..Default::default()
                    },
                    cyc,
                    end_eip: end,
                });
                break;
            }
            let Some((body, effect, cyc)) = Self::classify_body(&d) else {
                self.regs.eip = insn_start;
                break;
            };
            cands.push(Cand {
                body: Some(body),
                term: None,
                effect,
                cyc,
                end_eip: end,
            });
            total_len += d.len as u32;
        }

        // Pass 2: forward flag simulation, taking the longest prefix that ends
        // with no garbage flag and no unsatisfied flag read.
        let (mut exact, mut const0, mut const1, mut garbage) = (0u32, 0u32, 0u32, 0u32);
        let mut cut = 0usize;
        let mut mat = (0u32, 0u32, 0u32);
        let mut exit = Term::Fall { eip: start_eip };
        for (i, c) in cands.iter().enumerate() {
            if c.effect.reads & !exact != 0 {
                break; // reads a non-exact flag
            }
            let w = c.effect.written();
            exact = (exact & !w) | c.effect.exact;
            const0 = (const0 & !w) | c.effect.const0;
            const1 = (const1 & !w) | c.effect.const1;
            garbage = (garbage & !w) | c.effect.garbage;
            if garbage == 0 {
                cut = i + 1;
                mat = (exact, const0, const1);
                exit = c.term.unwrap_or(Term::Fall { eip: c.end_eip });
            }
        }
        if cut == 0 {
            return None;
        }

        let mut body = Vec::new();
        let mut body_cycles = 0u32;
        let mut ninsns = 0u32;
        for c in &cands[..cut] {
            ninsns += 1;
            if let Some(b) = c.body {
                body.push(b);
                body_cycles += c.cyc;
            }
        }
        let self_loop = match exit {
            Term::Jcc { taken, .. } | Term::Jmp { target: taken } => taken == start_eip,
            Term::Fall { .. } => false,
        };
        Some(Plan {
            body,
            term: exit,
            self_loop,
            mat_from_host: mat.0,
            mat_const0: mat.1,
            mat_const1: mat.2,
            ninsns,
            body_cycles,
            start_eip,
        })
    }
}

/// Flag effect of a shift/rotate with a known immediate `count` (already
/// masked to 5 bits).
fn shift_effect(shop: u8, count: u32) -> FlagEffect {
    if count == 0 {
        return FlagEffect::default(); // no flags change
    }
    let of = if count == 1 { OF } else { 0 };
    let of_garbage = if count == 1 { 0 } else { OF };
    match shop {
        4 | 5 => FlagEffect {
            // SHL/SHR: CF/SF/ZF/PF exact, AF cleared, OF exact only at count 1.
            exact: CF | SF | ZF | PF | of,
            const0: AF,
            garbage: of_garbage,
            ..Default::default()
        },
        7 => FlagEffect {
            // SAR: CF/SF/ZF/PF exact, OF and AF cleared.
            exact: CF | SF | ZF | PF,
            const0: AF | OF,
            ..Default::default()
        },
        0 | 1 => FlagEffect {
            // ROL/ROR: CF exact, OF exact only at count 1; SF/ZF/AF/PF kept.
            exact: CF | of,
            garbage: of_garbage,
            ..Default::default()
        },
        _ => FlagEffect {
            garbage: ALL,
            ..Default::default()
        },
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

/// Emit `dst = dst OP src` for the guest register slot at `dst` and the value
/// already in `eax`. `aluop` 7 is CMP and `wb == false` (with a writing
/// selector) is TEST — neither writes the destination.
fn emit_alu(ops: &mut Assembler, aluop: u8, dst: i32, wb: bool) {
    if aluop == 7 {
        dynasm!(ops ; .arch x64 ; cmp DWORD [rbp + dst], eax);
    } else if !wb {
        // AND without write-back is TEST (the only non-writing ALU selector).
        dynasm!(ops ; .arch x64 ; test DWORD [rbp + dst], eax);
    } else {
        match aluop {
            0 => dynasm!(ops ; .arch x64 ; add DWORD [rbp + dst], eax),
            1 => dynasm!(ops ; .arch x64 ; or DWORD [rbp + dst], eax),
            2 => dynasm!(ops ; .arch x64 ; adc DWORD [rbp + dst], eax),
            3 => dynasm!(ops ; .arch x64 ; sbb DWORD [rbp + dst], eax),
            4 => dynasm!(ops ; .arch x64 ; and DWORD [rbp + dst], eax),
            5 => dynasm!(ops ; .arch x64 ; sub DWORD [rbp + dst], eax),
            _ => dynasm!(ops ; .arch x64 ; xor DWORD [rbp + dst], eax),
        }
    }
}

/// Emit `dst = dst OP imm` for a guest register slot.
fn emit_alu_imm(ops: &mut Assembler, aluop: u8, dst: i32, imm: i32, wb: bool) {
    if aluop == 7 {
        dynasm!(ops ; .arch x64 ; cmp DWORD [rbp + dst], imm);
    } else if !wb {
        dynasm!(ops ; .arch x64 ; test DWORD [rbp + dst], imm);
    } else {
        match aluop {
            0 => dynasm!(ops ; .arch x64 ; add DWORD [rbp + dst], imm),
            1 => dynasm!(ops ; .arch x64 ; or DWORD [rbp + dst], imm),
            2 => dynasm!(ops ; .arch x64 ; adc DWORD [rbp + dst], imm),
            3 => dynasm!(ops ; .arch x64 ; sbb DWORD [rbp + dst], imm),
            4 => dynasm!(ops ; .arch x64 ; and DWORD [rbp + dst], imm),
            5 => dynasm!(ops ; .arch x64 ; sub DWORD [rbp + dst], imm),
            _ => dynasm!(ops ; .arch x64 ; xor DWORD [rbp + dst], imm),
        }
    }
}

/// Emit a body op.
fn emit_body(ops: &mut Assembler, b: &Body) {
    match *b {
        Body::StoreImm { reg, val } => {
            let off = gpr_off(reg);
            dynasm!(ops ; .arch x64 ; mov DWORD [rbp + off], DWORD val as i32);
        }
        Body::IncDec { reg, dec } => {
            let off = gpr_off(reg);
            if dec {
                dynasm!(ops ; .arch x64 ; dec DWORD [rbp + off]);
            } else {
                dynasm!(ops ; .arch x64 ; inc DWORD [rbp + off]);
            }
        }
        Body::AluRR {
            dst,
            src,
            aluop,
            wb,
        } => {
            let (d, s) = (gpr_off(dst), gpr_off(src));
            dynasm!(ops ; .arch x64 ; mov eax, [rbp + s]);
            emit_alu(ops, aluop, d, wb);
        }
        Body::AluImm {
            dst,
            aluop,
            imm,
            wb,
        } => {
            emit_alu_imm(ops, aluop, gpr_off(dst), imm, wb);
        }
        Body::Shift { reg, shop, count } => {
            let off = gpr_off(reg);
            let c = count as i8;
            match shop {
                0 => dynasm!(ops ; .arch x64 ; rol DWORD [rbp + off], c),
                1 => dynasm!(ops ; .arch x64 ; ror DWORD [rbp + off], c),
                4 => dynasm!(ops ; .arch x64 ; shl DWORD [rbp + off], c),
                5 => dynasm!(ops ; .arch x64 ; shr DWORD [rbp + off], c),
                _ => dynasm!(ops ; .arch x64 ; sar DWORD [rbp + off], c),
            }
        }
        Body::Imul { dst, src } => {
            let (d, s) = (gpr_off(dst), gpr_off(src));
            dynasm!(ops ; .arch x64
                ; mov eax, [rbp + d]
                ; imul eax, [rbp + s]
                ; mov [rbp + d], eax
            );
        }
    }
}

/// Materialize the live host status flags into `regs.eflags`: `from_host` bits
/// come from EFLAGS, `const0`/`const1` are forced, everything else is kept.
///
/// Uses `LAHF`/`SETO` to read the flags — universal on modern x86-64 (the
/// host this JIT targets).
fn emit_materialize(ops: &mut Assembler, from_host: u32, const0: u32, const1: u32) {
    let clear = (from_host | const0 | const1) as i32;
    if clear == 0 {
        return;
    }
    let low = (from_host & 0xD5) as i32;
    let of = (from_host & OF) as i32;
    let c1 = const1 as i32;
    dynasm!(ops ; .arch x64
        ; lahf
        ; seto al
        ; movzx ecx, al
        ; shl ecx, 11
        ; and ecx, DWORD of
        ; movzx eax, ah
        ; and eax, DWORD low
        ; or eax, ecx
        ; or eax, DWORD c1
        ; mov ecx, [rbp + OFF_EFLAGS]
        ; and ecx, DWORD !clear
        ; or ecx, eax
        ; mov [rbp + OFF_EFLAGS], ecx
    );
}

/// Cycles retired on an exit path.
enum Cyc {
    R11,
    Const(u32),
}

/// Emit an exit: materialize flags, flush cycles, set EIP, load retired count.
fn emit_exit(ops: &mut Assembler, plan: &Plan, eip: u32, retired_r8: bool, ninsns: u32, cyc: Cyc) {
    if plan.flags_dirty() {
        emit_materialize(ops, plan.mat_from_host, plan.mat_const0, plan.mat_const1);
    }
    match cyc {
        Cyc::R11 => dynasm!(ops ; .arch x64 ; add [rbp + OFF_CYCLES], r11),
        Cyc::Const(c) => dynasm!(ops ; .arch x64 ; add QWORD [rbp + OFF_CYCLES], DWORD c as i32),
    }
    dynasm!(ops ; .arch x64
        ; mov DWORD [rbp + OFF_EIP], DWORD eip as i32
    );
    if retired_r8 {
        dynasm!(ops ; .arch x64 ; mov rax, r8);
    } else {
        dynasm!(ops ; .arch x64 ; mov eax, DWORD ninsns as i32);
    }
    dynasm!(ops ; .arch x64 ; pop rbp ; ret);
}

/// Emit a self-loop back-edge that decrements the `loop` counter (rcx) and,
/// while it is nonzero, jumps to `body_top` — via a two-hop bounce so the
/// distance is unbounded. `loop` has only an 8-bit displacement, but the loop
/// body can span far more than 127 bytes; the `loop` here only reaches the
/// adjacent `cont` trampoline, whose near `jmp` reaches any block size. When
/// rcx reaches 0 (budget exhausted) it falls through past `budget_done`.
fn emit_backedge(ops: &mut Assembler, body_top: dynasmrt::DynamicLabel) {
    let cont = ops.new_dynamic_label();
    let budget_done = ops.new_dynamic_label();
    dynasm!(ops ; .arch x64
        ; loop =>cont            // rcx-- ; if rcx != 0 -> cont (adjacent: rel8-safe)
        ; jmp =>budget_done      // rcx == 0 -> fall out to the budget exit
        ; =>cont
        ; jmp =>body_top         // near back-edge, reaches any block size
        ; =>budget_done
    );
}

/// Emit a whole block and return its entry offset. Calling convention
/// (win64): `fn(cpu: *mut Cpu, budget: u64) -> retired: u64`.
fn emit_block(ops: &mut Assembler, plan: &Plan, version: u64, stamp_slot: usize) -> AssemblyOffset {
    let stale = ops.new_dynamic_label();
    let stamp_off = (stamp_slot * 8) as i32;
    let entry = ops.offset();

    dynasm!(ops ; .arch x64
        ; push rbp
        ; mov rbp, rcx
        ; mov rax, QWORD version as i64
        ; mov r10, [rbp + OFF_STAMPS]
        ; cmp rax, [r10 + stamp_off]
        ; jb =>stale
        ; cmp rax, [rbp + OFF_INVAL]
        ; jb =>stale
    );

    let ninsns = plan.ninsns;
    let bc = plan.body_cycles;

    if plan.self_loop {
        let body_top = ops.new_dynamic_label();
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
                jcc_to!(ops, cc ^ 1, exit_ft);
                dynasm!(ops ; .arch x64
                    ; lea r8, [r8 + ninsns as i32]
                    ; lea r11, [r11 + cyc_iter]
                );
                emit_backedge(ops, body_top);
                // rcx == 0: budget exhausted after a taken branch.
                emit_exit(ops, plan, plan.start_eip, true, ninsns, Cyc::R11);
                dynasm!(ops ; .arch x64
                    ; =>exit_ft
                    ; lea r8, [r8 + ninsns as i32]
                    ; lea r11, [r11 + cyc_final]
                );
                emit_exit(ops, plan, fallthrough, true, ninsns, Cyc::R11);
            }
            Term::Jmp { .. } => {
                let cyc_iter = (bc + CYC_JMP) as i32;
                dynasm!(ops ; .arch x64
                    ; lea r8, [r8 + ninsns as i32]
                    ; lea r11, [r11 + cyc_iter]
                );
                emit_backedge(ops, body_top);
                emit_exit(ops, plan, plan.start_eip, true, ninsns, Cyc::R11);
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
                emit_exit(
                    ops,
                    plan,
                    fallthrough,
                    false,
                    ninsns,
                    Cyc::Const(bc + CYC_JCC_NOT),
                );
                dynasm!(ops ; .arch x64 ; =>exit_taken);
                emit_exit(
                    ops,
                    plan,
                    taken,
                    false,
                    ninsns,
                    Cyc::Const(bc + CYC_JCC_TAKEN),
                );
            }
            Term::Jmp { target } => {
                emit_exit(ops, plan, target, false, ninsns, Cyc::Const(bc + CYC_JMP));
            }
            Term::Fall { eip } => {
                emit_exit(ops, plan, eip, false, ninsns, Cyc::Const(bc));
            }
        }
    }

    dynasm!(ops ; .arch x64
        ; =>stale
        ; xor eax, eax
        ; pop rbp
        ; ret
    );

    entry
}
