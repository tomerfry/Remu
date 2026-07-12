//! Decoupled decode for the hot instruction subset — the x86-64 port of
//! `x86_32/decode.rs`; see that file for the design contract. Notable
//! differences:
//!
//! - REX is part of the decode products (`rex` raw byte; register fields
//!   are stored pre-extended) and the *resolved* operand size — including
//!   the default-64 stack/branch promotions — is baked into `flags`, so
//!   arms never call `stack_osize`/`branch_osize`.
//! - RIP-relative EAs store the [`RIP`] sentinel as their base and resolve
//!   as `start_rip + len + disp`: with the total length known, the fused
//!   path's `imm_len` pre-set and `fixup_rip_rel` dance disappears.
//! - Instructions with a `67` prefix in 64-bit mode are never decoded here
//!   (the RIP-relative + 32-bit-EA truncation corner of `F6`/`F7` is not
//!   representable in the formula); they run fused.

use super::OpSize::{self, O16, O32, O64};
use super::execute::ALU8;
use super::modrm::{ModRm, Operand};
use super::registers::reg;
use super::{Bus, Cpu, Exception, Exec};

/// "No register" sentinel for [`DecodedInsn::base`]/[`DecodedInsn::index`].
pub(crate) const NONE: u8 = 0xFF;
/// RIP-relative sentinel for [`DecodedInsn::base`].
pub(crate) const RIP: u8 = 0xFE;

/// Handler discriminant of a decoded instruction (see `x86_32/decode.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Op {
    /// ALU `r/m8, r8` (00 family; TEST 84).
    AluRmR8,
    /// ALU `r/m, r` at the operand size (01 family; TEST 85).
    AluRmRW,
    /// ALU `r8, r/m8` (02 family).
    AluRRm8,
    /// ALU `r, r/m` (03 family).
    AluRRmW,
    /// ALU `AL, imm8` (04 family; TEST A8).
    AluAccImm8,
    /// ALU `eAX/rAX, imm` (05 family; TEST A9).
    AluAccImmW,
    /// Group 80: ALU `r/m8, imm8`.
    AluGrpImm8,
    /// Group 81/83: ALU `r/m, imm` (pre-extended).
    AluGrpImmW,
    /// MOV `r/m8, r8` (88).
    MovRmR8,
    /// MOV `r/m, r` (89).
    MovRmRW,
    /// MOV `r8, r/m8` (8A).
    MovRRm8,
    /// MOV `r, r/m` (8B).
    MovRRmW,
    /// MOV `r8, imm8` (B0–B7).
    MovRegImm8,
    /// MOV `r, imm` (B8–BF; O64 is the one true imm64).
    MovRegImmW,
    /// MOV `r/m8, imm8` (C6 /0).
    MovRmImm8,
    /// MOV `r/m, imm` (C7 /0).
    MovRmImmW,
    /// LEA `r, m` (8D).
    Lea,
    /// INC r (40–47, legacy modes only — REX in 64-bit).
    IncReg,
    /// DEC r (48–4F, legacy modes only).
    DecReg,
    /// PUSH r (50–57; stack size pre-resolved).
    PushReg,
    /// POP r (58–5F).
    PopReg,
    /// Group FE /0 /1: INC/DEC `r/m8` (`aux` = sub-op).
    IncDecRm8,
    /// Group FF /0 /1: INC/DEC `r/m` (`aux` = sub-op).
    IncDecRmW,
    /// Group FF /2: near indirect CALL (branch size pre-resolved).
    CallRm,
    /// Group FF /4: near indirect JMP.
    JmpRm,
    /// Group FF /6: PUSH `r/m` (stack size pre-resolved).
    PushRm,
    /// Shift/rotate group on `r/m8` (C0/D0/D2; `aux` bit 7 = count from CL).
    ShiftRm8,
    /// Shift/rotate group at the operand size (C1/D1/D3).
    ShiftRmW,
    /// Jcc rel (70–7F, 0F 80–8F; `aux` = condition, `imm` = rel).
    Jcc,
    /// JMP rel (E9/EB).
    JmpRel,
    /// CALL rel (E8).
    CallRel,
    /// RET near (C3).
    RetNear,
    /// RET near, imm16 stack release (C2).
    RetNearImm,
    /// IMUL `r, r/m` (0F AF).
    ImulRRmW,
    /// PUSH imm (68/6A; stack size pre-resolved, imm pre-extended).
    PushImm,
    /// IMUL `r, r/m, imm` (69/6B; imm pre-extended).
    ImulRmImm,
    /// MOVSXD `r, r/m32` (63, 64-bit mode only).
    Movsxd,
    /// XCHG `r/m8, r8` (86).
    XchgRm8,
    /// XCHG `r/m, r` (87).
    XchgRmW,
    /// XCHG rAX, r (90–97 with a register bit).
    XchgAcc,
    /// One-byte NOP (90 without REX.B; F3 90 PAUSE never decodes here).
    Nop,
    /// MOVZX `r, r/m8` (0F B6).
    MovzxB,
    /// MOVZX `r, r/m16` (0F B7).
    MovzxW,
    /// MOVSX `r, r/m8` (0F BE).
    MovsxB,
    /// MOVSX `r, r/m16` (0F BF).
    MovsxW,
    /// SETcc `r/m8` (0F 90–9F; `aux` = condition).
    Setcc,
    /// CMOVcc `r, r/m` (0F 40–4F).
    CmovW,
}

/// Operand shape of the decoded ModRM `rm` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpKind {
    None,
    Reg,
    Mem,
}

/// Register-independent decode products of one instruction (24 bytes).
/// Register fields are REX-extended (0–15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodedInsn {
    /// Handler discriminant.
    pub op: Op,
    /// Total instruction length in bytes, prefixes included.
    pub len: u8,
    /// Bits 0–1 the *resolved* [`OpSize`] (stack/branch promotions
    /// included), bit 2 write-back, bits 3–4 [`OpKind`], bits 5–7
    /// legacy-prefix count (REX excluded, matching the fused cycle rule).
    flags: u8,
    /// ModRM `reg` field / `+r` register, REX-extended.
    pub reg: u8,
    /// Op-specific selector (ALU index, sub-op, condition; shift bit 7 =
    /// count from CL).
    pub aux: u8,
    /// EA base register, [`NONE`], or [`RIP`].
    pub base: u8,
    /// EA index register or [`NONE`].
    pub index: u8,
    /// Bits 0–2 resolved segment, bits 3–4 scale, bits 5–7 the raw
    /// override prefix (6 = none).
    segscale: u8,
    /// Raw REX byte (0 = absent) — `reg8` semantics depend on presence.
    pub rex: u8,
    /// Bits 0–1 the effective address size as an [`OpSize`].
    xflags: u8,
    _pad: u16,
    /// Displacement, sign-extended at execution.
    pub disp: i32,
    /// Immediate, pre-extended per the encoding (imm64 for `MOV r64`).
    pub imm: u64,
}

const F_WB: u8 = 1 << 2;
const KIND_SHIFT: u8 = 3;
const NPFX_SHIFT: u8 = 5;
const OVR_NONE: u8 = 6;

fn opsize_bits(s: OpSize) -> u8 {
    match s {
        O16 => 0,
        O32 => 1,
        O64 => 2,
    }
}

fn bits_opsize(b: u8) -> OpSize {
    match b & 3 {
        0 => O16,
        1 => O32,
        _ => O64,
    }
}

impl Default for DecodedInsn {
    fn default() -> Self {
        DecodedInsn {
            op: Op::RetNear, // placeholder; every decode arm overwrites it
            len: 0,
            flags: 0,
            reg: 0,
            aux: 0,
            base: NONE,
            index: NONE,
            segscale: OVR_NONE << 5,
            rex: 0,
            xflags: 0,
            _pad: 0,
            disp: 0,
            imm: 0,
        }
    }
}

impl DecodedInsn {
    /// A blank instruction carrying the current prefix state.
    fn new(osize: OpSize, asize: OpSize, seg_override: Option<u8>, rex: Option<u8>) -> Self {
        DecodedInsn {
            flags: opsize_bits(osize),
            segscale: seg_override.unwrap_or(OVR_NONE) << 5,
            rex: rex.unwrap_or(0),
            xflags: opsize_bits(asize),
            ..DecodedInsn::default()
        }
    }

    #[inline(always)]
    pub fn osize(&self) -> OpSize {
        bits_opsize(self.flags)
    }

    fn set_osize(&mut self, s: OpSize) {
        self.flags = (self.flags & !3) | opsize_bits(s);
    }

    #[inline(always)]
    pub fn asize(&self) -> OpSize {
        bits_opsize(self.xflags)
    }

    #[inline(always)]
    pub fn wb(&self) -> bool {
        self.flags & F_WB != 0
    }

    fn set_wb(&mut self, wb: bool) {
        self.flags |= (wb as u8) << 2;
    }

    #[inline(always)]
    pub fn kind(&self) -> OpKind {
        match (self.flags >> KIND_SHIFT) & 3 {
            0 => OpKind::None,
            1 => OpKind::Reg,
            _ => OpKind::Mem,
        }
    }

    fn set_kind(&mut self, k: OpKind) {
        self.flags = (self.flags & !(3 << KIND_SHIFT)) | (k as u8) << KIND_SHIFT;
    }

    #[inline(always)]
    pub fn npfx(&self) -> u8 {
        self.flags >> NPFX_SHIFT
    }

    fn set_npfx(&mut self, n: u8) {
        self.flags |= n << NPFX_SHIFT;
    }

    /// Resolved segment register of the memory operand.
    #[inline(always)]
    pub fn seg(&self) -> u8 {
        self.segscale & 7
    }

    fn set_seg(&mut self, seg: u8) {
        self.segscale = (self.segscale & !7) | seg;
    }

    /// Index scale (0–3).
    #[inline(always)]
    pub fn scale(&self) -> u8 {
        (self.segscale >> 3) & 3
    }

    fn set_scale(&mut self, s: u8) {
        self.segscale = (self.segscale & !(3 << 3)) | s << 3;
    }

    /// The raw segment-override prefix, for state reconstruction on a hit.
    #[inline(always)]
    pub fn seg_override(&self) -> Option<u8> {
        let o = self.segscale >> 5;
        if o == OVR_NONE { None } else { Some(o) }
    }

    /// The REX prefix, for state reconstruction on a hit.
    #[inline(always)]
    pub fn rex_opt(&self) -> Option<u8> {
        if self.rex == 0 { None } else { Some(self.rex) }
    }
}

/// Outcome of [`Cpu::try_decode`] (see `x86_32/decode.rs`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decoded {
    Hot,
    Cold,
    Cold0F(u8),
}

impl Cpu {
    /// Try to decode the instruction whose opcode byte (already consumed)
    /// is `opcode` into `d`; on failure EIP/ilen rewind to the post-opcode
    /// position and the fused path reproduces the exact fault ordering.
    /// Never called with LOCK/REP prefixes.
    pub(crate) fn try_decode<B: Bus>(
        &mut self,
        bus: &mut B,
        opcode: u8,
        npfx: u32,
        d: &mut DecodedInsn,
    ) -> Exec<Decoded> {
        debug_assert!(!self.lock && self.rep.is_none());
        // 67-prefixed instructions in 64-bit mode stay fused (RIP-relative
        // truncation corners); degenerate prefix runs are un-cacheable.
        if npfx > 7 || (self.m64 && self.asize == O32) {
            return Ok(Decoded::Cold);
        }
        let rip0 = self.regs.rip;
        let ilen0 = self.ilen;
        match self.decode_op(bus, opcode, d) {
            Ok(Decoded::Hot) => {
                d.len = self.ilen;
                d.set_npfx(npfx as u8);
                Ok(Decoded::Hot)
            }
            Ok(c @ Decoded::Cold0F(_)) => Ok(c),
            Ok(Decoded::Cold) | Err(_) => {
                self.regs.rip = rip0;
                self.ilen = ilen0;
                Ok(Decoded::Cold)
            }
        }
    }

    /// The hot-subset decode match (fetches through the normal path).
    fn decode_op<B: Bus>(&mut self, bus: &mut B, opcode: u8, d: &mut DecodedInsn) -> Exec<Decoded> {
        *d = DecodedInsn::new(self.osize, self.asize, self.seg_override, self.rex);
        match opcode {
            // --- ALU: ADD OR ADC SBB AND SUB XOR CMP ------------------------
            0x00 | 0x08 | 0x10 | 0x18 | 0x20 | 0x28 | 0x30 | 0x38 => {
                d.op = Op::AluRmR8;
                d.aux = opcode >> 3;
                d.set_wb(opcode != 0x38);
                self.decode_modrm(bus, d)?;
            }
            0x01 | 0x09 | 0x11 | 0x19 | 0x21 | 0x29 | 0x31 | 0x39 => {
                d.op = Op::AluRmRW;
                d.aux = opcode >> 3;
                d.set_wb(opcode != 0x39);
                self.decode_modrm(bus, d)?;
            }
            0x02 | 0x0A | 0x12 | 0x1A | 0x22 | 0x2A | 0x32 | 0x3A => {
                d.op = Op::AluRRm8;
                d.aux = opcode >> 3;
                d.set_wb(opcode != 0x3A);
                self.decode_modrm(bus, d)?;
            }
            0x03 | 0x0B | 0x13 | 0x1B | 0x23 | 0x2B | 0x33 | 0x3B => {
                d.op = Op::AluRRmW;
                d.aux = opcode >> 3;
                d.set_wb(opcode != 0x3B);
                self.decode_modrm(bus, d)?;
            }
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 | 0x3C => {
                d.op = Op::AluAccImm8;
                d.aux = opcode >> 3;
                d.set_wb(opcode != 0x3C);
                d.imm = self.fetch8(bus)? as u64;
            }
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D => {
                d.op = Op::AluAccImmW;
                d.aux = opcode >> 3;
                d.set_wb(opcode != 0x3D);
                d.imm = self.fetch_imm(bus)?;
            }
            0x80 | 0x82 => {
                if opcode == 0x82 && self.m64 {
                    return Err(Exception::ud());
                }
                d.op = Op::AluGrpImm8;
                let sub = self.decode_modrm(bus, d)?;
                d.aux = sub;
                d.set_wb(sub != 7);
                d.imm = self.fetch8(bus)? as u64;
            }
            0x81 | 0x83 => {
                d.op = Op::AluGrpImmW;
                let sub = self.decode_modrm(bus, d)?;
                d.aux = sub;
                d.set_wb(sub != 7);
                d.imm = if opcode == 0x81 {
                    self.fetch_imm(bus)?
                } else {
                    self.fetch8(bus)? as i8 as i64 as u64
                };
            }

            // --- TEST (ALU AND without write-back) --------------------------
            0x84 => {
                d.op = Op::AluRmR8;
                d.aux = 4;
                self.decode_modrm(bus, d)?;
            }
            0x85 => {
                d.op = Op::AluRmRW;
                d.aux = 4;
                self.decode_modrm(bus, d)?;
            }
            0xA8 => {
                d.op = Op::AluAccImm8;
                d.aux = 4;
                d.imm = self.fetch8(bus)? as u64;
            }
            0xA9 => {
                d.op = Op::AluAccImmW;
                d.aux = 4;
                d.imm = self.fetch_imm(bus)?;
            }

            // --- MOV ---------------------------------------------------------
            0x88 => {
                d.op = Op::MovRmR8;
                self.decode_modrm(bus, d)?;
            }
            0x89 => {
                d.op = Op::MovRmRW;
                self.decode_modrm(bus, d)?;
            }
            0x8A => {
                d.op = Op::MovRRm8;
                self.decode_modrm(bus, d)?;
            }
            0x8B => {
                d.op = Op::MovRRmW;
                self.decode_modrm(bus, d)?;
            }
            0x8D => {
                d.op = Op::Lea;
                self.decode_modrm(bus, d)?;
                if d.kind() != OpKind::Mem {
                    return Err(Exception::ud());
                }
            }
            0xB0..=0xB7 => {
                d.op = Op::MovRegImm8;
                d.reg = (opcode & 7) | self.rex_b() << 3;
                d.imm = self.fetch8(bus)? as u64;
            }
            0xB8..=0xBF => {
                d.op = Op::MovRegImmW;
                d.reg = (opcode & 7) | self.rex_b() << 3;
                d.imm = match self.osize {
                    O16 => self.fetch16(bus)? as u64,
                    O32 => self.fetch32(bus)? as u64,
                    // The one true imm64: MOV r64, imm64.
                    O64 => self.fetch64(bus)?,
                };
            }
            0xC6 => {
                d.op = Op::MovRmImm8;
                if self.decode_modrm(bus, d)? != 0 {
                    return Err(Exception::ud());
                }
                d.imm = self.fetch8(bus)? as u64;
            }
            0xC7 => {
                d.op = Op::MovRmImmW;
                if self.decode_modrm(bus, d)? != 0 {
                    return Err(Exception::ud());
                }
                d.imm = self.fetch_imm(bus)?;
            }

            // --- PUSH imm / IMUL r,r/m,imm / MOVSXD -----------------------------
            0x68 => {
                d.op = Op::PushImm;
                d.set_osize(self.stack_osize());
                // fetch_imm at the resolved stack size (imm32 sign-extends
                // under a 64-bit operand size, as in the fused handler).
                d.imm = match d.osize() {
                    O16 => self.fetch16(bus)? as u64,
                    O32 => self.fetch32(bus)? as u64,
                    O64 => self.fetch32(bus)? as i32 as i64 as u64,
                };
            }
            0x6A => {
                d.op = Op::PushImm;
                d.set_osize(self.stack_osize());
                d.imm = self.fetch8(bus)? as i8 as i64 as u64;
            }
            0x69 | 0x6B => {
                d.op = Op::ImulRmImm;
                self.decode_modrm(bus, d)?;
                d.imm = if opcode == 0x69 {
                    self.fetch_imm(bus)?
                } else {
                    self.fetch8(bus)? as i8 as i64 as u64
                };
            }
            0x63 if self.m64 => {
                d.op = Op::Movsxd;
                self.decode_modrm(bus, d)?;
            }

            // --- XCHG / NOP -----------------------------------------------------
            0x86 => {
                d.op = Op::XchgRm8;
                self.decode_modrm(bus, d)?;
            }
            0x87 => {
                d.op = Op::XchgRmW;
                self.decode_modrm(bus, d)?;
            }
            0x90..=0x97 => {
                if opcode == 0x90 && self.rex_b() == 0 {
                    d.op = Op::Nop;
                } else {
                    d.op = Op::XchgAcc;
                    d.reg = (opcode & 7) | self.rex_b() << 3;
                }
            }

            // --- INC/DEC r (legacy encodings; REX in 64-bit mode) --------------
            0x40..=0x47 => {
                d.op = Op::IncReg;
                d.reg = opcode & 7;
            }
            0x48..=0x4F => {
                d.op = Op::DecReg;
                d.reg = opcode & 7;
            }

            // --- PUSH/POP r (stack size pre-resolved) ---------------------------
            0x50..=0x57 => {
                d.op = Op::PushReg;
                d.reg = (opcode & 7) | self.rex_b() << 3;
                d.set_osize(self.stack_osize());
            }
            0x58..=0x5F => {
                d.op = Op::PopReg;
                d.reg = (opcode & 7) | self.rex_b() << 3;
                d.set_osize(self.stack_osize());
            }

            // --- Groups FE/FF --------------------------------------------------
            0xFE => {
                d.op = Op::IncDecRm8;
                let sub = self.decode_modrm(bus, d)?;
                if sub > 1 {
                    return Err(Exception::ud());
                }
                d.aux = sub;
            }
            0xFF => {
                let sub = self.decode_modrm(bus, d)?;
                d.aux = sub;
                d.op = match sub {
                    0 | 1 => Op::IncDecRmW,
                    2 => {
                        d.set_osize(self.branch_osize());
                        Op::CallRm
                    }
                    4 => {
                        d.set_osize(self.branch_osize());
                        Op::JmpRm
                    }
                    6 => {
                        d.set_osize(self.stack_osize());
                        Op::PushRm
                    }
                    // Far forms (and /7) run fused; the caller rewinds.
                    _ => return Ok(Decoded::Cold),
                };
            }

            // --- Shift/rotate groups -------------------------------------------
            0xC0 => {
                d.op = Op::ShiftRm8;
                d.aux = self.decode_modrm(bus, d)?;
                d.imm = self.fetch8(bus)? as u64;
            }
            0xC1 => {
                d.op = Op::ShiftRmW;
                d.aux = self.decode_modrm(bus, d)?;
                d.imm = self.fetch8(bus)? as u64;
            }
            0xD0 => {
                d.op = Op::ShiftRm8;
                d.aux = self.decode_modrm(bus, d)?;
                d.imm = 1;
            }
            0xD1 => {
                d.op = Op::ShiftRmW;
                d.aux = self.decode_modrm(bus, d)?;
                d.imm = 1;
            }
            0xD2 => {
                d.op = Op::ShiftRm8;
                d.aux = self.decode_modrm(bus, d)? | 0x80;
            }
            0xD3 => {
                d.op = Op::ShiftRmW;
                d.aux = self.decode_modrm(bus, d)? | 0x80;
            }

            // --- Control flow (branch size pre-resolved) ------------------------
            0x70..=0x7F => {
                d.op = Op::Jcc;
                d.aux = opcode & 0xF;
                d.set_osize(self.branch_osize());
                d.imm = self.fetch8(bus)? as i8 as i64 as u64;
            }
            0xE8 => {
                d.op = Op::CallRel;
                d.set_osize(self.branch_osize());
                d.imm = self.decode_rel(bus, d.osize())? as u64;
            }
            0xE9 => {
                d.op = Op::JmpRel;
                d.set_osize(self.branch_osize());
                d.imm = self.decode_rel(bus, d.osize())? as u64;
            }
            0xEB => {
                d.op = Op::JmpRel;
                d.set_osize(self.branch_osize());
                d.imm = self.fetch8(bus)? as i8 as i64 as u64;
            }
            0xC3 => {
                d.op = Op::RetNear;
                d.set_osize(self.branch_osize());
            }
            0xC2 => {
                d.op = Op::RetNearImm;
                d.set_osize(self.branch_osize());
                d.imm = self.fetch16(bus)? as u64;
            }

            // --- Two-byte escape -------------------------------------------------
            0x0F => {
                let op2 = self.fetch8(bus)?;
                match op2 {
                    0x80..=0x8F => {
                        d.op = Op::Jcc;
                        d.aux = op2 & 0xF;
                        d.set_osize(self.branch_osize());
                        d.imm = self.decode_rel(bus, d.osize())? as u64;
                    }
                    0xAF => {
                        d.op = Op::ImulRRmW;
                        self.decode_modrm(bus, d)?;
                    }
                    0xB6 => {
                        d.op = Op::MovzxB;
                        self.decode_modrm(bus, d)?;
                    }
                    0xB7 => {
                        d.op = Op::MovzxW;
                        self.decode_modrm(bus, d)?;
                    }
                    0xBE => {
                        d.op = Op::MovsxB;
                        self.decode_modrm(bus, d)?;
                    }
                    0xBF => {
                        d.op = Op::MovsxW;
                        self.decode_modrm(bus, d)?;
                    }
                    0x90..=0x9F => {
                        d.op = Op::Setcc;
                        d.aux = op2 & 0xF;
                        self.decode_modrm(bus, d)?;
                    }
                    0x40..=0x4F => {
                        d.op = Op::CmovW;
                        d.aux = op2 & 0xF;
                        self.decode_modrm(bus, d)?;
                    }
                    _ => return Ok(Decoded::Cold0F(op2)),
                }
            }

            _ => return Ok(Decoded::Cold),
        }
        Ok(Decoded::Hot)
    }

    /// Fetch a rel16/rel32 branch displacement at the given (resolved
    /// branch) operand size, sign-extended — `fetch_rel` without reading
    /// `self.osize`, which decode does not mutate.
    #[inline]
    fn decode_rel<B: Bus>(&mut self, bus: &mut B, osize: OpSize) -> Exec<i64> {
        if osize == O16 {
            Ok(self.fetch16(bus)? as i16 as i64)
        } else {
            Ok(self.fetch32(bus)? as i32 as i64)
        }
    }

    /// Fetch ModRM (plus SIB/displacement) into `d` as an EA *formula*.
    /// Returns the REX-extended `reg` field (also stored in `d.reg`); for
    /// groups the raw `sub` field is identical to `reg` minus the REX
    /// extension, so callers use the return value masked appropriately.
    fn decode_modrm<B: Bus>(&mut self, bus: &mut B, d: &mut DecodedInsn) -> Exec<u8> {
        self.stat(|s| s.modrm_calls += 1);
        let m = ModRm {
            byte: self.fetch8(bus)?,
            r: self.rex_r() << 3,
            b: self.rex_b() << 3,
        };
        d.reg = m.reg();
        if m.md() == 3 {
            d.set_kind(OpKind::Reg);
            d.base = m.rm();
            return Ok(m.sub());
        }
        d.set_kind(OpKind::Mem);
        if self.asize == O16 {
            self.ea16_formula(bus, m, d)?;
        } else {
            self.ea_wide_formula(bus, m, d)?;
        }
        Ok(m.sub())
    }

    /// 16-bit EA formula (legacy modes; mirrors `ea16`).
    fn ea16_formula<B: Bus>(&mut self, bus: &mut B, m: ModRm, d: &mut DecodedInsn) -> Exec<()> {
        if m.md() == 0 && m.byte & 7 == 6 {
            d.disp = self.fetch16(bus)? as i32;
            d.set_seg(self.seg_or(reg::DS));
            return Ok(());
        }
        let (base, index, seg) = match m.byte & 7 {
            0 => (reg::RBX, reg::RSI, reg::DS),
            1 => (reg::RBX, reg::RDI, reg::DS),
            2 => (reg::RBP, reg::RSI, reg::SS),
            3 => (reg::RBP, reg::RDI, reg::SS),
            4 => (reg::RSI, NONE, reg::DS),
            5 => (reg::RDI, NONE, reg::DS),
            6 => (reg::RBP, NONE, reg::SS),
            _ => (reg::RBX, NONE, reg::DS),
        };
        d.base = base;
        d.index = index;
        // The resolver masks the sum to 16 bits, so a zero-extended 16-bit
        // displacement is congruent to `ea16`'s u16 arithmetic.
        d.disp = match m.md() {
            0 => 0,
            1 => self.fetch8(bus)? as i8 as u16 as i32,
            _ => self.fetch16(bus)? as i32,
        };
        d.set_seg(self.seg_or(seg));
        Ok(())
    }

    /// 32/64-bit EA formula with SIB and RIP-relative (mirrors
    /// `ea_wide`/`sib`; truncation happens in the resolver, which is
    /// congruent because low bits of a sum depend only on low bits of the
    /// addends, including through the index shift).
    fn ea_wide_formula<B: Bus>(&mut self, bus: &mut B, m: ModRm, d: &mut DecodedInsn) -> Exec<()> {
        // mod == 00, rm == 101 (REX.B ignored): disp32 in legacy modes,
        // RIP-relative disp32 in 64-bit mode.
        if m.md() == 0 && m.byte & 7 == 5 {
            d.disp = self.fetch32(bus)? as i32;
            if self.m64 {
                d.base = RIP;
            }
            d.set_seg(self.seg_or(reg::DS));
            return Ok(());
        }

        let seg = if m.byte & 7 == 4 {
            self.sib_formula(bus, m, d)?
        } else {
            d.base = (m.byte & 7) | m.b;
            if m.byte & 7 == reg::RBP {
                reg::SS
            } else {
                reg::DS
            }
        };

        let disp = match m.md() {
            0 => 0,
            1 => self.fetch8(bus)? as i8 as i32,
            _ => self.fetch32(bus)? as i32,
        };
        // The mod==00/base==101 SIB form already parked its disp32 in `d`.
        d.disp = d.disp.wrapping_add(disp);
        d.set_seg(self.seg_or(seg));
        Ok(())
    }

    /// SIB formula; returns the default segment. No 386 scaled-base quirk
    /// on this architecture; `index == 100` without REX.X means no index.
    fn sib_formula<B: Bus>(&mut self, bus: &mut B, m: ModRm, d: &mut DecodedInsn) -> Exec<u8> {
        let sib = self.fetch8(bus)?;
        let (scale, index3, base3) = (sib >> 6, (sib >> 3) & 7, sib & 7);
        let index = index3 | (self.rex_x() << 3);

        let seg = if base3 == 5 && m.md() == 0 {
            d.disp = self.fetch32(bus)? as i32;
            reg::DS
        } else {
            d.base = base3 | m.b;
            if base3 == reg::RSP || base3 == reg::RBP {
                reg::SS
            } else {
                reg::DS
            }
        };

        if index != 4 {
            d.index = index;
            d.set_scale(scale);
        }
        Ok(seg)
    }

    /// Reconstruct the post-decode machine state from a cached instruction
    /// (see `x86_32/decode.rs` for the field-selection rationale). `m64`
    /// was already refreshed by `exec_one` before the probe.
    #[inline(always)]
    pub(crate) fn begin_decoded(&mut self, d: &DecodedInsn) {
        self.seg_override = d.seg_override();
        self.rex = d.rex_opt();
        self.commit_on_fault = false;
        self.cold_saved = false;
        self.supervisor_override = false;
        self.osize = d.osize();
        self.asize = d.asize();
        self.regs.rip = self.start_rip.wrapping_add(d.len as u64);
    }

    /// Debug-build differential for an icache hit (see `x86_32/decode.rs`).
    #[cfg(debug_assertions)]
    pub(crate) fn icache_differential<B: Bus>(&mut self, bus: &mut B, cached: &DecodedInsn) {
        let (rip0, ilen0) = (self.regs.rip, self.ilen);
        self.ilen = 0;
        self.seg_override = None;
        self.rep = None;
        self.lock = false;
        self.lock_ok = false;
        self.rex = None;
        self.m64 = self.mode64();
        let fresh = self.fresh_decode(bus);
        assert_eq!(
            fresh.as_ref(),
            Some(cached),
            "icache: stale entry at rip {:#014x} (missed invalidation or decoder drift)",
            self.start_rip
        );
        self.regs.rip = rip0;
        self.ilen = ilen0;
    }

    /// The full decode an icache hit claims to be equivalent to.
    #[cfg(debug_assertions)]
    fn fresh_decode<B: Bus>(&mut self, bus: &mut B) -> Option<DecodedInsn> {
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

    /// Resolve the cached EA formula into the exact [`Operand`] the fused
    /// `modrm()` would produce, reading live registers.
    #[inline(always)]
    pub(crate) fn ea_operand(&self, d: &DecodedInsn) -> Operand {
        if d.kind() == OpKind::Reg {
            return Operand::Reg(d.base);
        }
        let mut off = d.disp as i64 as u64;
        if d.base == RIP {
            off = off.wrapping_add(self.start_rip.wrapping_add(d.len as u64));
        } else if d.base != NONE {
            off = off.wrapping_add(self.regs.gpr[(d.base & 15) as usize]);
        }
        if d.index != NONE {
            off = off.wrapping_add(self.regs.gpr[(d.index & 15) as usize] << d.scale());
        }
        match d.asize() {
            O16 => off &= 0xFFFF,
            O32 => off &= 0xFFFF_FFFF,
            O64 => {}
        }
        Operand::Mem { seg: d.seg(), off }
    }

    /// Execute a decoded instruction: verbatim fused handler bodies with
    /// the EA formula resolved up front and promoted operand sizes
    /// pre-baked (arms install `self.osize` where the fused arm would call
    /// `branch_osize`/`stack_osize`, so this is correct on both the miss
    /// path — where `self.osize` is still the general size — and on hits).
    #[inline(always)]
    pub(crate) fn exec_decoded<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        match d.op {
            Op::AluRmR8 => {
                let op = self.ea_operand(d);
                let a = self.read_op8(bus, op)?;
                let b = self.gpr8(d.reg);
                let r = ALU8[d.aux as usize](self, a, b);
                if d.wb() {
                    self.write_op8(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            Op::AluRmRW => {
                let op = self.ea_operand(d);
                let a = self.read_op(bus, op)?;
                let b = self.regs.reg64(d.reg);
                let r = self.alu(d.aux as usize, a, b);
                if d.wb() {
                    self.write_op(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            Op::AluRRm8 => {
                let op = self.ea_operand(d);
                let a = self.gpr8(d.reg);
                let b = self.read_op8(bus, op)?;
                let r = ALU8[d.aux as usize](self, a, b);
                if d.wb() {
                    self.set_gpr8(d.reg, r);
                }
                Ok(if op.is_mem() { 6 } else { 2 })
            }
            Op::AluRRmW => {
                let op = self.ea_operand(d);
                let a = match self.osize {
                    O16 => self.regs.reg16(d.reg) as u64,
                    O32 => self.regs.reg32(d.reg) as u64,
                    O64 => self.regs.reg64(d.reg),
                };
                let b = self.read_op(bus, op)?;
                let r = self.alu(d.aux as usize, a, b);
                if d.wb() {
                    self.write_reg_osize(d.reg, r);
                }
                Ok(if op.is_mem() { 6 } else { 2 })
            }
            Op::AluAccImm8 => {
                let r = ALU8[d.aux as usize](self, self.gpr8(0), d.imm as u8);
                if d.wb() {
                    self.set_gpr8(0, r);
                }
                Ok(2)
            }
            Op::AluAccImmW => {
                let a = self.acc();
                let r = self.alu(d.aux as usize, a, d.imm);
                if d.wb() {
                    self.set_acc(r);
                }
                Ok(2)
            }
            Op::AluGrpImm8 => {
                let op = self.ea_operand(d);
                let a = self.read_op8(bus, op)?;
                let r = ALU8[d.aux as usize](self, a, d.imm as u8);
                if d.wb() {
                    self.write_op8(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            Op::AluGrpImmW => {
                let op = self.ea_operand(d);
                let a = self.read_op(bus, op)?;
                let r = self.alu(d.aux as usize, a, d.imm);
                if d.wb() {
                    self.write_op(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            Op::MovRmR8 => {
                let op = self.ea_operand(d);
                let v = self.gpr8(d.reg);
                self.write_op8(bus, op, v)?;
                Ok(2)
            }
            Op::MovRmRW => {
                let op = self.ea_operand(d);
                match self.osize {
                    O16 => {
                        let v = self.regs.reg16(d.reg);
                        self.write_op16(bus, op, v)?;
                    }
                    O32 => {
                        let v = self.regs.reg32(d.reg);
                        self.write_op32(bus, op, v)?;
                    }
                    O64 => {
                        let v = self.regs.reg64(d.reg);
                        self.write_op64(bus, op, v)?;
                    }
                }
                Ok(2)
            }
            Op::MovRRm8 => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)?;
                self.set_gpr8(d.reg, v);
                Ok(if op.is_mem() { 4 } else { 2 })
            }
            Op::MovRRmW => {
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                self.write_reg_osize(d.reg, v);
                Ok(if op.is_mem() { 4 } else { 2 })
            }
            Op::MovRegImm8 => {
                self.set_gpr8(d.reg, d.imm as u8);
                Ok(2)
            }
            Op::MovRegImmW => {
                match self.osize {
                    O16 => self.regs.set_reg16(d.reg, d.imm as u16),
                    O32 => self.regs.set_reg32(d.reg, d.imm as u32),
                    O64 => self.regs.set_reg64(d.reg, d.imm),
                }
                Ok(2)
            }
            Op::MovRmImm8 => {
                let op = self.ea_operand(d);
                self.write_op8(bus, op, d.imm as u8)?;
                Ok(2)
            }
            Op::MovRmImmW => {
                let op = self.ea_operand(d);
                self.write_op(bus, op, d.imm)?;
                Ok(2)
            }
            Op::Lea => {
                let Operand::Mem { off, .. } = self.ea_operand(d) else {
                    return Err(Exception::ud());
                };
                self.write_reg_osize(d.reg, off);
                Ok(2)
            }
            Op::IncReg => {
                let i = d.reg;
                match self.osize {
                    O16 => {
                        let r = self.inc16(self.regs.reg16(i));
                        self.regs.set_reg16(i, r);
                    }
                    _ => {
                        let r = self.inc32(self.regs.reg32(i));
                        self.regs.set_reg32(i, r);
                    }
                }
                Ok(2)
            }
            Op::DecReg => {
                let i = d.reg;
                match self.osize {
                    O16 => {
                        let r = self.dec16(self.regs.reg16(i));
                        self.regs.set_reg16(i, r);
                    }
                    _ => {
                        let r = self.dec32(self.regs.reg32(i));
                        self.regs.set_reg32(i, r);
                    }
                }
                Ok(2)
            }
            Op::PushReg => {
                self.osize = d.osize();
                let v = self.regs.reg64(d.reg);
                self.push(bus, v)?;
                Ok(2)
            }
            Op::PopReg => {
                self.osize = d.osize();
                let v = self.pop(bus)?;
                match self.osize {
                    O16 => self.regs.set_reg16(d.reg, v as u16),
                    O32 => self.regs.set_reg32(d.reg, v as u32),
                    O64 => self.regs.set_reg64(d.reg, v),
                }
                Ok(4)
            }
            Op::IncDecRm8 => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)?;
                let r = if d.aux == 0 {
                    self.inc8(v)
                } else {
                    self.dec8(v)
                };
                self.write_op8(bus, op, r)?;
                Ok(if op.is_mem() { 6 } else { 2 })
            }
            Op::IncDecRmW => {
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                let r = match (self.osize, d.aux) {
                    (O16, 0) => self.inc16(v as u16) as u64,
                    (O16, _) => self.dec16(v as u16) as u64,
                    (O32, 0) => self.inc32(v as u32) as u64,
                    (O32, _) => self.dec32(v as u32) as u64,
                    (O64, 0) => self.inc64(v),
                    (O64, _) => self.dec64(v),
                };
                self.write_op(bus, op, r)?;
                Ok(if op.is_mem() { 6 } else { 2 })
            }
            Op::CallRm => {
                self.osize = d.osize();
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                let ret = self.regs.rip;
                self.push(bus, ret)?;
                self.set_ip(v)?;
                Ok(if op.is_mem() { 10 } else { 7 })
            }
            Op::JmpRm => {
                self.osize = d.osize();
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                self.set_ip(v)?;
                Ok(if op.is_mem() { 10 } else { 7 })
            }
            Op::PushRm => {
                self.osize = d.osize();
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                self.push(bus, v)?;
                Ok(if op.is_mem() { 5 } else { 2 })
            }
            Op::ShiftRm8 => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)?;
                let n = if d.aux & 0x80 != 0 {
                    self.regs.reg8(1, false) as u32
                } else {
                    d.imm as u32
                };
                let r = self.shift_dispatch8(d.aux & 7, v, n);
                self.write_op8(bus, op, r)?;
                Ok(if op.is_mem() { 7 } else { 3 })
            }
            Op::ShiftRmW => {
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                let n = if d.aux & 0x80 != 0 {
                    self.regs.reg8(1, false) as u32
                } else {
                    d.imm as u32
                };
                let r = self.shift_dispatch(d.aux & 7, v, n);
                self.write_op(bus, op, r)?;
                Ok(if op.is_mem() { 7 } else { 3 })
            }
            Op::Jcc => {
                self.osize = d.osize();
                if self.cond(d.aux) {
                    self.jump_rel(d.imm as i64)?;
                    Ok(7)
                } else {
                    Ok(3)
                }
            }
            Op::JmpRel => {
                self.osize = d.osize();
                self.jump_rel(d.imm as i64)?;
                Ok(7)
            }
            Op::CallRel => {
                self.osize = d.osize();
                let ret = self.regs.rip;
                self.push(bus, ret)?;
                self.jump_rel(d.imm as i64)?;
                Ok(7)
            }
            Op::RetNear => {
                self.osize = d.osize();
                let ip = self.pop(bus)?;
                self.set_ip(ip)?;
                Ok(10)
            }
            Op::RetNearImm => {
                self.osize = d.osize();
                let ip = self.pop(bus)?;
                self.set_ip(ip)?;
                self.adjust_sp(d.imm as i64);
                Ok(10)
            }
            Op::ImulRRmW => {
                let op = self.ea_operand(d);
                let b = self.read_op(bus, op)?;
                let a = match self.osize {
                    O16 => self.regs.reg16(d.reg) as u64,
                    O32 => self.regs.reg32(d.reg) as u64,
                    O64 => self.regs.reg64(d.reg),
                };
                match self.osize {
                    O16 => {
                        let r = self.imul_trunc16(a as u16, b as u16);
                        self.regs.set_reg16(d.reg, r);
                    }
                    O32 => {
                        let r = self.imul_trunc32(a as u32, b as u32);
                        self.regs.set_reg32(d.reg, r);
                    }
                    O64 => {
                        let r = self.imul_trunc64(a, b);
                        self.regs.set_reg64(d.reg, r);
                    }
                }
                Ok(if op.is_mem() { 22 } else { 20 })
            }
            // Extended-coverage families live out of line so the hot
            // match stays small (codegen: this match is inlined per site).
            _ => self.exec_decoded_ext(bus, d),
        }
    }

    /// Execution arms for the extended-coverage families — out of line
    /// to keep `exec_decoded`'s inlined footprint bounded as coverage
    /// grows. Only reached for the ops not matched there.
    fn exec_decoded_ext<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        match d.op {
            Op::PushImm => {
                self.osize = d.osize();
                self.push(bus, d.imm)?;
                Ok(2)
            }
            Op::ImulRmImm => {
                let op = self.ea_operand(d);
                let a = self.read_op(bus, op)?;
                let b = d.imm;
                match self.osize {
                    O16 => {
                        let r = self.imul_trunc16(a as u16, b as u16);
                        self.regs.set_reg16(d.reg, r);
                    }
                    O32 => {
                        let r = self.imul_trunc32(a as u32, b as u32);
                        self.regs.set_reg32(d.reg, r);
                    }
                    O64 => {
                        let r = self.imul_trunc64(a, b);
                        self.regs.set_reg64(d.reg, r);
                    }
                }
                Ok(if op.is_mem() { 22 } else { 20 })
            }
            Op::Movsxd => {
                let op = self.ea_operand(d);
                match self.osize {
                    O64 => {
                        let v = self.read_op32(bus, op)? as i32 as i64 as u64;
                        self.regs.set_reg64(d.reg, v);
                    }
                    O32 => {
                        let v = self.read_op32(bus, op)?;
                        self.regs.set_reg32(d.reg, v);
                    }
                    O16 => {
                        let v = self.read_op16(bus, op)?;
                        self.regs.set_reg16(d.reg, v);
                    }
                }
                Ok(2)
            }
            Op::XchgRm8 => {
                let op = self.ea_operand(d);
                let a = self.read_op8(bus, op)?;
                let b = self.gpr8(d.reg);
                self.write_op8(bus, op, b)?;
                self.set_gpr8(d.reg, a);
                Ok(if op.is_mem() { 5 } else { 3 })
            }
            Op::XchgRmW => {
                let op = self.ea_operand(d);
                let a = self.read_op(bus, op)?;
                let b = self.regs.reg64(d.reg);
                match self.osize {
                    O16 => {
                        self.write_op16(bus, op, b as u16)?;
                        self.regs.set_reg16(d.reg, a as u16);
                    }
                    O32 => {
                        self.write_op32(bus, op, b as u32)?;
                        self.regs.set_reg32(d.reg, a as u32);
                    }
                    O64 => {
                        self.write_op64(bus, op, b)?;
                        self.regs.set_reg64(d.reg, a);
                    }
                }
                Ok(if op.is_mem() { 5 } else { 3 })
            }
            Op::XchgAcc => {
                let i = d.reg;
                match self.osize {
                    O16 => {
                        let t = self.regs.reg16(0);
                        let v = self.regs.reg16(i);
                        self.regs.set_reg16(0, v);
                        self.regs.set_reg16(i, t);
                    }
                    O32 => {
                        let t = self.regs.reg32(0);
                        let v = self.regs.reg32(i);
                        self.regs.set_reg32(0, v);
                        self.regs.set_reg32(i, t);
                    }
                    O64 => {
                        self.regs.gpr.swap(0, (i & 15) as usize);
                    }
                }
                Ok(3)
            }
            Op::Nop => Ok(1),
            Op::MovzxB => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)? as u64;
                self.write_reg_osize(d.reg, v);
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            Op::MovzxW => {
                let op = self.ea_operand(d);
                let v = self.read_op16(bus, op)? as u64;
                self.write_reg_osize(d.reg, v);
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            Op::MovsxB => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)? as i8 as i64 as u64;
                self.write_reg_osize(d.reg, v);
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            Op::MovsxW => {
                let op = self.ea_operand(d);
                let v = self.read_op16(bus, op)? as i16 as i64 as u64;
                self.write_reg_osize(d.reg, v);
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            Op::Setcc => {
                let op = self.ea_operand(d);
                let v = self.cond(d.aux) as u8;
                self.write_op8(bus, op, v)?;
                Ok(4)
            }
            Op::CmovW => {
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                // The destination is written even on a false condition (it
                // keeps its value) — in 64-bit mode that still zeroes the
                // upper half of a 32-bit destination.
                let taken = self.cond(d.aux);
                let cur = match self.osize {
                    O16 => self.regs.reg16(d.reg) as u64,
                    O32 => self.regs.reg32(d.reg) as u64,
                    O64 => self.regs.reg64(d.reg),
                };
                self.write_reg_osize(d.reg, if taken { v } else { cur });
                Ok(if op.is_mem() { 5 } else { 4 })
            }
            _ => unreachable!("hot-subset op reached the extended arm"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::OpSize::{O16, O32, O64};
    use super::super::registers::reg;
    use super::super::{Cpu, LinearMemory};
    use super::*;

    /// A CPU in 64-bit long mode with `bytes` at 0x1_0000 and distinctive
    /// register values.
    fn setup64(bytes: &[u8]) -> (Cpu, LinearMemory) {
        let mut mem = LinearMemory::new();
        mem.load(0x1_0000, bytes);
        let mut cpu = Cpu::new();
        cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);
        for i in 0..16 {
            cpu.regs.gpr[i] = 0x0101_0101_0101_0101u64
                .wrapping_mul(i as u64 + 1)
                .wrapping_add(0x0BAD_F00D << (i & 7));
        }
        cpu.regs.gpr[reg::RSP as usize] = 0x20_0000;
        (cpu, mem)
    }

    /// Decode the ModRM stream at RIP both ways under the current prefix
    /// state and compare the resolved operands and consumed lengths.
    fn check_ea_state(cpu: &mut Cpu, mem: &mut LinearMemory) {
        cpu.regs.rip = 0x1_0000;
        cpu.ilen = 0;
        let mut d = DecodedInsn::new(cpu.osize, cpu.asize, cpu.seg_override, cpu.rex);
        // start_rip anchors the RIP-relative resolution; emulate a len-only
        // instruction: start at 0x1_0000 with no opcode bytes before ModRM.
        cpu.start_rip = 0x1_0000;
        let sub = cpu.decode_modrm(mem, &mut d);
        let Ok(sub) = sub else {
            // Both paths must fail identically.
            cpu.regs.rip = 0x1_0000;
            cpu.ilen = 0;
            assert!(cpu.modrm(mem).is_err());
            return;
        };
        d.len = cpu.ilen; // no trailing immediate in these tests
        let got = cpu.ea_operand(&d);
        let (got_rip, got_ilen) = (cpu.regs.rip, cpu.ilen);

        cpu.regs.rip = 0x1_0000;
        cpu.ilen = 0;
        cpu.imm_len = 0; // matches d.len == bytes consumed by ModRM stream
        cpu.used_rip_rel = false;
        let (m, want) = cpu.modrm(mem).unwrap();
        // Fused RIP-relative EAs are relative to the *current* rip (plus
        // imm_len, zero here); the formula resolves via start_rip + len,
        // which is the same address.
        assert_eq!(got, want, "operand mismatch");
        assert_eq!(sub, m.sub(), "sub field mismatch");
        assert_eq!(d.reg, m.reg(), "reg field mismatch");
        assert_eq!(got_rip, cpu.regs.rip, "length mismatch");
        assert_eq!(got_ilen, cpu.ilen, "ilen mismatch");
    }

    /// Every ModRM byte, and every SIB byte under each mod, in 64-bit mode
    /// with every REX combination affecting EAs (B, X, R).
    #[test]
    fn ea_formula_matches_modrm_exhaustive_64() {
        let (mut cpu, mut mem) = setup64(&[]);
        for rex in [
            None,
            Some(0x40u8),
            Some(0x41),
            Some(0x42),
            Some(0x44),
            Some(0x4F),
        ] {
            for modrm in 0..=255u8 {
                let bytes = [modrm, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
                mem.load(0x1_0000, &bytes);
                cpu.rex = rex;
                cpu.m64 = true;
                cpu.osize = O32;
                cpu.asize = O64;
                cpu.seg_override = None;
                check_ea_state(&mut cpu, &mut mem);
            }
            for md in [0x04u8, 0x44, 0x84] {
                for sib in 0..=255u8 {
                    let bytes = [md, sib, 0xEF, 0xBE, 0xAD, 0xDE, 0x77];
                    mem.load(0x1_0000, &bytes);
                    cpu.rex = rex;
                    cpu.m64 = true;
                    cpu.osize = O32;
                    cpu.asize = O64;
                    cpu.seg_override = None;
                    check_ea_state(&mut cpu, &mut mem);
                }
            }
        }
    }

    /// Legacy 16-bit and 32-bit address sizes (compat-style state).
    #[test]
    fn ea_formula_matches_modrm_legacy_sizes() {
        let (mut cpu, mut mem) = setup64(&[]);
        for asize in [O16, O32] {
            for modrm in 0..=255u8 {
                let bytes = [modrm, 0x10, 0x20, 0x30, 0x40, 0x50];
                mem.load(0x1_0000, &bytes);
                cpu.rex = None;
                cpu.m64 = false;
                cpu.osize = O32;
                cpu.asize = asize;
                cpu.seg_override = None;
                check_ea_state(&mut cpu, &mut mem);
            }
        }
    }

    /// The decoupled path and the fused path leave identical state and
    /// cycles, two passes (fills, then pure hits).
    #[test]
    fn decoded_execution_matches_fused_64() {
        #[rustfmt::skip]
        let program: &[u8] = &[
            0x48, 0xC7, 0xC0, 0x34, 0x12, 0x00, 0x00,                   // MOV RAX, 0x1234
            0x48, 0xB9, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x08, // MOV RCX, imm64
            0x48, 0xC7, 0xC3, 0x00, 0x00, 0x03, 0x00,                   // MOV RBX, 0x30000
            0x48, 0x89, 0x03,                                           // MOV [RBX], RAX
            0x48, 0x8B, 0x13,                                           // MOV RDX, [RBX]
            0x48, 0x01, 0xD0,                                           // ADD RAX, RDX
            0x48, 0x83, 0xC0, 0x7B,                                     // ADD RAX, 0x7B
            0x48, 0x81, 0x2B, 0x10, 0x00, 0x00, 0x00,                   // SUB qword [RBX], 0x10
            0x48, 0x31, 0xC2,                                           // XOR RDX, RAX
            0x48, 0x85, 0xC2,                                           // TEST RDX, RAX
            0x48, 0x39, 0xC2,                                           // CMP RDX, RAX
            0x48, 0xFF, 0xC0,                                           // INC RAX
            0x48, 0xFF, 0xCA,                                           // DEC RDX
            0xFE, 0x03,                                                 // INC byte [RBX]
            0x48, 0xFF, 0x0B,                                           // DEC qword [RBX]
            0x50,                                                       // PUSH RAX
            0x5B,                                                       // POP RBX
            0x48, 0xC1, 0xC0, 0x03,                                     // ROL RAX, 3
            0x48, 0xD3, 0xE8,                                           // SHR RAX, CL
            0x48, 0x0F, 0xAF, 0xC2,                                     // IMUL RAX, RDX
            0x48, 0x8D, 0x43, 0x10,                                     // LEA RAX, [RBX+0x10]
            0x8B, 0x05, 0x04, 0x00, 0x00, 0x00,                         // MOV EAX, [RIP+4]
            0x3C, 0x00,                                                 // CMP AL, 0
            0x75, 0x01,                                                 // JNZ +1
            0x90,                                                       // NOP
            0xE8, 0x02, 0x00, 0x00, 0x00,                               // CALL +2
            0xEB, 0x04,                                                 // JMP over target
            0x48, 0xFF, 0xC0,                                           // INC RAX  <- call target
            0xC3,                                                       // RET
            0x68, 0x78, 0x56, 0x00, 0x00,                               // PUSH 0x5678
            0x6A, 0xF6,                                                 // PUSH -10
            0x48, 0x6B, 0xC2, 0x05,                                     // IMUL RAX, RDX, 5
            0x48, 0x63, 0xC8,                                           // MOVSXD RCX, EAX
            0x86, 0x03,                                                 // XCHG [RBX], AL
            0x48, 0x87, 0x13,                                           // XCHG [RBX], RDX
            0x48, 0x91,                                                 // XCHG RAX, RCX
            0x41, 0x90,                                                 // XCHG RAX, R8
            0x0F, 0xB6, 0xC1,                                           // MOVZX EAX, CL
            0x48, 0x0F, 0xBE, 0x0B,                                     // MOVSX RCX, byte [RBX]
            0x0F, 0x94, 0xC2,                                           // SETZ DL
            0x48, 0x0F, 0x44, 0xCA,                                     // CMOVZ RCX, RDX
            0xF4,                                                       // HLT
        ];

        let run = |fused_only: bool| {
            let mut mem = LinearMemory::new();
            mem.load(0x1_0000, program);
            let mut cpu = Cpu::new();
            cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);
            cpu.regs.gpr[reg::RCX as usize] = 5;
            cpu.fused_only = fused_only;
            let mut cycles = Vec::new();
            for _ in 0..2 {
                cpu.regs.rip = 0x1_0000;
                cpu.halted = false;
                for _ in 0..300 {
                    if cpu.halted {
                        break;
                    }
                    cycles.push(cpu.step(&mut mem));
                }
            }
            (cpu.regs, cycles)
        };

        let (regs_a, cycles_a) = run(false);
        let (regs_b, cycles_b) = run(true);
        assert_eq!(regs_a, regs_b, "register state diverged");
        assert_eq!(cycles_a, cycles_b, "cycle counts diverged");
    }
}
