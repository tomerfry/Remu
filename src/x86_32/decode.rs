//! Decoupled decode for the hot instruction subset: [`Cpu::try_decode`]
//! turns an instruction's bytes into a register-independent [`DecodedInsn`]
//! (prefix effects, handler discriminant, EA *formula*, immediate, length),
//! and [`Cpu::exec_decoded`] executes one. The decoded-instruction cache
//! stores these so a hot loop decodes each instruction once, not once per
//! iteration.
//!
//! Execution arms are verbatim copies of the fused handler bodies in
//! `execute.rs`/`execute_0f.rs` with `modrm()` replaced by
//! [`Cpu::ea_operand`], which re-evaluates the cached EA formula from live
//! registers into the identical [`Operand`] — everything downstream
//! (`read_op*`/`write_op*`, the ALU tables, `jump_rel`/`set_ip`, the stack
//! helpers) is reused unchanged, including the cycle formulas.
//!
//! Exactness rule: if decoding fails for *any* reason (a fetch faults, an
//! undefined encoding), the decoder rewinds to the post-opcode position and
//! the caller falls back to the fused path, which re-decodes and faults in
//! its exact historical order. Only fully-decoded instructions execute (and
//! later: cache) through this path. LOCK- and REP-prefixed instructions
//! never enter it.

use super::execute::{ALU8, ALU16, ALU32};
use super::modrm::{ModRm, Operand};
use super::registers::reg;
use super::{Bus, Cpu, Exception, Exec};

/// "No register" sentinel for [`DecodedInsn::base`]/[`DecodedInsn::index`].
pub(crate) const NONE: u8 = 0xFF;

/// Handler discriminant of a decoded instruction. One variant per fused
/// dispatch-arm shape; bodies in [`Cpu::exec_decoded`] mirror `execute.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Op {
    /// ALU `r/m8, r8` (00 family; TEST 84 as AND without write-back).
    AluRmR8,
    /// ALU `r/m, r` at the operand size (01 family; TEST 85).
    AluRmRW,
    /// ALU `r8, r/m8` (02 family).
    AluRRm8,
    /// ALU `r, r/m` (03 family).
    AluRRmW,
    /// ALU `AL, imm8` (04 family; TEST A8).
    AluAccImm8,
    /// ALU `eAX, imm` (05 family; TEST A9).
    AluAccImmW,
    /// Group 80/82: ALU `r/m8, imm8`.
    AluGrpImm8,
    /// Group 81/83: ALU `r/m, imm` (83's imm8 pre-sign-extended).
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
    /// MOV `r, imm` (B8–BF).
    MovRegImmW,
    /// MOV `r/m8, imm8` (C6 /0).
    MovRmImm8,
    /// MOV `r/m, imm` (C7 /0).
    MovRmImmW,
    /// LEA `r, m` (8D).
    Lea,
    /// INC r (40–47).
    IncReg,
    /// DEC r (48–4F).
    DecReg,
    /// PUSH r (50–57).
    PushReg,
    /// POP r (58–5F).
    PopReg,
    /// Group FE /0 /1: INC/DEC `r/m8` (`aux` = sub-op).
    IncDecRm8,
    /// Group FF /0 /1: INC/DEC `r/m` (`aux` = sub-op).
    IncDecRmW,
    /// Group FF /2: near indirect CALL.
    CallRm,
    /// Group FF /4: near indirect JMP.
    JmpRm,
    /// Group FF /6: PUSH `r/m`.
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
    /// PUSH imm (68/6A; imm pre-extended).
    PushImm,
    /// IMUL `r, r/m, imm` (69/6B; imm pre-extended).
    ImulRmImm,
    /// XCHG `r/m8, r8` (86).
    XchgRm8,
    /// XCHG `r/m, r` (87).
    XchgRmW,
    /// XCHG eAX, r (90–97).
    XchgAcc,
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
    /// CMOVcc `r, r/m` (0F 40–4F, `extensions` only — which is a context
    /// key bit, so a cached entry can never outlive the setting).
    CmovW,
}

/// Operand shape of the decoded ModRM `rm` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpKind {
    None,
    Reg,
    Mem,
}

/// The register-independent decode products of one instruction: everything
/// the prefix scan, opcode match and ModRM/SIB/displacement/immediate
/// fetches produce — but *not* anything read from a register. 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodedInsn {
    /// Handler discriminant.
    pub op: Op,
    /// Total instruction length in bytes, prefixes included.
    pub len: u8,
    /// Bit 0 `osize32`, bit 1 `asize32`, bit 2 write-back, bits 3–4
    /// [`OpKind`], bits 5–7 legacy-prefix count (cycle parity).
    flags: u8,
    /// ModRM `reg` field, or the `+r` register of a short encoding.
    pub reg: u8,
    /// Op-specific selector: ALU-table index, group sub-op, Jcc condition,
    /// shift sub-op (bit 7 = count comes from CL at execution).
    pub aux: u8,
    /// EA base register, or [`NONE`].
    pub base: u8,
    /// EA index register, or [`NONE`].
    pub index: u8,
    /// Bits 0–2 resolved segment (override folded in), bits 3–4 scale,
    /// bits 5–7 the raw override prefix (6 = none) for state reconstruction.
    segscale: u8,
    /// Displacement (sign-extended for 32-bit EAs; the 16-bit EA formula
    /// masks the sum, so zero-extension of a 16-bit disp is equivalent).
    pub disp: u32,
    /// Immediate, pre-extended per the encoding.
    pub imm: u32,
}

const F_OSIZE32: u8 = 1 << 0;
const F_ASIZE32: u8 = 1 << 1;
const F_WB: u8 = 1 << 2;
const KIND_SHIFT: u8 = 3;
const NPFX_SHIFT: u8 = 5;
/// `seg_override` encoding in bits 5–7 of `segscale`: 0–5 = segment, 6 = none.
const OVR_NONE: u8 = 6;

impl Default for DecodedInsn {
    /// A blank slot for the decoder to fill (see [`Cpu::try_decode`]).
    fn default() -> Self {
        DecodedInsn::new(false, false, None)
    }
}

impl DecodedInsn {
    /// A blank instruction carrying the current prefix state; the decoder
    /// fills in the rest.
    fn new(osize32: bool, asize32: bool, seg_override: Option<u8>) -> Self {
        DecodedInsn {
            op: Op::RetNear, // placeholder; every decode arm overwrites it
            len: 0,
            flags: (osize32 as u8) | (asize32 as u8) << 1,
            reg: 0,
            aux: 0,
            base: NONE,
            index: NONE,
            segscale: seg_override.unwrap_or(OVR_NONE) << 5,
            disp: 0,
            imm: 0,
        }
    }

    #[inline(always)]
    pub fn osize32(&self) -> bool {
        self.flags & F_OSIZE32 != 0
    }

    #[inline(always)]
    pub fn asize32(&self) -> bool {
        self.flags & F_ASIZE32 != 0
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

    /// The raw segment-override prefix this instruction carried, for
    /// decoder-state reconstruction on a cache hit.
    #[inline(always)]
    pub fn seg_override(&self) -> Option<u8> {
        let o = self.segscale >> 5;
        if o == OVR_NONE { None } else { Some(o) }
    }
}

/// Outcome of [`Cpu::try_decode`]; the decoded instruction itself is
/// written through the out-parameter (kept out of the return value so it
/// never moves through memory on the hot path).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decoded {
    /// Fully decoded into the out-parameter; all instruction bytes consumed.
    Hot,
    /// Not in the hot subset (or decode failed); nothing consumed beyond
    /// the opcode — run the fused dispatch.
    Cold,
    /// A `0F` opcode outside the subset; the second byte is consumed —
    /// hand it to `dispatch_0f` as the fused `0F` arm would.
    Cold0F(u8),
}

impl Cpu {
    /// Try to decode the instruction whose opcode byte (already consumed)
    /// is `opcode` into `d`. On any failure — a fetch fault or an
    /// undefined encoding — EIP/ilen rewind to the post-opcode position
    /// and the caller must run the fused path, which reproduces the exact
    /// fault ordering. Never called with LOCK/REP prefixes.
    pub(crate) fn try_decode<B: Bus>(
        &mut self,
        bus: &mut B,
        opcode: u8,
        npfx: u32,
        d: &mut DecodedInsn,
    ) -> Exec<Decoded> {
        debug_assert!(!self.lock && self.rep.is_none());
        if npfx > 7 {
            return Ok(Decoded::Cold); // degenerate; also un-cacheable
        }
        let eip0 = self.regs.eip;
        let ilen0 = self.ilen;
        match self.decode_op(bus, opcode, d) {
            Ok(Decoded::Hot) => {
                d.len = self.ilen;
                d.set_npfx(npfx as u8);
                Ok(Decoded::Hot)
            }
            Ok(c @ Decoded::Cold0F(_)) => Ok(c),
            // Cold may have consumed bytes (e.g. the ModRM of a group whose
            // sub-op runs fused); rewinding is a no-op when it didn't.
            Ok(Decoded::Cold) | Err(_) => {
                self.regs.eip = eip0;
                self.ilen = ilen0;
                Ok(Decoded::Cold)
            }
        }
    }

    /// The hot-subset decode match. Consumes instruction bytes via the
    /// normal fetch path (15-byte limit, CS limit checks, paging).
    fn decode_op<B: Bus>(&mut self, bus: &mut B, opcode: u8, d: &mut DecodedInsn) -> Exec<Decoded> {
        *d = DecodedInsn::new(self.osize32, self.asize32, self.seg_override);
        let d = &mut *d;
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
                d.imm = self.fetch8(bus)? as u32;
            }
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D => {
                d.op = Op::AluAccImmW;
                d.aux = opcode >> 3;
                d.set_wb(opcode != 0x3D);
                d.imm = self.fetch_imm(bus)?;
            }
            0x80 | 0x82 => {
                d.op = Op::AluGrpImm8;
                let sub = self.decode_modrm(bus, d)?;
                d.aux = sub;
                d.set_wb(sub != 7);
                d.imm = self.fetch8(bus)? as u32;
            }
            0x81 | 0x83 => {
                d.op = Op::AluGrpImmW;
                let sub = self.decode_modrm(bus, d)?;
                d.aux = sub;
                d.set_wb(sub != 7);
                d.imm = if opcode == 0x81 {
                    self.fetch_imm(bus)?
                } else {
                    self.fetch8(bus)? as i8 as i32 as u32
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
                d.imm = self.fetch8(bus)? as u32;
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
                d.reg = opcode & 7;
                d.imm = self.fetch8(bus)? as u32;
            }
            0xB8..=0xBF => {
                d.op = Op::MovRegImmW;
                d.reg = opcode & 7;
                d.imm = self.fetch_imm(bus)?;
            }
            0xC6 => {
                d.op = Op::MovRmImm8;
                if self.decode_modrm(bus, d)? != 0 {
                    return Err(Exception::ud());
                }
                d.imm = self.fetch8(bus)? as u32;
            }
            0xC7 => {
                d.op = Op::MovRmImmW;
                if self.decode_modrm(bus, d)? != 0 {
                    return Err(Exception::ud());
                }
                d.imm = self.fetch_imm(bus)?;
            }

            // --- PUSH imm / IMUL r,r/m,imm --------------------------------------
            0x68 => {
                d.op = Op::PushImm;
                d.imm = self.fetch_imm(bus)?;
            }
            0x6A => {
                d.op = Op::PushImm;
                d.imm = self.fetch8(bus)? as i8 as i32 as u32;
            }
            0x69 | 0x6B => {
                d.op = Op::ImulRmImm;
                self.decode_modrm(bus, d)?;
                d.imm = if opcode == 0x69 {
                    self.fetch_imm(bus)?
                } else {
                    self.fetch8(bus)? as i8 as i32 as u32
                };
            }

            // --- XCHG -----------------------------------------------------------
            0x86 => {
                d.op = Op::XchgRm8;
                self.decode_modrm(bus, d)?;
            }
            0x87 => {
                d.op = Op::XchgRmW;
                self.decode_modrm(bus, d)?;
            }
            0x90..=0x97 => {
                d.op = Op::XchgAcc;
                d.reg = opcode & 7;
            }

            // --- INC/DEC/PUSH/POP r -------------------------------------------
            0x40..=0x47 => {
                d.op = Op::IncReg;
                d.reg = opcode & 7;
            }
            0x48..=0x4F => {
                d.op = Op::DecReg;
                d.reg = opcode & 7;
            }
            0x50..=0x57 => {
                d.op = Op::PushReg;
                d.reg = opcode & 7;
            }
            0x58..=0x5F => {
                d.op = Op::PopReg;
                d.reg = opcode & 7;
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
                    2 => Op::CallRm,
                    4 => Op::JmpRm,
                    6 => Op::PushRm,
                    // Far forms (and the undefined /7) run fused; the caller
                    // rewinds the ModRM bytes consumed here.
                    _ => return Ok(Decoded::Cold),
                };
            }

            // --- Shift/rotate groups -------------------------------------------
            0xC0 => {
                d.op = Op::ShiftRm8;
                d.aux = self.decode_modrm(bus, d)?;
                d.imm = self.fetch8(bus)? as u32;
            }
            0xC1 => {
                d.op = Op::ShiftRmW;
                d.aux = self.decode_modrm(bus, d)?;
                d.imm = self.fetch8(bus)? as u32;
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

            // --- Control flow ---------------------------------------------------
            0x70..=0x7F => {
                d.op = Op::Jcc;
                d.aux = opcode & 0xF;
                d.imm = self.fetch8(bus)? as i8 as i32 as u32;
            }
            0xE8 => {
                d.op = Op::CallRel;
                d.imm = self.fetch_rel(bus)?;
            }
            0xE9 => {
                d.op = Op::JmpRel;
                d.imm = self.fetch_rel(bus)?;
            }
            0xEB => {
                d.op = Op::JmpRel;
                d.imm = self.fetch8(bus)? as i8 as i32 as u32;
            }
            0xC3 => d.op = Op::RetNear,
            0xC2 => {
                d.op = Op::RetNearImm;
                d.imm = self.fetch16(bus)? as u32;
            }

            // --- Two-byte escape -------------------------------------------------
            0x0F => {
                let op2 = self.fetch8(bus)?;
                match op2 {
                    0x80..=0x8F => {
                        d.op = Op::Jcc;
                        d.aux = op2 & 0xF;
                        d.imm = self.fetch_rel(bus)?;
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
                    0x40..=0x4F if self.extensions => {
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

    /// Fetch a rel16/rel32 branch displacement, sign-extended.
    #[inline]
    fn fetch_rel<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        Ok(if self.osize32 {
            self.fetch32(bus)?
        } else {
            self.fetch16(bus)? as i16 as i32 as u32
        })
    }

    /// Fetch ModRM (plus SIB/displacement) into `d` as an EA *formula* —
    /// register indices, scale and displacement, no register reads. Returns
    /// the ModRM `reg` field (also stored in `d.reg`). The byte order and
    /// fetch faults match [`Cpu::modrm`] exactly; equivalence of the
    /// resolved operand is asserted by this module's tests.
    fn decode_modrm<B: Bus>(&mut self, bus: &mut B, d: &mut DecodedInsn) -> Exec<u8> {
        self.stat(|s| s.modrm_calls += 1);
        let m = ModRm(self.fetch8(bus)?);
        d.reg = m.reg();
        if m.md() == 3 {
            d.set_kind(OpKind::Reg);
            d.base = m.rm();
            return Ok(m.reg());
        }
        d.set_kind(OpKind::Mem);
        if self.asize32 {
            self.ea32_formula(bus, m, d)?;
        } else {
            self.ea16_formula(bus, m, d)?;
        }
        Ok(m.reg())
    }

    /// 16-bit EA formula (mirrors `ea16`).
    fn ea16_formula<B: Bus>(&mut self, bus: &mut B, m: ModRm, d: &mut DecodedInsn) -> Exec<()> {
        // Direct address: mod == 00, rm == 110 is a bare disp16 (no BP).
        if m.md() == 0 && m.rm() == 6 {
            d.disp = self.fetch16(bus)? as u32;
            d.set_seg(self.seg_or(reg::DS));
            return Ok(());
        }

        let (base, index, seg) = match m.rm() {
            0 => (reg::EBX, reg::ESI, reg::DS),
            1 => (reg::EBX, reg::EDI, reg::DS),
            2 => (reg::EBP, reg::ESI, reg::SS),
            3 => (reg::EBP, reg::EDI, reg::SS),
            4 => (reg::ESI, NONE, reg::DS),
            5 => (reg::EDI, NONE, reg::DS),
            6 => (reg::EBP, NONE, reg::SS),
            _ => (reg::EBX, NONE, reg::DS),
        };
        d.base = base;
        d.index = index;
        // A 16-bit disp needs no sign extension: the resolver masks the sum
        // to 16 bits, so only the low 16 bits of the addend matter.
        d.disp = match m.md() {
            0 => 0,
            1 => self.fetch8(bus)? as i8 as u16 as u32,
            _ => self.fetch16(bus)? as u32,
        };
        d.set_seg(self.seg_or(seg));
        Ok(())
    }

    /// 32-bit EA formula, with the SIB byte (mirrors `ea32`/`sib`).
    fn ea32_formula<B: Bus>(&mut self, bus: &mut B, m: ModRm, d: &mut DecodedInsn) -> Exec<()> {
        // Direct address: mod == 00, rm == 101 is a bare disp32 (no EBP).
        if m.md() == 0 && m.rm() == 5 {
            d.disp = self.fetch32(bus)?;
            d.set_seg(self.seg_or(reg::DS));
            return Ok(());
        }

        let seg = if m.rm() == 4 {
            self.sib_formula(bus, m, d)?
        } else {
            d.base = m.rm();
            if m.rm() == reg::EBP { reg::SS } else { reg::DS }
        };

        let disp = match m.md() {
            0 => 0,
            1 => self.fetch8(bus)? as i8 as u32,
            _ => self.fetch32(bus)?,
        };
        // The mod==00/base==101 SIB form already parked its disp32 in `d`.
        d.disp = d.disp.wrapping_add(disp);
        d.set_seg(self.seg_or(seg));
        Ok(())
    }

    /// SIB formula; returns the default segment.
    fn sib_formula<B: Bus>(&mut self, bus: &mut B, m: ModRm, d: &mut DecodedInsn) -> Exec<u8> {
        let sib = self.fetch8(bus)?;
        let (scale, index, base_reg) = (sib >> 6, (sib >> 3) & 7, sib & 7);

        // mod == 00, base == 101: disp32 replaces the base register (and is
        // never scaled, even by the quirk below).
        let (seg, scalable) = if base_reg == 5 && m.md() == 0 {
            d.disp = self.fetch32(bus)?;
            (reg::DS, false)
        } else {
            d.base = base_reg;
            let seg = if base_reg == reg::ESP || base_reg == reg::EBP {
                reg::SS
            } else {
                reg::DS
            };
            (seg, true)
        };

        if index != 4 {
            d.index = index;
            d.set_scale(scale);
        } else if scalable {
            // The 386 scaled-base quirk (`base <<= scale`), canonicalized as
            // index = base with no base register — bit-identical result,
            // no special case in the resolver.
            d.index = d.base;
            d.base = NONE;
            d.set_scale(scale);
        }
        Ok(seg)
    }

    /// Reconstruct the post-decode machine state from a cached instruction:
    /// every decoder field read during *execution* is installed, and EIP
    /// moves past the instruction. Fields only the decode/dispatch path
    /// reads (`ilen`, `lock`, `lock_ok`, `rep`) are deliberately left
    /// stale — nothing on the hit path or in `step`'s fault handling looks
    /// at them, and the next fused decode resets them first.
    #[inline(always)]
    pub(crate) fn begin_decoded(&mut self, d: &DecodedInsn) {
        self.seg_override = d.seg_override();
        self.commit_on_fault = false;
        self.cold_saved = false;
        self.supervisor_override = false;
        self.osize32 = d.osize32();
        self.asize32 = d.asize32();
        self.regs.eip = self.start_eip.wrapping_add(d.len as u32);
    }

    /// Debug-build differential for an icache hit: re-decode the live bytes
    /// through the real fetch path and require every decode product to
    /// match the cached entry. A mismatch means a missed invalidation
    /// (write stamp, context bit, host write) or decoder drift. Restores
    /// EIP/ilen; the caller's `begin_decoded` re-derives the rest.
    #[cfg(debug_assertions)]
    pub(crate) fn icache_differential<B: Bus>(&mut self, bus: &mut B, cached: &DecodedInsn) {
        let (eip0, ilen0) = (self.regs.eip, self.ilen);
        self.ilen = 0;
        self.seg_override = None;
        self.rep = None;
        self.lock = false;
        self.lock_ok = false;
        let fresh = self.fresh_decode(bus);
        assert_eq!(
            fresh.as_ref(),
            Some(cached),
            "icache: stale entry at eip {:#010x} (missed invalidation or decoder drift)",
            self.start_eip
        );
        self.regs.eip = eip0;
        self.ilen = ilen0;
    }

    /// The full decode an icache hit claims to be equivalent to: prefix
    /// scan plus `try_decode`. `None` if the bytes no longer decode into
    /// the hot subset (which on a hit is always a staleness bug).
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
        let mut off = d.disp;
        if d.base != NONE {
            off = off.wrapping_add(self.regs.gpr[(d.base & 7) as usize]);
        }
        if d.index != NONE {
            off = off.wrapping_add(self.regs.gpr[(d.index & 7) as usize] << d.scale());
        }
        if !d.asize32() {
            // 16-bit EA arithmetic wraps at 64 KiB; masking the full-width
            // sum is congruent to `ea16`'s u16 component arithmetic.
            off &= 0xFFFF;
        }
        Operand::Mem { seg: d.seg(), off }
    }

    /// Execute a decoded instruction. Arms are verbatim fused handler
    /// bodies (minus `lock_check` — LOCK-prefixed instructions never decode
    /// into this path) with the EA formula resolved up front. EIP is
    /// already past the instruction, exactly as after the fused fetches.
    /// `self.lock`/`self.rep` may hold a *previous* instruction's values on
    /// a cache hit; no arm reads them.
    #[inline(always)]
    pub(crate) fn exec_decoded<B: Bus>(&mut self, bus: &mut B, d: &DecodedInsn) -> Exec<u32> {
        match d.op {
            Op::AluRmR8 => {
                let op = self.ea_operand(d);
                let a = self.read_op8(bus, op)?;
                let b = self.regs.reg8(d.reg);
                let r = ALU8[d.aux as usize](self, a, b);
                if d.wb() {
                    self.write_op8(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            Op::AluRmRW => {
                let op = self.ea_operand(d);
                if self.osize32 {
                    let a = self.read_op32(bus, op)?;
                    let b = self.regs.reg32(d.reg);
                    let r = ALU32[d.aux as usize](self, a, b);
                    if d.wb() {
                        self.write_op32(bus, op, r)?;
                    }
                } else {
                    let a = self.read_op16(bus, op)?;
                    let b = self.regs.reg16(d.reg);
                    let r = ALU16[d.aux as usize](self, a, b);
                    if d.wb() {
                        self.write_op16(bus, op, r)?;
                    }
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            Op::AluRRm8 => {
                let op = self.ea_operand(d);
                let a = self.regs.reg8(d.reg);
                let b = self.read_op8(bus, op)?;
                let r = ALU8[d.aux as usize](self, a, b);
                if d.wb() {
                    self.regs.set_reg8(d.reg, r);
                }
                Ok(if op.is_mem() { 6 } else { 2 })
            }
            Op::AluRRmW => {
                let op = self.ea_operand(d);
                if self.osize32 {
                    let a = self.regs.reg32(d.reg);
                    let b = self.read_op32(bus, op)?;
                    let r = ALU32[d.aux as usize](self, a, b);
                    if d.wb() {
                        self.regs.set_reg32(d.reg, r);
                    }
                } else {
                    let a = self.regs.reg16(d.reg);
                    let b = self.read_op16(bus, op)?;
                    let r = ALU16[d.aux as usize](self, a, b);
                    if d.wb() {
                        self.regs.set_reg16(d.reg, r);
                    }
                }
                Ok(if op.is_mem() { 6 } else { 2 })
            }
            Op::AluAccImm8 => {
                let r = ALU8[d.aux as usize](self, self.regs.reg8(0), d.imm as u8);
                if d.wb() {
                    self.regs.set_reg8(0, r);
                }
                Ok(2)
            }
            Op::AluAccImmW => {
                if self.osize32 {
                    let r = ALU32[d.aux as usize](self, self.regs.gpr[0], d.imm);
                    if d.wb() {
                        self.regs.gpr[0] = r;
                    }
                } else {
                    let r = ALU16[d.aux as usize](self, self.regs.reg16(0), d.imm as u16);
                    if d.wb() {
                        self.regs.set_reg16(0, r);
                    }
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
                let r = if self.osize32 {
                    ALU32[d.aux as usize](self, a, d.imm)
                } else {
                    ALU16[d.aux as usize](self, a as u16, d.imm as u16) as u32
                };
                if d.wb() {
                    self.write_op(bus, op, r)?;
                }
                Ok(if op.is_mem() { 7 } else { 2 })
            }
            Op::MovRmR8 => {
                let op = self.ea_operand(d);
                let v = self.regs.reg8(d.reg);
                self.write_op8(bus, op, v)?;
                Ok(2)
            }
            Op::MovRmRW => {
                let op = self.ea_operand(d);
                if self.osize32 {
                    let v = self.regs.reg32(d.reg);
                    self.write_op32(bus, op, v)?;
                } else {
                    let v = self.regs.reg16(d.reg);
                    self.write_op16(bus, op, v)?;
                }
                Ok(2)
            }
            Op::MovRRm8 => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)?;
                self.regs.set_reg8(d.reg, v);
                Ok(if op.is_mem() { 4 } else { 2 })
            }
            Op::MovRRmW => {
                let op = self.ea_operand(d);
                if self.osize32 {
                    let v = self.read_op32(bus, op)?;
                    self.regs.set_reg32(d.reg, v);
                } else {
                    let v = self.read_op16(bus, op)?;
                    self.regs.set_reg16(d.reg, v);
                }
                Ok(if op.is_mem() { 4 } else { 2 })
            }
            Op::MovRegImm8 => {
                self.regs.set_reg8(d.reg, d.imm as u8);
                Ok(2)
            }
            Op::MovRegImmW => {
                if self.osize32 {
                    self.regs.set_reg32(d.reg, d.imm);
                } else {
                    self.regs.set_reg16(d.reg, d.imm as u16);
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
                if self.osize32 {
                    self.regs.set_reg32(d.reg, off);
                } else {
                    self.regs.set_reg16(d.reg, off as u16);
                }
                Ok(2)
            }
            Op::IncReg => {
                let i = d.reg;
                if self.osize32 {
                    let r = self.inc32(self.regs.reg32(i));
                    self.regs.set_reg32(i, r);
                } else {
                    let r = self.inc16(self.regs.reg16(i));
                    self.regs.set_reg16(i, r);
                }
                Ok(2)
            }
            Op::DecReg => {
                let i = d.reg;
                if self.osize32 {
                    let r = self.dec32(self.regs.reg32(i));
                    self.regs.set_reg32(i, r);
                } else {
                    let r = self.dec16(self.regs.reg16(i));
                    self.regs.set_reg16(i, r);
                }
                Ok(2)
            }
            Op::PushReg => {
                // Unlike the 8086, PUSH (E)SP pushes the pre-decrement value.
                let v = self.regs.reg32(d.reg);
                self.push(bus, v)?;
                Ok(2)
            }
            Op::PopReg => {
                let v = self.pop(bus)?;
                if self.osize32 {
                    self.regs.set_reg32(d.reg, v);
                } else {
                    self.regs.set_reg16(d.reg, v as u16);
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
                let r = if self.osize32 {
                    if d.aux == 0 {
                        self.inc32(v)
                    } else {
                        self.dec32(v)
                    }
                } else if d.aux == 0 {
                    self.inc16(v as u16) as u32
                } else {
                    self.dec16(v as u16) as u32
                };
                self.write_op(bus, op, r)?;
                Ok(if op.is_mem() { 6 } else { 2 })
            }
            Op::CallRm => {
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                let ret = self.regs.eip;
                self.push(bus, ret)?;
                self.set_ip(v)?;
                Ok(if op.is_mem() { 10 } else { 7 })
            }
            Op::JmpRm => {
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                self.set_ip(v)?;
                Ok(if op.is_mem() { 10 } else { 7 })
            }
            Op::PushRm => {
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                self.push(bus, v)?;
                Ok(if op.is_mem() { 5 } else { 2 })
            }
            Op::ShiftRm8 => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)?;
                let n = if d.aux & 0x80 != 0 {
                    self.regs.reg8(1) as u32
                } else {
                    d.imm
                };
                let r = self.shift_dispatch8(d.aux & 7, v, n);
                self.write_op8(bus, op, r)?;
                Ok(if op.is_mem() { 7 } else { 3 })
            }
            Op::ShiftRmW => {
                let op = self.ea_operand(d);
                let v = self.read_op(bus, op)?;
                let n = if d.aux & 0x80 != 0 {
                    self.regs.reg8(1) as u32
                } else {
                    d.imm
                };
                let r = self.shift_dispatch(d.aux & 7, v, n);
                self.write_op(bus, op, r)?;
                Ok(if op.is_mem() { 7 } else { 3 })
            }
            Op::Jcc => {
                if self.cond(d.aux) {
                    self.jump_rel(d.imm as i32)?;
                    Ok(7)
                } else {
                    Ok(3)
                }
            }
            Op::JmpRel => {
                self.jump_rel(d.imm as i32)?;
                Ok(7)
            }
            Op::CallRel => {
                let ret = self.regs.eip;
                self.push(bus, ret)?;
                self.jump_rel(d.imm as i32)?;
                Ok(7)
            }
            Op::RetNear => {
                let ip = self.pop(bus)?;
                self.set_ip(ip)?;
                Ok(10)
            }
            Op::RetNearImm => {
                let ip = self.pop(bus)?;
                self.set_ip(ip)?;
                self.adjust_sp(d.imm as i32);
                Ok(10)
            }
            Op::ImulRRmW => {
                let op = self.ea_operand(d);
                let src = self.read_op(bus, op)?;
                if self.osize32 {
                    let a = self.regs.reg32(d.reg);
                    let r = self.imul_trunc32(a, src);
                    self.regs.set_reg32(d.reg, r);
                } else {
                    let a = self.regs.reg16(d.reg);
                    let r = self.imul_trunc16(a, src as u16);
                    self.regs.set_reg16(d.reg, r);
                }
                Ok(if op.is_mem() { 22 } else { 20 })
            }
            Op::PushImm => {
                self.push(bus, d.imm)?;
                Ok(2)
            }
            Op::ImulRmImm => {
                let op = self.ea_operand(d);
                let a = self.read_op(bus, op)?;
                if self.osize32 {
                    let r = self.imul_trunc32(a, d.imm);
                    self.regs.set_reg32(d.reg, r);
                } else {
                    let r = self.imul_trunc16(a as u16, d.imm as u16);
                    self.regs.set_reg16(d.reg, r);
                }
                Ok(if op.is_mem() { 22 } else { 20 })
            }
            Op::XchgRm8 => {
                let op = self.ea_operand(d);
                let a = self.read_op8(bus, op)?;
                let b = self.regs.reg8(d.reg);
                self.write_op8(bus, op, b)?;
                self.regs.set_reg8(d.reg, a);
                Ok(if op.is_mem() { 5 } else { 3 })
            }
            Op::XchgRmW => {
                let op = self.ea_operand(d);
                let a = self.read_op(bus, op)?;
                let b = self.regs.reg32(d.reg);
                if self.osize32 {
                    self.write_op32(bus, op, b)?;
                    self.regs.set_reg32(d.reg, a);
                } else {
                    self.write_op16(bus, op, b as u16)?;
                    self.regs.set_reg16(d.reg, a as u16);
                }
                Ok(if op.is_mem() { 5 } else { 3 })
            }
            Op::XchgAcc => {
                let i = d.reg;
                if self.osize32 {
                    let t = self.regs.gpr[0];
                    self.regs.gpr[0] = self.regs.reg32(i);
                    self.regs.set_reg32(i, t);
                } else {
                    let t = self.regs.reg16(0);
                    self.regs.set_reg16(0, self.regs.reg16(i));
                    self.regs.set_reg16(i, t);
                }
                Ok(3)
            }
            Op::MovzxB => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)? as u32;
                if self.osize32 {
                    self.regs.set_reg32(d.reg, v);
                } else {
                    self.regs.set_reg16(d.reg, v as u16);
                }
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            Op::MovzxW => {
                let op = self.ea_operand(d);
                let v = self.read_op16(bus, op)? as u32;
                if self.osize32 {
                    self.regs.set_reg32(d.reg, v);
                } else {
                    self.regs.set_reg16(d.reg, v as u16);
                }
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            Op::MovsxB => {
                let op = self.ea_operand(d);
                let v = self.read_op8(bus, op)? as i8 as i32 as u32;
                if self.osize32 {
                    self.regs.set_reg32(d.reg, v);
                } else {
                    self.regs.set_reg16(d.reg, v as u16);
                }
                Ok(if op.is_mem() { 6 } else { 3 })
            }
            Op::MovsxW => {
                let op = self.ea_operand(d);
                let v = self.read_op16(bus, op)? as i16 as i32 as u32;
                if self.osize32 {
                    self.regs.set_reg32(d.reg, v);
                } else {
                    self.regs.set_reg16(d.reg, v as u16);
                }
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
                if self.cond(d.aux) {
                    if self.osize32 {
                        self.regs.set_reg32(d.reg, v);
                    } else {
                        self.regs.set_reg16(d.reg, v as u16);
                    }
                }
                Ok(if op.is_mem() { 5 } else { 4 })
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::super::{Cpu, LinearMemory, registers::reg};
    use super::*;

    /// A CPU at 0000:1000 in real mode with distinctive register values, and
    /// memory holding `bytes` at that address.
    fn setup(bytes: &[u8]) -> (Cpu, LinearMemory) {
        let mut mem = LinearMemory::new();
        mem.load(0x1000, bytes);
        let mut cpu = Cpu::new();
        cpu.set_cs_ip(0x0000, 0x1000);
        for i in 0..8 {
            // Distinct per register, nonzero high and low halves, odd sums.
            cpu.regs.gpr[i] = 0x1111_1111u32.wrapping_mul(i as u32 + 1) ^ 0x0BAD_F00D;
        }
        (cpu, mem)
    }

    /// Decode `bytes` (starting at the ModRM byte) both ways under the given
    /// address size and override, and assert the resolved operands, the
    /// `reg` field and the consumed length all match.
    fn check_ea(bytes: &[u8], asize32: bool, seg_override: Option<u8>) {
        let (mut cpu, mut mem) = setup(bytes);
        cpu.asize32 = asize32;
        cpu.seg_override = seg_override;

        // Formula decode + resolve.
        cpu.regs.eip = 0x1000;
        cpu.ilen = 0;
        let mut d = DecodedInsn::new(cpu.osize32, asize32, seg_override);
        let sub = cpu.decode_modrm(&mut mem, &mut d).unwrap();
        let got = cpu.ea_operand(&d);
        let (got_eip, got_ilen) = (cpu.regs.eip, cpu.ilen);

        // Fused decode.
        cpu.regs.eip = 0x1000;
        cpu.ilen = 0;
        let (m, want) = cpu.modrm(&mut mem).unwrap();

        assert_eq!(
            got, want,
            "operand mismatch for bytes {bytes:02X?} asize32={asize32} ovr={seg_override:?}"
        );
        assert_eq!(sub, m.reg(), "reg field mismatch for {bytes:02X?}");
        assert_eq!(got_eip, cpu.regs.eip, "length mismatch for {bytes:02X?}");
        assert_eq!(got_ilen, cpu.ilen, "ilen mismatch for {bytes:02X?}");
    }

    /// Every ModRM byte (all mod/reg/rm combinations) in both address sizes,
    /// with generic displacement bytes following.
    #[test]
    fn ea_formula_matches_modrm_exhaustive() {
        for modrm in 0..=255u8 {
            for asize32 in [false, true] {
                let bytes = [modrm, 0x12, 0x34, 0x56, 0x78, 0x9A];
                check_ea(&bytes, asize32, None);
            }
        }
    }

    /// Every SIB byte under mod 00/01/10 — covers the 386 scaled-base quirk
    /// rows (index == 100 with a non-zero scale) and the disp32-base form.
    #[test]
    fn ea_formula_matches_modrm_sib_exhaustive() {
        for md in [0x04u8, 0x44, 0x84] {
            for sib in 0..=255u8 {
                let bytes = [md, sib, 0xEF, 0xBE, 0xAD, 0xDE];
                check_ea(&bytes, true, None);
            }
        }
    }

    /// Segment overrides fold into the resolved operand identically.
    #[test]
    fn ea_formula_honors_segment_overrides() {
        for ovr in [None, Some(reg::ES), Some(reg::CS), Some(reg::FS)] {
            for modrm in [0x00u8, 0x46, 0x86, 0x02, 0x03, 0x05] {
                check_ea(&[modrm, 0x10, 0x20, 0x30, 0x40], false, ovr);
                check_ea(&[modrm, 0x10, 0x20, 0x30, 0x40], true, ovr);
            }
            // SIB with SS-default base (EBP) and override on top.
            check_ea(&[0x44, 0x65, 0x7F], true, ovr);
        }
    }

    /// The decoupled path and the fused path must leave identical machine
    /// state and return identical cycle counts, instruction by instruction.
    #[test]
    fn decoded_execution_matches_fused() {
        // A battery of hot-subset instructions: ALU reg/mem forms, MOVs,
        // INC/DEC, PUSH/POP, shifts, IMUL, TEST, CMP, Jcc (taken and not),
        // CALL/RET, LEA, group forms with memory operands.
        #[rustfmt::skip]
        let program: &[u8] = &[
            0xBC, 0x00, 0x20,             // MOV SP, 0x2000
            0xB8, 0x34, 0x12,             // MOV AX, 0x1234
            0xB9, 0x05, 0x00,             // MOV CX, 5
            0xBB, 0x00, 0x30,             // MOV BX, 0x3000
            0x89, 0x07,                   // MOV [BX], AX
            0x8B, 0x17,                   // MOV DX, [BX]
            0x01, 0xD0,                   // ADD AX, DX
            0x83, 0xC0, 0x7B,             // ADD AX, 0x7B
            0x81, 0x2F, 0x10, 0x00,       // SUB word [BX], 0x10
            0x31, 0xC2,                   // XOR DX, AX
            0x85, 0xC2,                   // TEST DX, AX
            0x39, 0xC2,                   // CMP DX, AX
            0x40,                         // INC AX
            0x4A,                         // DEC DX
            0xFE, 0x07,                   // INC byte [BX]
            0xFF, 0x0F,                   // DEC word [BX]
            0x50,                         // PUSH AX
            0x5B,                         // POP BX
            0xC1, 0xC0, 0x03,             // ROL AX, 3
            0xD3, 0xE8,                   // SHR AX, CL
            0x0F, 0xAF, 0xC2,             // IMUL AX, DX
            0x8D, 0x47, 0x10,             // LEA AX, [BX+0x10]
            0x3C, 0x00,                   // CMP AL, 0
            0x75, 0x01,                   // JNZ +1
            0x90,                         // NOP (XCHG AX,AX)
            0xE8, 0x02, 0x00,             // CALL +2
            0xEB, 0x02,                   // JMP +2 (over the RET)
            0x40,                         // INC AX   <- call target
            0xC3,                         // RET
            0x68, 0x34, 0x12,             // PUSH 0x1234
            0x6A, 0xF6,                   // PUSH -10
            0x6B, 0xC2, 0x05,             // IMUL AX, DX, 5
            0x86, 0x07,                   // XCHG [BX], AL
            0x87, 0x17,                   // XCHG [BX], DX
            0x91,                         // XCHG AX, CX
            0x0F, 0xB6, 0xC1,             // MOVZX AX, CL
            0x0F, 0xBE, 0x0F,             // MOVSX CX, byte [BX]
            0x0F, 0x94, 0xC2,             // SETZ DL
            0x0F, 0x44, 0xCA,             // CMOVZ CX, DX (extensions on)
            0xF4,                         // HLT
        ];

        // Two passes over the program: the first fills the icache, the
        // second runs entirely from hits — both must match the fused path
        // (and, transitively, each other) in state and cycles.
        let run = |fused_only: bool| {
            let mut mem = LinearMemory::new();
            mem.load(0x1000, program);
            let mut cpu = Cpu::new();
            cpu.extensions = true; // CMOVcc coverage
            cpu.fused_only = fused_only;
            let mut cycles = Vec::new();
            for _ in 0..2 {
                cpu.set_cs_ip(0x0000, 0x1000);
                cpu.halted = false;
                for _ in 0..200 {
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
