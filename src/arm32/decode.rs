//! Decoders for the ARM and Thumb instruction sets, producing the shared
//! register-independent [`DecodedInsn`] representation that the executor
//! dispatches on and the decoded-instruction cache stores.
//!
//! Both decoders are pure functions of the fetched word — ARM has no
//! prefixes, ModRM or trailing immediates, so unlike the x86 cores there is
//! no fused/decoupled split: every instruction decodes once into a
//! `DecodedInsn` and the same executor runs it, cached or not. Thumb
//! instructions map onto the ARM operations wherever the semantics match
//! (a Thumb `ADD Rd, #imm` *is* an ARM `ADDS Rd, Rd, #imm`), so the
//! executor stays single; the two genuinely Thumb-only shapes (the split
//! BL pair, the word-aligned PC of PC-relative forms) get an op and a
//! register sentinel of their own.

/// "PC, word-aligned" base-register sentinel ([`PCA`] > 15): the value
/// reads as `(pc + 4) & !3`, the Thumb ADR/LDR-literal rule. Never emitted
/// by the ARM decoder.
pub(crate) const PCA: u8 = 16;

/// ALU opcodes of the data-processing instructions (bits 24:21), also used
/// as-is by the Thumb decoder.
pub(crate) mod alu_op {
    pub const AND: u8 = 0x0;
    pub const EOR: u8 = 0x1;
    pub const SUB: u8 = 0x2;
    pub const RSB: u8 = 0x3;
    pub const ADD: u8 = 0x4;
    pub const ADC: u8 = 0x5;
    pub const SBC: u8 = 0x6;
    pub const RSC: u8 = 0x7;
    pub const TST: u8 = 0x8;
    pub const TEQ: u8 = 0x9;
    pub const CMP: u8 = 0xA;
    pub const CMN: u8 = 0xB;
    pub const ORR: u8 = 0xC;
    pub const MOV: u8 = 0xD;
    pub const BIC: u8 = 0xE;
    pub const MVN: u8 = 0xF;
}

/// [`DecodedInsn::aux`] layout for the data-processing ops: the ALU opcode,
/// the S bit, the shift type (register forms), and the rotated-immediate
/// marker (immediate form — the shifter carry becomes bit 31 of the
/// immediate).
pub(crate) mod dp {
    pub const OP: u8 = 0x0F;
    pub const S: u8 = 0x10;
    pub const TY_SHIFT: u8 = 5;
    pub const TY: u8 = 3 << TY_SHIFT;
    pub const ROT: u8 = 0x80;
}

/// [`DecodedInsn::aux`] layout for [`Op::Ldr`]/[`Op::Str`]. `WB` is
/// pre-normalized: post-indexed forms always write back, and their W bit
/// selects the user-mode access (`USER`, the LDRT/STRT forms) instead.
pub(crate) mod xfer {
    pub const BYTE: u8 = 0x01;
    pub const PRE: u8 = 0x02;
    pub const UP: u8 = 0x04;
    pub const WB: u8 = 0x08;
    pub const REG: u8 = 0x10;
    pub const USER: u8 = 0x20;
    pub const TY_SHIFT: u8 = 6;
}

/// [`DecodedInsn::aux`] layout for the halfword/signed transfers
/// ([`Op::LdrMisc`]/[`Op::StrMisc`]). `SH` holds bits 6:5 of the encoding:
/// `01` = halfword, `10` = signed byte, `11` = signed halfword.
pub(crate) mod hw {
    pub const SH: u8 = 0x03;
    pub const H: u8 = 0x01;
    pub const SB: u8 = 0x02;
    pub const SHW: u8 = 0x03;
    pub const PRE: u8 = 0x04;
    pub const UP: u8 = 0x08;
    pub const WB: u8 = 0x10;
    pub const REG: u8 = 0x20;
}

/// [`DecodedInsn::aux`] layout for [`Op::Ldm`]/[`Op::Stm`]. `EMPTY` marks
/// the empty-register-list quirk (transfers PC only, steps the base by
/// 0x40).
pub(crate) mod blk {
    pub const PRE: u8 = 0x01;
    pub const UP: u8 = 0x02;
    pub const S: u8 = 0x04;
    pub const WB: u8 = 0x08;
    pub const EMPTY: u8 = 0x10;
}

/// [`DecodedInsn::aux`] layout for the multiplies.
pub(crate) mod mul {
    pub const ACC: u8 = 0x01;
    pub const SIGNED: u8 = 0x02;
    pub const S: u8 = 0x10;
}

/// [`DecodedInsn::aux`] layout for MRS/MSR: the CPSR/SPSR select and, for
/// MSR, the four-bit field mask (c/x/s/f) in the high nibble.
pub(crate) mod sr {
    pub const SPSR: u8 = 0x01;
    pub const FIELD_SHIFT: u8 = 4;
}

/// [`DecodedInsn::aux`] bit for [`Op::BranchImm`]: link (BL).
pub(crate) const BR_LINK: u8 = 0x01;
/// [`DecodedInsn::aux`] bit for [`Op::ThumbBl`]: the suffix half (H=1).
pub(crate) const TBL_SUFFIX: u8 = 0x01;

/// Handler discriminant of a decoded instruction.
///
/// The all-zero bit pattern of a cache [`Entry`](super::icache::Entry) must
/// be a valid `DecodedInsn`, so the discriminant 0 variant is meaningful
/// (dead entries are fenced by their version, never by their contents).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Op {
    /// Data processing, operand2 = rotated immediate (pre-rotated into
    /// `imm`).
    DpImm = 0,
    /// Data processing, operand2 = `Rm` shifted by an immediate (`rs` =
    /// amount).
    DpShImm,
    /// Data processing, operand2 = `Rm` shifted by `Rs` (+4 on the PC
    /// reads, one extra cycle).
    DpShReg,
    /// MUL/MLA.
    Mul,
    /// UMULL/UMLAL/SMULL/SMLAL (`rd` = RdHi, `rn` = RdLo).
    Mull,
    /// MRS `rd, CPSR/SPSR`.
    Mrs,
    /// MSR `CPSR/SPSR_fields, Rm`.
    MsrReg,
    /// MSR `CPSR/SPSR_fields, #imm` (pre-rotated into `imm`).
    MsrImm,
    /// B/BL (`imm` = pre-shifted signed offset from `pc+8`).
    BranchImm,
    /// BX `Rm` (Thumb interworking).
    Bx,
    /// LDR/LDRB (rotation quirks live in the executor).
    Ldr,
    /// STR/STRB.
    Str,
    /// LDRH/LDRSB/LDRSH.
    LdrMisc,
    /// STRH.
    StrMisc,
    /// LDM (all addressing modes, S-bit user bank / SPSR restore).
    Ldm,
    /// STM.
    Stm,
    /// SWP/SWPB.
    Swp,
    /// SWI (`imm` = comment field).
    Swi,
    /// Undefined instruction (also every coprocessor encoding — this core
    /// has no coprocessors, exactly like an ARM7TDMI with none attached).
    Undef,
    /// The two-halfword Thumb BL pair.
    ThumbBl,
}

/// Register-independent decode products of one instruction (12 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodedInsn {
    /// Handler discriminant.
    pub op: Op,
    /// Condition field (0–14; 15 = never on this core).
    pub cond: u8,
    /// Destination register (RdHi for long multiplies).
    pub rd: u8,
    /// First operand / base register (RdLo for long multiplies); may be
    /// the [`PCA`] sentinel in Thumb code.
    pub rn: u8,
    /// Second operand / offset register.
    pub rm: u8,
    /// Shift-amount register (register-shift forms and multiplies) or
    /// immediate shift amount (immediate-shift forms).
    pub rs: u8,
    /// Op-specific bits (see the `dp`/`xfer`/`hw`/`blk`/`mul`/`sr`
    /// modules).
    pub aux: u8,
    /// Instruction length in bytes: 4 (ARM) or 2 (Thumb).
    pub len: u8,
    /// Immediate: rotated operand, offset, register list, or SWI comment.
    pub imm: u32,
}

impl DecodedInsn {
    /// A skeleton with every field zero except `op`, `cond` and `len`.
    #[inline]
    fn new(op: Op, cond: u8, len: u8) -> Self {
        DecodedInsn {
            op,
            cond,
            rd: 0,
            rn: 0,
            rm: 0,
            rs: 0,
            aux: 0,
            len,
            imm: 0,
        }
    }
}

/// Condition code for "always".
pub(crate) const COND_AL: u8 = 14;

// --- ARM decoder --------------------------------------------------------------

/// Decode one 32-bit ARM instruction.
///
/// Unallocated corners decode deterministically: coprocessor space and the
/// architecturally *undefined* hole (`011x...1`) raise the
/// undefined-instruction trap, while the UNPREDICTABLE data-processing
/// leftovers (TST..CMN with S=0 that are not MRS/MSR/BX) execute as the
/// corresponding data-processing operation without flag writes.
pub(crate) fn decode_arm(w: u32) -> DecodedInsn {
    let cond = (w >> 28) as u8;
    let mut d = DecodedInsn::new(Op::Undef, cond, 4);
    match (w >> 25) & 7 {
        0b000 => {
            if w & 0x0FFF_FFF0 == 0x012F_FF10 {
                d.op = Op::Bx;
                d.rm = (w & 0xF) as u8;
            } else if w & 0x0FB0_0FF0 == 0x0100_0090 {
                d.op = Op::Swp;
                d.rn = ((w >> 16) & 0xF) as u8;
                d.rd = ((w >> 12) & 0xF) as u8;
                d.rm = (w & 0xF) as u8;
                d.aux = ((w >> 22) & 1) as u8; // byte
            } else if w & 0x0FC0_00F0 == 0x0000_0090 {
                d.op = Op::Mul;
                d.rd = ((w >> 16) & 0xF) as u8;
                d.rn = ((w >> 12) & 0xF) as u8; // accumulator
                d.rs = ((w >> 8) & 0xF) as u8;
                d.rm = (w & 0xF) as u8;
                d.aux = (((w >> 21) & 1) as u8 * mul::ACC) | (((w >> 20) & 1) as u8 * mul::S);
            } else if w & 0x0F80_00F0 == 0x0080_0090 {
                d.op = Op::Mull;
                d.rd = ((w >> 16) & 0xF) as u8; // RdHi
                d.rn = ((w >> 12) & 0xF) as u8; // RdLo
                d.rs = ((w >> 8) & 0xF) as u8;
                d.rm = (w & 0xF) as u8;
                d.aux = (((w >> 21) & 1) as u8 * mul::ACC)
                    | (((w >> 22) & 1) as u8 * mul::SIGNED)
                    | (((w >> 20) & 1) as u8 * mul::S);
            } else if w & 0x0E00_0090 == 0x0000_0090 {
                // Halfword/signed transfer space (SH != 00; SH == 00 is the
                // multiply/SWP space, whose leftovers stay Undef).
                let sh = ((w >> 5) & 3) as u8;
                let load = w & (1 << 20) != 0;
                if sh != 0 && (load || sh == hw::H) {
                    d.op = if load { Op::LdrMisc } else { Op::StrMisc };
                    d.rn = ((w >> 16) & 0xF) as u8;
                    d.rd = ((w >> 12) & 0xF) as u8;
                    let pre = w & (1 << 24) != 0;
                    d.aux = sh
                        | if pre { hw::PRE } else { 0 }
                        | if w & (1 << 23) != 0 { hw::UP } else { 0 }
                        | if !pre || w & (1 << 21) != 0 {
                            hw::WB
                        } else {
                            0
                        };
                    if w & (1 << 22) != 0 {
                        d.imm = ((w >> 4) & 0xF0) | (w & 0xF);
                    } else {
                        d.aux |= hw::REG;
                        d.rm = (w & 0xF) as u8;
                    }
                }
                // A signed *store* (SH = 10/11 with L=0) stays Undef.
            } else if w & 0x0FBF_0FFF == 0x010F_0000 {
                d.op = Op::Mrs;
                d.rd = ((w >> 12) & 0xF) as u8;
                d.aux = ((w >> 22) & 1) as u8; // SPSR
            } else if w & 0x0FB0_FFF0 == 0x0120_F000 {
                d.op = Op::MsrReg;
                d.rm = (w & 0xF) as u8;
                d.aux = (((w >> 22) & 1) as u8 * sr::SPSR)
                    | ((((w >> 16) & 0xF) as u8) << sr::FIELD_SHIFT);
            } else {
                decode_dp_reg(w, &mut d);
            }
        }
        0b001 => {
            if w & 0x0FB0_F000 == 0x0320_F000 {
                d.op = Op::MsrImm;
                d.aux = (((w >> 22) & 1) as u8 * sr::SPSR)
                    | ((((w >> 16) & 0xF) as u8) << sr::FIELD_SHIFT);
                d.imm = (w & 0xFF).rotate_right(((w >> 8) & 0xF) * 2);
            } else {
                d.op = Op::DpImm;
                d.rn = ((w >> 16) & 0xF) as u8;
                d.rd = ((w >> 12) & 0xF) as u8;
                let rot = (w >> 8) & 0xF;
                d.imm = (w & 0xFF).rotate_right(rot * 2);
                d.aux = ((w >> 21) & 0xF) as u8
                    | (((w >> 20) & 1) as u8 * dp::S)
                    | if rot != 0 { dp::ROT } else { 0 };
            }
        }
        0b010 | 0b011 => {
            if w & (1 << 25) != 0 && w & (1 << 4) != 0 {
                // The architecturally undefined hole.
                return d;
            }
            let load = w & (1 << 20) != 0;
            d.op = if load { Op::Ldr } else { Op::Str };
            d.rn = ((w >> 16) & 0xF) as u8;
            d.rd = ((w >> 12) & 0xF) as u8;
            let pre = w & (1 << 24) != 0;
            let wbit = w & (1 << 21) != 0;
            d.aux = (((w >> 22) & 1) as u8 * xfer::BYTE)
                | if pre { xfer::PRE } else { 0 }
                | if w & (1 << 23) != 0 { xfer::UP } else { 0 }
                | if !pre || wbit { xfer::WB } else { 0 }
                | if !pre && wbit { xfer::USER } else { 0 };
            if w & (1 << 25) != 0 {
                d.aux |= xfer::REG | ((((w >> 5) & 3) as u8) << xfer::TY_SHIFT);
                d.rm = (w & 0xF) as u8;
                d.rs = ((w >> 7) & 0x1F) as u8;
            } else {
                d.imm = w & 0xFFF;
            }
        }
        0b100 => {
            let load = w & (1 << 20) != 0;
            d.op = if load { Op::Ldm } else { Op::Stm };
            d.rn = ((w >> 16) & 0xF) as u8;
            d.aux = (((w >> 24) & 1) as u8 * blk::PRE)
                | (((w >> 23) & 1) as u8 * blk::UP)
                | (((w >> 22) & 1) as u8 * blk::S)
                | (((w >> 21) & 1) as u8 * blk::WB);
            d.imm = w & 0xFFFF;
            if d.imm == 0 {
                // Empty list: transfers PC only, steps the base by 0x40.
                d.imm = 1 << 15;
                d.aux |= blk::EMPTY;
            }
        }
        0b101 => {
            d.op = Op::BranchImm;
            d.aux = ((w >> 24) & 1) as u8 * BR_LINK;
            d.imm = (((w & 0x00FF_FFFF) << 8) as i32 >> 6) as u32; // sext(off24) * 4
        }
        0b110 => {} // LDC/STC: no coprocessors — Undef.
        _ => {
            if w & (1 << 24) != 0 {
                d.op = Op::Swi;
                d.imm = w & 0x00FF_FFFF;
            }
            // CDP/MCR/MRC: no coprocessors — Undef.
        }
    }
    d
}

/// The register-operand data-processing forms (group 000 leftovers).
fn decode_dp_reg(w: u32, d: &mut DecodedInsn) {
    d.rn = ((w >> 16) & 0xF) as u8;
    d.rd = ((w >> 12) & 0xF) as u8;
    d.rm = (w & 0xF) as u8;
    d.aux = ((w >> 21) & 0xF) as u8
        | (((w >> 20) & 1) as u8 * dp::S)
        | ((((w >> 5) & 3) as u8) << dp::TY_SHIFT);
    if w & (1 << 4) != 0 {
        // Bit 7 set with bit 4 was consumed by the multiply/transfer space.
        d.op = Op::DpShReg;
        d.rs = ((w >> 8) & 0xF) as u8;
    } else {
        d.op = Op::DpShImm;
        d.rs = ((w >> 7) & 0x1F) as u8;
    }
}

// --- Thumb decoder --------------------------------------------------------------

/// Decode one 16-bit Thumb instruction into the shared representation.
/// Every product has `cond` = always (except format 16's conditional
/// branch) and `len` = 2.
pub(crate) fn decode_thumb(hw: u16) -> DecodedInsn {
    let w = hw as u32;
    let mut d = DecodedInsn::new(Op::Undef, COND_AL, 2);
    let rd = (w & 7) as u8;
    let rs = ((w >> 3) & 7) as u8;
    match w >> 11 {
        // Format 1: LSL/LSR/ASR Rd, Rs, #imm5 → MOVS Rd, Rs shift #imm.
        0b00000..=0b00010 => {
            d.op = Op::DpShImm;
            d.rd = rd;
            d.rm = rs;
            d.rs = ((w >> 6) & 0x1F) as u8;
            d.aux = alu_op::MOV | dp::S | ((((w >> 11) & 3) as u8) << dp::TY_SHIFT);
        }
        // Format 2: ADD/SUB Rd, Rs, Rn|#imm3.
        0b00011 => {
            let op = if w & (1 << 9) != 0 {
                alu_op::SUB
            } else {
                alu_op::ADD
            };
            d.rd = rd;
            d.rn = rs;
            if w & (1 << 10) != 0 {
                d.op = Op::DpImm;
                d.imm = (w >> 6) & 7;
            } else {
                d.op = Op::DpShImm;
                d.rm = ((w >> 6) & 7) as u8;
            }
            d.aux = op | dp::S;
        }
        // Format 3: MOV/CMP/ADD/SUB Rd, #imm8.
        0b00100..=0b00111 => {
            d.op = Op::DpImm;
            let r = ((w >> 8) & 7) as u8;
            d.rd = r;
            d.rn = r;
            d.imm = w & 0xFF;
            d.aux = dp::S
                | match (w >> 11) & 3 {
                    0 => alu_op::MOV,
                    1 => alu_op::CMP,
                    2 => alu_op::ADD,
                    _ => alu_op::SUB,
                };
        }
        0b01000 => {
            if w & (1 << 10) == 0 {
                // Format 4: register ALU operations.
                d.rd = rd;
                match (w >> 6) & 0xF {
                    // The shift-by-register group → MOVS Rd, Rd shift Rs.
                    0x2 | 0x3 | 0x4 | 0x7 => {
                        d.op = Op::DpShReg;
                        d.rm = rd;
                        d.rs = rs;
                        let ty = match (w >> 6) & 0xF {
                            0x2 => 0u8, // LSL
                            0x3 => 1,   // LSR
                            0x4 => 2,   // ASR
                            _ => 3,     // ROR
                        };
                        d.aux = alu_op::MOV | dp::S | (ty << dp::TY_SHIFT);
                    }
                    0x9 => {
                        // NEG Rd, Rs → RSBS Rd, Rs, #0.
                        d.op = Op::DpImm;
                        d.rn = rs;
                        d.aux = alu_op::RSB | dp::S;
                    }
                    0xD => {
                        // MUL Rd, Rs → MULS Rd, Rs, Rd.
                        d.op = Op::Mul;
                        d.rm = rd;
                        d.rs = rs;
                        d.aux = mul::S;
                    }
                    op => {
                        // AND EOR ADC SBC TST CMP CMN ORR BIC MVN, straight
                        // through (same opcode numbering as ARM for all).
                        d.op = Op::DpShImm;
                        d.rn = rd;
                        d.rm = rs;
                        d.aux = op as u8 | dp::S;
                    }
                }
            } else {
                // Format 5: hi-register ADD/CMP/MOV and BX.
                let rd_full = rd | (((w >> 7) & 1) as u8) << 3;
                let rs_full = ((w >> 3) & 0xF) as u8;
                match (w >> 8) & 3 {
                    0 => {
                        d.op = Op::DpShImm;
                        d.rd = rd_full;
                        d.rn = rd_full;
                        d.rm = rs_full;
                        d.aux = alu_op::ADD; // no flags
                    }
                    1 => {
                        d.op = Op::DpShImm;
                        d.rn = rd_full;
                        d.rm = rs_full;
                        d.aux = alu_op::CMP | dp::S;
                    }
                    2 => {
                        d.op = Op::DpShImm;
                        d.rd = rd_full;
                        d.rm = rs_full;
                        d.aux = alu_op::MOV; // no flags
                    }
                    _ => {
                        // BX Rs. The H1 bit (BLX on ARMv5) is simply
                        // ignored by the ARM7TDMI — suite-pinned.
                        d.op = Op::Bx;
                        d.rm = rs_full;
                    }
                }
            }
        }
        // Format 6: LDR Rd, [PC, #imm8*4] (PC word-aligned).
        0b01001 => {
            d.op = Op::Ldr;
            d.rd = ((w >> 8) & 7) as u8;
            d.rn = PCA;
            d.imm = (w & 0xFF) * 4;
            d.aux = xfer::PRE | xfer::UP;
        }
        // Formats 7/8: register-offset loads/stores.
        0b01010 | 0b01011 => {
            d.rd = rd;
            d.rn = rs;
            d.rm = ((w >> 6) & 7) as u8;
            if w & (1 << 9) == 0 {
                // Format 7: STR/STRB/LDR/LDRB.
                let load = w & (1 << 11) != 0;
                d.op = if load { Op::Ldr } else { Op::Str };
                d.aux = xfer::PRE
                    | xfer::UP
                    | xfer::REG
                    | if w & (1 << 10) != 0 { xfer::BYTE } else { 0 };
            } else {
                // Format 8: STRH/LDRH/LDRSB/LDRSH.
                let (op, sh) = match (w >> 10) & 3 {
                    0 => (Op::StrMisc, hw::H),
                    1 => (Op::LdrMisc, hw::SB),
                    2 => (Op::LdrMisc, hw::H),
                    _ => (Op::LdrMisc, hw::SHW),
                };
                d.op = op;
                d.aux = sh | hw::PRE | hw::UP | hw::REG;
            }
        }
        // Format 9: STR/LDR/STRB/LDRB Rd, [Rb, #imm5].
        0b01100..=0b01111 => {
            let byte = w & (1 << 12) != 0;
            let load = w & (1 << 11) != 0;
            d.op = if load { Op::Ldr } else { Op::Str };
            d.rd = rd;
            d.rn = rs;
            d.imm = (w >> 6) & 0x1F;
            if !byte {
                d.imm *= 4;
            }
            d.aux = xfer::PRE | xfer::UP | if byte { xfer::BYTE } else { 0 };
        }
        // Format 10: STRH/LDRH Rd, [Rb, #imm5*2].
        0b10000 | 0b10001 => {
            let load = w & (1 << 11) != 0;
            d.op = if load { Op::LdrMisc } else { Op::StrMisc };
            d.rd = rd;
            d.rn = rs;
            d.imm = ((w >> 6) & 0x1F) * 2;
            d.aux = hw::H | hw::PRE | hw::UP;
        }
        // Format 11: STR/LDR Rd, [SP, #imm8*4].
        0b10010 | 0b10011 => {
            let load = w & (1 << 11) != 0;
            d.op = if load { Op::Ldr } else { Op::Str };
            d.rd = ((w >> 8) & 7) as u8;
            d.rn = 13;
            d.imm = (w & 0xFF) * 4;
            d.aux = xfer::PRE | xfer::UP;
        }
        // Format 12: ADD Rd, PC|SP, #imm8*4.
        0b10100 | 0b10101 => {
            d.op = Op::DpImm;
            d.rd = ((w >> 8) & 7) as u8;
            d.rn = if w & (1 << 11) != 0 { 13 } else { PCA };
            d.imm = (w & 0xFF) * 4;
            d.aux = alu_op::ADD; // no flags
        }
        0b10110 | 0b10111 => {
            match (w >> 8) & 0xF {
                // Format 13: ADD SP, #±imm7*4.
                0b0000 => {
                    d.op = Op::DpImm;
                    d.rd = 13;
                    d.rn = 13;
                    d.imm = (w & 0x7F) * 4;
                    d.aux = if w & (1 << 7) != 0 {
                        alu_op::SUB
                    } else {
                        alu_op::ADD
                    };
                }
                // Format 14: PUSH/POP {rlist, LR/PC}.
                0b0100 | 0b0101 | 0b1100 | 0b1101 => {
                    let load = w & (1 << 11) != 0;
                    d.rn = 13;
                    d.imm = w & 0xFF;
                    if load {
                        d.op = Op::Ldm;
                        d.aux = blk::UP | blk::WB; // LDMIA SP!
                        if w & (1 << 8) != 0 {
                            d.imm |= 1 << 15; // PC
                        }
                    } else {
                        d.op = Op::Stm;
                        d.aux = blk::PRE | blk::WB; // STMDB SP!
                        if w & (1 << 8) != 0 {
                            d.imm |= 1 << 14; // LR
                        }
                    }
                    if d.imm == 0 {
                        d.imm = 1 << 15;
                        d.aux |= blk::EMPTY;
                    }
                }
                _ => {} // BKPT etc. are ARMv5 — Undef.
            }
        }
        // Format 15: STMIA/LDMIA Rb!, {rlist}.
        0b11000 | 0b11001 => {
            let load = w & (1 << 11) != 0;
            d.op = if load { Op::Ldm } else { Op::Stm };
            d.rn = ((w >> 8) & 7) as u8;
            d.imm = w & 0xFF;
            d.aux = blk::UP | blk::WB;
            if d.imm == 0 {
                d.imm = 1 << 15;
                d.aux |= blk::EMPTY;
            }
        }
        // Formats 16/17: conditional branch and SWI. The 0xE condition
        // (documented as undefined on ARMv4T) executes as an
        // always-taken branch on the ARM7TDMI — suite-pinned.
        0b11010 | 0b11011 => {
            let cond = ((w >> 8) & 0xF) as u8;
            if cond == 0xF {
                d.op = Op::Swi;
                d.imm = w & 0xFF;
            } else {
                d.op = Op::BranchImm;
                d.cond = cond;
                d.imm = (((w & 0xFF) << 24) as i32 >> 23) as u32; // sext(off8) * 2
            }
        }
        // Format 18: B (unconditional).
        0b11100 => {
            d.op = Op::BranchImm;
            d.imm = (((w & 0x7FF) << 21) as i32 >> 20) as u32; // sext(off11) * 2
        }
        // Format 19: the BL pair (11101 would be BLX — ARMv5, Undef here).
        0b11110 | 0b11111 => {
            d.op = Op::ThumbBl;
            d.aux = if w & (1 << 11) != 0 { TBL_SUFFIX } else { 0 };
            d.imm = w & 0x7FF;
        }
        _ => {}
    }
    d
}
