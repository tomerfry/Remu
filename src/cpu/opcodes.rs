//! The instruction set: the [`Operation`] semantics, the [`OpInfo`] metadata
//! record, and the 256-entry [`OPCODES`] decode table.
//!
//! Decoding splits each opcode into two orthogonal axes — *what* it does
//! ([`Operation`]) and *how* it finds its operand
//! ([`AddressingMode`](crate::cpu::addressing::AddressingMode)). That lets the
//! ~56 operations and ~13 addressing modes be implemented once each, instead of
//! duplicating logic across all 151 valid encodings.

use crate::cpu::addressing::AddressingMode;

/// What an instruction does, independent of how it addresses memory.
///
/// Variant names are the canonical 6502 mnemonics, so `{:?}` yields the mnemonic
/// directly (used by the disassembler).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Operation {
    ADC,
    AND,
    ASL,
    BCC,
    BCS,
    BEQ,
    BIT,
    BMI,
    BNE,
    BPL,
    BRK,
    BVC,
    BVS,
    CLC,
    CLD,
    CLI,
    CLV,
    CMP,
    CPX,
    CPY,
    DEC,
    DEX,
    DEY,
    EOR,
    INC,
    INX,
    INY,
    JMP,
    JSR,
    LDA,
    LDX,
    LDY,
    LSR,
    NOP,
    ORA,
    PHA,
    PHP,
    PLA,
    PLP,
    ROL,
    ROR,
    RTI,
    RTS,
    SBC,
    SEC,
    SED,
    SEI,
    STA,
    STX,
    STY,
    TAX,
    TAY,
    TSX,
    TXA,
    TXS,
    TYA,
    /// Illegal/undocumented opcode that jams the processor (KIL/JAM/HLT).
    KIL,
}

/// Static metadata for one opcode.
#[derive(Debug, Clone, Copy)]
pub struct OpInfo {
    /// The operation to perform.
    pub operation: Operation,
    /// How to resolve the operand.
    pub mode: AddressingMode,
    /// Base cycle count (before page-cross / branch penalties).
    pub cycles: u8,
    /// Whether this opcode pays the `+1` page-cross penalty on an indexed read.
    /// True only for read instructions in `AbsoluteX/Y` and `(zp),Y`; stores and
    /// read-modify-write instructions always take a fixed count.
    pub page_penalty: bool,
}

impl OpInfo {
    /// Total length of the instruction in bytes (opcode + operand bytes).
    #[inline]
    pub fn length(&self) -> u8 {
        self.mode.length()
    }
}

/// Convenience constructor for a table entry.
const fn op(operation: Operation, mode: AddressingMode, cycles: u8, page_penalty: bool) -> OpInfo {
    OpInfo {
        operation,
        mode,
        cycles,
        page_penalty,
    }
}

/// The decode table: index by opcode byte to get its [`OpInfo`].
///
/// Unfilled entries are illegal/undocumented opcodes, left as `KIL` for now.
///
/// A `const` (not `static`) so the fused dispatch handlers can evaluate their
/// entry at compile time.
pub const OPCODES: [OpInfo; 256] = build_table();

const fn build_table() -> [OpInfo; 256] {
    use AddressingMode::*;
    use Operation::*;

    // Default every slot to an illegal/jam entry, then fill the official ones.
    let mut t = [op(KIL, Implied, 2, false); 256];

    // --- ADC ---
    t[0x69] = op(ADC, Immediate, 2, false);
    t[0x65] = op(ADC, ZeroPage, 3, false);
    t[0x75] = op(ADC, ZeroPageX, 4, false);
    t[0x6D] = op(ADC, Absolute, 4, false);
    t[0x7D] = op(ADC, AbsoluteX, 4, true);
    t[0x79] = op(ADC, AbsoluteY, 4, true);
    t[0x61] = op(ADC, IndexedIndirect, 6, false);
    t[0x71] = op(ADC, IndirectIndexed, 5, true);

    // --- AND ---
    t[0x29] = op(AND, Immediate, 2, false);
    t[0x25] = op(AND, ZeroPage, 3, false);
    t[0x35] = op(AND, ZeroPageX, 4, false);
    t[0x2D] = op(AND, Absolute, 4, false);
    t[0x3D] = op(AND, AbsoluteX, 4, true);
    t[0x39] = op(AND, AbsoluteY, 4, true);
    t[0x21] = op(AND, IndexedIndirect, 6, false);
    t[0x31] = op(AND, IndirectIndexed, 5, true);

    // --- ASL ---
    t[0x0A] = op(ASL, Accumulator, 2, false);
    t[0x06] = op(ASL, ZeroPage, 5, false);
    t[0x16] = op(ASL, ZeroPageX, 6, false);
    t[0x0E] = op(ASL, Absolute, 6, false);
    t[0x1E] = op(ASL, AbsoluteX, 7, false);

    // --- Branches ---
    t[0x90] = op(BCC, Relative, 2, false);
    t[0xB0] = op(BCS, Relative, 2, false);
    t[0xF0] = op(BEQ, Relative, 2, false);
    t[0x30] = op(BMI, Relative, 2, false);
    t[0xD0] = op(BNE, Relative, 2, false);
    t[0x10] = op(BPL, Relative, 2, false);
    t[0x50] = op(BVC, Relative, 2, false);
    t[0x70] = op(BVS, Relative, 2, false);

    // --- BIT ---
    t[0x24] = op(BIT, ZeroPage, 3, false);
    t[0x2C] = op(BIT, Absolute, 4, false);

    // --- BRK ---
    t[0x00] = op(BRK, Implied, 7, false);

    // --- Flag clears/sets ---
    t[0x18] = op(CLC, Implied, 2, false);
    t[0xD8] = op(CLD, Implied, 2, false);
    t[0x58] = op(CLI, Implied, 2, false);
    t[0xB8] = op(CLV, Implied, 2, false);
    t[0x38] = op(SEC, Implied, 2, false);
    t[0xF8] = op(SED, Implied, 2, false);
    t[0x78] = op(SEI, Implied, 2, false);

    // --- CMP ---
    t[0xC9] = op(CMP, Immediate, 2, false);
    t[0xC5] = op(CMP, ZeroPage, 3, false);
    t[0xD5] = op(CMP, ZeroPageX, 4, false);
    t[0xCD] = op(CMP, Absolute, 4, false);
    t[0xDD] = op(CMP, AbsoluteX, 4, true);
    t[0xD9] = op(CMP, AbsoluteY, 4, true);
    t[0xC1] = op(CMP, IndexedIndirect, 6, false);
    t[0xD1] = op(CMP, IndirectIndexed, 5, true);

    // --- CPX / CPY ---
    t[0xE0] = op(CPX, Immediate, 2, false);
    t[0xE4] = op(CPX, ZeroPage, 3, false);
    t[0xEC] = op(CPX, Absolute, 4, false);
    t[0xC0] = op(CPY, Immediate, 2, false);
    t[0xC4] = op(CPY, ZeroPage, 3, false);
    t[0xCC] = op(CPY, Absolute, 4, false);

    // --- DEC / DEX / DEY ---
    t[0xC6] = op(DEC, ZeroPage, 5, false);
    t[0xD6] = op(DEC, ZeroPageX, 6, false);
    t[0xCE] = op(DEC, Absolute, 6, false);
    t[0xDE] = op(DEC, AbsoluteX, 7, false);
    t[0xCA] = op(DEX, Implied, 2, false);
    t[0x88] = op(DEY, Implied, 2, false);

    // --- EOR ---
    t[0x49] = op(EOR, Immediate, 2, false);
    t[0x45] = op(EOR, ZeroPage, 3, false);
    t[0x55] = op(EOR, ZeroPageX, 4, false);
    t[0x4D] = op(EOR, Absolute, 4, false);
    t[0x5D] = op(EOR, AbsoluteX, 4, true);
    t[0x59] = op(EOR, AbsoluteY, 4, true);
    t[0x41] = op(EOR, IndexedIndirect, 6, false);
    t[0x51] = op(EOR, IndirectIndexed, 5, true);

    // --- INC / INX / INY ---
    t[0xE6] = op(INC, ZeroPage, 5, false);
    t[0xF6] = op(INC, ZeroPageX, 6, false);
    t[0xEE] = op(INC, Absolute, 6, false);
    t[0xFE] = op(INC, AbsoluteX, 7, false);
    t[0xE8] = op(INX, Implied, 2, false);
    t[0xC8] = op(INY, Implied, 2, false);

    // --- JMP / JSR ---
    t[0x4C] = op(JMP, Absolute, 3, false);
    t[0x6C] = op(JMP, Indirect, 5, false);
    t[0x20] = op(JSR, JsrAbsolute, 6, false);

    // --- LDA ---
    t[0xA9] = op(LDA, Immediate, 2, false);
    t[0xA5] = op(LDA, ZeroPage, 3, false);
    t[0xB5] = op(LDA, ZeroPageX, 4, false);
    t[0xAD] = op(LDA, Absolute, 4, false);
    t[0xBD] = op(LDA, AbsoluteX, 4, true);
    t[0xB9] = op(LDA, AbsoluteY, 4, true);
    t[0xA1] = op(LDA, IndexedIndirect, 6, false);
    t[0xB1] = op(LDA, IndirectIndexed, 5, true);

    // --- LDX ---
    t[0xA2] = op(LDX, Immediate, 2, false);
    t[0xA6] = op(LDX, ZeroPage, 3, false);
    t[0xB6] = op(LDX, ZeroPageY, 4, false);
    t[0xAE] = op(LDX, Absolute, 4, false);
    t[0xBE] = op(LDX, AbsoluteY, 4, true);

    // --- LDY ---
    t[0xA0] = op(LDY, Immediate, 2, false);
    t[0xA4] = op(LDY, ZeroPage, 3, false);
    t[0xB4] = op(LDY, ZeroPageX, 4, false);
    t[0xAC] = op(LDY, Absolute, 4, false);
    t[0xBC] = op(LDY, AbsoluteX, 4, true);

    // --- LSR ---
    t[0x4A] = op(LSR, Accumulator, 2, false);
    t[0x46] = op(LSR, ZeroPage, 5, false);
    t[0x56] = op(LSR, ZeroPageX, 6, false);
    t[0x4E] = op(LSR, Absolute, 6, false);
    t[0x5E] = op(LSR, AbsoluteX, 7, false);

    // --- NOP ---
    t[0xEA] = op(NOP, Implied, 2, false);

    // --- ORA ---
    t[0x09] = op(ORA, Immediate, 2, false);
    t[0x05] = op(ORA, ZeroPage, 3, false);
    t[0x15] = op(ORA, ZeroPageX, 4, false);
    t[0x0D] = op(ORA, Absolute, 4, false);
    t[0x1D] = op(ORA, AbsoluteX, 4, true);
    t[0x19] = op(ORA, AbsoluteY, 4, true);
    t[0x01] = op(ORA, IndexedIndirect, 6, false);
    t[0x11] = op(ORA, IndirectIndexed, 5, true);

    // --- Stack ---
    t[0x48] = op(PHA, Implied, 3, false);
    t[0x08] = op(PHP, Implied, 3, false);
    t[0x68] = op(PLA, Implied, 4, false);
    t[0x28] = op(PLP, Implied, 4, false);

    // --- ROL / ROR ---
    t[0x2A] = op(ROL, Accumulator, 2, false);
    t[0x26] = op(ROL, ZeroPage, 5, false);
    t[0x36] = op(ROL, ZeroPageX, 6, false);
    t[0x2E] = op(ROL, Absolute, 6, false);
    t[0x3E] = op(ROL, AbsoluteX, 7, false);
    t[0x6A] = op(ROR, Accumulator, 2, false);
    t[0x66] = op(ROR, ZeroPage, 5, false);
    t[0x76] = op(ROR, ZeroPageX, 6, false);
    t[0x6E] = op(ROR, Absolute, 6, false);
    t[0x7E] = op(ROR, AbsoluteX, 7, false);

    // --- RTI / RTS ---
    t[0x40] = op(RTI, Implied, 6, false);
    t[0x60] = op(RTS, Implied, 6, false);

    // --- SBC ---
    t[0xE9] = op(SBC, Immediate, 2, false);
    t[0xE5] = op(SBC, ZeroPage, 3, false);
    t[0xF5] = op(SBC, ZeroPageX, 4, false);
    t[0xED] = op(SBC, Absolute, 4, false);
    t[0xFD] = op(SBC, AbsoluteX, 4, true);
    t[0xF9] = op(SBC, AbsoluteY, 4, true);
    t[0xE1] = op(SBC, IndexedIndirect, 6, false);
    t[0xF1] = op(SBC, IndirectIndexed, 5, true);

    // --- STA ---
    t[0x85] = op(STA, ZeroPage, 3, false);
    t[0x95] = op(STA, ZeroPageX, 4, false);
    t[0x8D] = op(STA, Absolute, 4, false);
    t[0x9D] = op(STA, AbsoluteX, 5, false);
    t[0x99] = op(STA, AbsoluteY, 5, false);
    t[0x81] = op(STA, IndexedIndirect, 6, false);
    t[0x91] = op(STA, IndirectIndexed, 6, false);

    // --- STX / STY ---
    t[0x86] = op(STX, ZeroPage, 3, false);
    t[0x96] = op(STX, ZeroPageY, 4, false);
    t[0x8E] = op(STX, Absolute, 4, false);
    t[0x84] = op(STY, ZeroPage, 3, false);
    t[0x94] = op(STY, ZeroPageX, 4, false);
    t[0x8C] = op(STY, Absolute, 4, false);

    // --- Transfers ---
    t[0xAA] = op(TAX, Implied, 2, false);
    t[0xA8] = op(TAY, Implied, 2, false);
    t[0xBA] = op(TSX, Implied, 2, false);
    t[0x8A] = op(TXA, Implied, 2, false);
    t[0x9A] = op(TXS, Implied, 2, false);
    t[0x98] = op(TYA, Implied, 2, false);

    t
}
