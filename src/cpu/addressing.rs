//! Addressing modes and operand resolution.
//!
//! Each [`AddressingMode`] knows how to turn the bytes following an opcode into an
//! effective address (and whether resolving it crossed a page boundary, for the
//! cycle penalty). Instruction handlers then read/write through that address, so
//! every operation is written once regardless of how it's addressed.

use crate::bus::Bus;
use crate::cpu::Cpu;

/// How an instruction locates its operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressingMode {
    /// No operand (e.g. `INX`, `NOP`).
    Implied,
    /// Operates on the accumulator (e.g. `ASL A`).
    Accumulator,
    /// Operand is the next byte (`LDA #$nn`).
    Immediate,
    /// Zero-page address (`LDA $nn`).
    ZeroPage,
    /// Zero-page address indexed by X, wrapping within the page (`LDA $nn,X`).
    ZeroPageX,
    /// Zero-page address indexed by Y, wrapping within the page (`LDX $nn,Y`).
    ZeroPageY,
    /// Absolute 16-bit address (`LDA $nnnn`).
    Absolute,
    /// Absolute address indexed by X (`LDA $nnnn,X`).
    AbsoluteX,
    /// Absolute address indexed by Y (`LDA $nnnn,Y`).
    AbsoluteY,
    /// Indirect, used only by `JMP ($nnnn)` — includes the page-boundary bug.
    Indirect,
    /// `JSR`'s absolute target: the handler fetches it itself, interleaved with
    /// the stack pushes, so `resolve` leaves `PC` untouched.
    JsrAbsolute,
    /// Indexed indirect `($nn,X)`: add X to the zero-page pointer, then deref.
    IndexedIndirect,
    /// Indirect indexed `($nn),Y`: deref the zero-page pointer, then add Y.
    IndirectIndexed,
    /// PC-relative signed offset, for branches.
    Relative,
}

impl AddressingMode {
    /// Total instruction length in bytes for this mode (opcode included).
    pub const fn length(&self) -> u8 {
        use AddressingMode::*;
        match self {
            Implied | Accumulator => 1,
            Immediate | ZeroPage | ZeroPageX | ZeroPageY | IndexedIndirect
            | IndirectIndexed | Relative => 2,
            Absolute | AbsoluteX | AbsoluteY | Indirect | JsrAbsolute => 3,
        }
    }
}

/// A resolved operand: the effective address plus whether resolving it crossed a
/// page boundary (only meaningful for indexed reads and branches).
#[derive(Debug, Clone, Copy)]
pub struct Operand {
    /// The effective address. Meaningless for `Implied`/`Accumulator`; for
    /// `Immediate` it points at the operand byte in the instruction stream; for
    /// `Relative` it is the branch target.
    pub addr: u16,
    /// Whether indexing carried into a new page (the `+1` cycle penalty).
    pub page_crossed: bool,
}

impl Operand {
    const fn at(addr: u16) -> Self {
        Operand { addr, page_crossed: false }
    }
}

impl Cpu {
    /// Resolve the operand for `mode`, advancing `PC` past the operand bytes.
    ///
    /// `inline(always)`: the fused dispatch handlers call this with `mode` a
    /// compile-time constant, and inlining lets the `match` fold to one arm.
    #[inline(always)]
    pub(crate) fn resolve<B: Bus>(&mut self, bus: &mut B, mode: AddressingMode) -> Operand {
        use AddressingMode::*;
        match mode {
            Implied | Accumulator => Operand::at(0),

            // JSR fetches its own operand (see `Cpu::jsr`).
            JsrAbsolute => Operand::at(0),

            Immediate => {
                let addr = self.regs.pc;
                self.regs.pc = self.regs.pc.wrapping_add(1);
                Operand::at(addr)
            }

            ZeroPage => Operand::at(self.fetch_byte(bus) as u16),

            ZeroPageX => {
                let base = self.fetch_byte(bus);
                Operand::at(base.wrapping_add(self.regs.x) as u16)
            }

            ZeroPageY => {
                let base = self.fetch_byte(bus);
                Operand::at(base.wrapping_add(self.regs.y) as u16)
            }

            Absolute => Operand::at(self.fetch_word(bus)),

            AbsoluteX => {
                let base = self.fetch_word(bus);
                let addr = base.wrapping_add(self.regs.x as u16);
                Operand { addr, page_crossed: page_crossed(base, addr) }
            }

            AbsoluteY => {
                let base = self.fetch_word(bus);
                let addr = base.wrapping_add(self.regs.y as u16);
                Operand { addr, page_crossed: page_crossed(base, addr) }
            }

            Indirect => {
                // JMP ($nnnn) — the 6502 fails to carry into the high byte when
                // the pointer's low byte is $FF, reading the high byte from the
                // start of the same page instead.
                let ptr = self.fetch_word(bus);
                let lo = self.read(bus, ptr) as u16;
                let hi = self.read(bus, (ptr & 0xFF00) | (ptr.wrapping_add(1) & 0x00FF)) as u16;
                Operand::at(lo | (hi << 8))
            }

            IndexedIndirect => {
                // ($nn,X): pointer arithmetic wraps within the zero page.
                let base = self.fetch_byte(bus);
                let ptr = base.wrapping_add(self.regs.x);
                let lo = self.read(bus, ptr as u16) as u16;
                let hi = self.read(bus, ptr.wrapping_add(1) as u16) as u16;
                Operand::at(lo | (hi << 8))
            }

            IndirectIndexed => {
                // ($nn),Y: deref the zero-page pointer (wrapping), then add Y.
                let base = self.fetch_byte(bus);
                let lo = self.read(bus, base as u16) as u16;
                let hi = self.read(bus, base.wrapping_add(1) as u16) as u16;
                let pointer = lo | (hi << 8);
                let addr = pointer.wrapping_add(self.regs.y as u16);
                Operand { addr, page_crossed: page_crossed(pointer, addr) }
            }

            Relative => {
                let offset = self.fetch_byte(bus) as i8 as u16;
                let base = self.regs.pc;
                let addr = base.wrapping_add(offset);
                Operand { addr, page_crossed: page_crossed(base, addr) }
            }
        }
    }
}

/// Whether two addresses lie in different 256-byte pages.
#[inline]
fn page_crossed(a: u16, b: u16) -> bool {
    (a & 0xFF00) != (b & 0xFF00)
}
