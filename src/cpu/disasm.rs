//! A small disassembler over the same [`OPCODES`] table the interpreter uses.
//!
//! Invaluable when a test case fails and you want to see which instruction blew
//! up. It reads through a [`Bus`], so it works against any memory implementation.

use crate::bus::Bus;
use crate::cpu::addressing::AddressingMode;
use crate::cpu::opcodes::OPCODES;

/// Disassemble the instruction at `addr`, returning the formatted text and the
/// address of the following instruction.
pub fn disassemble<B: Bus>(bus: &mut B, addr: u16) -> (String, u16) {
    use AddressingMode::*;

    let opcode = bus.read(addr);
    let info = OPCODES[opcode as usize];
    let mnemonic = format!("{:?}", info.operation);

    // Read only the operand bytes the instruction actually has: bus reads can
    // have device side effects, so a phantom read would perturb the traced run.
    let len = info.length();
    let b1 = if len >= 2 { bus.read(addr.wrapping_add(1)) } else { 0 };
    let b2 = if len >= 3 { bus.read(addr.wrapping_add(2)) } else { 0 };
    let word = (b1 as u16) | ((b2 as u16) << 8);

    let operand = match info.mode {
        Implied => String::new(),
        Accumulator => "A".to_string(),
        Immediate => format!("#${:02X}", b1),
        ZeroPage => format!("${:02X}", b1),
        ZeroPageX => format!("${:02X},X", b1),
        ZeroPageY => format!("${:02X},Y", b1),
        Absolute | JsrAbsolute => format!("${:04X}", word),
        AbsoluteX => format!("${:04X},X", word),
        AbsoluteY => format!("${:04X},Y", word),
        Indirect => format!("(${:04X})", word),
        IndexedIndirect => format!("(${:02X},X)", b1),
        IndirectIndexed => format!("(${:02X}),Y", b1),
        Relative => {
            let target = addr
                .wrapping_add(2)
                .wrapping_add(b1 as i8 as u16);
            format!("${:04X}", target)
        }
    };

    let text = if operand.is_empty() {
        format!("{:04X}  {}", addr, mnemonic)
    } else {
        format!("{:04X}  {} {}", addr, mnemonic, operand)
    };

    (text, addr.wrapping_add(info.length() as u16))
}
