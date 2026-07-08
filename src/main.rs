//! Thin frontend: load a raw 6502 binary into a flat 64 KiB memory and run it,
//! optionally tracing each instruction.
//!
//! Usage:
//!   remu <program.bin> [--load <hex_addr>] [--start <hex_addr>] [--trace] [--steps <n>]
//!
//! Defaults: load address `$0600`, start at the load address, 1,000,000 steps.

use std::process::ExitCode;

use remu::cpu::disasm::disassemble;
use remu::memory::FlatMemory;
use remu::Cpu;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <program.bin> [--load <hex>] [--start <hex>] [--trace] [--steps <n>]",
            args.first().map(String::as_str).unwrap_or("remu")
        );
        return ExitCode::FAILURE;
    }

    let path = &args[1];
    let mut load_addr: u16 = 0x0600;
    let mut start_addr: Option<u16> = None;
    let mut trace = false;
    let mut max_steps: u64 = 1_000_000;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--load" => {
                i += 1;
                load_addr = parse_hex(args.get(i));
            }
            "--start" => {
                i += 1;
                start_addr = Some(parse_hex(args.get(i)));
            }
            "--trace" => trace = true,
            "--steps" => {
                i += 1;
                max_steps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(max_steps);
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let program = match std::fs::read(path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut mem = FlatMemory::new();
    mem.load(load_addr, &program);
    let start = start_addr.unwrap_or(load_addr);
    mem.set_reset_vector(start);

    let mut cpu = Cpu::new();
    cpu.reset(&mut mem);

    for _ in 0..max_steps {
        if cpu.halted {
            println!("halted (KIL) at ${:04X} after {} cycles", cpu.regs.pc, cpu.cycles);
            break;
        }
        if trace {
            let (text, _) = disassemble(&mut mem, cpu.regs.pc);
            println!(
                "{text:<20} A:{:02X} X:{:02X} Y:{:02X} SP:{:02X} P:{:02X}",
                cpu.regs.a, cpu.regs.x, cpu.regs.y, cpu.regs.sp, cpu.regs.p.bits()
            );
        }
        cpu.step(&mut mem);
    }

    ExitCode::SUCCESS
}

fn parse_hex(s: Option<&String>) -> u16 {
    s.and_then(|s| u16::from_str_radix(s.trim_start_matches("0x").trim_start_matches('$'), 16).ok())
        .unwrap_or(0)
}
