//! Thin frontend: load a raw 6502 binary into a flat 64 KiB memory and run it,
//! optionally tracing each instruction.
//!
//! Usage:
//!   remu-cli <program.bin> [--load <hex_addr>] [--start <hex_addr>] [--trace] [--steps <n>]
//!
//! Defaults: load address `$0600`, start at the load address, 1,000,000 steps.

use std::process::ExitCode;

use remu::Cpu;
use remu::cpu::disasm::disassemble;
use remu::memory::FlatMemory;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <program.bin> [--load <hex>] [--start <hex>] [--trace] [--steps <n>]",
            args.first().map(String::as_str).unwrap_or("remu-cli")
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
                load_addr = match parse_hex("--load", args.get(i)) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("{e}");
                        return ExitCode::FAILURE;
                    }
                };
            }
            "--start" => {
                i += 1;
                start_addr = match parse_hex("--start", args.get(i)) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!("{e}");
                        return ExitCode::FAILURE;
                    }
                };
            }
            "--trace" => trace = true,
            "--steps" => {
                i += 1;
                max_steps = match args.get(i).and_then(|s| s.parse().ok()) {
                    Some(n) => n,
                    None => {
                        eprintln!("--steps requires a decimal step count");
                        return ExitCode::FAILURE;
                    }
                };
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
            // PC has already advanced past the KIL opcode byte.
            let jam_addr = cpu.regs.pc.wrapping_sub(1);
            println!(
                "halted (KIL) at ${jam_addr:04X} after {} cycles",
                cpu.cycles
            );
            break;
        }
        if trace {
            let (text, _) = disassemble(&mut mem, cpu.regs.pc);
            println!(
                "{text:<20} A:{:02X} X:{:02X} Y:{:02X} SP:{:02X} P:{:02X}",
                cpu.regs.a,
                cpu.regs.x,
                cpu.regs.y,
                cpu.regs.sp,
                cpu.regs.p.bits()
            );
        }
        cpu.step(&mut mem);
    }

    ExitCode::SUCCESS
}

fn parse_hex(flag: &str, s: Option<&String>) -> Result<u16, String> {
    let s = s.ok_or_else(|| format!("{flag} requires a hex address"))?;
    u16::from_str_radix(s.trim_start_matches("0x").trim_start_matches('$'), 16)
        .map_err(|_| format!("{flag}: invalid hex address `{s}`"))
}
