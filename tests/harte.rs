//! Tom Harte / SingleStepTests harness for the 6502 (the `65x02/nes6502` or
//! `6502` JSON suite).
//!
//! Each JSON file is named after an opcode (`a9.json`) and contains thousands of
//! cases, each with an initial CPU+RAM state, the expected final state, and a
//! per-cycle bus access list. We set the CPU to `initial`, run one `step()`, and
//! assert the final registers, RAM, and total cycle count.
//!
//! The data is large (tens of MB) and is **not** vendored. Point the test at a
//! local checkout via the `REMU_HARTE_DIR` environment variable:
//!
//! ```text
//! REMU_HARTE_DIR=/path/to/tests/6502/v1 cargo test --release --test harte
//! ```
//!
//! When the variable is unset the test prints a notice and passes, so CI without
//! the data stays green.
//!
//! The stock suite has one file per all 256 opcodes; only files for opcodes we
//! implement are run — undocumented ones decode to `KIL` here and would fail
//! their randomized cases.

use std::path::PathBuf;

use remu::cpu::opcodes::{OPCODES, Operation};
use remu::memory::FlatMemory;
use remu::{Cpu, Status};
use serde::Deserialize;

#[derive(Deserialize)]
struct State {
    pc: u16,
    s: u8,
    a: u8,
    x: u8,
    y: u8,
    p: u8,
    ram: Vec<(u16, u8)>,
}

#[derive(Deserialize)]
struct TestCase {
    name: String,
    initial: State,
    #[serde(rename = "final")]
    final_state: State,
    cycles: Vec<(u16, u8, String)>,
}

/// Run every case in one opcode file. Returns the number of failures (0 = pass).
fn run_file(path: &PathBuf) -> usize {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let cases: Vec<TestCase> = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));

    let mut failures = 0;
    // One flat memory reused across cases: each case writes its bytes up front
    // and zeroes every address it touched afterwards (initial RAM plus every
    // bus access the instruction made).
    let mut mem = FlatMemory::new();
    for case in &cases {
        for &(addr, val) in &case.initial.ram {
            mem.ram[addr as usize] = val;
        }

        let mut cpu = Cpu::new();
        cpu.regs.pc = case.initial.pc;
        cpu.regs.a = case.initial.a;
        cpu.regs.x = case.initial.x;
        cpu.regs.y = case.initial.y;
        cpu.regs.sp = case.initial.s;
        cpu.regs.p = Status::from_bits_retain(case.initial.p);
        cpu.cycles = 0;

        let cycles = cpu.step(&mut mem);

        let mut ok = cpu.regs.pc == case.final_state.pc
            && cpu.regs.a == case.final_state.a
            && cpu.regs.x == case.final_state.x
            && cpu.regs.y == case.final_state.y
            && cpu.regs.sp == case.final_state.s
            && cpu.regs.p.bits() == case.final_state.p
            && cycles as usize == case.cycles.len();

        if ok {
            for &(addr, val) in &case.final_state.ram {
                if mem.ram[addr as usize] != val {
                    ok = false;
                    break;
                }
            }
        }

        if !ok {
            if failures < 5 {
                eprintln!(
                    "FAIL {} [{}]:\n  exp pc:{:04X} a:{:02X} x:{:02X} y:{:02X} s:{:02X} p:{:02X} cyc:{}\n  got pc:{:04X} a:{:02X} x:{:02X} y:{:02X} s:{:02X} p:{:02X} cyc:{}",
                    path.file_name().unwrap().to_string_lossy(),
                    case.name,
                    case.final_state.pc,
                    case.final_state.a,
                    case.final_state.x,
                    case.final_state.y,
                    case.final_state.s,
                    case.final_state.p,
                    case.cycles.len(),
                    cpu.regs.pc,
                    cpu.regs.a,
                    cpu.regs.x,
                    cpu.regs.y,
                    cpu.regs.sp,
                    cpu.regs.p.bits(),
                    cycles,
                );
            }
            failures += 1;
        }

        for &(addr, _) in &case.initial.ram {
            mem.ram[addr as usize] = 0;
        }
        for &(addr, _, _) in &case.cycles {
            mem.ram[addr as usize] = 0;
        }
    }
    failures
}

#[test]
fn harte_suite() {
    let Ok(dir) = std::env::var("REMU_HARTE_DIR") else {
        eprintln!("REMU_HARTE_DIR not set — skipping Tom Harte suite.");
        return;
    };

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .filter(|p| {
            // Keep only files named after an opcode we implement.
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| u8::from_str_radix(s, 16).ok())
                .map(|op| OPCODES[op as usize].operation != Operation::KIL)
                .unwrap_or(false)
        })
        .collect();
    entries.sort();

    let mut total_failures = 0;
    let mut failed_opcodes = Vec::new();
    for path in &entries {
        let f = run_file(path);
        if f > 0 {
            total_failures += f;
            failed_opcodes.push(format!(
                "{} ({f})",
                path.file_name().unwrap().to_string_lossy()
            ));
        }
    }

    if total_failures > 0 {
        panic!(
            "{} failing cases across {} opcode files: {}",
            total_failures,
            failed_opcodes.len(),
            failed_opcodes.join(", ")
        );
    }
    eprintln!(
        "Tom Harte suite passed: {} official-opcode files.",
        entries.len()
    );
}
