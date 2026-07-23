//! SingleStepTests harness for the ARM32 core (the `ARM7TDMI` suite,
//! captured from real hardware by the SingleStepTests project).
//!
//! Each JSON file holds up to 20,000 cases for one encoding category. A
//! case carries the full initial and final CPU state — all banked
//! registers, CPSR, the five SPSRs, and the prefetch pipeline — plus the
//! opcode and the bus transactions it performed. We install `initial`,
//! seed a sparse RAM from the read transactions (and the pipeline words),
//! execute one instruction, and assert the final registers and every
//! write. Cycle counts and bus-access ordering are NOT validated: the
//! suite captures per-cycle ARM7TDMI bus activity, which this core does
//! not model.
//!
//! The suite's R15 carries the visible pipeline offset (+8 in ARM state,
//! +4 in Thumb); this core keeps `gpr[15]` at the instruction itself, so
//! the harness translates on the way in and out.
//!
//! One knowing divergence (see `src/arm32/mod.rs`): the S-bit multiplies
//! leave C (and, for the long forms, V) unchanged, where hardware corrupts
//! them via the Booth multiplier internals. The harness masks C/V for
//! exactly those encodings and reports how many cases were masked.
//!
//! The data is large and is **not** vendored. The repository ships
//! `.json.bin` files plus a `transcode_json.py` script that converts them
//! to plain `.json`; point the harness at a directory of converted files:
//!
//! ```text
//! git clone --depth 1 https://github.com/SingleStepTests/ARM7TDMI.git
//! cd ARM7TDMI && python transcode_json.py   # produces v1/*.json
//! REMU_HARTE_ARM7_DIR=/path/to/ARM7TDMI/v1 cargo test --release --test harte_arm7tdmi
//! ```
//!
//! When the variable is unset the test prints a notice and passes, so CI
//! without the data stays green.

use std::collections::HashMap;
use std::path::PathBuf;

use remu::arm32::{Bus, Cpu, Mode, Psr};
use serde::Deserialize;

/// A sparse 4 GiB RAM: the suite uses random addresses across the whole
/// space, so a flat power-of-two RAM would alias.
#[derive(Default)]
struct SparseBus {
    mem: HashMap<u32, u8>,
}

impl Bus for SparseBus {
    fn read(&mut self, addr: u32) -> u8 {
        self.mem.get(&addr).copied().unwrap_or(0)
    }
    fn write(&mut self, addr: u32, value: u8) {
        self.mem.insert(addr, value);
    }
}

impl SparseBus {
    fn write_sized(&mut self, addr: u32, size: u8, data: u32) {
        for i in 0..size as u32 {
            self.write(addr.wrapping_add(i), (data >> (8 * i)) as u8);
        }
    }
    fn read_sized(&mut self, addr: u32, size: u8) -> u32 {
        (0..size as u32).fold(0, |acc, i| {
            acc | (self.read(addr.wrapping_add(i)) as u32) << (8 * i)
        })
    }
}

#[derive(Deserialize)]
struct State {
    #[serde(rename = "R")]
    r: [u32; 16],
    #[serde(rename = "R_fiq")]
    r_fiq: [u32; 7],
    #[serde(rename = "R_svc")]
    r_svc: [u32; 2],
    #[serde(rename = "R_abt")]
    r_abt: [u32; 2],
    #[serde(rename = "R_irq")]
    r_irq: [u32; 2],
    #[serde(rename = "R_und")]
    r_und: [u32; 2],
    #[serde(rename = "CPSR")]
    cpsr: u32,
    #[serde(rename = "SPSR")]
    spsr: [u32; 5],
    pipeline: [u32; 2],
}

#[derive(Deserialize)]
struct Transaction {
    /// 0 = instruction read, 1 = data read, 2 = write.
    kind: u8,
    size: u8,
    addr: u32,
    data: u32,
}

#[derive(Deserialize)]
struct Case {
    initial: State,
    #[serde(rename = "final")]
    end: State,
    opcode: u32,
    base_addr: u32,
    transactions: Vec<Transaction>,
}

/// SPSR bank order used by both the suite and `Registers`: FIQ, SVC, ABT,
/// IRQ, UND.
const SPSR_MODES: [Mode; 5] = [Mode::Fiq, Mode::Svc, Mode::Abt, Mode::Irq, Mode::Und];

/// Whether `opcode` (ARM state) is an S-bit multiply, whose C — and V for
/// the long forms — the hardware corrupts and this core leaves unchanged.
/// Returns the CPSR mask to *exclude* from comparison.
fn multiply_flag_mask(opcode: u32, thumb: bool) -> u32 {
    const C: u32 = 1 << 29;
    const V: u32 = 1 << 28;
    if thumb {
        // Format 4 MULS: 010000 1101 rs rd.
        if opcode & 0xFFC0 == 0x4340 {
            return C;
        }
        return 0;
    }
    if opcode & 0x0FD0_00F0 == 0x0010_0090 {
        return C; // MULS/MLAS
    }
    if opcode & 0x0F90_00F0 == 0x0090_0090 {
        return C | V; // UMULLS/UMLALS/SMULLS/SMLALS
    }
    0
}

/// Run one case; returns whether the multiply C/V mask was applied, or a
/// description of the first mismatch.
fn run_case(t: &Case, thumb: bool) -> Result<bool, String> {
    let mut bus = SparseBus::default();
    let pipe = if thumb { 2u32 } else { 4u32 };
    let exec_addr = t.initial.r[15].wrapping_sub(pipe * 2);

    // Seed the instruction stream: the opcode and the two prefetched
    // words behind it.
    let isize = if thumb { 2 } else { 4 };
    bus.write_sized(t.base_addr, isize, t.opcode);
    bus.write_sized(exec_addr, isize, t.opcode);
    bus.write_sized(exec_addr.wrapping_add(pipe), isize, t.initial.pipeline[0]);
    bus.write_sized(
        exec_addr.wrapping_add(pipe * 2),
        isize,
        t.initial.pipeline[1],
    );
    // Seed data reads (bus-level: aligned address, raw data).
    for tr in t.transactions.iter().filter(|tr| tr.kind == 1) {
        bus.write_sized(tr.addr & !(tr.size as u32 - 1), tr.size, tr.data);
    }

    // Install the initial state: banked registers first, then the mode.
    let mut cpu = Cpu::new();
    cpu.trap_faults = false;
    let cpsr = Psr::from_bits(t.initial.cpsr);
    cpu.regs.set_mode(cpsr.mode());
    cpu.regs.cpsr = cpsr;
    for i in 0..15 {
        cpu.regs.set_reg_of(Mode::Usr, i, t.initial.r[i]);
    }
    cpu.regs.gpr[15] = exec_addr;
    for (i, &v) in t.initial.r_fiq.iter().enumerate() {
        cpu.regs.set_reg_of(Mode::Fiq, 8 + i, v);
    }
    for (bank, vals) in [
        (Mode::Svc, &t.initial.r_svc),
        (Mode::Abt, &t.initial.r_abt),
        (Mode::Irq, &t.initial.r_irq),
        (Mode::Und, &t.initial.r_und),
    ] {
        cpu.regs.set_reg_of(bank, 13, vals[0]);
        cpu.regs.set_reg_of(bank, 14, vals[1]);
    }
    for (m, &v) in SPSR_MODES.iter().zip(&t.initial.spsr) {
        cpu.regs.set_spsr_of(*m, Psr::from_bits(v));
    }

    cpu.step(&mut bus);

    // Compare the final state. The suite's R15 carries the pipeline
    // offset of the *final* state — except after an MSR CPSR that flips
    // the T bit, which changes state without reloading the pipeline; the
    // dumped R15 still reflects the initial state's fetch width there.
    let msr_cpsr = !thumb
        && (t.opcode & 0x0FF0_FFF0 == 0x0120_F000 // register form (not BX!)
            || t.opcode & 0x0FF0_F000 == 0x0320_F000); // immediate form
    let end_thumb = if msr_cpsr {
        thumb
    } else {
        t.end.cpsr & (1 << 5) != 0
    };
    let end_pipe = if end_thumb { 2u32 } else { 4u32 };
    let want_pc = t.end.r[15].wrapping_sub(end_pipe * 2);
    if cpu.regs.gpr[15] != want_pc {
        return Err(format!(
            "PC: got {:#010X}, want {:#010X} (suite R15 {:#010X})",
            cpu.regs.gpr[15], want_pc, t.end.r[15]
        ));
    }
    let flag_mask = multiply_flag_mask(t.opcode, thumb);
    let got_cpsr = cpu.regs.cpsr.bits() & !flag_mask;
    let want_cpsr = Psr::from_bits(t.end.cpsr).bits() & !flag_mask;
    if got_cpsr != want_cpsr {
        return Err(format!(
            "CPSR: got {got_cpsr:#010X}, want {want_cpsr:#010X}"
        ));
    }
    for i in 0..15 {
        let got = cpu.regs.reg_of(Mode::Usr, i);
        if got != t.end.r[i] {
            return Err(format!("r{i}: got {got:#010X}, want {:#010X}", t.end.r[i]));
        }
    }
    for (i, &want) in t.end.r_fiq.iter().enumerate() {
        let got = cpu.regs.reg_of(Mode::Fiq, 8 + i);
        if got != want {
            return Err(format!(
                "r{}_fiq: got {got:#010X}, want {want:#010X}",
                8 + i
            ));
        }
    }
    for (bank, vals) in [
        (Mode::Svc, &t.end.r_svc),
        (Mode::Abt, &t.end.r_abt),
        (Mode::Irq, &t.end.r_irq),
        (Mode::Und, &t.end.r_und),
    ] {
        for (i, &want) in vals.iter().enumerate() {
            let got = cpu.regs.reg_of(bank, 13 + i);
            if got != want {
                return Err(format!(
                    "r{}_{bank:?}: got {got:#010X}, want {want:#010X}",
                    13 + i
                ));
            }
        }
    }
    for (m, &want) in SPSR_MODES.iter().zip(&t.end.spsr) {
        let got = cpu.regs.spsr_of(*m).bits();
        let want = Psr::from_bits(want).bits();
        if got != want {
            return Err(format!("SPSR_{m:?}: got {got:#010X}, want {want:#010X}"));
        }
    }
    // Every write transaction must have landed (bus-level: aligned).
    for tr in t.transactions.iter().filter(|tr| tr.kind == 2) {
        let addr = tr.addr & !(tr.size as u32 - 1);
        let got = bus.read_sized(addr, tr.size);
        let want = tr.data & (u32::MAX >> (32 - 8 * tr.size as u32));
        if got != want {
            return Err(format!(
                "mem[{addr:#010X}]/{}: got {got:#010X}, want {want:#010X}",
                tr.size
            ));
        }
    }
    Ok(flag_mask != 0)
}

#[test]
fn harte_arm7tdmi() {
    let Some(dir) = std::env::var_os("REMU_HARTE_ARM7_DIR") else {
        eprintln!(
            "REMU_HARTE_ARM7_DIR not set; skipping the SingleStepTests \
             ARM7TDMI suite (see this file's header for setup)."
        );
        return;
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("REMU_HARTE_ARM7_DIR is not readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no .json files in {dir:?} — run the repo's transcode_json.py first"
    );

    let (mut total, mut skipped, mut masked, mut failed_cases) = (0u64, 0u64, 0u64, 0u64);
    let mut failed_files = Vec::new();
    for path in &files {
        let data = std::fs::read(path).expect("test file unreadable");
        let cases: Vec<Case> =
            serde_json::from_slice(&data).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let mut file_failed = 0u64;
        for (i, case) in cases.iter().enumerate() {
            total += 1;
            let thumb = case.initial.cpsr & (1 << 5) != 0;
            // A handful of cases read or write the instruction's own
            // memory: the suite's CPU had the opcode in its pipeline
            // already, so its data access sees the canned RAM value. A
            // pipeline-less interpreter fetches and reads the same byte —
            // unresolvable, so these are skipped (and counted).
            let isz = if thumb { 2u32 } else { 4 };
            let exec = case.initial.r[15].wrapping_sub(isz * 2);
            if case.transactions.iter().any(|tr| {
                let a = tr.addr & !(tr.size as u32 - 1);
                tr.kind != 0 && a < exec.wrapping_add(isz) && a.wrapping_add(tr.size as u32) > exec
            }) {
                skipped += 1;
                continue;
            }
            match run_case(case, thumb) {
                Ok(true) => masked += 1,
                Ok(false) => {}
                Err(msg) => {
                    failed_cases += 1;
                    file_failed += 1;
                    if file_failed <= 3 {
                        eprintln!(
                            "{}[{}] opcode {:#010X}: {}",
                            path.file_name().unwrap().to_string_lossy(),
                            i,
                            case.opcode,
                            msg
                        );
                    }
                }
            }
        }
        if file_failed > 0 {
            failed_files.push(format!(
                "{}: {file_failed}/{}",
                path.file_name().unwrap().to_string_lossy(),
                cases.len()
            ));
        }
    }
    eprintln!(
        "ARM7TDMI suite: {total} cases, {failed_cases} failed, \
         {skipped} skipped (self-referencing memory), \
         {masked} compared with multiply C/V masked"
    );
    assert_eq!(
        failed_cases,
        0,
        "failing files:\n{}",
        failed_files.join("\n")
    );
}
