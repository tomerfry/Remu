//! Wall-clock throughput of the Remu x86 cores, paired with
//! `examples/bench_unicorn.py` which runs the *byte-identical* programs on
//! Unicorn Engine (QEMU/TCG). Together they answer "how fast is Remu next to
//! Unicorn?" in millions of emulated instructions per second (MIPS).
//!
//! ```text
//! cargo run --release --example bench_x86
//! python examples/bench_unicorn.py
//! ```
//!
//! Methodology
//! - Each program initializes every register it uses, loops a statically
//!   known number of times, and falls through to its end address, so the
//!   instruction count per run is exact and identical on both engines.
//! - Remu steps exactly `instructions` times and asserts it landed on the
//!   end address; Unicorn runs `emu_start(begin, end)`. Only the execution
//!   loop is timed.
//! - `RUNS` timed runs per workload (registers persist across runs on both
//!   engines); the best run is reported, so Unicorn's numbers are for a warm
//!   translation cache.
//! - Both harnesses print one TSV row per workload:
//!   `engine  mode  name  instructions  best_s  mips  run_s,...  check`
//!   where `check` is a final register value (AX/EAX/RAX family) that must
//!   match between the engines — a cross-emulator correctness check.
//!
//! Modes: 16-bit real (8086 core), 32-bit flat protected (386 core), and
//! 64-bit long mode (x86-64 core, identity paging via `setup_long_flat`).

use remu::{x86, x86_32, x86_64};
use std::time::Instant;

/// Timed runs per workload; best is reported.
const RUNS: usize = 5;

/// Inner-loop trip count for the 16-bit workloads (fits CX).
const INNER16: u64 = 0xFFFF;
/// Outer trip counts, sized so every 16-bit workload is ~10M instructions.
const TIGHT16_OUTER: u64 = 80;
const ALU16_OUTER: u64 = 25;
const MEM16_OUTER: u64 = 25;
const CALL16_OUTER: u64 = 30;

/// Loop trip counts for the 32/64-bit workloads (imm32 in the code bytes).
const TIGHT_N: u64 = 20_000_000; // 0x01312D00
const ALU_N: u64 = 3_000_000; // 0x002DC6C0
const MEM_N: u64 = 4_000_000; // 0x003D0900
const CALL_N: u64 = 4_000_000;

/// Source-buffer fill for the mem_rw workloads, identical in both harnesses.
fn pat(i: usize) -> u8 {
    (i.wrapping_mul(37).wrapping_add(11)) as u8
}

/// Print one TSV result row (same shape as the Unicorn harness).
fn report(mode: &str, name: &str, instructions: u64, times: &[f64], check: u64) {
    let best = times.iter().copied().fold(f64::INFINITY, f64::min);
    let mips = instructions as f64 / best / 1e6;
    let runs: Vec<String> = times.iter().map(|t| format!("{t:.4}")).collect();
    println!(
        "remu\t{mode}\t{name}\t{instructions}\t{best:.4}\t{mips:.1}\t{}\t{check:#x}",
        runs.join(",")
    );
}

// --- 16-bit real mode (8086 core), programs at 0000:1000 --------------------

#[rustfmt::skip]
const TIGHT16: &[u8] = &[
    0xBA, 0x50, 0x00,       // 1000: MOV DX, 80
    0xB9, 0xFF, 0xFF,       // 1003: MOV CX, 0xFFFF
    0x49,                   // 1006: DEC CX
    0x75, 0xFD,             // 1007: JNZ 1006
    0x4A,                   // 1009: DEC DX
    0x75, 0xF7,             // 100A: JNZ 1003
]; // 100C: end

#[rustfmt::skip]
const ALU16: &[u8] = &[
    0xB8, 0x34, 0x12,       // 1000: MOV AX, 0x1234
    0xBA, 0x19, 0x00,       // 1003: MOV DX, 25
    0xB9, 0xFF, 0xFF,       // 1006: MOV CX, 0xFFFF
    0x05, 0xB9, 0x79,       // 1009: ADD AX, 0x79B9
    0x35, 0x5A, 0x5A,       // 100C: XOR AX, 0x5A5A
    0xD1, 0xC0,             // 100F: ROL AX, 1
    0x29, 0xC8,             // 1011: SUB AX, CX
    0x49,                   // 1013: DEC CX
    0x75, 0xF3,             // 1014: JNZ 1009
    0x4A,                   // 1016: DEC DX
    0x75, 0xED,             // 1017: JNZ 1006
]; // 1019: end

#[rustfmt::skip]
const MEM16: &[u8] = &[
    0xBA, 0x19, 0x00,       // 1000: MOV DX, 25
    0xBE, 0x00, 0x20,       // 1003: MOV SI, 0x2000 (src)
    0xBF, 0x00, 0x30,       // 1006: MOV DI, 0x3000 (dst)
    0xB9, 0xFF, 0xFF,       // 1009: MOV CX, 0xFFFF
    0x89, 0xCB,             // 100C: MOV BX, CX
    0x81, 0xE3, 0xFF, 0x03, // 100E: AND BX, 0x3FF
    0x8B, 0x00,             // 1012: MOV AX, [BX+SI]
    0x89, 0x01,             // 1014: MOV [BX+DI], AX
    0x49,                   // 1016: DEC CX
    0x75, 0xF3,             // 1017: JNZ 100C
    0x4A,                   // 1019: DEC DX
    0x75, 0xED,             // 101A: JNZ 1009
]; // 101C: end

#[rustfmt::skip]
const CALL16: &[u8] = &[
    0xBC, 0x00, 0xFF,       // 1000: MOV SP, 0xFF00
    0xBA, 0x1E, 0x00,       // 1003: MOV DX, 30
    0xB9, 0xFF, 0xFF,       // 1006: MOV CX, 0xFFFF
    0xE8, 0x08, 0x00,       // 1009: CALL 1014
    0x49,                   // 100C: DEC CX
    0x75, 0xFA,             // 100D: JNZ 1009
    0x4A,                   // 100F: DEC DX
    0x75, 0xF4,             // 1010: JNZ 1006
    0xEB, 0x02,             // 1012: JMP 1016
    0x40,                   // 1014: INC AX
    0xC3,                   // 1015: RET
]; // 1016: end

fn bench_8086(name: &str, program: &[u8], instructions: u64, check: fn(&x86::Cpu) -> u64) {
    let mut mem = x86::LinearMemory::new();
    mem.load(0x1000, program);
    let src: Vec<u8> = (0..0x404).map(pat).collect();
    mem.load(0x2000, &src);

    let mut cpu = x86::Cpu::new();
    cpu.regs.cs = 0x0000;
    cpu.regs.ds = 0x0000;
    cpu.regs.es = 0x0000;
    cpu.regs.ss = 0x0000;

    let end = 0x1000 + program.len() as u16;
    let mut times = Vec::new();
    for _ in 0..RUNS {
        cpu.regs.ip = 0x1000;
        let t = Instant::now();
        for _ in 0..instructions {
            cpu.step(&mut mem);
        }
        times.push(t.elapsed().as_secs_f64());
        assert_eq!(cpu.regs.ip, end, "{name}: did not land on the end address");
        assert!(!cpu.halted, "{name}: halted");
    }
    report("16", name, instructions, &times, check(&cpu));
}

// --- 32-bit flat protected mode (386 core), programs at 0x1000 --------------

#[rustfmt::skip]
const TIGHT32: &[u8] = &[
    0xB9, 0x00, 0x2D, 0x31, 0x01, // 1000: MOV ECX, 20000000
    0x49,                         // 1005: DEC ECX
    0x75, 0xFD,                   // 1006: JNZ 1005
]; // 1008: end

#[rustfmt::skip]
const ALU32: &[u8] = &[
    0xB9, 0xC0, 0xC6, 0x2D, 0x00, // 1000: MOV ECX, 3000000
    0xB8, 0x78, 0x56, 0x34, 0x12, // 1005: MOV EAX, 0x12345678
    0x05, 0xB9, 0x79, 0x37, 0x9E, // 100A: ADD EAX, 0x9E3779B9
    0x35, 0x5A, 0x5A, 0x5A, 0x5A, // 100F: XOR EAX, 0x5A5A5A5A
    0xC1, 0xC0, 0x07,             // 1014: ROL EAX, 7
    0x0F, 0xAF, 0xC0,             // 1017: IMUL EAX, EAX
    0x29, 0xC8,                   // 101A: SUB EAX, ECX
    0x49,                         // 101C: DEC ECX
    0x75, 0xEB,                   // 101D: JNZ 100A
]; // 101F: end

#[rustfmt::skip]
const MEM32: &[u8] = &[
    0xB9, 0x00, 0x09, 0x3D, 0x00,       // 1000: MOV ECX, 4000000
    0xBE, 0x00, 0x00, 0x10, 0x00,       // 1005: MOV ESI, 0x100000 (src)
    0xBF, 0x00, 0x10, 0x10, 0x00,       // 100A: MOV EDI, 0x101000 (dst)
    0x89, 0xCA,                         // 100F: MOV EDX, ECX
    0x81, 0xE2, 0xFF, 0x03, 0x00, 0x00, // 1011: AND EDX, 0x3FF
    0x8B, 0x04, 0x96,                   // 1017: MOV EAX, [ESI+EDX*4]
    0x89, 0x04, 0x97,                   // 101A: MOV [EDI+EDX*4], EAX
    0x49,                               // 101D: DEC ECX
    0x75, 0xEF,                         // 101E: JNZ 100F
]; // 1020: end

#[rustfmt::skip]
const CALL32: &[u8] = &[
    0xBC, 0x00, 0x00, 0x20, 0x00, // 1000: MOV ESP, 0x200000
    0xB9, 0x00, 0x09, 0x3D, 0x00, // 1005: MOV ECX, 4000000
    0xE8, 0x05, 0x00, 0x00, 0x00, // 100A: CALL 1014
    0x49,                         // 100F: DEC ECX
    0x75, 0xF8,                   // 1010: JNZ 100A
    0xEB, 0x02,                   // 1012: JMP 1016
    0x40,                         // 1014: INC EAX
    0xC3,                         // 1015: RET
]; // 1016: end

/// 50,000 × `REP MOVSD` of 4 KiB (≈205 MB copied). A REP counts as one
/// instruction in both harnesses, so the "MIPS" column is a relative
/// string-throughput number, not literal instructions — QEMU inlines
/// string ops aggressively and this is the honest worst case.
const REP_OUTER: u64 = 50_000;
#[rustfmt::skip]
const REP32: &[u8] = &[
    0xBA, 0x50, 0xC3, 0x00, 0x00, // 1000: MOV EDX, 50000
    0xBE, 0x00, 0x00, 0x10, 0x00, // 1005: MOV ESI, 0x100000 (src)
    0xBF, 0x00, 0x40, 0x10, 0x00, // 100A: MOV EDI, 0x104000 (dst)
    0xB9, 0x00, 0x04, 0x00, 0x00, // 100F: MOV ECX, 1024
    0xF3, 0xA5,                   // 1014: REP MOVSD
    0x4A,                         // 1016: DEC EDX
    0x75, 0xEC,                   // 1017: JNZ 1005
    0xA1, 0xFC, 0x4F, 0x10, 0x00, // 1019: MOV EAX, [0x104FFC] (last dword)
]; // 101E: end

fn bench_386(name: &str, program: &[u8], instructions: u64, check: fn(&x86_32::Cpu) -> u64) {
    bench_386_inner(name, program, instructions, false, false, check);
}

/// The same 386 rig with CR0.PG set and identity 4 KiB page tables for the
/// low 4 MiB (accessed/dirty preset), exercising the TLB paths.
fn bench_386_paged(name: &str, program: &[u8], instructions: u64, check: fn(&x86_32::Cpu) -> u64) {
    bench_386_inner(name, program, instructions, true, false, check);
}

/// The same 386 rig at CPL 3 (flat user-privilege segments, paging off),
/// exercising the user-mode protection checks the os/usermode layers pay.
fn bench_386_ring3(name: &str, program: &[u8], instructions: u64, check: fn(&x86_32::Cpu) -> u64) {
    bench_386_inner(name, program, instructions, false, true, check);
}

fn bench_386_inner(
    name: &str,
    program: &[u8],
    instructions: u64,
    paging: bool,
    ring3: bool,
    check: fn(&x86_32::Cpu) -> u64,
) {
    use x86_32::{SegReg, cr0, reg};

    let mut mem = x86_32::LinearMemory::new();
    mem.load(0x1000, program);
    let src: Vec<u8> = (0..0x1008).map(pat).collect();
    mem.load(0x10_0000, &src);

    // Flat protected mode, entered by loading the caches directly (same
    // trick as the usermode layer); ring 0 unless `ring3`.
    let mut cpu = x86_32::Cpu::new();
    cpu.regs.cr0 |= cr0::PE;
    if paging {
        // PD at 3 MiB, one PT mapping 0..4 MiB identity (P|RW|US|A|D so the
        // steady state performs no A/D write-backs). Same tables as the
        // Unicorn harness.
        mem.load(0x30_0000, &(0x30_1000u32 | 0x67).to_le_bytes());
        for page in 0..1024u32 {
            let pte = (page << 12) | 0x67;
            mem.load(0x30_1000 + page * 4, &pte.to_le_bytes());
        }
        cpu.regs.cr3 = 0x30_0000;
        cpu.regs.cr0 |= cr0::PG;
    }
    let rpl = if ring3 { 3 } else { 0 };
    let dpl = if ring3 { 0x60 } else { 0 };
    cpu.regs.seg[reg::CS as usize] = SegReg {
        sel: 0x08 | rpl,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0C9B | dpl, // present, code, exec/read, accessed; G+D
    };
    for s in [reg::SS, reg::DS, reg::ES, reg::FS, reg::GS] {
        cpu.regs.seg[s as usize] = SegReg {
            sel: 0x10 | rpl,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: 0x0C93 | dpl, // present, data, read/write, accessed; G+D
        };
    }

    let end = 0x1000 + program.len() as u32;
    let mut times = Vec::new();
    for _ in 0..RUNS {
        cpu.regs.eip = 0x1000;
        let t = Instant::now();
        let r = cpu.run(&mut mem, instructions);
        times.push(t.elapsed().as_secs_f64());
        assert_eq!(
            r.executed, instructions,
            "{name}: run() stopped early ({:?})",
            r.exit
        );
        assert_eq!(cpu.regs.eip, end, "{name}: did not land on the end address");
        assert!(!cpu.halted && !cpu.shutdown, "{name}: halted/shutdown");
    }
    report("32", name, instructions, &times, check(&cpu));
    #[cfg(feature = "perf-stats")]
    print!("{}", cpu.stats.report());
}

// --- 64-bit long mode (x86-64 core), programs at 0x10000 --------------------

#[rustfmt::skip]
const TIGHT64: &[u8] = &[
    0x48, 0xC7, 0xC1, 0x00, 0x2D, 0x31, 0x01, // 10000: MOV RCX, 20000000
    0x48, 0xFF, 0xC9,                         // 10007: DEC RCX
    0x75, 0xFB,                               // 1000A: JNZ 10007
]; // 1000C: end

#[rustfmt::skip]
const ALU64: &[u8] = &[
    0x48, 0xC7, 0xC1, 0xC0, 0xC6, 0x2D, 0x00,                   // 10000: MOV RCX, 3000000
    0x48, 0xB8, 0x78, 0x56, 0x34, 0x12, 0xEF, 0xCD, 0xAB, 0x89, // 10007: MOV RAX, 0x89ABCDEF12345678
    0x48, 0x05, 0xB9, 0x79, 0x37, 0x9E,                         // 10011: ADD RAX, -0x61C88647
    0x48, 0x35, 0x5A, 0x5A, 0x5A, 0x5A,                         // 10017: XOR RAX, 0x5A5A5A5A
    0x48, 0xC1, 0xC0, 0x07,                                     // 1001D: ROL RAX, 7
    0x48, 0x0F, 0xAF, 0xC0,                                     // 10021: IMUL RAX, RAX
    0x48, 0x29, 0xC8,                                           // 10025: SUB RAX, RCX
    0x48, 0xFF, 0xC9,                                           // 10028: DEC RCX
    0x75, 0xE4,                                                 // 1002B: JNZ 10011
]; // 1002D: end

#[rustfmt::skip]
const MEM64: &[u8] = &[
    0x48, 0xC7, 0xC1, 0x00, 0x09, 0x3D, 0x00, // 10000: MOV RCX, 4000000
    0x48, 0xC7, 0xC6, 0x00, 0x00, 0x10, 0x00, // 10007: MOV RSI, 0x100000 (src)
    0x48, 0xC7, 0xC7, 0x00, 0x10, 0x10, 0x00, // 1000E: MOV RDI, 0x101000 (dst)
    0x48, 0x89, 0xCA,                         // 10015: MOV RDX, RCX
    0x48, 0x81, 0xE2, 0xFF, 0x03, 0x00, 0x00, // 10018: AND RDX, 0x3FF
    0x48, 0x8B, 0x04, 0x96,                   // 1001F: MOV RAX, [RSI+RDX*4]
    0x48, 0x89, 0x04, 0x97,                   // 10023: MOV [RDI+RDX*4], RAX
    0x48, 0xFF, 0xC9,                         // 10027: DEC RCX
    0x75, 0xE9,                               // 1002A: JNZ 10015
]; // 1002C: end

#[rustfmt::skip]
const CALL64: &[u8] = &[
    0x48, 0xC7, 0xC4, 0x00, 0x00, 0x20, 0x00, // 10000: MOV RSP, 0x200000
    0x48, 0xC7, 0xC1, 0x00, 0x09, 0x3D, 0x00, // 10007: MOV RCX, 4000000
    0xE8, 0x07, 0x00, 0x00, 0x00,             // 1000E: CALL 1001A
    0x48, 0xFF, 0xC9,                         // 10013: DEC RCX
    0x75, 0xF6,                               // 10016: JNZ 1000E
    0xEB, 0x04,                               // 10018: JMP 1001E
    0x48, 0xFF, 0xC0,                         // 1001A: INC RAX
    0xC3,                                     // 1001D: RET
]; // 1001E: end

fn bench_x64(name: &str, program: &[u8], instructions: u64, check: fn(&x86_64::Cpu) -> u64) {
    let mut mem = x86_64::LinearMemory::new();
    mem.load(0x1_0000, program);
    let src: Vec<u8> = (0..0x1008).map(pat).collect();
    mem.load(0x10_0000, &src);

    let mut cpu = x86_64::Cpu::new();
    cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);

    let end = 0x1_0000 + program.len() as u64;
    let mut times = Vec::new();
    for _ in 0..RUNS {
        cpu.regs.rip = 0x1_0000;
        let t = Instant::now();
        let r = cpu.run(&mut mem, instructions);
        times.push(t.elapsed().as_secs_f64());
        assert_eq!(
            r.executed, instructions,
            "{name}: run() stopped early ({:?})",
            r.exit
        );
        assert_eq!(cpu.regs.rip, end, "{name}: did not land on the end address");
        assert!(!cpu.halted && !cpu.shutdown, "{name}: halted/shutdown");
    }
    report("64", name, instructions, &times, check(&cpu));
    #[cfg(feature = "perf-stats")]
    print!("{}", cpu.stats.report());
}

fn main() {
    println!("engine\tmode\tname\tinstructions\tbest_s\tmips\trun_s\tcheck");

    let per16 = |body: u64, outer: u64| outer * (1 + body * INNER16 + 2);
    bench_8086("tight_loop", TIGHT16, 1 + per16(2, TIGHT16_OUTER), |c| {
        c.regs.cx as u64
    });
    bench_8086("alu_mix", ALU16, 2 + per16(6, ALU16_OUTER), |c| {
        c.regs.ax as u64
    });
    bench_8086("mem_rw", MEM16, 3 + per16(6, MEM16_OUTER), |c| {
        c.regs.ax as u64
    });
    bench_8086("call_ret", CALL16, 2 + per16(5, CALL16_OUTER) + 1, |c| {
        c.regs.ax as u64
    });

    use x86_32::reg::{EAX, ECX};
    bench_386("tight_loop", TIGHT32, 1 + 2 * TIGHT_N, |c| {
        c.regs.gpr[ECX as usize] as u64
    });
    bench_386("alu_mix", ALU32, 2 + 7 * ALU_N, |c| {
        c.regs.gpr[EAX as usize] as u64
    });
    bench_386("mem_rw", MEM32, 3 + 6 * MEM_N, |c| {
        c.regs.gpr[EAX as usize] as u64
    });
    bench_386("call_ret", CALL32, 3 + 5 * CALL_N, |c| {
        c.regs.gpr[EAX as usize] as u64
    });
    bench_386_paged("tight_loop_pg", TIGHT32, 1 + 2 * TIGHT_N, |c| {
        c.regs.gpr[ECX as usize] as u64
    });
    bench_386_paged("mem_rw_pg", MEM32, 3 + 6 * MEM_N, |c| {
        c.regs.gpr[EAX as usize] as u64
    });
    bench_386_ring3("tight_loop_r3", TIGHT32, 1 + 2 * TIGHT_N, |c| {
        c.regs.gpr[ECX as usize] as u64
    });
    bench_386_ring3("mem_rw_r3", MEM32, 3 + 6 * MEM_N, |c| {
        c.regs.gpr[EAX as usize] as u64
    });
    bench_386("rep_movs", REP32, 2 + 6 * REP_OUTER, |c| {
        c.regs.gpr[EAX as usize] as u64
    });

    use x86_64::reg::{RAX, RCX};
    bench_x64("tight_loop", TIGHT64, 1 + 2 * TIGHT_N, |c| {
        c.regs.gpr[RCX as usize]
    });
    bench_x64("alu_mix", ALU64, 2 + 7 * ALU_N, |c| {
        c.regs.gpr[RAX as usize]
    });
    bench_x64("mem_rw", MEM64, 3 + 6 * MEM_N, |c| c.regs.gpr[RAX as usize]);
    bench_x64("call_ret", CALL64, 3 + 5 * CALL_N, |c| {
        c.regs.gpr[RAX as usize]
    });
}
