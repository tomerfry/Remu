//! Concolic-overlay tests for the ARM32 core (feature `symbolic`). Validates
//! the shared engine on a genuinely different architecture: NZCV flags, the
//! (result, carry, overflow) ALU, and per-instruction predication as the fork.
#![cfg(feature = "symbolic")]

use remu::arm32::{Cpu, LinearMemory};

const ORG: u32 = 0x100;

/// A CPU + flat memory with ARM `program` at ORG, PC there, a sane stack.
fn arm(program: &[u32]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    for (i, w) in program.iter().enumerate() {
        mem.load(ORG + i as u32 * 4, &w.to_le_bytes());
    }
    let mut cpu = Cpu::new();
    cpu.regs.gpr[15] = ORG;
    cpu.regs.gpr[13] = 0x8000;
    (cpu, mem)
}

/// Symbolic data-processing (ADDS/SUBS immediate) keeps the golden invariant —
/// including the NZCV flags.
#[test]
fn arm_golden_invariant() {
    // ADDS r0, r0, #5 ; SUBS r0, r0, #2
    let (mut cpu, mut mem) = arm(&[0xE290_0005, 0xE250_0002]);
    cpu.regs.gpr[0] = 0x1234;
    cpu.sym_init();
    cpu.sym_symbolize_reg(0, "r0");

    for _ in 0..2 {
        cpu.step(&mut mem);
        assert!(cpu.sym_check_invariant(), "invariant at pc {:#x}", cpu.regs.gpr[15]);
    }
    assert_eq!(cpu.regs.gpr[0], 0x1234 + 5 - 2);
}

/// Overlay inert until enabled.
#[test]
fn arm_overlay_inert() {
    let (mut cpu, mut mem) = arm(&[0xE290_0005]); // ADDS r0, r0, #5
    cpu.regs.gpr[0] = 1;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 6);
    assert!(cpu.sym_constraints().is_empty());
    assert!(cpu.sym_check_invariant());
}

/// End-to-end on ARM: `CMP r0, #0x2A ; BNE` on a symbolic r0 records the branch
/// condition; solving the flipped branch recovers the magic value. Skips
/// without a solver.
#[cfg(feature = "symbolic-solver")]
#[test]
fn arm_solve_for_input() {
    if !Cpu::sym_solver_available() {
        eprintln!("no SMT solver on PATH (set REMU_SMT_SOLVER) — skipping");
        return;
    }
    let (mut cpu, mut mem) = arm(&[
        0xE350_002A, // CMP r0, #0x2A
        0x1A00_0000, // BNE +0 (target = ORG + 0xC)
        0xE1A0_0000, // NOP (MOV r0, r0)
        0xE1A0_0000, // NOP
    ]);
    cpu.regs.gpr[0] = 7; // seed ≠ 0x2A
    cpu.sym_init();
    cpu.sym_symbolize_reg(0, "r0");

    cpu.step(&mut mem); // CMP r0, #0x2A  (sets symbolic NZCV)
    cpu.step(&mut mem); // BNE  (NE taken; records the constraint)

    assert!(cpu.sym_check_invariant());
    assert_eq!(cpu.sym_constraints().len(), 1);
    let model = cpu.sym_solve_flip(0).expect("constraints should be satisfiable");
    assert_eq!(model.get("r0"), Some(&0x2A), "solver recovered the magic input");
}
