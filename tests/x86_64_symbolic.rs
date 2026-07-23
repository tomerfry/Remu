//! Concolic-overlay tests for the x86-64 core (feature `symbolic`). Exercises
//! the shared engine on a second core: 64-bit symbolic dataflow in long mode,
//! MOV propagation, and end-to-end solve-for-input.
#![cfg(feature = "symbolic")]

use remu::x86_64::{Cpu, LinearMemory};

const CODE: u64 = 0x10_0000;
const STACK: u64 = 0x20_0000;

/// A CPU in 64-bit long mode (flat ring-0 + identity paging), `program` at CODE.
fn long(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(CODE, program);
    let mut cpu = Cpu::new();
    cpu.setup_long_flat(&mut mem, CODE, STACK);
    (cpu, mem)
}

/// 64-bit symbolic arithmetic (REX.W) keeps the golden invariant, including
/// values above 32 bits (where the width-safe carry logic matters).
#[test]
fn x64_golden_invariant() {
    let (mut cpu, mut mem) = long(&[
        0x48, 0x05, 0x05, 0x00, 0x00, 0x00, // ADD RAX, 5
        0x48, 0x2D, 0x02, 0x00, 0x00, 0x00, // SUB RAX, 2
        0x48, 0x35, 0xFF, 0x00, 0x00, 0x00, // XOR RAX, 0xFF
    ]);
    cpu.regs.gpr[0] = 0x1_0000_0000; // > 32 bits
    cpu.sym_init();
    cpu.sym_symbolize_reg64(0, "rax");

    for _ in 0..3 {
        cpu.step(&mut mem);
        assert!(cpu.sym_check_invariant(), "invariant at rip {:#x}", cpu.regs.rip);
    }
    let expect = ((0x1_0000_0000u64 + 5) - 2) ^ 0xFF;
    assert_eq!(cpu.regs.gpr[0], expect);
}

/// `MOV RBX, RAX` propagates taint to RBX; a later compare branches symbolically.
#[test]
fn x64_mov_propagates() {
    let (mut cpu, mut mem) = long(&[
        0x48, 0x89, 0xC3, // MOV RBX, RAX
        0x48, 0x81, 0xFB, 0x78, 0x56, 0x34, 0x12, // CMP RBX, 0x12345678
        0x75, 0x02, 0x90, 0x90, // JNE +2 / NOPs
    ]);
    cpu.regs.gpr[0] = 1;
    cpu.sym_init();
    cpu.sym_symbolize_reg64(0, "rax");

    cpu.step(&mut mem); // MOV RBX, RAX
    cpu.step(&mut mem); // CMP RBX, MAGIC
    cpu.step(&mut mem); // JNE
    assert!(cpu.sym_check_invariant());
    assert_eq!(cpu.sym_constraints().len(), 1);
}

/// End-to-end on the x86-64 core: solve for the RAX value that flips the branch.
#[cfg(feature = "symbolic-solver")]
#[test]
fn x64_solve_for_input() {
    if !Cpu::sym_solver_available() {
        eprintln!("no SMT solver on PATH (set REMU_SMT_SOLVER) — skipping");
        return;
    }
    const MAGIC: u64 = 0x1234_5678;
    let (mut cpu, mut mem) = long(&[
        0x48, 0x3D, 0x78, 0x56, 0x34, 0x12, // CMP RAX, 0x12345678
        0x75, 0x02, 0x90, 0x90, // JNE +2 / NOPs
    ]);
    cpu.regs.gpr[0] = 7; // seed ≠ MAGIC
    cpu.sym_init();
    cpu.sym_symbolize_reg64(0, "rax");

    cpu.step(&mut mem); // CMP RAX, MAGIC
    cpu.step(&mut mem); // JNE (taken)

    let model = cpu.sym_solve_flip(0).expect("constraints should be satisfiable");
    assert_eq!(model.get("rax"), Some(&MAGIC), "solver recovered the magic input");
}
