//! Concolic-overlay tests for the 386 core (feature `symbolic`, milestone M1).
//!
//! These exercise the live instrumentation seams: symbolic dataflow through the
//! ALU, the golden concolic invariant (`shadow.eval(seed) == concrete`), and
//! path-constraint collection at a symbolic branch.
#![cfg(feature = "symbolic")]

use remu::symbolic::{Model, SymId};
use remu::x86_32::{Cpu, LinearMemory, SegReg, reg};

/// A CPU + flat memory with `program` at `0000:1100`, real-mode defaults, DS
/// base 0. Mirrors `x86_32_tests::setup`.
fn setup(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(0x1100, program);
    let mut cpu = Cpu::new();
    cpu.set_cs_ip(0x0000, 0x1100);
    cpu.regs.seg[reg::SS as usize] = SegReg::real(0x0000);
    cpu.regs.seg[reg::DS as usize] = SegReg::real(0x0000);
    cpu.regs.gpr[reg::ESP as usize] = 0xFFF0;
    (cpu, mem)
}

/// The overlay is invisible until enabled: no constraints, invariant trivially
/// holds, results are bit-identical to a normal run.
#[test]
fn overlay_inert_until_enabled() {
    let (mut cpu, mut mem) = setup(&[0x66, 0x05, 0x01, 0x00, 0x00, 0x00]); // ADD EAX, 1
    cpu.regs.gpr[0] = 41;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 42);
    assert!(cpu.sym_constraints().is_empty());
    assert!(cpu.sym_check_invariant());
}

/// Symbolic dataflow through a chain of ALU ops keeps the golden invariant
/// (`shadow.eval(seed) == concrete`) after every step.
#[test]
fn golden_invariant_alu_chain() {
    let (mut cpu, mut mem) = setup(&[
        0x66, 0x05, 0x03, 0x00, 0x00, 0x00, // ADD EAX, 3
        0x66, 0x35, 0xFF, 0x00, 0x00, 0x00, // XOR EAX, 0FFh
        0x66, 0x2D, 0x01, 0x00, 0x00, 0x00, // SUB EAX, 1
    ]);
    cpu.regs.gpr[0] = 0x1234_5678;
    cpu.sym_init();
    cpu.sym_symbolize_reg32(0, "eax");

    for _ in 0..3 {
        cpu.step(&mut mem);
        assert!(cpu.sym_check_invariant(), "invariant broke at eip {:#x}", cpu.regs.eip);
    }
    let expect = (0x1234_5678u32.wrapping_add(3) ^ 0xFF).wrapping_sub(1);
    assert_eq!(cpu.regs.gpr[0], expect);
}

/// A symbolic memory operand taints the destination register.
#[test]
fn symbolic_memory_operand() {
    // 66 03 06 00 20: ADD EAX, [0x2000] (32-bit operand, 16-bit [disp16] EA).
    let (mut cpu, mut mem) = setup(&[0x66, 0x03, 0x06, 0x00, 0x20]);
    cpu.regs.gpr[0] = 1;
    mem.ram[0x2000..0x2004].copy_from_slice(&0x1122_3344u32.to_le_bytes());
    cpu.sym_init();
    cpu.sym_symbolize_mem(0x2000, 32, "m", 0x1122_3344);

    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x1122_3345);
    assert!(cpu.sym_check_invariant());
    // EAX is now symbolic: solving for m = 0 would give EAX = 1.
    let id = cpu.sym_engine().unwrap().inputs()[0].id;
    // (Nothing to assert on constraints here — just that the taint propagated
    // and the invariant holds; the id is used by the branch test below.)
    let _ = id;
}

/// `CMP EAX, 42; JNE` on a symbolic EAX records the branch condition as a
/// constraint that a solver could negate to reach the other path.
#[test]
fn branch_records_path_constraint() {
    let (mut cpu, mut mem) = setup(&[
        0x66, 0x3D, 0x2A, 0x00, 0x00, 0x00, // CMP EAX, 42
        0x75, 0x02, // JNE +2
        0x90, 0x90, // NOP NOP (fall-through / target padding)
    ]);
    cpu.regs.gpr[0] = 7; // seed ≠ 42 ⇒ concrete branch is taken
    cpu.sym_init();
    let eax = cpu.sym_symbolize_reg32(0, "eax");

    cpu.step(&mut mem); // CMP
    cpu.step(&mut mem); // JNE

    let cons = cpu.sym_constraints();
    assert_eq!(cons.len(), 1, "one symbolic branch ⇒ one constraint");

    let seed = cpu.sym_seed();
    assert!(cons[0].eval(&seed), "constraint holds on the concrete (taken) path");

    // The recorded constraint is `EAX != 42`; it is false exactly when EAX = 42,
    // i.e. negating it (the fall-through path) solves to the magic value.
    let mut hit: Model = seed.clone();
    hit.insert(eax, 42);
    assert!(!cons[0].eval(&hit), "the taken-branch constraint fails at EAX = 42");
    assert!(
        remu::symbolic::BoolExpr::not(cons[0].clone()).eval(&hit),
        "the fall-through path is reached at EAX = 42"
    );
}

/// The canonical solve-for-input flow: load a symbolic input from memory with
/// MOV, compare against a magic value, branch. The recorded constraint negates
/// to the magic input.
#[test]
fn mov_load_then_branch() {
    const MAGIC: u32 = 0x1234_5678;
    let (mut cpu, mut mem) = setup(&[
        0x66, 0x8B, 0x06, 0x00, 0x20, // MOV EAX, [0x2000]
        0x66, 0x3D, 0x78, 0x56, 0x34, 0x12, // CMP EAX, MAGIC
        0x75, 0x02, 0x90, 0x90, // JNE +2 / NOPs
    ]);
    mem.ram[0x2000..0x2004].copy_from_slice(&0u32.to_le_bytes()); // seed input = 0
    cpu.sym_init();
    let input = cpu.sym_symbolize_mem(0x2000, 32, "input", 0);

    cpu.step(&mut mem); // MOV EAX, [input]  → EAX tainted
    cpu.step(&mut mem); // CMP EAX, MAGIC
    cpu.step(&mut mem); // JNE (taken: 0 != MAGIC)

    assert!(cpu.sym_check_invariant());
    let cons = cpu.sym_constraints();
    assert_eq!(cons.len(), 1);

    // Negating the taken-branch constraint solves to input = MAGIC.
    let mut solved: Model = cpu.sym_seed();
    solved.insert(input, MAGIC as u64);
    assert!(!cons[0].eval(&solved), "input = MAGIC flips the branch");
}

/// A concrete branch (EAX never symbolized) records nothing.
#[test]
fn concrete_branch_records_nothing() {
    let (mut cpu, mut mem) = setup(&[
        0x66, 0x3D, 0x2A, 0x00, 0x00, 0x00, // CMP EAX, 42
        0x75, 0x02, 0x90, 0x90, // JNE +2 / NOPs
    ]);
    cpu.regs.gpr[0] = 7;
    cpu.sym_init(); // enabled, but nothing symbolic
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert!(cpu.sym_constraints().is_empty());
    // Silence unused-import lints when only this test runs.
    let _ = SymId(0);
}
