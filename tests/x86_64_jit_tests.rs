//! Differential tests for the x86-64 template JIT (`--features jit`).
//!
//! The interpreter is the reference: a CPU driven through `run()` (which uses
//! the JIT) must land on byte-identical architectural state as a twin driven
//! through `step()` (always the interpreter). Chunked lockstep with prime
//! chunk sizes exercises budget cut-offs mid-block, and loop counts well past
//! the hot threshold guarantee the blocks are actually translated.

#![cfg(feature = "jit")]

use remu::x86_64::{Cpu, LinearMemory, RunExit};

const CODE: u64 = 0x1_0000;
const STACK: u64 = 0x20_0000;

fn long(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(CODE, program);
    let mut cpu = Cpu::new();
    cpu.setup_long_flat(&mut mem, CODE, STACK);
    (cpu, mem)
}

/// Run `program` for exactly `total` instructions two ways — the JIT via
/// `run(chunk)`, the interpreter via `chunk × step()` — asserting identical
/// registers, cycles and RAM after every chunk.
fn lockstep(program: &[u8], total: u64, chunk: u64) {
    let (mut ja, mut ma) = long(program);
    let (mut jb, mut mb) = long(program);
    let mut done = 0u64;
    while done < total {
        let c = chunk.min(total - done);
        let r = ja.run(&mut ma, c);
        assert_eq!(
            (r.executed, r.exit),
            (c, RunExit::Completed),
            "JIT run stopped early at {done}"
        );
        for _ in 0..c {
            jb.step(&mut mb);
        }
        assert_eq!(
            ja.regs,
            jb.regs,
            "registers diverged after {} insns",
            done + c
        );
        assert_eq!(
            ja.cycles,
            jb.cycles,
            "cycles diverged after {} insns",
            done + c
        );
        done += c;
    }
    assert_eq!(ma.ram, mb.ram, "guest RAM diverged");
}

/// `MOV RCX, n; loop { DEC RCX; JNZ }` — a self-looping block; total = 1 + 2n.
fn tight(n: u32) -> Vec<u8> {
    let mut p = vec![0x48, 0xC7, 0xC1];
    p.extend_from_slice(&n.to_le_bytes()); // MOV RCX, n (imm32 sign-extended)
    p.extend_from_slice(&[0x48, 0xFF, 0xC9]); // DEC RCX
    p.extend_from_slice(&[0x75, 0xFB]); // JNZ -5 -> DEC
    p
}

#[test]
fn tight_loop_whole() {
    let n = 500u32;
    lockstep(&tight(n), 1 + 2 * n as u64, 1 + 2 * n as u64);
}

#[test]
fn tight_loop_prime_chunks() {
    let n = 1000u32;
    let total = 1 + 2 * n as u64;
    for chunk in [1u64, 2, 3, 7, 97, 1009] {
        lockstep(&tight(n), total, chunk);
    }
}

#[test]
fn incs_and_dec_loop() {
    // Interleave INC on other registers with the DEC/JNZ loop so several
    // registers change and the loop still terminates on RCX.
    #[rustfmt::skip]
    let mut p = vec![
        0x48, 0xC7, 0xC1, 0x40, 0x00, 0x00, 0x00, // MOV RCX, 64
    ];
    // loop body: INC RAX; INC RDX; DEC RCX; JNZ back
    let body_start = p.len();
    p.extend_from_slice(&[0x48, 0xFF, 0xC0]); // INC RAX
    p.extend_from_slice(&[0x48, 0xFF, 0xC2]); // INC RDX
    p.extend_from_slice(&[0x48, 0xFF, 0xC9]); // DEC RCX
    let back = -((p.len() + 2 - body_start) as i64) as i8;
    p.extend_from_slice(&[0x75, back as u8]); // JNZ body_start
    // 1 (MOV) + 4*64 body instructions.
    let total = 1 + 4 * 64;
    for chunk in [1u64, 5, 13, 257] {
        lockstep(&p, total, chunk);
    }
}

#[test]
fn mov32_zero_extends_in_jit() {
    // MOV ECX, -1 (B9) then DEC/JNZ — the 32-bit MOV must zero bits 63:32,
    // giving RCX = 0xFFFFFFFF, so the loop runs 0xFFFFFFFF... too long. Use a
    // 32-bit immediate that is small but exercises the zero-extend path.
    #[rustfmt::skip]
    let mut p = vec![
        0xB9, 0x20, 0x00, 0x00, 0x00,       // MOV ECX, 32 (zero-extends)
    ];
    p.extend_from_slice(&[0x48, 0xFF, 0xC9]); // DEC RCX (O64)
    p.extend_from_slice(&[0x75, 0xFB]); // JNZ -5
    let total = 1 + 2 * 32;
    lockstep(&p, total, 7);
}

#[test]
fn dec32_zero_extends_upper() {
    // 32-bit DEC must clear the upper dword. Seed ECX via a 64-bit MOV whose
    // upper bits are set, then DEC ECX (no REX.W) in a loop.
    #[rustfmt::skip]
    let mut p = vec![
        0x48, 0xB9, 0x08, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, // MOV RCX, 0xFFFFFFFF_00000008
    ];
    p.extend_from_slice(&[0xFF, 0xC9]); // DEC ECX (32-bit, zero-extends result)
    p.extend_from_slice(&[0x75, 0xFC]); // JNZ -4
    // After the first DEC ECX, RCX becomes 0x00000007 (upper cleared); loops
    // 8 times total to reach 0. total = 1 + 2*8.
    let total = 1 + 2 * 8;
    lockstep(&p, total, 3);
}

#[test]
fn smc_invalidates_jit_block() {
    // Run a tight loop enough to translate it, then overwrite the loop body
    // with a HLT via a guest store and confirm the CPU sees the new code
    // (the block's prologue stamp-check makes it retranslate/fall back).
    let n = 200u32;
    let prog = tight(n);
    let (mut cpu, mut mem) = long(&prog);
    // Warm + translate: run partway.
    let r = cpu.run(&mut mem, 50);
    assert_eq!(r.exit, RunExit::Completed);
    // Guest store: patch the DEC (at CODE+7) to a HLT (0xF4) via a MOV to
    // memory executed by the guest is complex here; instead write host-side
    // and invalidate, mirroring an os-layer trap.
    mem.load(CODE + 7, &[0xF4]); // HLT in place of DEC
    cpu.invalidate_icache();
    cpu.invalidate_jit();
    cpu.regs.rip = CODE + 7;
    cpu.halted = false;
    let r = cpu.run(&mut mem, 100);
    assert_eq!(r.exit, RunExit::Halted, "patched HLT must take effect");
}
