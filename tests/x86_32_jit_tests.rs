//! Differential tests for the 80386 template JIT (`--features jit`).
//!
//! The interpreter is the reference: a CPU driven through `run()` (which uses
//! the JIT) must land on byte-identical architectural state as a twin driven
//! through `step()` (always the interpreter). Chunked lockstep with prime
//! chunk sizes exercises budget cut-offs mid-block, and loop counts well past
//! the hot threshold guarantee the blocks are actually translated.
//!
//! All programs run in 32-bit flat protected mode (ring 0, paging off, CS.D =
//! 1) — the mode the JIT translates, matching `examples/bench_x86.rs`.

#![cfg(feature = "jit")]

use remu::x86_32::{Cpu, LinearMemory, RunExit, SegReg, cr0, reg};

const CODE: u32 = 0x1000;
const STACK: u32 = 0x20_0000;

/// A CPU in 32-bit flat protected mode with `program` loaded at [`CODE`].
fn flat32(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(CODE, program);
    let mut cpu = Cpu::new();
    cpu.regs.cr0 |= cr0::PE;
    cpu.regs.seg[reg::CS as usize] = SegReg {
        sel: 0x08,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0C9B, // present, code, exec/read, accessed; G + D (32-bit)
    };
    for s in [reg::SS, reg::DS, reg::ES, reg::FS, reg::GS] {
        cpu.regs.seg[s as usize] = SegReg {
            sel: 0x10,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: 0x0C93, // present, data, read/write, accessed; G + D
        };
    }
    cpu.regs.gpr[reg::ESP as usize] = STACK;
    cpu.regs.eip = CODE;
    (cpu, mem)
}

/// Run `program` for exactly `total` instructions two ways — the JIT via
/// `run(chunk)`, the interpreter via `chunk × step()` — asserting identical
/// registers, cycles and RAM after every chunk.
fn lockstep(program: &[u8], total: u64, chunk: u64) {
    let (mut ja, mut ma) = flat32(program);
    let (mut jb, mut mb) = flat32(program);
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
        assert_eq!(ja.regs, jb.regs, "registers diverged after {} insns", done + c);
        assert_eq!(ja.cycles, jb.cycles, "cycles diverged after {} insns", done + c);
        done += c;
    }
    assert_eq!(ma.ram, mb.ram, "guest RAM diverged");
}

/// `MOV ECX, n; loop { DEC ECX; JNZ }` — a self-looping block; total = 1 + 2n.
/// Uses the single-byte `DEC ECX` (0x49) form.
fn tight(n: u32) -> Vec<u8> {
    let mut p = vec![0xB9];
    p.extend_from_slice(&n.to_le_bytes()); // MOV ECX, n
    p.push(0x49); // DEC ECX
    p.extend_from_slice(&[0x75, 0xFD]); // JNZ -3 -> DEC ECX
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
fn ff_incdec_form_loop() {
    // The `FF /1` (IncDecRmW) DEC ECX form rather than the 0x49 short form, so
    // the group-decoded INC/DEC path is exercised too.
    #[rustfmt::skip]
    let mut p = vec![
        0xB9, 0x40, 0x00, 0x00, 0x00, // MOV ECX, 64
    ];
    p.extend_from_slice(&[0xFF, 0xC9]); // DEC ECX (FF /1, modrm C9)
    p.extend_from_slice(&[0x75, 0xFC]); // JNZ -4 -> DEC ECX
    let total = 1 + 2 * 64;
    for chunk in [1u64, 5, 13, 257] {
        lockstep(&p, total, chunk);
    }
}

#[test]
fn incs_and_dec_loop() {
    // Interleave single-byte INC on other registers with the DEC/JNZ loop so
    // several registers change and the loop still terminates on ECX. INC/DEC
    // preserve CF, so the block must leave the guest CF untouched.
    #[rustfmt::skip]
    let mut p = vec![
        0xB9, 0x40, 0x00, 0x00, 0x00, // MOV ECX, 64
    ];
    let body_start = p.len();
    p.push(0x40); // INC EAX
    p.push(0x42); // INC EDX
    p.push(0x49); // DEC ECX
    let back = -((p.len() + 2 - body_start) as i64) as i8;
    p.extend_from_slice(&[0x75, back as u8]); // JNZ body_start
    let total = 1 + 4 * 64;
    for chunk in [1u64, 5, 13, 257] {
        lockstep(&p, total, chunk);
    }
}

#[test]
fn alu_mix_loop() {
    // The bench's alu_mix body in 32-bit encodings: ADD/XOR imm, ROL imm,
    // IMUL, SUB reg, DEC/JNZ. Exercises the flag-exactness machine (XOR's
    // undefined AF, ROL's undefined OF and IMUL's undefined SF/ZF/PF are all
    // dead by the DEC at the exit).
    #[rustfmt::skip]
    let mut p = vec![
        0xB9, 0x00, 0x08, 0x00, 0x00,       // MOV ECX, 2048
        0xB8, 0x78, 0x56, 0x34, 0x12,       // MOV EAX, 0x12345678
    ];
    let loop_start = p.len();
    p.extend_from_slice(&[0x05, 0xB9, 0x79, 0x37, 0x9E]); // ADD EAX, 0x9E3779B9
    p.extend_from_slice(&[0x35, 0x5A, 0x5A, 0x5A, 0x5A]); // XOR EAX, 0x5A5A5A5A
    p.extend_from_slice(&[0xC1, 0xC0, 0x07]); // ROL EAX, 7
    p.extend_from_slice(&[0x0F, 0xAF, 0xC0]); // IMUL EAX, EAX
    p.extend_from_slice(&[0x29, 0xC8]); // SUB EAX, ECX
    p.push(0x49); // DEC ECX
    let back = -((p.len() + 2 - loop_start) as i64) as i8;
    p.extend_from_slice(&[0x75, back as u8]); // JNZ loop_start
    // 2 setup + 7 body instructions per iteration × 2048.
    let total = 2 + 7 * 2048;
    for chunk in [1u64, 3, 29, 337, 7919] {
        lockstep(&p, total, chunk);
    }
}

#[test]
fn alu_ops_and_shifts() {
    // A straight sweep of ALU/shift forms wrapped in a DEC/JNZ loop, so several
    // registers and every status flag change each iteration.
    #[rustfmt::skip]
    let mut p = vec![
        0xB9, 0x1E, 0x00, 0x00, 0x00, // MOV ECX, 30
        0xB8, 0xFF, 0x00, 0x00, 0x00, // MOV EAX, 255
        0xBB, 0x0F, 0x00, 0x00, 0x00, // MOV EBX, 15
    ];
    let loop_start = p.len();
    p.extend_from_slice(&[0x01, 0xD8]); // ADD EAX, EBX
    p.extend_from_slice(&[0x21, 0xD8]); // AND EAX, EBX
    p.extend_from_slice(&[0x09, 0xD8]); // OR EAX, EBX
    p.extend_from_slice(&[0xC1, 0xE0, 0x03]); // SHL EAX, 3
    p.extend_from_slice(&[0xC1, 0xE8, 0x02]); // SHR EAX, 2
    p.extend_from_slice(&[0x83, 0xF0, 0x11]); // XOR EAX, 0x11 (grp imm8)
    p.push(0x49); // DEC ECX
    let back = -((p.len() + 2 - loop_start) as i64) as i8;
    p.extend_from_slice(&[0x75, back as u8]); // JNZ
    let total = 3 + 8 * 30;
    for chunk in [1u64, 11, 121] {
        lockstep(&p, total, chunk);
    }
}

#[test]
fn shift_by_one_and_sar() {
    // Shift-by-1 forms (D1) exercise the OF-exact path; SAR exercises OF
    // cleared. Wrapped in a loop that reseeds EAX each iteration so the shift
    // operates on a known value.
    #[rustfmt::skip]
    let mut p = vec![
        0xB9, 0x20, 0x00, 0x00, 0x00, // MOV ECX, 32
    ];
    let loop_start = p.len();
    p.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x80]); // MOV EAX, 0x80000001
    p.extend_from_slice(&[0xD1, 0xE0]); // SHL EAX, 1  (D1 /4)
    p.extend_from_slice(&[0xD1, 0xF8]); // SAR EAX, 1  (D1 /7)
    p.extend_from_slice(&[0xC1, 0xC0, 0x04]); // ROL EAX, 4
    p.push(0x49); // DEC ECX
    let back = -((p.len() + 2 - loop_start) as i64) as i8;
    p.extend_from_slice(&[0x75, back as u8]); // JNZ
    let total = 1 + 5 * 32;
    for chunk in [1u64, 7, 41] {
        lockstep(&p, total, chunk);
    }
}

#[test]
fn test_and_cmp_no_writeback() {
    // TEST (AND without write-back) and CMP (SUB without write-back) must set
    // flags without touching the destination register.
    #[rustfmt::skip]
    let mut p = vec![
        0xB9, 0x28, 0x00, 0x00, 0x00, // MOV ECX, 40
        0xB8, 0xF0, 0x00, 0x00, 0x00, // MOV EAX, 0xF0
        0xBB, 0x0F, 0x00, 0x00, 0x00, // MOV EBX, 0x0F
    ];
    let loop_start = p.len();
    p.extend_from_slice(&[0x85, 0xD8]); // TEST EAX, EBX  (85 /r)
    p.extend_from_slice(&[0x39, 0xC8]); // CMP EAX, ECX   (39 /r)
    p.extend_from_slice(&[0xA9, 0x0F, 0x00, 0x00, 0x00]); // TEST EAX, 0x0F (imm)
    p.push(0x40); // INC EAX (so EAX changes and the loop is not trivial)
    p.push(0x49); // DEC ECX
    let back = -((p.len() + 2 - loop_start) as i64) as i8;
    p.extend_from_slice(&[0x75, back as u8]); // JNZ
    let total = 3 + 5 * 40;
    for chunk in [1u64, 3, 17, 211] {
        lockstep(&p, total, chunk);
    }
}

#[test]
fn large_self_loop_backedge() {
    // A self-loop whose body is far larger than a `loop` rel8 back-edge can
    // reach (24 ALU ops before DEC/JNZ). Exercises the bounce trampoline.
    #[rustfmt::skip]
    let mut p = vec![
        0xB9, 0x40, 0x00, 0x00, 0x00, // MOV ECX, 64
        0xB8, 0x01, 0x00, 0x00, 0x00, // MOV EAX, 1
        0xBB, 0x03, 0x00, 0x00, 0x00, // MOV EBX, 3
    ];
    let loop_start = p.len();
    for _ in 0..24 {
        p.extend_from_slice(&[0x01, 0xD8]); // ADD EAX, EBX
    }
    p.push(0x49); // DEC ECX
    let back = -((p.len() + 2 - loop_start) as i64) as i8;
    p.extend_from_slice(&[0x75, back as u8]); // JNZ loop_start
    let total = 3 + (24 + 1) * 64;
    for chunk in [1u64, 41, 733] {
        lockstep(&p, total, chunk);
    }
}

#[test]
fn smc_invalidates_jit_block() {
    // Warm + translate a tight loop, then overwrite the loop body with a HLT
    // via a host-side store + invalidation (mirroring an os-layer trap) and
    // confirm the CPU sees the new code.
    let n = 200u32;
    let prog = tight(n);
    let (mut cpu, mut mem) = flat32(&prog);
    let r = cpu.run(&mut mem, 50);
    assert_eq!(r.exit, RunExit::Completed);
    // Patch the DEC ECX (at CODE+5) to a HLT (0xF4).
    mem.load(CODE + 5, &[0xF4]);
    cpu.invalidate_icache();
    cpu.invalidate_jit();
    cpu.regs.eip = CODE + 5;
    cpu.halted = false;
    let r = cpu.run(&mut mem, 100);
    assert_eq!(r.exit, RunExit::Halted, "patched HLT must take effect");
}
