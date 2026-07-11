//! CPU core throughput benchmarks.
//!
//! Each workload is a small hand-assembled 6502 program that loops forever;
//! we measure a fixed number of `step()` calls, so Criterion's `Melem/s`
//! throughput reads directly as *millions of emulated instructions per second*.
//!
//! Run with `cargo bench`. Criterion stores the previous run as a baseline and
//! reports the delta, so run once before and once after a change to compare.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use remu::{Cpu, memory::FlatMemory};
use std::hint::black_box;

/// Instructions executed per benchmark iteration.
const STEPS: u64 = 10_000;

/// Benchmark `STEPS` instructions of `program`, loaded and entered at `$0600`.
fn bench_program(c: &mut Criterion, name: &str, program: &[u8], setup: fn(&mut FlatMemory)) {
    let mut mem = FlatMemory::new();
    mem.load(0x0600, program);
    mem.set_reset_vector(0x0600);
    setup(&mut mem);

    let mut cpu = Cpu::new();
    cpu.reset(&mut mem);

    let mut group = c.benchmark_group("cpu");
    group.throughput(Throughput::Elements(STEPS));
    group.bench_function(name, |b| {
        b.iter(|| {
            for _ in 0..STEPS {
                cpu.step(&mut mem);
            }
            black_box(cpu.cycles)
        })
    });
    group.finish();

    assert!(!cpu.halted, "benchmark program {name} hit a KIL opcode");
}

/// Branch-heavy counting loop: DEX/BNE with a JMP restart.
fn tight_loop(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0xA2, 0xFF,             // 0600: LDX #$FF
        0xCA,                   // 0602: DEX
        0xD0, 0xFD,             // 0603: BNE $0602
        0x4C, 0x00, 0x06,       // 0605: JMP $0600
    ];
    bench_program(c, "tight_loop", &program, |_| {});
}

/// Page copy via absolute-indexed loads/stores: LDA abs,Y / STA abs,Y / INY.
fn memcpy(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0xA0, 0x00,             // 0600: LDY #$00
        0xB9, 0x00, 0x07,       // 0602: LDA $0700,Y
        0x99, 0x00, 0x08,       // 0605: STA $0800,Y
        0xC8,                   // 0608: INY
        0xD0, 0xF7,             // 0609: BNE $0602
        0x4C, 0x00, 0x06,       // 060B: JMP $0600
    ];
    bench_program(c, "memcpy", &program, |mem| {
        let src: Vec<u8> = (0..=255).collect();
        mem.load(0x0700, &src);
    });
}

/// Arithmetic/flag mix on the zero page: ADC, SBC, CMP, EOR.
fn arith(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0x18,                   // 0600: CLC
        0xA5, 0x10,             // 0601: LDA $10
        0x69, 0x07,             // 0603: ADC #$07
        0x85, 0x10,             // 0605: STA $10
        0xA5, 0x11,             // 0607: LDA $11
        0xE9, 0x03,             // 0609: SBC #$03
        0x85, 0x11,             // 060B: STA $11
        0x45, 0x10,             // 060D: EOR $10
        0xC5, 0x11,             // 060F: CMP $11
        0x4C, 0x00, 0x06,       // 0611: JMP $0600
    ];
    bench_program(c, "arith", &program, |mem| {
        mem.load(0x0010, &[0x39, 0xA4]);
    });
}

/// Subroutine call/return: JSR/RTS stack traffic.
fn jsr_rts(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0x20, 0x06, 0x06,       // 0600: JSR $0606
        0x4C, 0x00, 0x06,       // 0603: JMP $0600
        0xE8,                   // 0606: INX
        0x60,                   // 0607: RTS
    ];
    bench_program(c, "jsr_rts", &program, |_| {});
}

// --- 80386 core -------------------------------------------------------------

/// Benchmark `STEPS` instructions of a 386 real-mode `program` at `0000:1000`.
fn bench_386(c: &mut Criterion, name: &str, program: &[u8]) {
    use remu::x86_32;

    let mut mem = x86_32::LinearMemory::new();
    mem.load(0x1000, program);

    let mut cpu = x86_32::Cpu::new();
    cpu.set_cs_ip(0x0000, 0x1000);
    cpu.regs.seg[x86_32::reg::SS as usize] = x86_32::SegReg::real(0x9000);
    cpu.regs.gpr[x86_32::reg::ESP as usize] = 0xFF00;

    let mut group = c.benchmark_group("cpu386");
    group.throughput(Throughput::Elements(STEPS));
    group.bench_function(name, |b| {
        b.iter(|| {
            for _ in 0..STEPS {
                cpu.step(&mut mem);
            }
            black_box(cpu.cycles)
        })
    });
    group.finish();

    assert!(!cpu.halted, "benchmark program {name} halted");
}

/// 32-bit counting loop: DEC ECX / JNZ with a JMP restart.
fn tight_loop_386(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0x66, 0xB9, 0xFF, 0x00, 0x00, 0x00, // 1000: MOV ECX, 0xFF
        0x66, 0x49,                         // 1006: DEC ECX
        0x75, 0xFC,                         // 1008: JNZ 1006
        0xEB, 0xF4,                         // 100A: JMP 1000
    ];
    bench_386(c, "tight_loop", &program);
}

/// 32-bit ALU mix on registers and memory.
fn arith_386(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0x66, 0xB8, 0x78, 0x56, 0x34, 0x12, // 1000: MOV EAX, 12345678h
        0x66, 0x05, 0x01, 0x00, 0x00, 0x00, // 1006: ADD EAX, 1
        0x66, 0x31, 0x06, 0x00, 0x20,       // 100C: XOR [2000h], EAX
        0x66, 0xC1, 0xC0, 0x07,             // 1011: ROL EAX, 7
        0x66, 0x0F, 0xAF, 0xC0,             // 1015: IMUL EAX, EAX
        0xEB, 0xE5,                         // 1019: JMP 1000
    ];
    bench_386(c, "arith", &program);
}

/// CALL/RET stack traffic.
fn call_ret_386(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0xE8, 0x03, 0x00,                   // 1000: CALL 1006
        0xEB, 0xFB,                         // 1003: JMP 1000
        0x90,                               // 1005: NOP (padding)
        0x40,                               // 1006: INC AX
        0xC3,                               // 1007: RET
    ];
    bench_386(c, "call_ret", &program);
}

// --- x86-64 core ------------------------------------------------------------

/// Benchmark `STEPS` instructions of a 64-bit `program` at `0x1_0000`, in
/// long mode with flat identity paging (`setup_long_flat`).
fn bench_x64(c: &mut Criterion, name: &str, program: &[u8]) {
    use remu::x86_64;

    let mut mem = x86_64::LinearMemory::new();
    mem.load(0x1_0000, program);

    let mut cpu = x86_64::Cpu::new();
    cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);

    let mut group = c.benchmark_group("cpux64");
    group.throughput(Throughput::Elements(STEPS));
    group.bench_function(name, |b| {
        b.iter(|| {
            for _ in 0..STEPS {
                cpu.step(&mut mem);
            }
            black_box(cpu.cycles)
        })
    });
    group.finish();

    assert!(!cpu.halted, "benchmark program {name} halted");
}

/// 64-bit counting loop: DEC RCX / JNZ with a JMP restart.
fn tight_loop_x64(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0x48, 0xC7, 0xC1, 0xFF, 0x00, 0x00, 0x00,       // 10000: MOV RCX, 0xFF
        0x48, 0xFF, 0xC9,                               // 10007: DEC RCX
        0x75, 0xFB,                                     // 1000A: JNZ 10007
        0xEB, 0xF2,                                     // 1000C: JMP 10000
    ];
    bench_x64(c, "tight_loop", &program);
}

/// 64-bit ALU mix on registers and memory.
fn arith_x64(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0x48, 0xC7, 0xC0, 0x78, 0x56, 0x34, 0x12,       // 10000: MOV RAX, 12345678h
        0x48, 0x05, 0x01, 0x00, 0x00, 0x00,             // 10007: ADD RAX, 1
        0x48, 0x31, 0x04, 0x25, 0x00, 0x00, 0x02, 0x00, // 1000D: XOR [20000h], RAX
        0x48, 0xC1, 0xC0, 0x07,                         // 10015: ROL RAX, 7
        0x48, 0x0F, 0xAF, 0xC0,                         // 10019: IMUL RAX, RAX
        0xEB, 0xE1,                                     // 1001D: JMP 10000
    ];
    bench_x64(c, "arith", &program);
}

/// CALL/RET stack traffic.
fn call_ret_x64(c: &mut Criterion) {
    #[rustfmt::skip]
    let program = [
        0xE8, 0x03, 0x00, 0x00, 0x00,                   // 10000: CALL 10008
        0xEB, 0xF9,                                     // 10005: JMP 10000
        0x90,                                           // 10007: NOP (padding)
        0x48, 0xFF, 0xC0,                               // 10008: INC RAX
        0xC3,                                           // 1000B: RET
    ];
    bench_x64(c, "call_ret", &program);
}

criterion_group!(benches, tight_loop, memcpy, arith, jsr_rts);
criterion_group!(benches386, tight_loop_386, arith_386, call_ret_386);
criterion_group!(benchesx64, tight_loop_x64, arith_x64, call_ret_x64);
criterion_main!(benches, benches386, benchesx64);
