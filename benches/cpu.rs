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

criterion_group!(benches, tight_loop, memcpy, arith, jsr_rts);
criterion_main!(benches);
