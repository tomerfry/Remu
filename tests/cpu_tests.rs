//! Focused unit tests for CPU behavior and the nastier 6502 quirks. These give
//! readable failures before reaching for the exhaustive Tom Harte suite.

use remu::bus::Bus;
use remu::memory::FlatMemory;
use remu::{Cpu, Status};

/// Build a CPU + flat memory with `program` at `$0600` and the reset vector set,
/// then reset the CPU so `PC == $0600`.
fn setup(program: &[u8]) -> (Cpu, FlatMemory) {
    let mut mem = FlatMemory::new();
    mem.load(0x0600, program);
    mem.set_reset_vector(0x0600);
    let mut cpu = Cpu::new();
    cpu.reset(&mut mem);
    (cpu, mem)
}

#[test]
fn reset_loads_pc_from_vector() {
    let mut mem = FlatMemory::new();
    mem.set_reset_vector(0x1234);
    let mut cpu = Cpu::new();
    cpu.reset(&mut mem);
    assert_eq!(cpu.regs.pc, 0x1234);
    assert_eq!(cpu.regs.sp, 0xFD);
    assert!(cpu.regs.p.contains(Status::I));
}

#[test]
fn lda_immediate_sets_zn() {
    let (mut cpu, mut mem) = setup(&[0xA9, 0x00]); // LDA #$00
    let cycles = cpu.step(&mut mem);
    assert_eq!(cpu.regs.a, 0x00);
    assert!(cpu.regs.p.contains(Status::Z));
    assert!(!cpu.regs.p.contains(Status::N));
    assert_eq!(cycles, 2);

    let (mut cpu, mut mem) = setup(&[0xA9, 0x80]); // LDA #$80
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.a, 0x80);
    assert!(cpu.regs.p.contains(Status::N));
    assert!(!cpu.regs.p.contains(Status::Z));
}

#[test]
fn lda_absolute_x_page_cross_costs_extra_cycle() {
    // LDA $12FF,X with X=1 -> reads $1300, crossing a page (+1 cycle).
    let mut mem = FlatMemory::new();
    mem.load(0x0600, &[0xBD, 0xFF, 0x12]);
    mem.ram[0x1300] = 0x77;
    mem.set_reset_vector(0x0600);
    let mut cpu = Cpu::new();
    cpu.reset(&mut mem);
    cpu.regs.x = 0x01;

    let cycles = cpu.step(&mut mem);
    assert_eq!(cpu.regs.a, 0x77);
    assert_eq!(cycles, 5); // base 4 + page cross
}

#[test]
fn sta_zero_page_writes_memory() {
    let (mut cpu, mut mem) = setup(&[0x85, 0x10]); // STA $10
    cpu.regs.a = 0xAB;
    cpu.step(&mut mem);
    assert_eq!(mem.read(0x10), 0xAB);
}

#[test]
fn adc_overflow_and_carry() {
    // 0x50 + 0x50 = 0xA0: signed overflow, negative, no carry.
    let (mut cpu, mut mem) = setup(&[0x69, 0x50]); // ADC #$50
    cpu.regs.a = 0x50;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.a, 0xA0);
    assert!(cpu.regs.p.contains(Status::V));
    assert!(cpu.regs.p.contains(Status::N));
    assert!(!cpu.regs.p.contains(Status::C));
}

#[test]
fn adc_decimal_mode() {
    // BCD: 0x09 + 0x01 = 0x10 in decimal mode.
    let (mut cpu, mut mem) = setup(&[0x69, 0x01]); // ADC #$01
    cpu.regs.a = 0x09;
    cpu.regs.p.insert(Status::D);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.a, 0x10);
    assert!(!cpu.regs.p.contains(Status::C));

    // BCD: 0x99 + 0x01 = 0x00 with carry.
    let (mut cpu, mut mem) = setup(&[0x69, 0x01]);
    cpu.regs.a = 0x99;
    cpu.regs.p.insert(Status::D);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.a, 0x00);
    assert!(cpu.regs.p.contains(Status::C));
}

#[test]
fn sbc_decimal_mode() {
    // BCD: 0x10 - 0x01 = 0x09 (carry set means no borrow in).
    let (mut cpu, mut mem) = setup(&[0xE9, 0x01]); // SBC #$01
    cpu.regs.a = 0x10;
    cpu.regs.p.insert(Status::D | Status::C);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.a, 0x09);
}

#[test]
fn branch_taken_page_cross_cycles() {
    // Taken, same page: BEQ +$10 from PC=$0602 -> $0612 (both page $06).
    let (mut cpu, mut mem) = setup(&[0xF0, 0x10]); // BEQ +$10
    cpu.regs.p.insert(Status::Z);
    let cycles = cpu.step(&mut mem);
    assert_eq!(cpu.regs.pc, 0x0612);
    assert_eq!(cycles, 3); // base 2 + taken (same page)

    // Taken, crossing a page: BEQ +$20 from PC=$06F2 -> $0712 (page $06 -> $07).
    let mut mem = FlatMemory::new();
    mem.load(0x06F0, &[0xF0, 0x20]);
    mem.set_reset_vector(0x06F0);
    let mut cpu = Cpu::new();
    cpu.reset(&mut mem);
    cpu.regs.p.insert(Status::Z);
    let cycles = cpu.step(&mut mem);
    assert_eq!(cpu.regs.pc, 0x0712);
    assert_eq!(cycles, 4); // base 2 + taken + page cross

    // Not taken: base 2 cycles, falls through.
    let (mut cpu, mut mem) = setup(&[0xF0, 0x10]); // BEQ +$10
    cpu.regs.p.remove(Status::Z);
    let cycles = cpu.step(&mut mem);
    assert_eq!(cpu.regs.pc, 0x0602);
    assert_eq!(cycles, 2);
}

#[test]
fn jmp_indirect_page_boundary_bug() {
    // JMP ($30FF): low byte from $30FF, high byte from $3000 (not $3100).
    let mut mem = FlatMemory::new();
    mem.load(0x0600, &[0x6C, 0xFF, 0x30]);
    mem.ram[0x30FF] = 0x34;
    mem.ram[0x3000] = 0x12; // buggy high-byte source
    mem.ram[0x3100] = 0xCD; // would be used if the bug were absent
    mem.set_reset_vector(0x0600);
    let mut cpu = Cpu::new();
    cpu.reset(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.pc, 0x1234);
}

#[test]
fn jsr_rts_round_trip() {
    // JSR $0610 ; (at $0610) RTS -> returns to $0603.
    let mut mem = FlatMemory::new();
    mem.load(0x0600, &[0x20, 0x10, 0x06]); // JSR $0610
    mem.load(0x0610, &[0x60]); // RTS
    mem.set_reset_vector(0x0600);
    let mut cpu = Cpu::new();
    cpu.reset(&mut mem);

    cpu.step(&mut mem); // JSR
    assert_eq!(cpu.regs.pc, 0x0610);
    cpu.step(&mut mem); // RTS
    assert_eq!(cpu.regs.pc, 0x0603);
}

#[test]
fn php_plp_b_and_u_bits() {
    // PHP pushes with B and U set; the value on the stack should reflect that.
    let (mut cpu, mut mem) = setup(&[0x08]); // PHP
    cpu.regs.p = Status::from_bits_retain(0b1000_0001); // N and C only
    let sp_before = cpu.regs.sp;
    cpu.step(&mut mem);
    let pushed = mem.read(0x0100 | sp_before as u16);
    assert_eq!(pushed, 0b1011_0001); // N, U, B, C
}

#[test]
fn nmi_latches_on_falling_edge_and_services_once() {
    let (mut cpu, mut mem) = setup(&[0xEA, 0xEA, 0xEA]); // NOPs
    mem.load(0xFFFA, &[0x00, 0x80]); // NMI vector -> $8000
    mem.load(0x8000, &[0xEA, 0xEA]); // NOPs at the handler

    cpu.set_nmi(true); // line idle (high): no edge
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.pc, 0x0601);

    cpu.set_nmi(false); // high -> low: latch the NMI
    let cycles = cpu.step(&mut mem);
    assert_eq!(cpu.regs.pc, 0x8000);
    assert_eq!(cycles, 7);

    // Holding the line low does not retrigger — edge, not level.
    cpu.set_nmi(false);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.pc, 0x8001);
}

#[test]
fn irq_is_masked_by_i_flag_and_pushes_b_clear() {
    let (mut cpu, mut mem) = setup(&[0x58, 0xEA]); // CLI; NOP
    mem.load(0xFFFE, &[0x00, 0x90]); // IRQ vector -> $9000
    mem.load(0x9000, &[0xEA]);

    cpu.set_irq(true);
    cpu.step(&mut mem); // I is set from reset: IRQ stays masked, CLI runs
    assert_eq!(cpu.regs.pc, 0x0601);

    let sp_before = cpu.regs.sp;
    let cycles = cpu.step(&mut mem); // now unmasked: serviced before the NOP
    assert_eq!(cpu.regs.pc, 0x9000);
    assert_eq!(cycles, 7);
    assert!(cpu.regs.p.contains(Status::I)); // I set on entry

    // Hardware interrupts push the status with B clear (unlike BRK/PHP).
    let pushed_p = mem.read(0x0100 | sp_before.wrapping_sub(2) as u16);
    assert_eq!(pushed_p & 0b0001_0000, 0);
    cpu.set_irq(false);
}

#[test]
fn brk_rti_round_trip() {
    let (mut cpu, mut mem) = setup(&[0x00, 0xFF, 0xEA]); // BRK; padding; NOP
    mem.load(0xFFFE, &[0x00, 0x90]); // IRQ/BRK vector -> $9000
    mem.load(0x9000, &[0x40]); // RTI
    cpu.regs.p.remove(Status::I);
    let sp_before = cpu.regs.sp;

    cpu.step(&mut mem); // BRK
    assert_eq!(cpu.regs.pc, 0x9000);
    assert!(cpu.regs.p.contains(Status::I));

    // BRK pushes the status with B set, and a return address that skips the
    // padding byte ($0602).
    let pushed_p = mem.read(0x0100 | sp_before.wrapping_sub(2) as u16);
    assert_ne!(pushed_p & 0b0001_0000, 0);
    let lo = mem.read(0x0100 | sp_before.wrapping_sub(1) as u16) as u16;
    let hi = mem.read(0x0100 | sp_before as u16) as u16;
    assert_eq!((hi << 8) | lo, 0x0602);

    cpu.step(&mut mem); // RTI
    assert_eq!(cpu.regs.pc, 0x0602);
    assert_eq!(cpu.regs.sp, sp_before);
    assert!(!cpu.regs.p.contains(Status::I)); // pre-BRK status restored
}

#[test]
fn kil_halts_the_cpu() {
    let (mut cpu, mut mem) = setup(&[0x02]); // KIL
    cpu.step(&mut mem);
    assert!(cpu.halted);
    let pc = cpu.regs.pc;
    assert_eq!(cpu.step(&mut mem), 0); // stays halted, no state changes
    assert_eq!(cpu.regs.pc, pc);
}

#[test]
fn stack_push_pull_wraps() {
    // PHA then PLA round-trips the accumulator and restores SP.
    let (mut cpu, mut mem) = setup(&[0x48, 0xA9, 0x00, 0x68]); // PHA; LDA #$00; PLA
    cpu.regs.a = 0x42;
    let sp = cpu.regs.sp;
    cpu.step(&mut mem); // PHA
    cpu.step(&mut mem); // LDA #$00 (clobber A)
    assert_eq!(cpu.regs.a, 0x00);
    cpu.step(&mut mem); // PLA
    assert_eq!(cpu.regs.a, 0x42);
    assert_eq!(cpu.regs.sp, sp);
}
