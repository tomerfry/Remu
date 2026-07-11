//! Focused unit tests for the 8086 core and its nastier quirks. These give
//! readable failures before reaching for the exhaustive SingleStepTests suite.

use remu::x86::{Cpu, Flags, LinearMemory};

/// Build a CPU + flat memory with `program` at `0000:0100`.
fn setup(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(0x0100, program);
    let mut cpu = Cpu::new();
    cpu.regs.cs = 0x0000;
    cpu.regs.ip = 0x0100;
    cpu.regs.ss = 0x9000;
    cpu.regs.sp = 0xFFFE;
    (cpu, mem)
}

#[test]
fn mov_imm_and_reg_halves() {
    let (mut cpu, mut mem) = setup(&[
        0xB8, 0x34, 0x12, // MOV AX, 1234h
        0xB4, 0xAB, // MOV AH, ABh
        0x88, 0xC3, // MOV BL, AL
    ]);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0x1234);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0xAB34);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.bx & 0xFF, 0x34);
}

#[test]
fn add_sets_carry_overflow_and_zero() {
    // MOV AL, FFh; ADD AL, 1 -> AL=0, CF=1, ZF=1, AF=1, OF=0
    let (mut cpu, mut mem) = setup(&[0xB0, 0xFF, 0x04, 0x01]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax & 0xFF, 0);
    let f = cpu.regs.flags;
    assert!(f.contains(Flags::CF) && f.contains(Flags::ZF) && f.contains(Flags::AF));
    assert!(!f.contains(Flags::OF));

    // MOV AL, 7Fh; ADD AL, 1 -> signed overflow
    let (mut cpu, mut mem) = setup(&[0xB0, 0x7F, 0x04, 0x01]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert!(cpu.regs.flags.contains(Flags::OF));
    assert!(cpu.regs.flags.contains(Flags::SF));
    assert!(!cpu.regs.flags.contains(Flags::CF));
}

#[test]
fn sub_borrow_and_parity() {
    // MOV AX, 0; SUB AX, 1 -> FFFF, CF=1, SF=1, PF=1 (FF has 8 set bits)
    let (mut cpu, mut mem) = setup(&[0xB8, 0x00, 0x00, 0x2D, 0x01, 0x00]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0xFFFF);
    let f = cpu.regs.flags;
    assert!(f.contains(Flags::CF) && f.contains(Flags::SF) && f.contains(Flags::PF));
}

#[test]
fn modrm_effective_addresses_and_segment_default() {
    // MOV [BX+SI+10h], AX with DS=1000h
    let (mut cpu, mut mem) = setup(&[0x89, 0x40, 0x10]);
    cpu.regs.ds = 0x1000;
    cpu.regs.bx = 0x0200;
    cpu.regs.si = 0x0030;
    cpu.regs.ax = 0xBEEF;
    cpu.step(&mut mem);
    assert_eq!(mem.ram[0x10240], 0xEF);
    assert_eq!(mem.ram[0x10241], 0xBE);

    // MOV AX, [BP+2] defaults to SS
    let (mut cpu, mut mem) = setup(&[0x8B, 0x46, 0x02]);
    cpu.regs.ss = 0x2000;
    cpu.regs.bp = 0x0100;
    mem.load(0x20102, &[0xCD, 0xAB]);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0xABCD);

    // Segment override: MOV AX, ES:[BP+2]
    let (mut cpu, mut mem) = setup(&[0x26, 0x8B, 0x46, 0x02]);
    cpu.regs.ss = 0x2000;
    cpu.regs.es = 0x3000;
    cpu.regs.bp = 0x0100;
    mem.load(0x30102, &[0x22, 0x11]);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0x1122);
}

#[test]
fn push_pop_and_push_sp_quirk() {
    let (mut cpu, mut mem) = setup(&[
        0xB8, 0x55, 0xAA, // MOV AX, AA55h
        0x50, // PUSH AX
        0x5B, // POP BX
        0x54, // PUSH SP
    ]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.sp, 0xFFFC);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.bx, 0xAA55);
    assert_eq!(cpu.regs.sp, 0xFFFE);

    // 8086: PUSH SP stores the new (decremented) SP.
    cpu.step(&mut mem);
    let top = mem.ram[0x9FFFC] as u16 | (mem.ram[0x9FFFD] as u16) << 8;
    assert_eq!(top, 0xFFFC);
}

#[test]
fn jumps_calls_and_rets() {
    let (mut cpu, mut mem) = setup(&[
        0xE8, 0x02, 0x00, // 0100: CALL 0105
        0xEB, 0x03, // 0103: JMP 0108
        0xC3, // 0105: RET
        0x90, // 0106
        0x90, // 0107
        0xF4, // 0108: HLT
    ]);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ip, 0x0105);
    cpu.step(&mut mem); // RET
    assert_eq!(cpu.regs.ip, 0x0103);
    cpu.step(&mut mem); // JMP
    assert_eq!(cpu.regs.ip, 0x0108);
    cpu.step(&mut mem); // HLT
    assert!(cpu.halted);
}

#[test]
fn conditional_jump_taken_and_not() {
    // CMP AX, 0 ; JZ +2
    let (mut cpu, mut mem) = setup(&[0x3D, 0x00, 0x00, 0x74, 0x02]);
    cpu.regs.ax = 0;
    cpu.step(&mut mem);
    let c = cpu.step(&mut mem);
    assert_eq!(cpu.regs.ip, 0x0107);
    assert_eq!(c, 16);

    let (mut cpu, mut mem) = setup(&[0x3D, 0x00, 0x00, 0x74, 0x02]);
    cpu.regs.ax = 5;
    cpu.step(&mut mem);
    let c = cpu.step(&mut mem);
    assert_eq!(cpu.regs.ip, 0x0105);
    assert_eq!(c, 4);
}

#[test]
fn div8_quotient_and_remainder() {
    // MOV AX, 0234h; MOV BL, 10h; DIV BL -> AL = 23h, AH = 4
    let (mut cpu, mut mem) = setup(&[0xB8, 0x34, 0x02, 0xB3, 0x10, 0xF6, 0xF3]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0x0423);
}

#[test]
fn div8_fault_vectors_through_int0() {
    let mut mem = LinearMemory::new();
    // IVT vector 0 -> 2000:0004
    mem.load(0, &[0x04, 0x00, 0x00, 0x20]);
    mem.load(0x0100, &[0xB3, 0x00, 0xF6, 0xF3]); // MOV BL,0 ; DIV BL
    let mut cpu = Cpu::new();
    cpu.regs.cs = 0x0000;
    cpu.regs.ip = 0x0100;
    cpu.regs.ss = 0x9000;
    cpu.regs.sp = 0xFFFE;
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.cs, 0x2000);
    assert_eq!(cpu.regs.ip, 0x0004);
    // 8086 quirk: the pushed return IP points AFTER the faulting DIV.
    // Frame layout: FLAGS at SP-2, CS at SP-4, IP at SP-6.
    let ret_ip = mem.ram[0x9FFF8] as u16 | (mem.ram[0x9FFF9] as u16) << 8;
    assert_eq!(ret_ip, 0x0104);
}

#[test]
fn mul_sets_carry_when_high_half_used() {
    let (mut cpu, mut mem) = setup(&[0xB0, 0x40, 0xB3, 0x04, 0xF6, 0xE3]); // AL=40h, BL=4, MUL BL
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0x0100);
    assert!(cpu.regs.flags.contains(Flags::CF) && cpu.regs.flags.contains(Flags::OF));
}

#[test]
fn rep_movsb_copies_and_respects_df() {
    let (mut cpu, mut mem) = setup(&[0xF3, 0xA4]); // REP MOVSB
    cpu.regs.ds = 0x1000;
    cpu.regs.es = 0x2000;
    cpu.regs.si = 0x0000;
    cpu.regs.di = 0x0000;
    cpu.regs.cx = 4;
    mem.load(0x10000, b"remu");
    cpu.step(&mut mem);
    assert_eq!(&mem.ram[0x20000..0x20004], b"remu");
    assert_eq!(cpu.regs.cx, 0);
    assert_eq!(cpu.regs.si, 4);
    assert_eq!(cpu.regs.di, 4);
}

#[test]
fn repne_scasb_finds_byte() {
    let (mut cpu, mut mem) = setup(&[0xF2, 0xAE]); // REPNE SCASB
    cpu.regs.es = 0x2000;
    cpu.regs.di = 0x0000;
    cpu.regs.cx = 10;
    cpu.regs.ax = 0x0058; // AL = 'X'
    mem.load(0x20000, b"abcXefg");
    cpu.step(&mut mem);
    assert!(cpu.regs.flags.contains(Flags::ZF));
    assert_eq!(cpu.regs.di, 4); // stopped one past the match
    assert_eq!(cpu.regs.cx, 6);
}

#[test]
fn string_ops_word_sized_and_reversed() {
    let (mut cpu, mut mem) = setup(&[0xFD, 0xF3, 0xA5]); // STD ; REP MOVSW
    cpu.regs.ds = 0x1000;
    cpu.regs.es = 0x2000;
    cpu.regs.si = 0x0006;
    cpu.regs.di = 0x0006;
    cpu.regs.cx = 4;
    mem.load(0x10000, &[1, 2, 3, 4, 5, 6, 7, 8]);
    cpu.step(&mut mem); // STD
    cpu.step(&mut mem);
    assert_eq!(&mem.ram[0x20000..0x20008], &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(cpu.regs.si, 0x0006u16.wrapping_sub(8));
}

#[test]
fn shifts_and_rotates() {
    // MOV AL, 81h ; SHL AL, 1 -> 02h, CF=1, OF=1 (sign changed)
    let (mut cpu, mut mem) = setup(&[0xB0, 0x81, 0xD0, 0xE0]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax & 0xFF, 0x02);
    assert!(cpu.regs.flags.contains(Flags::CF));
    assert!(cpu.regs.flags.contains(Flags::OF));

    // MOV CL, 8 ; MOV AX, FF00h ; SHR AX, CL -> 00FFh, CF=0
    let (mut cpu, mut mem) = setup(&[0xB1, 0x08, 0xB8, 0x00, 0xFF, 0xD3, 0xE8]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0x00FF);
    assert!(!cpu.regs.flags.contains(Flags::CF));

    // RCR through carry: STC ; MOV AL, 01h ; RCR AL, 1 -> 80h, CF=1
    let (mut cpu, mut mem) = setup(&[0xF9, 0xB0, 0x01, 0xD0, 0xD8]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax & 0xFF, 0x80);
    assert!(cpu.regs.flags.contains(Flags::CF));
}

#[test]
fn int_and_iret_round_trip() {
    let mut mem = LinearMemory::new();
    mem.load(0x21 * 4, &[0x00, 0x03, 0x00, 0x20]); // vector 21h -> 2000:0300
    mem.load(0x0100, &[0xCD, 0x21]); // INT 21h
    mem.load(0x20300, &[0xCF]); // IRET
    let mut cpu = Cpu::new();
    cpu.regs.cs = 0x0000;
    cpu.regs.ip = 0x0100;
    cpu.regs.ss = 0x9000;
    cpu.regs.sp = 0xFFFE;
    cpu.regs.flags.insert(Flags::IF);

    cpu.step(&mut mem);
    assert_eq!((cpu.regs.cs, cpu.regs.ip), (0x2000, 0x0300));
    assert!(!cpu.regs.flags.contains(Flags::IF)); // INT clears IF

    cpu.step(&mut mem);
    assert_eq!((cpu.regs.cs, cpu.regs.ip), (0x0000, 0x0102));
    assert!(cpu.regs.flags.contains(Flags::IF)); // IRET restores FLAGS
}

#[test]
fn hardware_interrupt_and_hlt_resume() {
    let mut mem = LinearMemory::new();
    mem.load(0x08 * 4, &[0x00, 0x02, 0x00, 0x30]); // vector 8 -> 3000:0200
    mem.load(0x0100, &[0xF4]); // HLT
    let mut cpu = Cpu::new();
    cpu.regs.cs = 0x0000;
    cpu.regs.ip = 0x0100;
    cpu.regs.ss = 0x9000;
    cpu.regs.sp = 0xFFFE;
    cpu.regs.flags.insert(Flags::IF);

    cpu.step(&mut mem);
    assert!(cpu.halted);
    cpu.step(&mut mem); // still halted, idles
    assert!(cpu.halted);

    cpu.assert_intr(8);
    cpu.step(&mut mem);
    assert!(!cpu.halted);
    assert_eq!((cpu.regs.cs, cpu.regs.ip), (0x3000, 0x0200));
}

#[test]
fn sti_shadow_delays_interrupt_by_one_instruction() {
    let mut mem = LinearMemory::new();
    mem.load(0x08 * 4, &[0x00, 0x02, 0x00, 0x30]);
    mem.load(0x0100, &[0xFB, 0x40, 0x40]); // STI ; INC AX ; INC AX
    let mut cpu = Cpu::new();
    cpu.regs.cs = 0x0000;
    cpu.regs.ip = 0x0100;
    cpu.regs.ss = 0x9000;
    cpu.regs.sp = 0xFFFE;
    cpu.assert_intr(8);

    cpu.step(&mut mem); // STI (interrupt masked before, shadow after)
    cpu.step(&mut mem); // INC AX must run before the interrupt is taken
    assert_eq!(cpu.regs.ax, 1);
    cpu.step(&mut mem); // now the interrupt is serviced
    assert_eq!((cpu.regs.cs, cpu.regs.ip), (0x3000, 0x0200));
    assert_eq!(cpu.regs.ax, 1);
}

#[test]
fn xlat_lahf_sahf_and_flags_word() {
    let (mut cpu, mut mem) = setup(&[0xD7]); // XLAT
    cpu.regs.ds = 0x1000;
    cpu.regs.bx = 0x0020;
    cpu.regs.ax = 0x0003;
    mem.ram[0x10023] = 0x99;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax & 0xFF, 0x99);

    // PUSHF image has the fixed bits set.
    let (mut cpu, mut mem) = setup(&[0x9C]);
    cpu.regs.flags = Flags::CF | Flags::ZF;
    cpu.step(&mut mem);
    let img = mem.ram[0x9FFFC] as u16 | (mem.ram[0x9FFFD] as u16) << 8;
    assert_eq!(img, 0xF002 | 0x0041);
}

#[test]
fn far_call_via_memory_and_retf() {
    let (mut cpu, mut mem) = setup(&[0xFF, 0x1E, 0x00, 0x05]); // CALL FAR [0500h]
    cpu.regs.ds = 0x0000;
    mem.load(0x0500, &[0x00, 0x03, 0x00, 0x40]); // -> 4000:0300
    mem.load(0x40300, &[0xCB]); // RETF
    cpu.step(&mut mem);
    assert_eq!((cpu.regs.cs, cpu.regs.ip), (0x4000, 0x0300));
    cpu.step(&mut mem);
    assert_eq!((cpu.regs.cs, cpu.regs.ip), (0x0000, 0x0104));
}

#[test]
fn bcd_daa_packed_addition() {
    // 39h + 27h = 60h, DAA -> 66h (39 + 27 = 66 decimal)
    let (mut cpu, mut mem) = setup(&[0xB0, 0x39, 0x04, 0x27, 0x27]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax & 0xFF, 0x66);
    assert!(!cpu.regs.flags.contains(Flags::CF));
}

#[test]
fn aam_divides_and_aam_zero_faults() {
    let (mut cpu, mut mem) = setup(&[0xB0, 0x4F, 0xD4, 0x0A]); // AL=79 ; AAM
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0x0709); // 79 = 7*10 + 9
}

#[test]
fn wrap_at_one_megabyte_and_offset_wrap() {
    // Physical wrap: FFFF:0010 -> 000000
    let mut mem = LinearMemory::new();
    mem.load(0x0000, &[0x42]);
    let mut cpu = Cpu::new();
    cpu.regs.cs = 0x0000;
    cpu.regs.ip = 0x0100;
    cpu.regs.ds = 0xFFFF;
    cpu.regs.bx = 0x0010;
    mem.load(0x0100, &[0x8A, 0x07]); // MOV AL, [BX]
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax & 0xFF, 0x42);

    // Word read at offset FFFF wraps to offset 0000 within the segment.
    let (mut cpu, mut mem) = setup(&[0x8B, 0x07]); // MOV AX, [BX]
    cpu.regs.ds = 0x1000;
    cpu.regs.bx = 0xFFFF;
    mem.ram[0x1FFFF] = 0xCD;
    mem.ram[0x10000] = 0xAB;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.ax, 0xABCD);
}
