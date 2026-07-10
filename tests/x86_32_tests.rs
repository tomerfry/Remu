//! Focused unit tests for the 386 core: 16/32-bit operand and address sizes,
//! SIB addressing, the new-to-386 instructions, real-mode exceptions, and the
//! protected-mode plumbing. These give readable failures before reaching for
//! the exhaustive SingleStepTests suite.

use remu::x86_32::{Cpu, DescTable, EFlags, LinearMemory, SegReg, reg};

/// Build a CPU + flat memory with `program` at `0000:1100`, a sane stack and
/// real-mode defaults (mirrors the SingleStepTests rig conventions).
fn setup(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(0x1100, program);
    let mut cpu = Cpu::new();
    cpu.set_cs_ip(0x0000, 0x1100);
    cpu.regs.seg[reg::SS as usize] = remu::x86_32::SegReg::real(0x9000);
    cpu.regs.gpr[reg::ESP as usize] = 0xFFF0;
    (cpu, mem)
}

#[test]
fn mov_imm_operand_sizes() {
    // 16-bit code: MOV AX, imm16; 66: MOV EAX, imm32; MOV AH, imm8.
    let (mut cpu, mut mem) = setup(&[
        0xB8, 0x34, 0x12, // MOV AX, 1234h
        0x66, 0xB8, 0x78, 0x56, 0x34, 0x12, // MOV EAX, 12345678h
        0xB4, 0xAB, // MOV AH, ABh
    ]);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0] & 0xFFFF, 0x1234);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x1234_AB78);
}

#[test]
fn alu_32bit_flags() {
    // MOV EAX, FFFFFFFFh; ADD EAX, 1 -> 0, CF ZF AF set, OF clear.
    let (mut cpu, mut mem) = setup(&[
        0x66, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0x66, 0x05, 0x01, 0x00, 0x00, 0x00,
    ]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0);
    let f = cpu.regs.eflags;
    assert!(f.contains(EFlags::CF) && f.contains(EFlags::ZF) && f.contains(EFlags::AF));
    assert!(!f.contains(EFlags::OF));

    // MOV EAX, 7FFFFFFFh; INC EAX -> signed overflow, CF untouched.
    let (mut cpu, mut mem) = setup(&[0x66, 0xB8, 0xFF, 0xFF, 0xFF, 0x7F, 0x66, 0x40]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x8000_0000);
    assert!(cpu.regs.eflags.contains(EFlags::OF));
    assert!(cpu.regs.eflags.contains(EFlags::SF));
    assert!(!cpu.regs.eflags.contains(EFlags::CF));
}

#[test]
fn sib_addressing() {
    // 67 66 89 44 88 10: MOV [EAX + ECX*4 + 10h], EAX (32-bit EA from
    // 16-bit code via the 67 prefix).
    let (mut cpu, mut mem) = setup(&[0x67, 0x66, 0x89, 0x44, 0x88, 0x10]);
    cpu.regs.gpr[reg::EAX as usize] = 0x0000_2000;
    cpu.regs.gpr[reg::ECX as usize] = 0x0000_0004;
    cpu.step(&mut mem);
    // EA = 2000 + 4*4 + 10 = 2020, DS base 0.
    assert_eq!(mem.ram[0x2020], 0x00);
    assert_eq!(mem.ram[0x2021], 0x20);
}

#[test]
fn sib_scaled_base_quirk() {
    // SIB with index == 100 (none) and scale == 1 applies the scale to the
    // base on a real 386: EA = (EBX << 1) + disp8.
    // 67 88 44 63 10 : MOV [sib+disp8], AL with sib = 01 100 011 (scale 1,
    // no index, base EBX).
    let (mut cpu, mut mem) = setup(&[0x67, 0x88, 0x44, 0x63, 0x10]);
    cpu.regs.gpr[reg::EBX as usize] = 0x0000_1000;
    cpu.regs.gpr[reg::EAX as usize] = 0xAB;
    cpu.step(&mut mem);
    assert_eq!(mem.ram[0x2010], 0xAB); // (1000h << 1) + 10h
}

#[test]
fn push_pop_and_pusha() {
    let (mut cpu, mut mem) = setup(&[
        0x66, 0x68, 0x78, 0x56, 0x34, 0x12, // PUSH 12345678h
        0x66, 0x5B, // POP EBX
        0x60, // PUSHA
        0x61, // POPA
    ]);
    let sp0 = cpu.regs.gpr[reg::ESP as usize];
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[reg::ESP as usize], sp0 - 4);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[reg::EBX as usize], 0x1234_5678);
    assert_eq!(cpu.regs.gpr[reg::ESP as usize], sp0);

    let before = cpu.regs.gpr;
    cpu.step(&mut mem); // PUSHA
    assert_eq!(cpu.regs.gpr[reg::ESP as usize], sp0 - 16);
    cpu.step(&mut mem); // POPA
    assert_eq!(cpu.regs.gpr, before);
}

#[test]
fn push_esp_is_pre_decrement_value() {
    // 386 fixes the 8086 PUSH SP quirk: the pushed value is the old SP.
    let (mut cpu, mut mem) = setup(&[0x54]); // PUSH SP
    let sp0 = cpu.regs.gpr[reg::ESP as usize] as u16;
    cpu.step(&mut mem);
    let top = 0x90000 + (sp0 as usize - 2);
    let pushed = mem.ram[top] as u16 | (mem.ram[top + 1] as u16) << 8;
    assert_eq!(pushed, sp0);
}

#[test]
fn movzx_movsx_setcc() {
    let (mut cpu, mut mem) = setup(&[
        0xB3, 0x80, // MOV BL, 80h
        0x66, 0x0F, 0xB6, 0xC3, // MOVZX EAX, BL
        0x66, 0x0F, 0xBE, 0xCB, // MOVSX ECX, BL
        0x0F, 0x94, 0xC2, // SETZ DL
    ]);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x80);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[1], 0xFFFF_FF80);
    cpu.regs.eflags.insert(EFlags::ZF);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg8(2), 1);
}

#[test]
fn bt_family() {
    // BTS reg: set bit 9 of AX.
    let (mut cpu, mut mem) = setup(&[
        0x0F, 0xAB, 0xD8, // BTS AX, BX
        0x0F, 0xBA, 0xE0, 0x09, // BT AX, 9
    ]);
    cpu.regs.set_reg16(0, 0);
    cpu.regs.set_reg16(3, 9);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg16(0), 0x200);
    assert!(!cpu.regs.eflags.contains(EFlags::CF));
    cpu.step(&mut mem);
    assert!(cpu.regs.eflags.contains(EFlags::CF));
}

#[test]
fn bt_memory_bit_offset_is_signed() {
    // BT [0x2000], BX with BX = -16 tests bit 0 of the word at 0x1FFE.
    let (mut cpu, mut mem) = setup(&[0x0F, 0xA3, 0x1E, 0x00, 0x20]);
    cpu.regs.set_reg16(3, (-16i16) as u16);
    mem.ram[0x1FFE] = 0x01;
    cpu.step(&mut mem);
    assert!(cpu.regs.eflags.contains(EFlags::CF));
}

#[test]
fn shld_shrd() {
    let (mut cpu, mut mem) = setup(&[
        0x66, 0x0F, 0xA4, 0xD8, 0x04, // SHLD EAX, EBX, 4
        0x66, 0x0F, 0xAC, 0xD9, 0x08, // SHRD ECX, EBX, 8
    ]);
    cpu.regs.gpr[0] = 0x1234_5678;
    cpu.regs.gpr[3] = 0x9ABC_DEF0;
    cpu.regs.gpr[1] = 0x1111_2222;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x2345_6789);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[1], 0xF011_1122);
}

#[test]
fn imul_forms() {
    let (mut cpu, mut mem) = setup(&[
        0x6B, 0xC3, 0x10, // IMUL AX, BX, 16
        0x66, 0x0F, 0xAF, 0xC3, // IMUL EAX, EBX
    ]);
    cpu.regs.set_reg16(3, 100);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg16(0), 1600);
    cpu.regs.gpr[0] = 3;
    cpu.regs.gpr[3] = 0xFFFF_FFFF; // -1
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0xFFFF_FFFD); // -3
    assert!(!cpu.regs.eflags.contains(EFlags::CF));
}

#[test]
fn mul_div_32() {
    let (mut cpu, mut mem) = setup(&[
        0x66, 0xF7, 0xE3, // MUL EBX
        0x66, 0xF7, 0xF1, // DIV ECX
    ]);
    cpu.regs.gpr[0] = 0x1000_0000;
    cpu.regs.gpr[3] = 0x10;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0);
    assert_eq!(cpu.regs.gpr[2], 1); // EDX:EAX = 1_0000_0000
    assert!(cpu.regs.eflags.contains(EFlags::CF));
    cpu.regs.gpr[1] = 0x10;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x1000_0000);
    assert_eq!(cpu.regs.gpr[2], 0);
}

#[test]
fn divide_fault_is_a_fault() {
    // DIV by zero: the pushed IP must point AT the DIV (fault semantics,
    // unlike the 8086 which pushes the next IP).
    let (mut cpu, mut mem) = setup(&[0xF6, 0xF3]); // DIV BL with BL=0
    // IVT entry 0 -> 0x0000:0x3000.
    mem.load(0, &[0x00, 0x30, 0x00, 0x00]);
    cpu.regs.set_reg8(3, 0);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x3000);
    // Pushed IP is at SS:SP+0.
    let sp = cpu.regs.gpr[reg::ESP as usize] as usize;
    let ip = mem.ram[0x90000 + sp] as u16 | (mem.ram[0x90000 + sp + 1] as u16) << 8;
    assert_eq!(ip, 0x1100);
}

#[test]
fn strings_rep_ecx() {
    // 67 F3 66 AB: REP STOSD with ECX (address-size 32).
    let (mut cpu, mut mem) = setup(&[0x67, 0xF3, 0x66, 0xAB]);
    cpu.regs.gpr[0] = 0xDEAD_BEEF;
    cpu.regs.gpr[reg::ECX as usize] = 3;
    cpu.regs.gpr[reg::EDI as usize] = 0x4000;
    cpu.regs.seg[reg::ES as usize] = remu::x86_32::SegReg::real(0);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[reg::ECX as usize], 0);
    assert_eq!(cpu.regs.gpr[reg::EDI as usize], 0x400C);
    assert_eq!(mem.ram[0x4000], 0xEF);
    assert_eq!(mem.ram[0x400B], 0xDE);
}

#[test]
fn cmps_repe_stops_on_mismatch() {
    let (mut cpu, mut mem) = setup(&[0xF3, 0xA6]); // REPE CMPSB
    mem.load(0x5000, b"abcX");
    mem.load(0x6000, b"abcY");
    cpu.regs.set_reg16(reg::ESI, 0x5000);
    cpu.regs.set_reg16(reg::EDI, 0x6000);
    cpu.regs.set_reg16(reg::ECX, 8);
    cpu.regs.seg[reg::ES as usize] = remu::x86_32::SegReg::real(0);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg16(reg::ECX), 4); // stopped after the 4th element
    assert!(!cpu.regs.eflags.contains(EFlags::ZF));
}

#[test]
fn enter_leave() {
    let (mut cpu, mut mem) = setup(&[
        0xC8, 0x10, 0x00, 0x00, // ENTER 16, 0
        0xC9, // LEAVE
    ]);
    let sp0 = cpu.regs.gpr[reg::ESP as usize] as u16;
    let bp0 = 0x1234u16;
    cpu.regs.set_reg16(reg::EBP, bp0);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg16(reg::EBP), sp0 - 2);
    assert_eq!(cpu.regs.reg16(reg::ESP), sp0 - 2 - 16);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg16(reg::EBP), bp0);
    assert_eq!(cpu.regs.reg16(reg::ESP), sp0);
}

#[test]
fn jcxz_uses_address_size() {
    // 67 E3 xx uses ECX; plain E3 uses CX.
    let (mut cpu, mut mem) = setup(&[0x67, 0xE3, 0x02, 0x90, 0x90, 0x90]);
    cpu.regs.gpr[reg::ECX as usize] = 0x0001_0000; // CX == 0, ECX != 0
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x1103); // not taken (ECX != 0)

    let (mut cpu, mut mem) = setup(&[0xE3, 0x02, 0x90, 0x90]);
    cpu.regs.gpr[reg::ECX as usize] = 0x0001_0000;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x1104); // taken (CX == 0)
}

#[test]
fn limit_violation_faults() {
    // 67-prefixed access beyond the 64 KiB real-mode limit -> #GP -> IVT 13.
    let (mut cpu, mut mem) = setup(&[0x67, 0x8A, 0x83, 0x00, 0x00, 0x01, 0x00]); // MOV AL,[EBX+10000h]
    mem.load(13 * 4, &[0x00, 0x40, 0x00, 0x00]); // IVT 13 -> 0:4000
    cpu.regs.gpr[reg::EBX as usize] = 0;
    let eax0 = cpu.regs.gpr[0];
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x4000);
    assert_eq!(
        cpu.regs.gpr[0], eax0,
        "faulting instruction must not modify EAX"
    );
    // Pushed IP points at the instruction (fault).
    let sp = cpu.regs.gpr[reg::ESP as usize] as usize;
    let ip = mem.ram[0x90000 + sp] as u16 | (mem.ram[0x90000 + sp + 1] as u16) << 8;
    assert_eq!(ip, 0x1100);
}

#[test]
fn word_access_straddling_limit_faults() {
    // MOV AX, [0xFFFF] straddles the 64 KiB limit: #GP on a 386 (the 8086
    // wrapped instead).
    let (mut cpu, mut mem) = setup(&[0xA1, 0xFF, 0xFF]);
    mem.load(13 * 4, &[0x00, 0x40, 0x00, 0x00]);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x4000);
}

#[test]
fn int_and_iret_frame() {
    let (mut cpu, mut mem) = setup(&[0xCD, 0x21]); // INT 21h
    mem.load(0x21 * 4, &[0x00, 0x50, 0x00, 0x20]); // IVT 21h -> 2000:5000
    mem.load(0x25000, &[0xCF]); // IRET at the handler
    cpu.regs.eflags.insert(EFlags::DF);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.seg[reg::CS as usize].sel, 0x2000);
    assert_eq!(cpu.regs.seg[reg::CS as usize].base, 0x20000);
    assert_eq!(cpu.regs.eip, 0x5000);
    cpu.step(&mut mem); // IRET
    assert_eq!(cpu.regs.seg[reg::CS as usize].sel, 0x0000);
    assert_eq!(cpu.regs.eip, 0x1102);
    assert!(cpu.regs.eflags.contains(EFlags::DF));
}

#[test]
fn undefined_opcode_faults() {
    // ARPL (63) in real mode -> #UD (IVT 6).
    let (mut cpu, mut mem) = setup(&[0x63, 0xC0]);
    mem.load(6 * 4, &[0x00, 0x60, 0x00, 0x00]);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x6000);
}

#[test]
fn lock_on_unlockable_instruction_faults() {
    // LOCK MOV AX, BX -> #UD.
    let (mut cpu, mut mem) = setup(&[0xF0, 0x89, 0xD8]);
    mem.load(6 * 4, &[0x00, 0x60, 0x00, 0x00]);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x6000);

    // LOCK ADD [mem], AX is legal.
    let (mut cpu, mut mem) = setup(&[0xF0, 0x01, 0x06, 0x00, 0x20]);
    mem.load(6 * 4, &[0x00, 0x60, 0x00, 0x00]);
    cpu.step(&mut mem);
    assert_ne!(cpu.regs.eip, 0x6000);
}

#[test]
fn xlat_and_cwde_cdq() {
    let (mut cpu, mut mem) = setup(&[
        0xD7, // XLAT
        0x66, 0x98, // CWDE
        0x66, 0x99, // CDQ
    ]);
    cpu.regs.set_reg16(3, 0x3000);
    cpu.regs.set_reg8(0, 5);
    mem.ram[0x3005] = 0x99;
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg8(0), 0x99);
    cpu.regs.set_reg16(0, 0x8000);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[0], 0xFFFF_8000);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.gpr[2], 0xFFFF_FFFF);
}

#[test]
fn bound_raises_br() {
    let (mut cpu, mut mem) = setup(&[0x62, 0x06, 0x00, 0x20]); // BOUND AX, [2000h]
    mem.load(5 * 4, &[0x00, 0x70, 0x00, 0x00]); // IVT 5
    mem.load(0x2000, &[0x10, 0x00, 0x20, 0x00]); // bounds [16, 32]
    cpu.regs.set_reg16(0, 100);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x7000);

    let (mut cpu, mut mem) = setup(&[0x62, 0x06, 0x00, 0x20]);
    mem.load(0x2000, &[0x10, 0x00, 0x20, 0x00]);
    cpu.regs.set_reg16(0, 20);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x1104); // in range: falls through
}

#[test]
fn call_far_and_retf() {
    let (mut cpu, mut mem) = setup(&[0x9A, 0x00, 0x50, 0x00, 0x30]); // CALL 3000:5000
    mem.load(0x35000, &[0xCB]); // RETF
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.seg[reg::CS as usize].sel, 0x3000);
    assert_eq!(cpu.regs.eip, 0x5000);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.seg[reg::CS as usize].sel, 0x0000);
    assert_eq!(cpu.regs.eip, 0x1105);
}

#[test]
fn loop_uses_address_size_counter() {
    // 67 E2 FD: LOOP with ECX counter.
    let (mut cpu, mut mem) = setup(&[0x90, 0x67, 0xE2, 0xFC]); // NOP; LOOP -4
    cpu.regs.gpr[reg::ECX as usize] = 3;
    cpu.step(&mut mem); // NOP
    cpu.step(&mut mem); // LOOP -> taken
    assert_eq!(cpu.regs.eip, 0x1100);
    assert_eq!(cpu.regs.gpr[reg::ECX as usize], 2);
}

#[test]
fn salc_and_lahf() {
    let (mut cpu, mut mem) = setup(&[0xF9, 0xD6, 0x9F]); // STC; SALC; LAHF
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg8(0), 0xFF);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg8(4) & 0x01, 0x01); // CF in AH bit 0
    assert_eq!(cpu.regs.reg8(4) & 0x02, 0x02); // bit 1 reads 1
}

#[test]
fn smsw_sgdt_real_mode() {
    let (mut cpu, mut mem) = setup(&[
        0x0F, 0x01, 0x26, 0x00, 0x20, // SMSW [2000h]
        0x0F, 0x01, 0x06, 0x00, 0x21, // SGDT [2100h]
    ]);
    cpu.regs.cr0 = 0x7FFF_FFE0 | 0x10;
    cpu.regs.gdtr = remu::x86_32::DescTable {
        base: 0x0012_3456,
        limit: 0x27,
    };
    cpu.step(&mut mem);
    assert_eq!(mem.ram[0x2000], 0xF0);
    assert_eq!(mem.ram[0x2001], 0xFF);
    cpu.step(&mut mem);
    assert_eq!(mem.ram[0x2100], 0x27);
    assert_eq!(mem.ram[0x2101], 0x00);
    assert_eq!(mem.ram[0x2102], 0x56);
    assert_eq!(mem.ram[0x2103], 0x34);
    assert_eq!(mem.ram[0x2104], 0x12);
}

// --- Protected mode -----------------------------------------------------------

/// Descriptor bytes for base/limit/access/flags.
fn descriptor(base: u32, limit: u32, access: u8, flags: u8) -> [u8; 8] {
    [
        limit as u8,
        (limit >> 8) as u8,
        base as u8,
        (base >> 8) as u8,
        (base >> 16) as u8,
        access,
        ((limit >> 16) as u8 & 0x0F) | (flags << 4),
        (base >> 24) as u8,
    ]
}

/// Enter protected mode with a small GDT: 08 = flat 32-bit code,
/// 10 = flat 32-bit data, 18 = 16-bit code (for returning), stack in data.
fn setup_pm() -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    // GDT at 0x0500.
    mem.load(0x0508, &descriptor(0, 0xF_FFFF, 0x9A, 0xC)); // 08: code, 4K gran, D
    mem.load(0x0510, &descriptor(0, 0xF_FFFF, 0x92, 0xC)); // 10: data
    mem.load(0x0518, &descriptor(0, 0xFFFF, 0x9A, 0x0)); // 18: 16-bit code

    // Real-mode stub at 0x1100:
    //   LGDT [0x0560]; MOV EAX,CR0; OR AL,1; MOV CR0,EAX; JMP far 08:00002000
    mem.load(0x0560, &[0x47, 0x00, 0x00, 0x05, 0x00, 0x00]); // limit 47, base 0x500
    mem.load(
        0x1100,
        &[
            0x0F, 0x01, 0x16, 0x60, 0x05, // LGDT [0x0560]
            0x0F, 0x20, 0xC0, // MOV EAX, CR0
            0x0C, 0x01, // OR AL, 1
            0x0F, 0x22, 0xC0, // MOV CR0, EAX
            0x66, 0xEA, 0x00, 0x20, 0x00, 0x00, 0x08, 0x00, // JMP 0008:00002000
        ],
    );
    let mut cpu = Cpu::new();
    cpu.set_cs_ip(0x0000, 0x1100);
    cpu.regs.seg[reg::SS as usize] = remu::x86_32::SegReg::real(0x9000);
    cpu.regs.gpr[reg::ESP as usize] = 0xFFF0;
    for _ in 0..5 {
        cpu.step(&mut mem);
    }
    (cpu, mem)
}

#[test]
fn enter_protected_mode_and_load_segments() {
    let (mut cpu, mut mem) = setup_pm();
    assert!(cpu.protected_mode());
    assert_eq!(cpu.regs.seg[reg::CS as usize].sel, 0x08);
    assert_eq!(cpu.regs.eip, 0x2000);
    assert!(cpu.regs.seg[reg::CS as usize].db());

    // MOV AX, 0x10; MOV DS, AX; MOV [0x00345678], EAX (flat 4 GiB data).
    mem.load(
        0x2000,
        &[
            0x66, 0xB8, 0x10, 0x00, // MOV AX, 10h (osize16 via 66 in 32-bit code)
            0x8E, 0xD8, // MOV DS, AX
            0xB8, 0xEF, 0xBE, 0x00, 0x00, // MOV EAX, 0BEEFh
            0xA3, 0x78, 0x56, 0x34, 0x00, // MOV [345678h], EAX
        ],
    );
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.reg16(0), 0x10);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.seg[reg::DS as usize].base, 0);
    assert_eq!(cpu.regs.seg[reg::DS as usize].limit, 0xFFFF_FFFF);
    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(mem.ram[0x345678], 0xEF);
    assert_eq!(mem.ram[0x345679], 0xBE);

    // The GDT data descriptor got its accessed bit set by the load.
    assert_eq!(mem.ram[0x515] & 1, 1);
}

#[test]
fn pm_interrupt_gate() {
    let (mut cpu, mut mem) = setup_pm();
    // IDT at 0x0800: vector 32 -> 32-bit interrupt gate, sel 08, offset 0x2800.
    let mut gate = [0u8; 8];
    gate[0] = 0x00;
    gate[1] = 0x28; // offset low
    gate[2] = 0x08;
    gate[3] = 0x00; // selector
    gate[5] = 0x8E; // present, DPL0, 32-bit interrupt gate
    mem.load(0x0800 + 32 * 8, &gate);
    mem.load(0x0860, &[0xFF, 0x01, 0x00, 0x08, 0x00, 0x00]); // IDTR image

    // LIDT [0x0860]; MOV AX,10h; MOV SS,AX; MOV ESP, 0x9F000; STI?; INT 32
    mem.load(
        0x2000,
        &[
            0x0F, 0x01, 0x1D, 0x60, 0x08, 0x00, 0x00, // LIDT [860h]
            0x66, 0xB8, 0x10, 0x00, // MOV AX, 10h
            0x8E, 0xD0, // MOV SS, AX
            0xBC, 0x00, 0xF0, 0x09, 0x00, // MOV ESP, 9F000h
            0xCD, 0x20, // INT 32
        ],
    );
    mem.load(0x2800, &[0xCF]); // IRETD
    for _ in 0..5 {
        cpu.step(&mut mem);
    }
    assert_eq!(cpu.regs.eip, 0x2800);
    assert!(cpu.protected_mode());
    // Frame: EFLAGS, CS, EIP (32-bit each).
    let sp = cpu.regs.gpr[reg::ESP as usize] as usize;
    assert_eq!(sp, 0x9F000 - 12);
    let ret_eip = u32::from_le_bytes(mem.ram[sp..sp + 4].try_into().unwrap());
    assert_eq!(ret_eip, 0x2014);
    // IF cleared by the interrupt gate.
    assert!(!cpu.regs.eflags.contains(EFlags::IF));

    cpu.step(&mut mem); // IRETD
    assert_eq!(cpu.regs.eip, 0x2014);
    assert_eq!(cpu.regs.gpr[reg::ESP as usize], 0x9F000);
}

#[test]
fn pm_privilege_fault_on_hlt() {
    let (mut cpu, mut mem) = setup_pm();
    // Add ring-3 code/data descriptors and a call path down to ring 3 via
    // IRETD, then attempt HLT -> #GP. Descriptors: 20|3 code DPL3, 28|3 data.
    mem.load(0x0520, &descriptor(0, 0xF_FFFF, 0xFA, 0xC)); // ring3 code
    mem.load(0x0528, &descriptor(0, 0xF_FFFF, 0xF2, 0xC)); // ring3 data
    // IDT gate for #GP (vector 13) -> ring0 handler at 0x2C00.
    let mut gate = [0u8; 8];
    gate[0] = 0x00;
    gate[1] = 0x2C;
    gate[2] = 0x08;
    gate[3] = 0x00;
    gate[5] = 0x8E;
    mem.load(0x0800 + 13 * 8, &gate);
    mem.load(0x0860, &[0xFF, 0x01, 0x00, 0x08, 0x00, 0x00]);

    // TSS for the return path (ring0 stack), TR not strictly needed for the
    // downward transition, but the #GP handler entry from ring 3 needs SS0.
    // Build a minimal 32-bit TSS at 0x3000: SS0:ESP0 = 10h:0x9E000.
    let mut tss = [0u8; 0x68];
    tss[4..8].copy_from_slice(&0x0009_E000u32.to_le_bytes());
    tss[8..10].copy_from_slice(&0x0010u16.to_le_bytes());
    tss[0x66] = 0x68; // iobase beyond limit -> all ports fault at CPL3
    mem.load(0x3000, &tss);
    mem.load(0x0530, &descriptor(0x3000, 0x67, 0x89, 0x0)); // 30: avail 32-bit TSS

    mem.load(
        0x2000,
        &[
            0x0F, 0x01, 0x1D, 0x60, 0x08, 0x00, 0x00, // LIDT [860h]
            0x66, 0xB8, 0x30, 0x00, // MOV AX, 30h
            0x0F, 0x00, 0xD8, // LTR AX
            0x66, 0xB8, 0x10, 0x00, // MOV AX, 10h
            0x8E, 0xD0, // MOV SS, AX
            0xBC, 0x00, 0xF0, 0x09, 0x00, // MOV ESP, 9F000h
            // Build IRETD frame to ring 3: SS=2B, ESP=9C000, EFL=2, CS=23, EIP=2A00
            0x6A, 0x2B, // PUSH 2B
            0x68, 0x00, 0xC0, 0x09, 0x00, // PUSH 9C000h
            0x6A, 0x02, // PUSH 2
            0x6A, 0x23, // PUSH 23h
            0x68, 0x00, 0x2A, 0x00, 0x00, // PUSH 2A00h
            0xCF, // IRETD
        ],
    );
    mem.load(0x2A00, &[0xF4]); // ring-3 code: HLT -> #GP
    for _ in 0..12 {
        cpu.step(&mut mem);
    }
    assert_eq!(cpu.cpl(), 3, "should be in ring 3 after IRETD");
    assert_eq!(cpu.regs.eip, 0x2A00);

    cpu.step(&mut mem); // HLT faults -> #GP handler at ring 0
    assert_eq!(cpu.regs.eip, 0x2C00);
    assert_eq!(cpu.cpl(), 0);
    // Inner frame: SS3:ESP3 pushed above EFLAGS/CS/EIP/error.
    assert_eq!(cpu.regs.gpr[reg::ESP as usize], 0x9E000 - 24);
    assert!(!cpu.halted);
}

// --- Paging ---------------------------------------------------------------------

#[test]
fn paging_translation_and_fault() {
    let (mut cpu, mut mem) = setup_pm();
    // Identity-map the first 4 MiB via one page table, EXCEPT page 0x5000
    // which is left not-present. Page dir at 0x10000, page table at 0x11000.
    mem.load(0x10000, &0x0001_1003u32.to_le_bytes()); // PDE 0: table @ 11000, P|RW
    for i in 0..1024u32 {
        let pte: u32 = (i << 12) | 3;
        let pte = if i == 5 { 0 } else { pte };
        mem.load(0x11000 + i * 4, &pte.to_le_bytes());
    }
    // IDT gate 14 (#PF) -> handler 0x2E00.
    let mut gate = [0u8; 8];
    gate[0] = 0x00;
    gate[1] = 0x2E;
    gate[2] = 0x08;
    gate[3] = 0x00;
    gate[5] = 0x8E;
    mem.load(0x0800 + 14 * 8, &gate);
    mem.load(0x0860, &[0xFF, 0x01, 0x00, 0x08, 0x00, 0x00]);

    mem.load(
        0x2000,
        &[
            0x0F, 0x01, 0x1D, 0x60, 0x08, 0x00, 0x00, // LIDT [860h]
            0x66, 0xB8, 0x10, 0x00, // MOV AX, 10h
            0x8E, 0xD0, // MOV SS, AX
            0xBC, 0x00, 0xF0, 0x09, 0x00, // MOV ESP, 9F000h
            0xB8, 0x00, 0x00, 0x01, 0x00, // MOV EAX, 10000h (page dir)
            0x0F, 0x22, 0xD8, // MOV CR3, EAX
            0x0F, 0x20, 0xC0, // MOV EAX, CR0
            0x0D, 0x00, 0x00, 0x00, 0x80, // OR EAX, 80000000h
            0x0F, 0x22, 0xC0, // MOV CR0, EAX (PG on)
            0xB8, 0x11, 0x00, 0x00, 0x00, // MOV EAX, 11h
            0xA3, 0x00, 0x40, 0x00, 0x00, // MOV [4000h], EAX (mapped)
            0xA3, 0x00, 0x50, 0x00, 0x00, // MOV [5000h], EAX (#PF)
        ],
    );
    mem.load(0x2E00, &[0x90]);
    for _ in 0..11 {
        cpu.step(&mut mem);
    }
    assert_eq!(mem.ram[0x4000], 0x11, "write through identity mapping");
    // PTE 4 accessed+dirty.
    assert_eq!(mem.ram[0x11000 + 4 * 4] & 0x60, 0x60);

    cpu.step(&mut mem); // MOV [5000h] -> #PF
    assert_eq!(cpu.regs.eip, 0x2E00);
    assert_eq!(cpu.regs.cr2, 0x5000);
    // Error code on the stack: bit0=0 (not present), bit1=1 (write).
    let sp = cpu.regs.gpr[reg::ESP as usize] as usize;
    assert_eq!(mem.ram[sp], 0x02);
}

// --- Privilege-transition regression tests ------------------------------------
//
// These cover the paths the SingleStepTests suite cannot reach (it is real
// mode only): implicit supervisor accesses under paging, nested-fault
// classification, V86 delivery, and the TSS/LDT/IRET edge cases.

/// An interrupt/trap gate: `access` is 0x8E (32-bit int), 0x86 (16-bit int),
/// 0x8F (32-bit trap), plus `0x60` to raise the gate DPL to 3.
fn gate(offset: u32, sel: u16, access: u8) -> [u8; 8] {
    [
        offset as u8,
        (offset >> 8) as u8,
        sel as u8,
        (sel >> 8) as u8,
        0,
        access,
        (offset >> 16) as u8,
        (offset >> 24) as u8,
    ]
}

/// A 32-bit available TSS whose ring-0 stack is `ss0:esp0`.
fn tss(esp0: u32, ss0: u16) -> [u8; 0x68] {
    let mut t = [0u8; 0x68];
    t[4..8].copy_from_slice(&esp0.to_le_bytes());
    t[8..10].copy_from_slice(&ss0.to_le_bytes());
    t[0x66] = 0x68; // I/O bitmap base beyond the limit: all ports trap
    t
}

/// Load SS with the flat 32-bit data segment (`setup_pm` leaves the
/// real-mode stack in place, which no protected-mode test wants).
fn flat_ss(cpu: &mut Cpu, esp: u32) {
    cpu.regs.seg[reg::SS as usize] = SegReg {
        sel: 0x10,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0C93,
    };
    cpu.regs.gpr[reg::ESP as usize] = esp;
}

/// Attach an IDT at `base` and a task register pointing at `tss_base`.
fn attach_idt_tss(cpu: &mut Cpu, idt_base: u32, tss_base: u32, tss_sel: u16) {
    cpu.regs.idtr = DescTable {
        base: idt_base,
        limit: 0x7FF,
    };
    cpu.regs.tr = SegReg {
        sel: tss_sel,
        base: tss_base,
        limit: 0x67,
        attrs: 0x0089, // present, 32-bit available TSS
    };
}

#[test]
fn paging_treats_implicit_accesses_as_supervisor() {
    // A ring-3 `INT n` must reach its ring-0 handler even when the GDT, IDT,
    // TSS and kernel stack live on supervisor-only pages: the descriptor
    // reads and the inner-stack pushes happen while CPL is still 3, but the
    // 386 performs them with supervisor privilege.
    let (mut cpu, mut mem) = setup_pm();

    mem.load(0x0520, &descriptor(0, 0xF_FFFF, 0xFA, 0xC)); // 20: ring-3 code
    mem.load(0x0528, &descriptor(0, 0xF_FFFF, 0xF2, 0xC)); // 28: ring-3 data
    mem.load(0x0530, &descriptor(0x3000, 0x67, 0x89, 0x0)); // 30: TSS
    mem.load(0x3000, &tss(0x7000, 0x10)); // ring-0 stack: page 6

    // Gates: INT 40h (DPL 3) -> 2C00h; #PF -> 2E00h; #DF -> 2F00h.
    mem.load(0x0800 + 0x40 * 8, &gate(0x2C00, 0x08, 0xEE));
    mem.load(0x0800 + 14 * 8, &gate(0x2E00, 0x08, 0x8E));
    mem.load(0x0800 + 8 * 8, &gate(0x2F00, 0x08, 0x8E));
    attach_idt_tss(&mut cpu, 0x0800, 0x3000, 0x30);

    // Identity-map 0..0x11FFF. Only the ring-3 code (page 4) and ring-3
    // stack (page 5) are user pages; the GDT/IDT (page 0), kernel code
    // (page 2), TSS (page 3) and kernel stack (page 6) are supervisor-only.
    mem.load(0x10000, &0x0001_1007u32.to_le_bytes()); // PDE: P|RW|US
    for i in 0..0x12u32 {
        let user = matches!(i, 4 | 5);
        let pte = (i << 12) | if user { 7 } else { 3 };
        mem.load(0x11000 + i * 4, &pte.to_le_bytes());
    }
    cpu.regs.cr3 = 0x10000;
    cpu.regs.cr0 |= remu::x86_32::cr0::PG;

    // Ring-0 stub: build an IRETD frame and drop to ring 3 at 0x4000.
    flat_ss(&mut cpu, 0x7000);
    mem.load(
        0x2000,
        &[
            0x6A, 0x2B, // PUSH 2Bh   (ring-3 SS)
            0x68, 0x00, 0x60, 0x00, 0x00, // PUSH 6000h (ring-3 ESP)
            0x6A, 0x02, // PUSH 2     (EFLAGS)
            0x6A, 0x23, // PUSH 23h   (ring-3 CS)
            0x68, 0x00, 0x40, 0x00, 0x00, // PUSH 4000h (ring-3 EIP)
            0xCF, // IRETD
        ],
    );
    mem.load(0x4000, &[0xCD, 0x40]); // ring 3: INT 40h
    mem.load(0x2C00, &[0x90]);

    for _ in 0..6 {
        cpu.step(&mut mem);
    }
    assert_eq!(cpu.cpl(), 3, "IRETD should reach ring 3");
    assert_eq!(cpu.regs.eip, 0x4000);

    cpu.step(&mut mem); // INT 40h
    assert!(!cpu.shutdown, "delivery must not triple-fault");
    assert_eq!(
        cpu.regs.eip, 0x2C00,
        "expected the INT 40h handler, not #PF ({:#X}) or #DF ({:#X})",
        0x2E00, 0x2F00
    );
    assert_eq!(cpu.cpl(), 0);
    // The frame landed on the supervisor-only kernel stack.
    assert_eq!(cpu.regs.gpr[reg::ESP as usize], 0x7000 - 20);
}

#[test]
fn nested_fault_during_external_interrupt_is_serial() {
    // An external interrupt is benign, so a fault raised while delivering it
    // is handled serially rather than escalating to #DF. The error code of
    // that fault carries the EXT bit, since the event was external.
    let (mut cpu, mut mem) = setup_pm();

    // 38: a code segment that is NOT present -> #NP when the gate loads it.
    mem.load(0x0538, &descriptor(0, 0xF_FFFF, 0x1A, 0xC));
    mem.load(0x0800 + 0x20 * 8, &gate(0x2C00, 0x38, 0x8E)); // IRQ -> absent CS
    mem.load(0x0800 + 11 * 8, &gate(0x2D00, 0x08, 0x8E)); // #NP handler
    mem.load(0x0800 + 8 * 8, &gate(0x2F00, 0x08, 0x8E)); // #DF handler
    attach_idt_tss(&mut cpu, 0x0800, 0x3000, 0x30);

    flat_ss(&mut cpu, 0x7000);
    cpu.regs.eflags.insert(EFlags::IF);
    cpu.assert_intr(0x20);
    cpu.step(&mut mem);

    assert!(!cpu.shutdown);
    assert_eq!(
        cpu.regs.eip, 0x2D00,
        "a benign external interrupt + #NP is handled serially, not as #DF"
    );
    // Error code at the top of the frame: selector 0x38 with EXT set.
    let sp = cpu.regs.gpr[reg::ESP as usize] as usize;
    let err = u32::from_le_bytes(mem.ram[sp..sp + 4].try_into().unwrap());
    assert_eq!(err, 0x39, "EXT bit must be set for an external event");
}

#[test]
fn v86_delivery_fault_preserves_vm() {
    // A fault partway through V86 interrupt delivery must roll back the
    // cleared VM flag, so the nested fault is delivered as a V86 event (it
    // pushes the four V86 data selectors and nulls them) rather than through
    // the plain protected-mode path.
    let (mut cpu, mut mem) = setup_pm();

    // 38: ring-0 stack, base 7000h, limit 3Fh, 16-bit (B=0).
    mem.load(0x0538, &descriptor(0x7000, 0x3F, 0x92, 0x0));
    mem.load(0x0530, &descriptor(0x3000, 0x67, 0x89, 0x0));
    // ESP0 = 20h: a 32-bit V86 frame (36 bytes) overflows the stack limit,
    // a 16-bit one (20 bytes) fits.
    mem.load(0x3000, &tss(0x20, 0x38));
    mem.load(0x0800 + 0x50 * 8, &gate(0x2C00, 0x08, 0xEE)); // INT 50h: 32-bit gate
    mem.load(0x0800 + 12 * 8, &gate(0x2D00, 0x08, 0x86)); // #SS: 16-bit gate
    mem.load(0x0800 + 8 * 8, &gate(0x2F00, 0x08, 0x8E)); // #DF
    attach_idt_tss(&mut cpu, 0x0800, 0x3000, 0x30);

    // Enter V86 via IRETD: EIP CS EFLAGS ESP SS ES DS FS GS.
    flat_ss(&mut cpu, 0x6000);
    let frame: [u32; 9] = [
        0x0000,                                      // EIP
        0x0800,                                      // CS
        EFlags::VM.bits() | EFlags::IOPL.bits() | 2, // EFLAGS (IOPL 3)
        0x0100,                                      // ESP
        0x0900,                                      // SS
        0x0A00,                                      // ES
        0x0B00,                                      // DS
        0x0C00,                                      // FS
        0x0D00,                                      // GS
    ];
    for (i, v) in frame.iter().enumerate() {
        mem.load(0x6000 - 36 + i as u32 * 4, &v.to_le_bytes());
    }
    cpu.regs.gpr[reg::ESP as usize] = 0x6000 - 36;
    mem.load(0x2000, &[0xCF]); // IRETD
    mem.load(0x8000, &[0xCD, 0x50]); // V86 code at 0800:0000 -> INT 50h

    cpu.step(&mut mem);
    assert!(
        cpu.regs.eflags.contains(EFlags::VM),
        "should be in V86 mode"
    );
    assert_eq!(cpu.cpl(), 3);

    cpu.step(&mut mem); // INT 50h -> #SS mid-frame -> nested #SS delivery
    assert!(!cpu.shutdown);
    assert_eq!(cpu.regs.eip, 0x2D00, "expected the #SS handler");
    assert!(!cpu.regs.eflags.contains(EFlags::VM));

    // A V86 frame nulls the data segments and stacks the four V86 selectors.
    for idx in [reg::ES, reg::DS, reg::FS, reg::GS] {
        assert_eq!(
            cpu.regs.seg[idx as usize].sel, 0,
            "V86 data segments must be nulled by a V86 delivery"
        );
    }
    let word = |off: usize| mem.ram[off] as u16 | (mem.ram[off + 1] as u16) << 8;
    assert_eq!(word(0x7000 + 0x1E), 0x0D00, "GS");
    assert_eq!(word(0x7000 + 0x1C), 0x0C00, "FS");
    assert_eq!(word(0x7000 + 0x1A), 0x0B00, "DS");
    assert_eq!(word(0x7000 + 0x18), 0x0A00, "ES");
}

#[test]
fn iret_does_not_commit_eflags_when_the_outer_stack_faults() {
    // IRET to an outer ring with a bad SS faults; the popped EFLAGS image
    // must not survive into the fault handler.
    let (mut cpu, mut mem) = setup_pm();
    mem.load(0x0520, &descriptor(0, 0xF_FFFF, 0xFA, 0xC)); // 20: ring-3 code
    mem.load(0x0800 + 13 * 8, &gate(0x2D00, 0x08, 0x8E)); // #GP handler
    attach_idt_tss(&mut cpu, 0x0800, 0x3000, 0x30);

    // Frame: EIP, CS=23h, EFLAGS with DF set, ESP, SS=0FFFCh (bad RPL).
    flat_ss(&mut cpu, 0x7000);
    let frame: [u32; 5] = [0x4000, 0x23, EFlags::DF.bits() | 2, 0x6000, 0xFFFC];
    for (i, v) in frame.iter().enumerate() {
        mem.load(0x7000 + i as u32 * 4, &v.to_le_bytes());
    }
    mem.load(0x2000, &[0xCF]); // IRETD

    assert!(!cpu.regs.eflags.contains(EFlags::DF));
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.eip, 0x2D00, "expected #GP on the bad outer SS");
    assert!(
        !cpu.regs.eflags.contains(EFlags::DF),
        "the returned-to EFLAGS image must not be committed before the SS load"
    );
}

#[test]
fn iretd_at_cpl3_ignores_a_set_vm_bit() {
    // VM is not writable at CPL > 0, so IRETD simply ignores it (no #GP).
    let (mut cpu, mut mem) = setup_pm();
    mem.load(0x0520, &descriptor(0, 0xF_FFFF, 0xFA, 0xC)); // 20: ring-3 code
    mem.load(0x0528, &descriptor(0, 0xF_FFFF, 0xF2, 0xC)); // 28: ring-3 data

    // Drop straight to ring 3 by loading the caches directly.
    cpu.regs.seg[reg::CS as usize] = SegReg {
        sel: 0x23,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0CFA,
    };
    cpu.regs.seg[reg::SS as usize] = SegReg {
        sel: 0x2B,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0CF2,
    };
    assert_eq!(cpu.cpl(), 3);

    cpu.regs.eip = 0x4000;
    cpu.regs.gpr[reg::ESP as usize] = 0x6000;
    let frame: [u32; 3] = [0x4100, 0x23, EFlags::VM.bits() | 2];
    for (i, v) in frame.iter().enumerate() {
        mem.load(0x6000 + i as u32 * 4, &v.to_le_bytes());
    }
    mem.load(0x4000, &[0xCF]); // IRETD

    cpu.step(&mut mem);
    assert!(!cpu.shutdown);
    assert_eq!(cpu.regs.eip, 0x4100, "the return must not raise #GP");
    assert!(!cpu.regs.eflags.contains(EFlags::VM));
    assert_eq!(cpu.cpl(), 3);
}

#[test]
fn ldt_limit_is_not_truncated_to_16_bits() {
    // A page-granular LDT has a byte limit above 0xFFFF; selectors beyond
    // index 0x1000 must still resolve.
    let (mut cpu, mut mem) = setup_pm();
    cpu.regs.ldtr = SegReg {
        sel: 0x40,
        base: 0x2_0000,
        limit: 0x1_0FFF, // granular: 0x10 pages
        attrs: 0x0082,
    };
    // A data descriptor at LDT offset 0x1000 -> selector (0x200 << 3) | 4.
    mem.load(0x2_1000, &descriptor(0x5000, 0xFFFF, 0x92, 0x0));
    mem.load(0x2000, &[0x66, 0xB8, 0x04, 0x10, 0x8E, 0xD8]); // MOV AX,1004h; MOV DS,AX

    cpu.step(&mut mem);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.seg[reg::DS as usize].sel, 0x1004);
    assert_eq!(cpu.regs.seg[reg::DS as usize].base, 0x5000);
}

#[test]
fn tss_sourced_selector_out_of_range_raises_ts() {
    // The inner SS selector comes from the TSS: an out-of-range one is #TS,
    // not #GP.
    let (mut cpu, mut mem) = setup_pm();
    mem.load(0x0520, &descriptor(0, 0xF_FFFF, 0xFA, 0xC)); // 20: ring-3 code
    mem.load(0x0528, &descriptor(0, 0xF_FFFF, 0xF2, 0xC)); // 28: ring-3 data
    mem.load(0x0530, &descriptor(0x3000, 0x67, 0x89, 0x0)); // 30: TSS
    mem.load(0x3000, &tss(0x7000, 0x00F8)); // SS0 index 31: past the GDT limit

    // The #TS and #GP handlers are ring-3 code, so their delivery needs no
    // stack switch and the cascade stops there.
    mem.load(0x0800 + 0x40 * 8, &gate(0x2C00, 0x08, 0xEE)); // INT 40h -> ring 0
    mem.load(0x0800 + 10 * 8, &gate(0x4100, 0x23, 0x8E)); // #TS -> ring 3
    mem.load(0x0800 + 13 * 8, &gate(0x4200, 0x23, 0x8E)); // #GP -> ring 3
    attach_idt_tss(&mut cpu, 0x0800, 0x3000, 0x30);

    cpu.regs.seg[reg::CS as usize] = SegReg {
        sel: 0x23,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0CFA,
    };
    cpu.regs.seg[reg::SS as usize] = SegReg {
        sel: 0x2B,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0CF2,
    };
    cpu.regs.eip = 0x4000;
    cpu.regs.gpr[reg::ESP as usize] = 0x6000;
    mem.load(0x4000, &[0xCD, 0x40]); // INT 40h

    cpu.step(&mut mem);
    assert!(!cpu.shutdown);
    assert_eq!(cpu.regs.eip, 0x4100, "expected #TS, not #GP");
    let sp = cpu.regs.gpr[reg::ESP as usize] as usize;
    let err = u32::from_le_bytes(mem.ram[sp..sp + 4].try_into().unwrap());
    assert_eq!(err, 0xF8, "#TS carries the offending SS selector");
}

#[test]
fn tss_stack_read_respects_the_limit_exactly() {
    // ESP0/SS0 for level 0 occupy TSS bytes 4..9, so a limit of 8 is one
    // byte short and must raise #TS rather than read past it.
    let (mut cpu, mut mem) = setup_pm();
    mem.load(0x0520, &descriptor(0, 0xF_FFFF, 0xFA, 0xC));
    mem.load(0x0528, &descriptor(0, 0xF_FFFF, 0xF2, 0xC));
    mem.load(0x3000, &tss(0x7000, 0x10)); // a perfectly valid ring-0 stack
    mem.load(0x0800 + 0x40 * 8, &gate(0x2C00, 0x08, 0xEE));
    mem.load(0x0800 + 10 * 8, &gate(0x4100, 0x23, 0x8E)); // #TS -> ring 3
    attach_idt_tss(&mut cpu, 0x0800, 0x3000, 0x30);
    cpu.regs.tr.limit = 8; // one byte short of SS0's last byte

    cpu.regs.seg[reg::CS as usize] = SegReg {
        sel: 0x23,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0CFA,
    };
    cpu.regs.seg[reg::SS as usize] = SegReg {
        sel: 0x2B,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0CF2,
    };
    cpu.regs.eip = 0x4000;
    cpu.regs.gpr[reg::ESP as usize] = 0x6000;
    mem.load(0x4000, &[0xCD, 0x40]);
    mem.load(0x2C00, &[0x90]);

    cpu.step(&mut mem);
    assert_eq!(
        cpu.regs.eip, 0x4100,
        "the truncated TSS must raise #TS, not deliver the interrupt"
    );
}

#[test]
fn rep_string_is_interruptible() {
    // A 32-bit REP can run for billions of iterations; hardware recognizes
    // interrupts between them, resuming at the prefix afterwards.
    // STI's one-instruction shadow defers delivery past the REP's own
    // boundary, so the pending request is first seen *inside* the loop.
    let (mut cpu, mut mem) = setup(&[0xFB, 0x67, 0xF3, 0x66, 0xAB]); // STI; a32 REP STOSD
    mem.load(0x20 * 4, &[0x00, 0x80, 0x00, 0x00]); // IVT 20h -> 0000:8000
    cpu.regs.gpr[0] = 0xDEAD_BEEF;
    cpu.regs.gpr[reg::ECX as usize] = 1000;
    cpu.regs.gpr[reg::EDI as usize] = 0x4000;
    cpu.regs.seg[reg::ES as usize] = SegReg::real(0);
    cpu.assert_intr(0x20);

    cpu.step(&mut mem); // STI
    cpu.step(&mut mem); // REP STOSD: one iteration, then yields
    assert_eq!(
        cpu.regs.eip, 0x1101,
        "EIP rewinds to the prefix so the REP resumes after the handler"
    );
    assert_eq!(cpu.regs.gpr[reg::ECX as usize], 999, "one iteration ran");
    assert_eq!(cpu.regs.gpr[reg::EDI as usize], 0x4004);

    cpu.step(&mut mem); // the pending interrupt is taken
    assert_eq!(cpu.regs.eip, 0x8000);
}
