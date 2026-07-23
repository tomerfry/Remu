//! Focused unit and integration tests for the ARM32 core: the data
//! processing set with the barrel shifter and its carry rules, the R15
//! pipeline offsets (+8/+12), multiplies, PSR transfers and mode banking,
//! single/halfword/block transfers with the ARM7TDMI alignment and
//! base-in-list quirks, SWP, exceptions (SWI/undefined/IRQ/FIQ) with mode
//! returns, the Thumb instruction set including the BL pair, host traps,
//! and the batched [`Cpu::run`] entry point.

use remu::arm32::{Cpu, Exception, HostTrap, LinearMemory, Mode, RunExit, psr};

/// Code origin: clear of the exception vectors at 0x00–0x1C.
const ORG: u32 = 0x100;

/// A CPU + flat memory with ARM `program` at [`ORG`], PC there, and a sane
/// stack.
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

/// A CPU + flat memory with Thumb `program` at [`ORG`], already in Thumb
/// state.
fn thumb(program: &[u16]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    for (i, h) in program.iter().enumerate() {
        mem.load(ORG + i as u32 * 2, &h.to_le_bytes());
    }
    let mut cpu = Cpu::new();
    cpu.regs.cpsr.set(psr::T, true);
    cpu.regs.gpr[15] = ORG;
    cpu.regs.gpr[13] = 0x8000;
    (cpu, mem)
}

/// Step `n` instructions.
fn steps(cpu: &mut Cpu, mem: &mut LinearMemory, n: u32) {
    for _ in 0..n {
        cpu.step(mem);
    }
}

/// Shorthand for the four condition flags as a tuple.
fn nzcv(cpu: &Cpu) -> (bool, bool, bool, bool) {
    let p = cpu.regs.cpsr;
    (p.n(), p.z(), p.c(), p.v())
}

// --- Data processing ---------------------------------------------------------

#[test]
fn dp_basic_ops() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0007, // MOV r0, #7
        0xE3A0_1003, // MOV r1, #3
        0xE080_2001, // ADD r2, r0, r1
        0xE050_3001, // SUBS r3, r0, r1
        0xE020_4001, // EOR r4, r0, r1
        0xE180_5001, // ORR r5, r0, r1
        0xE000_6001, // AND r6, r0, r1
        0xE1C0_7001, // BIC r7, r0, r1
        0xE1E0_8001, // MVN r8, r1
        0xE260_9001, // RSB r9, r0, #1
    ]);
    steps(&mut cpu, &mut mem, 10);
    assert_eq!(cpu.regs.gpr[2], 10);
    assert_eq!(cpu.regs.gpr[3], 4);
    assert_eq!(cpu.regs.gpr[4], 4);
    assert_eq!(cpu.regs.gpr[5], 7);
    assert_eq!(cpu.regs.gpr[6], 3);
    assert_eq!(cpu.regs.gpr[7], 4);
    assert_eq!(cpu.regs.gpr[8], !3u32);
    assert_eq!(cpu.regs.gpr[9], 1u32.wrapping_sub(7));
}

#[test]
fn dp_flags_add_sub() {
    // ADDS overflow: 0x7FFFFFFF + 1; SUBS borrow: 0 - 1; carry: FFFFFFFF+1.
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0102, // MOV r0, #0x80000000
        0xE250_1001, // SUBS r1, r0, #1     ; 0x7FFFFFFF: N0 Z0 C1 V1
        0xE091_2000, // ADDS r2, r1, r0     ; 0xFFFFFFFF: N1 Z0 C0 V0
        0xE092_3001, // ADDS r3, r2, r1     ; FFFFFFFF+7FFFFFFF: C1
        0xE3A0_4000, // MOV r4, #0
        0xE254_5001, // SUBS r5, r4, #1     ; borrow: C0
    ]);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[1], 0x7FFF_FFFF);
    assert_eq!(nzcv(&cpu), (false, false, true, true));
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[2], 0xFFFF_FFFF);
    assert_eq!(nzcv(&cpu), (true, false, false, false));
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[3], 0x7FFF_FFFE);
    assert_eq!(nzcv(&cpu), (false, false, true, false));
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[5], 0xFFFF_FFFF);
    assert_eq!(nzcv(&cpu), (true, false, false, false));
}

#[test]
fn dp_adc_sbc_rsc_use_carry() {
    let (mut cpu, mut mem) = arm(&[
        0xE3B0_0000, // MOVS r0, #0        ; clears N/Z sets Z, C unchanged(0)
        0xE2A0_1005, // ADC r1, r0, #5     ; C=0: 5
        0xE3A0_2003, // MOV r2, #3
        0xE0D2_3000, // SBCS r3, r2, r0    ; 3 - 0 - !C(1) = 2, C=1
        0xE0A1_4001, // ADC r4, r1, r1     ; 5+5+C(1) = 11
        0xE2E2_5009, // RSC r5, r2, #9     ; 9 - 3 - !C(0) = 6
    ]);
    steps(&mut cpu, &mut mem, 6);
    assert_eq!(cpu.regs.gpr[1], 5);
    assert_eq!(cpu.regs.gpr[3], 2);
    assert!(cpu.regs.cpsr.c());
    assert_eq!(cpu.regs.gpr[4], 11);
    assert_eq!(cpu.regs.gpr[5], 6);
}

#[test]
fn dp_logical_s_takes_shifter_carry() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0001, // MOV r0, #1
        0xE1B0_1FC0, // MOVS r1, r0, ASR #31 ; 1>>31 = 0: Z=1, C = bit30... = 0
        0xE3A0_0102, // MOV r0, #0x80000000
        0xE1B0_1FC0, // MOVS r1, r0, ASR #31 ; -1, C = 1 (bit 30 chain), N=1
        0xE1B0_2020, // MOVS r2, r0, LSR #32 ; (encoded LSR #0): 0, C = bit31 = 1
        0xE3B0_3102, // MOVS r3, #0x80000000 ; rotated imm: C = bit31 of imm
    ]);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[1], 0);
    assert!(!nzcv(&cpu).2);
    assert!(cpu.regs.cpsr.z());
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[1], 0xFFFF_FFFF);
    // ASR #31 carries out bit 30 (= 0 for 0x80000000).
    assert!(cpu.regs.cpsr.n() && !cpu.regs.cpsr.c());
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[2], 0);
    assert!(cpu.regs.cpsr.z() && cpu.regs.cpsr.c());
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[3], 0x8000_0000);
    assert!(cpu.regs.cpsr.c(), "rotated immediate sets shifter carry");
}

#[test]
fn dp_rrx_rotates_through_carry() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0003, // MOV r0, #3
        0xE1B0_1060, // MOVS r1, r0, RRX ; C in (0): 1, C out = 1
        0xE1B0_2060, // MOVS r2, r0, RRX ; C in (1): 0x80000001, C out = 1
    ]);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[1], 1);
    assert!(cpu.regs.cpsr.c());
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[2], 0x8000_0001);
}

#[test]
fn dp_shift_by_register_uses_low_byte() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0001, // MOV r0, #1
        0xE3A0_1004, // MOV r1, #4
        0xE1A0_2110, // MOV r2, r0, LSL r1   ; 0x10
        0xE3A0_1C01, // MOV r1, #0x100       ; low byte 0 → unchanged
        0xE1B0_3110, // MOVS r3, r0, LSL r1  ; 1, carry unchanged
        0xE3A0_1021, // MOV r1, #33
        0xE1B0_4110, // MOVS r4, r0, LSL r1  ; shift 33 → 0, C=0
    ]);
    steps(&mut cpu, &mut mem, 7);
    assert_eq!(cpu.regs.gpr[2], 0x10);
    assert_eq!(cpu.regs.gpr[3], 1);
    assert_eq!(cpu.regs.gpr[4], 0);
    assert!(!cpu.regs.cpsr.c());
}

#[test]
fn r15_reads_plus_8_and_plus_12() {
    let (mut cpu, mut mem) = arm(&[
        0xE1A0_000F, // MOV r0, pc            ; ORG + 8
        0xE3A0_2000, // MOV r2, #0
        0xE1A0_121F, // MOV r1, pc, LSL r2    ; reg-shift: ORG+8 + 12
    ]);
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[0], ORG + 8);
    assert_eq!(cpu.regs.gpr[1], ORG + 8 + 12);
}

#[test]
fn dp_pc_write_branches() {
    let (mut cpu, mut mem) = arm(&[
        0xE28F_F004, // ADD pc, pc, #4  ; lands at ORG+8+4 = ORG+12
        0xE3A0_0001, // MOV r0, #1      (skipped)
        0xE3A0_0002, // MOV r0, #2      (skipped)
        0xE3A0_0003, // MOV r0, #3      (executed)
    ]);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[0], 3);
}

#[test]
fn condition_codes_skip_and_pass() {
    let (mut cpu, mut mem) = arm(&[
        0xE3B0_0000, // MOVS r0, #0     ; Z=1
        0x03A0_1001, // MOVEQ r1, #1    ; executes
        0x13A0_2001, // MOVNE r2, #1    ; skipped
        0x23A0_3001, // MOVCS r3, #1    ; skipped (C=0)
        0x33A0_4001, // MOVCC r4, #1    ; executes
        0xF3A0_5001, // (NV) MOV r5, #1 ; never executes on this core
    ]);
    steps(&mut cpu, &mut mem, 6);
    assert_eq!(cpu.regs.gpr[1], 1);
    assert_eq!(cpu.regs.gpr[2], 0);
    assert_eq!(cpu.regs.gpr[3], 0);
    assert_eq!(cpu.regs.gpr[4], 1);
    assert_eq!(cpu.regs.gpr[5], 0);
}

// --- Multiplies ---------------------------------------------------------------

#[test]
fn multiplies() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1007, // MOV r1, #7
        0xE3A0_2006, // MOV r2, #6
        0xE3A0_3005, // MOV r3, #5
        0xE000_0291, // MUL r0, r1, r2            ; 42
        0xE020_3291, // MLA r0, r1, r2, r3        ; 47
        0xE3E0_1000, // MVN r1, #0                ; 0xFFFFFFFF
        0xE081_0392, // UMULL r0, r1, r2, r3      ; wait: operands below
    ]);
    steps(&mut cpu, &mut mem, 4);
    assert_eq!(cpu.regs.gpr[0], 42);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[0], 47);
    // UMULL r0(lo), r1(hi), r2, r3: r2=6, r3=5 → 30.
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[0], 30);
    assert_eq!(cpu.regs.gpr[1], 0);
}

#[test]
fn long_multiplies_signed_unsigned_accumulate() {
    let (mut cpu, mut mem) = arm(&[
        0xE3E0_2000, // MVN r2, #0            ; -1 / 0xFFFFFFFF
        0xE3A0_3002, // MOV r3, #2
        0xE0C1_0392, // SMULL r0, r1, r2, r3  ; -2 → lo FFFFFFFE hi FFFFFFFF
        0xE081_4392, // UMULL r4, r1, r2, r3  ; 0x1_FFFF_FFFE
        0xE3A0_0001, // MOV r0, #1
        0xE3A0_1000, // MOV r1, #0
        0xE0E1_0392, // SMLAL r0, r1, r2, r3  ; acc(1) + -2 = -1
    ]);
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[0], 0xFFFF_FFFE);
    assert_eq!(cpu.regs.gpr[1], 0xFFFF_FFFF);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[4], 0xFFFF_FFFE);
    assert_eq!(cpu.regs.gpr[1], 1);
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[0], 0xFFFF_FFFF);
    assert_eq!(cpu.regs.gpr[1], 0xFFFF_FFFF);
}

#[test]
fn muls_sets_nz_only() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1000, // MOV r1, #0
        0xE3A0_2005, // MOV r2, #5
        0xE010_0291, // MULS r0, r1, r2  ; 0 → Z
    ]);
    cpu.regs.cpsr.set(psr::C, true);
    cpu.regs.cpsr.set(psr::V, true);
    steps(&mut cpu, &mut mem, 3);
    assert!(cpu.regs.cpsr.z());
    assert!(
        cpu.regs.cpsr.c(),
        "C untouched by MULS (deterministic pick)"
    );
    assert!(cpu.regs.cpsr.v(), "V untouched by MULS");
}

// --- PSR transfers and modes ----------------------------------------------------

#[test]
fn mrs_reads_cpsr() {
    let (mut cpu, mut mem) = arm(&[0xE10F_0000]); // MRS r0, CPSR
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[0], cpu.regs.cpsr.bits());
    assert_eq!(cpu.regs.gpr[0] & 0x1F, Mode::Svc as u32);
}

#[test]
fn msr_flags_and_mode_switch() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0102, // MOV r0, #0x80000000
        0xE128_F000, // MSR CPSR_f, r0       ; N set
        0xE3A0_00D2, // MOV r0, #0xD2        ; IRQ mode, I+F
        0xE121_F000, // MSR CPSR_c, r0       ; switch SVC → IRQ
    ]);
    let svc_sp = 0x8000;
    steps(&mut cpu, &mut mem, 2);
    assert!(cpu.regs.cpsr.n());
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Svc);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Irq);
    assert_ne!(cpu.regs.gpr[13], svc_sp, "banked SP swapped in");
    assert_eq!(cpu.regs.reg_of(Mode::Svc, 13), svc_sp);
}

#[test]
fn msr_user_mode_cannot_touch_control() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0010, // MOV r0, #0x10        ; user mode bits
        0xE121_F000, // MSR CPSR_c, r0       ; drop to user
        0xE3A0_00D3, // MOV r0, #0xD3        ; try SVC + I + F
        0xE121_F000, // MSR CPSR_c, r0       ; must be ignored in user mode
        0xE3A0_0201, // MOV r0, #0x10000000  ; V flag
        0xE128_F000, // MSR CPSR_f, r0       ; flags still writable
    ]);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Usr);
    assert!(
        !cpu.regs.cpsr.i(),
        "entering user mode via MSR clears I? no —"
    );
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(
        cpu.regs.cpsr.mode(),
        Mode::Usr,
        "user mode cannot switch modes"
    );
    steps(&mut cpu, &mut mem, 2);
    assert!(cpu.regs.cpsr.v(), "user mode may write flags");
}

#[test]
fn spsr_read_write_in_exception_mode() {
    let (mut cpu, mut mem) = arm(&[
        0xE14F_0000, // MRS r0, SPSR
        0xE3A0_1201, // MOV r1, #0x10000000
        0xE169_F001, // MSR SPSR_fc, r1
        0xE14F_2000, // MRS r2, SPSR
    ]);
    // Fresh CPU is in SVC; its SPSR is architecturally accessible.
    steps(&mut cpu, &mut mem, 4);
    assert_eq!(cpu.regs.gpr[2] & 0xF000_0000, 0x1000_0000);
}

#[test]
fn movs_pc_returns_mode_and_state() {
    // Drop to user via a privileged CPSR write, do SWI from user, observe
    // the SVC entry, and return with MOVS pc, lr (CPSR = SPSR).
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0010, // 100: MOV r0, #0x10
        0xE121_F000, // 104: MSR CPSR_c, r0      ; → user mode
        0xEF00_002A, // 108: SWI 0x2A            ; → SVC, LR = 0x10C
        0xE3A0_5005, // 10C: MOV r5, #5          ; runs after return, in user
    ]);
    // SVC handler at the SWI vector: set a marker, then MOVS pc, lr.
    mem.load(0x08, &0xE3A0_4001u32.to_le_bytes()); // MOV r4, #1
    mem.load(0x0C, &0xE1B0_F00Eu32.to_le_bytes()); // MOVS pc, lr
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Svc);
    assert!(cpu.regs.cpsr.i(), "SWI entry disables IRQ");
    assert_eq!(cpu.regs.gpr[14], ORG + 0xC);
    assert_eq!(cpu.regs.spsr().mode(), Mode::Usr);
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[4], 1);
    assert_eq!(cpu.regs.gpr[5], 5);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Usr, "MOVS pc, lr restored mode");
    assert!(!cpu.regs.cpsr.i(), "restored CPSR re-enables IRQ");
}

// --- Branches -------------------------------------------------------------------

#[test]
fn b_and_bl() {
    let (mut cpu, mut mem) = arm(&[
        0xEA00_0001, // 100: B 0x10C
        0xE3A0_0001, // 104: MOV r0, #1 (skipped)
        0xE3A0_0002, // 108: MOV r0, #2 (skipped)
        0xEB00_0000, // 10C: BL 0x114
        0xE3A0_1001, // 110: MOV r1, #1 (returned to)
        0xE3A0_2001, // 114: MOV r2, #1
        0xE1A0_F00E, // 118: MOV pc, lr
    ]);
    steps(&mut cpu, &mut mem, 3); // B, BL, MOV r2
    assert_eq!(cpu.regs.gpr[2], 1);
    assert_eq!(cpu.regs.gpr[14], ORG + 0x10);
    steps(&mut cpu, &mut mem, 2); // MOV pc, lr; MOV r1
    assert_eq!(cpu.regs.gpr[1], 1);
    assert_eq!(cpu.regs.gpr[0], 0);
}

#[test]
fn bx_interworks_to_thumb_and_back() {
    let (mut cpu, mut mem) = arm(&[
        0xE28F_0009, // 100: ADD r0, pc, #9   ; 0x111 = Thumb code | 1
        0xE12F_FF10, // 104: BX r0
        0xE3A0_5005, // 108: MOV r5, #5       ; executed after return
        0xEAFF_FFFE, // 10C: B .              (never)
    ]);
    // Thumb at 0x110: MOV r1, #7; BX lr — but LR is stale; load return via
    // r2. Use: MOV r1,#7 (2107); LDR r2,[pc,#0] → pool; BX r2 (4710).
    mem.load(0x110, &0x2107u16.to_le_bytes()); // MOVS r1, #7
    mem.load(0x112, &0x4A00u16.to_le_bytes()); // LDR r2, [pc, #0] ; (0x114+0)&~2 → 0x114
    mem.load(0x114, &0x4710u16.to_le_bytes()); // BX r2  — overlaps pool! move it
    // Rebuild the Thumb block cleanly: the literal must sit at a word
    // boundary after the code.
    // 110: 2107      MOVS r1, #7
    // 112: 4A01      LDR r2, [pc, #4]   ; base (0x112+4)&~2 = 0x114, +4 = 0x118
    // 114: 4710      BX r2
    // 116: 46C0      NOP (MOV r8, r8)
    // 118: .word 0x108
    mem.load(0x110, &0x2107u16.to_le_bytes());
    mem.load(0x112, &0x4A01u16.to_le_bytes());
    mem.load(0x114, &0x4710u16.to_le_bytes());
    mem.load(0x116, &0x46C0u16.to_le_bytes());
    mem.load(0x118, &0x108u32.to_le_bytes());
    steps(&mut cpu, &mut mem, 2);
    assert!(cpu.regs.cpsr.t(), "BX with bit 0 set enters Thumb");
    assert_eq!(cpu.regs.gpr[15], 0x110, "BX consumes bit 0");
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[1], 7);
    assert!(!cpu.regs.cpsr.t(), "BX with bit 0 clear returns to ARM");
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[5], 5);
}

// --- Single data transfers -------------------------------------------------------

#[test]
fn ldr_str_offsets_and_writeback() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE3A0_0042, // MOV r0, #0x42
        0xE581_0004, // STR r0, [r1, #4]
        0xE591_2004, // LDR r2, [r1, #4]
        0xE5A1_0008, // STR r0, [r1, #8]!    ; r1 = 0x408
        0xE491_3004, // LDR r3, [r1], #4     ; loads [0x408], r1 = 0x40C
        0xE511_4008, // LDR r4, [r1, #-8]    ; [0x404]
    ]);
    steps(&mut cpu, &mut mem, 7);
    assert_eq!(cpu.regs.gpr[2], 0x42);
    assert_eq!(cpu.regs.gpr[3], 0x42);
    assert_eq!(cpu.regs.gpr[1], 0x40C);
    assert_eq!(cpu.regs.gpr[4], 0x42);
}

#[test]
fn ldr_reg_offset_scaled() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE3A0_2004, // MOV r2, #4
        0xE3A0_0055, // MOV r0, #0x55
        0xE781_0102, // STR r0, [r1, r2, LSL #2] ; [0x410]
        0xE791_3102, // LDR r3, [r1, r2, LSL #2]
        0xE711_4002, // LDR r4, [r1, -r2]        ; [0x3FC] (zero)
    ]);
    steps(&mut cpu, &mut mem, 6);
    assert_eq!(cpu.regs.gpr[3], 0x55);
    assert_eq!(cpu.regs.gpr[4], 0);
    assert_eq!(mem.ram[0x410], 0x55);
}

#[test]
fn ldrb_strb() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE3A0_0EFF, // MOV r0, #0xFF0
        0xE5C1_0000, // STRB r0, [r1]        ; stores 0xF0
        0xE5D1_2000, // LDRB r2, [r1]
    ]);
    steps(&mut cpu, &mut mem, 4);
    assert_eq!(cpu.regs.gpr[2], 0xF0);
}

#[test]
fn ldr_unaligned_rotates() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE591_0001, // LDR r0, [r1, #1]
        0xE591_2002, // LDR r2, [r1, #2]
    ]);
    mem.load(0x400, &0x1122_3344u32.to_le_bytes());
    steps(&mut cpu, &mut mem, 3);
    // Aligned word 0x11223344 rotated so byte at +1 (0x33) lands low:
    assert_eq!(cpu.regs.gpr[0], 0x4411_2233);
    assert_eq!(cpu.regs.gpr[2], 0x3344_1122);
}

#[test]
fn str_of_pc_stores_plus_12() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE581_F000, // STR pc, [r1]     ; stores (ORG+4) + 12
    ]);
    steps(&mut cpu, &mut mem, 2);
    let stored = u32::from_le_bytes(mem.ram[0x400..0x404].try_into().unwrap());
    assert_eq!(stored, ORG + 4 + 12);
}

#[test]
fn ldr_to_pc_branches() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE591_F000, // LDR pc, [r1]
        0xE3A0_0001, // MOV r0, #1 (skipped)
    ]);
    mem.load(0x400, &(ORG + 0x40).to_le_bytes());
    mem.load(ORG + 0x40, &0xE3A0_0007u32.to_le_bytes()); // MOV r0, #7
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[0], 7);
}

// --- Halfword and signed transfers -------------------------------------------------

#[test]
fn halfword_and_signed_loads() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE1D1_00B0, // LDRH r0, [r1]
        0xE1D1_20B2, // LDRH r2, [r1, #2]
        0xE1D1_30D1, // LDRSB r3, [r1, #1]
        0xE1D1_40F0, // LDRSH r4, [r1]
        0xE1D1_50F2, // LDRSH r5, [r1, #2]
    ]);
    mem.load(0x400, &0x8001_FF80u32.to_le_bytes()); // bytes 80 FF 01 80
    steps(&mut cpu, &mut mem, 6);
    assert_eq!(cpu.regs.gpr[0], 0xFF80);
    assert_eq!(cpu.regs.gpr[2], 0x8001);
    assert_eq!(cpu.regs.gpr[3], 0xFFFF_FFFF); // sext(0xFF)
    assert_eq!(cpu.regs.gpr[4], 0xFFFF_FF80); // sext(0xFF80)
    assert_eq!(cpu.regs.gpr[5], 0xFFFF_8001); // sext(0x8001)
}

#[test]
fn strh_and_odd_address_quirks() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE3A0_0C23, // MOV r0, #0x2300
        0xE380_0045, // ORR r0, r0, #0x45   ; 0x2345
        0xE1C1_00B4, // STRH r0, [r1, #4]
        0xE1D1_20B1, // LDRH r2, [r1, #1]   ; odd: halfword ROR 8
        0xE1D1_30F1, // LDRSH r3, [r1, #1]  ; odd: degrades to LDRSB
    ]);
    mem.load(0x400, &0x0000_A55Au32.to_le_bytes()); // bytes 5A A5 00 00
    steps(&mut cpu, &mut mem, 4);
    assert_eq!(
        u16::from_le_bytes(mem.ram[0x404..0x406].try_into().unwrap()),
        0x2345
    );
    steps(&mut cpu, &mut mem, 1);
    // Halfword at 0x400 = 0xA55A, rotated by 8: byte A5 low, 5A at 31:24.
    assert_eq!(cpu.regs.gpr[2], 0x5A00_00A5);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[3], 0xFFFF_FFA5, "odd LDRSH acts as LDRSB");
}

// --- Block transfers --------------------------------------------------------------

#[test]
fn ldm_stm_addressing_modes() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0B01, // MOV r0, #0x400
        0xE3A0_1001, // MOV r1, #1
        0xE3A0_2002, // MOV r2, #2
        0xE8A0_0006, // STMIA r0!, {r1, r2}   ; [400]=1 [404]=2, r0=408
        0xE920_0006, // STMDB r0!, {r1, r2}   ; back to 400.. overwrites; r0=400
        0xE8B0_0018, // LDMIA r0!, {r3, r4}   ; r3=1 r4=2, r0=408
        0xE910_0060, // LDMDB r0, {r5, r6}    ; [400],[404] → r5=1 r6=2
    ]);
    steps(&mut cpu, &mut mem, 7);
    assert_eq!(cpu.regs.gpr[3], 1);
    assert_eq!(cpu.regs.gpr[4], 2);
    assert_eq!(cpu.regs.gpr[0], 0x408);
    assert_eq!(cpu.regs.gpr[5], 1);
    assert_eq!(cpu.regs.gpr[6], 2);
}

#[test]
fn stm_base_in_list_first_stores_original() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0B01, // MOV r0, #0x400
        0xE8A0_0003, // STMIA r0!, {r0, r1}  ; r0 first: stores original base
    ]);
    cpu.regs.gpr[1] = 0x77;
    steps(&mut cpu, &mut mem, 2);
    let w0 = u32::from_le_bytes(mem.ram[0x400..0x404].try_into().unwrap());
    assert_eq!(
        w0, 0x400,
        "base first in list stores the pre-writeback value"
    );
    assert_eq!(cpu.regs.gpr[0], 0x408);
}

#[test]
fn stm_base_in_list_not_first_stores_written_back() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_1B01, // MOV r1, #0x400
        0xE8A1_0006, // STMIA r1!, {r1, r2}  ; r1 not lowest? r1 < r2 → first!
    ]);
    // Make the base *not* the lowest: use {r0, r1}.
    mem.load(ORG + 4, &0xE8A1_0003u32.to_le_bytes()); // STMIA r1!, {r0, r1}
    cpu.regs.gpr[0] = 0x11;
    steps(&mut cpu, &mut mem, 2);
    let w1 = u32::from_le_bytes(mem.ram[0x404..0x408].try_into().unwrap());
    assert_eq!(
        w1, 0x408,
        "base later in list stores the written-back value"
    );
}

#[test]
fn ldm_base_in_list_loaded_value_wins() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0B01, // MOV r0, #0x400
        0xE8B0_0009, // LDMIA r0!, {r0, r3}  ; loaded r0 wins over writeback
    ]);
    mem.load(0x400, &0xDEAD_0000u32.to_le_bytes());
    mem.load(0x404, &0x0000_0033u32.to_le_bytes());
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[0], 0xDEAD_0000);
    assert_eq!(cpu.regs.gpr[3], 0x33);
}

#[test]
fn stm_user_bank_from_exception_mode() {
    // In FIQ mode, STM with S stores the *user* r8–r14.
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0B01, // MOV r0, #0x400
        0xE8C0_4100, // STMIA r0, {r8, lr}^   ; user bank
    ]);
    cpu.regs.gpr[8] = 0x1111; // SVC gpr8 == user gpr8 (shared)
    cpu.regs.set_user_reg(14, 0x2222);
    cpu.regs.set_mode(Mode::Fiq);
    cpu.regs.gpr[8] = 0xFFFF; // FIQ-banked r8
    cpu.regs.gpr[14] = 0xEEEE; // FIQ-banked lr
    cpu.regs.gpr[15] = ORG;
    cpu.regs.gpr[0] = 0; // will be set by MOV
    steps(&mut cpu, &mut mem, 2);
    let w0 = u32::from_le_bytes(mem.ram[0x400..0x404].try_into().unwrap());
    let w1 = u32::from_le_bytes(mem.ram[0x404..0x408].try_into().unwrap());
    assert_eq!(w0, 0x1111, "user r8, not the FIQ bank");
    assert_eq!(w1, 0x2222, "user lr, not the FIQ bank");
}

#[test]
fn ldm_pc_with_s_restores_cpsr() {
    // SWI from user → SVC handler does LDMFD sp!, {pc}^ to return.
    let mut mem = LinearMemory::new();
    let program: &[u32] = &[
        0xE3A0_0010, // 100: MOV r0, #0x10
        0xE121_F000, // 104: MSR CPSR_c, r0  ; → user
        0xEF00_0000, // 108: SWI 0
        0xE3A0_7007, // 10C: MOV r7, #7      ; after return
    ];
    for (i, w) in program.iter().enumerate() {
        mem.load(ORG + i as u32 * 4, &w.to_le_bytes());
    }
    // Handler: push lr, pop pc with ^.
    mem.load(0x08, &0xE92D_4000u32.to_le_bytes()); // STMDB sp!, {lr}
    mem.load(0x0C, &0xE8FD_8000u32.to_le_bytes()); // LDMIA sp!, {pc}^
    let mut cpu = Cpu::new();
    cpu.regs.gpr[15] = ORG;
    cpu.regs.gpr[13] = 0x8000; // SVC stack
    steps(&mut cpu, &mut mem, 5);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Usr, "^ with PC restored CPSR");
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[7], 7);
}

#[test]
fn ldm_stm_empty_list_quirk() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0B01, // MOV r0, #0x400
        0xE8A0_0000, // STMIA r0!, {}    ; stores PC, r0 += 0x40
    ]);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[0], 0x440);
    let w0 = u32::from_le_bytes(mem.ram[0x400..0x404].try_into().unwrap());
    assert_eq!(w0, ORG + 4 + 12, "empty list stores PC (+12)");
}

// --- SWP ---------------------------------------------------------------------------

#[test]
fn swp_and_swpb() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_2B01, // MOV r2, #0x400
        0xE3A0_1055, // MOV r1, #0x55
        0xE102_0091, // SWP r0, r1, [r2]
        0xE142_3091, // SWPB r3, r1, [r2]
    ]);
    mem.load(0x400, &0x1234_5678u32.to_le_bytes());
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
    assert_eq!(
        u32::from_le_bytes(mem.ram[0x400..0x404].try_into().unwrap()),
        0x55
    );
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[3], 0x55);
}

// --- Exceptions ----------------------------------------------------------------------

#[test]
fn swi_vectors_to_svc() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0010, // MOV r0, #0x10
        0xE121_F000, // MSR CPSR_c, r0   ; → user mode
        0xEF00_00FF, // SWI 0xFF
    ]);
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Svc);
    assert_eq!(cpu.regs.gpr[15], 0x08);
    assert_eq!(cpu.regs.gpr[14], ORG + 0xC, "LR_svc = SWI + 4");
    assert_eq!(cpu.regs.spsr().mode(), Mode::Usr);
    assert!(cpu.regs.cpsr.i());
    assert!(!cpu.regs.cpsr.f(), "SWI does not touch F");
}

#[test]
fn undefined_instruction_vectors() {
    let (mut cpu, mut mem) = arm(&[
        0xE7F0_00F0, // the canonical undefined encoding
    ]);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Und);
    assert_eq!(cpu.regs.gpr[15], 0x04);
    assert_eq!(cpu.regs.gpr[14], ORG + 4, "LR_und = undef + 4");
}

#[test]
fn coprocessor_ops_are_undefined() {
    for insn in [
        0xEE10_0F10u32, // MRC p15, 0, r0, c0, c0, 0
        0xEE01_0F10,    // MCR p15, 0, r0, c1, c0, 0
        0xED91_0100,    // LDC p1, c0, [r1]
    ] {
        let (mut cpu, mut mem) = arm(&[insn]);
        steps(&mut cpu, &mut mem, 1);
        assert_eq!(
            cpu.regs.cpsr.mode(),
            Mode::Und,
            "{insn:#010X} must take the undefined trap"
        );
    }
}

#[test]
fn irq_delivery_and_return() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0001, // 100: MOV r0, #1
        0xE280_0001, // 104: ADD r0, r0, #1
        0xE280_0001, // 108: ADD r0, r0, #1
    ]);
    // IRQ handler at 0x18: MOV r6, #1; SUBS pc, lr, #4.
    mem.load(0x18, &0xE3A0_6001u32.to_le_bytes());
    mem.load(0x1C, &0xE25E_F004u32.to_le_bytes());
    cpu.regs.cpsr.set(psr::I, false);
    steps(&mut cpu, &mut mem, 1); // MOV at 100
    cpu.set_irq(true);
    steps(&mut cpu, &mut mem, 1); // delivery, not an instruction
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Irq);
    assert!(cpu.regs.cpsr.i());
    assert_eq!(cpu.regs.gpr[15], 0x18);
    assert_eq!(cpu.regs.gpr[14], ORG + 4 + 4, "LR_irq = next + 4");
    cpu.set_irq(false); // handler "acks the device"
    steps(&mut cpu, &mut mem, 2); // MOV r6; SUBS pc, lr, #4
    assert_eq!(cpu.regs.gpr[6], 1);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Svc, "SUBS pc restored mode");
    assert!(!cpu.regs.cpsr.i(), "I restored from SPSR");
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[0], 3, "resumed exactly where interrupted");
}

#[test]
fn irq_masked_by_i_bit_and_fiq_outranks_irq() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0001, // MOV r0, #1
        0xE3A0_0002, // MOV r0, #2
    ]);
    cpu.set_irq(true); // I is set on reset: masked
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Svc);
    assert_eq!(cpu.regs.gpr[0], 1, "masked IRQ does not preempt");

    // Unmask both, assert both: FIQ wins.
    cpu.regs.cpsr.set(psr::I, false);
    cpu.regs.cpsr.set(psr::F, false);
    cpu.set_fiq(true);
    cpu.step(&mut mem);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Fiq);
    assert!(cpu.regs.cpsr.i() && cpu.regs.cpsr.f(), "FIQ masks both");
}

// --- Host traps -----------------------------------------------------------------------

#[test]
fn trap_swi_records_syscall() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0007, // MOV r0, #7
        0xEF90_0004, // SWI 0x900004 (Linux OABI-style comment)
    ]);
    cpu.trap_swi = true;
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.host_trap.take(), Some(HostTrap::Syscall));
    assert_eq!(cpu.regs.gpr[15], ORG + 8, "PC points past the SWI");
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Svc, "no mode change on trap");
}

#[test]
fn trap_faults_hands_undef_to_host() {
    let (mut cpu, mut mem) = arm(&[0xE7F0_00F0]);
    cpu.trap_faults = true;
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(
        cpu.host_trap.take(),
        Some(HostTrap::Exception(Exception::Undefined))
    );
    assert_eq!(
        cpu.regs.gpr[15], ORG,
        "PC rewound to the faulting instruction"
    );
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Svc, "no vectoring");
}

// --- run() ------------------------------------------------------------------------------

#[test]
fn run_batches_and_stops_on_trap() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0000, // 100: MOV r0, #0
        0xE280_0001, // 104: ADD r0, r0, #1
        0xEAFF_FFFD, // 108: B 104
    ]);
    let r = cpu.run(&mut mem, 21);
    assert!(matches!(r.exit, RunExit::Completed));
    assert_eq!(r.executed, 21);
    assert_eq!(cpu.regs.gpr[0], 10, "1 + 10×(ADD+B)");

    // Same program, but a SWI stops the batch.
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0000, // MOV r0, #0
        0xEF00_0000, // SWI
        0xE280_0001, // ADD r0, r0, #1 (not reached before the trap is taken)
    ]);
    cpu.trap_swi = true;
    let r = cpu.run(&mut mem, 100);
    assert!(matches!(r.exit, RunExit::HostTrap));
    assert_eq!(r.executed, 2);
    assert_eq!(cpu.regs.gpr[0], 0);
    assert_eq!(cpu.host_trap.take(), Some(HostTrap::Syscall));
    // Resumes past the SWI.
    let r = cpu.run(&mut mem, 1);
    assert!(matches!(r.exit, RunExit::Completed));
    assert_eq!(cpu.regs.gpr[0], 1);
}

#[test]
fn run_reports_halted_and_wakes_on_irq() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0001, // MOV r0, #1
    ]);
    mem.load(0x18, &0xE3A0_6001u32.to_le_bytes()); // IRQ: MOV r6, #1
    cpu.halted = true;
    let r = cpu.run(&mut mem, 10);
    assert!(matches!(r.exit, RunExit::Halted));
    assert_eq!(r.executed, 0);

    cpu.regs.cpsr.set(psr::I, false);
    cpu.set_irq(true);
    let r = cpu.run(&mut mem, 2);
    assert!(matches!(r.exit, RunExit::Completed));
    assert!(!cpu.halted, "delivered IRQ clears the halt");
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Irq);
    cpu.set_irq(false);
    cpu.run(&mut mem, 1);
    assert_eq!(cpu.regs.gpr[6], 1);
}

#[test]
fn run_matches_step_cycles() {
    let program = &[
        0xE3A0_0005, // MOV r0, #5
        0xE250_0001, // SUBS r0, r0, #1
        0x1AFF_FFFD, // BNE -3... offset: target 104 from 108+8 → -12 → -3
        0xE3A0_1001, // MOV r1, #1
        0xEAFF_FFFE, // B .
    ];
    let (mut a, mut mem_a) = arm(program);
    let (mut b, mut mem_b) = arm(program);
    let mut step_cycles = 0u64;
    for _ in 0..40 {
        a.step(&mut mem_a);
    }
    step_cycles += a.cycles;
    let r = b.run(&mut mem_b, 40);
    assert_eq!(r.executed, 40);
    assert_eq!(b.cycles, step_cycles, "run() and step() agree on cycles");
    assert_eq!(a.regs, b.regs);
}

// --- Thumb ---------------------------------------------------------------------------

#[test]
fn thumb_move_shift_arith() {
    let (mut cpu, mut mem) = thumb(&[
        0x2005, // MOVS r0, #5
        0x3003, // ADDS r0, #3
        0x2803, // CMP r0, #3
        0x0081, // LSLS r1, r0, #2
        0x1842, // ADDS r2, r0, r1
        0x1A83, // SUBS r3, r0, r2
        0x1CC4, // ADDS r4, r0, #3
    ]);
    steps(&mut cpu, &mut mem, 7);
    assert_eq!(cpu.regs.gpr[0], 8);
    assert_eq!(cpu.regs.gpr[1], 32);
    assert_eq!(cpu.regs.gpr[2], 40);
    assert_eq!(cpu.regs.gpr[3], 8u32.wrapping_sub(40));
    assert_eq!(cpu.regs.gpr[4], 11);
    assert!(!cpu.regs.cpsr.z());
}

#[test]
fn thumb_alu_register_ops() {
    let (mut cpu, mut mem) = thumb(&[
        0x2006, // MOVS r0, #6
        0x2103, // MOVS r1, #3
        0x4048, // EORS r0, r1      ; 5
        0x4348, // MULS r0, r1      ; 15
        0x43C8, // MVNS r0, r1      ; !3
        0x4248, // NEG r0, r1       ; -3
        0x4188, // SBCS r0, r1      ; -3 - 3 - !C
    ]);
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[0], 5);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[0], 15);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[0], !3u32);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[0], 3u32.wrapping_neg());
    assert!(cpu.regs.cpsr.n());
}

#[test]
fn thumb_shifts_by_register() {
    let (mut cpu, mut mem) = thumb(&[
        0x2001, // MOVS r0, #1
        0x2104, // MOVS r1, #4
        0x4088, // LSLS r0, r1     ; 0x10
        0x40C8, // LSRS r0, r1     ; 1
        0x2120, // MOVS r1, #32
        0x4088, // LSLS r0, r1     ; 0, C = old bit 0 = 1
    ]);
    steps(&mut cpu, &mut mem, 4);
    assert_eq!(cpu.regs.gpr[0], 1);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[0], 0);
    assert!(cpu.regs.cpsr.c(), "LSL #32 carries out bit 0");
    assert!(cpu.regs.cpsr.z());
}

#[test]
fn thumb_hi_register_ops_and_pc() {
    let (mut cpu, mut mem) = thumb(&[
        0x2005, // MOVS r0, #5
        0x4680, // MOV r8, r0
        0x2000, // MOVS r0, #0
        0x4440, // ADD r0, r8     ; 5
        0x4645, // MOV r5, r8
        0x4685, // MOV sp?? — keep simple: MOV r13, r0
    ]);
    steps(&mut cpu, &mut mem, 5);
    assert_eq!(cpu.regs.gpr[0], 5);
    assert_eq!(cpu.regs.gpr[5], 5);
    assert_eq!(cpu.regs.reg_of(Mode::Svc, 8), 5);
}

#[test]
fn thumb_pc_relative_load_aligns() {
    // Place the code so the LDR sits at an address where pc+4 is not
    // word-aligned without the &!2 rule.
    let (mut cpu, mut mem) = thumb(&[
        0x46C0, // 100: NOP
        0x4A01, // 102: LDR r2, [pc, #4]  ; base (0x102+4)&!2 = 0x104, +4 = 0x108
        0x46C0, // 104: NOP
        0x46C0, // 106: NOP
        0x5678, // 108: pool low
        0x1234, // 10A: pool high
    ]);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[2], 0x1234_5678);
}

#[test]
fn thumb_loads_stores() {
    let (mut cpu, mut mem) = thumb(&[
        0x2140, // MOVS r1, #0x40
        0x0209, // LSLS r1, r1, #8   ; 0x4000
        0x2055, // MOVS r0, #0x55
        0x6008, // STR r0, [r1]
        0x684A, // LDR r2, [r1, #4]
        0x6808, // LDR r0, [r1]
        0x2203, // MOVS r2, #3
        0x5288, // STRH r0, [r1, r2]... rm=r2 rb=r1 rd=r0: at 0x4003? aligned to 0x4002
        0x8888, // LDRH r0, [r1, #4]
        0x7048, // STRB r0, [r1, #1]
        0x7888, // LDRB r0, [r1, #2]
    ]);
    steps(&mut cpu, &mut mem, 6);
    assert_eq!(cpu.regs.gpr[0], 0x55);
    assert_eq!(cpu.regs.gpr[2], 0);
    steps(&mut cpu, &mut mem, 5);
    // STRH 0x55 at (0x4000+3)&!1 = 0x4002; LDRH [0x4004] = 0.
    assert_eq!(
        u16::from_le_bytes(mem.ram[0x4002..0x4004].try_into().unwrap()),
        0x55
    );
}

#[test]
fn thumb_sp_ops_push_pop() {
    let (mut cpu, mut mem) = thumb(&[
        0xB082, // SUB sp, #8
        0x2007, // MOVS r0, #7
        0x9001, // STR r0, [sp, #4]
        0x9901, // LDR r1, [sp, #4]
        0xB402, // PUSH {r1}
        0xBC04, // POP {r2}
        0xB002, // ADD sp, #8
        0xA902, // ADD r1, sp, #8
        0xA001, // ADD r0, pc, #4
    ]);
    let sp0 = cpu.regs.gpr[13];
    steps(&mut cpu, &mut mem, 6);
    assert_eq!(cpu.regs.gpr[1], 7);
    assert_eq!(cpu.regs.gpr[2], 7);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[13], sp0);
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[1], sp0 + 8);
    // ADD r0, pc, #4 at 0x110: ((0x110+4)&!2) + 4 = 0x118.
    assert_eq!(cpu.regs.gpr[0], 0x118);
}

#[test]
fn thumb_push_pop_lr_pc_roundtrip() {
    let (mut cpu, mut mem) = thumb(&[
        0x2005, // 100: MOVS r0, #5
        0xF000, // 102: BL prefix →
        0xF802, // 104: BL suffix → 0x10A
        0x2101, // 106: MOVS r1, #1   ; after return
        0xE7FE, // 108: B .
        0xB500, // 10A: PUSH {lr}
        0x3001, // 10C: ADDS r0, #1
        0xBD00, // 10E: POP {pc}      ; returns to 0x106
    ]);
    steps(&mut cpu, &mut mem, 3); // MOVS, prefix, suffix
    assert_eq!(cpu.regs.gpr[15], 0x10A);
    assert_eq!(cpu.regs.gpr[14], 0x107, "LR = return | 1");
    steps(&mut cpu, &mut mem, 3); // PUSH, ADDS, POP pc
    assert_eq!(cpu.regs.gpr[15], 0x106, "Thumb PC writes drop bit 0");
    assert!(cpu.regs.cpsr.t(), "POP pc stays in Thumb on ARMv4T");
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.gpr[0], 6);
    assert_eq!(cpu.regs.gpr[1], 1);
}

#[test]
fn thumb_ldmia_stmia() {
    let (mut cpu, mut mem) = thumb(&[
        0x2140, // MOVS r1, #0x40
        0x0209, // LSLS r1, r1, #8    ; 0x4000
        0x2211, // MOVS r2, #0x11
        0x2322, // MOVS r3, #0x22
        0xC10C, // STMIA r1!, {r2, r3}
        0x390A, // SUBS r1, #10... use #8: 0x3908
        0xC930, // LDMIA r1!, {r4, r5}
    ]);
    mem.load(ORG + 10, &0x3908u16.to_le_bytes()); // SUBS r1, #8
    steps(&mut cpu, &mut mem, 7);
    assert_eq!(cpu.regs.gpr[4], 0x11);
    assert_eq!(cpu.regs.gpr[5], 0x22);
    assert_eq!(cpu.regs.gpr[1], 0x4008);
}

#[test]
fn thumb_conditional_branches() {
    let (mut cpu, mut mem) = thumb(&[
        0x2800, // 100: CMP r0, #0    ; r0 = 0 → Z
        0xD001, // 102: BEQ 0x108
        0x2101, // 104: MOVS r1, #1   (skipped)
        0xE7FE, // 106: B .
        0x2202, // 108: MOVS r2, #2
        0xD1FE, // 10A: BNE .         (not taken: Z from CMP still set? no —
        //      MOVS r2 cleared Z) — make it BNE +0 → taken to 0x10E
        0x2303, // 10C: MOVS r3, #3   (skipped if BNE taken)
        0x2404, // 10E: MOVS r4, #4
    ]);
    // Fix the BNE: target 0x10E from pc 0x10A: off = (0x10E-0x10A-4)/2 = 0.
    mem.load(ORG + 10, &0xD100u16.to_le_bytes());
    steps(&mut cpu, &mut mem, 5);
    assert_eq!(cpu.regs.gpr[1], 0);
    assert_eq!(cpu.regs.gpr[2], 2);
    assert_eq!(cpu.regs.gpr[3], 0);
    assert_eq!(cpu.regs.gpr[4], 4);
}

#[test]
fn thumb_swi_enters_arm_svc() {
    let (mut cpu, mut mem) = thumb(&[
        0xDF2A, // SWI 0x2A
    ]);
    steps(&mut cpu, &mut mem, 1);
    assert_eq!(cpu.regs.cpsr.mode(), Mode::Svc);
    assert!(!cpu.regs.cpsr.t(), "exceptions are taken in ARM state");
    assert_eq!(cpu.regs.gpr[15], 0x08);
    assert_eq!(cpu.regs.gpr[14], ORG + 2, "LR_svc = SWI + 2 in Thumb");
}

#[test]
fn thumb_bl_negative_offset() {
    let (mut cpu, mut mem) = thumb(&[
        0xE002, // 100: B 0x108
        0x2107, // 102: MOVS r1, #7   ; the subroutine
        0x4770, // 104: BX lr
        0x46C0, // 106: NOP
        0xF7FF, // 108: BL prefix (offset high = -1)
        0xFFFB, // 10A: BL suffix → LR+2*0x7FB... target 0x102
    ]);
    // target = 0x108+4 + (sext(0x7FF)<<12) + (0x7FB<<1)
    //        = 0x10C - 0x1000 + 0xFF6 = 0x102 ✓
    steps(&mut cpu, &mut mem, 3);
    assert_eq!(cpu.regs.gpr[15], 0x102);
    assert_eq!(cpu.regs.gpr[14], 0x10D, "return past the pair, |1");
    steps(&mut cpu, &mut mem, 2);
    assert_eq!(cpu.regs.gpr[1], 7);
    assert_eq!(cpu.regs.gpr[15], 0x10C, "BX consumes bit 0");
}

// --- Cycle accounting ------------------------------------------------------------------

#[test]
fn cycles_accumulate_deterministically() {
    let (mut cpu, mut mem) = arm(&[
        0xE3A0_0003, // MOV r0, #3           ; 1 cycle
        0xE1A0_1110, // MOV r1, r0, LSL r0   ; 2 cycles (reg shift)
        0xEA00_0000, // B +0                 ; 3 cycles
        0xE5A1_0000, // (skipped by B) —
        0xE591_2000, // LDR r2, [r1]         ; 3 cycles
    ]);
    assert_eq!(cpu.step(&mut mem), 1);
    assert_eq!(cpu.step(&mut mem), 2);
    assert_eq!(cpu.step(&mut mem), 3);
    assert_eq!(cpu.step(&mut mem), 3);
    assert_eq!(cpu.cycles, 9);
}
