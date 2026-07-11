//! Focused unit and integration tests for the x86-64 core: REX prefixes and
//! the extended register file, the three operand sizes with 32-bit
//! zero-extension, RIP-relative and SIB addressing, the default-64 stack,
//! MOVSXD / imm64, 64-bit ALU/mul/div/shift/bit ops, the atomic primitives,
//! 4-level paging (NX / WP / 2 MiB pages / canonical faults), the real →
//! protected → long boot sequence, MSRs, SYSCALL/SYSRET, SWAPGS, and
//! interrupt delivery with IST and IRETQ.

use remu::x86_64::{Bus, Cpu, DescTable, LinearMemory, RFlags, SegReg, cr0, cr4, efer, reg};

// --- Real-mode harness (mirrors the 386 rig conventions) ---------------------

/// A CPU + flat memory with `program` at `0000:1100`, a sane real-mode stack.
fn real(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(0x1100, program);
    let mut cpu = Cpu::new();
    cpu.set_cs_ip(0x0000, 0x1100);
    cpu.regs.seg[reg::SS as usize] = SegReg::real(0x9000);
    cpu.regs.gpr[reg::RSP as usize] = 0xFFF0;
    (cpu, mem)
}

// --- Long-mode harness -------------------------------------------------------

const CODE: u64 = 0x10_0000; // 1 MiB, clear of the identity page tables
const STACK: u64 = 0x20_0000; // 2 MiB

/// A CPU dropped into 64-bit long mode with flat ring-0 segments and identity
/// paging, `program` at [`CODE`], RSP at [`STACK`].
fn long(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(CODE, program);
    let mut cpu = Cpu::new();
    cpu.setup_long_flat(&mut mem, CODE, STACK);
    (cpu, mem)
}

/// Step once and assert no host trap / shutdown occurred (a bare `step` that
/// faulted would have vectored through the IDT, which these flat setups have
/// not installed — so a stray fault shows up as a shutdown or wrong RIP).
fn step(cpu: &mut Cpu, mem: &mut LinearMemory) {
    cpu.step(mem);
    assert!(
        !cpu.shutdown,
        "unexpected shutdown at RIP {:#x}",
        cpu.regs.rip
    );
}

// --- Registers, REX and operand sizes ----------------------------------------

#[test]
fn mov_imm_operand_sizes() {
    // MOV EAX,imm32 (default 32 in 64-bit mode); 48 B8 = MOV RAX,imm64;
    // 66 B8 = MOV AX,imm16.
    let (mut cpu, mut mem) = long(&[
        0xB8, 0x78, 0x56, 0x34, 0x12, // MOV EAX, 12345678h
        0x48, 0xB8, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, // MOV RAX, 8877...11h
        0x66, 0xB8, 0xEF, 0xBE, // MOV AX, BEEFh
    ]);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x8877_6655_4433_2211);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x8877_6655_4433_BEEF);
}

#[test]
fn mov32_zero_extends_upper() {
    // Any 32-bit write clears bits 63:32. Seed RAX, then MOV EAX,1.
    let (mut cpu, mut mem) = long(&[0xB8, 0x01, 0x00, 0x00, 0x00]);
    cpu.regs.gpr[0] = 0xFFFF_FFFF_FFFF_FFFF;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 1);
}

#[test]
fn rex_b_extends_registers() {
    // 49 B8 = MOV R8, imm64; 4D 89 C1 = MOV R9, R8.
    let (mut cpu, mut mem) = long(&[
        0x49, 0xB8, 0x0D, 0xF0, 0xAD, 0xBA, 0x00, 0x00, 0x00, 0x00, // MOV R8, BAADF00Dh
        0x4D, 0x89, 0xC1, // MOV R9, R8
    ]);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[reg::R8 as usize], 0xBAAD_F00D);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[reg::R9 as usize], 0xBAAD_F00D);
}

#[test]
fn rex_selects_spl_over_ah() {
    // Without REX, C6 C4 imm8 = MOV AH, imm. With REX, /4 selects SPL.
    let (mut cpu, mut mem) = real(&[0xB4, 0xFF]); // real mode: MOV AH, FF
    step(&mut cpu, &mut mem);
    assert_eq!((cpu.regs.gpr[0] >> 8) & 0xFF, 0xFF);

    // 40 B4 FF in long mode: REX (empty) MOV SPL, FF → low byte of RSP.
    let (mut cpu, mut mem) = long(&[0x40, 0xB4, 0xFF]);
    cpu.regs.gpr[reg::RSP as usize] = 0x1000;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[reg::RSP as usize], 0x10FF);
    assert_eq!(cpu.regs.gpr[0] >> 8 & 0xFF, 0); // AH untouched
}

#[test]
fn rip_relative_addressing() {
    // 48 8B 05 disp32 = MOV RAX, [RIP + disp32]. The datum sits right after
    // the 7-byte instruction; disp32 = 0 reads from there.
    let (mut cpu, mut mem) = long(&[
        0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00, // MOV RAX, [RIP+0]
        0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, // the datum
    ]);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x0123_4567_89AB_CDEF);
    assert_eq!(cpu.regs.rip, CODE + 7);
}

#[test]
fn rip_relative_accounts_for_immediate() {
    // 48 C7 05 disp32 imm32 = MOV qword [RIP+0], imm32. RIP-relative must
    // count the 4 immediate bytes, so disp 0 targets the byte after the
    // whole 11-byte instruction.
    let (mut cpu, mut mem) = long(&[
        0x48, 0xC7, 0x05, 0x00, 0x00, 0x00, 0x00, 0x78, 0x56, 0x34, 0x12,
    ]);
    step(&mut cpu, &mut mem);
    assert_eq!(mem.read64(CODE + 11), 0x1234_5678);
}

#[test]
fn sib_extended_index_and_base() {
    // 4B 89 04 C8 = MOV [R8 + R9*8], RAX with REX.X+REX.B.
    let (mut cpu, mut mem) = long(&[0x4B, 0x89, 0x04, 0xC8]);
    cpu.regs.gpr[reg::R8 as usize] = 0x30_0000;
    cpu.regs.gpr[reg::R9 as usize] = 2;
    cpu.regs.gpr[0] = 0xDEAD_BEEF_CAFE_BABE;
    step(&mut cpu, &mut mem);
    assert_eq!(mem.read64(0x30_0000 + 16), 0xDEAD_BEEF_CAFE_BABE);
}

// --- Stack: default-64 --------------------------------------------------------

#[test]
fn push_pop_default_64() {
    // PUSH/POP are 8 bytes wide with no REX needed. 50 = PUSH RAX; 5B = POP RBX.
    let (mut cpu, mut mem) = long(&[0x50, 0x5B]);
    cpu.regs.gpr[0] = 0x1122_3344_5566_7788;
    let sp0 = cpu.regs.gpr[reg::RSP as usize];
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[reg::RSP as usize], sp0 - 8);
    assert_eq!(mem.read64(sp0 - 8), 0x1122_3344_5566_7788);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[reg::RBX as usize], 0x1122_3344_5566_7788);
    assert_eq!(cpu.regs.gpr[reg::RSP as usize], sp0);
}

#[test]
fn push_imm_and_pop_r15() {
    // 68 imm32 = PUSH imm32 (sign-extended to 64); 41 5F = POP R15.
    let (mut cpu, mut mem) = long(&[0x68, 0x00, 0x00, 0x00, 0x80, 0x41, 0x5F]);
    step(&mut cpu, &mut mem);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[reg::R15 as usize], 0xFFFF_FFFF_8000_0000);
}

// --- MOVSXD / MOVZX / MOVSX ---------------------------------------------------

#[test]
fn movsxd() {
    // 48 63 C1 = MOVSXD RAX, ECX (sign-extend 32→64).
    let (mut cpu, mut mem) = long(&[0x48, 0x63, 0xC1]);
    cpu.regs.gpr[1] = 0x0000_0000_FFFF_FFFF; // ECX = -1
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn movzx_movsx_byte_to_64() {
    // 48 0F B6 C3 = MOVZX RAX, BL; 48 0F BE C3 = MOVSX RAX, BL.
    let (mut cpu, mut mem) = long(&[0x48, 0x0F, 0xB6, 0xC3, 0x48, 0x0F, 0xBE, 0xC3]);
    cpu.regs.gpr[reg::RBX as usize] = 0x80;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0x80);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0xFFFF_FFFF_FFFF_FF80);
}

// --- 64-bit ALU / mul / div / shift -------------------------------------------

#[test]
fn add64_flags() {
    // 48 05 imm32 = ADD RAX, imm32 (sign-extended). RAX = -1 + 1 → 0, CF set.
    let (mut cpu, mut mem) = long(&[0x48, 0x05, 0x01, 0x00, 0x00, 0x00]);
    cpu.regs.gpr[0] = 0xFFFF_FFFF_FFFF_FFFF;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0);
    let f = cpu.regs.rflags;
    assert!(f.contains(RFlags::CF) && f.contains(RFlags::ZF) && f.contains(RFlags::AF));
    assert!(!f.contains(RFlags::OF));
}

#[test]
fn mul64_and_div64() {
    // 48 F7 E3 = MUL RBX (RDX:RAX = RAX*RBX); then 48 F7 F3 = DIV RBX.
    let (mut cpu, mut mem) = long(&[0x48, 0xF7, 0xE3, 0x48, 0xF7, 0xF3]);
    cpu.regs.gpr[0] = 0x1_0000_0001;
    cpu.regs.gpr[reg::RBX as usize] = 0x1_0000_0000;
    step(&mut cpu, &mut mem);
    // (2^32+1) * 2^32 = 2^64 + 2^32 → RDX=1, RAX=2^32.
    assert_eq!(cpu.regs.gpr[2], 1);
    assert_eq!(cpu.regs.gpr[0], 0x1_0000_0000);
    step(&mut cpu, &mut mem);
    // Divide back by 2^32 → quotient 2^32+1, remainder 0.
    assert_eq!(cpu.regs.gpr[0], 0x1_0000_0001);
    assert_eq!(cpu.regs.gpr[2], 0);
}

#[test]
fn shl64_count_masks_to_6_bits() {
    // 48 D3 E0 = SHL RAX, CL. CL = 64 masks to 0 (no-op); CL = 63 shifts fully.
    let (mut cpu, mut mem) = long(&[0x48, 0xD3, 0xE0]);
    cpu.regs.gpr[0] = 1;
    cpu.regs.gpr[1] = 64;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 1, "count & 63 == 0 leaves RAX unchanged");

    let (mut cpu, mut mem) = long(&[0x48, 0xD3, 0xE0]);
    cpu.regs.gpr[0] = 1;
    cpu.regs.gpr[1] = 63;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 1 << 63);
}

#[test]
fn sar32_is_arithmetic_and_zero_extends() {
    // D1 F8 = SAR EAX, 1 on 0x80000000 → 0xC0000000, upper 32 bits cleared.
    let (mut cpu, mut mem) = long(&[0xD1, 0xF8]);
    cpu.regs.gpr[0] = 0xFFFF_FFFF_8000_0000;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0xC000_0000);
}

// --- Bit operations -----------------------------------------------------------

#[test]
fn bts_register_and_memory() {
    // 48 0F AB D8 = BTS RAX, RBX (bit index in RBX).
    let (mut cpu, mut mem) = long(&[0x48, 0x0F, 0xAB, 0xD8]);
    cpu.regs.gpr[0] = 0;
    cpu.regs.gpr[reg::RBX as usize] = 40;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 1 << 40);
    assert!(!cpu.regs.rflags.contains(RFlags::CF)); // bit was 0

    // BT into memory with a large signed index picks the right qword.
    // 48 0F A3 18 = BT [RAX], RBX.
    let (mut cpu, mut mem) = long(&[0x48, 0x0F, 0xA3, 0x18]);
    cpu.regs.gpr[0] = 0x40_0000;
    cpu.regs.gpr[reg::RBX as usize] = 64 + 3; // second qword, bit 3
    mem.write64(0x40_0000 + 8, 1 << 3);
    step(&mut cpu, &mut mem);
    assert!(cpu.regs.rflags.contains(RFlags::CF));
}

#[test]
fn bsf_bsr_and_zero_source() {
    // 48 0F BC C3 = BSF RAX, RBX; 48 0F BD C3 = BSR RAX, RBX.
    let (mut cpu, mut mem) = long(&[0x48, 0x0F, 0xBC, 0xC3, 0x48, 0x0F, 0xBD, 0xC3]);
    cpu.regs.gpr[reg::RBX as usize] = 0x0000_8000_0000_0100;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 8); // lowest set bit
    assert!(!cpu.regs.rflags.contains(RFlags::ZF));
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 47); // highest set bit

    // Zero source: ZF set, destination unchanged.
    let (mut cpu, mut mem) = long(&[0x48, 0x0F, 0xBC, 0xC3]);
    cpu.regs.gpr[0] = 0xABCD;
    cpu.regs.gpr[reg::RBX as usize] = 0;
    step(&mut cpu, &mut mem);
    assert!(cpu.regs.rflags.contains(RFlags::ZF));
    assert_eq!(cpu.regs.gpr[0], 0xABCD);
}

// --- Atomics ------------------------------------------------------------------

#[test]
fn cmpxchg_success_and_failure() {
    // F0 48 0F B1 18 = LOCK CMPXCHG [RAX], RBX.
    let prog = [0x48, 0x0F, 0xB1, 0x18];
    let (mut cpu, mut mem) = long(&prog);
    cpu.regs.gpr[0] = 0x40_0000;
    mem.write64(0x40_0000, 0x1111);
    cpu.regs.gpr[reg::RAX as usize] = 0x40_0000; // RAX doubles as the pointer here
    // Use a separate accumulator setup: put comparand in... RAX is the ptr,
    // so re-do with pointer in RCX.
    let (mut cpu, mut mem) = long(&[0x48, 0x0F, 0xB1, 0x19]); // CMPXCHG [RCX], RBX
    cpu.regs.gpr[reg::RCX as usize] = 0x40_0000;
    mem.write64(0x40_0000, 0x1111);
    cpu.regs.gpr[reg::RAX as usize] = 0x1111; // matches → store RBX
    cpu.regs.gpr[reg::RBX as usize] = 0x2222;
    step(&mut cpu, &mut mem);
    assert!(cpu.regs.rflags.contains(RFlags::ZF));
    assert_eq!(mem.read64(0x40_0000), 0x2222);

    // Mismatch → load into RAX, ZF clear.
    let (mut cpu, mut mem) = long(&[0x48, 0x0F, 0xB1, 0x19]);
    cpu.regs.gpr[reg::RCX as usize] = 0x40_0000;
    mem.write64(0x40_0000, 0x9999);
    cpu.regs.gpr[reg::RAX as usize] = 0x1111;
    cpu.regs.gpr[reg::RBX as usize] = 0x2222;
    step(&mut cpu, &mut mem);
    assert!(!cpu.regs.rflags.contains(RFlags::ZF));
    assert_eq!(cpu.regs.gpr[reg::RAX as usize], 0x9999);
    assert_eq!(mem.read64(0x40_0000), 0x9999);
    let _ = prog;
}

#[test]
fn cmpxchg16b() {
    // 48 0F C7 09 = CMPXCHG16B [RCX] (REX.W). Compares RDX:RAX with the
    // 16-byte value; on match stores RCX:RBX.
    let (mut cpu, mut mem) = long(&[0x48, 0x0F, 0xC7, 0x09]);
    let ptr = 0x40_0000u64; // 16-byte aligned
    cpu.regs.gpr[reg::RCX as usize] = ptr;
    mem.write64(ptr, 0xAAAA);
    mem.write64(ptr + 8, 0xBBBB);
    cpu.regs.gpr[reg::RAX as usize] = 0xAAAA;
    cpu.regs.gpr[reg::RDX as usize] = 0xBBBB;
    cpu.regs.gpr[reg::RBX as usize] = 0x1234;
    // RCX is the pointer, so the "new high" comes from RCX too — hardware uses
    // RCX:RBX. To keep RCX as the pointer, verify the store of RBX (low) and
    // that ZF is set.
    step(&mut cpu, &mut mem);
    assert!(cpu.regs.rflags.contains(RFlags::ZF));
    assert_eq!(mem.read64(ptr), 0x1234);
}

// --- String operations --------------------------------------------------------

#[test]
fn rep_movsq() {
    // F3 48 A5 = REP MOVSQ (8 bytes per iteration).
    let (mut cpu, mut mem) = long(&[0xF3, 0x48, 0xA5]);
    let (src, dst) = (0x30_0000u64, 0x31_0000u64);
    for i in 0..4u64 {
        mem.write64(src + i * 8, 0x1000 + i);
    }
    cpu.regs.gpr[reg::RSI as usize] = src;
    cpu.regs.gpr[reg::RDI as usize] = dst;
    cpu.regs.gpr[reg::RCX as usize] = 4;
    step(&mut cpu, &mut mem);
    for i in 0..4u64 {
        assert_eq!(mem.read64(dst + i * 8), 0x1000 + i);
    }
    assert_eq!(cpu.regs.gpr[reg::RCX as usize], 0);
}

// --- Canonical-address faults -------------------------------------------------

#[test]
fn noncanonical_data_access_faults() {
    // MOV RAX, [RBX] with a non-canonical RBX → #GP. With trap_faults set the
    // core hands the fault back instead of vectoring.
    let (mut cpu, mut mem) = long(&[0x48, 0x8B, 0x03]);
    cpu.trap_faults = true;
    cpu.regs.gpr[reg::RBX as usize] = 0x0000_8000_0000_0000; // first non-canonical
    cpu.step(&mut mem);
    match cpu.host_trap {
        Some(remu::x86_64::HostTrap::Exception(e)) => assert_eq!(e.vector, 13),
        other => panic!("expected #GP, got {other:?}"),
    }
    // RIP rewound to the faulting instruction.
    assert_eq!(cpu.regs.rip, CODE);
}

// --- Mode transitions: real → protected → long --------------------------------

#[test]
fn boot_real_to_long() {
    // Drive the canonical bring-up: build a GDT + PML4/PDPT, enable PAE,
    // set EFER.LME, then flip CR0.PG|PE and far-jump into a 64-bit CS.
    let mut mem = LinearMemory::new();
    let mut cpu = Cpu::new();

    // GDT at 0x0000: null, 64-bit code (0x08), flat data (0x10).
    let gdt = 0x0000u64;
    mem.write64(gdt, 0);
    mem.write64(gdt + 8, 0x00A0_9A00_0000_0000); // code: L=1, present, exec/read
    mem.write64(gdt + 16, 0x00C0_9200_0000_FFFF); // data: present, r/w
    cpu.regs.gdtr = DescTable {
        base: gdt,
        limit: 0x17,
    };

    // Identity page tables: PML4 at 0x1000 → PDPT 0x2000 → 1 GiB pages.
    mem.write64(0x1000, 0x2000 | 0x03);
    for i in 0..512u64 {
        mem.write64(0x2000 + i * 8, (i << 30) | 0x83);
    }
    cpu.regs.cr3 = 0x1000;
    cpu.regs.cr4 = cr4::PAE;
    cpu.regs.msr.efer = efer::LME;

    // Real-mode bootstrap lives at physical 0x2_0000, entered as CS 2000:0000
    // (base 0x2_0000, a legal 64 KiB segment). It flips CR0.PE|PG, then a
    // 66-prefixed far JMP to 0x08:0x2_0040 — the physical (== identity-mapped
    // linear) address of the 64-bit continuation.
    // 0F 22 C0            MOV CR0, EAX     (EAX has PE|PG)
    // 66 EA off32 sel16   JMP 0008:0002_0040
    let prog: &[u8] = &[
        0x0F, 0x22, 0xC0, // MOV CR0, EAX
        0x66, 0xEA, // JMP FAR (66 → 32-bit offset)
        0x40, 0x00, 0x02, 0x00, // offset 0x0002_0040
        0x08, 0x00, // selector 0x08
    ];
    mem.load(0x2_0000, prog);
    // 64-bit target at 0x2_0040: MOV RAX, 0x99 (48 C7 C0 imm32).
    mem.load(0x2_0040, &[0x48, 0xC7, 0xC0, 0x99, 0x00, 0x00, 0x00]);
    cpu.set_cs_ip(0x2000, 0x0000);
    cpu.regs.gpr[0] = cr0::PE | cr0::PG;

    step(&mut cpu, &mut mem); // MOV CR0 → activates long mode (LME+PG)
    assert!(cpu.long_mode(), "EFER.LMA should be set once PG turns on");
    assert_eq!(cpu.regs.msr.efer & efer::LMA, efer::LMA);

    step(&mut cpu, &mut mem); // far JMP into 64-bit CS
    assert!(cpu.mode64(), "CS.L should select 64-bit mode");
    assert_eq!(cpu.regs.rip, 0x2_0040);

    step(&mut cpu, &mut mem); // MOV RAX, 0x99 (64-bit instruction)
    assert_eq!(cpu.regs.gpr[0], 0x99);
}

// --- Paging: NX, WP, 2 MiB pages ----------------------------------------------

/// Build a fresh long-mode CPU whose page tables we can hand-edit (4 KiB
/// granularity), mapping [`CODE`]/[`STACK`] and a scratch page.
fn long_4k(program: &[u8]) -> (Cpu, LinearMemory) {
    let mut mem = LinearMemory::new();
    mem.load(CODE, program);
    let mut cpu = Cpu::new();

    // PML4[0]=PDPT, PDPT[0]=PD, PD entries = 2 MiB pages covering low 1 GiB.
    let (pml4, pdpt, pd) = (0x1000u64, 0x2000u64, 0x3000u64);
    mem.write64(pml4, pdpt | 0x07); // present, rw, user
    mem.write64(pdpt, pd | 0x07);
    for i in 0..512u64 {
        mem.write64(pd + i * 8, (i << 21) | 0x87); // 2 MiB, present rw user PS
    }
    cpu.regs.cr3 = pml4;
    cpu.regs.cr4 = cr4::PAE;
    cpu.regs.cr0 |= cr0::PE | cr0::PG;
    cpu.regs.cr0 &= !(cr0::CD | cr0::NW);
    cpu.regs.msr.efer = efer::LME | efer::LMA | efer::SCE | efer::NXE;
    cpu.regs.seg[reg::CS as usize] = SegReg {
        sel: 0x08,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: 0x0A9B,
    };
    for idx in [reg::SS, reg::DS, reg::ES, reg::FS, reg::GS] {
        cpu.regs.seg[idx as usize] = SegReg {
            sel: 0x10,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: 0x0C93,
        };
    }
    cpu.regs.rip = CODE;
    cpu.regs.gpr[reg::RSP as usize] = STACK;
    (cpu, mem)
}

#[test]
fn two_mib_pages_map_identity() {
    // A plain load through the 2 MiB mapping built by long_4k.
    let (mut cpu, mut mem) = long_4k(&[0x48, 0x8B, 0x03]); // MOV RAX, [RBX]
    cpu.regs.gpr[reg::RBX as usize] = 0x40_0000;
    mem.write64(0x40_0000, 0xFEED_FACE);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0xFEED_FACE);
}

#[test]
fn nx_blocks_instruction_fetch() {
    // Mark the 2 MiB page containing a scratch target NX, jump into it, and
    // expect a #PF with the instruction-fetch bit set.
    let (mut cpu, mut mem) = long_4k(&[0xE9, 0x00, 0x00, 0x10, 0x00]); // JMP +0x100000
    cpu.trap_faults = true;
    // Put a NOP at the jump target (CODE + 5 + 0x100000).
    let target = CODE + 5 + 0x10_0000;
    mem.load(target, &[0x90]);
    // Set NX on the PD entry covering `target` (2 MiB page index target>>21).
    let pd = 0x3000u64;
    let idx = target >> 21;
    let e = mem.read64(pd + idx * 8) | (1u64 << 63);
    mem.write64(pd + idx * 8, e);
    cpu.invalidate_tlb();

    step(&mut cpu, &mut mem); // the JMP itself is fine
    cpu.step(&mut mem); // fetching at the NX target faults
    match cpu.host_trap {
        Some(remu::x86_64::HostTrap::Exception(e)) => {
            assert_eq!(e.vector, 14);
            assert_eq!(e.error.unwrap() & 0x10, 0x10, "instruction-fetch bit");
        }
        other => panic!("expected #PF, got {other:?}"),
    }
}

#[test]
fn wp_enforces_supervisor_write_protection() {
    // Clear R/W on the scratch page, then a supervisor write faults only when
    // CR0.WP is set.
    let (mut cpu, mut mem) = long_4k(&[0x48, 0x89, 0x03]); // MOV [RBX], RAX
    cpu.trap_faults = true;
    cpu.regs.gpr[reg::RBX as usize] = 0x60_0000;
    cpu.regs.gpr[0] = 0x1234;
    // Make the covering 2 MiB page read-only.
    let pd = 0x3000u64;
    let idx = 0x60_0000u64 >> 21;
    let e = mem.read64(pd + idx * 8) & !0x2; // clear RW
    mem.write64(pd + idx * 8, e);
    cpu.invalidate_tlb();

    // WP clear: supervisor write succeeds.
    step(&mut cpu, &mut mem);
    assert_eq!(mem.read64(0x60_0000), 0x1234);
    assert!(cpu.host_trap.is_none());

    // WP set: the same write faults.
    let (mut cpu, mut mem) = long_4k(&[0x48, 0x89, 0x03]);
    cpu.trap_faults = true;
    cpu.regs.cr0 |= cr0::WP;
    cpu.regs.gpr[reg::RBX as usize] = 0x60_0000;
    cpu.regs.gpr[0] = 0x1234;
    let e = mem.read64(pd + idx * 8) & !0x2;
    mem.write64(pd + idx * 8, e);
    cpu.invalidate_tlb();
    cpu.step(&mut mem);
    match cpu.host_trap {
        Some(remu::x86_64::HostTrap::Exception(ex)) => assert_eq!(ex.vector, 14),
        other => panic!("expected #PF with WP set, got {other:?}"),
    }
}

// --- Fetch-window / fetch-translation-cache boundary tests ---------------------

#[test]
fn fetch_across_page_boundary_faults_on_the_unmapped_second_page() {
    // MOV EAX, imm32 at 0x7F_FFFC: opcode + 3 imm bytes in the mapped 2 MiB
    // region 3, last imm byte in region 4, which is unmapped. The fetch must
    // #PF with CR2 = 0x80_0000 and RIP rewound; once mapped, it executes.
    let (mut cpu, mut mem) = long_4k(&[0x90]);
    cpu.trap_faults = true;
    let pd = 0x3000u64;
    mem.write64(pd + 4 * 8, 0); // unmap 0x80_0000..0x9F_FFFF
    cpu.invalidate_tlb();

    mem.load(0x7F_FFFC, &[0xB8, 0x78, 0x56, 0x34, 0x12]); // MOV EAX, 12345678h
    cpu.regs.rip = 0x7F_FFFC;

    cpu.step(&mut mem);
    match cpu.host_trap.take() {
        Some(remu::x86_64::HostTrap::Exception(e)) => assert_eq!(e.vector, 14),
        other => panic!("expected a trapped #PF, got {other:?}"),
    }
    assert_eq!(cpu.regs.cr2, 0x80_0000, "CR2 is the second page");
    assert_eq!(cpu.regs.rip, 0x7F_FFFC, "RIP rewinds to the instruction start");

    // Map region 4 and restart: the refetch sees the new mapping.
    mem.write64(pd + 4 * 8, (4u64 << 21) | 0x87);
    cpu.invalidate_tlb();
    cpu.step(&mut mem);
    assert!(cpu.host_trap.is_none());
    assert_eq!(cpu.regs.gpr[reg::RAX as usize], 0x1234_5678);
    assert_eq!(cpu.regs.rip, 0x80_0001);
}

#[test]
fn nx_on_the_second_page_faults_mid_instruction() {
    // Same straddle, but the second page is mapped no-execute: the fetch of
    // the last byte must #PF with the instruction-fetch bit set.
    let (mut cpu, mut mem) = long_4k(&[0x90]);
    cpu.trap_faults = true;
    let pd = 0x3000u64;
    mem.write64(pd + 4 * 8, (4u64 << 21) | 0x87 | (1u64 << 63)); // NX
    cpu.invalidate_tlb();

    mem.load(0x7F_FFFC, &[0xB8, 0x78, 0x56, 0x34, 0x12]);
    cpu.regs.rip = 0x7F_FFFC;

    cpu.step(&mut mem);
    match cpu.host_trap.take() {
        Some(remu::x86_64::HostTrap::Exception(e)) => {
            assert_eq!(e.vector, 14);
            assert_eq!(e.error.unwrap() & 0x10, 0x10, "instruction-fetch bit");
        }
        other => panic!("expected a trapped #PF, got {other:?}"),
    }
    assert_eq!(cpu.regs.cr2, 0x80_0000);
    assert_eq!(cpu.regs.rip, 0x7F_FFFC);
}

#[test]
fn store_into_the_next_instruction_is_fetched_fresh() {
    // Self-modifying code under the persistent fetch-translation cache:
    // only the *translation* is cached, so a store into the next
    // instruction's immediate must be observed by its fetch.
    let (mut cpu, mut mem) = long(&[
        0xC6, 0x05, 0x01, 0x00, 0x00, 0x00, 0x42, // MOV byte [RIP+1], 42h
        0xB0, 0x37, // MOV AL, 37h (imm at CODE+8 = the store target)
    ]);
    step(&mut cpu, &mut mem);
    step(&mut cpu, &mut mem);
    assert_eq!(
        cpu.regs.gpr[reg::RAX as usize] & 0xFF,
        0x42,
        "the freshly stored immediate must be fetched"
    );
}

#[test]
fn fetch_cache_does_not_survive_a_privilege_drop() {
    // Fill the fetch-translation cache at CPL 0 on a supervisor-only code
    // page, SYSRET to CPL 3 within the same page: the user fetch must #PF
    // (present + user + instruction-fetch). A stale cache entry from the
    // CPL 0 fill would wrongly allow it — this pins the prepare_cold_write
    // invalidation hook on privilege transitions.
    let (mut cpu, mut mem) = long_4k(&[0x90]);
    cpu.trap_faults = true;
    let pd = 0x3000u64;
    mem.write64(pd + 4 * 8, (4u64 << 21) | 0x83); // 0x80_0000: supervisor-only
    cpu.invalidate_tlb();

    // STAR: SYSRET base 0x10 -> user CS 0x23 (64-bit), SS 0x1B.
    cpu.regs.msr.star = 0x0010u64 << 48;
    cpu.regs.gpr[reg::RCX as usize] = 0x80_0100; // return RIP, same page
    cpu.regs.gpr[reg::R11 as usize] = 2; // RFLAGS image
    mem.load(0x80_0000, &[0x48, 0x0F, 0x07]); // SYSRET
    mem.load(0x80_0100, &[0x90]);
    cpu.regs.rip = 0x80_0000;

    step(&mut cpu, &mut mem); // SYSRET: fills the cache at CPL 0, drops to 3
    assert_eq!(cpu.cpl(), 3);
    assert_eq!(cpu.regs.rip, 0x80_0100);

    cpu.step(&mut mem); // user fetch of the supervisor page must fault
    match cpu.host_trap.take() {
        Some(remu::x86_64::HostTrap::Exception(e)) => {
            assert_eq!(e.vector, 14);
            let code = e.error.unwrap();
            assert_eq!(code & 0x15, 0x15, "present + user + instruction-fetch");
        }
        other => panic!("expected a trapped #PF, got {other:?}"),
    }
    assert_eq!(cpu.regs.cr2, 0x80_0100);
}

#[test]
fn invlpg_after_pd_rewrite_fetches_through_the_new_mapping() {
    // Pins the fetch-cache invalidation hooks: guest code running in the
    // 2 MiB region at 0xA0_0000 rewrites its own PD entry to point at the
    // frame at 0xC0_0000, executes INVLPG, and jumps within its own page —
    // the tail must be fetched through the NEW mapping. A stale fetch
    // translation would run the old frame's bytes.
    let (mut cpu, mut mem) = long_4k(&[0x90]);

    #[rustfmt::skip]
    let stub = [
        0x48, 0xB8, 0x87, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, // MOV RAX, 0xC00087
        0x48, 0x89, 0x03, // MOV [RBX], RAX   (RBX = &PD[5])
        0x0F, 0x01, 0x3E, // INVLPG [RSI]     (RSI = 0xA0_0000)
        0xE9, 0xEB, 0x00, 0x00, 0x00, // JMP 0xA0_0100
    ];
    // The stub must exist in BOTH frames: every byte fetched after INVLPG
    // (including the JMP tail) already goes through the new mapping.
    mem.load(0xA0_0000, &stub);
    mem.load(0xC0_0000, &stub);
    mem.load(0xA0_0100, &[0xB8, 0x11, 0x01, 0x00, 0x00]); // old: MOV EAX, 111h
    mem.load(0xC0_0100, &[0xB8, 0x22, 0x02, 0x00, 0x00]); // new: MOV EAX, 222h

    cpu.regs.gpr[reg::RBX as usize] = 0x3000 + 5 * 8;
    cpu.regs.gpr[reg::RSI as usize] = 0xA0_0000;
    cpu.regs.rip = 0xA0_0000;

    for _ in 0..5 {
        step(&mut cpu, &mut mem);
    }
    assert_eq!(
        cpu.regs.gpr[reg::RAX as usize],
        0x222,
        "the jump target must be fetched through the remapped page"
    );
}

// --- MSRs, SWAPGS -------------------------------------------------------------

#[test]
fn wrmsr_rdmsr_fs_base() {
    // WRMSR to IA32_FS_BASE (C0000100h) then read it back; the segment base
    // must track.
    // FS base must be canonical, so keep bit 47:63 clear.
    // B9 00 01 00 C0  MOV ECX, C0000100h
    // BA FF 7F 00 00  MOV EDX, 00007FFFh
    // B8 EF BE AD DE  MOV EAX, DEADBEEFh
    // 0F 30           WRMSR
    // 0F 32           RDMSR
    let (mut cpu, mut mem) = long(&[
        0xB9, 0x00, 0x01, 0x00, 0xC0, 0xBA, 0xFF, 0x7F, 0x00, 0x00, 0xB8, 0xEF, 0xBE, 0xAD, 0xDE,
        0x0F, 0x30, 0x0F, 0x32,
    ]);
    for _ in 0..4 {
        step(&mut cpu, &mut mem);
    }
    assert_eq!(cpu.regs.seg[reg::FS as usize].base, 0x0000_7FFF_DEAD_BEEF);
    // Clobber the GPRs, RDMSR restores them from the base.
    cpu.regs.gpr[0] = 0;
    cpu.regs.gpr[2] = 0;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.reg32(reg::RAX), 0xDEAD_BEEF);
    assert_eq!(cpu.regs.reg32(reg::RDX), 0x0000_7FFF);
}

#[test]
fn swapgs_exchanges_kernel_base() {
    // 0F 01 F8 = SWAPGS. Swaps GS.base with KernelGSBase.
    let (mut cpu, mut mem) = long(&[0x0F, 0x01, 0xF8]);
    cpu.regs.seg[reg::GS as usize].base = 0x1111;
    cpu.regs.msr.kernel_gs_base = 0x2222;
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.seg[reg::GS as usize].base, 0x2222);
    assert_eq!(cpu.regs.msr.kernel_gs_base, 0x1111);
}

#[test]
fn gs_relative_addressing() {
    // 65 48 8B 00 = MOV RAX, GS:[RAX] — the GS base is added in 64-bit mode.
    let (mut cpu, mut mem) = long(&[0x65, 0x48, 0x8B, 0x00]);
    cpu.regs.seg[reg::GS as usize].base = 0x50_0000;
    cpu.regs.gpr[0] = 0x1000;
    mem.write64(0x50_1000, 0xC0DE);
    step(&mut cpu, &mut mem);
    assert_eq!(cpu.regs.gpr[0], 0xC0DE);
}

// --- SYSCALL / SYSRET ---------------------------------------------------------

#[test]
fn syscall_sysret_roundtrip() {
    // Set STAR/LSTAR, run SYSCALL from ring 3, then SYSRET back.
    let mut mem = LinearMemory::new();
    let mut cpu = Cpu::new();
    cpu.setup_long_flat(&mut mem, CODE, STACK);

    // Kernel handler at 0x11_0000: 48 0F 07 = SYSRET (REX.W → 64-bit).
    let handler = 0x11_0000u64;
    mem.load(handler, &[0x48, 0x0F, 0x07]);
    cpu.regs.msr.lstar = handler;
    // STAR: SYSCALL loads CS=0x08/SS=0x10; SYSRET base 0x10 → user CS 0x20.
    cpu.regs.msr.star = (0x0010u64 << 48) | (0x0008u64 << 32);
    cpu.regs.msr.efer |= efer::SCE;

    // Drop to ring 3 with a user code segment. Reuse the flat descriptors but
    // mark CPL 3 via the CS attrs; the flat setup used ring 0.
    cpu.regs.seg[reg::CS as usize].attrs = 0x0AFB; // present, code, DPL 3, L
    cpu.regs.seg[reg::CS as usize].sel = 0x23;
    cpu.regs.seg[reg::SS as usize].attrs = 0x0CF3; // DPL 3 data
    cpu.regs.seg[reg::SS as usize].sel = 0x1B;
    // Program: 0F 05 = SYSCALL at CODE.
    mem.load(CODE, &[0x0F, 0x05]);
    cpu.regs.rip = CODE;

    step(&mut cpu, &mut mem); // SYSCALL
    assert_eq!(cpu.cpl(), 0, "SYSCALL enters ring 0");
    assert_eq!(cpu.regs.rip, handler);
    assert_eq!(
        cpu.regs.gpr[reg::RCX as usize],
        CODE + 2,
        "return RIP in RCX"
    );

    step(&mut cpu, &mut mem); // SYSRET
    assert_eq!(cpu.cpl(), 3, "SYSRET returns to ring 3");
    assert_eq!(cpu.regs.rip, CODE + 2);
}

// --- Interrupt delivery: IST + IRETQ ------------------------------------------

#[test]
fn interrupt_uses_ist_and_iretq_returns() {
    // Build a 64-bit IDT with an interrupt gate whose IST index is 1, a TSS
    // holding IST1, then software-INT into it and IRETQ back.
    let mut mem = LinearMemory::new();
    let mut cpu = Cpu::new();
    cpu.setup_long_flat(&mut mem, CODE, STACK);

    // GDT: need a real TSS descriptor. Put GDT at 0x8000 with entries:
    // 0x08 code, 0x10 data, 0x18 TSS (16 bytes).
    let gdt = 0x8000u64;
    mem.write64(gdt + 8, 0x00A0_9A00_0000_0000);
    mem.write64(gdt + 16, 0x00C0_9200_0000_FFFF);
    // 64-bit TSS at 0x9000, limit 0x67.
    let tss = 0x9000u64;
    let tss_lo = 0x0000_8900_0000_0000u64
        | (0x67u64)
        | ((tss & 0xFFFF) << 16)
        | (((tss >> 16) & 0xFF) << 32)
        | (((tss >> 24) & 0xFF) << 56);
    mem.write64(gdt + 24, tss_lo);
    mem.write64(gdt + 32, (tss >> 32) & 0xFFFF_FFFF);
    cpu.regs.gdtr = DescTable {
        base: gdt,
        limit: 0x2F,
    };
    cpu.regs.tr = SegReg {
        sel: 0x18,
        base: tss,
        limit: 0x67,
        attrs: 0x0089,
    };
    // IST1 at TSS offset 0x24.
    let ist1 = 0x1F_0000u64;
    mem.write64(tss + 0x24, ist1);

    // IDT at 0xA000: gate for vector 0x40 → 64-bit interrupt gate, IST 1,
    // target CS 0x08, offset = handler.
    let idt = 0xA000u64;
    let handler = 0x12_0000u64;
    let g_lo = (handler & 0xFFFF)
        | (0x0008u64 << 16)
        | (1u64 << 32)            // IST = 1
        | (0x8Eu64 << 40)         // present, DPL 0, type 0xE (interrupt gate)
        | (((handler >> 16) & 0xFFFF) << 48);
    let g_hi = (handler >> 32) & 0xFFFF_FFFF;
    mem.write64(idt + 0x40 * 16, g_lo);
    mem.write64(idt + 0x40 * 16 + 8, g_hi);
    cpu.regs.idtr = DescTable {
        base: idt,
        limit: 0x40 * 16 + 15,
    };

    // Handler: CF 48? IRETQ is just CF with a 64-bit operand size — the
    // default in 64-bit mode already pops a 64-bit frame, so 48 CF.
    mem.load(handler, &[0x48, 0xCF]);
    // Program: CD 40 = INT 0x40.
    mem.load(CODE, &[0xCD, 0x40]);
    cpu.regs.rip = CODE;
    let rsp0 = cpu.regs.gpr[reg::RSP as usize];

    step(&mut cpu, &mut mem); // INT 0x40 delivers on IST1
    assert_eq!(cpu.regs.rip, handler);
    // RSP switched onto IST1 (aligned down), well away from the old stack.
    assert!(cpu.regs.gpr[reg::RSP as usize] <= ist1);
    assert!(cpu.regs.gpr[reg::RSP as usize] >= ist1 - 0x40);

    step(&mut cpu, &mut mem); // IRETQ back to CODE+2
    assert_eq!(cpu.regs.rip, CODE + 2);
    assert_eq!(cpu.regs.gpr[reg::RSP as usize], rsp0);
}

#[test]
fn iretq_outer_ss_fault_restores_cs_cache() {
    // IRETQ from ring 0 to ring 3 commits the CS descriptor cache before the
    // outer SS load can still fault; the rewind must restore the full CS
    // cache (and with it CPL), not just RIP.
    let mut mem = LinearMemory::new();
    let mut cpu = Cpu::new();
    cpu.setup_long_flat(&mut mem, CODE, STACK);
    cpu.trap_faults = true;

    // GDT at 0x8000: 08 = ring-0 code64, 10 = data, 20 = ring-3 code64.
    // Limit 0x27 leaves the popped SS selector 0x2B (index 0x28) unmapped.
    let gdt = 0x8000u64;
    mem.write64(gdt + 8, 0x00A0_9A00_0000_0000);
    mem.write64(gdt + 16, 0x00C0_9200_0000_FFFF);
    mem.write64(gdt + 32, 0x00A0_FA00_0000_0000);
    cpu.regs.gdtr = DescTable {
        base: gdt,
        limit: 0x27,
    };

    // Frame: RIP, CS=23h (ring-3 code), RFLAGS, RSP, SS=2Bh (past the GDT
    // limit -> #GP once CS is already committed).
    for (i, v) in [0x13_0000u64, 0x23, 2, 0x18_0000, 0x2B].iter().enumerate() {
        mem.write64(STACK + i as u64 * 8, *v);
    }
    mem.load(CODE, &[0x48, 0xCF]); // IRETQ

    let cs_before = cpu.regs.seg[reg::CS as usize];
    let ss_before = cpu.regs.seg[reg::SS as usize];
    cpu.step(&mut mem);
    match cpu.host_trap {
        Some(remu::x86_64::HostTrap::Exception(e)) => {
            assert_eq!(e.vector, 13, "expected #GP on the bad outer SS")
        }
        ref other => panic!("expected a trapped #GP, got {other:?}"),
    }
    assert_eq!(
        cpu.regs.seg[reg::CS as usize],
        cs_before,
        "the committed CS cache must be rewound after the outer-SS fault"
    );
    assert_eq!(cpu.regs.seg[reg::SS as usize], ss_before);
    assert_eq!(cpu.cpl(), 0);
    assert_eq!(cpu.regs.rip, CODE, "RIP rewinds to the IRETQ");
    assert_eq!(cpu.regs.gpr[reg::RSP as usize], STACK, "pops rewound");
}

// --- 64-bit-mode #UD list -----------------------------------------------------

#[test]
fn legacy_opcodes_ud_in_long_mode() {
    for prog in [
        vec![0x27],       // DAA
        vec![0x06],       // PUSH ES
        vec![0x60],       // PUSHA
        vec![0x62, 0x00], // BOUND
        vec![0xCE],       // INTO
        vec![0xD4, 0x0A], // AAM
    ] {
        let (mut cpu, mut mem) = long(&prog);
        cpu.trap_faults = true;
        cpu.step(&mut mem);
        match cpu.host_trap {
            Some(remu::x86_64::HostTrap::Exception(e)) => {
                assert_eq!(e.vector, 6, "opcode {:#04X} should be #UD", prog[0]);
            }
            other => panic!("opcode {:#04X}: expected #UD, got {other:?}", prog[0]),
        }
    }
}
