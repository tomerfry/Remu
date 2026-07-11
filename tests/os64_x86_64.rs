//! End-to-end check of the x86-64 OS-emulation layer: synthesize a minimal
//! statically linked ELF64 that issues `write(1, msg, len)` and
//! `exit_group(0)` via raw `SYSCALL`, run it on the emulated core under long
//! mode + 4-level paging, and confirm the captured output and exit status.

use remu::os64::Emulator;
use remu::os64::fs::Fd;

const VBASE: u64 = 0x0040_0000;
const EHDR: usize = 64;
const PHDR: usize = 56;

fn put_u16(v: &mut [u8], off: usize, x: u16) {
    v[off..off + 2].copy_from_slice(&x.to_le_bytes());
}
fn put_u32(v: &mut [u8], off: usize, x: u32) {
    v[off..off + 4].copy_from_slice(&x.to_le_bytes());
}
fn put_u64(v: &mut [u8], off: usize, x: u64) {
    v[off..off + 8].copy_from_slice(&x.to_le_bytes());
}

/// Build a static `ET_EXEC` ELF64 whose single R+X `PT_LOAD` (mapped at
/// [`VBASE`]) contains the machine code and the message.
fn hello_elf(msg: &[u8]) -> Vec<u8> {
    let code_off = EHDR + PHDR;
    // Assemble the code once we know the message's virtual address.
    let msg_off = code_off + 36; // 36 = fixed length of the code below
    let msg_vaddr = VBASE + msg_off as u64;
    let len = msg.len() as u32;

    let mut code = Vec::new();
    code.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]); // mov edi, 1
    code.push(0x48);
    code.push(0xBE); // movabs rsi, imm64
    code.extend_from_slice(&msg_vaddr.to_le_bytes());
    code.push(0xBA); // mov edx, imm32
    code.extend_from_slice(&len.to_le_bytes());
    code.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]); // mov eax, 1 (write)
    code.extend_from_slice(&[0x0F, 0x05]); // syscall
    code.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi
    code.extend_from_slice(&[0xB8, 0xE7, 0x00, 0x00, 0x00]); // mov eax, 231 (exit_group)
    code.extend_from_slice(&[0x0F, 0x05]); // syscall
    assert_eq!(code.len(), 36, "code length must match the assumed msg offset");

    let filesz = (code_off + code.len() + msg.len()) as u64;
    let entry = VBASE + code_off as u64;

    let mut f = vec![0u8; code_off];
    // --- ELF header ---
    f[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
    f[4] = 2; // ELFCLASS64
    f[5] = 1; // little-endian
    f[6] = 1; // EV_CURRENT
    put_u16(&mut f, 16, 2); // ET_EXEC
    put_u16(&mut f, 18, 62); // EM_X86_64
    put_u32(&mut f, 20, 1); // e_version
    put_u64(&mut f, 24, entry);
    put_u64(&mut f, 32, EHDR as u64); // e_phoff
    put_u16(&mut f, 52, EHDR as u16); // e_ehsize
    put_u16(&mut f, 54, PHDR as u16); // e_phentsize
    put_u16(&mut f, 56, 1); // e_phnum
    // --- Program header (PT_LOAD, R+X) ---
    put_u32(&mut f, EHDR, 1); // p_type = PT_LOAD
    put_u32(&mut f, EHDR + 4, 5); // p_flags = R|X
    put_u64(&mut f, EHDR + 8, 0); // p_offset
    put_u64(&mut f, EHDR + 16, VBASE); // p_vaddr
    put_u64(&mut f, EHDR + 24, VBASE); // p_paddr
    put_u64(&mut f, EHDR + 32, filesz); // p_filesz
    put_u64(&mut f, EHDR + 40, filesz); // p_memsz
    put_u64(&mut f, EHDR + 48, 0x1000); // p_align

    f.extend_from_slice(&code);
    f.extend_from_slice(msg);
    f
}

#[test]
fn static_hello_world_runs() {
    let msg = b"hello, x86-64!\n";
    let elf = hello_elf(msg);

    let mut emu = Emulator::load_image(&elf, &["hello".into()], &[], None).expect("load");
    emu.vfs.install(1, Fd::Sink(Vec::new())); // capture stdout

    let code = emu.run_capped(10_000);
    assert_eq!(code, 0, "guest should exit_group(0)");
    assert_eq!(
        emu.vfs.sink_data(1).expect("fd 1 is a sink"),
        msg,
        "captured stdout must match the guest's write"
    );
}

#[test]
fn bad_write_buffer_returns_efault() {
    // Same shape, but point the write buffer at an unmapped address: the guest
    // still exits cleanly because write() returns -EFAULT (a syscall error, not
    // a CPU fault) and the program proceeds to exit_group(0). This confirms the
    // syscall marshalling's EFAULT path.
    let msg = b"unused";
    let mut elf = hello_elf(msg);
    // Rewrite the movabs immediate (code offset 7 = after `mov edi,1` and the
    // `48 be` movabs opcode) to a bogus, unmapped pointer.
    let imm_off = EHDR + PHDR + 7;
    elf[imm_off..imm_off + 8].copy_from_slice(&0x0000_1234_5678_9000u64.to_le_bytes());

    let mut emu = Emulator::load_image(&elf, &["hello".into()], &[], None).expect("load");
    emu.vfs.install(1, Fd::Sink(Vec::new()));
    let code = emu.run_capped(10_000);
    assert_eq!(code, 0, "guest exits normally; the bad write just returns EFAULT");
    assert!(emu.vfs.sink_data(1).unwrap().is_empty(), "nothing written");
}

#[test]
fn malformed_elf_huge_memsz_rejected() {
    // A crafted PT_LOAD with an absurd p_memsz must be rejected by the loader,
    // not overflow-panic or map an insane range. p_memsz is at phdr offset 40.
    let mut elf = hello_elf(b"x");
    let memsz_off = EHDR + 40;
    elf[memsz_off..memsz_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    let r = Emulator::load_image(&elf, &["x".into()], &[], None);
    assert!(r.is_err(), "segment escaping the address space must be rejected");
}
