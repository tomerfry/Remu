//! Integration tests for the Linux i386 OS-emulation layer.
//!
//! These use tiny, hand-assembled static ELF binaries that talk to the kernel
//! only through `int 0x80`, so they run in CI with no cross toolchain. Richer
//! tests against a real rootfs are gated on `REMU_LINUX_ROOTFS` (see below).

use remu::os::Emulator;

/// Assemble a minimal static `ET_EXEC` ELF32 whose entry point runs `code`,
/// with `data` appended right after it (its base address passed back so the
/// caller can reference it). Returns the ELF bytes.
fn build_elf(mut code_with_dataref: impl FnMut(u32) -> (Vec<u8>, Vec<u8>)) -> Vec<u8> {
    let base: u32 = 0x0804_8000;
    let code_off: u32 = 52 + 32; // ELF header + one program header

    // First pass with a dummy data address to learn the code length, then a
    // second pass with the real address (code length is stable).
    let (code0, _) = code_with_dataref(0);
    let data_addr = base + code_off + code0.len() as u32;
    let (code, data) = code_with_dataref(data_addr);
    assert_eq!(code.len(), code0.len(), "code length must not depend on the data address");

    let entry = base + code_off;
    let total = code_off + code.len() as u32 + data.len() as u32;

    let mut f = Vec::new();
    f.extend_from_slice(&[0x7F, b'E', b'L', b'F', 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    f.extend_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
    f.extend_from_slice(&3u16.to_le_bytes()); // e_machine = EM_386
    f.extend_from_slice(&1u32.to_le_bytes()); // e_version
    f.extend_from_slice(&entry.to_le_bytes());
    f.extend_from_slice(&52u32.to_le_bytes()); // e_phoff
    f.extend_from_slice(&0u32.to_le_bytes()); // e_shoff
    f.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    f.extend_from_slice(&52u16.to_le_bytes()); // e_ehsize
    f.extend_from_slice(&32u16.to_le_bytes()); // e_phentsize
    f.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
    f.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
    f.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    f.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
    assert_eq!(f.len(), 52);

    f.extend_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
    f.extend_from_slice(&0u32.to_le_bytes()); // p_offset
    f.extend_from_slice(&base.to_le_bytes()); // p_vaddr
    f.extend_from_slice(&base.to_le_bytes()); // p_paddr
    f.extend_from_slice(&total.to_le_bytes()); // p_filesz
    f.extend_from_slice(&total.to_le_bytes()); // p_memsz
    f.extend_from_slice(&7u32.to_le_bytes()); // p_flags = R|W|X
    f.extend_from_slice(&0x1000u32.to_le_bytes()); // p_align
    assert_eq!(f.len(), code_off as usize);

    f.extend_from_slice(&code);
    f.extend_from_slice(&data);
    assert_eq!(f.len() as u32, total);
    f
}

/// `mov r32, imm32` (opcode B8+reg).
fn mov_imm(reg: u8, imm: u32) -> Vec<u8> {
    let mut v = vec![0xB8 + reg];
    v.extend_from_slice(&imm.to_le_bytes());
    v
}

const EAX: u8 = 0;
const ECX: u8 = 1;
const EDX: u8 = 2;
const EBX: u8 = 3;

fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// `write(1, msg, len); exit_group(code)`.
fn hello_program(msg: &[u8], exit_code: u32) -> Vec<u8> {
    let m = msg.to_vec();
    build_elf(|data_addr| {
        let mut code = Vec::new();
        code.extend(mov_imm(EAX, 4)); // __NR_write
        code.extend(mov_imm(EBX, 1)); // fd = stdout
        code.extend(mov_imm(ECX, data_addr)); // buf
        code.extend(mov_imm(EDX, m.len() as u32)); // count
        code.extend([0xCD, 0x80]); // int 0x80
        code.extend(mov_imm(EAX, 252)); // __NR_exit_group
        code.extend(mov_imm(EBX, exit_code)); // status
        code.extend([0xCD, 0x80]); // int 0x80
        (code, m.clone())
    })
}

#[test]
fn static_write_and_exit() {
    let elf = hello_program(b"hi from guest\n", 0);
    let path = write_temp("remu_hello_static.elf", &elf);

    let mut emu = Emulator::load(&path, &["/hello".into()], &[], None).unwrap();
    emu.vfs.capture_output();
    let code = emu.run_capped(100_000);

    assert_eq!(emu.vfs.captured(), b"hi from guest\n");
    assert_eq!(code, 0);
}

#[test]
fn exit_code_is_propagated() {
    let elf = hello_program(b"", 42);
    let path = write_temp("remu_exit42.elf", &elf);

    let mut emu = Emulator::load(&path, &["/x".into()], &[], None).unwrap();
    emu.vfs.capture_output();
    let code = emu.run_capped(100_000);
    assert_eq!(code, 42);
}

#[test]
fn checked_in_fixture_runs() {
    // Exercise the deterministic fixture committed under tests/fixtures.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("hello_static.elf");
    let mut emu = Emulator::load(&path, &["/hello".into()], &[], None).unwrap();
    emu.vfs.capture_output();
    let code = emu.run_capped(100_000);
    assert_eq!(emu.vfs.captured(), b"hi from guest\n");
    assert_eq!(code, 0);
}

#[test]
fn reads_argc_from_the_stack() {
    // The initial stack must hold argc at [esp]: exit with it.
    let elf = build_elf(|_data| {
        let mut code = Vec::new();
        code.extend([0x8B, 0x04, 0x24]); // mov eax, [esp]  (argc)
        code.extend([0x89, 0xC3]); // mov ebx, eax
        code.extend(mov_imm(EAX, 252)); // exit_group(argc)
        code.extend([0xCD, 0x80]);
        (code, Vec::new())
    });
    let path = write_temp("remu_argc.elf", &elf);
    let argv = vec!["/p".to_string(), "a".to_string(), "b".to_string()];
    let mut emu = Emulator::load(&path, &argv, &[], None).unwrap();
    let code = emu.run_capped(100_000);
    assert_eq!(code, 3); // argc == 3
}

#[test]
fn stack_is_usable() {
    // push "i\n" onto the stack, then write(1, esp, 2) and exit.
    let elf = build_elf(|_data| {
        let mut code = Vec::new();
        code.extend(mov_imm(EAX, 0x0A69)); // "i\n" in the low 16 bits
        code.push(0x50); // push eax
        code.extend(mov_imm(EAX, 4)); // __NR_write
        code.extend(mov_imm(EBX, 1)); // fd = stdout
        code.extend([0x89, 0xE1]); // mov ecx, esp  (buffer = top of stack)
        code.extend(mov_imm(EDX, 2)); // count = 2
        code.extend([0xCD, 0x80]); // int 0x80
        code.extend(mov_imm(EAX, 252)); // exit_group
        code.extend(mov_imm(EBX, 0));
        code.extend([0xCD, 0x80]);
        (code, Vec::new())
    });
    let path = write_temp("remu_stack.elf", &elf);
    let mut emu = Emulator::load(&path, &["/stack".into()], &[], None).unwrap();
    emu.vfs.capture_output();
    let code = emu.run_capped(100_000);
    assert_eq!(emu.vfs.captured(), b"i\n");
    assert_eq!(code, 0);
}

#[test]
fn stack_grows_on_demand() {
    // Touch an address below the initial stack VMA (but within the growth
    // limit): the #PF handler must map it and the instruction must restart.
    let target: u32 = 0xBFFB_0000;
    let elf = build_elf(|_data| {
        let mut code = Vec::new();
        code.extend(mov_imm(ECX, target));
        code.extend(mov_imm(EAX, 0x42));
        code.extend([0x89, 0x01]); // mov [ecx], eax  → fault → grow → retry
        code.extend([0x8B, 0x19]); // mov ebx, [ecx]  → reads back 0x42
        code.extend(mov_imm(EAX, 252)); // exit_group(ebx)
        code.extend([0xCD, 0x80]);
        (code, Vec::new())
    });
    let path = write_temp("remu_stackgrow.elf", &elf);
    let mut emu = Emulator::load(&path, &["/g".into()], &[], None).unwrap();
    emu.vfs.capture_output();
    let code = emu.run_capped(100_000);
    assert_eq!(code, 0x42);
}

#[test]
fn extension_opcodes_run() {
    // BSWAP (a post-386 opcode) is enabled via the OS layer's `extensions`.
    let elf = build_elf(|_data| {
        let mut code = Vec::new();
        code.extend(mov_imm(EAX, 0x7856_3412));
        code.extend([0x0F, 0xC8]); // bswap eax  → 0x12345678
        code.extend([0x89, 0xC3]); // mov ebx, eax
        code.extend(mov_imm(EAX, 252)); // exit_group(ebx) → 0x78 = 120
        code.extend([0xCD, 0x80]);
        (code, Vec::new())
    });
    let path = write_temp("remu_ext.elf", &elf);
    let mut emu = Emulator::load(&path, &["/e".into()], &[], None).unwrap();
    let code = emu.run_capped(100_000);
    assert_eq!(code, 0x78);
}

#[test]
fn brk_allocates_heap() {
    // brk(0) then brk(old+0x2000); write a byte into the new heap and read back.
    let elf = build_elf(|_data| {
        let mut code = Vec::new();
        code.extend(mov_imm(EAX, 45)); // __NR_brk
        code.extend(mov_imm(EBX, 0)); // query current break
        code.extend([0xCD, 0x80]); // int 0x80 → eax = brk
        code.extend([0x89, 0xC6]); // mov esi, eax  (save old break)
        code.extend([0x8D, 0x98, 0x00, 0x20, 0x00, 0x00]); // lea ebx, [eax+0x2000]
        code.extend(mov_imm(EAX, 45)); // brk(old+0x2000)
        code.extend([0xCD, 0x80]);
        // store 0x37 at old break, read it back into ebx
        code.extend(mov_imm(EAX, 0x37));
        code.extend([0x88, 0x06]); // mov [esi], al
        code.extend([0x0F, 0xB6, 0x1E]); // movzx ebx, byte [esi]
        code.extend(mov_imm(EAX, 252)); // exit_group(ebx) → 0x37 = 55
        code.extend([0xCD, 0x80]);
        (code, Vec::new())
    });
    let path = write_temp("remu_brk.elf", &elf);
    let mut emu = Emulator::load(&path, &["/b".into()], &[], None).unwrap();
    let code = emu.run_capped(100_000);
    assert_eq!(code, 0x37);
}

/// Run a real binary from an i386 rootfs, if one is provided.
///
/// Set `REMU_LINUX_ROOTFS` to an i386 root filesystem directory and
/// `REMU_LINUX_TEST_BIN` to a guest path within it (e.g. `/bin/busybox`); the
/// test then loads and runs it, expecting a clean (0) exit. When the vars are
/// unset the test prints a notice and passes, so CI stays green without a
/// rootfs (mirrors the SingleStepTests harness idiom).
#[test]
fn rootfs_binary_runs() {
    let (Ok(rootfs), Ok(bin)) = (
        std::env::var("REMU_LINUX_ROOTFS"),
        std::env::var("REMU_LINUX_TEST_BIN"),
    ) else {
        eprintln!("REMU_LINUX_ROOTFS / REMU_LINUX_TEST_BIN unset — skipping rootfs test");
        return;
    };
    let root = std::path::PathBuf::from(&rootfs);
    let host_bin = root.join(bin.trim_start_matches('/'));
    let mut emu = Emulator::load(&host_bin, std::slice::from_ref(&bin), &[], Some(root)).unwrap();
    emu.trace = true;
    let code = emu.run_capped(50_000_000);
    assert_eq!(code, 0, "guest {bin} exited with {code}");
}

/// A wild write to an unmapped address must terminate with SIGSEGV (128+11).
#[test]
fn wild_write_faults_to_sigsegv() {
    let elf = build_elf(|_data| {
        let mut code = Vec::new();
        // mov eax, [0x00000000]  →  #PF (null page unmapped)  → SIGSEGV
        code.extend(mov_imm(ECX, 0));
        code.extend([0x8B, 0x01]); // mov eax, [ecx]
        code.extend(mov_imm(EAX, 252));
        code.extend(mov_imm(EBX, 0));
        code.extend([0xCD, 0x80]);
        (code, Vec::new())
    });
    let path = write_temp("remu_segv.elf", &elf);
    let mut emu = Emulator::load(&path, &["/segv".into()], &[], None).unwrap();
    emu.vfs.capture_output();
    let code = emu.run_capped(100_000);
    assert_eq!(code, 139, "expected SIGSEGV exit (128+11)");
}
