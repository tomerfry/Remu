//! End-to-end tests for the user-mode layer, driven by tiny ELF images built
//! entirely in Rust (no cross toolchain needed).

use remu::usermode::elf::{self, LoadError};
use remu::usermode::linux::{self, Control, abi, fs::Fd};
use remu::usermode::{AddressSpace, Exit, UserArch, Usermode, addr_space};
use remu::x86_32::Cpu;

/// Base address mirroring a classic i386 `ld` layout.
const BASE: u32 = 0x0804_8000;
/// Code starts right after the 52-byte ehdr + one 32-byte phdr.
const CODE_OFF: u32 = 84;

/// Build a minimal valid ELF32 `ET_EXEC` image: one `PT_LOAD` mapping the
/// whole file at `base` (so code lands at `base + 84`, which is also the
/// entry point) plus `bss` extra zeroed bytes.
fn tiny_elf(code: &[u8], base: u32, bss: u32) -> Vec<u8> {
    let filesz = CODE_OFF + code.len() as u32;
    let mut v = Vec::new();
    // --- ehdr (52 bytes) ---
    v.extend_from_slice(b"\x7fELF");
    v.extend_from_slice(&[1, 1, 1, 0]); // ELFCLASS32, LSB, EV_CURRENT, SysV
    v.extend_from_slice(&[0; 8]); // padding
    v.extend_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
    v.extend_from_slice(&elf::EM_386.to_le_bytes()); // e_machine
    v.extend_from_slice(&1u32.to_le_bytes()); // e_version
    v.extend_from_slice(&(base + CODE_OFF).to_le_bytes()); // e_entry
    v.extend_from_slice(&52u32.to_le_bytes()); // e_phoff
    v.extend_from_slice(&0u32.to_le_bytes()); // e_shoff
    v.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    v.extend_from_slice(&52u16.to_le_bytes()); // e_ehsize
    v.extend_from_slice(&32u16.to_le_bytes()); // e_phentsize
    v.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
    v.extend_from_slice(&40u16.to_le_bytes()); // e_shentsize
    v.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    v.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
    assert_eq!(v.len(), 52);
    // --- phdr (32 bytes) ---
    v.extend_from_slice(&elf::PT_LOAD.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // p_offset
    v.extend_from_slice(&base.to_le_bytes()); // p_vaddr
    v.extend_from_slice(&base.to_le_bytes()); // p_paddr
    v.extend_from_slice(&filesz.to_le_bytes());
    v.extend_from_slice(&(filesz + bss).to_le_bytes()); // p_memsz
    v.extend_from_slice(&5u32.to_le_bytes()); // p_flags = R+X
    v.extend_from_slice(&0x1000u32.to_le_bytes()); // p_align
    assert_eq!(v.len(), CODE_OFF as usize);
    v.extend_from_slice(code);
    v
}

// --- ELF parsing / loading ---------------------------------------------------

#[test]
fn elf_parse_accepts_a_minimal_image() {
    let img = tiny_elf(&[0x90, 0xF4], BASE, 0x100);
    let e = elf::parse(&img).unwrap();
    assert_eq!(e.entry, BASE + CODE_OFF);
    assert_eq!(e.machine, elf::EM_386);
    assert_eq!(e.phnum, 1);
    assert_eq!(e.phdrs[0].memsz, e.phdrs[0].filesz + 0x100);
}

#[test]
fn elf_parse_rejects_bad_images() {
    let img = tiny_elf(&[0x90], BASE, 0);

    let mut bad = img.clone();
    bad[0] = b'M';
    assert_eq!(elf::parse(&bad).unwrap_err(), LoadError::NotElf);

    let mut bad = img.clone();
    bad[4] = 2; // ELFCLASS64
    assert_eq!(elf::parse(&bad).unwrap_err(), LoadError::Not32Bit);

    let mut bad = img.clone();
    bad[5] = 2; // big-endian
    assert_eq!(elf::parse(&bad).unwrap_err(), LoadError::NotLittleEndian);

    let mut bad = img.clone();
    bad[16] = 3; // ET_DYN
    assert_eq!(elf::parse(&bad).unwrap_err(), LoadError::NotExec);

    assert_eq!(elf::parse(&img[..40]).unwrap_err(), LoadError::Truncated);

    let mut bad = img.clone();
    bad[52 + 16] = 0xFF; // p_filesz beyond the file
    assert_eq!(elf::parse(&bad).unwrap_err(), LoadError::Truncated);

    let mut bad = img.clone();
    bad[52] = 0; // p_type = PT_NULL
    assert_eq!(elf::parse(&bad).unwrap_err(), LoadError::NoLoadSegments);
}

#[test]
fn elf_load_maps_segments_and_reports_layout() {
    let code = [0xB8, 0x2A, 0x00, 0x00, 0x00]; // MOV EAX, 42
    let img = tiny_elf(&code, BASE, 0x100);
    let e = elf::parse(&img).unwrap();
    let mut mem = AddressSpace::new();
    let (entry, brk_start, phdr_vaddr) = elf::load(&e, &img, &mut mem);

    assert_eq!(entry, BASE + CODE_OFF);
    let mut got = [0u8; 5];
    mem.read_bytes(entry, &mut got).unwrap();
    assert_eq!(got, code, "code bytes are at the entry point");
    assert_eq!(mem.read8(BASE), 0x7F, "the whole file maps at BASE");
    let file_end = BASE + CODE_OFF + code.len() as u32;
    assert_eq!(mem.read8(file_end), 0, "BSS is zero-filled");
    assert_eq!(
        brk_start,
        addr_space::align_up(file_end + 0x100, 4096).unwrap()
    );
    assert_eq!(phdr_vaddr, Some(BASE + 52), "phdrs are inside the PT_LOAD");
}

// --- x86-32 adapter ------------------------------------------------------------

#[test]
fn adapter_runs_flat_ring3_code_to_the_syscall_trap() {
    let code = [
        0xB8, 0x2A, 0x00, 0x00, 0x00, // MOV EAX, 42
        0xBB, 0x07, 0x00, 0x00, 0x00, // MOV EBX, 7
        0xCD, 0x80, // INT 80h
    ];
    let img = tiny_elf(&code, BASE, 0);
    let e = elf::parse(&img).unwrap();
    let mut mem = AddressSpace::new();
    let (entry, _, _) = elf::load(&e, &img, &mut mem);
    mem.map(addr_space::STACK_TOP - 0x1000, 0x1000);

    let mut cpu = <Cpu as UserArch>::new();
    cpu.setup(&mut mem, entry, addr_space::STACK_TOP - 16);
    assert_eq!(cpu.cpl(), 3, "user code runs at ring 3");

    let mut trapped = false;
    for _ in 0..16 {
        UserArch::step(&mut cpu, &mut mem);
        assert!(!cpu.dead(), "CPU died:\n{}", UserArch::dump(&cpu));
        if cpu.take_syscall() {
            trapped = true;
            break;
        }
    }
    assert!(trapped, "INT 80h reached the trap latch");
    assert_eq!(cpu.syscall_number(), 42);
    assert_eq!(cpu.syscall_args()[0], 7);
    assert_eq!(cpu.pc(), entry + code.len() as u32, "PC is past the INT");
    assert!(mem.take_segv().is_none());
}

// --- Initial process stack -----------------------------------------------------

#[test]
fn initial_stack_follows_the_i386_sysv_layout() {
    let img = tiny_elf(&[0x90], BASE, 0);
    let mut um: Usermode<Cpu> =
        Usermode::load(&img, &["prog", "arg1"], &["PATH=/bin", "TERM=dumb"]).unwrap();
    let esp = um.cpu.regs.gpr[4]; // ESP
    assert_eq!(esp % 16, 0, "ESP is 16-byte aligned");

    let mem = &mut um.mem;
    assert_eq!(mem.read_u32(esp), 2, "argc");
    let argv0 = mem.read_u32(esp + 4);
    let argv1 = mem.read_u32(esp + 8);
    assert_eq!(mem.read_cstr(argv0).unwrap(), b"prog");
    assert_eq!(mem.read_cstr(argv1).unwrap(), b"arg1");
    assert_eq!(mem.read_u32(esp + 12), 0, "argv terminator");
    let env0 = mem.read_u32(esp + 16);
    let env1 = mem.read_u32(esp + 20);
    assert_eq!(mem.read_cstr(env0).unwrap(), b"PATH=/bin");
    assert_eq!(mem.read_cstr(env1).unwrap(), b"TERM=dumb");
    assert_eq!(mem.read_u32(esp + 24), 0, "envp terminator");

    // Walk the auxv and collect the tags this loader promises.
    let mut aux = std::collections::HashMap::new();
    let mut p = esp + 28;
    loop {
        let (tag, val) = (mem.read_u32(p), mem.read_u32(p + 4));
        aux.insert(tag, val);
        p += 8;
        if tag == abi::AT_NULL {
            break;
        }
        assert!(p < addr_space::STACK_TOP, "unterminated auxv");
    }
    assert_eq!(aux[&abi::AT_PAGESZ], 4096);
    assert_eq!(aux[&abi::AT_ENTRY], BASE + CODE_OFF);
    assert_eq!(aux[&abi::AT_PHDR], BASE + 52, "phdrs mapped by PT_LOAD");
    assert_eq!(aux[&abi::AT_PHENT], 32);
    assert_eq!(aux[&abi::AT_PHNUM], 1);
    assert_eq!(aux[&abi::AT_UID], abi::GUEST_ID);
    let rand = aux[&abi::AT_RANDOM];
    let mut r = [0u8; 16];
    mem.read_bytes(rand, &mut r).unwrap();

    // brk starts at the page-aligned image end.
    assert_eq!(um.mem.brk, um.mem.brk_base);
    assert!(um.mem.brk > BASE + CODE_OFF);
}

#[test]
fn load_rejects_a_wrong_machine_elf() {
    let mut img = tiny_elf(&[0x90], BASE, 0);
    img[18] = 40; // EM_ARM
    let Err(e) = Usermode::<Cpu>::load(&img, &["prog"], &[]) else {
        panic!("load must reject EM_ARM");
    };
    assert_eq!(e, LoadError::WrongMachine(40));
}

// --- Full runner ---------------------------------------------------------------

fn run_elf(code: &[u8], bss: u32) -> (Usermode<Cpu>, Exit) {
    let img = tiny_elf(code, BASE, bss);
    let mut um: Usermode<Cpu> = Usermode::load(&img, &["prog"], &[]).unwrap();
    um.process.fds.install(1, Fd::Sink(Vec::new()));
    let exit = um.run();
    (um, exit)
}

#[test]
fn guest_exit_status_is_reported() {
    let code = [
        0xB8, 0x01, 0x00, 0x00, 0x00, // MOV EAX, 1 (exit)
        0xBB, 0x2A, 0x00, 0x00, 0x00, // MOV EBX, 42
        0xCD, 0x80, // INT 80h
    ];
    let (_, exit) = run_elf(&code, 0);
    assert!(matches!(exit, Exit::Exited(42)), "got {exit:?}");
}

#[test]
fn guest_write_reaches_the_fd_table() {
    let msg_addr = BASE + CODE_OFF + 31;
    let mut code = vec![
        0xB8, 0x04, 0x00, 0x00, 0x00, // MOV EAX, 4 (write)
        0xBB, 0x01, 0x00, 0x00, 0x00, // MOV EBX, 1 (stdout)
        0xB9, 0, 0, 0, 0, // MOV ECX, msg (patched below)
        0xBA, 0x06, 0x00, 0x00, 0x00, // MOV EDX, 6
        0xCD, 0x80, // INT 80h
        0xB8, 0xFC, 0x00, 0x00, 0x00, // MOV EAX, 252 (exit_group)
        0x31, 0xDB, // XOR EBX, EBX
        0xCD, 0x80, // INT 80h
    ];
    code[11..15].copy_from_slice(&msg_addr.to_le_bytes());
    code.extend_from_slice(b"hello\n");

    let (um, exit) = run_elf(&code, 0);
    assert!(matches!(exit, Exit::Exited(0)), "got {exit:?}");
    assert_eq!(um.process.fds.sink_data(1).unwrap(), b"hello\n");
}

#[test]
fn guest_segfault_produces_a_fault_report() {
    let code = [0xA1, 0x00, 0x00, 0x00, 0x00]; // MOV EAX, [0]
    let (_, exit) = run_elf(&code, 0);
    let Exit::Fault(msg) = exit else {
        panic!("expected a fault, got {exit:?}")
    };
    assert!(msg.contains("segmentation fault"), "{msg}");
    assert!(msg.contains("read of unmapped address 0x00000000"), "{msg}");
    assert!(
        msg.contains("EIP="),
        "the report carries a register dump: {msg}"
    );
}

#[test]
fn tls_via_set_thread_area_and_gs_roundtrips() {
    // set_thread_area(entry=-1, base=tls, limit=0xFFFFF pages, 32-bit), load
    // GS with the returned selector, then read back through GS:0 a marker
    // written through DS. Exits 0 iff the values match.
    let tls = BASE + CODE_OFF + 200; // inside the BSS
    let mut code = vec![
        0x83, 0xEC, 0x10, // SUB ESP, 16
        0xC7, 0x04, 0x24, 0xFF, 0xFF, 0xFF, 0xFF, // MOV [ESP], -1 (entry_number)
        0xC7, 0x44, 0x24, 0x04, 0, 0, 0, 0, // MOV [ESP+4], tls (patched)
        0xC7, 0x44, 0x24, 0x08, 0xFF, 0xFF, 0x0F, 0x00, // MOV [ESP+8], 0xFFFFF
        0xC7, 0x44, 0x24, 0x0C, 0x51, 0x00, 0x00, 0x00, // MOV [ESP+12], flags
        0xB8, 0xF3, 0x00, 0x00, 0x00, // MOV EAX, 243 (set_thread_area)
        0x89, 0xE3, // MOV EBX, ESP
        0xCD, 0x80, // INT 80h
        0x8B, 0x0C, 0x24, // MOV ECX, [ESP] (chosen entry)
        0x8D, 0x0C, 0xCD, 0x03, 0x00, 0x00, 0x00, // LEA ECX, [ECX*8+3]
        0x8E, 0xE9, // MOV GS, CX
        0xC7, 0x05, 0, 0, 0, 0, 0xEF, 0xBE, 0x37, 0x13, // MOV [tls], 1337BEEFh
        0x65, 0xA1, 0x00, 0x00, 0x00, 0x00, // MOV EAX, GS:[0]
        0x2D, 0xEF, 0xBE, 0x37, 0x13, // SUB EAX, 1337BEEFh
        0x89, 0xC3, // MOV EBX, EAX
        0xB8, 0xFC, 0x00, 0x00, 0x00, // MOV EAX, 252 (exit_group)
        0xCD, 0x80, // INT 80h
    ];
    code[14..18].copy_from_slice(&tls.to_le_bytes());
    let patch = code.iter().position(|&b| b == 0x05).unwrap(); // C7 05 <tls>
    code[patch + 1..patch + 5].copy_from_slice(&tls.to_le_bytes());

    let (um, exit) = run_elf(&code, 0x400);
    assert!(
        matches!(exit, Exit::Exited(0)),
        "GS:0 read the TLS marker back — got {exit:?}\n{}",
        UserArch::dump(&um.cpu)
    );
}

// --- Acceptance: a real toolchain-built binary ---------------------------------

#[test]
fn nolibc_acceptance_binary_runs() {
    // Built from tests/data/hello.c by a real gcc (see tests/data/README.md):
    // prints via write, echoes argv, allocates with brk, exits with argc.
    let bin = include_bytes!("data/hello-nolibc");
    let mut um: Usermode<Cpu> =
        Usermode::load(bin, &["hello-nolibc", "world"], &["TERM=dumb"]).unwrap();
    um.process.fds.install(1, Fd::Sink(Vec::new()));
    let exit = um.run();
    assert!(
        matches!(exit, Exit::Exited(2)),
        "exit status is argc — got {exit:?}"
    );
    let out = String::from_utf8_lossy(um.process.fds.sink_data(1).unwrap()).into_owned();
    assert!(out.contains("hello via write"), "{out}");
    assert!(out.contains("arg: hello-nolibc"), "{out}");
    assert!(out.contains("arg: world"), "{out}");
    assert!(out.contains("heap: brk works"), "{out}");
}

#[test]
fn dispatch_services_memory_syscalls() {
    let mut cpu = <Cpu as UserArch>::new();
    let mut mem = AddressSpace::new();
    let mut process = linux::Process::new();
    mem.brk_base = 0x0805_0000;
    mem.brk = 0x0805_0000;

    let call = |cpu: &mut Cpu,
                mem: &mut AddressSpace,
                process: &mut linux::Process,
                nr: u32,
                args: [u32; 6]| {
        match linux::dispatch(cpu, mem, process, nr, args) {
            Control::Ret(v) => v,
            Control::Exit(c) => panic!("unexpected exit({c})"),
        }
    };

    // brk: query, grow, and use the new memory.
    assert_eq!(
        call(&mut cpu, &mut mem, &mut process, abi::NR_BRK, [0; 6]),
        0x0805_0000
    );
    let r = call(
        &mut cpu,
        &mut mem,
        &mut process,
        abi::NR_BRK,
        [0x0805_3000, 0, 0, 0, 0, 0],
    );
    assert_eq!(r, 0x0805_3000);
    mem.write_u32(0x0805_2FFC, 0xAA55_AA55);
    assert_eq!(mem.read_u32(0x0805_2FFC), 0xAA55_AA55);

    // mmap2 anonymous, then munmap.
    let flags = abi::MAP_ANONYMOUS | 0x02; // MAP_PRIVATE
    let a = call(
        &mut cpu,
        &mut mem,
        &mut process,
        abi::NR_MMAP2,
        [0, 0x2000, 3, flags, u32::MAX, 0],
    );
    assert!((a as i32) > 0, "mmap2 returned {a:#x}");
    mem.write_u32(a, 42);
    assert_eq!(
        call(
            &mut cpu,
            &mut mem,
            &mut process,
            abi::NR_MUNMAP,
            [a, 0x2000, 0, 0, 0, 0]
        ),
        0
    );

    // File-backed mappings are refused in v1.
    let r = call(
        &mut cpu,
        &mut mem,
        &mut process,
        abi::NR_MMAP2,
        [0, 0x1000, 3, 0x02, 5, 0],
    );
    assert_eq!(r as i32, -(abi::ENODEV as i32));

    // Unknown syscall.
    let r = call(&mut cpu, &mut mem, &mut process, 9999, [0; 6]);
    assert_eq!(r as i32, -(abi::ENOSYS as i32));
}
