//! Initial process stack construction: the `argc` / `argv` / `envp` / auxiliary
//! vector image the x86-64 System V ABI hands to `_start` (and that `ld.so`
//! bootstraps dynamic linking from). The stack region must already be mapped.
//!
//! Everything is 64-bit here — pointers and auxv tag/value slots are 8 bytes —
//! and the entry `RSP` is 16-byte aligned, as the kernel guarantees.

use crate::os64::abi::auxv;
use crate::os64::loader::ElfImage;
use crate::os64::memory::{AddressSpace, PAGE_SIZE, PhysMem, STACK_TOP};

/// 16 fixed bytes for `AT_RANDOM` (determinism aids testing; true entropy is
/// not required for correctness).
const RANDOM16: [u8; 16] = [
    0x9e, 0x37, 0x79, 0xb9, 0x7f, 0x4a, 0x7c, 0x15, 0xf3, 0x9c, 0xc0, 0x60, 0x5c, 0xed, 0xc8, 0x34,
];

/// Push `bytes` below `sp`, returning the address they were written to.
fn push(aspace: &AddressSpace, mem: &mut PhysMem, sp: &mut u64, bytes: &[u8]) -> u64 {
    *sp -= bytes.len() as u64;
    aspace.write_bytes(mem, *sp, bytes);
    *sp
}

/// Build the initial stack and return the entry `RSP` (pointing at `argc`).
///
/// `interp_base` is `Some` for a dynamically-linked program (the `ld.so` load
/// address, for `AT_BASE`), `None` for a static one. `exec_path` is the guest
/// path used for `AT_EXECFN` and `argv[0]` fallback.
pub fn build_stack(
    aspace: &AddressSpace,
    mem: &mut PhysMem,
    image: &ElfImage,
    interp_base: Option<u64>,
    argv: &[Vec<u8>],
    envp: &[Vec<u8>],
    exec_path: &[u8],
) -> u64 {
    let mut sp = STACK_TOP;

    // --- String data (order is irrelevant; pointers reference it) -----------
    let mut argv_ptrs = Vec::with_capacity(argv.len());
    for a in argv {
        let mut s = a.clone();
        s.push(0);
        argv_ptrs.push(push(aspace, mem, &mut sp, &s));
    }
    let mut envp_ptrs = Vec::with_capacity(envp.len());
    for e in envp {
        let mut s = e.clone();
        s.push(0);
        envp_ptrs.push(push(aspace, mem, &mut sp, &s));
    }
    let platform = push(aspace, mem, &mut sp, b"x86_64\0");
    let mut execfn_bytes = exec_path.to_vec();
    execfn_bytes.push(0);
    let execfn = push(aspace, mem, &mut sp, &execfn_bytes);
    let random = push(aspace, mem, &mut sp, &RANDOM16);

    // --- Auxiliary vector ---------------------------------------------------
    let aux: Vec<(u64, u64)> = vec![
        (auxv::AT_PHDR, image.phdr),
        (auxv::AT_PHENT, image.phent as u64),
        (auxv::AT_PHNUM, image.phnum as u64),
        (auxv::AT_PAGESZ, PAGE_SIZE),
        (auxv::AT_BASE, interp_base.unwrap_or(0)),
        (auxv::AT_FLAGS, 0),
        (auxv::AT_ENTRY, image.entry),
        (auxv::AT_UID, 0),
        (auxv::AT_EUID, 0),
        (auxv::AT_GID, 0),
        (auxv::AT_EGID, 0),
        (auxv::AT_SECURE, 0),
        (auxv::AT_CLKTCK, 100),
        (auxv::AT_HWCAP, 0),
        (auxv::AT_RANDOM, random),
        (auxv::AT_PLATFORM, platform),
        (auxv::AT_EXECFN, execfn),
        (auxv::AT_NULL, 0),
    ];

    // --- Pointer block: argc, argv[], 0, envp[], 0, auxv[] ------------------
    // The block is 8-byte words; aligning its base to 16 keeps the entry RSP
    // 16-byte aligned as the ABI requires.
    let words = 1 + (argv_ptrs.len() + 1) + (envp_ptrs.len() + 1) + aux.len() * 2;
    let need = words as u64 * 8;
    let base = (sp - need) & !0xF;

    let mut p = base;
    let w = |mem: &mut PhysMem, p: &mut u64, v: u64| {
        aspace.write_bytes(mem, *p, &v.to_le_bytes());
        *p += 8;
    };
    w(mem, &mut p, argv_ptrs.len() as u64);
    for &ptr in &argv_ptrs {
        w(mem, &mut p, ptr);
    }
    w(mem, &mut p, 0);
    for &ptr in &envp_ptrs {
        w(mem, &mut p, ptr);
    }
    w(mem, &mut p, 0);
    for &(t, v) in &aux {
        w(mem, &mut p, t);
        w(mem, &mut p, v);
    }

    base
}
