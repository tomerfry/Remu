//! The i386 backend: syscall ABI, GDT construction, protected-mode ring-3 flat
//! entry with paging, and TLS descriptors for `set_thread_area`.

use super::TargetArch;
use crate::os::abi::gdt;
use crate::os::memory::{AddressSpace, PhysMem};
use crate::x86_32::{Cpu, DescTable, SegReg, cr0, reg};

/// The i386 target.
pub struct I386;

/// i386 syscall argument registers, in ABI order.
const ARG_REGS: [u8; 6] = [reg::EBX, reg::ECX, reg::EDX, reg::ESI, reg::EDI, reg::EBP];

impl TargetArch for I386 {
    const SYSCALL_INT: u8 = 0x80;
    const ELF_MACHINE: u16 = crate::os::abi::elf::EM_386;

    fn syscall_nr(cpu: &Cpu) -> u32 {
        cpu.regs.reg32(reg::EAX)
    }

    fn syscall_arg(cpu: &Cpu, i: usize) -> u32 {
        cpu.regs.reg32(ARG_REGS[i])
    }

    fn set_syscall_ret(cpu: &mut Cpu, v: u32) {
        cpu.regs.set_reg32(reg::EAX, v);
    }

    fn pc(cpu: &Cpu) -> u32 {
        cpu.regs.eip
    }
}

/// Encode an 8-byte GDT/LDT descriptor. `access` is the access byte (P/DPL/S
/// and type); `gran` is the flags nibble (G, D/B, L, AVL).
fn descriptor(base: u32, limit: u32, access: u8, gran: u8) -> [u8; 8] {
    [
        limit as u8,
        (limit >> 8) as u8,
        base as u8,
        (base >> 8) as u8,
        (base >> 16) as u8,
        access,
        ((gran & 0x0F) << 4) | ((limit >> 16) & 0x0F) as u8,
        (base >> 24) as u8,
    ]
}

/// A flat 4 GiB segment cache (base 0, limit 4 GiB, 32-bit) with the given
/// access byte — what the CPU actually uses per access.
fn flat_seg(sel: u16, access: u8) -> SegReg {
    SegReg {
        sel,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs: access as u16 | 0x0C00, // G=1, D/B=1
    }
}

/// Build the GDT in guest memory and enter protected mode: flat ring-3 segments,
/// paging on, `EIP`/`ESP` set, and the OS-emulation hooks armed. After this the
/// CPU is a Linux i386 user process about to execute at `entry`.
pub fn enter_user(
    cpu: &mut Cpu,
    aspace: &mut AddressSpace,
    mem: &mut PhysMem,
    entry: u32,
    esp: u32,
) {
    // Descriptors: null, ring-0 code/data, ring-3 code/data; TLS slots zeroed.
    let mut table = vec![0u8; gdt::ENTRIES as usize * 8];
    let put = |t: &mut [u8], idx: usize, d: [u8; 8]| t[idx * 8..idx * 8 + 8].copy_from_slice(&d);
    put(&mut table, 1, descriptor(0, 0xF_FFFF, 0x9A, 0xC)); // ring-0 code
    put(&mut table, 2, descriptor(0, 0xF_FFFF, 0x92, 0xC)); // ring-0 data
    put(&mut table, 3, descriptor(0, 0xF_FFFF, 0xFA, 0xC)); // ring-3 code
    put(&mut table, 4, descriptor(0, 0xF_FFFF, 0xF2, 0xC)); // ring-3 data

    aspace.map_kernel(mem, gdt::BASE, table.len() as u32);
    aspace.write_bytes(mem, gdt::BASE, &table);
    cpu.regs.gdtr = DescTable {
        base: gdt::BASE,
        limit: gdt::ENTRIES * 8 - 1,
    };

    // Protected mode + paging.
    cpu.regs.cr0 |= cr0::PE;
    cpu.regs.cr3 = aspace.cr3;
    cpu.regs.cr0 |= cr0::PG;

    // Flat ring-3 segment caches.
    cpu.regs.seg[reg::CS as usize] = flat_seg(gdt::USER_CS, 0xFA);
    for s in [reg::ES, reg::SS, reg::DS, reg::FS, reg::GS] {
        cpu.regs.seg[s as usize] = flat_seg(gdt::USER_DS, 0xF2);
    }

    cpu.regs.eip = entry;
    cpu.regs.set_reg32(reg::ESP, esp);

    // Arm the OS-emulation hooks (see the core `x86_32` changes).
    cpu.syscall_int = Some(I386::SYSCALL_INT);
    cpu.trap_faults = true;
    cpu.extensions = true;
}

/// Program a TLS GDT entry for `set_thread_area`. `entry` is the GDT index
/// (`gdt::TLS_MIN..`); writes a ring-3 data descriptor with the given base and
/// (page- or byte-granular) limit. Returns the selector the guest loads into
/// `%gs` (`(entry << 3) | 3`).
pub fn write_tls(
    aspace: &AddressSpace,
    mem: &mut PhysMem,
    entry: u16,
    base: u32,
    limit: u32,
    limit_in_pages: bool,
    writable: bool,
) -> u16 {
    let access = 0x80 | (3 << 5) | 0x10 | if writable { 0x02 } else { 0x00 }; // P|DPL3|S|data
    let gran = 0x4 | if limit_in_pages { 0x8 } else { 0x0 }; // D/B=1, G=limit_in_pages
    let d = descriptor(base, limit, access, gran);
    aspace.write_bytes(mem, gdt::BASE + entry as u32 * 8, &d);
    (entry << 3) | 3
}
