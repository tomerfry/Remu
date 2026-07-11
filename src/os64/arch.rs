//! Dropping the x86-64 core into a Linux user process: flat ring-3 long mode
//! with paging, the SYSCALL/fault host hooks armed, and `RIP`/`RSP` set.
//!
//! Unlike the i386 backend there is no GDT in guest memory: 64-bit code ignores
//! segment bases (except FS/GS) and glibc installs its thread pointer through
//! `arch_prctl(ARCH_SET_FS)`, which writes the FS base directly — so loading the
//! descriptor caches by hand, as the CPU would after a segment load, is enough.

use crate::os64::memory::AddressSpace;
use crate::x86_64::{Cpu, DescTable, RFlags, SegReg, cr0, cr4, efer, reg};

/// Ring-3 64-bit code selector (Linux `__USER_CS`), purely cosmetic here since
/// we load the cache directly; `cpl()` reads the cached DPL, not the selector.
pub const USER_CS: u16 = 0x33;
/// Ring-3 data selector (Linux `__USER_DS`).
pub const USER_DS: u16 = 0x2B;

/// Cached access rights for a present ring-3 64-bit code segment
/// (P, DPL=3, S, code, exec/read, accessed; flags nibble G=1, L=1).
const CODE_ATTRS: u16 = 0x0AFB;
/// Cached access rights for a present ring-3 data segment
/// (P, DPL=3, S, data, read/write, accessed; flags nibble G=1, D/B=1).
const DATA_ATTRS: u16 = 0x0CF3;

/// A flat segment cache (base 0, 4 GiB limit — unused in 64-bit mode) with the
/// given selector and cached attributes.
fn flat(sel: u16, attrs: u16) -> SegReg {
    SegReg {
        sel,
        base: 0,
        limit: 0xFFFF_FFFF,
        attrs,
    }
}

/// Enter 64-bit long mode at CPL 3: turn on PAE + paging, arm long mode, point
/// `CR3` at the process page tables, load flat ring-3 segment caches, set
/// `RIP`/`RSP`, and enable the OS-emulation hooks. After this the CPU is a
/// Linux x86-64 user process about to execute at `entry`.
pub fn enter_user(cpu: &mut Cpu, aspace: &AddressSpace, entry: u64, rsp: u64) {
    // Paging + long mode. SYSCALL is trapped to the host, so SCE only needs to
    // be set for the instruction not to #UD; STAR/LSTAR are irrelevant.
    cpu.regs.cr4 |= cr4::PAE;
    cpu.regs.cr0 |= cr0::PE | cr0::PG;
    cpu.regs.cr0 &= !(cr0::CD | cr0::NW);
    cpu.regs.msr.efer |= efer::LME | efer::LMA | efer::SCE;
    cpu.regs.cr3 = aspace.cr3;

    // Flat ring-3 segments. FS/GS bases start at 0; glibc sets FS via
    // arch_prctl(ARCH_SET_FS).
    cpu.regs.seg[reg::CS as usize] = flat(USER_CS, CODE_ATTRS);
    for s in [reg::SS, reg::DS, reg::ES, reg::FS, reg::GS] {
        cpu.regs.seg[s as usize] = flat(USER_DS, DATA_ATTRS);
    }

    // No guest kernel behind the descriptor tables: exceptions and syscalls go
    // to the host, so empty GDTR/IDTR are fine (any stray IDT delivery triple
    // faults cleanly instead of reading guest memory).
    cpu.regs.gdtr = DescTable { base: 0, limit: 0 };
    cpu.regs.idtr = DescTable { base: 0, limit: 0 };

    cpu.regs.rip = entry;
    cpu.regs.gpr[reg::RSP as usize] = rsp;
    cpu.regs.rflags = RFlags::IF;

    cpu.trap_syscall = true;
    cpu.trap_faults = true;
    cpu.invalidate_tlb();
}
