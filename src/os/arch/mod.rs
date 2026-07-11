//! Architecture-generic seam.
//!
//! [`TargetArch`] captures the CPU-specific facts the loader, syscall layer and
//! run loop need — the syscall ABI, the syscall trap vector, the ELF machine
//! type — so a future core plugs in without touching those layers. Only
//! [`i386::I386`] is implemented today; the trait is deliberately lean.

use crate::x86_32::Cpu;

pub mod i386;

pub use i386::I386;

/// The CPU-specific contract for the OS layer.
pub trait TargetArch {
    /// The `INT` vector used for syscalls (`0x80` on i386).
    const SYSCALL_INT: u8;
    /// The ELF `e_machine` value this arch loads (`EM_386`).
    const ELF_MACHINE: u16;

    /// The syscall number for the pending syscall (from `EAX` on i386).
    fn syscall_nr(cpu: &Cpu) -> u32;
    /// The `i`-th syscall argument (0-based). i386 order: EBX ECX EDX ESI EDI EBP.
    fn syscall_arg(cpu: &Cpu, i: usize) -> u32;
    /// Write the syscall return value (into `EAX`).
    fn set_syscall_ret(cpu: &mut Cpu, v: u32);
    /// The current instruction pointer (for diagnostics).
    fn pc(cpu: &Cpu) -> u32;
}
