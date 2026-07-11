//! # Remu
//!
//! An emulation framework with cycle-conscious interpreters for the MOS 6502 /
//! 65xx CPU ([`cpu`]), the Intel 8086/8088 ([`x86`]), the Intel 80386
//! ([`x86_32`]) and x86-64/AMD64 ([`x86_64`]).
//!
//! The design keeps each CPU decoupled from memory and devices behind a bus
//! trait ([`bus::Bus`] for the 6502, per-core `Bus` traits for the x86
//! cores), so the same cores power unit tests, integration images, and
//! (eventually) full emulated machines.
//!
//! The [`usermode`] layer runs statically linked Linux i386 ELF executables
//! on the 80386 core, emulating their syscalls on the host (qemu-user
//! style) — see the `remu-user` binary. The [`os`] and [`os64`] layers are the
//! qiling-style OS-emulation engines (hardware paging, ring 3) for Linux i386
//! and x86-64 respectively — see `remu-user64` for the 64-bit runner.
//!
//! ```
//! use remu::{Cpu, memory::FlatMemory, bus::Bus};
//!
//! let mut mem = FlatMemory::new();
//! mem.load(0x0600, &[0xA9, 0x42]); // LDA #$42
//! mem.set_reset_vector(0x0600);
//!
//! let mut cpu = Cpu::new();
//! cpu.reset(&mut mem);
//! cpu.step(&mut mem);
//! assert_eq!(cpu.regs.a, 0x42);
//! ```

pub mod bus;
pub mod cpu;
pub mod device;
pub mod interrupt;
pub mod memory;
pub mod os;
pub mod os64;
#[cfg(feature = "python")]
mod python;
pub mod usermode;
pub mod x86;
pub mod x86_32;
pub mod x86_64;

pub use cpu::Cpu;
pub use cpu::registers::{Registers, Status};
