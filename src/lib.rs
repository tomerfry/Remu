//! # Remu
//!
//! An emulation framework with cycle-conscious interpreters for the MOS 6502 /
//! 65xx CPU ([`cpu`]) and the Intel 8086/8088 ([`x86`]).
//!
//! The design keeps each CPU decoupled from memory and devices behind a bus
//! trait ([`bus::Bus`] for the 6502, [`x86::Bus`] for the 8086), so the same
//! cores power unit tests, integration images, and (eventually) full emulated
//! machines.
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
pub mod interrupt;
pub mod memory;
pub mod x86;
#[cfg(feature = "python")]
mod python;

pub use cpu::Cpu;
pub use cpu::registers::{Registers, Status};
