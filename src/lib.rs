//! # Remu
//!
//! An emulation framework, starting with a cycle-conscious interpreter for the
//! MOS 6502 / 65xx CPU.
//!
//! The design keeps the CPU decoupled from memory and devices behind the
//! [`Bus`](bus::Bus) trait, so the same core powers unit tests, integration
//! images, and (eventually) full emulated machines.
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

pub use cpu::Cpu;
pub use cpu::registers::{Registers, Status};
