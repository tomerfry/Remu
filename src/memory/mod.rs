//! Concrete [`Bus`] implementations for running and testing the CPU.

use crate::bus::Bus;

/// A flat 64 KiB RAM-backed address space — the simplest possible machine, useful
/// for running raw 6502 programs and integration test images.
pub struct FlatMemory {
    /// The full 16-bit address space.
    pub ram: Box<[u8; 0x10000]>,
}

impl FlatMemory {
    /// Create a zero-initialized 64 KiB memory.
    pub fn new() -> Self {
        FlatMemory {
            ram: Box::new([0u8; 0x10000]),
        }
    }

    /// Load `data` into memory starting at `addr` (wrapping at the top).
    pub fn load(&mut self, addr: u16, data: &[u8]) {
        for (i, &byte) in data.iter().enumerate() {
            self.ram[addr.wrapping_add(i as u16) as usize] = byte;
        }
    }

    /// Set the reset vector at `$FFFC`/`$FFFD` to `addr`.
    pub fn set_reset_vector(&mut self, addr: u16) {
        self.ram[0xFFFC] = addr as u8;
        self.ram[0xFFFD] = (addr >> 8) as u8;
    }
}

impl Default for FlatMemory {
    fn default() -> Self {
        FlatMemory::new()
    }
}

impl Bus for FlatMemory {
    fn read(&mut self, addr: u16) -> u8 {
        self.ram[addr as usize]
    }

    fn write(&mut self, addr: u16, value: u8) {
        self.ram[addr as usize] = value;
    }
}
