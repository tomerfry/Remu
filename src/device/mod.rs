//! Memory-mapped peripherals and the address decoder that dispatches to them.
//!
//! A [`Device`] is a peripheral occupying a small window of the address space
//! (typically a handful of registers). [`SystemBus`] implements [`Bus`] by
//! routing each access to the device whose window contains the address,
//! falling back to flat RAM — the "real machine" address decoder described in
//! [`crate::bus`].

use crate::bus::Bus;
use crate::memory::FlatMemory;

pub mod uart;

/// A memory-mapped peripheral.
///
/// Accesses arrive as offsets relative to the device's base address, so a
/// device never needs to know where it is mapped. Accesses are single bytes:
/// the 6502 issues every wider read as individual byte cycles, so there is no
/// wider transaction for a device to see.
///
/// Like [`Bus`], `read` takes `&mut self` — reading a hardware register can
/// have side effects (e.g. reading a UART receive buffer consumes the byte) —
/// and both methods are infallible.
///
/// Devices have no interrupt output yet. The planned extension is a defaulted
/// `fn irq_asserted(&self) -> bool { false }` on this trait, with the run loop
/// mirroring the OR of all device lines into
/// [`Cpu::set_irq`](crate::Cpu::set_irq) at each instruction boundary; the
/// read/write path stays as it is.
pub trait Device {
    /// Read the register at `offset` (relative to the device's base address).
    fn read(&mut self, offset: u16) -> u8;

    /// Write `value` to the register at `offset`.
    fn write(&mut self, offset: u16, value: u8);
}

/// One device mapped at `start`, responding to `len` consecutive addresses.
struct Mapping {
    start: u16,
    len: u16,
    device: Box<dyn Device>,
}

/// An address decoder: flat RAM with memory-mapped devices overlaid on top.
///
/// Accesses that hit a mapped window go to that device (as an offset from its
/// base); everything else falls through to [`FlatMemory`]. Windows are checked
/// in mapping order and the first match wins, so callers should not map
/// overlapping ranges.
pub struct SystemBus {
    /// Backing RAM covering the full address space. Public so program images
    /// can be loaded directly into RAM (bypassing device windows), e.g. via
    /// [`FlatMemory::load`] and [`FlatMemory::set_reset_vector`].
    pub ram: FlatMemory,
    mappings: Vec<Mapping>,
}

impl SystemBus {
    /// Create a bus with zeroed RAM and no devices mapped.
    pub fn new() -> Self {
        SystemBus {
            ram: FlatMemory::new(),
            mappings: Vec::new(),
        }
    }

    /// Map `device` at `base`, occupying `len` consecutive addresses.
    ///
    /// # Panics
    ///
    /// Panics if the window is empty or extends past the top of the address
    /// space (device windows do not wrap).
    pub fn map(&mut self, base: u16, len: u16, device: Box<dyn Device>) {
        assert!(
            len > 0 && base.checked_add(len - 1).is_some(),
            "device window ${base:04X}+{len} is empty or extends past $FFFF"
        );
        self.mappings.push(Mapping {
            start: base,
            len,
            device,
        });
    }
}

impl Default for SystemBus {
    fn default() -> Self {
        SystemBus::new()
    }
}

impl Bus for SystemBus {
    fn read(&mut self, addr: u16) -> u8 {
        for m in &mut self.mappings {
            // Windows never wrap, so a wrapped (underflowed) offset is >= len.
            let offset = addr.wrapping_sub(m.start);
            if offset < m.len {
                return m.device.read(offset);
            }
        }
        self.ram.read(addr)
    }

    fn write(&mut self, addr: u16, value: u8) {
        for m in &mut self.mappings {
            let offset = addr.wrapping_sub(m.start);
            if offset < m.len {
                return m.device.write(offset, value);
            }
        }
        self.ram.write(addr, value)
    }
}
