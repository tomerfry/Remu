//! The memory/IO bus abstraction — the seam between the CPU and everything else.
//!
//! The 6502 sees a flat 16-bit address space. A [`Bus`] is whatever sits on the
//! other end of the address/data lines: in a unit test it's a sparse RAM map; in
//! a real machine it's an address decoder dispatching to RAM, ROM and
//! memory-mapped devices. The CPU is generic over `B: Bus`, so the common case
//! monomorphizes and inlines, while the machine layer can still hold a
//! `Box<dyn Bus>` (the trait is object-safe).

/// A device that responds to CPU reads and writes across the 16-bit address space.
///
/// Both `read` and `write` take `&mut self`: real reads have side effects (reading
/// a hardware register can clear a flag or advance a FIFO), so the bus must be
/// able to mutate on read. Both are infallible — a 6502 read always yields a byte;
/// an unmapped ("open bus") address returns garbage, it does not fail.
pub trait Bus {
    /// Read one byte from `addr`.
    fn read(&mut self, addr: u16) -> u8;

    /// Write `value` to `addr`.
    fn write(&mut self, addr: u16, value: u8);

    /// Read a little-endian 16-bit word from `addr` and `addr + 1`.
    ///
    /// Convenience for *non-quirky* fetches only — interrupt/reset vectors. The
    /// 6502's quirky 16-bit reads (zero-page wraparound, the indirect-JMP page
    /// bug) are handled in the addressing-mode resolver, not here.
    fn read_u16(&mut self, addr: u16) -> u16 {
        let lo = self.read(addr) as u16;
        let hi = self.read(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }
}
