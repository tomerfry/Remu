//! The 6502 CPU core: state, the fetch/decode/execute loop, and the low-level
//! memory/stack/interrupt plumbing the instruction handlers build on.

pub mod addressing;
pub mod disasm;
mod execute;
pub mod opcodes;
pub mod registers;

use crate::bus::Bus;
use crate::interrupt::{INTERRUPT_CYCLES, IRQ_VECTOR, NMI_VECTOR, RESET_VECTOR};
use opcodes::OPCODES;
use registers::{Registers, Status};

/// A 6502 processor.
///
/// The CPU does not own its bus — call [`Cpu::step`] with a `&mut B: Bus`. This
/// keeps the CPU as one bus master among potentially several (DMA, other chips),
/// with the machine owning the bus and lending it out.
#[derive(Debug, Clone)]
pub struct Cpu {
    /// The register file.
    pub regs: Registers,
    /// Total cycles elapsed since construction/reset.
    pub cycles: u64,
    /// Set when a `KIL`/jam opcode has halted the processor.
    pub halted: bool,

    /// Latched pending non-maskable interrupt (edge-triggered).
    nmi_pending: bool,
    /// Previous NMI line level, for edge detection.
    prev_nmi: bool,
    /// Current IRQ line level (level-triggered).
    irq_line: bool,
}

impl Cpu {
    /// Create a CPU in its power-on register state. Call [`Cpu::reset`] to load
    /// the reset vector before running.
    pub fn new() -> Self {
        Cpu {
            regs: Registers::new(),
            cycles: 0,
            halted: false,
            nmi_pending: false,
            prev_nmi: false,
            irq_line: false,
        }
    }

    /// Perform a RESET: load `PC` from the reset vector, set the interrupt-disable
    /// flag, and reset the stack pointer. Takes 7 cycles.
    pub fn reset<B: Bus>(&mut self, bus: &mut B) {
        self.regs.pc = bus.read_u16(RESET_VECTOR);
        self.regs.sp = 0xFD;
        self.regs.p.insert(Status::I | Status::U);
        self.halted = false;
        self.nmi_pending = false;
        self.cycles += INTERRUPT_CYCLES as u64;
    }

    /// Execute one instruction (or service a pending interrupt). Returns the
    /// number of cycles consumed, and adds them to [`Cpu::cycles`].
    pub fn step<B: Bus>(&mut self, bus: &mut B) -> u8 {
        if self.halted {
            return 0;
        }

        // Interrupts are polled at instruction boundaries: NMI takes priority,
        // then IRQ if unmasked.
        if self.nmi_pending {
            self.nmi_pending = false;
            self.service_interrupt(bus, NMI_VECTOR, false);
            self.cycles += INTERRUPT_CYCLES as u64;
            return INTERRUPT_CYCLES;
        }
        if self.irq_line && !self.regs.p.contains(Status::I) {
            self.service_interrupt(bus, IRQ_VECTOR, false);
            self.cycles += INTERRUPT_CYCLES as u64;
            return INTERRUPT_CYCLES;
        }

        let opcode = self.fetch_byte(bus);
        let info = OPCODES[opcode as usize];

        // JSR interleaves operand fetch with the stack pushes, so it can't use the
        // generic Absolute resolver: it fetches the target low byte, pushes the
        // return address, and only *then* fetches the high byte. If the operand
        // lives in the stack page, the push overwrites the high byte before it's
        // read — a real hardware quirk the Tom Harte suite exercises.
        if let opcodes::Operation::JSR = info.operation {
            let lo = self.fetch_byte(bus) as u16; // PC now points at the high byte
            let ret = self.regs.pc; // address of the last JSR byte (RTS adds 1)
            self.push(bus, (ret >> 8) as u8);
            self.push(bus, ret as u8);
            let hi = self.read(bus, self.regs.pc) as u16;
            self.regs.pc = (hi << 8) | lo;
            self.cycles += info.cycles as u64;
            return info.cycles;
        }

        let operand = self.resolve(bus, info.mode);

        let extra = self.execute(bus, &info, &operand);

        let mut cycles = info.cycles;
        if info.page_penalty && operand.page_crossed {
            cycles += 1;
        }
        cycles += extra;

        self.cycles += cycles as u64;
        cycles
    }

    // --- Interrupt line control -------------------------------------------

    /// Update the NMI input line, latching a pending NMI on a high→low edge.
    pub fn set_nmi(&mut self, level: bool) {
        if level && !self.prev_nmi {
            self.nmi_pending = true;
        }
        self.prev_nmi = level;
    }

    /// Directly latch a pending NMI (convenience for `set_nmi(true)` edges).
    pub fn trigger_nmi(&mut self) {
        self.nmi_pending = true;
    }

    /// Set the IRQ input line level. Serviced between instructions while asserted
    /// and the `I` flag is clear.
    pub fn set_irq(&mut self, level: bool) {
        self.irq_line = level;
    }

    // --- Memory access (the seam for future cycle-accurate stepping) -------

    /// Read one byte through the bus. All instruction memory accesses funnel
    /// through here so sub-instruction cycle accuracy can be added later by
    /// filling in [`Cpu::tick_devices`].
    #[inline]
    pub(crate) fn read<B: Bus>(&mut self, bus: &mut B, addr: u16) -> u8 {
        self.tick_devices();
        bus.read(addr)
    }

    /// Write one byte through the bus.
    #[inline]
    pub(crate) fn write<B: Bus>(&mut self, bus: &mut B, addr: u16, value: u8) {
        self.tick_devices();
        bus.write(addr, value);
    }

    /// Per-access hook for advancing devices in lock-step with the CPU. A no-op in
    /// the current instruction-atomic (tier-2) cycle model; the place to add
    /// tier-3 sub-instruction timing later.
    #[inline]
    fn tick_devices(&mut self) {}

    /// Fetch the byte at `PC` and advance `PC`.
    #[inline]
    pub(crate) fn fetch_byte<B: Bus>(&mut self, bus: &mut B) -> u8 {
        let b = self.read(bus, self.regs.pc);
        self.regs.pc = self.regs.pc.wrapping_add(1);
        b
    }

    /// Fetch a little-endian word at `PC` and advance `PC` by two.
    #[inline]
    pub(crate) fn fetch_word<B: Bus>(&mut self, bus: &mut B) -> u16 {
        let lo = self.fetch_byte(bus) as u16;
        let hi = self.fetch_byte(bus) as u16;
        lo | (hi << 8)
    }

    // --- Stack -------------------------------------------------------------

    /// Push a byte onto the stack (`$0100 + SP`), decrementing `SP`.
    pub(crate) fn push<B: Bus>(&mut self, bus: &mut B, value: u8) {
        self.write(bus, 0x0100 | self.regs.sp as u16, value);
        self.regs.sp = self.regs.sp.wrapping_sub(1);
    }

    /// Pull a byte off the stack, incrementing `SP`.
    pub(crate) fn pull<B: Bus>(&mut self, bus: &mut B) -> u8 {
        self.regs.sp = self.regs.sp.wrapping_add(1);
        self.read(bus, 0x0100 | self.regs.sp as u16)
    }

    /// Push the status register. `with_b` selects the `B` bit value: set for
    /// `PHP`/`BRK`, clear for hardware IRQ/NMI. The `U` bit is always pushed set.
    pub(crate) fn push_status<B: Bus>(&mut self, bus: &mut B, with_b: bool) {
        let mut bits = self.regs.p.bits() | Status::U.bits();
        if with_b {
            bits |= Status::B.bits();
        } else {
            bits &= !Status::B.bits();
        }
        self.push(bus, bits);
    }

    /// Pull the status register (`PLP`/`RTI`). Bits 4 (`B`) and 5 (`U`) from the
    /// stack are discarded: `U` is forced set and `B` cleared in the in-memory
    /// representation.
    pub(crate) fn pull_status<B: Bus>(&mut self, bus: &mut B) {
        let v = self.pull(bus);
        self.regs.p = Status::from_bits_retain((v | Status::U.bits()) & !Status::B.bits());
    }

    /// Push `PC` and `P` and vector through `vector`, setting the `I` flag. Used
    /// for hardware IRQ/NMI (`with_b == false`).
    fn service_interrupt<B: Bus>(&mut self, bus: &mut B, vector: u16, with_b: bool) {
        let pc = self.regs.pc;
        self.push(bus, (pc >> 8) as u8);
        self.push(bus, pc as u8);
        self.push_status(bus, with_b);
        self.regs.p.insert(Status::I);
        self.regs.pc = bus.read_u16(vector);
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}
