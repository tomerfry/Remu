//! The Intel 8086/8088 CPU core: 16-bit real mode, full documented instruction
//! set (plus the well-known undocumented aliases), instruction-atomic timing.
//!
//! This is the second architecture in the framework, deliberately self-contained
//! (own [`Bus`] trait, own register file) so promoting each core to its own crate
//! later stays mechanical. Later x86 generations (186/286/386) extend this core.
//!
//! ```
//! use remu::x86::{Cpu, Bus, LinearMemory};
//!
//! let mut mem = LinearMemory::new();
//! mem.load(0x0_0100, &[0xB8, 0x34, 0x12]); // MOV AX, 0x1234
//! let mut cpu = Cpu::new();
//! cpu.regs.cs = 0x0000;
//! cpu.regs.ip = 0x0100;
//! cpu.step(&mut mem);
//! assert_eq!(cpu.regs.ax, 0x1234);
//! ```

mod alu;
mod div;
mod execute;
mod modrm;
pub mod registers;

pub use registers::{Flags, Registers};

use registers::reg;

/// The 8086 memory and I/O bus.
///
/// Memory addresses are 20-bit physical (`segment * 16 + offset`, wrapped at
/// 1 MiB by the CPU before they reach the bus). The 8086 has a separate 64 KiB
/// port I/O space; the `io_*` methods default to open-bus reads and ignored
/// writes so memory-only machines implement just `read`/`write`.
pub trait Bus {
    /// Read one byte from physical address `addr` (already masked to 20 bits).
    fn read(&mut self, addr: u32) -> u8;

    /// Write `value` to physical address `addr`.
    fn write(&mut self, addr: u32, value: u8);

    /// Read one byte from I/O port `port`.
    fn io_read(&mut self, port: u16) -> u8 {
        let _ = port;
        0xFF
    }

    /// Write `value` to I/O port `port`.
    fn io_write(&mut self, port: u16, value: u8) {
        let _ = (port, value);
    }
}

/// A flat 1 MiB RAM covering the whole real-mode address space — the simplest
/// possible 8086 machine, for tests and raw programs.
pub struct LinearMemory {
    /// The full 20-bit physical address space.
    pub ram: Box<[u8; 0x10_0000]>,
}

impl LinearMemory {
    /// Create a zero-initialized 1 MiB memory.
    pub fn new() -> Self {
        LinearMemory { ram: vec![0u8; 0x10_0000].into_boxed_slice().try_into().unwrap() }
    }

    /// Load `data` into memory starting at physical address `addr`.
    pub fn load(&mut self, addr: u32, data: &[u8]) {
        for (i, &byte) in data.iter().enumerate() {
            self.ram[(addr as usize + i) & 0xF_FFFF] = byte;
        }
    }
}

impl Default for LinearMemory {
    fn default() -> Self {
        LinearMemory::new()
    }
}

impl Bus for LinearMemory {
    fn read(&mut self, addr: u32) -> u8 {
        self.ram[addr as usize & 0xF_FFFF]
    }

    fn write(&mut self, addr: u32, value: u8) {
        self.ram[addr as usize & 0xF_FFFF] = value;
    }
}

/// Cycles consumed by servicing a hardware interrupt.
const INTERRUPT_CYCLES: u32 = 61;

/// An 8086 processor.
///
/// As with the other cores, the CPU does not own its bus — call [`Cpu::step`]
/// with a `&mut B: Bus`.
#[derive(Debug, Clone)]
pub struct Cpu {
    /// The register file.
    pub regs: Registers,
    /// Total cycles elapsed since construction/reset. Timing is
    /// instruction-atomic using documented 8086 base timings plus the
    /// effective-address penalty; the prefetch queue is not modeled, so counts
    /// are approximate.
    pub cycles: u64,
    /// Set while the processor is stopped by `HLT` (an interrupt resumes it).
    pub halted: bool,

    /// Latched pending non-maskable interrupt (edge-triggered, vector 2).
    nmi_pending: bool,
    /// Pending maskable interrupt request with its vector (as supplied by a
    /// PIC during the INTA cycle). Level-style: stays pending until serviced
    /// or cleared via [`Cpu::clear_intr`].
    intr: Option<u8>,
    /// Interrupts (and traps) are inhibited for one instruction after
    /// `MOV SS`/`POP SS`/`STI`.
    inhibit_interrupts: bool,

    /// Per-instruction segment-override prefix (segment register index).
    seg_override: Option<u8>,
    /// Per-instruction repeat prefix: `true` for `REP`/`REPE` (F3),
    /// `false` for `REPNE` (F2).
    rep: Option<bool>,
    /// Effective-address cycle penalty of the last decoded ModRM operand
    /// (0 for register operands), consumed by the instruction handlers.
    ea_cycles: u32,
}

impl Cpu {
    /// Create a CPU in its power-on state: execution begins at `FFFF:0000`.
    pub fn new() -> Self {
        Cpu {
            regs: Registers::new(),
            cycles: 0,
            halted: false,
            nmi_pending: false,
            intr: None,
            inhibit_interrupts: false,
            seg_override: None,
            rep: None,
            ea_cycles: 0,
        }
    }

    /// Perform a RESET: flags cleared, `CS:IP = FFFF:0000`, all else zero.
    pub fn reset(&mut self) {
        self.regs = Registers::new();
        self.halted = false;
        self.nmi_pending = false;
        self.intr = None;
        self.inhibit_interrupts = false;
    }

    /// Execute one instruction (or service a pending interrupt). Returns the
    /// number of cycles consumed, and adds them to [`Cpu::cycles`].
    pub fn step<B: Bus>(&mut self, bus: &mut B) -> u32 {
        // Interrupts are recognized at instruction boundaries, except for the
        // one-instruction shadow after MOV SS / POP SS / STI.
        let inhibited = self.inhibit_interrupts;
        self.inhibit_interrupts = false;
        if !inhibited {
            if self.nmi_pending {
                self.nmi_pending = false;
                self.halted = false;
                self.interrupt(bus, 2);
                self.cycles += INTERRUPT_CYCLES as u64;
                return INTERRUPT_CYCLES;
            }
            if let Some(vector) = self.intr
                && self.regs.flags.contains(Flags::IF)
            {
                self.intr = None;
                self.halted = false;
                self.interrupt(bus, vector);
                self.cycles += INTERRUPT_CYCLES as u64;
                return INTERRUPT_CYCLES;
            }
        }

        if self.halted {
            self.cycles += 1;
            return 1;
        }

        // Trap flag: a single-step interrupt fires after this instruction if
        // TF was set when it started (and the instruction didn't clear it or
        // enter an interrupt handler, which clears TF).
        let trap = self.regs.flags.contains(Flags::TF);

        // Consume prefixes, then dispatch the opcode byte.
        self.seg_override = None;
        self.rep = None;
        let mut cycles = 0u32;
        let opcode = loop {
            let b = self.fetch_byte(bus);
            match b {
                0x26 => self.seg_override = Some(reg::ES),
                0x2E => self.seg_override = Some(reg::CS),
                0x36 => self.seg_override = Some(reg::SS),
                0x3E => self.seg_override = Some(reg::DS),
                0xF0 | 0xF1 => (), // LOCK (F1 is its undocumented alias)
                0xF2 => self.rep = Some(false),
                0xF3 => self.rep = Some(true),
                _ => break b,
            }
            cycles += 2;
        };
        cycles += self.dispatch(bus, opcode);

        if trap && self.regs.flags.contains(Flags::TF) && !self.inhibit_interrupts {
            self.interrupt(bus, 1);
            cycles += INTERRUPT_CYCLES;
        }

        self.cycles += cycles as u64;
        cycles
    }

    // --- Interrupt line control -------------------------------------------

    /// Latch a pending NMI (vector 2, not maskable by `IF`).
    pub fn trigger_nmi(&mut self) {
        self.nmi_pending = true;
    }

    /// Assert the INTR line with `vector` (as a PIC would supply during the
    /// interrupt-acknowledge cycle). Serviced at the next instruction boundary
    /// with `IF` set.
    pub fn assert_intr(&mut self, vector: u8) {
        self.intr = Some(vector);
    }

    /// Deassert the INTR line.
    pub fn clear_intr(&mut self) {
        self.intr = None;
    }

    // --- Memory access (the seam for future cycle-accurate stepping) -------

    /// `segment:offset` → 20-bit physical address (wrapping at 1 MiB, as on
    /// the 8086 where A20 does not exist).
    #[inline]
    fn phys(seg: u16, off: u16) -> u32 {
        (((seg as u32) << 4) + off as u32) & 0xF_FFFF
    }

    /// Read one byte at `seg:off`. All instruction memory accesses funnel
    /// through these helpers (mirroring the 6502 core) so sub-instruction
    /// timing can be added later in one place.
    #[inline]
    pub(crate) fn read8<B: Bus>(&mut self, bus: &mut B, seg: u16, off: u16) -> u8 {
        bus.read(Self::phys(seg, off))
    }

    /// Write one byte at `seg:off`.
    #[inline]
    pub(crate) fn write8<B: Bus>(&mut self, bus: &mut B, seg: u16, off: u16, value: u8) {
        bus.write(Self::phys(seg, off), value);
    }

    /// Read a little-endian word at `seg:off`. The 16-bit offset wraps within
    /// the segment (`seg:FFFF` + 1 → `seg:0000`), as on real hardware.
    #[inline]
    pub(crate) fn read16<B: Bus>(&mut self, bus: &mut B, seg: u16, off: u16) -> u16 {
        let lo = self.read8(bus, seg, off) as u16;
        let hi = self.read8(bus, seg, off.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    /// Write a little-endian word at `seg:off` (offset wrapping in-segment).
    #[inline]
    pub(crate) fn write16<B: Bus>(&mut self, bus: &mut B, seg: u16, off: u16, value: u16) {
        self.write8(bus, seg, off, value as u8);
        self.write8(bus, seg, off.wrapping_add(1), (value >> 8) as u8);
    }

    /// Fetch the byte at `CS:IP` and advance `IP`.
    #[inline]
    pub(crate) fn fetch_byte<B: Bus>(&mut self, bus: &mut B) -> u8 {
        let b = self.read8(bus, self.regs.cs, self.regs.ip);
        self.regs.ip = self.regs.ip.wrapping_add(1);
        b
    }

    /// Fetch a little-endian word at `CS:IP` and advance `IP` by two.
    #[inline]
    pub(crate) fn fetch_word<B: Bus>(&mut self, bus: &mut B) -> u16 {
        let lo = self.fetch_byte(bus) as u16;
        let hi = self.fetch_byte(bus) as u16;
        lo | (hi << 8)
    }

    /// The effective segment for a data access whose default segment register
    /// index is `default`, honoring any override prefix.
    #[inline]
    pub(crate) fn seg_or(&self, default: u8) -> u16 {
        self.regs.seg(self.seg_override.unwrap_or(default))
    }

    // --- Stack --------------------------------------------------------------

    /// Push a word onto the stack (`SS:SP`), pre-decrementing `SP`.
    #[inline]
    pub(crate) fn push16<B: Bus>(&mut self, bus: &mut B, value: u16) {
        self.regs.sp = self.regs.sp.wrapping_sub(2);
        self.write16(bus, self.regs.ss, self.regs.sp, value);
    }

    /// Pop a word off the stack, post-incrementing `SP`.
    #[inline]
    pub(crate) fn pop16<B: Bus>(&mut self, bus: &mut B) -> u16 {
        let v = self.read16(bus, self.regs.ss, self.regs.sp);
        self.regs.sp = self.regs.sp.wrapping_add(2);
        v
    }

    // --- Interrupts ----------------------------------------------------------

    /// Take interrupt `vector`: push FLAGS/CS/IP, clear `IF` and `TF`, and load
    /// `CS:IP` from the interrupt vector table at `0000:vector*4`. Used by
    /// hardware interrupts, `INT`/`INT3`/`INTO`, divide errors, and traps.
    pub(crate) fn interrupt<B: Bus>(&mut self, bus: &mut B, vector: u8) {
        self.push16(bus, self.regs.flags.to_word());
        self.regs.flags.remove(Flags::IF | Flags::TF);
        self.push16(bus, self.regs.cs);
        self.push16(bus, self.regs.ip);
        let base = vector as u16 * 4;
        self.regs.ip = self.read16(bus, 0, base);
        self.regs.cs = self.read16(bus, 0, base.wrapping_add(2));
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}
