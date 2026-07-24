//! The ARM32 CPU core: an ARM7TDMI-class ARMv4T — the full ARM and Thumb
//! instruction sets, the seven processor modes with banked registers, and
//! the five exception vectors with IRQ/FIQ lines.
//!
//! Like the other cores, it is deliberately self-contained (own [`Bus`]
//! trait, own register file) so promoting each core to its own crate later
//! stays mechanical. There is no MMU or coprocessor — exactly like an
//! ARM7TDMI with nothing on its coprocessor bus, every CDP/LDC/STC/MRC/MCR
//! encoding takes the undefined-instruction trap. Memory is little-endian.
//!
//! Where the architecture leaves behavior UNPREDICTABLE the core follows
//! ARM7TDMI silicon as pinned by the SingleStepTests suite: unaligned LDR
//! rotates the addressed word, LDRH/LDRSH degrade at odd addresses,
//! STR/STM of R15 store `pc + 12`, LDM/STM with an empty list transfer R15
//! and step the base by 0x40, ARM-state PC writes keep their raw low bits
//! (the fetch aligns instead; Thumb drops bit 0), the multiplies and SWP
//! read R15 as `pc + 12`, and MRS reads back whatever raw PSR bits MSR
//! deposited. Validated against all 2.25 million cases of the
//! [SingleStepTests ARM7TDMI](https://github.com/SingleStepTests/ARM7TDMI)
//! suite (see `tests/harte_arm7tdmi.rs`), with one knowing divergence: the
//! C flag (and V for the long forms) is left unchanged by the S-bit
//! multiplies, where hardware corrupts it via the Booth multiplier
//! internals.
//!
//! At an instruction boundary `regs.gpr[15]` is the address of the next
//! instruction to execute; the +8/+4 pipeline offset ARM code observes
//! when it reads R15 is applied by the executor.
//!
//! ```
//! use remu::arm32::{Cpu, LinearMemory};
//!
//! let mut mem = LinearMemory::new();
//! mem.load(0, &0xE3A0_0042u32.to_le_bytes()); // MOV r0, #0x42
//! let mut cpu = Cpu::new();                   // reset: PC = 0, Supervisor
//! cpu.step(&mut mem);
//! assert_eq!(cpu.regs.gpr[0], 0x42);
//! ```

pub(crate) mod alu;
mod decode;
mod execute;
mod icache;
pub mod registers;

pub use registers::{Mode, Psr, Registers, psr, reg};

/// The memory bus.
///
/// Addresses are 32-bit physical (no MMU). The wide accessors have
/// byte-composed defaults so simple devices only implement `read`/`write`;
/// RAM-backed buses should override them for speed. The CPU aligns every
/// wide access itself (the ARM rotation quirks live in the core), so
/// overrides may assume alignment and therefore contiguity.
pub trait Bus {
    /// Read one byte from `addr`.
    fn read(&mut self, addr: u32) -> u8;

    /// Write `value` to `addr`.
    fn write(&mut self, addr: u32, value: u8);

    /// Read a little-endian halfword from `addr`.
    #[inline]
    fn read16(&mut self, addr: u32) -> u16 {
        self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
    }

    /// Read a little-endian word from `addr`.
    #[inline]
    fn read32(&mut self, addr: u32) -> u32 {
        self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
    }

    /// Write a little-endian halfword to `addr`.
    #[inline]
    fn write16(&mut self, addr: u32, value: u16) {
        self.write(addr, value as u8);
        self.write(addr.wrapping_add(1), (value >> 8) as u8);
    }

    /// Write a little-endian word to `addr`.
    #[inline]
    fn write32(&mut self, addr: u32, value: u32) {
        self.write16(addr, value as u16);
        self.write16(addr.wrapping_add(2), (value >> 16) as u16);
    }
}

/// Default [`LinearMemory`] size (16 MiB).
const DEFAULT_MEMORY: usize = 16 << 20;

/// A flat power-of-two RAM — the simplest possible machine, for tests and
/// raw programs. Addresses wrap at the memory size.
pub struct LinearMemory {
    /// Backing RAM (`len` is a power of two).
    pub ram: Box<[u8]>,
    mask: usize,
}

impl LinearMemory {
    /// Create a zero-initialized 16 MiB memory.
    pub fn new() -> Self {
        Self::with_size(DEFAULT_MEMORY)
    }

    /// Create a zero-initialized memory of `size` bytes (rounded up to a
    /// power of two, minimum 64 KiB).
    pub fn with_size(size: usize) -> Self {
        let size = size.max(1 << 16).next_power_of_two();
        LinearMemory {
            ram: vec![0u8; size].into_boxed_slice(),
            mask: size - 1,
        }
    }

    /// Load `data` into memory starting at `addr`.
    pub fn load(&mut self, addr: u32, data: &[u8]) {
        for (i, &byte) in data.iter().enumerate() {
            self.ram[(addr as usize + i) & self.mask] = byte;
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
        self.ram[addr as usize & self.mask]
    }

    fn write(&mut self, addr: u32, value: u8) {
        self.ram[addr as usize & self.mask] = value;
    }

    fn read16(&mut self, addr: u32) -> u16 {
        let a = addr as usize & self.mask;
        if a < self.mask {
            u16::from_le_bytes(self.ram[a..a + 2].try_into().unwrap())
        } else {
            self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
        }
    }

    fn read32(&mut self, addr: u32) -> u32 {
        let a = addr as usize & self.mask;
        if a + 3 <= self.mask {
            u32::from_le_bytes(self.ram[a..a + 4].try_into().unwrap())
        } else {
            self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
        }
    }

    fn write16(&mut self, addr: u32, value: u16) {
        let a = addr as usize & self.mask;
        if a < self.mask {
            self.ram[a..a + 2].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write(addr, value as u8);
            self.write(addr.wrapping_add(1), (value >> 8) as u8);
        }
    }

    fn write32(&mut self, addr: u32, value: u32) {
        let a = addr as usize & self.mask;
        if a + 3 <= self.mask {
            self.ram[a..a + 4].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write16(addr, value as u16);
            self.write16(addr.wrapping_add(2), (value >> 16) as u16);
        }
    }
}

/// A CPU exception. The vectors live at `0x00..=0x1C`; RESET is
/// [`Cpu::reset`], not a deliverable event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exception {
    /// Undefined instruction (vector 0x04) — also every coprocessor
    /// encoding on this coprocessor-less core.
    Undefined,
    /// Software interrupt (vector 0x08).
    Swi,
    /// Prefetch abort (vector 0x0C). Never raised by this core (no MMU);
    /// deliverable by an embedder via [`Cpu::raise`].
    PrefetchAbort,
    /// Data abort (vector 0x10). Never raised by this core (no MMU).
    DataAbort,
    /// Normal interrupt (vector 0x18), from the IRQ line.
    Irq,
    /// Fast interrupt (vector 0x1C), from the FIQ line.
    Fiq,
}

impl Exception {
    /// The exception vector address.
    pub fn vector(self) -> u32 {
        match self {
            Exception::Undefined => 0x04,
            Exception::Swi => 0x08,
            Exception::PrefetchAbort => 0x0C,
            Exception::DataAbort => 0x10,
            Exception::Irq => 0x18,
            Exception::Fiq => 0x1C,
        }
    }

    /// The mode the exception is taken in.
    pub fn mode(self) -> Mode {
        match self {
            Exception::Undefined => Mode::Und,
            Exception::Swi => Mode::Svc,
            Exception::PrefetchAbort | Exception::DataAbort => Mode::Abt,
            Exception::Irq => Mode::Irq,
            Exception::Fiq => Mode::Fiq,
        }
    }
}

/// Result type for anything that can raise a CPU exception.
pub(crate) type Exec<T> = Result<T, Exception>;

/// A request from the CPU for the host (an OS-emulation layer) to act, in
/// place of the normal in-guest delivery.
///
/// Populated in [`Cpu::host_trap`] only when the corresponding opt-in is
/// set ([`Cpu::trap_swi`] / [`Cpu::trap_faults`]); the CPU is otherwise
/// plain bare metal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTrap {
    /// The guest executed `SWI` while [`Cpu::trap_swi`] was set. The PC
    /// already points past the instruction; read the syscall number from
    /// `r7` (EABI) or the SWI comment field and write the result to `r0`.
    Syscall,
    /// The guest raised a CPU exception while [`Cpu::trap_faults`] was
    /// set. The PC has been rewound to the faulting instruction, so the
    /// host may fix the fault and resume, or terminate the process.
    Exception(Exception),
}

/// Why [`Cpu::run`] stopped before retiring all `n` requested step-units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive] // a JIT backend may add exit reasons
pub enum RunExit {
    /// All `n` units retired.
    Completed,
    /// [`Cpu::host_trap`] is set (SWI or trapped fault). Service and
    /// `take()` it, then call [`Cpu::run`] again — no guest instruction
    /// has executed past the trap.
    HostTrap,
    /// The CPU is halted with no wake event pending. Assert an interrupt
    /// (or give up), then call [`Cpu::run`] again. Unlike [`Cpu::step`],
    /// which burns one idle cycle per call while halted, [`Cpu::run`]
    /// returns instead of idling.
    Halted,
}

/// The outcome of [`Cpu::run`]: how much ran, and why it stopped.
#[derive(Debug, Clone, Copy)]
pub struct RunResult {
    /// [`Cpu::step`]-equivalents retired: instruction executions and
    /// interrupt deliveries each count one, exactly as one `step()` call
    /// would.
    pub executed: u64,
    /// Why the run ended.
    pub exit: RunExit,
}

/// Cycles consumed by an exception entry (2S + 1N, collapsed).
const EXCEPTION_CYCLES: u32 = 3;

/// Decode/dispatch profile counters, collected only with the `perf-stats`
/// feature; reported by [`PerfStats::report`].
#[derive(Debug, Clone, Default)]
pub struct PerfStats {
    /// Instructions executed (`exec_one` entries).
    pub insns: u64,
    /// Instructions executed in Thumb state.
    pub thumb: u64,
    /// Decoded-instruction-cache hits.
    pub icache_hits: u64,
    /// Decoded-instruction-cache fills.
    pub icache_fills: u64,
    /// Dispatch histogram by decoded-op discriminant.
    pub op_hist: [u64; 32],
}

/// Display names for the op histogram, in discriminant order (keep in sync
/// with `decode::Op`).
const OP_NAMES: [&str; 20] = [
    "dp.imm", "dp.shimm", "dp.shreg", "mul", "mull", "mrs", "msr.reg", "msr.imm", "b", "bx", "ldr",
    "str", "ldrx", "strx", "ldm", "stm", "swp", "swi", "undef", "thumb.bl",
];

impl PerfStats {
    /// Multi-line summary: per-instruction averages, then every executed
    /// op sorted by frequency.
    pub fn report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let n = self.insns.max(1) as f64;
        let _ = writeln!(
            out,
            "insns {}  thumb {}  icache hit/i {:.3} fills {}",
            self.insns,
            self.thumb,
            self.icache_hits as f64 / n,
            self.icache_fills,
        );
        let mut ops: Vec<(&str, u64)> = self
            .op_hist
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(i, &c)| (OP_NAMES.get(i).copied().unwrap_or("?"), c))
            .collect();
        ops.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
        for (name, count) in ops {
            let pct = count as f64 / n * 100.0;
            let _ = writeln!(out, "  {name:>9}  {count:>12}  {pct:5.1}%");
        }
        out
    }
}

/// [`Cpu::events`] bit: the IRQ line is asserted.
pub(crate) const EVT_IRQ: u8 = 1 << 0;
/// [`Cpu::events`] bit: the FIQ line is asserted.
pub(crate) const EVT_FIQ: u8 = 1 << 1;
/// [`Cpu::events`] bit: a host trap was just recorded — a doorbell so the
/// [`Cpu::run`] hot loop needs no separate `host_trap` probe per
/// instruction. `Cpu::host_trap` itself stays the truth; the bit is
/// cleared as soon as it is acted on (or found stale).
pub(crate) const EVT_HOST_TRAP: u8 = 1 << 2;

/// An ARM32 processor.
///
/// As with the other cores, the CPU does not own its bus — call
/// [`Cpu::step`] with a `&mut B: Bus`.
#[derive(Debug, Clone)]
pub struct Cpu {
    /// The register file.
    pub regs: Registers,
    /// Total cycles elapsed since construction/reset. Nominal ARM7TDMI
    /// figures with S/N/I cycles collapsed — rough relative weights, not
    /// bus-accurate timings.
    pub cycles: u64,
    /// An embedder-visible idle state (ARMv4T has no wait-for-interrupt
    /// instruction; a device model may set this). A delivered IRQ/FIQ
    /// clears it.
    pub halted: bool,

    /// Host-latched boundary events (`EVT_*` bits), folded into one byte
    /// so the [`Cpu::step`] hot path tests them with a single predictable
    /// branch. IRQ/FIQ are level-sensitive lines: the bits track
    /// [`Cpu::set_irq`]/[`Cpu::set_fiq`].
    events: u8,

    // --- Per-instruction decode state ---------------------------------------
    /// Address of the current instruction (R15 operand reads and fault
    /// rewind derive from it).
    start_pc: u32,
    /// Executing in Thumb state (CPSR.T, cached per instruction).
    thumb: bool,
    /// Decoded-instruction cache (see `icache.rs`).
    icache: icache::ICache,

    // --- Host (OS-emulation) hooks — all inert at their defaults -------------
    /// If `true`, `SWI` records [`HostTrap::Syscall`] instead of vectoring
    /// through `0x08` (the natural hook for a Linux OS-emulation layer).
    /// Default `false` — SWI behaves exactly like hardware.
    pub trap_swi: bool,
    /// If `true`, a CPU exception is handed back via
    /// [`HostTrap::Exception`] instead of being delivered through its
    /// vector (the PC is rewound first). Default `false`.
    pub trap_faults: bool,
    /// Output: the pending host request. The host takes it after each
    /// [`Cpu::step`]. Cleared on RESET.
    pub host_trap: Option<HostTrap>,

    /// Profile counters (present only with the `perf-stats` feature).
    #[cfg(feature = "perf-stats")]
    pub stats: PerfStats,

    /// Test-only switch disabling the decoded-instruction cache, for
    /// differential tests.
    #[cfg(test)]
    pub(crate) fused_only: bool,
}

impl Cpu {
    /// Create a CPU in its power-on state: Supervisor mode, ARM state,
    /// IRQ+FIQ disabled, PC at the reset vector (0).
    pub fn new() -> Self {
        Cpu {
            regs: Registers::new(),
            cycles: 0,
            halted: false,
            events: 0,
            start_pc: 0,
            thumb: false,
            icache: icache::ICache::new(),
            trap_swi: false,
            trap_faults: false,
            host_trap: None,
            #[cfg(feature = "perf-stats")]
            stats: PerfStats::default(),
            #[cfg(test)]
            fused_only: false,
        }
    }

    /// Bump a profile counter; compiles to nothing without `perf-stats`.
    #[inline(always)]
    pub(crate) fn stat(&mut self, f: impl FnOnce(&mut PerfStats)) {
        #[cfg(feature = "perf-stats")]
        f(&mut self.stats);
        #[cfg(not(feature = "perf-stats"))]
        let _ = f;
    }

    /// Perform a RESET: registers to power-on state, pending events and
    /// traps cleared. (The interrupt lines are host state and stay as the
    /// host last set them.)
    pub fn reset(&mut self) {
        self.regs = Registers::new();
        self.halted = false;
        self.events &= EVT_IRQ | EVT_FIQ;
        self.icache.invalidate_all();
        self.host_trap = None;
    }

    // --- Interrupt line control ---------------------------------------------

    /// Drive the (level-sensitive) IRQ line. Serviced at the next
    /// instruction boundary while CPSR.I is clear; deassert once the
    /// device is acknowledged.
    pub fn set_irq(&mut self, level: bool) {
        if level {
            self.events |= EVT_IRQ;
        } else {
            self.events &= !EVT_IRQ;
        }
    }

    /// Drive the (level-sensitive) FIQ line.
    pub fn set_fiq(&mut self, level: bool) {
        if level {
            self.events |= EVT_FIQ;
        } else {
            self.events &= !EVT_FIQ;
        }
    }

    /// Whether an interrupt would be taken at the next boundary.
    #[inline]
    pub(crate) fn interrupt_pending(&self) -> bool {
        (self.events & EVT_FIQ != 0 && !self.regs.cpsr.f())
            || (self.events & EVT_IRQ != 0 && !self.regs.cpsr.i())
    }

    /// Record a host trap and ring the [`Cpu::run`] doorbell bit.
    #[inline]
    pub(crate) fn set_host_trap(&mut self, t: HostTrap) {
        self.host_trap = Some(t);
        self.events |= EVT_HOST_TRAP;
    }

    // --- Exception entry ------------------------------------------------------

    /// Take exception `e` at the current architectural state: bank into
    /// its mode, save the CPSR to the new SPSR, load the banked LR with
    /// the architected return offset, disable IRQ (and FIQ for a FIQ),
    /// leave Thumb state, and vector.
    ///
    /// [`Exception::Irq`]/[`Exception::Fiq`] assume `regs.gpr[15]` is at
    /// an instruction boundary; the fault kinds assume the executor's
    /// in-instruction state. Embedders normally only need this to inject
    /// aborts; use [`Cpu::set_irq`]/[`Cpu::set_fiq`] for interrupts.
    pub fn raise(&mut self, e: Exception) {
        let ret = match e {
            // gpr[15] already points past the instruction; in Thumb this
            // is the architected +2 form.
            Exception::Undefined | Exception::Swi => self.regs.gpr[15],
            Exception::PrefetchAbort => self.start_pc.wrapping_add(4),
            Exception::DataAbort => self.start_pc.wrapping_add(8),
            Exception::Irq | Exception::Fiq => self.regs.gpr[15].wrapping_add(4),
        };
        let old = self.regs.cpsr;
        self.regs.set_mode(e.mode());
        self.regs.spsr_store(old);
        self.regs.gpr[14] = ret;
        self.regs.cpsr.set(psr::T, false);
        self.regs.cpsr.set(psr::I, true);
        if e == Exception::Fiq {
            self.regs.cpsr.set(psr::F, true);
        }
        self.regs.gpr[15] = e.vector();
    }

    // --- Step and run -----------------------------------------------------------

    /// Execute one instruction (or service a pending interrupt). Returns
    /// the number of cycles consumed, and adds them to [`Cpu::cycles`].
    pub fn step<B: Bus>(&mut self, bus: &mut B) -> u32 {
        if self.boundary_pending() {
            return self.step_slow(bus);
        }
        self.step_fast(bus)
    }

    /// Execute up to `n` [`Cpu::step`]-equivalents as one batch, so the
    /// per-step boundary checks stay out of the embedder's loop. Semantics
    /// match `n` individual `step()` calls, with two deliberate
    /// refinements: the run stops as soon as [`Cpu::host_trap`] is set,
    /// and a halted CPU with no wake event pending returns
    /// [`RunExit::Halted`] instead of burning idle cycles.
    pub fn run<B: Bus>(&mut self, bus: &mut B, n: u64) -> RunResult {
        // A trap the embedder has not yet taken stops the run before
        // anything executes.
        if self.host_trap.is_some() {
            return RunResult {
                executed: 0,
                exit: RunExit::HostTrap,
            };
        }
        let mut executed = 0u64;
        while executed < n {
            // One predicate on the hot path: recording a host trap rings
            // EVT_HOST_TRAP, so no separate `host_trap` probe is needed.
            if self.boundary_pending() {
                if let Some(exit) = self.run_boundary(bus) {
                    return RunResult { executed, exit };
                }
            } else {
                self.step_fast(bus);
            }
            executed += 1;
        }
        RunResult {
            executed,
            exit: RunExit::Completed,
        }
    }

    /// [`Cpu::run`]'s boundary arm, out of line to keep the run loop's
    /// body as lean as the step() loop's: either resolves the boundary as
    /// an exit reason, or performs one slow step-unit and returns `None`.
    #[cold]
    #[inline(never)]
    fn run_boundary<B: Bus>(&mut self, bus: &mut B) -> Option<RunExit> {
        if self.host_trap.is_some() {
            self.events &= !EVT_HOST_TRAP;
            return Some(RunExit::HostTrap);
        }
        if self.halted && !self.interrupt_pending() {
            return Some(RunExit::Halted);
        }
        self.step_slow(bus);
        None
    }

    /// Whether the next instruction boundary needs [`Cpu::step_slow`]: a
    /// latched event or the halted state. `halted` is a `pub` field
    /// (writable by embedders directly), so it is re-read rather than
    /// mirrored into `events` — a mirror of a `pub` field can go stale.
    #[inline(always)]
    fn boundary_pending(&self) -> bool {
        self.events != 0 || self.halted
    }

    /// The boundary slow path: interrupt delivery and the halted state.
    /// Out of line so the hot path pays only [`Cpu::boundary_pending`]'s
    /// single predicted-not-taken branch.
    #[cold]
    #[inline(never)]
    fn step_slow<B: Bus>(&mut self, bus: &mut B) -> u32 {
        // A stale run() doorbell (trap already taken, or the embedder
        // drives step() directly) must not pin every step onto this path.
        self.events &= !EVT_HOST_TRAP;

        // FIQ outranks IRQ; both are recognized only at boundaries.
        if self.events & EVT_FIQ != 0 && !self.regs.cpsr.f() {
            self.halted = false;
            self.raise(Exception::Fiq);
            self.cycles += EXCEPTION_CYCLES as u64;
            return EXCEPTION_CYCLES;
        }
        if self.events & EVT_IRQ != 0 && !self.regs.cpsr.i() {
            self.halted = false;
            self.raise(Exception::Irq);
            self.cycles += EXCEPTION_CYCLES as u64;
            return EXCEPTION_CYCLES;
        }

        if self.halted {
            self.cycles += 1;
            return 1;
        }

        self.step_fast(bus)
    }

    /// The boundary fast path: no event latched, not halted — just execute
    /// one instruction.
    #[inline(always)]
    fn step_fast<B: Bus>(&mut self, bus: &mut B) -> u32 {
        let cycles = self.exec_one(bus);
        self.cycles += cycles as u64;
        cycles
    }

    /// Fetch, decode (through the decoded-instruction cache) and execute
    /// the instruction at `gpr[15]`.
    fn exec_one<B: Bus>(&mut self, bus: &mut B) -> u32 {
        let thumb = self.regs.cpsr.t();
        self.thumb = thumb;
        let pc = self.regs.gpr[15] & if thumb { !1u32 } else { !3u32 };
        self.start_pc = pc;
        self.stat(|s| {
            s.insns += 1;
            s.thumb += thumb as u64;
        });

        #[cfg(test)]
        let try_cache = !self.fused_only;
        #[cfg(not(test))]
        let try_cache = true;

        // Decoded-instruction-cache probe: on a hit, skip the fetch and
        // decode entirely (see icache.rs for the validity rules).
        let key = pc as u64 | (thumb as u64) << 63;
        let slot = (pc >> 1) as usize & (icache::ICACHE_ENTRIES - 1);
        let e = &self.icache.entries[slot];
        let d = if try_cache
            && e.key == key
            && e.version >= self.icache.stamps[(pc >> 12) as usize & (icache::STAMP_SLOTS - 1)]
            && e.version >= self.icache.inval
        {
            let d = e.insn;
            #[cfg(any(test, debug_assertions))]
            {
                self.icache.hits += 1;
            }
            self.stat(|s| s.icache_hits += 1);
            #[cfg(debug_assertions)]
            self.icache_differential(bus, &d);
            d
        } else {
            let d = if thumb {
                decode::decode_thumb(bus.read16(pc))
            } else {
                decode::decode_arm(bus.read32(pc))
            };
            if try_cache {
                self.icache_fill(key, slot, &d);
            }
            d
        };
        self.stat(|s| s.op_hist[d.op as usize] += 1);

        self.regs.gpr[15] = pc.wrapping_add(d.len as u32);
        match self.exec_decoded(bus, &d) {
            Ok(c) => c,
            Err(e) => self.fault_epilogue(e),
        }
    }

    /// Dispose of a fault from the executor. Only decode-detected faults
    /// exist on this core (no MMU — memory never aborts), so the rewind is
    /// just the PC; no register snapshot machinery is needed, unlike the
    /// x86 cores.
    #[cold]
    #[inline(never)]
    fn fault_epilogue(&mut self, e: Exception) -> u32 {
        if self.trap_faults {
            self.regs.gpr[15] = self.start_pc;
            self.set_host_trap(HostTrap::Exception(e));
        } else {
            // gpr[15] already points past the instruction — exactly the
            // architected LR for the undefined-instruction trap.
            self.raise(e);
        }
        EXCEPTION_CYCLES
    }

    // --- Data access helpers ----------------------------------------------------
    // Every guest store funnels through these so the decoded-instruction
    // cache sees a write stamp; loads go straight to the bus.

    /// Guest store of one byte.
    #[inline]
    pub(crate) fn write8<B: Bus>(&mut self, bus: &mut B, addr: u32, v: u8) {
        self.icache.stamp_write(addr);
        bus.write(addr, v);
    }

    /// Guest store of a halfword (`addr` pre-aligned by the caller).
    #[inline]
    pub(crate) fn write16<B: Bus>(&mut self, bus: &mut B, addr: u32, v: u16) {
        self.icache.stamp_write(addr);
        bus.write16(addr, v);
    }

    /// Guest store of a word (`addr` pre-aligned by the caller).
    #[inline]
    pub(crate) fn write32<B: Bus>(&mut self, bus: &mut B, addr: u32, v: u32) {
        self.icache.stamp_write(addr);
        bus.write32(addr, v);
    }

    /// LDR's unaligned-word rule: read the aligned word, rotated so the
    /// addressed byte lands in bits 7:0.
    #[inline]
    pub(crate) fn read32_rot<B: Bus>(&mut self, bus: &mut B, addr: u32) -> u32 {
        bus.read32(addr & !3).rotate_right(8 * (addr & 3))
    }

    /// LDRH's odd-address rule: the aligned halfword rotated by 8.
    #[inline]
    pub(crate) fn read16_rot<B: Bus>(&mut self, bus: &mut B, addr: u32) -> u32 {
        let v = bus.read16(addr & !1) as u32;
        if addr & 1 != 0 { v.rotate_right(8) } else { v }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}
