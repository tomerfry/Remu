//! The Intel 80386 CPU core: 32-bit registers and addressing, real mode and
//! protected mode with paging, the full documented 386 integer instruction
//! set, instruction-atomic timing.
//!
//! Like the other cores, it is deliberately self-contained (own [`Bus`]
//! trait, own register file) so promoting each core to its own crate later
//! stays mechanical.
//!
//! ```
//! use remu::x86_32::{Cpu, Bus, LinearMemory};
//!
//! let mut mem = LinearMemory::new();
//! mem.load(0x0_1100, &[0x66, 0xB8, 0x78, 0x56, 0x34, 0x12]); // MOV EAX, 0x12345678
//! let mut cpu = Cpu::new();
//! cpu.set_cs_ip(0x0000, 0x1100);
//! cpu.step(&mut mem);
//! assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
//! ```

mod alu;
mod decode;
mod execute;
mod execute_0f;
mod icache;
mod modrm;
mod paging;
mod protected;
pub mod registers;

pub use registers::{DescTable, EFlags, Registers, SegReg, cr0, reg};

/// The 386 memory and I/O bus.
///
/// Memory addresses are 32-bit physical. The wide accessors have
/// byte-composed defaults so simple devices only implement `read`/`write`;
/// RAM-backed buses should override them for speed. The CPU only issues a
/// wide access when it does not cross a page boundary, so overrides may
/// assume contiguity.
pub trait Bus {
    /// Read one byte from physical address `addr`.
    fn read(&mut self, addr: u32) -> u8;

    /// Write `value` to physical address `addr`.
    fn write(&mut self, addr: u32, value: u8);

    /// Read a little-endian word from `addr`.
    #[inline]
    fn read16(&mut self, addr: u32) -> u16 {
        self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
    }

    /// Read a little-endian double-word from `addr`.
    #[inline]
    fn read32(&mut self, addr: u32) -> u32 {
        self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
    }

    /// Write a little-endian word to `addr`.
    #[inline]
    fn write16(&mut self, addr: u32, value: u16) {
        self.write(addr, value as u8);
        self.write(addr.wrapping_add(1), (value >> 8) as u8);
    }

    /// Write a little-endian double-word to `addr`.
    #[inline]
    fn write32(&mut self, addr: u32, value: u32) {
        self.write16(addr, value as u16);
        self.write16(addr.wrapping_add(2), (value >> 16) as u16);
    }

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

/// A flat 16 MiB RAM (the address space of the SingleStepTests 386EX rig) —
/// the simplest possible 386 machine, for tests and raw programs. Addresses
/// wrap at 16 MiB.
pub struct LinearMemory {
    /// 24-bit physical address space.
    pub ram: Box<[u8]>,
}

/// Address mask for [`LinearMemory`] (16 MiB).
const LINEAR_MASK: usize = 0xFF_FFFF;

impl LinearMemory {
    /// Create a zero-initialized 16 MiB memory.
    pub fn new() -> Self {
        LinearMemory {
            ram: vec![0u8; LINEAR_MASK + 1].into_boxed_slice(),
        }
    }

    /// Load `data` into memory starting at physical address `addr`.
    pub fn load(&mut self, addr: u32, data: &[u8]) {
        for (i, &byte) in data.iter().enumerate() {
            self.ram[(addr as usize + i) & LINEAR_MASK] = byte;
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
        self.ram[addr as usize & LINEAR_MASK]
    }

    fn write(&mut self, addr: u32, value: u8) {
        self.ram[addr as usize & LINEAR_MASK] = value;
    }

    fn read16(&mut self, addr: u32) -> u16 {
        let a = addr as usize & LINEAR_MASK;
        if a < LINEAR_MASK {
            u16::from_le_bytes(self.ram[a..a + 2].try_into().unwrap())
        } else {
            self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
        }
    }

    fn read32(&mut self, addr: u32) -> u32 {
        let a = addr as usize & LINEAR_MASK;
        if a + 3 <= LINEAR_MASK {
            u32::from_le_bytes(self.ram[a..a + 4].try_into().unwrap())
        } else {
            self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
        }
    }

    fn write16(&mut self, addr: u32, value: u16) {
        let a = addr as usize & LINEAR_MASK;
        if a < LINEAR_MASK {
            self.ram[a..a + 2].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write(addr, value as u8);
            self.write(addr.wrapping_add(1), (value >> 8) as u8);
        }
    }

    fn write32(&mut self, addr: u32, value: u32) {
        let a = addr as usize & LINEAR_MASK;
        if a + 3 <= LINEAR_MASK {
            self.ram[a..a + 4].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write16(addr, value as u16);
            self.write16(addr.wrapping_add(2), (value >> 16) as u16);
        }
    }
}

/// A CPU exception (or software interrupt turned fault) in flight.
///
/// `error` carries the error code pushed by protected-mode gates for the
/// faults that define one; it is ignored in real mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exception {
    pub vector: u8,
    pub error: Option<u16>,
}

impl Exception {
    /// #DE — divide error.
    pub(crate) fn de() -> Self {
        Exception {
            vector: 0,
            error: None,
        }
    }
    /// #BR — BOUND range exceeded.
    pub(crate) fn br() -> Self {
        Exception {
            vector: 5,
            error: None,
        }
    }
    /// #UD — invalid opcode.
    pub(crate) fn ud() -> Self {
        Exception {
            vector: 6,
            error: None,
        }
    }
    /// #NM — no math coprocessor.
    pub(crate) fn nm() -> Self {
        Exception {
            vector: 7,
            error: None,
        }
    }
    /// #TS — invalid TSS.
    pub(crate) fn ts(sel: u16) -> Self {
        Exception {
            vector: 10,
            error: Some(sel),
        }
    }
    /// #NP — segment not present.
    pub(crate) fn np(sel: u16) -> Self {
        Exception {
            vector: 11,
            error: Some(sel),
        }
    }
    /// #SS — stack fault.
    pub(crate) fn ss(sel: u16) -> Self {
        Exception {
            vector: 12,
            error: Some(sel),
        }
    }
    /// #GP — general protection fault.
    pub(crate) fn gp(sel: u16) -> Self {
        Exception {
            vector: 13,
            error: Some(sel),
        }
    }
    /// #PF — page fault (CR2 is set by the paging unit).
    pub(crate) fn pf(code: u16) -> Self {
        Exception {
            vector: 14,
            error: Some(code),
        }
    }
}

/// Result type for anything that can raise a CPU exception.
pub(crate) type Exec<T> = Result<T, Exception>;

/// How an interrupt or exception reached the delivery path. This selects the
/// gate privilege check (`INT n` requires gate DPL >= CPL), the `EXT` bit of
/// any error code generated during delivery, and the double-fault
/// classification of the event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Event {
    /// A processor-detected exception (fault, trap or abort).
    Fault,
    /// `INT n`, `INT3`, `INTO` or `ICEBP`.
    SoftInt,
    /// A hardware interrupt or NMI.
    External,
}

/// A request from the CPU for the host (an OS-emulation layer) to act, in
/// place of the normal in-guest IDT delivery.
///
/// Populated in [`Cpu::host_trap`] only when the corresponding opt-in is set
/// ([`Cpu::syscall_int`] / [`Cpu::trap_faults`]); the CPU is otherwise a plain
/// bare-metal 386. A userspace emulator drives [`Cpu::step`] and, after each
/// call, takes any pending trap to service it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTrap {
    /// The guest executed `INT n` for the vector named by [`Cpu::syscall_int`]
    /// (e.g. `int 0x80`). `EIP` already points past the instruction; read the
    /// syscall number/arguments from the registers and write the result to
    /// `EAX`.
    Syscall,
    /// The guest raised a CPU exception while [`Cpu::trap_faults`] was set.
    /// Register state has been rewound to the faulting instruction (a `#PF`
    /// leaves the linear address in `CR2` and its code in `Exception::error`),
    /// so the host may fix the fault and resume, or terminate the process.
    Exception(Exception),
}

/// Why [`Cpu::run`] stopped before retiring all `n` requested step-units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive] // a JIT backend may add exit reasons
pub enum RunExit {
    /// All `n` units retired.
    Completed,
    /// [`Cpu::host_trap`] is set (syscall or trapped fault). Service and
    /// `take()` it, then call [`Cpu::run`] again — no guest instruction has
    /// executed past the trap.
    HostTrap,
    /// The CPU is halted with no wake event pending (no NMI, no INTR with
    /// `IF`). Assert an interrupt (or give up), then call [`Cpu::run`]
    /// again. Unlike [`Cpu::step`], which burns one idle cycle per call
    /// while halted, [`Cpu::run`] returns instead of idling.
    Halted,
    /// Triple fault; only [`Cpu::reset`] recovers.
    Shutdown,
}

/// The outcome of [`Cpu::run`]: how much ran, and why it stopped.
#[derive(Debug, Clone, Copy)]
pub struct RunResult {
    /// [`Cpu::step`]-equivalents retired: instruction executions and
    /// interrupt/trap deliveries each count one, exactly as one `step()`
    /// call would.
    pub executed: u64,
    /// Why the run ended.
    pub exit: RunExit,
}

/// Cycles consumed by servicing a hardware interrupt (real-mode figure).
const INTERRUPT_CYCLES: u32 = 37;

/// One-byte opcodes that may legally carry a LOCK prefix (given a memory
/// destination, validated per-handler): the ALU r/m,r forms, the immediate
/// ALU groups, XCHG, and the NOT/NEG/INC/DEC groups. `0F` two-byte opcodes
/// are screened in `dispatch_0f`.
const LOCK_CANDIDATE: [bool; 256] = {
    let mut t = [false; 256];
    let mut op = 0x00;
    while op <= 0x31 {
        // 00/01, 08/09, ... 30/31 (rm,r forms of ADD OR ADC SBB AND SUB XOR).
        t[op] = true;
        t[op + 1] = true;
        op += 8;
    }
    t[0x80] = true;
    t[0x81] = true;
    t[0x82] = true;
    t[0x83] = true;
    t[0x86] = true;
    t[0x87] = true;
    t[0xF6] = true;
    t[0xF7] = true;
    t[0xFE] = true;
    t[0xFF] = true;
    t
};

/// Decode/dispatch profile counters, collected only with the `perf-stats`
/// feature. They quantify per-instruction decode work (the target of the
/// decoded-instruction-cache effort) and are reported by
/// [`PerfStats::report`].
#[derive(Debug, Clone)]
pub struct PerfStats {
    /// Instructions executed (`exec_one` entries).
    pub insns: u64,
    /// Prefix bytes consumed by the decode loop.
    pub prefix_bytes: u64,
    /// `modrm()` decodes (ModRM + SIB + displacement fetches).
    pub modrm_calls: u64,
    /// Operand-size immediates fetched via `fetch_imm`.
    pub imm_fetches: u64,
    /// One-byte-opcode dispatch histogram.
    pub opcode_hist: [u64; 256],
    /// Two-byte (`0F xx`) dispatch histogram.
    pub opcode_0f_hist: [u64; 256],
    /// Decoded-instruction-cache hits.
    pub icache_hits: u64,
    /// Decoded-instruction-cache fills.
    pub icache_fills: u64,
    /// Decoded instructions rejected by the cache (page-crossers).
    pub icache_uncacheable: u64,
}

impl Default for PerfStats {
    fn default() -> Self {
        PerfStats {
            insns: 0,
            prefix_bytes: 0,
            modrm_calls: 0,
            imm_fetches: 0,
            opcode_hist: [0; 256],
            opcode_0f_hist: [0; 256],
            icache_hits: 0,
            icache_fills: 0,
            icache_uncacheable: 0,
        }
    }
}

impl PerfStats {
    /// Multi-line summary: per-instruction averages, then every executed
    /// opcode sorted by frequency.
    pub fn report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let n = self.insns.max(1) as f64;
        let _ = writeln!(
            out,
            "insns {}  prefix/i {:.3}  modrm/i {:.3}  imm/i {:.3}  \
             icache hit/i {:.3} fills {} uncacheable {}",
            self.insns,
            self.prefix_bytes as f64 / n,
            self.modrm_calls as f64 / n,
            self.imm_fetches as f64 / n,
            self.icache_hits as f64 / n,
            self.icache_fills,
            self.icache_uncacheable,
        );
        let one = self.opcode_hist.iter().enumerate();
        let two = self.opcode_0f_hist.iter().enumerate();
        let mut ops: Vec<(String, u64)> = one
            .map(|(op, &c)| (format!("{op:02X}"), c))
            .chain(two.map(|(op, &c)| (format!("0F {op:02X}"), c)))
            .filter(|&(_, c)| c > 0)
            .collect();
        ops.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, count) in ops {
            let pct = count as f64 / n * 100.0;
            let _ = writeln!(out, "  {name:>5}  {count:>12}  {pct:5.1}%");
        }
        out
    }
}

/// [`Cpu::events`] bit: a latched non-maskable interrupt (vector 2).
pub(crate) const EVT_NMI: u8 = 1 << 0;
/// [`Cpu::events`] bit: the INTR line is asserted (vector in `Cpu::intr`).
pub(crate) const EVT_INTR: u8 = 1 << 1;
/// [`Cpu::events`] bit: interrupts (and traps) are inhibited for one
/// instruction after `MOV SS` / `POP SS` / `STI`.
pub(crate) const EVT_INHIBIT: u8 = 1 << 2;
/// [`Cpu::events`] bit: a host trap was just recorded — a doorbell so the
/// [`Cpu::run`] hot loop needs no separate `host_trap` probe per
/// instruction. `Cpu::host_trap` itself stays the truth; the bit is cleared
/// as soon as it is acted on (or found stale).
pub(crate) const EVT_HOST_TRAP: u8 = 1 << 3;

/// An 80386 processor.
///
/// As with the other cores, the CPU does not own its bus — call [`Cpu::step`]
/// with a `&mut B: Bus`.
#[derive(Debug, Clone)]
pub struct Cpu {
    /// The register file.
    pub regs: Registers,
    /// Total cycles elapsed since construction/reset. Timing is
    /// instruction-atomic using documented 386 base timings; the prefetch
    /// queue and dynamic bus sizing are not modeled, so counts are
    /// approximate.
    pub cycles: u64,
    /// Set while the processor is stopped by `HLT` (an interrupt resumes it).
    pub halted: bool,
    /// Set on triple fault; only RESET leaves this state.
    pub shutdown: bool,

    /// Host-latched boundary events (`EVT_*` bits: NMI, INTR line,
    /// one-instruction interrupt shadow), folded into one byte so the
    /// [`Cpu::step`] hot path tests them with a single predictable branch.
    /// The bits are the *only* storage for these latches; `EVT_INTR` mirrors
    /// `intr.is_some()`.
    events: u8,
    /// Pending maskable interrupt request with its vector (as supplied by a
    /// PIC during the INTA cycle). `Some` iff `EVT_INTR` is set.
    intr: Option<u8>,

    // --- Per-instruction decode state ---------------------------------------
    /// Segment-override prefix (segment register index).
    seg_override: Option<u8>,
    /// Repeat prefix: `true` for `REP`/`REPE` (F3), `false` for `REPNE` (F2).
    rep: Option<bool>,
    /// A LOCK prefix was decoded.
    lock: bool,
    /// The current instruction legally used its LOCK prefix (set by handlers
    /// of lockable instructions with a memory destination).
    lock_ok: bool,
    /// Effective operand size is 32-bit (CS.D xor `66` prefix).
    osize32: bool,
    /// Effective address size is 32-bit (CS.D xor `67` prefix).
    asize32: bool,
    /// EIP at the start of the current instruction (fault reporting).
    start_eip: u32,
    /// A fault should keep the current register state and only rewind EIP —
    /// set by string instructions, whose per-iteration progress is
    /// architecturally visible when an iteration faults.
    commit_on_fault: bool,
    /// Bytes consumed by the current instruction (15-byte limit).
    ilen: u8,
    /// Full register file captured by [`Cpu::prepare_cold_write`] at the
    /// first cold-register write of the current instruction; meaningful only
    /// while `cold_saved` is set.
    fault_regs: Registers,
    /// The current instruction wrote (or is about to write) a cold register,
    /// so a fault must rewind from `fault_regs`, not just the GPR snapshot.
    cold_saved: bool,
    /// Forces paging to treat the current access as a supervisor access
    /// regardless of CPL. The 386 performs its *implicit* accesses — reads of
    /// the descriptor tables, the IDT and the TSS, the accessed/busy bit
    /// writebacks, and the frame pushes onto an inner-privilege stack — with
    /// supervisor privilege even while CPL is still 3.
    supervisor_override: bool,
    /// TLB for paged address translation (see `paging.rs`).
    tlb: paging::Tlb,
    /// Decoded-instruction cache (see `icache.rs`).
    icache: icache::ICache,

    // --- Host (OS-emulation) hooks — all inert at their defaults -------------
    /// If set, `INT n` for this vector does not vector through the IDT; instead
    /// the CPU records [`HostTrap::Syscall`] and returns, letting a host
    /// syscall layer service it (e.g. `Some(0x80)` for Linux i386). Default
    /// `None` — `INT` behaves exactly like hardware.
    pub syscall_int: Option<u8>,
    /// If `true`, a CPU exception is handed back via [`HostTrap::Exception`]
    /// instead of being delivered through the IDT (register state is rewound
    /// first). A userspace emulator has no guest kernel behind the IDT, so it
    /// resolves faults on the host. Default `false`.
    pub trap_faults: bool,
    /// If `true`, decode a handful of post-386 opcodes (`CPUID`, `RDTSC`,
    /// `CMPXCHG`, `XADD`, `BSWAP`, `CMOVcc`, long `NOP`) that real 32-bit
    /// binaries use. Default `false` keeps strict 386 behavior (`#UD`), so the
    /// conformance suite is unaffected.
    pub extensions: bool,
    /// Output: the pending host request, set by the CPU when [`syscall_int`]
    /// matches or a fault is trapped. The host takes it after each
    /// [`Cpu::step`]. Cleared on RESET.
    ///
    /// [`syscall_int`]: Cpu::syscall_int
    pub host_trap: Option<HostTrap>,

    /// Profile counters (present only with the `perf-stats` feature).
    #[cfg(feature = "perf-stats")]
    pub stats: PerfStats,

    /// Test-only switch forcing the fused decode path, for differential
    /// tests of the decoupled decoder.
    #[cfg(test)]
    pub(crate) fused_only: bool,
}

impl Cpu {
    /// Create a CPU in its power-on state: execution begins at `F000:FFF0`
    /// (CS base `FFFF0000`).
    pub fn new() -> Self {
        Cpu {
            regs: Registers::new(),
            cycles: 0,
            halted: false,
            shutdown: false,
            events: 0,
            intr: None,
            seg_override: None,
            rep: None,
            lock: false,
            lock_ok: false,
            osize32: false,
            asize32: false,
            start_eip: 0,
            commit_on_fault: false,
            ilen: 0,
            fault_regs: Registers::new(),
            cold_saved: false,
            supervisor_override: false,
            tlb: paging::Tlb::new(),
            icache: icache::ICache::new(),
            syscall_int: None,
            trap_faults: false,
            extensions: false,
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

    /// Perform a RESET: registers to power-on state, pending interrupts and
    /// shutdown cleared.
    pub fn reset(&mut self) {
        self.regs = Registers::new();
        self.halted = false;
        self.shutdown = false;
        self.events = 0;
        self.intr = None;
        self.supervisor_override = false;
        self.tlb.flush();
        self.icache.invalidate_all();
        self.host_trap = None;
    }

    /// Convenience for tests and loaders: set `CS:EIP` with real-mode
    /// semantics (CS base = `sel * 16`).
    pub fn set_cs_ip(&mut self, sel: u16, eip: u32) {
        self.regs.seg[reg::CS as usize] = SegReg::real(sel);
        self.regs.eip = eip;
    }

    /// True once `CR0.PE` is set and the CPU is not in Virtual-8086 mode.
    #[inline]
    pub fn protected_mode(&self) -> bool {
        self.regs.cr0 & cr0::PE != 0 && !self.regs.eflags.contains(EFlags::VM)
    }

    /// Current privilege level (CPL). Real mode and V86 report their fixed
    /// levels through the CS cache attributes maintained by segment loads.
    #[inline]
    pub fn cpl(&self) -> u8 {
        if self.regs.eflags.contains(EFlags::VM) {
            3
        } else if self.protected_mode() {
            self.regs.seg[reg::CS as usize].dpl()
        } else {
            0
        }
    }

    /// Execute one instruction (or service a pending interrupt). Returns the
    /// number of cycles consumed, and adds them to [`Cpu::cycles`].
    pub fn step<B: Bus>(&mut self, bus: &mut B) -> u32 {
        if self.boundary_pending() {
            return self.step_slow(bus);
        }
        self.step_fast(bus)
    }

    /// Execute up to `n` [`Cpu::step`]-equivalents as one batch, so the
    /// per-step boundary checks stay out of the embedder's loop. Semantics
    /// match `n` individual `step()` calls, with two deliberate refinements:
    /// the run stops (rather than executing further instructions) as soon as
    /// [`Cpu::host_trap`] is set, and a halted CPU with no wake event
    /// pending returns [`RunExit::Halted`] instead of burning idle cycles.
    ///
    /// The STI/`MOV SS` shadow and a pending single-step trap are CPU state,
    /// so they carry correctly across `run` boundaries.
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

    /// [`Cpu::run`]'s boundary arm, out of line to keep the run loop's body
    /// as lean as the step() loop's: either resolves the boundary as an exit
    /// reason, or performs one slow step-unit (delivery, wake, shadow or
    /// trap-flag work) and returns `None`.
    #[cold]
    #[inline(never)]
    fn run_boundary<B: Bus>(&mut self, bus: &mut B) -> Option<RunExit> {
        if self.host_trap.is_some() {
            self.events &= !EVT_HOST_TRAP;
            return Some(RunExit::HostTrap);
        }
        if self.shutdown {
            return Some(RunExit::Shutdown);
        }
        if self.halted && !self.interrupt_pending() {
            return Some(RunExit::Halted);
        }
        self.step_slow(bus);
        None
    }

    /// Whether the next instruction boundary needs [`Cpu::step_slow`]: a
    /// latched event, HLT/shutdown state, or a pending single-step trap.
    ///
    /// Short-circuit `||`, not bitwise `|`: measured faster on both cores
    /// (the fused OR-chain stalls the branch on all four loads; the
    /// short-circuit chain is a run of individually predicted-not-taken
    /// branches). `halted`/`shutdown` are `pub` fields and TF is
    /// guest-visible in EFLAGS (all writable by embedders and tests
    /// directly), so they are re-read here rather than mirrored into
    /// `events` — a mirror of a `pub` field can go stale.
    #[inline(always)]
    fn boundary_pending(&self) -> bool {
        (self.events != 0)
            || self.halted
            || self.shutdown
            || (self.regs.eflags.bits() & EFlags::TF.bits() != 0)
    }

    /// The boundary slow path: shutdown/halt states, interrupt delivery, the
    /// one-instruction shadow, and single-step traps. Out of line so the hot
    /// path pays only [`Cpu::boundary_pending`]'s single predicted-not-taken
    /// branch.
    #[cold]
    #[inline(never)]
    fn step_slow<B: Bus>(&mut self, bus: &mut B) -> u32 {
        // A stale run() doorbell (trap already taken, or the embedder drives
        // step() directly) must not pin every step onto this slow path.
        self.events &= !EVT_HOST_TRAP;
        if self.shutdown {
            self.cycles += 1;
            return 1;
        }

        // Interrupts are recognized at instruction boundaries, except for the
        // one-instruction shadow after MOV SS / POP SS / STI.
        let inhibited = self.events & EVT_INHIBIT != 0;
        self.events &= !EVT_INHIBIT;
        if !inhibited {
            if self.events & EVT_NMI != 0 {
                self.events &= !EVT_NMI;
                self.halted = false;
                self.deliver(
                    bus,
                    Exception {
                        vector: 2,
                        error: None,
                    },
                    Event::External,
                );
                self.cycles += INTERRUPT_CYCLES as u64;
                return INTERRUPT_CYCLES;
            }
            if let Some(vector) = self.intr
                && self.regs.eflags.contains(EFlags::IF)
            {
                self.intr = None;
                self.events &= !EVT_INTR;
                self.halted = false;
                self.deliver(
                    bus,
                    Exception {
                        vector,
                        error: None,
                    },
                    Event::External,
                );
                self.cycles += INTERRUPT_CYCLES as u64;
                return INTERRUPT_CYCLES;
            }
        }

        if self.halted {
            self.cycles += 1;
            return 1;
        }

        // Trap flag: a single-step exception fires after this instruction if
        // TF was set when it started.
        let trap = self.regs.eflags.contains(EFlags::TF);

        let saved_gpr = self.regs.gpr;
        #[cfg(debug_assertions)]
        let saved_all = self.regs;
        let mut cycles = match self.exec_one(bus) {
            Ok(c) => c,
            Err(e) => match self.fault_epilogue(
                bus,
                e,
                &saved_gpr,
                #[cfg(debug_assertions)]
                &saved_all,
            ) {
                Some(c) => c,
                None => return INTERRUPT_CYCLES,
            },
        };

        if trap && self.regs.eflags.contains(EFlags::TF) && self.events & EVT_INHIBIT == 0 {
            self.deliver(
                bus,
                Exception {
                    vector: 1,
                    error: None,
                },
                Event::Fault,
            );
            cycles += INTERRUPT_CYCLES;
        }

        self.cycles += cycles as u64;
        cycles
    }

    /// The boundary fast path: no event latched, not halted/shut down, TF
    /// clear — just execute one instruction. The single-step epilogue is
    /// skipped because TF was clear when the instruction started; an
    /// instruction that sets TF (or arms the shadow) routes the *next* step
    /// through [`Cpu::step_slow`] via [`Cpu::boundary_pending`].
    #[inline(always)]
    fn step_fast<B: Bus>(&mut self, bus: &mut B) -> u32 {
        // Faults restore register state so the instruction can restart. Only
        // the GPRs are snapshotted here: EIP rewinds via `start_eip`, EFLAGS
        // keeps whatever the faulting computation left behind (the
        // divide-fault frame pushes those flags, and LOCK-#UD is decided at
        // decode time before anything runs), CR2 keeps the page-fault
        // address, and every other field is cold — its writers call
        // `prepare_cold_write` first, which captures the full register file.
        let saved_gpr = self.regs.gpr;
        #[cfg(debug_assertions)]
        let saved_all = self.regs;
        let cycles = match self.exec_one(bus) {
            Ok(c) => c,
            Err(e) => match self.fault_epilogue(
                bus,
                e,
                &saved_gpr,
                #[cfg(debug_assertions)]
                &saved_all,
            ) {
                Some(c) => c,
                None => return INTERRUPT_CYCLES,
            },
        };

        self.cycles += cycles as u64;
        cycles
    }

    /// Rewind and dispose of a fault from `exec_one`: restore pre-instruction
    /// register state, then either hand the exception to the host
    /// (`trap_faults`, returns `None` — cycles already accounted) or deliver
    /// it through the IDT (returns `Some(INTERRUPT_CYCLES)` for the caller's
    /// accounting).
    #[cold]
    #[inline(never)]
    fn fault_epilogue<B: Bus>(
        &mut self,
        bus: &mut B,
        e: Exception,
        saved_gpr: &[u32; 8],
        #[cfg(debug_assertions)] saved_all: &Registers,
    ) -> Option<u32> {
        if self.commit_on_fault {
            // String/PUSHA-style progress stays; only EIP rewinds.
            self.regs.eip = self.start_eip;
        } else {
            if self.cold_saved {
                let (eflags, cr2) = (self.regs.eflags, self.regs.cr2);
                self.regs = self.fault_regs;
                self.regs.eflags = eflags;
                self.regs.cr2 = cr2;
            }
            self.regs.gpr = *saved_gpr;
            self.regs.eip = self.start_eip;
            #[cfg(debug_assertions)]
            {
                // Differential check against the old whole-file
                // rewind: a mismatch means a cold-register writer is
                // missing its `prepare_cold_write` call.
                let mut want = *saved_all;
                want.eflags = self.regs.eflags;
                want.cr2 = self.regs.cr2;
                assert_eq!(
                    self.regs, want,
                    "fault rewind mismatch at {:#x}",
                    self.start_eip
                );
            }
        }
        // OS-emulation hook: hand the (rewound, restartable) fault to
        // the host instead of vectoring through the IDT.
        if self.trap_faults {
            self.set_host_trap(HostTrap::Exception(e));
            self.cycles += INTERRUPT_CYCLES as u64;
            return None;
        }
        self.deliver(bus, e, Event::Fault);
        Some(INTERRUPT_CYCLES)
    }

    /// Capture the register file before the first write to a cold register
    /// (anything besides `gpr`/`eip`/`eflags`/`cr2`) during the current
    /// instruction, so a later fault in the same instruction can rewind it.
    /// Every function that writes such a register must call this first.
    #[inline]
    pub(crate) fn prepare_cold_write(&mut self) {
        if !self.cold_saved {
            self.cold_saved = true;
            self.fault_regs = self.regs;
        }
    }

    /// Consume the prefix bytes at `CS:EIP`, recording their effects, and
    /// return the opcode byte plus the number of prefixes consumed. Resets
    /// the size flags from CS.D first; the caller resets the rest of the
    /// per-instruction decode state.
    #[inline(always)]
    fn scan_prefixes<B: Bus>(&mut self, bus: &mut B) -> Exec<(u8, u32)> {
        let db = self.regs.seg[reg::CS as usize].db();
        self.osize32 = db;
        self.asize32 = db;

        let mut npfx = 0u32;
        let opcode = loop {
            let b = self.fetch8(bus)?;
            match b {
                0x26 => self.seg_override = Some(reg::ES),
                0x2E => self.seg_override = Some(reg::CS),
                0x36 => self.seg_override = Some(reg::SS),
                0x3E => self.seg_override = Some(reg::DS),
                0x64 => self.seg_override = Some(reg::FS),
                0x65 => self.seg_override = Some(reg::GS),
                0x66 => self.osize32 = !db,
                0x67 => self.asize32 = !db,
                0xF0 => self.lock = true,
                0xF2 => self.rep = Some(false),
                0xF3 => self.rep = Some(true),
                _ => break b,
            }
            self.stat(|s| s.prefix_bytes += 1);
            npfx += 1;
        };
        Ok((opcode, npfx))
    }

    /// Decode prefixes and execute the instruction at `CS:EIP`.
    fn exec_one<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        self.start_eip = self.regs.eip;
        self.stat(|s| s.insns += 1);

        #[cfg(test)]
        let try_hot = !self.fused_only;
        #[cfg(not(test))]
        let try_hot = true;

        // Decoded-instruction-cache probe: on a hit, skip the fetch, prefix
        // scan and dispatch entirely (see icache.rs for the validity rules).
        let probe_phys = if try_hot { self.icache_phys() } else { None };
        if let Some(phys) = probe_phys {
            let slot = phys as usize & (icache::ICACHE_ENTRIES - 1);
            let e = &self.icache.entries[slot];
            if e.key == (phys as u64 | self.icache_ctx())
                && e.version >= self.icache.stamps[(phys >> 12) as usize & (icache::STAMP_SLOTS - 1)]
                && e.version >= self.icache.inval
                // Per-hit CS-limit revalidation, replacing the per-byte
                // `fetch_check` (a shrink misses here and runs fused → #GP).
                && self.start_eip as u64 + (e.insn.len - 1) as u64
                    <= self.regs.seg[reg::CS as usize].limit as u64
            {
                let d = e.insn;
                #[cfg(any(test, debug_assertions))]
                {
                    self.icache.hits += 1;
                }
                self.stat(|s| s.icache_hits += 1);
                #[cfg(debug_assertions)]
                self.icache_differential(bus, &d);
                self.begin_decoded(&d);
                return self.exec_decoded(bus, &d).map(|c| c + d.npfx() as u32);
            }
        }

        self.ilen = 0;
        self.seg_override = None;
        self.rep = None;
        self.lock = false;
        self.lock_ok = false;
        self.commit_on_fault = false;
        self.cold_saved = false;
        self.supervisor_override = false;

        let (opcode, npfx) = self.scan_prefixes(bus)?;
        let mut cycles = npfx;

        // LOCK legality is a decode-time check: an opcode that can never
        // lock raises #UD before any side effect. Candidate opcodes verify
        // their operand form (memory destination, group sub-opcode) right
        // after ModRM decode via `lock_check`.
        if self.lock && opcode != 0x0F && !LOCK_CANDIDATE[opcode as usize] {
            return Err(Exception::ud());
        }

        // Decoupled decode-then-execute for the hot subset. LOCK/REP forms
        // and everything outside the subset run the fused path unchanged;
        // a decode failure rewinds and re-runs fused, so fault ordering is
        // bit-identical (see decode.rs).
        if try_hot && !self.lock && self.rep.is_none() {
            let mut d = decode::DecodedInsn::default();
            match self.try_decode(bus, opcode, npfx, &mut d)? {
                decode::Decoded::Hot => {
                    if let Some(phys) = probe_phys {
                        self.icache_fill(phys, &d);
                    }
                    return Ok(cycles + self.exec_decoded(bus, &d)?);
                }
                decode::Decoded::Cold0F(op2) => {
                    return Ok(cycles + self.dispatch_0f(bus, op2)?);
                }
                decode::Decoded::Cold => {}
            }
        }

        cycles += self.dispatch(bus, opcode)?;

        // Backstop: a candidate whose handler never validated the prefix.
        if self.lock && !self.lock_ok {
            return Err(Exception::ud());
        }
        Ok(cycles)
    }

    /// Validate the LOCK prefix for a candidate instruction once its operand
    /// is known: legal only with a memory destination (`ok`).
    #[inline]
    pub(crate) fn lock_check(&mut self, ok: bool) -> Exec<()> {
        if self.lock {
            if !ok {
                return Err(Exception::ud());
            }
            self.lock_ok = true;
        }
        Ok(())
    }

    // --- Interrupt line control ---------------------------------------------

    /// Latch a pending NMI (vector 2, not maskable by `IF`).
    pub fn trigger_nmi(&mut self) {
        self.events |= EVT_NMI;
    }

    /// Assert the INTR line with `vector` (as a PIC would supply during the
    /// interrupt-acknowledge cycle). Serviced at the next instruction
    /// boundary with `IF` set.
    pub fn assert_intr(&mut self, vector: u8) {
        self.intr = Some(vector);
        self.events |= EVT_INTR;
    }

    /// Deassert the INTR line.
    pub fn clear_intr(&mut self) {
        self.intr = None;
        self.events &= !EVT_INTR;
    }

    /// Whether an interrupt would be taken at the next instruction boundary.
    /// Long `REP` string operations poll this between iterations so they stay
    /// interruptible, as on hardware.
    #[inline]
    pub(crate) fn interrupt_pending(&self) -> bool {
        self.events & EVT_NMI != 0
            || (self.events & EVT_INTR != 0 && self.regs.eflags.contains(EFlags::IF))
    }

    /// Record a host trap and ring the [`Cpu::run`] doorbell bit.
    #[inline]
    pub(crate) fn set_host_trap(&mut self, t: HostTrap) {
        self.host_trap = Some(t);
        self.events |= EVT_HOST_TRAP;
    }

    // --- Segment-relative memory access -------------------------------------
    // Every instruction memory access funnels through these: segment limit
    // check, base + offset to a linear address, paging, then the bus.

    /// Check that `off .. off+size-1` lies within `seg`'s limit and that the
    /// segment permits the access, raising #GP(0) — or #SS(0) for the stack
    /// segment — otherwise. `write` selects the write-permission rule; reads
    /// only reject execute-only code segments (instruction fetch bypasses
    /// this via [`Cpu::fetch8`]).
    #[inline]
    fn ea_check(&self, seg: u8, off: u32, size: u32, write: bool) -> Exec<()> {
        let s = &self.regs.seg[seg as usize];
        let fault = || {
            if seg == reg::SS {
                Err(Exception::ss(0))
            } else {
                Err(Exception::gp(0))
            }
        };

        // Present/usable (null-loaded segments have attrs == 0).
        if s.attrs & 0x80 == 0 {
            return fault();
        }
        // Type: writes need a writable data segment; reads reject
        // execute-only code. (Real/V86 attrs are writable data.)
        if write {
            if s.attrs & 0x1A != 0x12 {
                return fault();
            }
        } else if s.attrs & 0x0A == 0x08 {
            return fault();
        }

        let last = off as u64 + (size - 1) as u64;
        let ok = if s.expand_down() {
            // Valid offsets are (limit, top]; top is FFFF or FFFFFFFF by D/B.
            let top = if s.db() { u32::MAX as u64 } else { 0xFFFF };
            off as u64 > s.limit as u64 && last <= top
        } else {
            last <= s.limit as u64
        };
        if ok { Ok(()) } else { fault() }
    }

    /// Instruction-fetch check: limit only (execute-only segments fetch fine).
    #[inline]
    fn fetch_check(&self, off: u32) -> Exec<()> {
        let cs = &self.regs.seg[reg::CS as usize];
        if off as u64 <= cs.limit as u64 {
            Ok(())
        } else {
            Err(Exception::gp(0))
        }
    }

    /// Offset wrap mask for data accesses: multi-byte operands wrap at 64 KiB
    /// under a 16-bit address size (8086-compatible), at 4 GiB otherwise.
    #[inline]
    pub(crate) fn data_wrap(&self) -> u32 {
        if self.asize32 { u32::MAX } else { 0xFFFF }
    }

    /// Offset wrap mask for stack accesses (SS.B selects SP vs ESP width).
    #[inline]
    pub(crate) fn stack_wrap(&self) -> u32 {
        if self.regs.seg[reg::SS as usize].db() {
            u32::MAX
        } else {
            0xFFFF
        }
    }

    /// Read one byte at `seg:off`.
    #[inline]
    pub(crate) fn read8<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32) -> Exec<u8> {
        self.ea_check(seg, off, 1, false)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_read8(bus, lin)
    }

    /// Read a little-endian word at `seg:off`, component offsets wrapping at
    /// the data-address width.
    #[inline]
    pub(crate) fn read16<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32) -> Exec<u16> {
        let wrap = self.data_wrap();
        self.read16w(bus, seg, off, wrap)
    }

    /// Read a little-endian double-word at `seg:off` (data-width wrap).
    #[inline]
    pub(crate) fn read32<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32) -> Exec<u32> {
        let wrap = self.data_wrap();
        self.read32w(bus, seg, off, wrap)
    }

    /// Word read with an explicit component-offset wrap mask. A wrapped
    /// access straddling the limit faults (the 386 does not split operands
    /// across a 64 KiB wrap; only *separate* component accesses wrap).
    #[inline]
    pub(crate) fn read16w<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
        wrap: u32,
    ) -> Exec<u16> {
        let _ = wrap;
        self.ea_check(seg, off, 2, false)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_read16(bus, lin)
    }

    /// Double-word read (see [`Cpu::read16w`]).
    #[inline]
    pub(crate) fn read32w<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
        wrap: u32,
    ) -> Exec<u32> {
        let _ = wrap;
        self.ea_check(seg, off, 4, false)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_read32(bus, lin)
    }

    /// Write one byte at `seg:off`.
    #[inline]
    pub(crate) fn write8<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32, v: u8) -> Exec<()> {
        self.ea_check(seg, off, 1, true)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_write8(bus, lin, v)
    }

    /// Write a little-endian word at `seg:off` (data-width wrap).
    #[inline]
    pub(crate) fn write16<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32, v: u16) -> Exec<()> {
        let wrap = self.data_wrap();
        self.write16w(bus, seg, off, v, wrap)
    }

    /// Write a little-endian double-word at `seg:off` (data-width wrap).
    #[inline]
    pub(crate) fn write32<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32, v: u32) -> Exec<()> {
        let wrap = self.data_wrap();
        self.write32w(bus, seg, off, v, wrap)
    }

    /// Word write with an explicit component-offset wrap mask (straddles
    /// fault; see [`Cpu::read16w`]).
    #[inline]
    pub(crate) fn write16w<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
        v: u16,
        wrap: u32,
    ) -> Exec<()> {
        let _ = wrap;
        self.ea_check(seg, off, 2, true)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_write16(bus, lin, v)
    }

    /// Double-word write (see [`Cpu::read16w`]).
    #[inline]
    pub(crate) fn write32w<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
        v: u32,
        wrap: u32,
    ) -> Exec<()> {
        let _ = wrap;
        self.ea_check(seg, off, 4, true)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_write32(bus, lin, v)
    }

    /// The effective segment for a data access whose default segment is
    /// `default`, honoring any override prefix.
    #[inline]
    pub(crate) fn seg_or(&self, default: u8) -> u8 {
        self.seg_override.unwrap_or(default)
    }

    // --- Instruction fetch ----------------------------------------------------

    /// Fetch the byte at `CS:EIP` and advance `EIP`, enforcing the 15-byte
    /// instruction length limit.
    #[inline]
    pub(crate) fn fetch8<B: Bus>(&mut self, bus: &mut B) -> Exec<u8> {
        if self.ilen >= 15 {
            return Err(Exception::ud());
        }
        self.ilen += 1;
        self.fetch_check(self.regs.eip)?;
        let lin = self.regs.seg[reg::CS as usize]
            .base
            .wrapping_add(self.regs.eip);
        let b = self.lin_read8(bus, lin)?;
        self.regs.eip = self.regs.eip.wrapping_add(1);
        Ok(b)
    }

    /// Fetch a little-endian word at `CS:EIP`.
    #[inline]
    pub(crate) fn fetch16<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        let lo = self.fetch8(bus)? as u16;
        let hi = self.fetch8(bus)? as u16;
        Ok(lo | hi << 8)
    }

    /// Fetch a little-endian double-word at `CS:EIP`.
    #[inline]
    pub(crate) fn fetch32<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        let lo = self.fetch16(bus)? as u32;
        let hi = self.fetch16(bus)? as u32;
        Ok(lo | hi << 16)
    }

    /// Fetch an immediate of the current operand size, sign-extending a
    /// 16-bit immediate to 32 bits.
    #[inline]
    pub(crate) fn fetch_imm<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        self.stat(|s| s.imm_fetches += 1);
        if self.osize32 {
            self.fetch32(bus)
        } else {
            Ok(self.fetch16(bus)? as u32)
        }
    }

    // --- Stack ----------------------------------------------------------------
    // SS.B selects whether SP or ESP is the stack pointer; the operand size
    // of the push/pop selects the datum width independently.

    /// Current stack pointer honoring SS.B.
    #[inline]
    pub(crate) fn stack_ptr(&self) -> u32 {
        let sp = self.regs.gpr[reg::ESP as usize];
        if self.regs.seg[reg::SS as usize].db() {
            sp
        } else {
            sp & 0xFFFF
        }
    }

    /// Adjust the stack pointer by `delta` (SP or ESP per SS.B).
    #[inline]
    pub(crate) fn adjust_sp(&mut self, delta: i32) {
        let r = &mut self.regs.gpr[reg::ESP as usize];
        if self.regs.seg[reg::SS as usize].db() {
            *r = r.wrapping_add(delta as u32);
        } else {
            *r = (*r & 0xFFFF_0000) | (*r as u16).wrapping_add(delta as u16) as u32;
        }
    }

    /// Push a word onto the stack.
    #[inline]
    pub(crate) fn push16<B: Bus>(&mut self, bus: &mut B, v: u16) -> Exec<()> {
        let wrap = self.stack_wrap();
        let sp = self.stack_ptr().wrapping_sub(2) & wrap;
        self.write16w(bus, reg::SS, sp, v, wrap)?;
        self.adjust_sp(-2);
        Ok(())
    }

    /// Push a double-word onto the stack.
    #[inline]
    pub(crate) fn push32<B: Bus>(&mut self, bus: &mut B, v: u32) -> Exec<()> {
        let wrap = self.stack_wrap();
        let sp = self.stack_ptr().wrapping_sub(4) & wrap;
        self.write32w(bus, reg::SS, sp, v, wrap)?;
        self.adjust_sp(-4);
        Ok(())
    }

    /// Push an operand-sized value.
    #[inline]
    pub(crate) fn push<B: Bus>(&mut self, bus: &mut B, v: u32) -> Exec<()> {
        if self.osize32 {
            self.push32(bus, v)
        } else {
            self.push16(bus, v as u16)
        }
    }

    /// Pop a word off the stack.
    #[inline]
    pub(crate) fn pop16<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        let wrap = self.stack_wrap();
        let v = self.read16w(bus, reg::SS, self.stack_ptr(), wrap)?;
        self.adjust_sp(2);
        Ok(v)
    }

    /// Pop a double-word off the stack.
    #[inline]
    pub(crate) fn pop32<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        let wrap = self.stack_wrap();
        let v = self.read32w(bus, reg::SS, self.stack_ptr(), wrap)?;
        self.adjust_sp(4);
        Ok(v)
    }

    /// Pop an operand-sized value (zero-extended).
    #[inline]
    pub(crate) fn pop<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        if self.osize32 {
            self.pop32(bus)
        } else {
            Ok(self.pop16(bus)? as u32)
        }
    }

    /// Pop a segment selector: a 32-bit operand size still performs only a
    /// 16-bit read (no fault at SP = FFFE on a 64 KiB stack) but releases
    /// four bytes — 386 behavior asserted by the hardware test suite.
    pub(crate) fn pop_sreg<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        let wrap = self.stack_wrap();
        let v = self.read16w(bus, reg::SS, self.stack_ptr(), wrap)?;
        self.adjust_sp(if self.osize32 { 4 } else { 2 });
        Ok(v)
    }

    // --- Interrupt and exception delivery --------------------------------------

    /// Exceptions Intel classifies as *contributory* for double-fault
    /// detection (SDM Table 6-5). `#PF` forms its own class; every other
    /// exception — and all external interrupts and `INT n` — is benign.
    fn contributory(vector: u8) -> bool {
        matches!(vector, 0 | 10 | 11 | 12 | 13)
    }

    /// Whether `second`, raised while delivering `first`, escalates to `#DF`.
    ///
    /// Only a processor-detected exception can contribute: the vector of an
    /// external interrupt or `INT n` says nothing about its class, so those
    /// are always benign and their nested faults are handled serially.
    fn is_double_fault(first: Exception, class: Event, second: u8) -> bool {
        if class != Event::Fault {
            return false;
        }
        let first_contributory = Self::contributory(first.vector);
        let first_page_fault = first.vector == 14;
        if second == 14 {
            // Contributory → #PF is handled serially; only #PF → #PF faults.
            first_page_fault
        } else if Self::contributory(second) {
            first_contributory || first_page_fault
        } else {
            false
        }
    }

    /// Deliver exception/interrupt `e` at the current `CS:EIP`.
    ///
    /// A fault raised while delivering `e` is handled serially unless the
    /// pair forms a double fault; a fault while delivering `#DF` is a triple
    /// fault and shuts the processor down.
    #[cold]
    pub(crate) fn deliver<B: Bus>(&mut self, bus: &mut B, e: Exception, class: Event) {
        let (mut current, mut class) = (e, class);
        let mut in_double_fault = false;
        // Serial handling terminates on its own (nested vectors are all
        // contributory or #PF), but bound the chain regardless.
        for _ in 0..4 {
            let Err(nested) = self.raise(bus, current, class) else {
                return;
            };
            if in_double_fault {
                break; // triple fault
            }
            if Self::is_double_fault(current, class, nested.vector) {
                current = Exception {
                    vector: 8,
                    error: Some(0),
                };
                in_double_fault = true;
            } else {
                current = nested;
            }
            class = Event::Fault;
        }
        self.shutdown = true;
        self.halted = true;
    }

    /// The fallible part of delivery: real mode uses the IVT at `IDTR.base`;
    /// protected mode — including V86 — goes through the IDT gates (see
    /// `protected.rs`).
    ///
    /// Delivery is atomic: if it faults partway through, every register side
    /// effect is rolled back (CR2 excepted, so a nested `#PF` keeps its
    /// address) and the nested fault is delivered from the architectural
    /// state the CPU had before delivery began — otherwise a partially
    /// applied frame, a cleared `VM`, or a switched stack would leak into the
    /// handler.
    #[cold]
    pub(crate) fn raise<B: Bus>(&mut self, bus: &mut B, e: Exception, class: Event) -> Exec<()> {
        let snapshot = self.regs;
        let sup = self.supervisor_override;
        let r = if self.regs.cr0 & cr0::PE != 0 {
            self.interrupt_protected(bus, e, class)
        } else {
            self.interrupt_real(bus, e.vector, class)
        };
        self.supervisor_override = sup;
        if r.is_err() {
            let cr2 = self.regs.cr2;
            self.regs = snapshot;
            self.regs.cr2 = cr2;
        }
        r
    }

    /// Real-mode interrupt: push FLAGS/CS/IP (16-bit), clear IF/TF, and load
    /// `CS:IP` from the vector table described by IDTR.
    fn interrupt_real<B: Bus>(&mut self, bus: &mut B, vector: u8, class: Event) -> Exec<()> {
        self.prepare_cold_write(); // CS cache
        let entry = vector as u32 * 4;
        if entry + 3 > self.regs.idtr.limit as u32 {
            let ext = (class == Event::External) as u16;
            return Err(Exception::gp(vector as u16 * 8 + 2 + ext));
        }
        let base = self.regs.idtr.base;
        let ip = self.sys_read16(bus, base.wrapping_add(entry))?;
        let cs = self.sys_read16(bus, base.wrapping_add(entry + 2))?;

        let flags = self.regs.eflags.image16();
        self.push16(bus, flags)?;
        let (old_cs, old_ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.eip);
        self.push16(bus, old_cs)?;
        self.push16(bus, old_ip as u16)?;
        self.regs.eflags.remove(EFlags::IF | EFlags::TF);
        self.regs.seg[reg::CS as usize].sel = cs;
        self.regs.seg[reg::CS as usize].base = (cs as u32) << 4;
        self.regs.eip = ip as u32;
        Ok(())
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}
