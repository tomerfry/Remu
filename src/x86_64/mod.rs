//! The x86-64 (AMD64) CPU core: sixteen 64-bit registers, REX prefixes, real
//! mode, protected/compatibility mode and 64-bit long mode, 4-level paging
//! with NX, and the SYSCALL/MSR system interface.
//!
//! Like the other cores, it is deliberately self-contained (own [`Bus`]
//! trait, own register file) so promoting each core to its own crate later
//! stays mechanical.
//!
//! The integer instruction set is complete through the x86-64 baseline
//! (CMOVcc, CMPXCHG8B/16B, BSWAP, long NOPs); x87/SSE state does not exist
//! (ESC opcodes are consumed as no-ops, SSE encodings raise #UD) and CPUID
//! honestly reports that. Where hardware leaves flags or results *undefined*
//! this core picks deterministic values resembling common silicon — there is
//! no SingleStepTests suite for x86-64 to pin them against, unlike the 8088
//! and 80386 cores.
//!
//! ```
//! use remu::x86_64::{Cpu, Bus, LinearMemory};
//!
//! let mut mem = LinearMemory::new();
//! mem.load(0x1100, &[0x48, 0xC7, 0xC0, 0x78, 0x56, 0x34, 0x12]); // MOV RAX, 0x12345678
//! let mut cpu = Cpu::new();
//! cpu.setup_long_flat(&mut mem, 0x1100, 0x8_0000);
//! cpu.step(&mut mem);
//! assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
//! ```

mod alu;
mod decode;
mod execute;
mod execute_0f;
mod icache;
mod modrm;
mod msr;
mod paging;
mod protected;
pub mod registers;

pub use registers::{DescTable, Msrs, RFlags, Registers, SegReg, cr0, cr4, efer, reg};

/// The memory and I/O bus.
///
/// Memory addresses are 64-bit physical (the core masks them to its 52-bit
/// physical address width). The wide accessors have byte-composed defaults so
/// simple devices only implement `read`/`write`; RAM-backed buses should
/// override them for speed. The CPU only issues a wide access when it does
/// not cross a page boundary, so overrides may assume contiguity.
pub trait Bus {
    /// Read one byte from physical address `addr`.
    fn read(&mut self, addr: u64) -> u8;

    /// Write `value` to physical address `addr`.
    fn write(&mut self, addr: u64, value: u8);

    /// Read a little-endian word from `addr`.
    #[inline]
    fn read16(&mut self, addr: u64) -> u16 {
        self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
    }

    /// Read a little-endian double-word from `addr`.
    #[inline]
    fn read32(&mut self, addr: u64) -> u32 {
        self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
    }

    /// Read a little-endian quad-word from `addr`.
    #[inline]
    fn read64(&mut self, addr: u64) -> u64 {
        self.read32(addr) as u64 | (self.read32(addr.wrapping_add(4)) as u64) << 32
    }

    /// Write a little-endian word to `addr`.
    #[inline]
    fn write16(&mut self, addr: u64, value: u16) {
        self.write(addr, value as u8);
        self.write(addr.wrapping_add(1), (value >> 8) as u8);
    }

    /// Write a little-endian double-word to `addr`.
    #[inline]
    fn write32(&mut self, addr: u64, value: u32) {
        self.write16(addr, value as u16);
        self.write16(addr.wrapping_add(2), (value >> 16) as u16);
    }

    /// Write a little-endian quad-word to `addr`.
    #[inline]
    fn write64(&mut self, addr: u64, value: u64) {
        self.write32(addr, value as u32);
        self.write32(addr.wrapping_add(4), (value >> 32) as u32);
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

/// Default [`LinearMemory`] size (16 MiB) — roomy enough for page tables plus
/// test programs while staying cheap to allocate.
const DEFAULT_MEMORY: usize = 16 << 20;

/// A flat power-of-two RAM — the simplest possible machine, for tests and raw
/// programs. Addresses wrap at the memory size.
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

    /// Load `data` into memory starting at physical address `addr`.
    pub fn load(&mut self, addr: u64, data: &[u8]) {
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
    fn read(&mut self, addr: u64) -> u8 {
        self.ram[addr as usize & self.mask]
    }

    fn write(&mut self, addr: u64, value: u8) {
        self.ram[addr as usize & self.mask] = value;
    }

    fn read16(&mut self, addr: u64) -> u16 {
        let a = addr as usize & self.mask;
        if a < self.mask {
            u16::from_le_bytes(self.ram[a..a + 2].try_into().unwrap())
        } else {
            self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
        }
    }

    fn read32(&mut self, addr: u64) -> u32 {
        let a = addr as usize & self.mask;
        if a + 3 <= self.mask {
            u32::from_le_bytes(self.ram[a..a + 4].try_into().unwrap())
        } else {
            self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
        }
    }

    fn read64(&mut self, addr: u64) -> u64 {
        let a = addr as usize & self.mask;
        if a + 7 <= self.mask {
            u64::from_le_bytes(self.ram[a..a + 8].try_into().unwrap())
        } else {
            self.read32(addr) as u64 | (self.read32(addr.wrapping_add(4)) as u64) << 32
        }
    }

    fn write16(&mut self, addr: u64, value: u16) {
        let a = addr as usize & self.mask;
        if a < self.mask {
            self.ram[a..a + 2].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write(addr, value as u8);
            self.write(addr.wrapping_add(1), (value >> 8) as u8);
        }
    }

    fn write32(&mut self, addr: u64, value: u32) {
        let a = addr as usize & self.mask;
        if a + 3 <= self.mask {
            self.ram[a..a + 4].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write16(addr, value as u16);
            self.write16(addr.wrapping_add(2), (value >> 16) as u16);
        }
    }

    fn write64(&mut self, addr: u64, value: u64) {
        let a = addr as usize & self.mask;
        if a + 7 <= self.mask {
            self.ram[a..a + 8].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write32(addr, value as u32);
            self.write32(addr.wrapping_add(4), (value >> 32) as u32);
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
/// place of the normal in-guest delivery.
///
/// Populated in [`Cpu::host_trap`] only when the corresponding opt-in is set
/// ([`Cpu::syscall_int`] / [`Cpu::trap_syscall`] / [`Cpu::trap_faults`]); the
/// CPU is otherwise plain bare metal. A userspace emulator drives
/// [`Cpu::step`] and, after each call, takes any pending trap to service it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTrap {
    /// The guest executed `INT n` for the vector named by [`Cpu::syscall_int`]
    /// (e.g. `int 0x80`), or the `SYSCALL` instruction while
    /// [`Cpu::trap_syscall`] was set. `RIP` already points past the
    /// instruction; read the syscall number/arguments from the registers and
    /// write the result to `RAX`.
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

/// Cycles consumed by servicing a hardware interrupt (nominal figure).
const INTERRUPT_CYCLES: u32 = 40;

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

/// Effective operand size of the current instruction.
///
/// Legacy/compat modes pick 16 or 32 from CS.D xor the `66` prefix; 64-bit
/// mode defaults to 32 with `66` selecting 16 and REX.W selecting 64
/// (REX.W wins over `66`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpSize {
    O16,
    O32,
    O64,
}

pub(crate) use OpSize::{O16, O32, O64};

/// Decode/dispatch profile counters, collected only with the `perf-stats`
/// feature. They quantify per-instruction decode work (the target of the
/// decoded-instruction-cache effort) and are reported by
/// [`PerfStats::report`].
#[derive(Debug, Clone)]
pub struct PerfStats {
    /// Instructions executed (`exec_one` entries).
    pub insns: u64,
    /// Prefix bytes consumed by the decode loop (REX included).
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

/// An x86-64 processor.
///
/// As with the other cores, the CPU does not own its bus — call [`Cpu::step`]
/// with a `&mut B: Bus`.
#[derive(Debug, Clone)]
pub struct Cpu {
    /// The register file.
    pub regs: Registers,
    /// Total cycles elapsed since construction/reset. Timing is
    /// instruction-atomic using nominal per-instruction costs — modern
    /// pipelines make exact interpreter cycle counting meaningless, so these
    /// are rough relative weights, not validated figures.
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
    /// PIC/APIC during the acknowledge cycle). `Some` iff `EVT_INTR` is set.
    intr: Option<u8>,

    // --- Per-instruction decode state ---------------------------------------
    /// Segment-override prefix (segment register index). In 64-bit mode only
    /// FS/GS overrides have an effect, but all six are still decoded.
    seg_override: Option<u8>,
    /// Repeat prefix: `true` for `REP`/`REPE` (F3), `false` for `REPNE` (F2).
    rep: Option<bool>,
    /// A LOCK prefix was decoded.
    lock: bool,
    /// The current instruction legally used its LOCK prefix (set by handlers
    /// of lockable instructions with a memory destination).
    lock_ok: bool,
    /// REX prefix byte (64-bit mode only; `None` when absent). Only a REX
    /// that immediately precedes the opcode counts — any later legacy prefix
    /// voids it, as on hardware.
    rex: Option<u8>,
    /// A `66` operand-size prefix was decoded (needed independently of the
    /// resolved [`OpSize`] for the default-64 promotions).
    prefix66: bool,
    /// Effective operand size (see [`OpSize`]).
    osize: OpSize,
    /// Effective address size: `O64`/`O32` in 64-bit mode (`67` selects 32),
    /// `O32`/`O16` in legacy modes.
    asize: OpSize,
    /// Executing 64-bit code (EFER.LMA and CS.L) — refreshed per instruction.
    m64: bool,
    /// RIP at the start of the current instruction (fault reporting).
    start_rip: u64,
    /// A fault should keep the current register state and only rewind RIP —
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
    /// One-entry fetch-translation cache: linear page number whose
    /// translation is cached (`u64::MAX` = empty). Fetches on this page skip
    /// the TLB lookup entirely; a hot loop in one page pays zero translates.
    /// Invalidated by [`Cpu::fetch_invalidate`] — called from every TLB
    /// flush, `INVLPG`, and `prepare_cold_write` (whose callers are a
    /// superset of everything that can change CPL, mode, or CS).
    fetch_tag: u64,
    /// Physical page base for `fetch_tag`.
    fetch_page: u64,
    /// Length in bytes of the immediate that follows the ModRM/displacement
    /// of the current instruction — set by the dispatcher *before* ModRM
    /// decode, because a RIP-relative displacement is relative to the end of
    /// the whole instruction, immediate included.
    imm_len: u8,
    /// The current instruction's memory operand used RIP-relative
    /// addressing (lets `F6`/`F7 TEST` fix up the EA once the group
    /// sub-opcode — and with it the immediate length — is known).
    used_rip_rel: bool,
    /// Forces paging to treat the current access as a supervisor access
    /// regardless of CPL. The CPU performs its *implicit* accesses — reads of
    /// the descriptor tables, the IDT and the TSS, the accessed/busy bit
    /// writebacks, and the frame pushes onto an inner-privilege stack — with
    /// supervisor privilege even while CPL is still 3.
    supervisor_override: bool,
    /// TLB for paged address translation (see `paging.rs`).
    tlb: paging::Tlb,
    /// Decoded-instruction cache (see `icache.rs`).
    icache: icache::ICache,

    // --- Host (OS-emulation) hooks — all inert at their defaults -------------
    /// If set, `INT n` for this vector does not vector through the IDT;
    /// instead the CPU records [`HostTrap::Syscall`] and returns, letting a
    /// host syscall layer service it (e.g. `Some(0x80)` for Linux i386
    /// compatibility). Default `None` — `INT` behaves exactly like hardware.
    pub syscall_int: Option<u8>,
    /// If `true`, the `SYSCALL` instruction records [`HostTrap::Syscall`]
    /// instead of vectoring through the STAR/LSTAR MSRs (the natural hook for
    /// a Linux x86-64 OS-emulation layer). `RCX`/`R11` are still loaded with
    /// the return RIP and RFLAGS, as user code may rely on them. Default
    /// `false`.
    pub trap_syscall: bool,
    /// If `true`, a CPU exception is handed back via [`HostTrap::Exception`]
    /// instead of being delivered through the IDT (register state is rewound
    /// first). A userspace emulator has no guest kernel behind the IDT, so it
    /// resolves faults on the host. Default `false`.
    pub trap_faults: bool,
    /// Output: the pending host request, set by the CPU when [`syscall_int`]
    /// / [`trap_syscall`] matches or a fault is trapped. The host takes it
    /// after each [`Cpu::step`]. Cleared on RESET.
    ///
    /// [`syscall_int`]: Cpu::syscall_int
    /// [`trap_syscall`]: Cpu::trap_syscall
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
    /// Create a CPU in its power-on state: real mode, execution begins at
    /// `F000:FFF0` (CS base `FFFF0000`).
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
            rex: None,
            prefix66: false,
            osize: O16,
            asize: O16,
            m64: false,
            start_rip: 0,
            commit_on_fault: false,
            ilen: 0,
            fault_regs: Registers::new(),
            cold_saved: false,
            fetch_tag: u64::MAX,
            fetch_page: 0,
            imm_len: 0,
            used_rip_rel: false,
            supervisor_override: false,
            tlb: paging::Tlb::new(),
            icache: icache::ICache::new(),
            syscall_int: None,
            trap_syscall: false,
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

    /// Perform a RESET: registers to power-on state, pending interrupts and
    /// shutdown cleared.
    pub fn reset(&mut self) {
        self.regs = Registers::new();
        self.halted = false;
        self.shutdown = false;
        self.events = 0;
        self.intr = None;
        self.supervisor_override = false;
        self.fetch_invalidate();
        self.tlb.flush();
        self.icache.invalidate_all();
        self.host_trap = None;
    }

    /// Convenience for tests and loaders: set `CS:RIP` with real-mode
    /// semantics (CS base = `sel * 16`).
    pub fn set_cs_ip(&mut self, sel: u16, rip: u64) {
        self.fetch_invalidate();
        self.regs.seg[reg::CS as usize] = SegReg::real(sel);
        self.regs.rip = rip;
    }

    /// Convenience for tests and loaders: drop the CPU directly into 64-bit
    /// long mode with flat ring-0 segments, RIP at `rip`, RSP at `rsp`, and
    /// identity page tables for the low 512 GiB built at physical `0x1000`
    /// (CR3 = `0x1000`; two pages of tables using 1 GiB mappings).
    ///
    /// Long mode *requires* paging, so this writes the minimal table set
    /// through `bus` — real firmware does the same dance, just less tersely.
    pub fn setup_long_flat<B: Bus>(&mut self, bus: &mut B, rip: u64, rsp: u64) {
        self.fetch_invalidate();
        // The table writes below go straight to the bus (host-side writes).
        self.icache.invalidate_all();
        // PML4[0] -> PDPT at 0x2000; PDPT[n] = n GiB, present/write/user/PS.
        // The user bit keeps ring-3 test code runnable through the flat map.
        bus.write64(0x1000, 0x2000 | 0x07);
        for i in 0..512u64 {
            bus.write64(0x2000 + i * 8, (i << 30) | 0x87);
        }
        self.regs.cr3 = 0x1000;
        self.regs.cr4 |= cr4::PAE;
        self.regs.cr0 |= cr0::PE | cr0::PG;
        self.regs.cr0 &= !(cr0::CD | cr0::NW);
        self.regs.msr.efer |= efer::LME | efer::LMA | efer::SCE;
        self.regs.seg[reg::CS as usize] = SegReg {
            sel: 0x08,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: 0x0A9B, // present, code, exec/read, accessed; G+L
        };
        for idx in [reg::SS, reg::DS, reg::ES, reg::FS, reg::GS] {
            self.regs.seg[idx as usize] = SegReg {
                sel: 0x10,
                base: 0,
                limit: 0xFFFF_FFFF,
                attrs: 0x0C93, // present, data, read/write, accessed; G+D
            };
        }
        self.regs.rip = rip;
        self.regs.gpr[reg::RSP as usize] = rsp;
        self.tlb.flush();
    }

    /// True while EFER.LMA is set (long mode active — 64-bit or
    /// compatibility submode).
    #[inline]
    pub fn long_mode(&self) -> bool {
        self.regs.msr.efer & efer::LMA != 0
    }

    /// True when the *current* code segment executes 64-bit code.
    #[inline]
    pub fn mode64(&self) -> bool {
        self.long_mode() && self.regs.seg[reg::CS as usize].l()
    }

    /// True once `CR0.PE` is set and the CPU is not in Virtual-8086 mode.
    #[inline]
    pub fn protected_mode(&self) -> bool {
        self.regs.cr0 & cr0::PE != 0 && !self.regs.rflags.contains(RFlags::VM)
    }

    /// Current privilege level (CPL). Real mode and V86 report their fixed
    /// levels through the CS cache attributes maintained by segment loads.
    #[inline]
    pub fn cpl(&self) -> u8 {
        if self.regs.rflags.contains(RFlags::VM) {
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
    /// guest-visible in RFLAGS (all writable by embedders and tests
    /// directly), so they are re-read here rather than mirrored into
    /// `events` — a mirror of a `pub` field can go stale.
    #[inline(always)]
    fn boundary_pending(&self) -> bool {
        (self.events != 0)
            || self.halted
            || self.shutdown
            || (self.regs.rflags.bits() & RFlags::TF.bits() != 0)
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
                && self.regs.rflags.contains(RFlags::IF)
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
        let trap = self.regs.rflags.contains(RFlags::TF);

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

        if trap && self.regs.rflags.contains(RFlags::TF) && self.events & EVT_INHIBIT == 0 {
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
        // the GPRs are snapshotted here: RIP rewinds via `start_rip`, RFLAGS
        // keeps whatever the faulting computation left behind, CR2 keeps the
        // page-fault address, and every other field is cold — its writers
        // call `prepare_cold_write` first, which captures the full register
        // file.
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
        saved_gpr: &[u64; 16],
        #[cfg(debug_assertions)] saved_all: &Registers,
    ) -> Option<u32> {
        if self.commit_on_fault {
            // String-op progress stays; only RIP rewinds.
            self.regs.rip = self.start_rip;
        } else {
            if self.cold_saved {
                let (rflags, cr2) = (self.regs.rflags, self.regs.cr2);
                self.regs = self.fault_regs;
                self.regs.rflags = rflags;
                self.regs.cr2 = cr2;
            }
            self.regs.gpr = *saved_gpr;
            self.regs.rip = self.start_rip;
            #[cfg(debug_assertions)]
            {
                // Differential check against the old whole-file
                // rewind: a mismatch means a cold-register writer is
                // missing its `prepare_cold_write` call.
                let mut want = *saved_all;
                want.rflags = self.regs.rflags;
                want.cr2 = self.regs.cr2;
                assert_eq!(
                    self.regs, want,
                    "fault rewind mismatch at {:#x}",
                    self.start_rip
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
    /// (anything besides `gpr`/`rip`/`rflags`/`cr2`) during the current
    /// instruction, so a later fault in the same instruction can rewind it.
    /// Every function that writes such a register must call this first.
    ///
    /// Doubles as the fetch-cache invalidation point: cold writers are a
    /// superset of everything that can change CPL, the execution mode, or
    /// the CS base under the cached fetch translation.
    #[inline]
    pub(crate) fn prepare_cold_write(&mut self) {
        self.fetch_invalidate();
        if !self.cold_saved {
            self.cold_saved = true;
            self.fault_regs = self.regs;
        }
    }

    /// Drop the cached fetch translation (page-table or privilege change).
    #[inline]
    pub(crate) fn fetch_invalidate(&mut self) {
        self.fetch_tag = u64::MAX;
    }

    /// Decode prefixes and execute the instruction at `CS:RIP`.
    fn exec_one<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        self.start_rip = self.regs.rip;
        // Refreshed before the probe: the context key, the probe's address
        // masking and every handler read it.
        self.m64 = self.mode64();
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
            if e.key == (phys | self.icache_ctx())
                && e.version >= self.icache.stamps[(phys >> 12) as usize & (icache::STAMP_SLOTS - 1)]
                && e.version >= self.icache.inval
                // Per-hit revalidation replacing the per-byte fetch_check:
                // canonicality in 64-bit mode (the canonical boundary is
                // page-aligned, and page-crossers are never cached), the CS
                // limit in legacy/compat modes.
                && if self.m64 {
                    Self::canonical(
                        self.regs.seg[reg::CS as usize]
                            .base
                            .wrapping_add(self.start_rip),
                    )
                } else {
                    self.start_rip.saturating_add((e.insn.len - 1) as u64)
                        <= self.regs.seg[reg::CS as usize].limit as u64
                }
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
        self.rex = None;
        self.prefix66 = false;
        self.commit_on_fault = false;
        self.cold_saved = false;
        self.supervisor_override = false;
        self.imm_len = 0;
        self.used_rip_rel = false;

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

    /// Consume the prefix (and REX) bytes at `CS:RIP`, recording their
    /// effects and resolving the effective sizes; returns the opcode byte
    /// and the number of *legacy* prefixes consumed (REX bytes cost no
    /// cycle, as before). The caller resets the remaining decode state and
    /// refreshes `m64` first.
    #[inline(always)]
    fn scan_prefixes<B: Bus>(&mut self, bus: &mut B) -> Exec<(u8, u32)> {
        let db = self.regs.seg[reg::CS as usize].db();
        let mut p66 = false;
        let mut p67 = false;

        let mut npfx = 0u32;
        let opcode = loop {
            let b = self.fetch8(bus)?;
            match b {
                // In 64-bit mode 40–4F are REX prefixes; a REX only counts
                // when it immediately precedes the opcode, so any later
                // prefix byte voids the recorded one.
                0x40..=0x4F if self.m64 => {
                    self.rex = Some(b);
                    self.stat(|s| s.prefix_bytes += 1);
                    continue;
                }
                0x26 => self.seg_override = Some(reg::ES),
                0x2E => self.seg_override = Some(reg::CS),
                0x36 => self.seg_override = Some(reg::SS),
                0x3E => self.seg_override = Some(reg::DS),
                0x64 => self.seg_override = Some(reg::FS),
                0x65 => self.seg_override = Some(reg::GS),
                0x66 => p66 = true,
                0x67 => p67 = true,
                0xF0 => self.lock = true,
                0xF2 => self.rep = Some(false),
                0xF3 => self.rep = Some(true),
                _ => break b,
            }
            self.rex = None;
            self.stat(|s| s.prefix_bytes += 1);
            npfx += 1;
        };
        self.prefix66 = p66;

        // Resolve the effective sizes.
        if self.m64 {
            self.osize = if self.rex.is_some_and(|r| r & 8 != 0) {
                O64
            } else if p66 {
                O16
            } else {
                O32
            };
            self.asize = if p67 { O32 } else { O64 };
            // In 64-bit mode only FS/GS overrides change anything; the four
            // legacy overrides are accepted and ignored.
            if let Some(s) = self.seg_override
                && s != reg::FS
                && s != reg::GS
            {
                self.seg_override = None;
            }
        } else {
            self.osize = if db != p66 { O32 } else { O16 };
            self.asize = if db != p67 { O32 } else { O16 };
        }
        Ok((opcode, npfx))
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

    // --- REX accessors --------------------------------------------------------

    /// REX.W (also folded into [`Cpu::osize`]).
    #[inline]
    pub(crate) fn rex_w(&self) -> bool {
        self.rex.is_some_and(|r| r & 8 != 0)
    }

    /// REX.R: extends the ModRM `reg` field.
    #[inline]
    pub(crate) fn rex_r(&self) -> u8 {
        self.rex.map_or(0, |r| (r >> 2) & 1)
    }

    /// REX.X: extends the SIB `index` field.
    #[inline]
    pub(crate) fn rex_x(&self) -> u8 {
        self.rex.map_or(0, |r| (r >> 1) & 1)
    }

    /// REX.B: extends ModRM `rm`, SIB `base`, and opcode-embedded registers.
    #[inline]
    pub(crate) fn rex_b(&self) -> u8 {
        self.rex.map_or(0, |r| r & 1)
    }

    /// Read an 8-bit register honoring the REX high-byte rule.
    #[inline]
    pub(crate) fn gpr8(&self, i: u8) -> u8 {
        self.regs.reg8(i, self.rex.is_some())
    }

    /// Write an 8-bit register honoring the REX high-byte rule.
    #[inline]
    pub(crate) fn set_gpr8(&mut self, i: u8, v: u8) {
        self.regs.set_reg8(i, self.rex.is_some(), v);
    }

    // --- Default-64 operand-size promotions ------------------------------------

    /// Operand size for stack operations (PUSH/POP/PUSHF/ENTER/...): 64-bit
    /// mode defaults to 64 and cannot encode 32 (`66` still selects 16).
    #[inline]
    pub(crate) fn stack_osize(&self) -> OpSize {
        if self.m64 {
            if self.prefix66 && !self.rex_w() {
                O16
            } else {
                O64
            }
        } else {
            self.osize
        }
    }

    /// Operand size for near branches (JMP/CALL/RET/Jcc/LOOP): forced to 64
    /// in 64-bit mode (`66` is ignored, as on Intel silicon; AMD instead
    /// truncates RIP — this core follows Intel).
    #[inline]
    pub(crate) fn branch_osize(&self) -> OpSize {
        if self.m64 { O64 } else { self.osize }
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
            || (self.events & EVT_INTR != 0 && self.regs.rflags.contains(RFlags::IF))
    }

    /// Record a host trap and ring the [`Cpu::run`] doorbell bit.
    #[inline]
    pub(crate) fn set_host_trap(&mut self, t: HostTrap) {
        self.host_trap = Some(t);
        self.events |= EVT_HOST_TRAP;
    }

    // --- Canonical addresses -----------------------------------------------------

    /// Whether `addr` is canonical (bits 63:47 all equal bit 47) under the
    /// core's 48-bit virtual address width.
    #[inline]
    pub(crate) fn canonical(addr: u64) -> bool {
        let top = addr >> 47;
        top == 0 || top == 0x1_FFFF
    }

    // --- Segment-relative memory access -------------------------------------
    // Every instruction memory access funnels through these. Legacy modes
    // apply the 386-style segment limit/type checks; 64-bit mode ignores
    // limits, treats CS/DS/ES/SS bases as zero, keeps the FS/GS bases, and
    // requires canonical linear addresses.

    /// Resolve `seg:off` to a linear address for a `size`-byte access,
    /// applying the mode's protection rules. `write` selects the
    /// write-permission rule in legacy modes.
    #[inline]
    fn lin_addr(&self, seg: u8, off: u64, size: u32, write: bool) -> Exec<u64> {
        if self.m64 {
            let base = if seg == reg::FS || seg == reg::GS {
                self.regs.seg[seg as usize].base
            } else {
                0
            };
            let lin = base.wrapping_add(off);
            let last = lin.wrapping_add(size as u64 - 1);
            if !Self::canonical(lin) || !Self::canonical(last) {
                return Err(if seg == reg::SS {
                    Exception::ss(0)
                } else {
                    Exception::gp(0)
                });
            }
            Ok(lin)
        } else {
            self.ea_check(seg, off as u32, size, write)?;
            Ok((self.regs.seg[seg as usize].base as u32).wrapping_add(off as u32) as u64)
        }
    }

    /// Check that `off .. off+size-1` lies within `seg`'s limit and that the
    /// segment permits the access, raising #GP(0) — or #SS(0) for the stack
    /// segment — otherwise (legacy modes only). `write` selects the
    /// write-permission rule; reads only reject execute-only code segments
    /// (instruction fetch bypasses this via [`Cpu::fetch8`]).
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

    /// Instruction-fetch check: canonical RIP in 64-bit mode, CS limit in
    /// legacy modes (execute-only segments fetch fine).
    #[inline]
    fn fetch_check(&self, off: u64) -> Exec<()> {
        if self.m64 {
            if Self::canonical(self.regs.seg[reg::CS as usize].base.wrapping_add(off)) {
                Ok(())
            } else {
                Err(Exception::gp(0))
            }
        } else if off <= self.regs.seg[reg::CS as usize].limit as u64 {
            Ok(())
        } else {
            Err(Exception::gp(0))
        }
    }

    /// Offset wrap mask for data accesses: multi-byte operands wrap at 64 KiB
    /// under a 16-bit address size (8086-compatible), at the address width
    /// otherwise.
    #[inline]
    pub(crate) fn data_wrap(&self) -> u64 {
        match self.asize {
            O16 => 0xFFFF,
            O32 => u32::MAX as u64,
            O64 => u64::MAX,
        }
    }

    /// Offset wrap mask for stack accesses (64-bit mode uses the full RSP;
    /// legacy modes select SP vs ESP via SS.B).
    #[inline]
    pub(crate) fn stack_wrap(&self) -> u64 {
        if self.m64 {
            u64::MAX
        } else if self.regs.seg[reg::SS as usize].db() {
            u32::MAX as u64
        } else {
            0xFFFF
        }
    }

    /// Read one byte at `seg:off`.
    #[inline]
    pub(crate) fn read8<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u64) -> Exec<u8> {
        let lin = self.lin_addr(seg, off, 1, false)?;
        self.lin_read8(bus, lin)
    }

    /// Read a little-endian word at `seg:off`.
    #[inline]
    pub(crate) fn read16<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u64) -> Exec<u16> {
        let lin = self.lin_addr(seg, off, 2, false)?;
        self.lin_read16(bus, lin)
    }

    /// Read a little-endian double-word at `seg:off`.
    #[inline]
    pub(crate) fn read32<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u64) -> Exec<u32> {
        let lin = self.lin_addr(seg, off, 4, false)?;
        self.lin_read32(bus, lin)
    }

    /// Read a little-endian quad-word at `seg:off`.
    #[inline]
    pub(crate) fn read64<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u64) -> Exec<u64> {
        let lin = self.lin_addr(seg, off, 8, false)?;
        self.lin_read64(bus, lin)
    }

    /// Write one byte at `seg:off`.
    #[inline]
    pub(crate) fn write8<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u64, v: u8) -> Exec<()> {
        let lin = self.lin_addr(seg, off, 1, true)?;
        self.lin_write8(bus, lin, v)
    }

    /// Write a little-endian word at `seg:off`.
    #[inline]
    pub(crate) fn write16<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u64, v: u16) -> Exec<()> {
        let lin = self.lin_addr(seg, off, 2, true)?;
        self.lin_write16(bus, lin, v)
    }

    /// Write a little-endian double-word at `seg:off`.
    #[inline]
    pub(crate) fn write32<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u64, v: u32) -> Exec<()> {
        let lin = self.lin_addr(seg, off, 4, true)?;
        self.lin_write32(bus, lin, v)
    }

    /// Write a little-endian quad-word at `seg:off`.
    #[inline]
    pub(crate) fn write64<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u64, v: u64) -> Exec<()> {
        let lin = self.lin_addr(seg, off, 8, true)?;
        self.lin_write64(bus, lin, v)
    }

    /// The effective segment for a data access whose default segment is
    /// `default`, honoring any override prefix.
    #[inline]
    pub(crate) fn seg_or(&self, default: u8) -> u8 {
        self.seg_override.unwrap_or(default)
    }

    // --- Instruction fetch ----------------------------------------------------

    /// Fetch the byte at `CS:RIP` and advance `RIP`, enforcing the 15-byte
    /// instruction length limit.
    #[inline]
    pub(crate) fn fetch8<B: Bus>(&mut self, bus: &mut B) -> Exec<u8> {
        if self.ilen >= 15 {
            return Err(Exception::ud());
        }
        self.ilen += 1;
        self.fetch_check(self.regs.rip)?;
        let lin = self.regs.seg[reg::CS as usize]
            .base
            .wrapping_add(self.regs.rip);
        let lin = if self.m64 { lin } else { lin & 0xFFFF_FFFF };
        let b = if self.paging() {
            // One-entry fetch-translation cache: a hit skips the NX-aware
            // TLB lookup; the cheap checks above still run per byte, so
            // every exception fires at exactly the same byte as a per-byte
            // translation. The entry persists across instructions — a hot
            // loop within one page pays zero translates — and is dropped on
            // TLB flushes, INVLPG and every CPL/mode/CS change (see
            // `fetch_invalidate`).
            if lin >> 12 == self.fetch_tag {
                bus.read(self.fetch_page | (lin & 0xFFF))
            } else {
                self.fetch_miss(bus, lin)?
            }
        } else {
            bus.read(lin)
        };
        self.regs.rip = self.regs.rip.wrapping_add(1);
        Ok(b)
    }

    /// Fetch-cache miss: translate the code byte's page (NX-aware, so a #PF
    /// error code carries the instruction-fetch bit) and cache it.
    /// Deliberately not inlined: it is rare, and keeping it out of
    /// `fetch8`'s many inline sites keeps the hot path small.
    fn fetch_miss<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u8> {
        let phys = self.fetch_translate(bus, lin)?;
        self.fetch_tag = lin >> 12;
        self.fetch_page = phys & !0xFFF;
        Ok(bus.read(phys))
    }

    /// Try to fetch `n` (2/4/8) instruction bytes as one wide bus read:
    /// possible when they all sit in the cached fetch page (which the Bus
    /// contract requires for a wide access anyway), fit the 15-byte limit,
    /// and the whole range passes the canonical/limit check. `None` falls
    /// back to byte-wise fetching — page crosses, cache misses, paging off
    /// and every fault case take that path, so exceptions are untouched.
    #[inline]
    fn fetch_wide<B: Bus>(&mut self, bus: &mut B, n: u8) -> Option<u64> {
        let rip = self.regs.rip;
        let lin = self.regs.seg[reg::CS as usize].base.wrapping_add(rip);
        let lin = if self.m64 { lin } else { lin & 0xFFFF_FFFF };
        if lin >> 12 != self.fetch_tag || lin & 0xFFF > 0x1000 - n as u64 || self.ilen > 15 - n {
            return None;
        }
        let range_ok = if self.m64 {
            // One page: canonicality is uniform across it.
            Self::canonical(lin)
        } else {
            rip.wrapping_add(n as u64 - 1) <= self.regs.seg[reg::CS as usize].limit as u64
        };
        if !range_ok {
            return None;
        }
        let phys = self.fetch_page | (lin & 0xFFF);
        debug_assert!(matches!(n, 2 | 4 | 8));
        let v = match n {
            2 => bus.read16(phys) as u64,
            4 => bus.read32(phys) as u64,
            _ => bus.read64(phys),
        };
        self.ilen += n;
        self.regs.rip = rip.wrapping_add(n as u64);
        Some(v)
    }

    /// Fetch a little-endian word at `CS:RIP`.
    #[inline]
    pub(crate) fn fetch16<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        if let Some(v) = self.fetch_wide(bus, 2) {
            return Ok(v as u16);
        }
        let lo = self.fetch8(bus)? as u16;
        let hi = self.fetch8(bus)? as u16;
        Ok(lo | hi << 8)
    }

    /// Fetch a little-endian double-word at `CS:RIP`.
    #[inline]
    pub(crate) fn fetch32<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        if let Some(v) = self.fetch_wide(bus, 4) {
            return Ok(v as u32);
        }
        let lo = self.fetch16(bus)? as u32;
        let hi = self.fetch16(bus)? as u32;
        Ok(lo | hi << 16)
    }

    /// Fetch a little-endian quad-word at `CS:RIP`.
    #[inline]
    pub(crate) fn fetch64<B: Bus>(&mut self, bus: &mut B) -> Exec<u64> {
        if let Some(v) = self.fetch_wide(bus, 8) {
            return Ok(v);
        }
        let lo = self.fetch32(bus)? as u64;
        let hi = self.fetch32(bus)? as u64;
        Ok(lo | hi << 32)
    }

    /// Fetch an immediate of the current operand size. Immediates never
    /// exceed 32 bits (except `MOV r64, imm64`, handled inline): a 64-bit
    /// operand size fetches an imm32 and sign-extends it.
    #[inline]
    pub(crate) fn fetch_imm<B: Bus>(&mut self, bus: &mut B) -> Exec<u64> {
        self.stat(|s| s.imm_fetches += 1);
        match self.osize {
            O16 => Ok(self.fetch16(bus)? as u64),
            O32 => Ok(self.fetch32(bus)? as u64),
            O64 => Ok(self.fetch32(bus)? as i32 as i64 as u64),
        }
    }

    // --- Stack ----------------------------------------------------------------
    // 64-bit mode uses the full RSP; legacy modes select SP vs ESP via SS.B.
    // The operand size of the push/pop selects the datum width independently.

    /// Current stack pointer.
    #[inline]
    pub(crate) fn stack_ptr(&self) -> u64 {
        let sp = self.regs.gpr[reg::RSP as usize];
        if self.m64 {
            sp
        } else if self.regs.seg[reg::SS as usize].db() {
            sp & 0xFFFF_FFFF
        } else {
            sp & 0xFFFF
        }
    }

    /// Adjust the stack pointer by `delta` (RSP, ESP or SP per mode/SS.B).
    #[inline]
    pub(crate) fn adjust_sp(&mut self, delta: i64) {
        let r = &mut self.regs.gpr[reg::RSP as usize];
        if self.m64 {
            *r = r.wrapping_add(delta as u64);
        } else if self.regs.seg[reg::SS as usize].db() {
            *r = (*r as u32).wrapping_add(delta as u32) as u64;
        } else {
            *r = (*r & !0xFFFF) | (*r as u16).wrapping_add(delta as u16) as u64;
        }
    }

    /// Push a word onto the stack.
    #[inline]
    pub(crate) fn push16<B: Bus>(&mut self, bus: &mut B, v: u16) -> Exec<()> {
        let wrap = self.stack_wrap();
        let sp = self.stack_ptr().wrapping_sub(2) & wrap;
        self.write16(bus, reg::SS, sp, v)?;
        self.adjust_sp(-2);
        Ok(())
    }

    /// Push a double-word onto the stack.
    #[inline]
    pub(crate) fn push32<B: Bus>(&mut self, bus: &mut B, v: u32) -> Exec<()> {
        let wrap = self.stack_wrap();
        let sp = self.stack_ptr().wrapping_sub(4) & wrap;
        self.write32(bus, reg::SS, sp, v)?;
        self.adjust_sp(-4);
        Ok(())
    }

    /// Push a quad-word onto the stack.
    #[inline]
    pub(crate) fn push64<B: Bus>(&mut self, bus: &mut B, v: u64) -> Exec<()> {
        let wrap = self.stack_wrap();
        let sp = self.stack_ptr().wrapping_sub(8) & wrap;
        self.write64(bus, reg::SS, sp, v)?;
        self.adjust_sp(-8);
        Ok(())
    }

    /// Push a stack-operand-sized value (default 64 in 64-bit mode).
    #[inline]
    pub(crate) fn push<B: Bus>(&mut self, bus: &mut B, v: u64) -> Exec<()> {
        match self.stack_osize() {
            O16 => self.push16(bus, v as u16),
            O32 => self.push32(bus, v as u32),
            O64 => self.push64(bus, v),
        }
    }

    /// Pop a word off the stack.
    #[inline]
    pub(crate) fn pop16<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        let v = self.read16(bus, reg::SS, self.stack_ptr())?;
        self.adjust_sp(2);
        Ok(v)
    }

    /// Pop a double-word off the stack.
    #[inline]
    pub(crate) fn pop32<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        let v = self.read32(bus, reg::SS, self.stack_ptr())?;
        self.adjust_sp(4);
        Ok(v)
    }

    /// Pop a quad-word off the stack.
    #[inline]
    pub(crate) fn pop64<B: Bus>(&mut self, bus: &mut B) -> Exec<u64> {
        let v = self.read64(bus, reg::SS, self.stack_ptr())?;
        self.adjust_sp(8);
        Ok(v)
    }

    /// Pop a stack-operand-sized value (zero-extended).
    #[inline]
    pub(crate) fn pop<B: Bus>(&mut self, bus: &mut B) -> Exec<u64> {
        match self.stack_osize() {
            O16 => Ok(self.pop16(bus)? as u64),
            O32 => Ok(self.pop32(bus)? as u64),
            O64 => self.pop64(bus),
        }
    }

    /// Pop a segment selector: a wide operand size still performs only a
    /// 16-bit read but releases the full slot.
    pub(crate) fn pop_sreg<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        let v = self.read16(bus, reg::SS, self.stack_ptr())?;
        self.adjust_sp(match self.stack_osize() {
            O16 => 2,
            O32 => 4,
            O64 => 8,
        });
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

    /// Deliver exception/interrupt `e` at the current `CS:RIP`.
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
    /// protected and long modes go through the IDT gates (see
    /// `protected.rs`).
    ///
    /// Delivery is atomic: if it faults partway through, every register side
    /// effect is rolled back (CR2 excepted, so a nested `#PF` keeps its
    /// address) and the nested fault is delivered from the architectural
    /// state the CPU had before delivery began.
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
        let ip = self.sys_read16(bus, base.wrapping_add(entry as u64))?;
        let cs = self.sys_read16(bus, base.wrapping_add(entry as u64 + 2))?;

        let flags = self.regs.rflags.image16();
        self.push16(bus, flags)?;
        let (old_cs, old_ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.rip);
        self.push16(bus, old_cs)?;
        self.push16(bus, old_ip as u16)?;
        self.regs.rflags.remove(RFlags::IF | RFlags::TF);
        self.regs.seg[reg::CS as usize].sel = cs;
        self.regs.seg[reg::CS as usize].base = (cs as u64) << 4;
        self.regs.rip = ip as u64;
        Ok(())
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}
