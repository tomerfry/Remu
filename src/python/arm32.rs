//! Python bindings for the ARM7TDMI / ARMv4T core (`remu.arm32`).
//!
//! ```python
//! from remu.arm32 import Cpu, Memory, RunExit
//!
//! mem = Memory()
//! mem.load(0x1000, (0xE3A0002A).to_bytes(4, "little"))  # MOV r0, #42
//! cpu = Cpu()
//! cpu.pc = 0x1000
//! cpu.step(mem)
//! assert cpu.r0 == 42
//! ```

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{PyRunExit, util};
use crate::arm32::{Bus, Cpu, Exception, HostTrap, LinearMemory, Mode, Psr, RunExit, psr};

// --- Bus dispatch -------------------------------------------------------------

/// A [`Bus`] that forwards accesses to a Python object's `read`/`write`
/// methods (the ARM bus has no I/O ports). The wide accessors byte-compose
/// little-endian through the trait defaults. See [`util::cb_read`].
struct CallbackBus<'py> {
    obj: Bound<'py, PyAny>,
    error: Option<PyErr>,
}

/// The bus argument accepted by every CPU entry point: a native
/// [`MemoryArm`](PyMemoryArm) or a duck-typed Python object.
enum BusArg<'py> {
    Flat(PyRefMut<'py, PyMemoryArm>),
    Callback(CallbackBus<'py>),
}

impl<'py> BusArg<'py> {
    fn from_any(bus: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(mem) = bus.cast::<PyMemoryArm>() {
            return Ok(BusArg::Flat(mem.try_borrow_mut()?));
        }
        if util::is_duck_bus(bus)? {
            return Ok(BusArg::Callback(CallbackBus {
                obj: bus.clone(),
                error: None,
            }));
        }
        Err(util::bus_type_error("remu.arm32.Memory"))
    }

    /// Re-raise an exception stashed by a Python bus callback, if any.
    fn check(self) -> PyResult<()> {
        match self {
            BusArg::Flat(_) => Ok(()),
            BusArg::Callback(mut c) => match c.error.take() {
                Some(e) => Err(e),
                None => Ok(()),
            },
        }
    }
}

impl Bus for CallbackBus<'_> {
    fn read(&mut self, addr: u32) -> u8 {
        util::cb_read(&self.obj, &mut self.error, addr as u64)
    }

    fn write(&mut self, addr: u32, value: u8) {
        util::cb_write(&self.obj, &mut self.error, addr as u64, value)
    }
}

impl Bus for BusArg<'_> {
    fn read(&mut self, addr: u32) -> u8 {
        match self {
            BusArg::Flat(m) => m.inner.read(addr),
            BusArg::Callback(c) => c.read(addr),
        }
    }

    fn write(&mut self, addr: u32, value: u8) {
        match self {
            BusArg::Flat(m) => m.inner.write(addr, value),
            BusArg::Callback(c) => c.write(addr, value),
        }
    }

    // The wide accessors keep the native memory on its fast paths; the
    // callback side falls through to the trait's byte-composed defaults.

    fn read16(&mut self, addr: u32) -> u16 {
        match self {
            BusArg::Flat(m) => m.inner.read16(addr),
            BusArg::Callback(c) => c.read16(addr),
        }
    }

    fn read32(&mut self, addr: u32) -> u32 {
        match self {
            BusArg::Flat(m) => m.inner.read32(addr),
            BusArg::Callback(c) => c.read32(addr),
        }
    }

    fn write16(&mut self, addr: u32, value: u16) {
        match self {
            BusArg::Flat(m) => m.inner.write16(addr, value),
            BusArg::Callback(c) => c.write16(addr, value),
        }
    }

    fn write32(&mut self, addr: u32, value: u32) {
        match self {
            BusArg::Flat(m) => m.inner.write32(addr, value),
            BusArg::Callback(c) => c.write32(addr, value),
        }
    }
}

// --- Memory -------------------------------------------------------------------

/// Python-facing wrapper over the ARM32 [`LinearMemory`]: a flat
/// power-of-two RAM (16 MiB by default; addresses wrap at the size) with
/// `mem[addr]` / `mem[start:stop]` indexing.
#[pyclass(name = "MemoryArm", module = "remu._remu")]
pub struct PyMemoryArm {
    pub(crate) inner: LinearMemory,
}

#[pymethods]
impl PyMemoryArm {
    /// Create a zero-initialized memory: 16 MiB by default, else `size`
    /// bytes (rounded up to a power of two, minimum 64 KiB).
    #[new]
    #[pyo3(signature = (size=None))]
    fn new(size: Option<usize>) -> Self {
        PyMemoryArm {
            inner: match size {
                Some(size) => LinearMemory::with_size(size),
                None => LinearMemory::new(),
            },
        }
    }

    /// Load `data` (bytes-like or iterable of ints) at `addr`, wrapping at
    /// the memory size.
    fn load(&mut self, addr: u32, data: Vec<u8>) {
        self.inner.load(addr, &data);
    }

    /// Read one byte (also lets a `Memory` be used where a duck-typed bus is
    /// expected).
    fn read(&mut self, addr: u32) -> u8 {
        self.inner.read(addr)
    }

    /// Write one byte.
    fn write(&mut self, addr: u32, value: u8) {
        self.inner.write(addr, value);
    }

    fn __len__(&self) -> usize {
        self.inner.ram.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        util::ram_getitem(py, &self.inner.ram[..], index)
    }

    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        util::ram_setitem(&mut self.inner.ram[..], index, value)
    }

    /// The full RAM as `bytes` (supports `bytes(mem)`).
    fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.ram[..])
    }

    fn __repr__(&self) -> String {
        let len = self.inner.ram.len();
        if len >= 1 << 20 {
            format!("<remu.arm32.Memory {}MiB>", len >> 20)
        } else {
            format!("<remu.arm32.Memory {}KiB>", len >> 10)
        }
    }
}

// --- Cpu ----------------------------------------------------------------------

/// Getter/setter pairs for the sixteen GPRs and their conventional aliases
/// (all views into the current mode's `gpr` window).
macro_rules! gpr_props {
    ($($(#[doc = $doc:expr])* $idx:literal => $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpuArm {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> u32 {
                    self.inner.regs.gpr[$idx]
                }
                #[setter]
                fn $set(&mut self, v: u32) {
                    self.inner.regs.gpr[$idx] = v;
                }
            )+
        }
    };
}

/// Getter/setter pairs for individual CPSR bits.
macro_rules! psr_bit_props {
    ($($(#[doc = $doc:expr])* $bit:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpuArm {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> bool {
                    self.inner.regs.cpsr.bits() & psr::$bit != 0
                }
                #[setter]
                fn $set(&mut self, v: bool) {
                    self.inner.regs.cpsr.set(psr::$bit, v);
                }
            )+
        }
    };
}

/// The Python-facing name of an exception kind (as returned in host traps
/// and accepted by `raise_exception`).
fn exception_name(e: Exception) -> &'static str {
    match e {
        Exception::Undefined => "undefined",
        Exception::Swi => "swi",
        Exception::PrefetchAbort => "prefetch_abort",
        Exception::DataAbort => "data_abort",
        Exception::Irq => "irq",
        Exception::Fiq => "fiq",
    }
}

fn parse_exception(kind: &str) -> PyResult<Exception> {
    Ok(match kind {
        "undefined" => Exception::Undefined,
        "swi" => Exception::Swi,
        "prefetch_abort" => Exception::PrefetchAbort,
        "data_abort" => Exception::DataAbort,
        "irq" => Exception::Irq,
        "fiq" => Exception::Fiq,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown exception kind {kind:?} (expected one of: undefined, swi, \
                 prefetch_abort, data_abort, irq, fiq)"
            )));
        }
    })
}

/// The Python-facing name of a processor mode.
fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Usr => "usr",
        Mode::Fiq => "fiq",
        Mode::Irq => "irq",
        Mode::Svc => "svc",
        Mode::Abt => "abt",
        Mode::Und => "und",
        Mode::Sys => "sys",
    }
}

fn parse_mode(name: &str) -> PyResult<Mode> {
    Ok(match name {
        "usr" => Mode::Usr,
        "fiq" => Mode::Fiq,
        "irq" => Mode::Irq,
        "svc" => Mode::Svc,
        "abt" => Mode::Abt,
        "und" => Mode::Und,
        "sys" => Mode::Sys,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown mode {name:?} (expected one of: usr, fiq, irq, svc, abt, und, sys)"
            )));
        }
    })
}

/// A host trap as a Python value: `"syscall"` or `("exception", kind)`.
fn trap_to_py(py: Python<'_>, t: HostTrap) -> PyResult<Py<PyAny>> {
    match t {
        HostTrap::Syscall => "syscall".into_py_any(py),
        HostTrap::Exception(e) => ("exception", exception_name(e)).into_py_any(py),
    }
}

/// Map the core's [`RunExit`] (`#[non_exhaustive]`) onto the shared Python
/// enum.
fn map_exit(exit: RunExit) -> PyRunExit {
    match exit {
        RunExit::Completed => PyRunExit::Completed,
        RunExit::HostTrap => PyRunExit::HostTrap,
        // Halted, plus any exit a future backend may add: the CPU stopped.
        _ => PyRunExit::Halted,
    }
}

/// Python-facing wrapper over the ARM32 [`Cpu`]. Registers and CPSR bits are
/// flat read/write properties; every method that touches memory takes the
/// bus as an argument (the CPU does not own its bus).
#[pyclass(name = "CpuArm", module = "remu._remu")]
pub struct PyCpuArm {
    pub(crate) inner: Cpu,
}

#[pymethods]
impl PyCpuArm {
    /// Create a CPU in its power-on state: Supervisor mode, ARM state,
    /// IRQ+FIQ disabled, PC at the reset vector (0).
    #[new]
    fn new() -> Self {
        PyCpuArm { inner: Cpu::new() }
    }

    /// RESET: registers back to the power-on state, halt latch and pending
    /// host trap cleared. (The IRQ/FIQ lines are host state and stay as
    /// last set.)
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Execute one instruction (or service a pending interrupt). Returns
    /// the cycles consumed. A halted CPU burns one idle cycle per call
    /// until an interrupt arrives. If a bus callback raises, the exception
    /// propagates after the instruction finishes against a bus that reads
    /// as 0 / drops writes — CPU state reflects that partial execution.
    fn step(&mut self, bus: &Bound<'_, PyAny>) -> PyResult<u32> {
        let mut b = BusArg::from_any(bus)?;
        let cycles = self.inner.step(&mut b);
        b.check()?;
        Ok(cycles)
    }

    /// Execute up to `instructions` step-units (instruction executions and
    /// interrupt deliveries each count one) and return `(executed, exit)`.
    /// The run stops early at a host trap (`RunExit.HostTrap` — service
    /// `take_host_trap()` and call `run` again), when `halted` is set with
    /// no wake event pending (`RunExit.Halted` — unlike `step`, `run`
    /// returns instead of idling), or when a Python bus callback raises
    /// (see `step` for its bus semantics). The hot loop runs in Rust.
    ///
    /// Every 64 Ki instructions the loop polls for signals (Ctrl-C raises
    /// `KeyboardInterrupt`) and briefly releases the GIL.
    fn run(&mut self, bus: &Bound<'_, PyAny>, instructions: u64) -> PyResult<(u64, PyRunExit)> {
        /// Rare enough to cost nothing, frequent enough that Ctrl-C and
        /// waiting threads feel it instantly.
        const CHUNK: u64 = 0x10000;

        let py = bus.py();
        let mut executed: u64 = 0;
        let mut exit = PyRunExit::Completed;
        while executed < instructions {
            let target = (instructions - executed).min(CHUNK);
            // Re-borrowed each chunk so the borrow is dropped before detaching.
            let mut b = BusArg::from_any(bus)?;
            let n = match &mut b {
                BusArg::Flat(m) => {
                    let r = self.inner.run(&mut m.inner, target);
                    exit = map_exit(r.exit);
                    r.executed
                }
                BusArg::Callback(c) => {
                    // A manual step loop mirroring the core's `run`
                    // semantics, so a stashed callback error also stops it.
                    let mut n: u64 = 0;
                    while n < target
                        && !self.inner.halted
                        && self.inner.host_trap.is_none()
                        && c.error.is_none()
                    {
                        self.inner.step(c);
                        n += 1;
                    }
                    exit = if self.inner.host_trap.is_some() {
                        PyRunExit::HostTrap
                    } else if self.inner.halted {
                        PyRunExit::Halted
                    } else {
                        PyRunExit::Completed
                    };
                    n
                }
            };
            executed += n;
            b.check()?;
            py.check_signals()?;
            py.detach(|| {});
            if exit != PyRunExit::Completed {
                break;
            }
        }
        Ok((executed, exit))
    }

    /// Drive the level-sensitive IRQ line. Serviced at the next
    /// instruction boundary while CPSR.I is clear (`irq_disable` False);
    /// deassert once the device is acknowledged. FIQ outranks IRQ.
    fn set_irq(&mut self, level: bool) {
        self.inner.set_irq(level);
    }

    /// Drive the level-sensitive FIQ line. Serviced at the next
    /// instruction boundary while CPSR.F is clear (`fiq_disable` False);
    /// FIQ outranks IRQ.
    fn set_fiq(&mut self, level: bool) {
        self.inner.set_fiq(level);
    }

    /// Deliver a CPU exception at the current architectural state: bank
    /// into the exception mode, save the CPSR to the new SPSR, load the
    /// banked LR with the architected return offset, disable IRQ (and FIQ
    /// for a FIQ), leave Thumb state, and vector. `kind` is one of
    /// "undefined", "swi", "prefetch_abort", "data_abort", "irq", "fiq"
    /// (ValueError otherwise). Embedders normally only need this to inject
    /// aborts — use `set_irq`/`set_fiq` for interrupts. (Named
    /// `raise_exception` because `raise` is a Python keyword.)
    fn raise_exception(&mut self, kind: &str) -> PyResult<()> {
        self.inner.raise(parse_exception(kind)?);
        Ok(())
    }

    /// Consume and return the pending host trap: `None`, `"syscall"`, or
    /// `("exception", kind)` with `kind` named as in `raise_exception`.
    /// Service it (e.g. read the syscall number from `r7`, write the
    /// result to `r0`), then call `run` again to resume.
    fn take_host_trap(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.inner.host_trap.take() {
            Some(t) => trap_to_py(py, t),
            None => Ok(py.None()),
        }
    }

    /// The pending host trap, without consuming it (same values as
    /// `take_host_trap`). `run` stops with `RunExit.HostTrap` while this
    /// is set.
    #[getter]
    fn get_host_trap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.inner.host_trap {
            Some(t) => trap_to_py(py, t),
            None => Ok(py.None()),
        }
    }

    /// OS-emulation hook: when True, SWI records a `"syscall"` host trap
    /// (with the PC already past the instruction) instead of vectoring
    /// through 0x08. Default False — SWI behaves exactly like hardware.
    #[getter]
    fn get_trap_swi(&self) -> bool {
        self.inner.trap_swi
    }
    #[setter]
    fn set_trap_swi(&mut self, v: bool) {
        self.inner.trap_swi = v;
    }

    /// OS-emulation hook: when True, a CPU exception is handed back as an
    /// `("exception", kind)` host trap with the PC rewound to the faulting
    /// instruction — the host may fix the fault and resume — instead of
    /// being delivered through its vector. Default False.
    #[getter]
    fn get_trap_faults(&self) -> bool {
        self.inner.trap_faults
    }
    #[setter]
    fn set_trap_faults(&mut self, v: bool) {
        self.inner.trap_faults = v;
    }

    /// Raw CPSR image. Writing switches register banks when the mode field
    /// changes (like MSR), then deposits the raw bits — reserved encodings
    /// included, exactly as the core stores them.
    #[getter]
    fn get_cpsr(&self) -> u32 {
        self.inner.regs.cpsr.bits()
    }
    #[setter]
    fn set_cpsr(&mut self, v: u32) {
        let next = Psr::from_bits(v);
        self.inner.regs.set_mode(next.mode());
        self.inner.regs.cpsr = next;
    }

    /// Raw SPSR image of the current mode. Only exception modes have an
    /// SPSR: in User/System a read returns the CPSR (the core's
    /// suite-pinned MRS behavior) and a write is discarded.
    #[getter]
    fn get_spsr(&self) -> u32 {
        self.inner.regs.spsr_bits()
    }
    #[setter]
    fn set_spsr(&mut self, v: u32) {
        self.inner.regs.set_spsr(v, u32::MAX);
    }

    /// The processor mode, as one of the strings "usr", "fiq", "irq",
    /// "svc", "abt", "und", "sys". Writing switches modes with proper
    /// register banking (like `set_mode` in Rust); ValueError on an
    /// unknown name.
    #[getter]
    fn get_mode(&self) -> &'static str {
        mode_name(self.inner.regs.cpsr.mode())
    }
    #[setter]
    fn set_mode(&mut self, name: &str) -> PyResult<()> {
        self.inner.regs.set_mode(parse_mode(name)?);
        Ok(())
    }

    /// True while the CPU is idle. ARMv4T has no wait-for-interrupt
    /// instruction, so halting is embedder-controlled: set this to park
    /// the CPU (`run` returns `RunExit.Halted`; `step` burns one idle
    /// cycle per call). A delivered IRQ/FIQ clears it.
    #[getter]
    fn get_halted(&self) -> bool {
        self.inner.halted
    }
    #[setter]
    fn set_halted(&mut self, v: bool) {
        self.inner.halted = v;
    }

    /// Total cycles elapsed since construction/reset (nominal ARM7TDMI
    /// figures with S/N/I cycles collapsed — rough relative weights, not
    /// bus-accurate timings).
    #[getter]
    fn get_cycles(&self) -> u64 {
        self.inner.cycles
    }

    fn __repr__(&self) -> String {
        let r = &self.inner.regs;
        format!(
            "<remu.arm32.Cpu pc=0x{:08X} r0=0x{:08X} {} {} cycles={}{}>",
            r.gpr[15],
            r.gpr[0],
            mode_name(r.cpsr.mode()),
            if r.cpsr.t() { "thumb" } else { "arm" },
            self.inner.cycles,
            if self.inner.halted { " HALTED" } else { "" },
        )
    }
}

gpr_props! {
    /// General-purpose register r0.
    0 => get_r0 / set_r0,
    /// General-purpose register r1.
    1 => get_r1 / set_r1,
    /// General-purpose register r2.
    2 => get_r2 / set_r2,
    /// General-purpose register r3.
    3 => get_r3 / set_r3,
    /// General-purpose register r4.
    4 => get_r4 / set_r4,
    /// General-purpose register r5.
    5 => get_r5 / set_r5,
    /// General-purpose register r6.
    6 => get_r6 / set_r6,
    /// General-purpose register r7 (the EABI syscall-number register).
    7 => get_r7 / set_r7,
    /// General-purpose register r8.
    8 => get_r8 / set_r8,
    /// General-purpose register r9.
    9 => get_r9 / set_r9,
    /// General-purpose register r10.
    10 => get_r10 / set_r10,
    /// General-purpose register r11.
    11 => get_r11 / set_r11,
    /// General-purpose register r12.
    12 => get_r12 / set_r12,
    /// r13, the stack pointer (the current mode's banked copy).
    13 => get_r13 / set_r13,
    /// r14, the link register (the current mode's banked copy).
    14 => get_r14 / set_r14,
    /// r15, the program counter — see `pc`.
    15 => get_r15 / set_r15,
    /// Stack pointer — alias of r13.
    13 => get_sp / set_sp,
    /// Link register — alias of r14.
    14 => get_lr / set_lr,
    /// Program counter — alias of r15: the address of the NEXT instruction
    /// to execute (the +8/+4 pipeline offset ARM code observes when
    /// reading R15 is applied by the executor, not stored here).
    15 => get_pc / set_pc,
}

psr_bit_props! {
    /// Negative flag (CPSR.N).
    N: get_negative / set_negative,
    /// Zero flag (CPSR.Z).
    Z: get_zero / set_zero,
    /// Carry flag (CPSR.C; for subtraction: NOT borrow).
    C: get_carry / set_carry,
    /// Signed-overflow flag (CPSR.V).
    V: get_overflow / set_overflow,
    /// IRQ-disable bit (CPSR.I).
    I: get_irq_disable / set_irq_disable,
    /// FIQ-disable bit (CPSR.F).
    F: get_fiq_disable / set_fiq_disable,
    /// Thumb-state bit (CPSR.T): False = ARM, True = Thumb. Takes effect
    /// at the next fetch.
    T: get_thumb / set_thumb,
}
