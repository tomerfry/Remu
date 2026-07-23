//! Python bindings for the x86-64 / AMD64 core (`remu.x86_64`).
//!
//! ```python
//! from remu.x86_64 import Cpu, Memory, RunExit
//!
//! mem = Memory()
//! mem.load(0x10000, b"\x48\xC7\xC0\x2A\x00\x00\x00\xF4")  # MOV RAX, 42; HLT
//! cpu = Cpu()
//! cpu.setup_long_flat(mem, 0x10000, 0x20000)
//! executed, exit = cpu.run(mem, 10)
//! assert cpu.rax == 42 and exit == RunExit.Halted
//! ```

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{PyRunExit, util};
use crate::x86_64::{Bus, Cpu, DescTable, HostTrap, LinearMemory, RFlags, RunExit, SegReg, reg};

// --- Bus dispatch -------------------------------------------------------------

/// A [`Bus`] that forwards accesses to a Python object's `read`/`write` (and
/// optional `io_read`/`io_write`) methods. See [`util::cb_read`]. Wide
/// accesses decompose into one Python call per byte.
struct CallbackBus<'py> {
    obj: Bound<'py, PyAny>,
    has_io_read: bool,
    has_io_write: bool,
    error: Option<PyErr>,
}

/// The bus argument accepted by every CPU entry point: a native
/// [`MemoryX64`](PyMemoryX64) or a duck-typed Python object.
enum BusArg<'py> {
    Flat(PyRefMut<'py, PyMemoryX64>),
    Callback(CallbackBus<'py>),
}

impl<'py> BusArg<'py> {
    fn from_any(bus: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(mem) = bus.cast::<PyMemoryX64>() {
            return Ok(BusArg::Flat(mem.try_borrow_mut()?));
        }
        if util::is_duck_bus(bus)? {
            return Ok(BusArg::Callback(CallbackBus {
                has_io_read: util::has_attr(bus, "io_read"),
                has_io_write: util::has_attr(bus, "io_write"),
                obj: bus.clone(),
                error: None,
            }));
        }
        Err(util::bus_type_error("remu.x86_64.Memory"))
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
    fn read(&mut self, addr: u64) -> u8 {
        util::cb_read(&self.obj, &mut self.error, addr)
    }

    fn write(&mut self, addr: u64, value: u8) {
        util::cb_write(&self.obj, &mut self.error, addr, value)
    }

    // Wide accesses byte-compose little-endian through the Python `read`/
    // `write` callbacks (the duck protocol is byte-only).
    fn read16(&mut self, addr: u64) -> u16 {
        self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
    }

    fn read32(&mut self, addr: u64) -> u32 {
        self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
    }

    fn read64(&mut self, addr: u64) -> u64 {
        self.read32(addr) as u64 | (self.read32(addr.wrapping_add(4)) as u64) << 32
    }

    fn write16(&mut self, addr: u64, value: u16) {
        self.write(addr, value as u8);
        self.write(addr.wrapping_add(1), (value >> 8) as u8);
    }

    fn write32(&mut self, addr: u64, value: u32) {
        self.write16(addr, value as u16);
        self.write16(addr.wrapping_add(2), (value >> 16) as u16);
    }

    fn write64(&mut self, addr: u64, value: u64) {
        self.write32(addr, value as u32);
        self.write32(addr.wrapping_add(4), (value >> 32) as u32);
    }

    fn io_read(&mut self, port: u16) -> u8 {
        util::cb_io_read(&self.obj, self.has_io_read, &mut self.error, port)
    }

    fn io_write(&mut self, port: u16, value: u8) {
        util::cb_io_write(&self.obj, self.has_io_write, &mut self.error, port, value)
    }
}

impl Bus for BusArg<'_> {
    fn read(&mut self, addr: u64) -> u8 {
        match self {
            BusArg::Flat(m) => m.inner.read(addr),
            BusArg::Callback(c) => c.read(addr),
        }
    }

    fn write(&mut self, addr: u64, value: u8) {
        match self {
            BusArg::Flat(m) => m.inner.write(addr, value),
            BusArg::Callback(c) => c.write(addr, value),
        }
    }

    // Forward the wide accessors so a native memory keeps its fast paths
    // (the trait defaults would byte-compose through `BusArg::read`).
    fn read16(&mut self, addr: u64) -> u16 {
        match self {
            BusArg::Flat(m) => m.inner.read16(addr),
            BusArg::Callback(c) => c.read16(addr),
        }
    }

    fn read32(&mut self, addr: u64) -> u32 {
        match self {
            BusArg::Flat(m) => m.inner.read32(addr),
            BusArg::Callback(c) => c.read32(addr),
        }
    }

    fn read64(&mut self, addr: u64) -> u64 {
        match self {
            BusArg::Flat(m) => m.inner.read64(addr),
            BusArg::Callback(c) => c.read64(addr),
        }
    }

    fn write16(&mut self, addr: u64, value: u16) {
        match self {
            BusArg::Flat(m) => m.inner.write16(addr, value),
            BusArg::Callback(c) => c.write16(addr, value),
        }
    }

    fn write32(&mut self, addr: u64, value: u32) {
        match self {
            BusArg::Flat(m) => m.inner.write32(addr, value),
            BusArg::Callback(c) => c.write32(addr, value),
        }
    }

    fn write64(&mut self, addr: u64, value: u64) {
        match self {
            BusArg::Flat(m) => m.inner.write64(addr, value),
            BusArg::Callback(c) => c.write64(addr, value),
        }
    }

    fn io_read(&mut self, port: u16) -> u8 {
        match self {
            BusArg::Flat(m) => m.inner.io_read(port),
            BusArg::Callback(c) => c.io_read(port),
        }
    }

    fn io_write(&mut self, port: u16, value: u8) {
        match self {
            BusArg::Flat(m) => m.inner.io_write(port, value),
            BusArg::Callback(c) => c.io_write(port, value),
        }
    }
}

// --- Memory -------------------------------------------------------------------

/// Python-facing wrapper over the x86-64 [`LinearMemory`]: a flat
/// power-of-two RAM (default 16 MiB) with `mem[addr]` / `mem[start:stop]`
/// indexing. Addresses wrap at the memory size.
#[pyclass(name = "MemoryX64", module = "remu._remu")]
pub struct PyMemoryX64 {
    pub(crate) inner: LinearMemory,
}

#[pymethods]
impl PyMemoryX64 {
    /// Create a zero-initialized memory of `size` bytes (rounded up to a
    /// power of two, minimum 64 KiB; default 16 MiB).
    #[new]
    #[pyo3(signature = (size=None))]
    fn new(size: Option<usize>) -> Self {
        PyMemoryX64 {
            inner: match size {
                Some(size) => LinearMemory::with_size(size),
                None => LinearMemory::new(),
            },
        }
    }

    /// Load `data` (bytes-like or iterable of ints) at physical `addr`,
    /// wrapping at the memory size.
    fn load(&mut self, addr: u64, data: Vec<u8>) {
        self.inner.load(addr, &data);
    }

    /// Read one byte (also lets a `Memory` be used where a duck-typed bus is
    /// expected).
    fn read(&mut self, addr: u64) -> u8 {
        self.inner.read(addr)
    }

    /// Write one byte.
    fn write(&mut self, addr: u64, value: u8) {
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
            format!("<remu.x86_64.Memory {}MiB>", len >> 20)
        } else {
            format!("<remu.x86_64.Memory {}KiB>", len >> 10)
        }
    }
}

// --- Cpu ----------------------------------------------------------------------

/// Getter/setter pairs for 64-bit general registers, by [`reg`] index.
macro_rules! gpr_props {
    ($($(#[doc = $doc:expr])* $idx:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpuX64 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> u64 {
                    self.inner.regs.gpr[reg::$idx as usize]
                }
                #[setter]
                fn $set(&mut self, v: u64) {
                    self.inner.regs.gpr[reg::$idx as usize] = v;
                }
            )+
        }
    };
}

/// Getter/setter pairs for plain `u64` fields of the register file
/// (RIP and the control registers).
macro_rules! reg64_props {
    ($($(#[doc = $doc:expr])* $name:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpuX64 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> u64 {
                    self.inner.regs.$name
                }
                #[setter]
                fn $set(&mut self, v: u64) {
                    self.inner.regs.$name = v;
                }
            )+
        }
    };
}

/// Getter/setter pairs for model-specific registers ([`crate::x86_64::Msrs`]
/// fields).
macro_rules! msr_props {
    ($($(#[doc = $doc:expr])* $name:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpuX64 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> u64 {
                    self.inner.regs.msr.$name
                }
                #[setter]
                fn $set(&mut self, v: u64) {
                    self.inner.regs.msr.$name = v;
                }
            )+
        }
    };
}

/// Getter/setter pairs for individual RFLAGS bits.
macro_rules! flag_props {
    ($($(#[doc = $doc:expr])* $bit:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpuX64 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> bool {
                    self.inner.regs.rflags.contains(RFlags::$bit)
                }
                #[setter]
                fn $set(&mut self, v: bool) {
                    self.inner.regs.rflags.set(RFlags::$bit, v);
                }
            )+
        }
    };
}

/// Getter/setter pairs for segment registers, by [`reg`] index. Getters
/// return the full descriptor cache as a `(sel, base, limit, attrs)` tuple;
/// setters accept a bare selector (real-mode semantics) or such a tuple.
macro_rules! seg_props {
    ($($(#[doc = $doc:expr])* $idx:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpuX64 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> (u16, u64, u32, u16) {
                    let s = self.inner.regs.seg[reg::$idx as usize];
                    (s.sel, s.base, s.limit, s.attrs)
                }
                #[setter]
                fn $set(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
                    self.inner.regs.seg[reg::$idx as usize] = seg_from_any(v)?;
                    Ok(())
                }
            )+
        }
    };
}

/// A segment-register value from Python: a bare selector int (real-mode
/// semantics, base = `sel * 16`) or a `(sel, base, limit, attrs)` tuple
/// naming the descriptor cache directly.
fn seg_from_any(v: &Bound<'_, PyAny>) -> PyResult<SegReg> {
    if v.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(SegReg::real(v.extract()?));
    }
    if let Ok((sel, base, limit, attrs)) = v.extract::<(u16, u64, u32, u16)>() {
        return Ok(SegReg {
            sel,
            base,
            limit,
            attrs,
        });
    }
    Err(PyTypeError::new_err(
        "segment must be a selector int or a (sel, base, limit, attrs) tuple",
    ))
}

/// Map the core's [`RunExit`] onto the shared Python enum.
fn map_exit(exit: RunExit) -> PyRunExit {
    match exit {
        RunExit::Completed => PyRunExit::Completed,
        RunExit::HostTrap => PyRunExit::HostTrap,
        RunExit::Halted => PyRunExit::Halted,
        RunExit::Shutdown => PyRunExit::Shutdown,
        // `RunExit` is #[non_exhaustive]; treat unknown future reasons as a
        // plain stop.
        #[allow(unreachable_patterns)]
        _ => PyRunExit::Completed,
    }
}

/// An optional [`HostTrap`] in its Python shape: `None`, `"syscall"`, or
/// `("exception", vector, error_code_or_None)`.
fn trap_to_py(py: Python<'_>, trap: Option<HostTrap>) -> PyResult<Py<PyAny>> {
    match trap {
        None => Ok(py.None()),
        Some(HostTrap::Syscall) => "syscall".into_py_any(py),
        Some(HostTrap::Exception(e)) => ("exception", e.vector, e.error).into_py_any(py),
    }
}

/// Python-facing wrapper over the x86-64 [`Cpu`]. Registers, flags, control
/// registers and MSRs are flat read/write properties; every method that
/// touches memory takes the bus as an argument (the CPU does not own its
/// bus).
#[pyclass(name = "CpuX64", module = "remu._remu")]
pub struct PyCpuX64 {
    inner: Cpu,
}

#[pymethods]
impl PyCpuX64 {
    /// Create a CPU in its power-on state: real mode, execution begins at
    /// `F000:FFF0`.
    #[new]
    fn new() -> Self {
        PyCpuX64 { inner: Cpu::new() }
    }

    /// RESET: back to the power-on state (registers, halt latch, lines,
    /// pending host trap, translation caches).
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Set `CS:RIP` with real-mode semantics (CS base = `sel * 16`).
    fn set_cs_ip(&mut self, sel: u16, rip: u64) {
        self.inner.set_cs_ip(sel, rip);
    }

    /// Drop the CPU directly into 64-bit long mode with flat ring-0 segments
    /// and identity paging: RIP at `rip`, RSP at `rsp`, and identity page
    /// tables for the low 512 GiB written through `bus` at physical
    /// `0x1000`/`0x2000` (CR3 = `0x1000`). Long mode requires paging, so the
    /// minimal table set is written to memory — keep those pages free.
    fn setup_long_flat(&mut self, bus: &Bound<'_, PyAny>, rip: u64, rsp: u64) -> PyResult<()> {
        let mut b = BusArg::from_any(bus)?;
        self.inner.setup_long_flat(&mut b, rip, rsp);
        b.check()
    }

    /// Execute one instruction (or service a pending interrupt). Returns the
    /// cycles consumed. A halted CPU burns one idle cycle per call until an
    /// interrupt arrives. If a bus callback raises, the exception propagates
    /// after the instruction finishes against a bus that reads as 0 / drops
    /// writes — CPU state reflects that partial execution.
    fn step(&mut self, bus: &Bound<'_, PyAny>) -> PyResult<u32> {
        let mut b = BusArg::from_any(bus)?;
        let cycles = self.inner.step(&mut b);
        b.check()?;
        Ok(cycles)
    }

    /// Execute up to `instructions` instructions as one batch, returning
    /// `(executed, RunExit)`. Stops early at a host trap
    /// (`RunExit.HostTrap` — service `take_host_trap()` and run again), on
    /// HLT (`RunExit.Halted` — assert an interrupt and step to resume), on
    /// triple fault (`RunExit.Shutdown` — only `reset()` recovers), or when
    /// a Python bus callback raises (see `step` for its bus semantics). The
    /// hot loop runs in Rust.
    ///
    /// Every 64 Ki instructions the loop polls for signals (Ctrl-C raises
    /// `KeyboardInterrupt`) and briefly releases the GIL.
    fn run(&mut self, bus: &Bound<'_, PyAny>, instructions: u64) -> PyResult<(u64, PyRunExit)> {
        /// Rare enough to cost nothing, frequent enough that Ctrl-C and
        /// waiting threads feel it instantly.
        const CHUNK: u64 = 0x10000;

        let py = bus.py();
        // A trap the embedder has not yet taken stops the run before
        // anything executes, as in the core's `run`.
        if self.inner.host_trap.is_some() {
            return Ok((0, PyRunExit::HostTrap));
        }
        let mut executed: u64 = 0;
        let mut exit = PyRunExit::Completed;
        while executed < instructions {
            let target = (instructions - executed).min(CHUNK);
            // Re-borrowed each chunk so the borrow is dropped before detaching.
            let mut b = BusArg::from_any(bus)?;
            let (n, why) = match &mut b {
                BusArg::Flat(m) => {
                    let r = self.inner.run(&mut m.inner, target);
                    (r.executed, map_exit(r.exit))
                }
                BusArg::Callback(c) => {
                    let mut n: u64 = 0;
                    while n < target
                        && !self.inner.halted
                        && !self.inner.shutdown
                        && self.inner.host_trap.is_none()
                        && c.error.is_none()
                    {
                        self.inner.step(c);
                        n += 1;
                    }
                    let why = if self.inner.host_trap.is_some() {
                        PyRunExit::HostTrap
                    } else if self.inner.halted {
                        PyRunExit::Halted
                    } else if self.inner.shutdown {
                        PyRunExit::Shutdown
                    } else {
                        PyRunExit::Completed
                    };
                    (n, why)
                }
            };
            executed += n;
            exit = why;
            b.check()?;
            py.check_signals()?;
            py.detach(|| {});
            if exit != PyRunExit::Completed {
                break;
            }
        }
        Ok((executed, exit))
    }

    /// Latch a non-maskable interrupt (vector 2).
    fn trigger_nmi(&mut self) {
        self.inner.trigger_nmi();
    }

    /// Assert the maskable INTR line with `vector` (serviced while IF is set).
    fn assert_intr(&mut self, vector: u8) {
        self.inner.assert_intr(vector);
    }

    /// Deassert the INTR line.
    fn clear_intr(&mut self) {
        self.inner.clear_intr();
    }

    /// Flush the JIT translation cache — call after writing guest code
    /// memory behind the CPU's back (stores performed by the CPU itself are
    /// already tracked). A no-op when the extension was built without the
    /// `jit` feature.
    fn invalidate_jit(&mut self) {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        self.inner.invalidate_jit();
    }

    /// RFLAGS as materialized by `PUSHFQ` (bit 1 reads as 1; bits 3/5/15 and
    /// VM/RF as 0).
    #[getter]
    fn get_rflags(&self) -> u64 {
        self.inner.regs.rflags.image()
    }
    #[setter]
    fn set_rflags(&mut self, v: u64) {
        self.inner.regs.rflags = RFlags::from_bits_truncate(v as u32);
    }

    /// True while EFER.LMA is set (long mode active — 64-bit or
    /// compatibility submode).
    #[getter]
    fn get_long_mode(&self) -> bool {
        self.inner.long_mode()
    }

    /// True when the current code segment executes 64-bit code.
    #[getter]
    fn get_mode64(&self) -> bool {
        self.inner.mode64()
    }

    /// True once CR0.PE is set and the CPU is not in Virtual-8086 mode.
    #[getter]
    fn get_protected_mode(&self) -> bool {
        self.inner.protected_mode()
    }

    /// Current privilege level (0–3; real mode reports 0, V86 reports 3).
    #[getter]
    fn get_cpl(&self) -> u8 {
        self.inner.cpl()
    }

    /// Total cycles elapsed since construction (nominal per-instruction
    /// weights, not validated hardware figures).
    #[getter]
    fn get_cycles(&self) -> u64 {
        self.inner.cycles
    }

    /// True while a HLT has the processor stopped (an interrupt resumes it).
    #[getter]
    fn get_halted(&self) -> bool {
        self.inner.halted
    }

    /// True after a triple fault shut the machine down; only `reset()`
    /// recovers.
    #[getter]
    fn get_shutdown(&self) -> bool {
        self.inner.shutdown
    }

    /// OS-emulation host hook: if set to a vector (e.g. `0x80`), `INT n` for
    /// that vector records a `"syscall"` host trap instead of vectoring
    /// through the IDT. `None` (the default) leaves INT hardware-faithful.
    #[getter]
    fn get_syscall_int(&self) -> Option<u8> {
        self.inner.syscall_int
    }
    #[setter]
    fn set_syscall_int(&mut self, v: Option<u8>) {
        self.inner.syscall_int = v;
    }

    /// OS-emulation host hook: when True the SYSCALL instruction records a
    /// `"syscall"` host trap instead of vectoring through the STAR/LSTAR
    /// MSRs (RCX/R11 still receive the return RIP and RFLAGS). Default
    /// False.
    #[getter]
    fn get_trap_syscall(&self) -> bool {
        self.inner.trap_syscall
    }
    #[setter]
    fn set_trap_syscall(&mut self, v: bool) {
        self.inner.trap_syscall = v;
    }

    /// OS-emulation host hook: when True a CPU exception is handed back as
    /// an `("exception", vector, error)` host trap — register state rewound
    /// to the faulting instruction — instead of being delivered through the
    /// IDT. Default False.
    #[getter]
    fn get_trap_faults(&self) -> bool {
        self.inner.trap_faults
    }
    #[setter]
    fn set_trap_faults(&mut self, v: bool) {
        self.inner.trap_faults = v;
    }

    /// The pending host trap without consuming it: `None`, `"syscall"`, or
    /// `("exception", vector, error_code_or_None)`.
    #[getter]
    fn get_host_trap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        trap_to_py(py, self.inner.host_trap)
    }

    /// Take (return and clear) the pending host trap — same shapes as
    /// `host_trap`. Service it, then call `run` again to resume the guest.
    fn take_host_trap(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        trap_to_py(py, self.inner.host_trap.take())
    }

    /// GDTR as a `(base, limit)` tuple.
    #[getter]
    fn get_gdtr(&self) -> (u64, u16) {
        (self.inner.regs.gdtr.base, self.inner.regs.gdtr.limit)
    }
    #[setter]
    fn set_gdtr(&mut self, v: (u64, u16)) {
        self.inner.regs.gdtr = DescTable {
            base: v.0,
            limit: v.1,
        };
    }

    /// IDTR as a `(base, limit)` tuple (in real mode it locates the
    /// interrupt vector table).
    #[getter]
    fn get_idtr(&self) -> (u64, u16) {
        (self.inner.regs.idtr.base, self.inner.regs.idtr.limit)
    }
    #[setter]
    fn set_idtr(&mut self, v: (u64, u16)) {
        self.inner.regs.idtr = DescTable {
            base: v.0,
            limit: v.1,
        };
    }

    fn __repr__(&self) -> String {
        let r = &self.inner.regs;
        let mode = if self.inner.long_mode() {
            "long"
        } else if self.inner.protected_mode() {
            "prot"
        } else {
            "real"
        };
        format!(
            "<remu.x86_64.Cpu rip=0x{:016X} rax=0x{:016X} {} cycles={}{}{}>",
            r.rip,
            r.gpr[reg::RAX as usize],
            mode,
            self.inner.cycles,
            if self.inner.halted { " HALTED" } else { "" },
            if self.inner.shutdown { " SHUTDOWN" } else { "" },
        )
    }
}

gpr_props! {
    /// Accumulator (RAX).
    RAX: get_rax / set_rax,
    /// Count register (RCX).
    RCX: get_rcx / set_rcx,
    /// Data register (RDX).
    RDX: get_rdx / set_rdx,
    /// Base register (RBX).
    RBX: get_rbx / set_rbx,
    /// Stack pointer (RSP).
    RSP: get_rsp / set_rsp,
    /// Base pointer (RBP).
    RBP: get_rbp / set_rbp,
    /// Source index (RSI).
    RSI: get_rsi / set_rsi,
    /// Destination index (RDI).
    RDI: get_rdi / set_rdi,
    /// General register R8.
    R8: get_r8 / set_r8,
    /// General register R9.
    R9: get_r9 / set_r9,
    /// General register R10.
    R10: get_r10 / set_r10,
    /// General register R11.
    R11: get_r11 / set_r11,
    /// General register R12.
    R12: get_r12 / set_r12,
    /// General register R13.
    R13: get_r13 / set_r13,
    /// General register R14.
    R14: get_r14 / set_r14,
    /// General register R15.
    R15: get_r15 / set_r15,
}

reg64_props! {
    /// Instruction pointer.
    rip: get_rip / set_rip,
    /// CR0 — protection/paging control bits.
    cr0: get_cr0 / set_cr0,
    /// CR2 — page-fault linear address.
    cr2: get_cr2 / set_cr2,
    /// CR3 — page-table base. Assigning here does not flush the TLB, unlike
    /// a guest `MOV CR3`.
    cr3: get_cr3 / set_cr3,
    /// CR4 — feature-enable bits.
    cr4: get_cr4 / set_cr4,
    /// CR8 — task-priority register (storage only; no local APIC is
    /// modeled).
    cr8: get_cr8 / set_cr8,
}

msr_props! {
    /// The IA32_EFER MSR — long-mode/SYSCALL enables (SCE/LME/LMA/NXE).
    efer: get_efer / set_efer,
    /// The IA32_STAR MSR — SYSCALL/SYSRET segment selectors.
    star: get_star / set_star,
    /// The IA32_LSTAR MSR — 64-bit SYSCALL target RIP.
    lstar: get_lstar / set_lstar,
    /// The IA32_CSTAR MSR — compatibility-mode SYSCALL target RIP (storage
    /// only, as on Intel hardware).
    cstar: get_cstar / set_cstar,
    /// The IA32_FMASK MSR — SYSCALL RFLAGS clear mask.
    sfmask: get_sfmask / set_sfmask,
    /// The IA32_KERNEL_GS_BASE MSR — the hidden GS base swapped in by
    /// SWAPGS.
    kernel_gs_base: get_kernel_gs_base / set_kernel_gs_base,
}

flag_props! {
    /// Carry flag.
    CF: get_carry / set_carry,
    /// Parity flag.
    PF: get_parity / set_parity,
    /// Auxiliary (BCD half-carry) flag.
    AF: get_adjust / set_adjust,
    /// Zero flag.
    ZF: get_zero / set_zero,
    /// Sign flag.
    SF: get_sign / set_sign,
    /// Trap (single-step) flag.
    TF: get_trap / set_trap,
    /// Interrupt-enable flag.
    IF: get_interrupt / set_interrupt,
    /// Direction flag.
    DF: get_direction / set_direction,
    /// Overflow flag.
    OF: get_overflow / set_overflow,
}

seg_props! {
    /// Extra segment as `(sel, base, limit, attrs)`; assign a selector int
    /// (real-mode semantics) or a 4-tuple.
    ES: get_es / set_es,
    /// Code segment as `(sel, base, limit, attrs)`; assign a selector int
    /// (real-mode semantics) or a 4-tuple.
    CS: get_cs / set_cs,
    /// Stack segment as `(sel, base, limit, attrs)`; assign a selector int
    /// (real-mode semantics) or a 4-tuple.
    SS: get_ss / set_ss,
    /// Data segment as `(sel, base, limit, attrs)`; assign a selector int
    /// (real-mode semantics) or a 4-tuple.
    DS: get_ds / set_ds,
    /// FS segment as `(sel, base, limit, attrs)` (64-bit base in long
    /// mode); assign a selector int (real-mode semantics) or a 4-tuple.
    FS: get_fs / set_fs,
    /// GS segment as `(sel, base, limit, attrs)` (64-bit base in long
    /// mode); assign a selector int (real-mode semantics) or a 4-tuple.
    GS: get_gs / set_gs,
}
