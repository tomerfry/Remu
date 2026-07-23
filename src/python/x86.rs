//! Python bindings for the Intel 8086/8088 core (`remu.x86`).
//!
//! ```python
//! from remu.x86 import Cpu, Memory
//!
//! mem = Memory()
//! mem.load(0x1000, b"\xB8\x34\x12\xF4")  # MOV AX, 0x1234; HLT
//! cpu = Cpu()
//! cpu.cs, cpu.ip = 0x0000, 0x1000
//! cpu.run(mem, 10)
//! assert cpu.ax == 0x1234 and cpu.halted
//! ```

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::util;
use crate::x86::{Bus, Cpu, Flags, LinearMemory};

const ADDRESS_SPACE: usize = 0x10_0000; // 1 MiB (20-bit physical)

// --- Bus dispatch -------------------------------------------------------------

/// A [`Bus`] that forwards accesses to a Python object's `read`/`write` (and
/// optional `io_read`/`io_write`) methods. See [`util::cb_read`].
struct CallbackBus<'py> {
    obj: Bound<'py, PyAny>,
    has_io_read: bool,
    has_io_write: bool,
    error: Option<PyErr>,
}

/// The bus argument accepted by every CPU entry point: a native
/// [`Memory8086`](PyMemory8086) or a duck-typed Python object.
enum BusArg<'py> {
    Flat(PyRefMut<'py, PyMemory8086>),
    Callback(CallbackBus<'py>),
}

impl<'py> BusArg<'py> {
    fn from_any(bus: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(mem) = bus.cast::<PyMemory8086>() {
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
        Err(util::bus_type_error("remu.x86.Memory"))
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

    fn io_read(&mut self, port: u16) -> u8 {
        util::cb_io_read(&self.obj, self.has_io_read, &mut self.error, port)
    }

    fn io_write(&mut self, port: u16, value: u8) {
        util::cb_io_write(&self.obj, self.has_io_write, &mut self.error, port, value)
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

/// Python-facing wrapper over the 8086 [`LinearMemory`]: a flat 1 MiB address
/// space with `mem[addr]` / `mem[start:stop]` indexing.
#[pyclass(name = "Memory8086", module = "remu._remu")]
pub struct PyMemory8086 {
    pub(crate) inner: LinearMemory,
}

#[pymethods]
impl PyMemory8086 {
    /// Create a zero-initialized 1 MiB memory.
    #[new]
    fn new() -> Self {
        PyMemory8086 {
            inner: LinearMemory::new(),
        }
    }

    /// Load `data` (bytes-like or iterable of ints) at `addr`, wrapping at 1 MiB.
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
        ADDRESS_SPACE
    }

    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        util::ram_getitem(py, &self.inner.ram[..], index)
    }

    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        util::ram_setitem(&mut self.inner.ram[..], index, value)
    }

    /// The full 1 MiB as `bytes` (supports `bytes(mem)`).
    fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.ram[..])
    }

    fn __repr__(&self) -> &'static str {
        "<remu.x86.Memory 1MiB>"
    }
}

// --- Cpu ----------------------------------------------------------------------

/// Getter/setter pairs for plain `u16` register fields.
macro_rules! reg16_props {
    ($($(#[doc = $doc:expr])* $name:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu8086 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> u16 {
                    self.inner.regs.$name
                }
                #[setter]
                fn $set(&mut self, v: u16) {
                    self.inner.regs.$name = v;
                }
            )+
        }
    };
}

/// Getter/setter pairs for the 8-bit halves of a 16-bit register.
macro_rules! reg8_props {
    ($($reg:ident : $lo_get:ident / $lo_set:ident, $hi_get:ident / $hi_set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu8086 {
            $(
                #[getter]
                fn $lo_get(&self) -> u8 {
                    self.inner.regs.$reg as u8
                }
                #[setter]
                fn $lo_set(&mut self, v: u8) {
                    self.inner.regs.$reg = (self.inner.regs.$reg & 0xFF00) | v as u16;
                }
                #[getter]
                fn $hi_get(&self) -> u8 {
                    (self.inner.regs.$reg >> 8) as u8
                }
                #[setter]
                fn $hi_set(&mut self, v: u8) {
                    self.inner.regs.$reg = (self.inner.regs.$reg & 0x00FF) | ((v as u16) << 8);
                }
            )+
        }
    };
}

/// Getter/setter pairs for individual FLAGS bits.
macro_rules! flag_props {
    ($($(#[doc = $doc:expr])* $bit:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu8086 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> bool {
                    self.inner.regs.flags.contains(Flags::$bit)
                }
                #[setter]
                fn $set(&mut self, v: bool) {
                    self.inner.regs.flags.set(Flags::$bit, v);
                }
            )+
        }
    };
}

/// Python-facing wrapper over the 8086 [`Cpu`]. Registers and flags are flat
/// read/write properties; every method that touches memory takes the bus as
/// an argument (the CPU does not own its bus).
#[pyclass(name = "Cpu8086", module = "remu._remu")]
pub struct PyCpu8086 {
    inner: Cpu,
}

#[pymethods]
impl PyCpu8086 {
    /// Create a CPU in its power-on state (`CS:IP = FFFF:0000`).
    #[new]
    fn new() -> Self {
        PyCpu8086 { inner: Cpu::new() }
    }

    /// RESET: back to the power-on state (registers, halt latch, lines).
    fn reset(&mut self) {
        self.inner.reset();
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

    /// Execute up to `instructions` instructions, stopping early on HLT
    /// (assert an interrupt and call `run` again to resume) or when a Python
    /// bus callback raises (see `step` for its bus semantics). Returns the
    /// number of instructions actually executed. The hot loop runs in Rust.
    ///
    /// Every 64 Ki instructions the loop polls for signals (Ctrl-C raises
    /// `KeyboardInterrupt`) and briefly releases the GIL.
    fn run(&mut self, bus: &Bound<'_, PyAny>, instructions: u64) -> PyResult<u64> {
        /// Rare enough to cost nothing, frequent enough that Ctrl-C and
        /// waiting threads feel it instantly.
        const CHUNK: u64 = 0x10000;

        let py = bus.py();
        let mut executed: u64 = 0;
        while executed < instructions && !self.inner.halted {
            let target = (instructions - executed).min(CHUNK);
            // Re-borrowed each chunk so the borrow is dropped before detaching.
            let mut b = BusArg::from_any(bus)?;
            let mut n: u64 = 0;
            match &mut b {
                BusArg::Flat(m) => {
                    let mem = &mut m.inner;
                    while n < target && !self.inner.halted {
                        self.inner.step(mem);
                        n += 1;
                    }
                }
                BusArg::Callback(c) => {
                    while n < target && !self.inner.halted && c.error.is_none() {
                        self.inner.step(c);
                        n += 1;
                    }
                }
            }
            executed += n;
            b.check()?;
            py.check_signals()?;
            py.detach(|| {});
        }
        Ok(executed)
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

    /// Raw FLAGS word (bits 12-15/1 read as set, 3/5 clear, like the 8086).
    #[getter]
    fn get_flags(&self) -> u16 {
        self.inner.regs.flags.to_word()
    }
    #[setter]
    fn set_flags(&mut self, v: u16) {
        self.inner.regs.flags = Flags::from_word(v);
    }

    /// Total cycles elapsed since construction.
    #[getter]
    fn get_cycles(&self) -> u64 {
        self.inner.cycles
    }

    /// True while a HLT has the processor stopped (an interrupt resumes it).
    #[getter]
    fn get_halted(&self) -> bool {
        self.inner.halted
    }

    fn __repr__(&self) -> String {
        let r = &self.inner.regs;
        format!(
            "<remu.x86.Cpu cs:ip={:04X}:{:04X} ax={:04X} bx={:04X} cx={:04X} dx={:04X} \
             flags={:04X} cycles={}{}>",
            r.cs,
            r.ip,
            r.ax,
            r.bx,
            r.cx,
            r.dx,
            r.flags.to_word(),
            self.inner.cycles,
            if self.inner.halted { " HALTED" } else { "" },
        )
    }
}

reg16_props! {
    /// Accumulator.
    ax: get_ax / set_ax,
    /// Base register.
    bx: get_bx / set_bx,
    /// Count register.
    cx: get_cx / set_cx,
    /// Data register.
    dx: get_dx / set_dx,
    /// Stack pointer.
    sp: get_sp / set_sp,
    /// Base pointer.
    bp: get_bp / set_bp,
    /// Source index.
    si: get_si / set_si,
    /// Destination index.
    di: get_di / set_di,
    /// Instruction pointer.
    ip: get_ip / set_ip,
    /// Extra segment.
    es: get_es / set_es,
    /// Code segment.
    cs: get_cs / set_cs,
    /// Stack segment.
    ss: get_ss / set_ss,
    /// Data segment.
    ds: get_ds / set_ds,
}

reg8_props! {
    ax: get_al / set_al, get_ah / set_ah,
    bx: get_bl / set_bl, get_bh / set_bh,
    cx: get_cl / set_cl, get_ch / set_ch,
    dx: get_dl / set_dl, get_dh / set_dh,
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
