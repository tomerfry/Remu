//! Python bindings (PyO3), compiled only with the `python` cargo feature.
//!
//! Exposes the crate as the `remu` extension module, built with
//! [maturin](https://www.maturin.rs) — see `pyproject.toml`. The API mirrors the
//! Rust one but leans pythonic:
//!
//! ```python
//! import remu
//!
//! mem = remu.Memory()
//! mem[0x0600:0x0602] = b"\xA9\x42"   # LDA #$42
//! mem.set_reset_vector(0x0600)
//!
//! cpu = remu.Cpu()
//! cpu.reset(mem)
//! cpu.step(mem)
//! assert cpu.a == 0x42
//! ```
//!
//! Anywhere a bus is expected, either a [`Memory`](PyMemory) (fast, stays in
//! Rust) or any Python object with `read(addr)` / `write(addr, value)` methods
//! (flexible, for MMIO experiments) is accepted.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PySlice};

use crate::bus::Bus;
use crate::cpu::disasm;
use crate::memory::FlatMemory;
use crate::{Cpu, Status};

const ADDRESS_SPACE: isize = 0x10000;

// --- Bus dispatch -----------------------------------------------------------

/// A [`Bus`] that forwards each access to a Python object's `read`/`write`
/// methods. The first exception raised by a callback is stashed (the 6502 has
/// no bus fault to map it to, so the access reads as 0 / the write is dropped)
/// and re-raised once the current instruction finishes.
struct CallbackBus<'py> {
    obj: Bound<'py, PyAny>,
    error: Option<PyErr>,
}

impl Bus for CallbackBus<'_> {
    fn read(&mut self, addr: u16) -> u8 {
        if self.error.is_some() {
            return 0;
        }
        let read = intern!(self.obj.py(), "read");
        match self
            .obj
            .call_method1(read, (addr,))
            .and_then(|v| v.extract::<u8>())
        {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                0
            }
        }
    }

    fn write(&mut self, addr: u16, value: u8) {
        if self.error.is_some() {
            return;
        }
        let write = intern!(self.obj.py(), "write");
        if let Err(e) = self.obj.call_method1(write, (addr, value)) {
            self.error = Some(e);
        }
    }
}

/// The bus argument accepted by every CPU entry point: a native [`PyMemory`]
/// (borrowed mutably for the duration of the call, no Python overhead per
/// access) or a duck-typed Python object.
enum BusArg<'py> {
    Flat(PyRefMut<'py, PyMemory>),
    Callback(CallbackBus<'py>),
}

impl<'py> BusArg<'py> {
    fn from_any(bus: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(mem) = bus.cast::<PyMemory>() {
            return Ok(BusArg::Flat(mem.try_borrow_mut()?));
        }
        let py = bus.py();
        if bus.hasattr(intern!(py, "read"))? && bus.hasattr(intern!(py, "write"))? {
            return Ok(BusArg::Callback(CallbackBus {
                obj: bus.clone(),
                error: None,
            }));
        }
        Err(PyTypeError::new_err(
            "bus must be a remu.Memory or an object with read(addr) and write(addr, value) methods",
        ))
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

impl Bus for BusArg<'_> {
    fn read(&mut self, addr: u16) -> u8 {
        match self {
            BusArg::Flat(m) => m.inner.read(addr),
            BusArg::Callback(c) => c.read(addr),
        }
    }

    fn write(&mut self, addr: u16, value: u8) {
        match self {
            BusArg::Flat(m) => m.inner.write(addr, value),
            BusArg::Callback(c) => c.write(addr, value),
        }
    }
}

// --- Memory ------------------------------------------------------------------

/// Python-facing wrapper over [`FlatMemory`]: a flat 64 KiB address space with
/// `mem[addr]` / `mem[start:stop]` indexing.
#[pyclass(name = "Memory", module = "remu")]
pub struct PyMemory {
    inner: FlatMemory,
}

fn normalize_index(index: &Bound<'_, PyAny>) -> PyResult<usize> {
    if !index.is_instance_of::<pyo3::types::PyInt>() {
        return Err(PyTypeError::new_err(
            "memory indices must be integers or slices",
        ));
    }
    // An int that doesn't fit isize is out of range like any other (list semantics).
    let i: isize = index
        .extract()
        .map_err(|_| PyIndexError::new_err("address out of range (0..=0xFFFF)"))?;
    let i = if i < 0 { i + ADDRESS_SPACE } else { i };
    if !(0..ADDRESS_SPACE).contains(&i) {
        return Err(PyIndexError::new_err("address out of range (0..=0xFFFF)"));
    }
    Ok(i as usize)
}

#[pymethods]
impl PyMemory {
    /// Create a zero-initialized 64 KiB memory.
    #[new]
    fn new() -> Self {
        PyMemory {
            inner: FlatMemory::new(),
        }
    }

    /// Load `data` (bytes-like or iterable of ints) at `addr`, wrapping at the top.
    fn load(&mut self, addr: u16, data: Vec<u8>) {
        self.inner.load(addr, &data);
    }

    /// Point the reset vector (`$FFFC/$FFFD`) at `addr`.
    fn set_reset_vector(&mut self, addr: u16) {
        self.inner.set_reset_vector(addr);
    }

    /// Read one byte (also lets a `Memory` be used where a duck-typed bus is expected).
    fn read(&mut self, addr: u16) -> u8 {
        self.inner.read(addr)
    }

    /// Write one byte.
    fn write(&mut self, addr: u16, value: u8) {
        self.inner.write(addr, value);
    }

    fn __len__(&self) -> usize {
        ADDRESS_SPACE as usize
    }

    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(slice) = index.cast::<PySlice>() {
            let idx = slice.indices(ADDRESS_SPACE)?;
            let mut out = Vec::with_capacity(idx.slicelength);
            let mut i = idx.start;
            for _ in 0..idx.slicelength {
                out.push(self.inner.ram[i as usize]);
                i += idx.step;
            }
            PyBytes::new(py, &out).into_py_any(py)
        } else {
            self.inner.ram[normalize_index(index)?].into_py_any(py)
        }
    }

    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(slice) = index.cast::<PySlice>() {
            let idx = slice.indices(ADDRESS_SPACE)?;
            let data: Vec<u8> = value.extract().map_err(|_| {
                PyTypeError::new_err("expected bytes-like or iterable of ints in 0..=255")
            })?;
            if data.len() != idx.slicelength {
                return Err(PyValueError::new_err(format!(
                    "cannot assign {} bytes to slice of length {}",
                    data.len(),
                    idx.slicelength
                )));
            }
            let mut i = idx.start;
            for b in data {
                self.inner.ram[i as usize] = b;
                i += idx.step;
            }
            Ok(())
        } else {
            let i = normalize_index(index)?;
            self.inner.ram[i] = value.extract()?;
            Ok(())
        }
    }

    /// The full 64 KiB as `bytes` (supports `bytes(mem)`).
    fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.ram[..])
    }

    fn __repr__(&self) -> &'static str {
        "<remu.Memory 64KiB>"
    }
}

// --- Cpu ----------------------------------------------------------------------

/// Render `P` as `nv--dizc` with set flags uppercased (B/U shown as `--`).
fn flags_str(p: Status) -> String {
    let f = |bit: Status, ch: char| {
        if p.contains(bit) {
            ch.to_ascii_uppercase()
        } else {
            ch
        }
    };
    format!(
        "{}{}--{}{}{}{}",
        f(Status::N, 'n'),
        f(Status::V, 'v'),
        f(Status::D, 'd'),
        f(Status::I, 'i'),
        f(Status::Z, 'z'),
        f(Status::C, 'c'),
    )
}

/// Python-facing wrapper over [`Cpu`]. Registers and flags are flat read/write
/// properties; every method that touches memory takes the bus as an argument,
/// mirroring the Rust design (the CPU does not own its bus).
#[pyclass(name = "Cpu", module = "remu")]
pub struct PyCpu {
    inner: Cpu,
}

#[pymethods]
impl PyCpu {
    /// Create a CPU in its power-on state. Call `reset(bus)` before running.
    #[new]
    fn new() -> Self {
        PyCpu { inner: Cpu::new() }
    }

    /// RESET: load `pc` from the reset vector ($FFFC), set `i`, `sp = $FD`.
    fn reset(&mut self, bus: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut b = BusArg::from_any(bus)?;
        self.inner.reset(&mut b);
        b.check()
    }

    /// Execute one instruction (or service a pending interrupt). Returns the
    /// cycles consumed (0 if the CPU is jammed). If a bus callback raises, the
    /// exception propagates after the instruction finishes against a bus that
    /// reads as 0 / drops writes — CPU state reflects that partial execution.
    fn step(&mut self, bus: &Bound<'_, PyAny>) -> PyResult<u8> {
        let mut b = BusArg::from_any(bus)?;
        let cycles = self.inner.step(&mut b);
        b.check()?;
        Ok(cycles)
    }

    /// Execute up to `instructions` instructions, stopping early if the CPU
    /// jams (KIL) or a Python bus callback raises (the aborted instruction is
    /// included in the count; see `step` for its bus semantics). Returns the
    /// number of instructions actually executed. The hot loop runs in Rust,
    /// so this is the fast way to drive the CPU from Python.
    ///
    /// Every 64 Ki instructions the loop polls for signals (Ctrl-C raises
    /// `KeyboardInterrupt` instead of spinning to the budget) and briefly
    /// releases the GIL so other Python threads aren't starved.
    fn run(&mut self, bus: &Bound<'_, PyAny>, instructions: u64) -> PyResult<u64> {
        /// Chunk size: rare enough to cost nothing (~200 µs of emulation),
        /// frequent enough that Ctrl-C and waiting threads feel it instantly.
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

    /// Set the IRQ line level (level-triggered, serviced while `i` is clear).
    fn set_irq(&mut self, level: bool) {
        self.inner.set_irq(level);
    }

    /// Set the NMI line level (a low→high transition latches an NMI).
    fn set_nmi(&mut self, level: bool) {
        self.inner.set_nmi(level);
    }

    /// Directly latch a pending NMI.
    fn trigger_nmi(&mut self) {
        self.inner.trigger_nmi();
    }

    // --- Registers ---

    /// Accumulator.
    #[getter]
    fn get_a(&self) -> u8 {
        self.inner.regs.a
    }
    #[setter]
    fn set_a(&mut self, v: u8) {
        self.inner.regs.a = v;
    }

    /// Index register X.
    #[getter]
    fn get_x(&self) -> u8 {
        self.inner.regs.x
    }
    #[setter]
    fn set_x(&mut self, v: u8) {
        self.inner.regs.x = v;
    }

    /// Index register Y.
    #[getter]
    fn get_y(&self) -> u8 {
        self.inner.regs.y
    }
    #[setter]
    fn set_y(&mut self, v: u8) {
        self.inner.regs.y = v;
    }

    /// Stack pointer (low byte; the stack lives at $0100..=$01FF).
    #[getter]
    fn get_sp(&self) -> u8 {
        self.inner.regs.sp
    }
    #[setter]
    fn set_sp(&mut self, v: u8) {
        self.inner.regs.sp = v;
    }

    /// Program counter.
    #[getter]
    fn get_pc(&self) -> u16 {
        self.inner.regs.pc
    }
    #[setter]
    fn set_pc(&mut self, v: u16) {
        self.inner.regs.pc = v;
    }

    /// Raw status byte. Writes keep the in-memory convention: `U` forced set,
    /// `B` forced clear (neither exists as real storage on the 6502).
    #[getter]
    fn get_p(&self) -> u8 {
        self.inner.regs.p.bits()
    }
    #[setter]
    fn set_p(&mut self, v: u8) {
        self.inner.regs.p = Status::from_bits_retain((v | Status::U.bits()) & !Status::B.bits());
    }

    // --- Flags ---

    /// Carry flag.
    #[getter]
    fn get_carry(&self) -> bool {
        self.inner.regs.p.contains(Status::C)
    }
    #[setter]
    fn set_carry(&mut self, v: bool) {
        self.inner.regs.p.set(Status::C, v);
    }

    /// Zero flag.
    #[getter]
    fn get_zero(&self) -> bool {
        self.inner.regs.p.contains(Status::Z)
    }
    #[setter]
    fn set_zero(&mut self, v: bool) {
        self.inner.regs.p.set(Status::Z, v);
    }

    /// Interrupt-disable flag.
    #[getter]
    fn get_interrupt_disable(&self) -> bool {
        self.inner.regs.p.contains(Status::I)
    }
    #[setter]
    fn set_interrupt_disable(&mut self, v: bool) {
        self.inner.regs.p.set(Status::I, v);
    }

    /// Decimal-mode flag.
    #[getter]
    fn get_decimal(&self) -> bool {
        self.inner.regs.p.contains(Status::D)
    }
    #[setter]
    fn set_decimal(&mut self, v: bool) {
        self.inner.regs.p.set(Status::D, v);
    }

    /// Overflow flag.
    #[getter]
    fn get_overflow(&self) -> bool {
        self.inner.regs.p.contains(Status::V)
    }
    #[setter]
    fn set_overflow(&mut self, v: bool) {
        self.inner.regs.p.set(Status::V, v);
    }

    /// Negative flag.
    #[getter]
    fn get_negative(&self) -> bool {
        self.inner.regs.p.contains(Status::N)
    }
    #[setter]
    fn set_negative(&mut self, v: bool) {
        self.inner.regs.p.set(Status::N, v);
    }

    // --- State ---

    /// Total cycles elapsed since construction.
    #[getter]
    fn get_cycles(&self) -> u64 {
        self.inner.cycles
    }

    /// True once a KIL/jam opcode has halted the processor (cleared by `reset`).
    #[getter]
    fn get_halted(&self) -> bool {
        self.inner.halted
    }

    fn __repr__(&self) -> String {
        let r = &self.inner.regs;
        format!(
            "<remu.Cpu pc=${:04X} a=${:02X} x=${:02X} y=${:02X} sp=${:02X} p={} cycles={}{}>",
            r.pc,
            r.a,
            r.x,
            r.y,
            r.sp,
            flags_str(r.p),
            self.inner.cycles,
            if self.inner.halted { " HALTED" } else { "" },
        )
    }
}

// --- Module -------------------------------------------------------------------

/// Disassemble the instruction at `addr`, returning
/// `(text, address_of_next_instruction)`.
#[pyfunction]
fn disassemble(bus: &Bound<'_, PyAny>, addr: u16) -> PyResult<(String, u16)> {
    let mut b = BusArg::from_any(bus)?;
    let result = disasm::disassemble(&mut b, addr);
    b.check()?;
    Ok(result)
}

#[pymodule]
fn remu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCpu>()?;
    m.add_class::<PyMemory>()?;
    m.add_function(wrap_pyfunction!(disassemble, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
