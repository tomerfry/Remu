//! Python bindings for the MOS 6502 core (`remu.mos6502`, also re-exported at
//! the top level as `remu.Cpu` / `remu.Memory` / `remu.disassemble`).
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

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::util;
use crate::bus::Bus;
use crate::cpu::disasm;
use crate::memory::FlatMemory;
use crate::{Cpu, Status};

// --- Bus dispatch -------------------------------------------------------------

/// A [`Bus`] that forwards each access to a Python object's `read`/`write`
/// methods (the 6502 has no bus fault to map an exception to, so the access
/// reads as 0 / the write is dropped and the exception re-raises once the
/// current instruction finishes).
struct CallbackBus<'py> {
    obj: Bound<'py, PyAny>,
    error: Option<PyErr>,
}

/// The bus argument accepted by every CPU entry point: a native
/// [`Memory6502`](PyMemory6502) (borrowed mutably for the duration of the
/// call, no Python overhead per access) or a duck-typed Python object.
enum BusArg<'py> {
    Flat(PyRefMut<'py, PyMemory6502>),
    Callback(CallbackBus<'py>),
}

impl<'py> BusArg<'py> {
    fn from_any(bus: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(mem) = bus.cast::<PyMemory6502>() {
            return Ok(BusArg::Flat(mem.try_borrow_mut()?));
        }
        if util::is_duck_bus(bus)? {
            return Ok(BusArg::Callback(CallbackBus {
                obj: bus.clone(),
                error: None,
            }));
        }
        Err(util::bus_type_error("remu.Memory"))
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
    fn read(&mut self, addr: u16) -> u8 {
        util::cb_read(&self.obj, &mut self.error, addr as u64)
    }

    fn write(&mut self, addr: u16, value: u8) {
        util::cb_write(&self.obj, &mut self.error, addr as u64, value)
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

// --- Memory -------------------------------------------------------------------

/// Python-facing wrapper over [`FlatMemory`]: a flat 64 KiB address space with
/// `mem[addr]` / `mem[start:stop]` indexing.
#[pyclass(name = "Memory6502", module = "remu._remu")]
pub struct PyMemory6502 {
    inner: FlatMemory,
}

#[pymethods]
impl PyMemory6502 {
    /// Create a zero-initialized 64 KiB memory.
    #[new]
    fn new() -> Self {
        PyMemory6502 {
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
        0x10000
    }

    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        util::ram_getitem(py, &self.inner.ram[..], index)
    }

    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        util::ram_setitem(&mut self.inner.ram[..], index, value)
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

/// Getter/setter pairs for plain `u8`/`u16` register fields.
macro_rules! reg_props {
    ($($(#[doc = $doc:expr])* $name:ident : $ty:ty = $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu6502 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> $ty {
                    self.inner.regs.$name
                }
                #[setter]
                fn $set(&mut self, v: $ty) {
                    self.inner.regs.$name = v;
                }
            )+
        }
    };
}

/// Getter/setter pairs for individual status-register bits.
macro_rules! flag_props {
    ($($(#[doc = $doc:expr])* $bit:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu6502 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> bool {
                    self.inner.regs.p.contains(Status::$bit)
                }
                #[setter]
                fn $set(&mut self, v: bool) {
                    self.inner.regs.p.set(Status::$bit, v);
                }
            )+
        }
    };
}

/// Python-facing wrapper over [`Cpu`]. Registers and flags are flat read/write
/// properties; every method that touches memory takes the bus as an argument,
/// mirroring the Rust design (the CPU does not own its bus).
#[pyclass(name = "Cpu6502", module = "remu._remu")]
pub struct PyCpu6502 {
    inner: Cpu,
}

#[pymethods]
impl PyCpu6502 {
    /// Create a CPU in its power-on state. Call `reset(bus)` before running.
    #[new]
    fn new() -> Self {
        PyCpu6502 { inner: Cpu::new() }
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

    /// Set the NMI line level; the line idles high, and a high→low
    /// (asserting) edge latches an NMI.
    fn set_nmi(&mut self, level: bool) {
        self.inner.set_nmi(level);
    }

    /// Directly latch a pending NMI.
    fn trigger_nmi(&mut self) {
        self.inner.trigger_nmi();
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

reg_props! {
    /// Accumulator.
    a: u8 = get_a / set_a,
    /// Index register X.
    x: u8 = get_x / set_x,
    /// Index register Y.
    y: u8 = get_y / set_y,
    /// Stack pointer (low byte; the stack lives at $0100..=$01FF).
    sp: u8 = get_sp / set_sp,
    /// Program counter.
    pc: u16 = get_pc / set_pc,
}

flag_props! {
    /// Carry flag.
    C: get_carry / set_carry,
    /// Zero flag.
    Z: get_zero / set_zero,
    /// Interrupt-disable flag.
    I: get_interrupt_disable / set_interrupt_disable,
    /// Decimal-mode flag.
    D: get_decimal / set_decimal,
    /// Overflow flag.
    V: get_overflow / set_overflow,
    /// Negative flag.
    N: get_negative / set_negative,
}

// --- Functions ----------------------------------------------------------------

/// Disassemble the instruction at `addr`, returning
/// `(text, address_of_next_instruction)`.
#[pyfunction]
#[pyo3(name = "disassemble6502")]
pub fn disassemble6502(bus: &Bound<'_, PyAny>, addr: u16) -> PyResult<(String, u16)> {
    let mut b = BusArg::from_any(bus)?;
    let result = disasm::disassemble(&mut b, addr);
    b.check()?;
    Ok(result)
}
