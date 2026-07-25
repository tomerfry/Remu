//! Python bindings for the Intel 80386 core (`remu.x86_32`).
//!
//! ```python
//! from remu.x86_32 import Cpu, Memory, RunExit
//!
//! mem = Memory()
//! mem.load(0x1000, b"\xB8\x78\x56\x34\x12\xF4")  # MOV EAX, 0x12345678; HLT
//! cpu = Cpu()
//! cpu.enter_flat_protected()
//! cpu.eip, cpu.esp = 0x1000, 0x8000
//! executed, exit = cpu.run(mem, 10)
//! assert cpu.eax == 0x12345678 and exit == RunExit.Halted
//! ```

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::util;
use crate::x86_32::{
    Bus, Cpu, DescTable, EFlags, HostTrap, LinearMemory, RunExit, SegReg, cr0, reg,
};

const ADDRESS_SPACE: usize = 0x100_0000; // 16 MiB (24-bit physical)

// --- Bus dispatch -------------------------------------------------------------

/// A [`Bus`] that forwards accesses to a Python object's `read`/`write` (and
/// optional `io_read`/`io_write`) methods. See [`util::cb_read`]. Wide
/// accesses are byte-composed, little-endian, so duck-typed buses only ever
/// see byte traffic.
struct CallbackBus<'py> {
    obj: Bound<'py, PyAny>,
    has_io_read: bool,
    has_io_write: bool,
    error: Option<PyErr>,
}

/// The bus argument accepted by every CPU entry point: a native
/// [`Memory386`](PyMemory386) or a duck-typed Python object.
enum BusArg<'py> {
    Flat(PyRefMut<'py, PyMemory386>),
    Callback(CallbackBus<'py>),
}

impl<'py> BusArg<'py> {
    fn from_any(bus: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(mem) = bus.cast::<PyMemory386>() {
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
        Err(util::bus_type_error("remu.x86_32.Memory"))
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

    fn read16(&mut self, addr: u32) -> u16 {
        self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
    }

    fn read32(&mut self, addr: u32) -> u32 {
        self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
    }

    fn write16(&mut self, addr: u32, value: u16) {
        self.write(addr, value as u8);
        self.write(addr.wrapping_add(1), (value >> 8) as u8);
    }

    fn write32(&mut self, addr: u32, value: u32) {
        self.write16(addr, value as u16);
        self.write16(addr.wrapping_add(2), (value >> 16) as u16);
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

/// Python-facing wrapper over the 386 [`LinearMemory`]: a flat 16 MiB address
/// space with `mem[addr]` / `mem[start:stop]` indexing.
#[pyclass(name = "Memory386", module = "remu._remu")]
pub struct PyMemory386 {
    pub(crate) inner: LinearMemory,
}

#[pymethods]
impl PyMemory386 {
    /// Create a zero-initialized 16 MiB memory.
    #[new]
    fn new() -> Self {
        PyMemory386 {
            inner: LinearMemory::new(),
        }
    }

    /// Load `data` (bytes-like or iterable of ints) at `addr`, wrapping at
    /// 16 MiB.
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

    /// The full 16 MiB as `bytes` (supports `bytes(mem)`).
    fn __bytes__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.ram[..])
    }

    fn __repr__(&self) -> &'static str {
        "<remu.x86_32.Memory 16MiB>"
    }
}

// --- Cpu ----------------------------------------------------------------------

/// Getter/setter pairs for the 32-bit general registers (`gpr` slots).
macro_rules! gpr_props {
    ($($(#[doc = $doc:expr])* $idx:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu386 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> u32 {
                    self.inner.regs.gpr[reg::$idx as usize]
                }
                #[setter]
                fn $set(&mut self, v: u32) {
                    self.inner.regs.gpr[reg::$idx as usize] = v;
                }
            )+
        }
    };
}

/// Getter/setter pairs for plain `u32` register-file fields.
macro_rules! u32_props {
    ($($(#[doc = $doc:expr])* $name:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu386 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> u32 {
                    self.inner.regs.$name
                }
                #[setter]
                fn $set(&mut self, v: u32) {
                    self.inner.regs.$name = v;
                }
            )+
        }
    };
}

/// Getter/setter pairs for individual EFLAGS bits.
macro_rules! flag_props {
    ($($(#[doc = $doc:expr])* $bit:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu386 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> bool {
                    self.inner.regs.eflags.contains(EFlags::$bit)
                }
                #[setter]
                fn $set(&mut self, v: bool) {
                    self.inner.regs.eflags.set(EFlags::$bit, v);
                }
            )+
        }
    };
}

/// Getter/setter pairs for the segment registers, exposed as
/// `(sel, base, limit, attrs)` tuples; assigning an int loads the selector
/// with real-mode semantics (see [`seg_from_any`]).
macro_rules! seg_props {
    ($($(#[doc = $doc:expr])* $idx:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu386 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> (u16, u32, u32, u16) {
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

/// Getter/setter pairs for the descriptor-table registers (GDTR/IDTR),
/// exposed as `(base, limit)` tuples.
macro_rules! dtr_props {
    ($($(#[doc = $doc:expr])* $name:ident : $get:ident / $set:ident),+ $(,)?) => {
        #[pymethods]
        impl PyCpu386 {
            $(
                $(#[doc = $doc])*
                #[getter]
                fn $get(&self) -> (u32, u16) {
                    let t = self.inner.regs.$name;
                    (t.base, t.limit)
                }
                #[setter]
                fn $set(&mut self, v: (u32, u16)) {
                    self.inner.regs.$name = DescTable {
                        base: v.0,
                        limit: v.1,
                    };
                }
            )+
        }
    };
}

/// Convert a Python segment-property value: an int selector loads with
/// real-mode semantics (`base = sel * 16`, 64 KiB limit), a
/// `(sel, base, limit, attrs)` tuple loads the descriptor cache raw.
fn seg_from_any(v: &Bound<'_, PyAny>) -> PyResult<SegReg> {
    if v.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(SegReg::real(v.extract::<u16>()?));
    }
    let (sel, base, limit, attrs) = v.extract::<(u16, u32, u32, u16)>().map_err(|_| {
        PyTypeError::new_err(
            "segment must be an int selector or a (sel, base, limit, attrs) tuple",
        )
    })?;
    Ok(SegReg {
        sel,
        base,
        limit,
        attrs,
    })
}

/// A [`HostTrap`] as seen from Python: `None`, `"syscall"`, or
/// `("exception", vector, error_code_or_None)`.
fn host_trap_to_py(py: Python<'_>, trap: Option<HostTrap>) -> PyResult<Py<PyAny>> {
    match trap {
        None => Ok(py.None()),
        Some(HostTrap::Syscall) => "syscall".into_py_any(py),
        Some(HostTrap::Exception(e)) => ("exception", e.vector, e.error).into_py_any(py),
    }
}

/// Python-facing wrapper over the 386 [`Cpu`]. Registers and flags are flat
/// read/write properties; every method that touches memory takes the bus as
/// an argument (the CPU does not own its bus).
#[pyclass(name = "Cpu386", module = "remu._remu")]
pub struct PyCpu386 {
    inner: Cpu,
}

#[pymethods]
impl PyCpu386 {
    /// Create a CPU in its power-on state: real mode, execution begins at
    /// `F000:FFF0` (CS base `FFFF0000`, the 386 reset quirk).
    #[new]
    fn new() -> Self {
        PyCpu386 { inner: Cpu::new() }
    }

    /// RESET: back to the power-on state (registers, halt/shutdown latches,
    /// interrupt lines, pending host trap, translation caches).
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Set `CS:EIP` with real-mode semantics (CS base = `sel * 16`) — the
    /// entry-point helper for real-mode programs.
    fn set_cs_ip(&mut self, sel: u16, eip: u32) {
        self.inner.set_cs_ip(sel, eip);
    }

    /// Enter flat protected mode the fast way: set `CR0.PE` and load every
    /// segment cache with a flat 4 GiB ring-0 segment (32-bit code at
    /// selector `0x08`, data/stack at `0x10`, base 0, limit `0xFFFFFFFF`),
    /// with no GDT or far jump needed. `EIP` and `ESP` are left untouched —
    /// point them at your code and stack, then `run`.
    fn enter_flat_protected(&mut self) {
        self.inner.regs.cr0 |= cr0::PE;
        self.inner.regs.seg[reg::CS as usize] = SegReg {
            sel: 0x08,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: 0x0C9B, // present, code, exec/read, accessed; G+D
        };
        for s in [reg::SS, reg::DS, reg::ES, reg::FS, reg::GS] {
            self.inner.regs.seg[s as usize] = SegReg {
                sel: 0x10,
                base: 0,
                limit: 0xFFFF_FFFF,
                attrs: 0x0C93, // present, data, read/write, accessed; G+D
            };
        }
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

    /// Execute up to `instructions` step-units and return
    /// `(executed, RunExit)`: how many actually ran, and why the run ended.
    /// `RunExit.Completed` means the full budget ran; `HostTrap` means a
    /// syscall or trapped fault is pending (service `take_host_trap()` and
    /// call `run` again); `Halted` means HLT (assert an interrupt and `step`
    /// to resume); `Shutdown` means triple fault (only `reset`
    /// recovers). A Python bus callback raising also ends the run (see `step`
    /// for its bus semantics). The hot loop runs in Rust — with a native
    /// `Memory` it takes the core's batched (JIT-ready) path.
    ///
    /// Every 64 Ki instructions the loop polls for signals (Ctrl-C raises
    /// `KeyboardInterrupt`) and briefly releases the GIL.
    fn run(&mut self, bus: &Bound<'_, PyAny>, instructions: u64) -> PyResult<(u64, super::PyRunExit)> {
        /// Rare enough to cost nothing, frequent enough that Ctrl-C and
        /// waiting threads feel it instantly.
        const CHUNK: u64 = 0x10000;

        let py = bus.py();
        let mut executed: u64 = 0;
        let mut exit = super::PyRunExit::Completed;
        while executed < instructions {
            let target = (instructions - executed).min(CHUNK);
            // Re-borrowed each chunk so the borrow is dropped before detaching.
            let mut b = BusArg::from_any(bus)?;
            let n = match &mut b {
                BusArg::Flat(m) => {
                    let r = self.inner.run(&mut m.inner, target);
                    exit = match r.exit {
                        RunExit::Completed => super::PyRunExit::Completed,
                        RunExit::HostTrap => super::PyRunExit::HostTrap,
                        RunExit::Halted => super::PyRunExit::Halted,
                        _ => super::PyRunExit::Shutdown,
                    };
                    r.executed
                }
                BusArg::Callback(c) => {
                    let cpu = &mut self.inner;
                    let mut n: u64 = 0;
                    while n < target
                        && !cpu.halted
                        && !cpu.shutdown
                        && cpu.host_trap.is_none()
                        && c.error.is_none()
                    {
                        cpu.step(c);
                        n += 1;
                    }
                    exit = if cpu.host_trap.is_some() {
                        super::PyRunExit::HostTrap
                    } else if cpu.halted {
                        super::PyRunExit::Halted
                    } else if cpu.shutdown {
                        super::PyRunExit::Shutdown
                    } else {
                        super::PyRunExit::Completed
                    };
                    n
                }
            };
            executed += n;
            b.check()?;
            if exit != super::PyRunExit::Completed {
                break;
            }
            py.check_signals()?;
            py.detach(|| {});
        }
        Ok((executed, exit))
    }

    /// Latch a non-maskable interrupt (vector 2, not maskable by IF).
    fn trigger_nmi(&mut self) {
        self.inner.trigger_nmi();
    }

    /// Assert the INTR line with `vector` (as a PIC would supply during the
    /// interrupt-acknowledge cycle). Serviced while IF is set.
    fn assert_intr(&mut self, vector: u8) {
        self.inner.assert_intr(vector);
    }

    /// Deassert the INTR line.
    fn clear_intr(&mut self) {
        self.inner.clear_intr();
    }

    /// Flush any JIT-translated code, after the host writes to guest code
    /// memory behind the CPU's back (CPU-performed stores are tracked
    /// automatically). A no-op when the extension was built without the JIT.
    fn invalidate_jit(&mut self) {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        self.inner.invalidate_jit();
    }

    /// EFLAGS as materialized by `PUSHFD` (bit 1 reads as 1; bits 3/5/15 and
    /// the VM/RF bits read as 0). Assigning loads a raw 32-bit value,
    /// discarding undefined bits.
    #[getter]
    fn get_eflags(&self) -> u32 {
        self.inner.regs.eflags.image32()
    }
    #[setter]
    fn set_eflags(&mut self, v: u32) {
        self.inner.regs.eflags.load(v, EFlags::DEFINED);
    }

    /// True once `CR0.PE` is set and the CPU is not in Virtual-8086 mode.
    #[getter]
    fn get_protected_mode(&self) -> bool {
        self.inner.protected_mode()
    }

    /// Current privilege level (0-3). Real mode reports 0, V86 reports 3.
    #[getter]
    fn get_cpl(&self) -> u8 {
        self.inner.cpl()
    }

    /// True while a HLT has the processor stopped (an interrupt resumes it).
    #[getter]
    fn get_halted(&self) -> bool {
        self.inner.halted
    }

    /// True after a triple fault; only `reset` recovers.
    #[getter]
    fn get_shutdown(&self) -> bool {
        self.inner.shutdown
    }

    /// Total cycles elapsed since construction (documented 386 base timings,
    /// instruction-atomic).
    #[getter]
    fn get_cycles(&self) -> u64 {
        self.inner.cycles
    }

    /// OS-emulation host hook: if set to a vector (e.g. `0x80`), `INT n` for
    /// that vector bypasses the IDT and raises a `"syscall"` host trap
    /// instead, with EIP already past the instruction. `None` (the default)
    /// disables the hook — `INT` behaves exactly like hardware.
    #[getter]
    fn get_syscall_int(&self) -> Option<u8> {
        self.inner.syscall_int
    }
    #[setter]
    fn set_syscall_int(&mut self, v: Option<u8>) {
        self.inner.syscall_int = v;
    }

    /// OS-emulation host hook: when True, CPU exceptions are handed back as
    /// `("exception", vector, error)` host traps (register state rewound to
    /// the faulting instruction) instead of vectoring through the IDT.
    /// Default False.
    #[getter]
    fn get_trap_faults(&self) -> bool {
        self.inner.trap_faults
    }
    #[setter]
    fn set_trap_faults(&mut self, v: bool) {
        self.inner.trap_faults = v;
    }

    /// OS-emulation host hook: when True, decode a handful of post-386
    /// opcodes (CPUID, RDTSC, CMPXCHG, XADD, BSWAP, CMOVcc, long NOP) that
    /// real 32-bit binaries use. Default False keeps strict 386 behavior
    /// (`#UD`).
    #[getter]
    fn get_extensions(&self) -> bool {
        self.inner.extensions
    }
    #[setter]
    fn set_extensions(&mut self, v: bool) {
        self.inner.extensions = v;
    }

    /// Peek at the pending host trap without clearing it: `None`,
    /// `"syscall"`, or `("exception", vector, error_code_or_None)`.
    #[getter]
    fn get_host_trap(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        host_trap_to_py(py, self.inner.host_trap)
    }

    /// Return the pending host trap and clear it, so `run` can proceed past
    /// the trap point: `None`, `"syscall"`, or
    /// `("exception", vector, error_code_or_None)`.
    fn take_host_trap(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let trap = self.inner.host_trap.take();
        host_trap_to_py(py, trap)
    }

    fn __repr__(&self) -> String {
        let r = &self.inner.regs;
        format!(
            "<remu.x86_32.Cpu eip=0x{:08X} eax=0x{:08X} {} cycles={}{}{}>",
            r.eip,
            r.gpr[reg::EAX as usize],
            if self.inner.protected_mode() {
                "prot"
            } else {
                "real"
            },
            self.inner.cycles,
            if self.inner.halted { " HALTED" } else { "" },
            if self.inner.shutdown { " SHUTDOWN" } else { "" },
        )
    }
}

gpr_props! {
    /// Accumulator.
    EAX: get_eax / set_eax,
    /// Count register.
    ECX: get_ecx / set_ecx,
    /// Data register.
    EDX: get_edx / set_edx,
    /// Base register.
    EBX: get_ebx / set_ebx,
    /// Stack pointer.
    ESP: get_esp / set_esp,
    /// Base pointer.
    EBP: get_ebp / set_ebp,
    /// Source index.
    ESI: get_esi / set_esi,
    /// Destination index.
    EDI: get_edi / set_edi,
}

u32_props! {
    /// Instruction pointer.
    eip: get_eip / set_eip,
    /// Control register 0 (PE bit 0, PG bit 31, ...). Raw field write — no
    /// TLB flush, unlike a guest `MOV CR0` (irrelevant on a fresh CPU).
    cr0: get_cr0 / set_cr0,
    /// Control register 2: the page-fault linear address.
    cr2: get_cr2 / set_cr2,
    /// Control register 3: the page-directory base. Raw field write — no
    /// TLB flush, unlike a guest `MOV CR3` (irrelevant on a fresh CPU).
    cr3: get_cr3 / set_cr3,
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
    /// Extra segment: `(sel, base, limit, attrs)`; assign an int selector
    /// for real-mode semantics or a 4-tuple to load the cache raw.
    ES: get_es / set_es,
    /// Code segment: `(sel, base, limit, attrs)`; assign an int selector
    /// for real-mode semantics or a 4-tuple to load the cache raw.
    CS: get_cs / set_cs,
    /// Stack segment: `(sel, base, limit, attrs)`; assign an int selector
    /// for real-mode semantics or a 4-tuple to load the cache raw.
    SS: get_ss / set_ss,
    /// Data segment: `(sel, base, limit, attrs)`; assign an int selector
    /// for real-mode semantics or a 4-tuple to load the cache raw.
    DS: get_ds / set_ds,
    /// F segment: `(sel, base, limit, attrs)`; assign an int selector
    /// for real-mode semantics or a 4-tuple to load the cache raw.
    FS: get_fs / set_fs,
    /// G segment: `(sel, base, limit, attrs)`; assign an int selector
    /// for real-mode semantics or a 4-tuple to load the cache raw.
    GS: get_gs / set_gs,
}

dtr_props! {
    /// Global descriptor table register: `(base, limit)`.
    gdtr: get_gdtr / set_gdtr,
    /// Interrupt descriptor table register: `(base, limit)`. At reset it
    /// covers the real-mode IVT (base 0, limit 0x3FF).
    idtr: get_idtr / set_idtr,
}
