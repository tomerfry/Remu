//! Python bindings for the Linux OS-emulation layers (`remu.usermode`,
//! `remu.os`, `remu.os64`).
//!
//! Unlike the CPU bindings, each class here wraps a *whole* emulator — CPU,
//! guest memory and host-side OS state — because the layers own their address
//! spaces; there is no bus argument anywhere.
//!
//! ```python
//! from remu.usermode import Usermode
//!
//! um = Usermode(open("hello", "rb").read(), argv=["hello", "world"])
//! um.capture_fd(1)          # keep guest stdout off the real console
//! code = um.run()           # Ctrl-C friendly; RuntimeError on a guest fault
//! print(code, um.fd_data(1))
//! ```

use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::usermode::{Exit, UserArch, Usermode};
use crate::x86_32::EFlags;
use crate::x86_64::RFlags;

// --- Shared helpers -----------------------------------------------------------

/// Signal-poll cadence for the user-mode step loop (instructions). Rare enough
/// to cost nothing, frequent enough that Ctrl-C feels instant.
const STEP_CHUNK: u64 = 0x10000;
/// Signal-poll cadence for the OS layers' batched `run_capped` loops.
const RUN_CHUNK: u64 = 0x10_0000;

/// GPR index (instruction-encoding order) for an i386 register name.
fn gpr32_index(name: &str) -> Option<usize> {
    Some(match name {
        "eax" => 0,
        "ecx" => 1,
        "edx" => 2,
        "ebx" => 3,
        "esp" => 4,
        "ebp" => 5,
        "esi" => 6,
        "edi" => 7,
        _ => return None,
    })
}

/// GPR index (instruction-encoding order) for an x86-64 register name.
fn gpr64_index(name: &str) -> Option<usize> {
    Some(match name {
        "rax" => 0,
        "rcx" => 1,
        "rdx" => 2,
        "rbx" => 3,
        "rsp" => 4,
        "rbp" => 5,
        "rsi" => 6,
        "rdi" => 7,
        "r8" => 8,
        "r9" => 9,
        "r10" => 10,
        "r11" => 11,
        "r12" => 12,
        "r13" => 13,
        "r14" => 14,
        "r15" => 15,
        _ => return None,
    })
}

/// The `ValueError` for an unrecognized register name.
fn unknown_reg(name: &str) -> PyErr {
    PyValueError::new_err(format!("unknown register {name:?}"))
}

/// Check that a Python-supplied register value fits a 32-bit register.
fn fit32(value: u64) -> PyResult<u32> {
    u32::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("value {value:#x} does not fit in a 32-bit register"))
    })
}

/// The `ValueError` for a guest access that touched unmapped memory.
fn unmapped(addr: u64, len: usize) -> PyErr {
    PyValueError::new_err(format!(
        "unmapped guest memory in [{:#x}, {:#x})",
        addr,
        addr + len as u64
    ))
}

/// Read `length` guest bytes via `read_at(addr, buf) -> mapped?`, in 64 KiB
/// chunks so the output grows only as reads succeed. An absurd `length` over
/// unmapped memory thus raises `ValueError` at the first unmapped chunk
/// instead of first attempting one huge up-front allocation (whose failure
/// would abort the whole process, not raise).
fn read_guest(
    addr: u64,
    length: usize,
    mut read_at: impl FnMut(u64, &mut [u8]) -> bool,
) -> PyResult<Vec<u8>> {
    const CHUNK: usize = 0x10000;
    let mut out = Vec::new();
    let mut done = 0;
    while done < length {
        let n = (length - done).min(CHUNK);
        let start = out.len();
        out.resize(start + n, 0);
        let a = addr.wrapping_add(done as u64);
        if !read_at(a, &mut out[start..]) {
            return Err(unmapped(a, n));
        }
        done += n;
    }
    Ok(out)
}

// --- Usermode (qemu-user style, Linux i386) -----------------------------------

/// How a finished process ended, kept so later `run`/`step`/`__repr__` calls
/// can replay the outcome instead of stepping a dead guest.
enum Finished {
    Exited(i32),
    Fault(String),
}

/// A qemu-user style emulated Linux i386 process: a statically linked ELF
/// executable loaded from bytes, run on the 80386 core with its syscalls
/// serviced on the host.
///
/// The image is fully prepared at construction (segments mapped, brk placed,
/// argv/envp/auxv stack built, CPU in flat user mode at the entry point);
/// `run()` executes it to completion.
#[pyclass(name = "Usermode", module = "remu._remu")]
pub struct PyUsermode {
    inner: Usermode<crate::x86_32::Cpu>,
    finished: Option<Finished>,
}

impl PyUsermode {
    /// Record the process outcome and convert it to the Python result:
    /// exit code, or `RuntimeError` for a fault.
    fn finish(&mut self, exit: Exit) -> PyResult<i32> {
        match exit {
            Exit::Exited(code) => {
                self.finished = Some(Finished::Exited(code));
                Ok(code)
            }
            Exit::Fault(msg) => {
                self.finished = Some(Finished::Fault(msg.clone()));
                Err(PyRuntimeError::new_err(msg))
            }
        }
    }

    /// Replay a previously recorded outcome, if any.
    fn replay(&self) -> Option<PyResult<i32>> {
        match self.finished.as_ref()? {
            Finished::Exited(code) => Some(Ok(*code)),
            Finished::Fault(msg) => Some(Err(PyRuntimeError::new_err(msg.clone()))),
        }
    }
}

#[pymethods]
impl PyUsermode {
    /// Load a statically linked Linux i386 ELF executable from `program`
    /// (bytes) and prepare it to run. `argv` defaults to `["a.out"]`, `envp`
    /// to `[]`. Raises `ValueError` if the image cannot be loaded.
    #[new]
    #[pyo3(signature = (program, argv=None, envp=None))]
    fn new(
        program: Vec<u8>,
        argv: Option<Vec<String>>,
        envp: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let argv = argv.unwrap_or_else(|| vec!["a.out".to_string()]);
        let envp = envp.unwrap_or_default();
        let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
        let envp: Vec<&str> = envp.iter().map(String::as_str).collect();
        let inner = Usermode::load(&program, &argv, &envp)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyUsermode {
            inner,
            finished: None,
        })
    }

    /// Run the process to completion and return its exit code. A guest fault
    /// (unmapped access, unhandled CPU exception) raises `RuntimeError` whose
    /// message includes a register dump. The hot loop runs in Rust; every
    /// 64 Ki instructions it polls for signals (Ctrl-C raises
    /// `KeyboardInterrupt`) and briefly releases the GIL. Calling `run` again
    /// after the process ended replays the recorded outcome.
    fn run(&mut self, py: Python<'_>) -> PyResult<i32> {
        if let Some(done) = self.replay() {
            return done;
        }
        loop {
            for _ in 0..STEP_CHUNK {
                if let Some(exit) = self.inner.step_one() {
                    return self.finish(exit);
                }
            }
            py.check_signals()?;
            py.detach(|| {});
        }
    }

    /// Advance one instruction (servicing any syscall it raised). Returns
    /// `None` while the process is running, the exit code once it has exited,
    /// and raises `RuntimeError` on a guest fault.
    fn step(&mut self) -> PyResult<Option<i32>> {
        if let Some(done) = self.replay() {
            return done.map(Some);
        }
        match self.inner.step_one() {
            None => Ok(None),
            Some(exit) => self.finish(exit).map(Some),
        }
    }

    /// Log one line per syscall to stderr.
    #[getter]
    fn get_strace(&self) -> bool {
        self.inner.process.strace
    }
    #[setter]
    fn set_strace(&mut self, v: bool) {
        self.inner.process.strace = v;
    }

    /// Replace `fd` with an in-memory sink: guest writes to it are captured
    /// (readable via `fd_data`) instead of reaching the host. Call before
    /// running, e.g. `capture_fd(1)` to capture stdout.
    fn capture_fd(&mut self, fd: u32) {
        self.inner
            .process
            .fds
            .install(fd, crate::usermode::linux::fs::Fd::Sink(Vec::new()));
    }

    /// The bytes captured by the sink at `fd`, or `None` if `fd` is not a
    /// sink (see `capture_fd`).
    fn fd_data<'py>(&self, py: Python<'py>, fd: u32) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .process
            .fds
            .sink_data(fd)
            .map(|d| PyBytes::new(py, d))
    }

    /// Read `length` bytes of guest virtual memory at `addr`. Raises
    /// `ValueError` if any byte falls on unmapped memory.
    fn read<'py>(
        &mut self,
        py: Python<'py>,
        addr: u32,
        length: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mem = &mut self.inner.mem;
        let buf = read_guest(addr as u64, length, |a, b| {
            mem.read_bytes(a as u32, b).is_ok()
        })?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Write `data` into guest virtual memory at `addr`. Raises `ValueError`
    /// if any byte falls on unmapped memory (a prefix may have been written).
    fn write(&mut self, addr: u32, data: Vec<u8>) -> PyResult<()> {
        self.inner
            .mem
            .write_bytes(addr, &data)
            .map_err(|_| unmapped(addr as u64, data.len()))
    }

    /// Read a NUL-terminated string at `addr` (without the NUL), bounded by
    /// `max` bytes — an unterminated string returns the first `max` bytes.
    /// Raises `ValueError` on unmapped memory.
    #[pyo3(signature = (addr, max=4096))]
    fn read_cstr<'py>(
        &mut self,
        py: Python<'py>,
        addr: u32,
        max: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut out = Vec::new();
        for i in 0..max as u32 {
            let mut b = [0u8; 1];
            let a = addr.wrapping_add(i);
            self.inner
                .mem
                .read_bytes(a, &mut b)
                .map_err(|_| unmapped(a as u64, 1))?;
            if b[0] == 0 {
                break;
            }
            out.push(b[0]);
        }
        Ok(PyBytes::new(py, &out))
    }

    /// Read a register by name: `eax ecx edx ebx esp ebp esi edi eip eflags
    /// cr2` (case-insensitive). Raises `ValueError` on an unknown name.
    fn reg(&self, name: &str) -> PyResult<u64> {
        let name = name.to_ascii_lowercase();
        let r = &self.inner.cpu.regs;
        Ok(match name.as_str() {
            "eip" => r.eip as u64,
            "eflags" => r.eflags.image32() as u64,
            "cr2" => r.cr2 as u64,
            n => match gpr32_index(n) {
                Some(i) => r.gpr[i] as u64,
                None => return Err(unknown_reg(&name)),
            },
        })
    }

    /// Write a register by name (see `reg`). Raises `ValueError` on an
    /// unknown name or a value that does not fit in 32 bits.
    fn set_reg(&mut self, name: &str, value: u64) -> PyResult<()> {
        let name = name.to_ascii_lowercase();
        let v = fit32(value)?;
        let r = &mut self.inner.cpu.regs;
        match name.as_str() {
            "eip" => r.eip = v,
            "eflags" => r.eflags = EFlags::from_bits_truncate(v),
            "cr2" => r.cr2 = v,
            n => match gpr32_index(n) {
                Some(i) => r.gpr[i] = v,
                None => return Err(unknown_reg(&name)),
            },
        }
        Ok(())
    }

    /// The current program counter (EIP).
    #[getter]
    fn pc(&self) -> u32 {
        self.inner.cpu.regs.eip
    }

    /// Multi-line register dump (the same text fault reports include).
    fn dump(&self) -> String {
        UserArch::dump(&self.inner.cpu)
    }

    fn __repr__(&self) -> String {
        let state = match &self.finished {
            None => "running".to_string(),
            Some(Finished::Exited(code)) => format!("exited({code})"),
            Some(Finished::Fault(_)) => "faulted".to_string(),
        };
        format!(
            "<remu.usermode.Usermode eip={:#010x} {state}>",
            self.inner.cpu.regs.eip
        )
    }
}

// --- Emulator386 (qiling style, Linux i386) -----------------------------------

/// A qiling-style emulated Linux i386 process: an ELF executable loaded from
/// a host path, run on the 80386 core under real hardware paging in ring 3.
///
/// Dynamically linked binaries need a `rootfs` directory providing the
/// interpreter (`ld-linux.so.2`); guest filesystem access is sandboxed to it.
#[pyclass(name = "Emulator386", module = "remu._remu")]
pub struct PyEmulator386 {
    inner: crate::os::Emulator,
    /// Whether `capture_fd` enabled the layer's stdout/stderr capture.
    capturing: bool,
}

#[pymethods]
impl PyEmulator386 {
    /// Load a Linux i386 ELF executable from `path` and prepare it to run.
    /// An empty/omitted `argv` makes the layer substitute `["/a.out"]`.
    /// Raises `ValueError` if the image cannot be loaded.
    #[new]
    #[pyo3(signature = (path, argv=None, envp=None, rootfs=None))]
    fn new(
        path: PathBuf,
        argv: Option<Vec<String>>,
        envp: Option<Vec<String>>,
        rootfs: Option<PathBuf>,
    ) -> PyResult<Self> {
        let argv = argv.unwrap_or_default();
        let envp = envp.unwrap_or_default();
        let inner = crate::os::Emulator::load(&path, &argv, &envp, rootfs)
            .map_err(PyValueError::new_err)?;
        Ok(PyEmulator386 {
            inner,
            capturing: false,
        })
    }

    /// Run the process. With `max_instructions=None`, run to completion and
    /// return the exit code. With a budget, run at most about that many
    /// step-units (the budget is approximate — syscall servicing counts
    /// toward it): returns the exit code if the process finished, or `None`
    /// if the budget ran out first — the process stays resumable, call `run`
    /// again to continue. Every ~1 Mi instructions the loop polls for signals
    /// (Ctrl-C raises `KeyboardInterrupt`) and briefly releases the GIL.
    #[pyo3(signature = (max_instructions=None))]
    fn run(&mut self, py: Python<'_>, max_instructions: Option<u64>) -> PyResult<Option<i32>> {
        let mut budget = max_instructions.unwrap_or(u64::MAX);
        while self.inner.running {
            if budget == 0 {
                return Ok(None);
            }
            let n = budget.min(RUN_CHUNK);
            let code = self.inner.run_capped(n);
            if max_instructions.is_some() {
                budget -= n;
            }
            if !self.inner.running {
                return Ok(Some(code));
            }
            py.check_signals()?;
            py.detach(|| {});
        }
        Ok(Some(self.inner.exit_code))
    }

    /// Exit status (low byte of `exit`, or 128+signal on a fatal fault).
    /// Meaningful once `running` is `False`.
    #[getter]
    fn get_exit_code(&self) -> i32 {
        self.inner.exit_code
    }

    /// Whether the process is still running (cleared by `exit` or a fatal
    /// fault).
    #[getter]
    fn get_running(&self) -> bool {
        self.inner.running
    }

    /// Print diagnostics for unimplemented syscalls and fatal faults.
    #[getter]
    fn get_trace(&self) -> bool {
        self.inner.trace
    }
    #[setter]
    fn set_trace(&mut self, v: bool) {
        self.inner.trace = v;
    }

    /// Capture guest output in memory instead of the host console. This layer
    /// captures stdout and stderr *jointly* into one buffer, so `fd` must be
    /// 1 or 2 (raises `ValueError` otherwise) and both streams land in the
    /// same capture. Call before running.
    fn capture_fd(&mut self, fd: u32) -> PyResult<()> {
        if fd != 1 && fd != 2 {
            return Err(PyValueError::new_err(
                "this layer can only capture the standard streams (fd 1 or 2)",
            ));
        }
        self.inner.vfs.capture_output();
        self.capturing = true;
        Ok(())
    }

    /// The captured stdout+stderr bytes (see `capture_fd`), or `None` if
    /// capture is not enabled or `fd` is not 1 or 2.
    fn fd_data<'py>(&self, py: Python<'py>, fd: u32) -> Option<Bound<'py, PyBytes>> {
        if self.capturing && (fd == 1 || fd == 2) {
            Some(PyBytes::new(py, self.inner.vfs.captured()))
        } else {
            None
        }
    }

    /// Read `length` bytes of guest linear memory at `addr` (kernel
    /// privilege: page protection does not apply). Raises `ValueError` if any
    /// page in the range is unmapped.
    fn read<'py>(
        &self,
        py: Python<'py>,
        addr: u32,
        length: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let emu = &self.inner;
        let buf = read_guest(addr as u64, length, |a, b| {
            emu.aspace.read_bytes(&emu.mem, a as u32, b)
        })?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Write `data` into guest linear memory at `addr` (kernel privilege:
    /// read-only pages are writable from here). Raises `ValueError` if any
    /// page in the range is unmapped (a prefix may have been written).
    fn write(&mut self, addr: u32, data: Vec<u8>) -> PyResult<()> {
        let emu = &mut self.inner;
        if !emu.aspace.write_bytes(&mut emu.mem, addr, &data) {
            return Err(unmapped(addr as u64, data.len()));
        }
        Ok(())
    }

    /// Read a NUL-terminated string at `addr` (without the NUL), bounded by
    /// `max` bytes — an unterminated string returns the first `max` bytes.
    /// Raises `ValueError` on unmapped memory.
    #[pyo3(signature = (addr, max=4096))]
    fn read_cstr<'py>(
        &self,
        py: Python<'py>,
        addr: u32,
        max: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        match self.inner.aspace.read_cstr(&self.inner.mem, addr, max) {
            Some(s) => Ok(PyBytes::new(py, &s)),
            None => Err(unmapped(addr as u64, max)),
        }
    }

    /// Read a register by name: `eax ecx edx ebx esp ebp esi edi eip eflags
    /// cr2 cr3` (case-insensitive). Raises `ValueError` on an unknown name.
    fn reg(&self, name: &str) -> PyResult<u64> {
        let name = name.to_ascii_lowercase();
        let r = &self.inner.cpu.regs;
        Ok(match name.as_str() {
            "eip" => r.eip as u64,
            "eflags" => r.eflags.image32() as u64,
            "cr2" => r.cr2 as u64,
            "cr3" => r.cr3 as u64,
            n => match gpr32_index(n) {
                Some(i) => r.gpr[i] as u64,
                None => return Err(unknown_reg(&name)),
            },
        })
    }

    /// Write a register by name (see `reg`). Raises `ValueError` on an
    /// unknown name or a value that does not fit in 32 bits.
    fn set_reg(&mut self, name: &str, value: u64) -> PyResult<()> {
        let name = name.to_ascii_lowercase();
        let v = fit32(value)?;
        let r = &mut self.inner.cpu.regs;
        match name.as_str() {
            "eip" => r.eip = v,
            "eflags" => r.eflags = EFlags::from_bits_truncate(v),
            "cr2" => r.cr2 = v,
            "cr3" => r.cr3 = v,
            n => match gpr32_index(n) {
                Some(i) => r.gpr[i] = v,
                None => return Err(unknown_reg(&name)),
            },
        }
        Ok(())
    }

    /// The current program counter (EIP).
    #[getter]
    fn pc(&self) -> u32 {
        self.inner.cpu.regs.eip
    }

    fn __repr__(&self) -> String {
        let state = if self.inner.running {
            "running".to_string()
        } else {
            format!("exited({})", self.inner.exit_code)
        };
        format!(
            "<remu.os.Emulator eip={:#010x} {state}>",
            self.inner.cpu.regs.eip
        )
    }
}

// --- Emulator64 (qiling style, Linux x86-64) ----------------------------------

/// A qiling-style emulated Linux x86-64 process: an ELF executable loaded
/// from bytes or a host path, run on the x86-64 core under long-mode 4-level
/// paging in ring 3.
///
/// Statically linked `ET_EXEC` images run today; dynamically linked binaries
/// need a `rootfs` providing `ld-linux-x86-64.so.2` (support in progress).
#[pyclass(name = "Emulator64", module = "remu._remu")]
pub struct PyEmulator64 {
    inner: crate::os64::Emulator,
}

#[pymethods]
impl PyEmulator64 {
    /// Load a Linux x86-64 ELF executable and prepare it to run. `program`
    /// is either the ELF image as bytes/bytearray or a path (str or
    /// os.PathLike). An empty/omitted `argv` makes the layer substitute
    /// `["/a.out"]`. Raises `ValueError` if the image cannot be loaded,
    /// `TypeError` if `program` is neither bytes nor a path.
    #[new]
    #[pyo3(signature = (program, argv=None, envp=None, rootfs=None))]
    fn new(
        program: &Bound<'_, PyAny>,
        argv: Option<Vec<String>>,
        envp: Option<Vec<String>>,
        rootfs: Option<PathBuf>,
    ) -> PyResult<Self> {
        let argv = argv.unwrap_or_default();
        let envp = envp.unwrap_or_default();
        // bytes first: a Python `str` extracts as PathBuf but not as Vec<u8>,
        // so this order routes bytes → load_image and str/PathLike → load.
        let inner = if let Ok(image) = program.extract::<Vec<u8>>() {
            crate::os64::Emulator::load_image(&image, &argv, &envp, rootfs)
        } else if let Ok(path) = program.extract::<PathBuf>() {
            crate::os64::Emulator::load(&path, &argv, &envp, rootfs)
        } else {
            return Err(PyTypeError::new_err(
                "program must be bytes (an ELF image) or a path (str / os.PathLike)",
            ));
        }
        .map_err(PyValueError::new_err)?;
        Ok(PyEmulator64 { inner })
    }

    /// Run the process. With `max_instructions=None`, run to completion and
    /// return the exit code. With a budget, run at most about that many
    /// step-units (the budget is approximate — syscall servicing counts
    /// toward it): returns the exit code if the process finished, or `None`
    /// if the budget ran out first — the process stays resumable, call `run`
    /// again to continue. Every ~1 Mi instructions the loop polls for signals
    /// (Ctrl-C raises `KeyboardInterrupt`) and briefly releases the GIL.
    #[pyo3(signature = (max_instructions=None))]
    fn run(&mut self, py: Python<'_>, max_instructions: Option<u64>) -> PyResult<Option<i32>> {
        let mut budget = max_instructions.unwrap_or(u64::MAX);
        while self.inner.running {
            if budget == 0 {
                return Ok(None);
            }
            let n = budget.min(RUN_CHUNK);
            let code = self.inner.run_capped(n);
            if max_instructions.is_some() {
                budget -= n;
            }
            if !self.inner.running {
                return Ok(Some(code));
            }
            py.check_signals()?;
            py.detach(|| {});
        }
        Ok(Some(self.inner.exit_code))
    }

    /// Exit status (low byte of `exit`, or 128+signal on a fatal fault).
    /// Meaningful once `running` is `False`.
    #[getter]
    fn get_exit_code(&self) -> i32 {
        self.inner.exit_code
    }

    /// Whether the process is still running (cleared by `exit` or a fatal
    /// fault).
    #[getter]
    fn get_running(&self) -> bool {
        self.inner.running
    }

    /// Print diagnostics for unimplemented syscalls and fatal faults.
    #[getter]
    fn get_trace(&self) -> bool {
        self.inner.trace
    }
    #[setter]
    fn set_trace(&mut self, v: bool) {
        self.inner.trace = v;
    }

    /// Replace `fd` with an in-memory sink: guest writes to it are captured
    /// (readable via `fd_data`) instead of reaching the host. Call before
    /// running, e.g. `capture_fd(1)` to capture stdout.
    fn capture_fd(&mut self, fd: u32) {
        self.inner
            .vfs
            .install(fd as usize, crate::os64::fs::Fd::Sink(Vec::new()));
    }

    /// The bytes captured by the sink at `fd`, or `None` if `fd` is not a
    /// sink (see `capture_fd`).
    fn fd_data<'py>(&self, py: Python<'py>, fd: u32) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .vfs
            .sink_data(fd as usize)
            .map(|d| PyBytes::new(py, d))
    }

    /// Read `length` bytes of guest linear memory at `addr` (kernel
    /// privilege: page protection does not apply). Raises `ValueError` if any
    /// page in the range is unmapped.
    fn read<'py>(
        &self,
        py: Python<'py>,
        addr: u64,
        length: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let emu = &self.inner;
        let buf = read_guest(addr, length, |a, b| {
            emu.aspace.read_bytes(&emu.mem, a, b)
        })?;
        Ok(PyBytes::new(py, &buf))
    }

    /// Write `data` into guest linear memory at `addr` (kernel privilege:
    /// read-only pages are writable from here). Raises `ValueError` if any
    /// page in the range is unmapped (a prefix may have been written).
    fn write(&mut self, addr: u64, data: Vec<u8>) -> PyResult<()> {
        let emu = &mut self.inner;
        if !emu.aspace.write_bytes(&mut emu.mem, addr, &data) {
            return Err(unmapped(addr, data.len()));
        }
        Ok(())
    }

    /// Read a NUL-terminated string at `addr` (without the NUL), bounded by
    /// `max` bytes — an unterminated string returns the first `max` bytes.
    /// Raises `ValueError` on unmapped memory.
    #[pyo3(signature = (addr, max=4096))]
    fn read_cstr<'py>(
        &self,
        py: Python<'py>,
        addr: u64,
        max: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        match self.inner.aspace.read_cstr(&self.inner.mem, addr, max) {
            Some(s) => Ok(PyBytes::new(py, &s)),
            None => Err(unmapped(addr, max)),
        }
    }

    /// Read a register by name: `rax rcx rdx rbx rsp rbp rsi rdi r8..r15 rip
    /// rflags cr2 cr3` (case-insensitive). Raises `ValueError` on an unknown
    /// name.
    fn reg(&self, name: &str) -> PyResult<u64> {
        let name = name.to_ascii_lowercase();
        let r = &self.inner.cpu.regs;
        Ok(match name.as_str() {
            "rip" => r.rip,
            "rflags" => r.rflags.image(),
            "cr2" => r.cr2,
            "cr3" => r.cr3,
            n => match gpr64_index(n) {
                Some(i) => r.gpr[i],
                None => return Err(unknown_reg(&name)),
            },
        })
    }

    /// Write a register by name (see `reg`). Raises `ValueError` on an
    /// unknown name.
    fn set_reg(&mut self, name: &str, value: u64) -> PyResult<()> {
        let name = name.to_ascii_lowercase();
        let r = &mut self.inner.cpu.regs;
        match name.as_str() {
            "rip" => r.rip = value,
            "rflags" => r.rflags = RFlags::from_bits_truncate(value as u32),
            "cr2" => r.cr2 = value,
            "cr3" => r.cr3 = value,
            n => match gpr64_index(n) {
                Some(i) => r.gpr[i] = value,
                None => return Err(unknown_reg(&name)),
            },
        }
        Ok(())
    }

    /// The current program counter (RIP).
    #[getter]
    fn pc(&self) -> u64 {
        self.inner.cpu.regs.rip
    }

    fn __repr__(&self) -> String {
        let state = if self.inner.running {
            "running".to_string()
        } else {
            format!("exited({})", self.inner.exit_code)
        };
        format!(
            "<remu.os64.Emulator rip={:#018x} {state}>",
            self.inner.cpu.regs.rip
        )
    }
}
