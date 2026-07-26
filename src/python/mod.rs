//! Python bindings (PyO3), compiled only with the `python` cargo feature.
//!
//! The crate builds as the `remu._remu` extension module with
//! [maturin](https://www.maturin.rs) — see `pyproject.toml`. The thin shim
//! package in `python/remu/` re-exports it as one submodule per core,
//! mirroring the Rust layout:
//!
//! | Python module   | Rust module        | classes                       |
//! |-----------------|--------------------|-------------------------------|
//! | `remu.mos6502`  | [`crate::cpu`]     | `Cpu`, `Memory`, `disassemble`|
//! | `remu.x86`      | [`crate::x86`]     | `Cpu`, `Memory`               |
//! | `remu.x86_32`   | [`crate::x86_32`]  | `Cpu`, `Memory`, `RunExit`    |
//! | `remu.x86_64`   | [`crate::x86_64`]  | `Cpu`, `Memory`, `RunExit`    |
//! | `remu.arm32`    | [`crate::arm32`]   | `Cpu`, `Memory`, `RunExit`    |
//! | `remu.usermode` | [`crate::usermode`]| `Usermode`                    |
//! | `remu.os`       | [`crate::os`]      | `Emulator`                    |
//! | `remu.os64`     | [`crate::os64`]    | `Emulator`                    |
//!
//! The 6502 stays re-exported at the top level (`remu.Cpu`, `remu.Memory`,
//! `remu.disassemble`) for backward compatibility. Anywhere a bus is
//! expected, either the core's native `Memory` (fast, stays in Rust) or any
//! Python object with `read(addr)` / `write(addr, value)` methods (flexible,
//! for MMIO experiments) is accepted.

mod arm32;
mod mos6502;
mod oslayers;
#[cfg(feature = "symbolic")]
mod symbolic;
mod util;
mod x86;
mod x86_32;
mod x86_64;

use pyo3::prelude::*;

/// Why a batched `run(bus, n)` call returned, for the cores whose Rust `run`
/// reports an exit reason (386, x86-64, ARM32).
#[pyclass(
    name = "RunExit",
    module = "remu._remu",
    frozen,
    eq,
    eq_int,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyRunExit {
    /// The full instruction budget was executed.
    Completed,
    /// Stopped at a host trap (syscall or trapped fault) — see
    /// `cpu.take_host_trap()`.
    HostTrap,
    /// The CPU executed HLT (or, on ARM, `halted` was set by the embedder).
    Halted,
    /// Triple fault — the machine shut down (x86 cores only; only `reset`
    /// recovers).
    Shutdown,
}

#[pymodule]
fn _remu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // MOS 6502
    m.add_class::<mos6502::PyCpu6502>()?;
    m.add_class::<mos6502::PyMemory6502>()?;
    m.add_function(wrap_pyfunction!(mos6502::disassemble6502, m)?)?;
    // Intel 8086/8088
    m.add_class::<x86::PyCpu8086>()?;
    m.add_class::<x86::PyMemory8086>()?;
    // Intel 80386
    m.add_class::<x86_32::PyCpu386>()?;
    m.add_class::<x86_32::PyMemory386>()?;
    // x86-64 / AMD64
    m.add_class::<x86_64::PyCpuX64>()?;
    m.add_class::<x86_64::PyMemoryX64>()?;
    // ARM7TDMI / ARMv4T
    m.add_class::<arm32::PyCpuArm>()?;
    m.add_class::<arm32::PyMemoryArm>()?;
    // Linux OS-emulation layers
    m.add_class::<oslayers::PyUsermode>()?;
    m.add_class::<oslayers::PyEmulator386>()?;
    m.add_class::<oslayers::PyEmulator64>()?;
    // Shared
    m.add_class::<PyRunExit>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Concolic execution: the `sym_*` methods are attached to the 386, x86-64
    // and ARM32 CPU classes; these are the module-level entry points.
    #[cfg(feature = "symbolic")]
    {
        m.add_function(wrap_pyfunction!(symbolic::sym_solver_available, m)?)?;
        #[cfg(feature = "symbolic-solver")]
        m.add_function(wrap_pyfunction!(symbolic::sym_find_input, m)?)?;
    }
    m.add("__symbolic__", cfg!(feature = "symbolic"))?;
    Ok(())
}
