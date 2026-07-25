//! Python bindings for the concolic-execution overlay (`remu.symbolic`),
//! compiled only with the `symbolic` cargo feature.
//!
//! The overlay rides alongside the concrete interpreter on the 386, x86-64 and
//! ARM32 cores: mark a register or some memory symbolic, run normally, and the
//! engine collects the path constraints of the executed path. With an SMT
//! solver on `PATH` (feature `symbolic-solver`) those constraints can be solved
//! for an input that takes the *other* side of a branch.
//!
//! ```python
//! from remu.x86_32 import Cpu, Memory
//!
//! mem = Memory()
//! mem.load(0x1000, bytes([0x81, 0xF8, 0x78, 0x56, 0x34, 0x12,  # CMP EAX, 0x12345678
//!                         0x75, 0x01, 0xF4, 0xF4]))            # JNE +1; HLT; HLT
//! cpu = Cpu()
//! cpu.enter_flat_protected()
//! cpu.eip = 0x1000
//! cpu.sym_init()
//! cpu.sym_symbolize_reg(0, "eax")     # slot 0 = EAX
//! cpu.run(mem, 3)
//! cpu.sym_solve(flip=0)               # -> {'eax': 0x12345678}
//! ```
//!
//! Only scalars, strings and dicts cross the boundary — the bitvector AST stays
//! in Rust and is exported as SMT-LIB 2 text (`sym_smtlib`) when needed.

use std::collections::HashMap;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "symbolic-solver")]
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::symbolic::SymEngine;

/// Build a name-keyed model from an engine's seed (the engine stores it keyed
/// by the numeric symbol id).
fn seed_by_name(eng: &SymEngine) -> HashMap<String, u64> {
    eng.inputs()
        .iter()
        .filter_map(|i| eng.seed().get(&i.id).map(|v| (i.name.clone(), *v)))
        .collect()
}

/// The error raised when a symbolic entry point is used before `sym_init()`.
fn not_initialized() -> PyErr {
    PyRuntimeError::new_err("symbolic overlay is not enabled — call sym_init() first")
}

/// Generate the `#[pymethods]` block exposing the overlay on one core's CPU
/// class. The per-core Rust API differs only in the name of the
/// register-symbolizing method and the width of a memory address.
macro_rules! sym_methods {
    ($Py:ty, $symbolize_reg:ident, $addr:ty, $slots:expr) => {
        #[pymethods]
        impl $Py {
            /// Enable the concolic overlay (idempotent). Until this is called
            /// the core runs exactly as before, at full speed; while it is
            /// active the CPU is forced onto its reference interpreter path,
            /// so expect execution to be substantially slower.
            fn sym_init(&mut self) {
                self.inner.sym_init();
            }

            /// Whether the overlay is enabled.
            #[getter]
            fn sym_enabled(&self) -> bool {
                self.inner.sym_engine().is_some_and(|e| e.enabled)
            }

            /// Mark a full-width general register symbolic, seeded with its
            /// current concrete value, and return the input's numeric id.
            /// `slot` is the architectural register number.
            fn sym_symbolize_reg(&mut self, slot: u8, name: &str) -> PyResult<u32> {
                self.require_engine()?;
                if slot as usize >= $slots {
                    return Err(PyValueError::new_err(format!(
                        "register slot out of range (0..={})",
                        $slots - 1
                    )));
                }
                Ok(self.inner.$symbolize_reg(slot, name).0)
            }

            /// Mark `width / 8` bytes at linear address `addr` symbolic, seeded
            /// with `concrete` (little-endian), and return the input's id. The
            /// caller supplies the seed because the CPU does not own its bus —
            /// read it out of your `Memory` first.
            fn sym_symbolize_mem(
                &mut self,
                addr: $addr,
                width: u16,
                name: &str,
                concrete: u64,
            ) -> PyResult<u32> {
                self.require_engine()?;
                if !matches!(width, 8 | 16 | 32 | 64) {
                    return Err(PyValueError::new_err("width must be 8, 16, 32 or 64"));
                }
                Ok(self.inner.sym_symbolize_mem(addr, width, name, concrete).0)
            }

            /// How many path constraints the executed path has collected. Each
            /// index is a valid `flip` argument to `sym_smtlib` / `sym_solve`.
            #[getter]
            fn sym_constraint_count(&self) -> usize {
                self.inner.sym_constraints().len()
            }

            /// The declared symbolic inputs, as `(name, width_in_bits)`.
            fn sym_inputs(&self) -> Vec<(String, u16)> {
                match self.inner.sym_engine() {
                    Some(e) => e.inputs().iter().map(|i| (i.name.clone(), i.width)).collect(),
                    None => Vec::new(),
                }
            }

            /// The concolic seed: the concrete value standing in for each
            /// symbolic input on this run, keyed by input name.
            fn sym_seed(&self) -> HashMap<String, u64> {
                self.inner.sym_engine().map(seed_by_name).unwrap_or_default()
            }

            /// An SMT-LIB 2 (QF_BV) script for the current path, for feeding to
            /// any solver. With `flip=i`, branch `i` is negated (and later
            /// constraints dropped), so a satisfying model is an input that
            /// takes the *other* side of that branch.
            #[pyo3(signature = (flip = None))]
            fn sym_smtlib(&self, flip: Option<usize>) -> PyResult<String> {
                self.require_engine()?;
                self.check_flip(flip)?;
                Ok(match flip {
                    Some(i) => self.inner.sym_smtlib_flip(i),
                    None => self.inner.sym_smtlib(),
                })
            }

            /// Solve the current path's constraints, returning a model keyed by
            /// input name, or `None` if unsatisfiable. With `flip=i`, solve for
            /// an input that takes the other side of branch `i` — the
            /// "solve for the magic value" query.
            ///
            /// Requires an SMT solver binary (see
            /// `remu.symbolic.solver_available()`); the GIL is released while
            /// it runs.
            #[cfg(feature = "symbolic-solver")]
            #[pyo3(signature = (flip = None))]
            fn sym_solve(
                &self,
                py: Python<'_>,
                flip: Option<usize>,
            ) -> PyResult<Option<HashMap<String, u64>>> {
                self.require_engine()?;
                self.check_flip(flip)?;
                let cpu = &self.inner;
                Ok(py.detach(|| match flip {
                    Some(i) => cpu.sym_solve_flip(i),
                    None => cpu.sym_solve(),
                }))
            }

            /// Check the golden concolic invariant: every live symbolic shadow,
            /// evaluated under the seed, still equals the concrete machine
            /// state. Always true unless the overlay has a bug — useful as an
            /// assertion in tests.
            fn sym_check_invariant(&self) -> bool {
                self.inner.sym_check_invariant()
            }
        }

        impl $Py {
            /// Reject use before `sym_init()`. Checked before any argument, so
            /// the "you forgot to enable it" case never surfaces as the
            /// downstream complaint that the path has no branches.
            fn require_engine(&self) -> PyResult<()> {
                match self.inner.sym_engine() {
                    Some(_) => Ok(()),
                    None => Err(not_initialized()),
                }
            }

            /// Reject a `flip` index that names no collected branch, so the
            /// caller gets an `IndexError` instead of a silently empty script.
            fn check_flip(&self, flip: Option<usize>) -> PyResult<()> {
                let n = self.inner.sym_constraints().len();
                match flip {
                    Some(i) if i >= n => Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "no branch {i}: the path has {n} constraint(s)"
                    ))),
                    _ => Ok(()),
                }
            }
        }
    };
}

sym_methods!(super::x86_32::PyCpu386, sym_symbolize_reg32, u32, 8);
sym_methods!(super::x86_64::PyCpuX64, sym_symbolize_reg64, u64, 16);
sym_methods!(super::arm32::PyCpuArm, sym_symbolize_reg, u32, 16);

/// Take the overlay engine out of whichever core's CPU object the harness
/// returned. The engine is *moved* — the CPU is a throwaway built for one
/// exploration round.
#[cfg(feature = "symbolic-solver")]
fn take_engine(obj: &Bound<'_, PyAny>) -> PyResult<SymEngine> {
    macro_rules! try_core {
        ($t:ty) => {
            if let Ok(c) = obj.cast::<$t>() {
                return match c.try_borrow_mut()?.inner.sym.take() {
                    Some(e) => Ok(*e),
                    None => Err(PyRuntimeError::new_err(
                        "harness returned a CPU with no symbolic overlay — call sym_init() on it",
                    )),
                };
            }
        };
    }
    try_core!(super::x86_32::PyCpu386);
    try_core!(super::x86_64::PyCpuX64);
    try_core!(super::arm32::PyCpuArm);
    Err(PyTypeError::new_err(
        "harness must return (reached, cpu) where cpu is a remu 386, x86-64 or ARM32 Cpu",
    ))
}

/// One exploration round: hand the candidate input to the Python harness and
/// take back `(reached, engine)`.
#[cfg(feature = "symbolic-solver")]
fn call_harness(
    harness: &Bound<'_, PyAny>,
    inputs: &crate::symbolic::InputMap,
) -> PyResult<(bool, SymEngine)> {
    let py = harness.py();
    py.check_signals()?;
    let ret = harness.call1((inputs.clone().into_pyobject(py)?,))?;
    let (reached, cpu) = ret
        .extract::<(bool, Bound<'_, PyAny>)>()
        .map_err(|_| PyTypeError::new_err("harness must return a (reached, cpu) tuple"))?;
    Ok((reached, take_engine(&cpu)?))
}

/// Whether an SMT solver binary is launchable — `z3` by default, overridable
/// with the `REMU_SMT_SOLVER` environment variable. Everything except
/// `sym_solve` and `find_input` works without one.
#[pyfunction]
pub fn sym_solver_available() -> bool {
    #[cfg(feature = "symbolic-solver")]
    {
        crate::symbolic::Solver::new().available()
    }
    #[cfg(not(feature = "symbolic-solver"))]
    {
        false
    }
}

/// Search for an input that drives a program to a goal, by concolic
/// exploration (the generational search used by SAGE/Triton).
///
/// `harness(inputs)` is called once per candidate input. It must build a
/// **fresh** machine, apply `inputs` (a `{name: value}` dict; an absent name
/// keeps that input's default seed), symbolize the same inputs under the same
/// names, run the program, and return `(reached, cpu)` — whether the goal was
/// met, and the CPU it ran on. The overlay is taken from that CPU, so it must
/// have had `sym_init()` called on it.
///
/// Returns the first input dict that reaches the goal, or `None` after
/// `max_iters` rounds. Exceptions raised inside the harness propagate once the
/// search unwinds; Ctrl-C is honoured between rounds.
///
/// ```python
/// def harness(inputs):
///     cpu, mem = build_machine()
///     v = inputs.get("password", 0)
///     cpu.eax = v
///     cpu.sym_init()
///     cpu.sym_symbolize_reg(0, "password")
///     cpu.run(mem, 100)
///     return cpu.ebx == 0x600D, cpu
///
/// remu.symbolic.find_input(harness, max_iters=50)
/// ```
#[cfg(feature = "symbolic-solver")]
#[pyfunction]
#[pyo3(signature = (harness, max_iters = 50))]
pub fn sym_find_input(
    harness: &Bound<'_, PyAny>,
    max_iters: usize,
) -> PyResult<Option<HashMap<String, u64>>> {
    // `find_input` has no error channel, so a failing harness stashes its
    // exception and reports "not reached" with an inert engine (no
    // constraints ⇒ no children queued), which drains the search promptly.
    // The same stash-and-reraise contract the callback buses use.
    let mut error: Option<PyErr> = None;
    let mut closure = |inputs: &crate::symbolic::InputMap| -> (bool, SymEngine) {
        if error.is_some() {
            return (false, SymEngine::new(1, 32, 1));
        }
        match call_harness(harness, inputs) {
            Ok(v) => v,
            Err(e) => {
                error = Some(e);
                (false, SymEngine::new(1, 32, 1))
            }
        }
    };
    let found = crate::symbolic::find_input(max_iters, &mut closure);
    match error {
        Some(e) => Err(e),
        None => Ok(found),
    }
}
