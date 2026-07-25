//! Shared plumbing for the per-core Python bindings: `mem[...]` indexing over
//! a raw RAM slice, and the forwarding helpers behind every duck-typed
//! callback bus.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PySlice};

/// Normalize an integer index into a `space`-byte address space, with Python
/// list semantics (negative indices count from the end; out-of-range is an
/// `IndexError`, non-ints a `TypeError`).
pub fn normalize_index(index: &Bound<'_, PyAny>, space: usize) -> PyResult<usize> {
    if !index.is_instance_of::<pyo3::types::PyInt>() {
        return Err(PyTypeError::new_err(
            "memory indices must be integers or slices",
        ));
    }
    let oob = || PyIndexError::new_err(format!("address out of range (0..=0x{:X})", space - 1));
    // An int that doesn't fit isize is out of range like any other.
    let i: isize = index.extract().map_err(|_| oob())?;
    let i = if i < 0 { i + space as isize } else { i };
    if !(0..space as isize).contains(&i) {
        return Err(oob());
    }
    Ok(i as usize)
}

/// `mem[i]` / `mem[start:stop:step]` over a RAM slice (slices return `bytes`).
pub fn ram_getitem(py: Python<'_>, ram: &[u8], index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(slice) = index.cast::<PySlice>() {
        let idx = slice.indices(ram.len() as isize)?;
        let mut out = Vec::with_capacity(idx.slicelength);
        let mut i = idx.start;
        for _ in 0..idx.slicelength {
            out.push(ram[i as usize]);
            i += idx.step;
        }
        PyBytes::new(py, &out).into_py_any(py)
    } else {
        ram[normalize_index(index, ram.len())?].into_py_any(py)
    }
}

/// `mem[i] = v` / `mem[start:stop:step] = data` over a RAM slice.
pub fn ram_setitem(
    ram: &mut [u8],
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if let Ok(slice) = index.cast::<PySlice>() {
        let idx = slice.indices(ram.len() as isize)?;
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
            ram[i as usize] = b;
            i += idx.step;
        }
        Ok(())
    } else {
        let i = normalize_index(index, ram.len())?;
        ram[i] = value.extract()?;
        Ok(())
    }
}

// --- Callback-bus forwarding --------------------------------------------------
//
// Every core accepts, anywhere a bus is expected, either its native `Memory`
// (fast, stays in Rust) or any Python object with `read(addr)` /
// `write(addr, value)` methods (plus optional `io_read(port)` /
// `io_write(port, value)` on the x86 cores). The first exception raised by a
// callback is stashed (the cores' buses are infallible, so the access reads
// as 0 / the write is dropped) and re-raised once control returns to Python.

/// Forward a byte read to `obj.read(addr)`, stashing the first error.
pub fn cb_read(obj: &Bound<'_, PyAny>, error: &mut Option<PyErr>, addr: u64) -> u8 {
    if error.is_some() {
        return 0;
    }
    let read = intern!(obj.py(), "read");
    match obj
        .call_method1(read, (addr,))
        .and_then(|v| v.extract::<u8>())
    {
        Ok(v) => v,
        Err(e) => {
            *error = Some(e);
            0
        }
    }
}

/// Forward a byte write to `obj.write(addr, value)`, stashing the first error.
pub fn cb_write(obj: &Bound<'_, PyAny>, error: &mut Option<PyErr>, addr: u64, value: u8) {
    if error.is_some() {
        return;
    }
    let write = intern!(obj.py(), "write");
    if let Err(e) = obj.call_method1(write, (addr, value)) {
        *error = Some(e);
    }
}

/// Forward a port read to `obj.io_read(port)` when the object has one
/// (`has`), else float the open bus (`0xFF`, matching the `Bus` default).
pub fn cb_io_read(obj: &Bound<'_, PyAny>, has: bool, error: &mut Option<PyErr>, port: u16) -> u8 {
    if !has || error.is_some() {
        return 0xFF;
    }
    let io_read = intern!(obj.py(), "io_read");
    match obj
        .call_method1(io_read, (port,))
        .and_then(|v| v.extract::<u8>())
    {
        Ok(v) => v,
        Err(e) => {
            *error = Some(e);
            0xFF
        }
    }
}

/// Forward a port write to `obj.io_write(port, value)` when present, else
/// drop it (matching the `Bus` default).
pub fn cb_io_write(
    obj: &Bound<'_, PyAny>,
    has: bool,
    error: &mut Option<PyErr>,
    port: u16,
    value: u8,
) {
    if !has || error.is_some() {
        return;
    }
    let io_write = intern!(obj.py(), "io_write");
    if let Err(e) = obj.call_method1(io_write, (port, value)) {
        *error = Some(e);
    }
}

/// True if `obj` quacks like a bus (has `read` and `write` attributes).
pub fn is_duck_bus(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    Ok(obj.hasattr(intern!(py, "read"))? && obj.hasattr(intern!(py, "write"))?)
}

/// True if `obj` has attribute `name` (for optional `io_read`/`io_write`).
pub fn has_attr(obj: &Bound<'_, PyAny>, name: &str) -> bool {
    obj.hasattr(name).unwrap_or(false)
}

/// The `TypeError` raised when a bus argument is neither a native memory nor
/// a duck-typed object.
pub fn bus_type_error(kind: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "bus must be a {kind} or an object with read(addr) and write(addr, value) methods",
    ))
}
