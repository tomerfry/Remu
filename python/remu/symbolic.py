"""Concolic (symbolic) execution over the 386, x86-64 and ARM32 cores.

The concrete interpreter stays the source of truth; a sparse bitvector *shadow*
rides alongside it, tracking only the registers, flags and memory bytes derived
from an input you marked symbolic. Control flow is always driven by the real
run, and the conditions of the branches it took are collected as path
constraints. An SMT solver then negates one of them to answer "what input would
have taken the *other* side?".

The overlay lives on the CPU objects themselves, as ``sym_``-prefixed methods
(available on :class:`remu.x86_32.Cpu`, :class:`remu.x86_64.Cpu` and
:class:`remu.arm32.Cpu`)::

    cpu.sym_init()                          # enable the overlay
    cpu.sym_symbolize_reg(0, "eax")         # mark a register symbolic
    cpu.sym_symbolize_mem(addr, 32, "n", v) # ...or some memory
    cpu.run(mem, 100)                       # run normally
    cpu.sym_constraint_count                # branches recorded on this path
    cpu.sym_smtlib(flip=0)                  # SMT-LIB 2 for the flipped branch
    cpu.sym_solve(flip=0)                   # -> {"eax": 0x12345678} or None

``sym_solve`` and :func:`find_input` need an SMT solver binary on ``PATH`` —
``z3`` by default, or set ``REMU_SMT_SOLVER`` to another. Check with
:func:`solver_available`; everything else works without one.

Note that while the overlay is enabled the core is forced onto its reference
interpreter path (no instruction cache, no JIT), so execution is markedly
slower than a plain concrete run. Enabling it is per-CPU and opt-in, so
non-symbolic emulation is unaffected.

Coverage is deliberately partial: instructions the overlay does not model yet
concretize their result rather than tracking it, which keeps answers *sound*
(never a wrong solution) but may make them incomplete (a reachable path can be
missed). Symbolic memory addresses are concretized with a pinning constraint.
"""

from remu._remu import __symbolic__

if not __symbolic__:  # pragma: no cover - depends on how the wheel was built
    raise ImportError(
        "this build of remu was compiled without the `symbolic` feature; "
        "rebuild with `maturin develop --features symbolic-solver`"
    )

from remu._remu import sym_solver_available as solver_available

try:
    from remu._remu import sym_find_input as find_input
except ImportError:  # pragma: no cover - built with `symbolic` but no solver

    def find_input(harness, max_iters=50):
        """Unavailable: this build lacks the solver driver."""
        raise RuntimeError(
            "this build has the symbolic overlay but no solver driver; "
            "rebuild with `maturin develop --features symbolic-solver`"
        )

__all__ = ["find_input", "solver_available"]
