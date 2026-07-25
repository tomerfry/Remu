"""Remu — a fast CPU emulation framework in Rust.

One submodule per core, mirroring the Rust crate layout:

- :mod:`remu.mos6502` — MOS 6502 / 65xx
- :mod:`remu.x86` — Intel 8086/8088 (real mode)
- :mod:`remu.x86_32` — Intel 80386 (real + protected mode + paging)
- :mod:`remu.x86_64` — x86-64 / AMD64 (long mode, 4-level paging)
- :mod:`remu.arm32` — ARM7TDMI-class ARMv4T (ARM + Thumb)

and the Linux OS-emulation layers built on the x86 cores:

- :mod:`remu.usermode` — qemu-user style, Linux i386, statically linked ELF
- :mod:`remu.os` — qiling-style Linux i386 (hardware paging, ring 3)
- :mod:`remu.os64` — qiling-style Linux x86-64 (runs static ELF64)

Each core module exposes ``Cpu`` and ``Memory``; anywhere a bus is expected,
either the core's native ``Memory`` (fast, the whole call stays in Rust) or
any Python object with ``read(addr)`` / ``write(addr, value)`` methods (for
MMIO experiments) is accepted.

The 6502 classes stay re-exported at the top level (``remu.Cpu``,
``remu.Memory``, ``remu.disassemble``) for backward compatibility.

Builds compiled with the ``symbolic`` feature also carry
:mod:`remu.symbolic` — concolic execution on the 386, x86-64 and ARM32 cores.
"""

from remu._remu import (
    Cpu6502 as Cpu,
    Memory6502 as Memory,
    RunExit,
    __version__,
    disassemble6502 as disassemble,
)

from . import arm32, mos6502, os, os64, usermode, x86, x86_32, x86_64

__all__ = [
    "Cpu",
    "Memory",
    "RunExit",
    "__version__",
    "arm32",
    "disassemble",
    "mos6502",
    "os",
    "os64",
    "usermode",
    "x86",
    "x86_32",
    "x86_64",
]

# Only in builds compiled with the `symbolic` feature; the submodule itself
# raises ImportError otherwise.
try:
    from . import symbolic

    __all__.append("symbolic")
except ImportError:  # pragma: no cover - depends on the build's features
    pass
