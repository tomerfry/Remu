"""MOS 6502 / 65xx core.

``Memory`` is a flat 64 KiB address space with a reset vector helper; ``Cpu``
is the cycle-accurate 6502 (also re-exported at the top level as ``remu.Cpu``
/ ``remu.Memory`` / ``remu.disassemble``).
"""

from remu._remu import (
    Cpu6502 as Cpu,
    Memory6502 as Memory,
    disassemble6502 as disassemble,
)

__all__ = ["Cpu", "Memory", "disassemble"]
