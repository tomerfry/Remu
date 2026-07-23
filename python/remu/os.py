"""qiling-style Linux i386 OS emulation (hardware paging, ring 3).

``Emulator(path, argv=None, envp=None, rootfs=None)`` loads a Linux i386 ELF
from a path and runs it on the 80386 core with real page tables.
``run(max_instructions=None)`` returns the guest exit code, or ``None`` if
the instruction budget ran out first (call ``run`` again to resume).
"""

from remu._remu import Emulator386 as Emulator

__all__ = ["Emulator"]
