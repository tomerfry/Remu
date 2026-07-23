"""qiling-style Linux x86-64 OS emulation (long mode, 4-level paging).

``Emulator(program, argv=None, envp=None, rootfs=None)`` loads a statically
linked Linux x86-64 ELF from bytes or from a path and runs it on the x86-64
core. ``run(max_instructions=None)`` returns the guest exit code, or
``None`` if the instruction budget ran out first (call ``run`` again to
resume); ``capture_fd(1)`` / ``fd_data(1)`` capture guest stdout in memory.
"""

from remu._remu import Emulator64 as Emulator

__all__ = ["Emulator"]
