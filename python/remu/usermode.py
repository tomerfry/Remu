"""qemu-user style Linux i386 user-mode emulation.

``Usermode(program, argv=None, envp=None)`` loads a statically linked Linux
i386 ELF from bytes and emulates its syscalls on the host. ``run()`` returns
the guest exit code (raising ``RuntimeError`` with a register dump on a
guest fault); ``capture_fd(1)`` before running redirects guest stdout into
an in-memory buffer readable via ``fd_data(1)``.
"""

from remu._remu import Usermode

__all__ = ["Usermode"]
