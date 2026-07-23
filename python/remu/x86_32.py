"""Intel 80386 core: real mode, protected mode, and paging.

``Memory`` is a flat 16 MiB physical address space. ``Cpu`` powers on in real
mode at ``F000:FFF0``; use ``set_cs_ip`` for real-mode entry or
``enter_flat_protected`` for a flat 4 GiB ring-0 setup. ``run`` returns
``(executed, RunExit)``. The OS-emulation host hooks (``syscall_int``,
``trap_faults``, ``take_host_trap``) let Python code service guest syscalls.
"""

from remu._remu import Cpu386 as Cpu, Memory386 as Memory, RunExit

__all__ = ["Cpu", "Memory", "RunExit"]
