"""x86-64 / AMD64 core: long mode, 4-level paging, SYSCALL/MSRs.

``Memory(size=None)`` is a flat physical address space (16 MiB default,
power-of-two sizes). ``Cpu.setup_long_flat(bus, rip, rsp)`` bootstraps
64-bit long mode with flat segments and identity paging in one call. ``run``
returns ``(executed, RunExit)``; the host hooks (``trap_syscall``,
``syscall_int``, ``trap_faults``, ``take_host_trap``) support OS emulation.
"""

from remu._remu import CpuX64 as Cpu, MemoryX64 as Memory, RunExit

__all__ = ["Cpu", "Memory", "RunExit"]
