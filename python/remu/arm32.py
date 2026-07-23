"""ARM7TDMI-class ARMv4T core: ARM + Thumb, banked modes.

``Memory(size=None)`` is a flat physical address space (16 MiB default).
``Cpu`` powers on in Supervisor mode, ARM state, ``pc = 0``; assign ``pc``
(and ``thumb``) directly, then ``step`` or ``run``. ``mode`` switches
processor modes with proper register banking; ``trap_swi`` +
``take_host_trap`` expose SWI to Python for OS experiments.
"""

from remu._remu import CpuArm as Cpu, MemoryArm as Memory, RunExit

__all__ = ["Cpu", "Memory", "RunExit"]
