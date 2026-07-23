"""Intel 8086/8088 real-mode core.

``Memory`` is a flat 1 MiB (20-bit) address space. ``Cpu`` powers on at
``FFFF:0000``; set ``cs``/``ip`` (and friends) directly, then ``step`` or
``run`` against a bus. Duck-typed buses may also provide ``io_read(port)`` /
``io_write(port, value)`` for the IN/OUT instructions.
"""

from remu._remu import Cpu8086 as Cpu, Memory8086 as Memory

__all__ = ["Cpu", "Memory"]
