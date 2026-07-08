"""Type stubs for the `remu` extension module (6502 emulation)."""

from typing import Iterable, Protocol, Tuple, Union, overload

__version__: str

class BusLike(Protocol):
    """Anything with byte read/write across the 16-bit address space.

    Every function below that takes a bus accepts either a `Memory` (fast,
    the whole call stays in Rust) or any Python object implementing this
    protocol (e.g. for memory-mapped IO). Exceptions raised inside `read`/
    `write` propagate out of the CPU call that triggered them, after the
    current instruction finishes against a bus that reads as 0 and drops
    writes — so CPU state reflects that partial execution.

    This protocol exists only in the type stubs for annotation purposes;
    it is not importable from `remu` at runtime.
    """

    def read(self, addr: int) -> int: ...
    def write(self, addr: int, value: int) -> None: ...

class Memory:
    """A flat 64 KiB RAM-backed address space.

    Supports indexing and slicing: `mem[0x0600]`, `mem[0x0600:0x0610]` (returns
    `bytes`), `mem[0x0600] = 0xA9`, `mem[0x0600:0x0602] = b"\\xA9\\x42"`, and
    `bytes(mem)` for a full dump.
    """

    def __init__(self) -> None: ...
    def load(self, addr: int, data: Union[bytes, bytearray, Iterable[int]]) -> None:
        """Load `data` starting at `addr` (wrapping at the top of memory)."""

    def set_reset_vector(self, addr: int) -> None:
        """Point the reset vector ($FFFC/$FFFD) at `addr`."""

    def read(self, addr: int) -> int: ...
    def write(self, addr: int, value: int) -> None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> bytes: ...
    @overload
    def __setitem__(self, index: int, value: int) -> None: ...
    @overload
    def __setitem__(
        self, index: slice, value: Union[bytes, bytearray, Iterable[int]]
    ) -> None: ...
    def __bytes__(self) -> bytes: ...

class Cpu:
    """A MOS 6502 processor.

    The CPU does not own its bus — pass one to `reset`/`step`/`run`, which
    keeps it one bus master among many, exactly like the Rust API.
    """

    a: int
    """Accumulator."""
    x: int
    """Index register X."""
    y: int
    """Index register Y."""
    sp: int
    """Stack pointer (low byte; the stack lives at $0100..=$01FF)."""
    pc: int
    """Program counter."""
    p: int
    """Raw status byte (writes force U set / B clear, like the hardware)."""
    carry: bool
    zero: bool
    interrupt_disable: bool
    decimal: bool
    overflow: bool
    negative: bool
    @property
    def cycles(self) -> int:
        """Total cycles elapsed since construction."""
    @property
    def halted(self) -> bool:
        """True once a KIL/jam opcode halted the CPU (cleared by `reset`)."""

    def __init__(self) -> None: ...
    def reset(self, bus: BusLike) -> None:
        """RESET: load `pc` from the reset vector, set `interrupt_disable`, sp=$FD."""

    def step(self, bus: BusLike) -> int:
        """Execute one instruction (or service a pending interrupt).

        Returns the cycles consumed (0 if the CPU is jammed).
        """

    def run(self, bus: BusLike, instructions: int) -> int:
        """Execute up to `instructions` instructions entirely in Rust.

        Stops early if the CPU jams or a bus callback raises (the aborted
        instruction is included in the count). Returns the number of
        instructions actually executed. Every 64 Ki instructions the loop
        polls signals (Ctrl-C raises KeyboardInterrupt) and briefly releases
        the GIL so other Python threads can run.
        """

    def set_irq(self, level: bool) -> None:
        """Set the IRQ line level (level-triggered)."""

    def set_nmi(self, level: bool) -> None:
        """Set the NMI line level (low->high latches an NMI)."""

    def trigger_nmi(self) -> None:
        """Directly latch a pending NMI."""

def disassemble(bus: BusLike, addr: int) -> Tuple[str, int]:
    """Disassemble the instruction at `addr`.

    Returns `(text, next_addr)` — e.g. `("0600  LDA #$42", 0x0602)`.
    """
