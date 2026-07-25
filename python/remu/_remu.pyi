"""Type stubs for the `remu._remu` extension module.

The friendly import surface is the shim modules (`remu`, `remu.x86`,
`remu.arm32`, ...) which alias the flat names declared here.
"""

from pathlib import Path
from typing import Iterable, Optional, Protocol, Tuple, Union, overload

__version__: str

_Data = Union[bytes, bytearray, Iterable[int]]

class BusLike(Protocol):
    """Anything with byte read/write over the core's address space.

    Every CPU entry point that takes a bus accepts either the core's native
    `Memory` (fast, the whole call stays in Rust) or any Python object
    implementing this protocol (e.g. for memory-mapped IO). The x86 cores
    additionally forward IN/OUT to optional `io_read(port)` /
    `io_write(port, value)` methods when present. Exceptions raised inside
    callbacks propagate out of the CPU call that triggered them, after the
    current instruction finishes against a bus that reads as 0 and drops
    writes — CPU state reflects that partial execution.

    This protocol exists only in the type stubs for annotation purposes.
    """

    def read(self, addr: int) -> int: ...
    def write(self, addr: int, value: int) -> None: ...

class RunExit:
    """Why a batched `run(bus, n)` call returned (386 / x86-64 / ARM32)."""

    Completed: RunExit
    """The full instruction budget was executed."""
    HostTrap: RunExit
    """Stopped at a host trap (syscall or trapped fault) — see `take_host_trap()`."""
    Halted: RunExit
    """The CPU executed HLT (or, on ARM, `halted` was set by the embedder)."""
    Shutdown: RunExit
    """Triple fault — the machine shut down (x86 cores only; only `reset` recovers)."""

    def __eq__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...

# --- MOS 6502 (remu.mos6502, re-exported at the top level) --------------------

class Memory6502:
    """A flat 64 KiB RAM-backed address space.

    Supports indexing and slicing: `mem[0x0600]`, `mem[0x0600:0x0610]`
    (returns `bytes`), `mem[0x0600] = 0xA9`, and `bytes(mem)` for a full dump.
    """

    def __init__(self) -> None: ...
    def load(self, addr: int, data: _Data) -> None:
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
    def __setitem__(self, index: slice, value: _Data) -> None: ...
    def __bytes__(self) -> bytes: ...

class Cpu6502:
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
        """Execute one instruction; returns cycles consumed (0 if jammed)."""

    def run(self, bus: BusLike, instructions: int) -> int:
        """Execute up to `instructions` instructions entirely in Rust.

        Stops early if the CPU jams or a bus callback raises. Returns the
        number of instructions actually executed. Polls signals every 64 Ki
        instructions (Ctrl-C raises KeyboardInterrupt).
        """

    def set_irq(self, level: bool) -> None:
        """Set the IRQ line level (level-triggered)."""

    def set_nmi(self, level: bool) -> None:
        """Set the NMI line level (high→low latches an NMI)."""

    def trigger_nmi(self) -> None:
        """Directly latch a pending NMI."""

def disassemble6502(bus: BusLike, addr: int) -> Tuple[str, int]:
    """Disassemble the instruction at `addr` → `(text, next_addr)`."""

# --- Intel 8086/8088 (remu.x86) -----------------------------------------------

class Memory8086:
    """A flat 1 MiB (20-bit) RAM-backed address space with indexing/slicing."""

    def __init__(self) -> None: ...
    def load(self, addr: int, data: _Data) -> None:
        """Load `data` at physical `addr`, wrapping at 1 MiB."""

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
    def __setitem__(self, index: slice, value: _Data) -> None: ...
    def __bytes__(self) -> bytes: ...

class Cpu8086:
    """An Intel 8086/8088 (real mode). Powers on at CS:IP = FFFF:0000."""

    ax: int
    bx: int
    cx: int
    dx: int
    sp: int
    bp: int
    si: int
    di: int
    ip: int
    es: int
    cs: int
    ss: int
    ds: int
    al: int
    ah: int
    bl: int
    bh: int
    cl: int
    ch: int
    dl: int
    dh: int
    flags: int
    """Raw FLAGS word (reserved bits read as on the 8086)."""
    carry: bool
    parity: bool
    adjust: bool
    zero: bool
    sign: bool
    trap: bool
    interrupt: bool
    """IF — interrupt-enable flag."""
    direction: bool
    overflow: bool
    @property
    def cycles(self) -> int: ...
    @property
    def halted(self) -> bool:
        """True while a HLT has the processor stopped (an interrupt resumes it)."""

    def __init__(self) -> None: ...
    def reset(self) -> None:
        """Back to the power-on state (registers, halt latch, lines)."""

    def step(self, bus: BusLike) -> int:
        """Execute one instruction (or service a pending interrupt) → cycles."""

    def run(self, bus: BusLike, instructions: int) -> int:
        """Execute up to `instructions` instructions in Rust.

        Stops early on HLT (assert an interrupt and call `run` again to
        resume) or when a bus callback raises. Returns instructions executed.
        """

    def trigger_nmi(self) -> None:
        """Latch a non-maskable interrupt (vector 2)."""

    def assert_intr(self, vector: int) -> None:
        """Assert the maskable INTR line (serviced while IF is set)."""

    def clear_intr(self) -> None:
        """Deassert the INTR line."""

# --- Intel 80386 (remu.x86_32) ------------------------------------------------

_SegTuple = Tuple[int, int, int, int]
"""Segment register image: (selector, base, limit, attributes)."""

_HostTrap = Union[None, str, Tuple[str, int, Optional[int]]]
"""None, "syscall", or ("exception", vector, error_code_or_None)."""

class Memory386:
    """A flat 16 MiB physical address space with indexing/slicing."""

    def __init__(self) -> None: ...
    def load(self, addr: int, data: _Data) -> None:
        """Load `data` at physical `addr`, wrapping at 16 MiB."""

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
    def __setitem__(self, index: slice, value: _Data) -> None: ...
    def __bytes__(self) -> bytes: ...

class Cpu386:
    """An Intel 80386: real + protected mode, paging. Powers on at F000:FFF0."""

    eax: int
    ecx: int
    edx: int
    ebx: int
    esp: int
    ebp: int
    esi: int
    edi: int
    eip: int
    eflags: int
    """Materialized EFLAGS image (writes are masked to defined bits)."""
    carry: bool
    parity: bool
    adjust: bool
    zero: bool
    sign: bool
    trap: bool
    interrupt: bool
    direction: bool
    overflow: bool
    cr0: int
    cr2: int
    cr3: int
    es: _SegTuple
    """Segment registers read as (sel, base, limit, attrs); assign an int for
    real-mode semantics (base = sel*16) or a 4-tuple for a raw load."""
    cs: _SegTuple
    ss: _SegTuple
    ds: _SegTuple
    fs: _SegTuple
    gs: _SegTuple
    gdtr: Tuple[int, int]
    """Descriptor table register as (base, limit)."""
    idtr: Tuple[int, int]
    syscall_int: Optional[int]
    """INT vector trapped to the host as a syscall (OS-emulation hook)."""
    trap_faults: bool
    """Trap CPU exceptions to the host instead of the guest IDT."""
    extensions: bool
    """Enable the 486+/686 extension opcodes (CMPXCHG, CPUID, ...)."""
    @property
    def host_trap(self) -> _HostTrap:
        """Peek the pending host trap without clearing it."""
    @property
    def protected_mode(self) -> bool: ...
    @property
    def cpl(self) -> int:
        """Current privilege level (0-3)."""
    @property
    def halted(self) -> bool: ...
    @property
    def shutdown(self) -> bool:
        """True after a triple fault (only `reset` recovers)."""
    @property
    def cycles(self) -> int: ...

    def __init__(self) -> None: ...
    def reset(self) -> None: ...
    def set_cs_ip(self, sel: int, eip: int) -> None:
        """Real-mode entry helper: CS = sel (base = sel*16), EIP = eip."""

    def enter_flat_protected(self) -> None:
        """Enter ring-0 protected mode with flat 4 GiB segments (eip/esp kept)."""

    def step(self, bus: BusLike) -> int:
        """Execute one instruction → cycles consumed."""

    def run(self, bus: BusLike, instructions: int) -> Tuple[int, RunExit]:
        """Execute up to `instructions` in Rust → (executed, RunExit).

        Stops early on HLT, triple fault, a host trap, or a raising bus
        callback. Polls signals every 64 Ki instructions.
        """

    def trigger_nmi(self) -> None: ...
    def assert_intr(self, vector: int) -> None: ...
    def clear_intr(self) -> None: ...
    def invalidate_jit(self) -> None:
        """Flush translated code after writing to guest code memory (no-op
        without the `jit` build feature)."""

    def take_host_trap(self) -> _HostTrap:
        """Return and clear the pending host trap."""

# --- x86-64 / AMD64 (remu.x86_64) ---------------------------------------------

class MemoryX64:
    """A flat physical address space (16 MiB default, power-of-two sizes)."""

    def __init__(self, size: Optional[int] = None) -> None: ...
    def load(self, addr: int, data: _Data) -> None:
        """Load `data` at physical `addr`, wrapping at the memory size."""

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
    def __setitem__(self, index: slice, value: _Data) -> None: ...
    def __bytes__(self) -> bytes: ...

class CpuX64:
    """An x86-64 / AMD64 CPU: long mode, 4-level paging, SYSCALL/MSRs.

    Powers on in real mode at F000:FFF0; `setup_long_flat` bootstraps
    64-bit long mode in one call.
    """

    rax: int
    rcx: int
    rdx: int
    rbx: int
    rsp: int
    rbp: int
    rsi: int
    rdi: int
    r8: int
    r9: int
    r10: int
    r11: int
    r12: int
    r13: int
    r14: int
    r15: int
    rip: int
    rflags: int
    """Materialized RFLAGS image (writes are masked to defined bits)."""
    carry: bool
    parity: bool
    adjust: bool
    zero: bool
    sign: bool
    trap: bool
    interrupt: bool
    direction: bool
    overflow: bool
    cr0: int
    cr2: int
    cr3: int
    cr4: int
    cr8: int
    efer: int
    """EFER MSR (LME/LMA/SCE/NX)."""
    star: int
    lstar: int
    """LSTAR MSR — 64-bit SYSCALL entry point."""
    cstar: int
    sfmask: int
    kernel_gs_base: int
    es: _SegTuple
    cs: _SegTuple
    ss: _SegTuple
    ds: _SegTuple
    fs: _SegTuple
    gs: _SegTuple
    gdtr: Tuple[int, int]
    idtr: Tuple[int, int]
    syscall_int: Optional[int]
    """INT vector trapped to the host as a syscall (OS-emulation hook)."""
    trap_syscall: bool
    """Trap the SYSCALL instruction to the host (OS-emulation hook)."""
    trap_faults: bool
    @property
    def host_trap(self) -> _HostTrap:
        """Peek the pending host trap without clearing it."""
    @property
    def long_mode(self) -> bool: ...
    @property
    def mode64(self) -> bool:
        """True when executing 64-bit code (long mode + CS.L)."""
    @property
    def protected_mode(self) -> bool: ...
    @property
    def cpl(self) -> int: ...
    @property
    def halted(self) -> bool: ...
    @property
    def shutdown(self) -> bool: ...
    @property
    def cycles(self) -> int: ...

    def __init__(self) -> None: ...
    def reset(self) -> None: ...
    def set_cs_ip(self, sel: int, rip: int) -> None:
        """Real-mode entry helper: CS = sel (base = sel*16), RIP = rip."""

    def setup_long_flat(self, bus: BusLike, rip: int, rsp: int) -> None:
        """Enter 64-bit long mode with flat segments and identity paging
        (page tables are written at physical 0x1000/0x2000 through `bus`)."""

    def step(self, bus: BusLike) -> int:
        """Execute one instruction → cycles consumed."""

    def run(self, bus: BusLike, instructions: int) -> Tuple[int, RunExit]:
        """Execute up to `instructions` in Rust → (executed, RunExit)."""

    def trigger_nmi(self) -> None: ...
    def assert_intr(self, vector: int) -> None: ...
    def clear_intr(self) -> None: ...
    def invalidate_jit(self) -> None:
        """Flush translated code after writing to guest code memory (no-op
        without the `jit` build feature)."""

    def take_host_trap(self) -> _HostTrap:
        """Return and clear the pending host trap."""

# --- ARM7TDMI / ARMv4T (remu.arm32) -------------------------------------------

_ArmHostTrap = Union[None, str, Tuple[str, str]]
"""None, "syscall", or ("exception", kind) with kind one of "undefined",
"swi", "prefetch_abort", "data_abort", "irq", "fiq"."""

class MemoryArm:
    """A flat physical address space (16 MiB default, power-of-two sizes)."""

    def __init__(self, size: Optional[int] = None) -> None: ...
    def load(self, addr: int, data: _Data) -> None:
        """Load `data` at physical `addr`, wrapping at the memory size."""

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
    def __setitem__(self, index: slice, value: _Data) -> None: ...
    def __bytes__(self) -> bytes: ...

class CpuArm:
    """An ARM7TDMI-class ARMv4T CPU (ARM + Thumb, banked modes).

    Powers on in Supervisor mode, ARM state, pc = 0.
    """

    r0: int
    r1: int
    r2: int
    r3: int
    r4: int
    r5: int
    r6: int
    r7: int
    r8: int
    r9: int
    r10: int
    r11: int
    r12: int
    r13: int
    r14: int
    r15: int
    sp: int
    """Alias of r13 (current mode's bank)."""
    lr: int
    """Alias of r14 (current mode's bank)."""
    pc: int
    """Alias of r15 — the address of the NEXT instruction (no pipeline offset)."""
    cpsr: int
    """Raw CPSR bits (assigning a value with different mode bits switches
    register banks, like an MSR write)."""
    spsr: int
    """SPSR of the current mode (reads as CPSR in usr/sys)."""
    thumb: bool
    """Thumb state (CPSR.T)."""
    negative: bool
    zero: bool
    carry: bool
    overflow: bool
    irq_disable: bool
    fiq_disable: bool
    mode: str
    """Processor mode: "usr" "fiq" "irq" "svc" "abt" "und" "sys". Assigning
    switches modes with proper register banking."""
    halted: bool
    """Embedder-controlled halt (ARMv4T has no WFI); `run` returns Halted
    while set, and a delivered interrupt clears it."""
    trap_swi: bool
    """Trap SWI to the host instead of vectoring (OS-emulation hook)."""
    trap_faults: bool
    @property
    def host_trap(self) -> _ArmHostTrap:
        """Peek the pending host trap without clearing it."""
    @property
    def cycles(self) -> int: ...

    def __init__(self) -> None: ...
    def reset(self) -> None: ...
    def step(self, bus: BusLike) -> int:
        """Execute one instruction → cycles consumed."""

    def run(self, bus: BusLike, instructions: int) -> Tuple[int, RunExit]:
        """Execute up to `instructions` in Rust → (executed, RunExit)."""

    def set_irq(self, level: bool) -> None:
        """Set the IRQ line level (serviced while CPSR.I is clear)."""

    def set_fiq(self, level: bool) -> None:
        """Set the FIQ line level (FIQ outranks IRQ)."""

    def raise_exception(self, kind: str) -> None:
        """Enter an exception directly: "undefined" "swi" "prefetch_abort"
        "data_abort" "irq" "fiq"."""

    def take_host_trap(self) -> _ArmHostTrap:
        """Return and clear the pending host trap."""

# --- Linux OS-emulation layers (remu.usermode / remu.os / remu.os64) ----------

class Usermode:
    """qemu-user style Linux i386 emulation: statically linked ELF from bytes,
    syscalls emulated on the host."""

    strace: bool
    """Log one line per syscall to stderr."""
    @property
    def pc(self) -> int:
        """Current EIP."""

    def __init__(
        self,
        program: Union[bytes, bytearray],
        argv: Optional[list[str]] = None,
        envp: Optional[list[str]] = None,
    ) -> None: ...
    def run(self) -> int:
        """Run to completion → guest exit code.

        Raises RuntimeError (with a register dump) on a guest fault; polls
        signals so Ctrl-C works.
        """

    def step(self) -> Optional[int]:
        """One instruction (+ syscall service): None while running, the exit
        code once the process ends; RuntimeError on a fault."""

    def capture_fd(self, fd: int) -> None:
        """Redirect guest `fd` into an in-memory sink (call before running)."""

    def fd_data(self, fd: int) -> Optional[bytes]:
        """Data captured on a sink fd (None if `fd` isn't a sink)."""

    def read(self, addr: int, length: int) -> bytes:
        """Read guest virtual memory (ValueError on unmapped)."""

    def write(self, addr: int, data: _Data) -> None:
        """Write guest virtual memory (ValueError on unmapped)."""

    def read_cstr(self, addr: int, max: int = 4096) -> bytes:
        """Read a NUL-terminated string from guest memory."""

    def reg(self, name: str) -> int:
        """Read a register by name (eax..edi, eip, eflags, cr2)."""

    def set_reg(self, name: str, value: int) -> None: ...
    def dump(self) -> str:
        """Multi-line register dump."""

class Emulator386:
    """qiling-style Linux i386 OS emulation (hardware paging, ring 3);
    loads a statically linked ELF from a path."""

    trace: bool
    """Print unimplemented-syscall / fault diagnostics to stderr."""
    @property
    def exit_code(self) -> int: ...
    @property
    def running(self) -> bool: ...
    @property
    def pc(self) -> int:
        """Current EIP."""

    def __init__(
        self,
        path: Union[str, Path],
        argv: Optional[list[str]] = None,
        envp: Optional[list[str]] = None,
        rootfs: Optional[Union[str, Path]] = None,
    ) -> None: ...
    def run(self, max_instructions: Optional[int] = None) -> Optional[int]:
        """Run the guest → exit code, or None if `max_instructions` ran out
        first (still `running`; call `run` again to resume)."""

    def capture_fd(self, fd: int) -> None:
        """Capture guest output in memory. This layer captures stdout and
        stderr jointly, so `fd` must be 1 or 2 (both land in one buffer)."""

    def fd_data(self, fd: int) -> Optional[bytes]: ...
    def read(self, addr: int, length: int) -> bytes:
        """Read guest linear memory (ValueError on unmapped)."""

    def write(self, addr: int, data: _Data) -> None: ...
    def read_cstr(self, addr: int, max: int = 4096) -> bytes: ...
    def reg(self, name: str) -> int:
        """Read a register by name (eax..edi, eip, eflags, cr2, cr3)."""

    def set_reg(self, name: str, value: int) -> None: ...

class Emulator64:
    """qiling-style Linux x86-64 OS emulation (long mode, 4-level paging);
    loads a statically linked ELF64 from bytes or from a path."""

    trace: bool
    @property
    def exit_code(self) -> int: ...
    @property
    def running(self) -> bool: ...
    @property
    def pc(self) -> int:
        """Current RIP."""

    def __init__(
        self,
        program: Union[bytes, bytearray, str, Path],
        argv: Optional[list[str]] = None,
        envp: Optional[list[str]] = None,
        rootfs: Optional[Union[str, Path]] = None,
    ) -> None: ...
    def run(self, max_instructions: Optional[int] = None) -> Optional[int]:
        """Run the guest → exit code, or None if `max_instructions` ran out
        first (still `running`; call `run` again to resume)."""

    def capture_fd(self, fd: int) -> None: ...
    def fd_data(self, fd: int) -> Optional[bytes]: ...
    def read(self, addr: int, length: int) -> bytes: ...
    def write(self, addr: int, data: _Data) -> None: ...
    def read_cstr(self, addr: int, max: int = 4096) -> bytes: ...
    def reg(self, name: str) -> int:
        """Read a register by name (rax..r15, rip, rflags, cr2, cr3)."""

    def set_reg(self, name: str, value: int) -> None: ...
