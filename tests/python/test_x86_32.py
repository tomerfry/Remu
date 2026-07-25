"""Tests for the `remu.x86_32` (Intel 80386) bindings.

Run with the extension installed (`maturin develop`): `pytest tests/python`.
"""

import pytest

from remu.x86_32 import Cpu, Memory, RunExit

CODE = 0x1000

# MOV AX, 0x1234 ; HLT (16-bit real mode)
REAL_PROGRAM = b"\xB8\x34\x12\xF4"
# MOV EAX, 0x12345678 ; HLT (32-bit protected mode)
PROT_PROGRAM = b"\xB8\x78\x56\x34\x12\xF4"


def real_setup(program=REAL_PROGRAM):
    """A real-mode CPU with `program` at 0000:CODE."""
    mem = Memory()
    mem.load(CODE, program)
    cpu = Cpu()
    cpu.set_cs_ip(0x0000, CODE)
    return cpu, mem


def prot_setup(program=PROT_PROGRAM):
    """A flat protected-mode ring-0 CPU with `program` at CODE."""
    mem = Memory()
    mem.load(CODE, program)
    cpu = Cpu()
    cpu.enter_flat_protected()
    cpu.eip = CODE
    cpu.esp = 0x8000
    return cpu, mem


# --- Memory ------------------------------------------------------------------


class TestMemory:
    def test_starts_zeroed_and_sized(self):
        mem = Memory()
        assert len(mem) == 0x100_0000
        assert mem[0] == 0
        assert mem[0xFF_FFFF] == 0

    def test_index_read_write(self):
        mem = Memory()
        mem[0x12_3456] = 0xAB
        assert mem[0x12_3456] == 0xAB
        assert mem.read(0x12_3456) == 0xAB
        mem.write(0x12_3456, 0xCD)
        assert mem[0x12_3456] == 0xCD
        mem[-1] = 0x99
        assert mem[0xFF_FFFF] == 0x99

    def test_index_out_of_range(self):
        mem = Memory()
        with pytest.raises(IndexError):
            mem[0x100_0000]
        with pytest.raises(TypeError):
            mem["nope"]

    def test_slices(self):
        mem = Memory()
        mem[0x1000:0x1004] = b"\x01\x02\x03\x04"
        assert mem[0x1000:0x1004] == b"\x01\x02\x03\x04"
        assert mem[0x1000:0x1004:2] == b"\x01\x03"

    def test_load_wraps(self):
        mem = Memory()
        mem.load(0xFF_FFFF, b"\xAA\xBB")  # wraps to 0x000000
        assert mem[0xFF_FFFF] == 0xAA
        assert mem[0x00_0000] == 0xBB

    def test_repr(self):
        assert repr(Memory()) == "<remu.x86_32.Memory 16MiB>"


# --- Real mode ----------------------------------------------------------------


class TestRealMode:
    def test_step(self):
        cpu, mem = real_setup()
        assert not cpu.protected_mode
        cycles = cpu.step(mem)  # MOV AX, 0x1234
        assert cycles > 0
        assert cpu.eax & 0xFFFF == 0x1234
        cpu.step(mem)  # HLT
        assert cpu.halted

    def test_run_halts(self):
        cpu, mem = real_setup()
        executed, exit = cpu.run(mem, 100)
        assert executed == 2
        assert exit == RunExit.Halted
        assert cpu.eax & 0xFFFF == 0x1234
        assert cpu.halted

    def test_power_on_state(self):
        cpu = Cpu()
        assert cpu.eip == 0x0000_FFF0
        assert cpu.cs[0] == 0xF000
        assert cpu.cs[1] == 0xFFFF_0000  # the 386 reset quirk
        assert cpu.idtr == (0, 0x3FF)
        assert not cpu.halted and not cpu.shutdown
        assert cpu.cycles == 0


# --- Protected mode -----------------------------------------------------------


class TestProtectedMode:
    def test_flat_run(self):
        cpu, mem = prot_setup()
        executed, exit = cpu.run(mem, 100)
        assert exit == RunExit.Halted
        assert cpu.eax == 0x1234_5678
        assert cpu.protected_mode
        assert cpu.cpl == 0
        assert cpu.cr0 & 1  # PE
        assert cpu.cs == (0x08, 0, 0xFFFF_FFFF, 0x0C9B)
        assert cpu.ds == (0x10, 0, 0xFFFF_FFFF, 0x0C93)

    def test_flags(self):
        # ADD EAX, 1 ; HLT with EAX = 0xFFFFFFFF: wraps to 0, sets ZF and CF.
        cpu, mem = prot_setup(b"\x83\xC0\x01\xF4")
        cpu.eax = 0xFFFF_FFFF
        _, exit = cpu.run(mem, 100)
        assert exit == RunExit.Halted
        assert cpu.eax == 0
        assert cpu.zero
        assert cpu.carry
        assert not cpu.sign
        assert not cpu.overflow


# --- Duck-typed Python buses ----------------------------------------------------


class DictBus:
    """A minimal pure-Python bus backed by a dict (unwritten bytes read 0)."""

    def __init__(self):
        self.mem = {}

    def read(self, addr):
        return self.mem.get(addr, 0)

    def write(self, addr, value):
        self.mem[addr] = value

    def load(self, addr, data):
        for i, b in enumerate(data):
            self.mem[addr + i] = b


class TestPythonBus:
    def test_runs_program(self):
        bus = DictBus()
        bus.load(CODE, REAL_PROGRAM)
        cpu = Cpu()
        cpu.set_cs_ip(0x0000, CODE)
        executed, exit = cpu.run(bus, 100)
        assert executed == 2
        assert exit == RunExit.Halted
        assert cpu.eax & 0xFFFF == 0x1234

    def test_write_side_effects_visible(self):
        bus = DictBus()
        bus.load(CODE, b"\xA2\x00\x20")  # MOV [0x2000], AL
        cpu = Cpu()
        cpu.set_cs_ip(0x0000, CODE)
        cpu.eax = 0x77
        cpu.step(bus)
        assert bus.mem[0x2000] == 0x77

    def test_callback_exception_propagates(self):
        class ExplodingBus:
            def read(self, addr):
                raise RuntimeError("bus fault")

            def write(self, addr, value):
                pass

        cpu = Cpu()
        with pytest.raises(RuntimeError, match="bus fault"):
            cpu.step(ExplodingBus())

    def test_rejects_non_bus(self):
        cpu = Cpu()
        with pytest.raises(TypeError, match="bus"):
            cpu.step(42)


# --- Host traps (OS-emulation hooks) --------------------------------------------


class TestHostTrap:
    def test_syscall(self):
        cpu, mem = real_setup(b"\xCD\x80")  # INT 0x80
        cpu.syscall_int = 0x80
        executed, exit = cpu.run(mem, 10)
        assert executed == 1
        assert exit == RunExit.HostTrap
        assert cpu.host_trap == "syscall"  # peek does not clear
        assert cpu.host_trap == "syscall"
        assert cpu.take_host_trap() == "syscall"
        assert cpu.take_host_trap() is None
        assert cpu.host_trap is None

    def test_hook_properties(self):
        cpu = Cpu()
        assert cpu.syscall_int is None
        cpu.syscall_int = 0x80
        assert cpu.syscall_int == 0x80
        cpu.syscall_int = None
        assert cpu.syscall_int is None
        assert not cpu.trap_faults
        cpu.trap_faults = True
        assert cpu.trap_faults
        assert not cpu.extensions
        cpu.extensions = True
        assert cpu.extensions


# --- Segments and descriptor tables ---------------------------------------------


class TestSegments:
    def test_tuple_roundtrip(self):
        cpu = Cpu()
        cpu.ds = (0x10, 0x1_2345, 0xFFFF, 0x93)
        assert cpu.ds == (0x10, 0x1_2345, 0xFFFF, 0x93)

    def test_int_set_real_semantics(self):
        cpu = Cpu()
        cpu.cs = 0x9000
        assert cpu.cs == (0x9000, 0x9_0000, 0xFFFF, 0x93)

    def test_gdtr_idtr_roundtrip(self):
        cpu = Cpu()
        cpu.gdtr = (0x5000, 0x27)
        assert cpu.gdtr == (0x5000, 0x27)
        cpu.idtr = (0x6000, 0x7FF)
        assert cpu.idtr == (0x6000, 0x7FF)


# --- repr -----------------------------------------------------------------------


def test_repr():
    cpu = Cpu()
    text = repr(cpu)
    assert text.startswith("<remu.x86_32.Cpu")
    assert "eip=0x0000FFF0" in text
    assert "real" in text
    assert "HALTED" not in text

    cpu, mem = real_setup()
    cpu.run(mem, 10)
    assert "HALTED" in repr(cpu)

    cpu = Cpu()
    cpu.enter_flat_protected()
    assert "prot" in repr(cpu)
