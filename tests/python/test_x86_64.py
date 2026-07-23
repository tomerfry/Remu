"""Tests for the `remu.x86_64` bindings.

Run with the extension installed (`maturin develop`): `pytest tests/python`.
"""

import pytest

from remu.x86_64 import Cpu, Memory, RunExit

# MOV RAX, 42 (48 C7 C0 imm32) ; HLT
MOV_RAX_42_HLT = b"\x48\xC7\xC0\x2A\x00\x00\x00\xF4"
# MOV AX, 0x1234 (B8 imm16, 16-bit real mode) ; HLT
MOV_AX_1234_HLT = b"\xB8\x34\x12\xF4"
SYSCALL = b"\x0F\x05"


def long_mode_cpu(program, org=0x10000):
    """A CPU dropped into 64-bit long mode with `program` at `org`."""
    mem = Memory()
    mem.load(org, program)
    cpu = Cpu()
    cpu.setup_long_flat(mem, org, 0x20000)
    return cpu, mem


# --- Memory ------------------------------------------------------------------


class TestMemory:
    def test_default_size(self):
        mem = Memory()
        assert len(mem) == 16 * 1024 * 1024
        assert mem[0] == 0
        assert mem[len(mem) - 1] == 0

    def test_custom_size(self):
        mem = Memory(size=0x40000)
        assert len(mem) == 0x40000  # a power of two is kept as-is

    def test_size_rounding(self):
        assert len(Memory(size=0x11000)) == 0x20000  # up to a power of two
        assert len(Memory(size=1)) == 0x10000  # minimum 64 KiB

    def test_index_read_write(self):
        mem = Memory(size=0x10000)
        mem[0x1234] = 0xAB
        assert mem[0x1234] == 0xAB
        assert mem.read(0x1234) == 0xAB
        mem.write(0x1234, 0xCD)
        assert mem[0x1234] == 0xCD

    def test_negative_index(self):
        mem = Memory(size=0x10000)
        mem[-1] = 0x99
        assert mem[0xFFFF] == 0x99

    def test_slices(self):
        mem = Memory(size=0x10000)
        mem[0x600:0x604] = b"\x01\x02\x03\x04"
        assert mem[0x600:0x604] == b"\x01\x02\x03\x04"
        assert mem[0x600:0x604:2] == b"\x01\x03"

    def test_load_and_bytes(self):
        mem = Memory(size=0x10000)
        mem.load(0x200, b"\xAA\xBB")
        assert mem[0x200:0x202] == b"\xAA\xBB"
        dump = bytes(mem)
        assert len(dump) == 0x10000
        assert dump[0x200] == 0xAA


# --- Long mode ---------------------------------------------------------------


class TestLongMode:
    def test_mov_rax_hlt(self):
        cpu, mem = long_mode_cpu(MOV_RAX_42_HLT)
        assert cpu.long_mode
        assert cpu.mode64
        assert cpu.run(mem, 10) == (2, RunExit.Halted)
        assert cpu.rax == 42
        assert cpu.halted


# --- Real mode ---------------------------------------------------------------


class TestRealMode:
    def test_power_on_state(self):
        cpu = Cpu()
        assert not cpu.long_mode
        assert not cpu.protected_mode
        assert cpu.rip == 0xFFF0
        assert cpu.cs[0] == 0xF000  # (sel, base, limit, attrs)

    def test_mov_ax_hlt(self):
        mem = Memory()
        mem.load(0x1000, MOV_AX_1234_HLT)
        cpu = Cpu()
        cpu.set_cs_ip(0, 0x1000)
        cpu.step(mem)  # MOV AX, 0x1234
        cpu.step(mem)  # HLT
        assert cpu.rax & 0xFFFF == 0x1234
        assert cpu.halted


# --- SYSCALL host trap -------------------------------------------------------


class TestSyscallTrap:
    def test_syscall_traps_to_host(self):
        cpu, mem = long_mode_cpu(SYSCALL)
        cpu.trap_syscall = True
        executed, why = cpu.run(mem, 10)
        assert why == RunExit.HostTrap
        assert executed == 1
        assert cpu.host_trap == "syscall"  # peek does not consume
        assert cpu.take_host_trap() == "syscall"
        assert cpu.take_host_trap() is None


# --- Duck-typed Python buses -------------------------------------------------


class DictBus:
    """A minimal pure-Python bus over a dict (unmapped reads return 0)."""

    def __init__(self, image=b"", org=0):
        self.mem = dict(enumerate(image, start=org))

    def read(self, addr):
        return self.mem.get(addr, 0)

    def write(self, addr, value):
        self.mem[addr] = value


class TestPythonBus:
    def test_runs_real_mode_program(self):
        bus = DictBus(MOV_AX_1234_HLT, org=0x1000)
        cpu = Cpu()
        cpu.set_cs_ip(0, 0x1000)
        assert cpu.run(bus, 10) == (2, RunExit.Halted)
        assert cpu.rax & 0xFFFF == 0x1234

    def test_callback_exception_propagates(self):
        class ExplodingBus:
            def read(self, addr):
                raise RuntimeError("bus fault")

            def write(self, addr, value):
                pass

        cpu = Cpu()
        cpu.set_cs_ip(0, 0x1000)
        with pytest.raises(RuntimeError, match="bus fault"):
            cpu.step(ExplodingBus())
        cpu = Cpu()
        cpu.set_cs_ip(0, 0x1000)
        with pytest.raises(RuntimeError, match="bus fault"):
            cpu.run(ExplodingBus(), 10)

    def test_rejects_non_bus(self):
        cpu = Cpu()
        with pytest.raises(TypeError, match="bus"):
            cpu.step(42)


# --- Registers ---------------------------------------------------------------


class TestRegisters:
    def test_r8_r15_roundtrip(self):
        cpu = Cpu()
        names = ["r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
        for i, name in enumerate(names):
            setattr(cpu, name, 0x8877_6655_4433_2200 + i)
        for i, name in enumerate(names):
            assert getattr(cpu, name) == 0x8877_6655_4433_2200 + i

    def test_gpr_roundtrip(self):
        cpu = Cpu()
        names = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"]
        for i, name in enumerate(names):
            setattr(cpu, name, 0x1111_2222_3333_4440 + i)
        for i, name in enumerate(names):
            assert getattr(cpu, name) == 0x1111_2222_3333_4440 + i

    def test_msr_roundtrip(self):
        cpu = Cpu()
        cpu.lstar = 0xFFFF_8000_0000_1000  # a value above 2**63 fits
        assert cpu.lstar == 0xFFFF_8000_0000_1000

    def test_flag_bits(self):
        cpu = Cpu()
        cpu.zero = True
        assert cpu.rflags & 0x40
        cpu.rflags = 0
        assert not cpu.zero


# --- Repr --------------------------------------------------------------------


def test_repr():
    cpu = Cpu()
    text = repr(cpu)
    assert "rip=0x" in text and "real" in text
    mem = Memory()
    cpu.setup_long_flat(mem, 0x10000, 0x20000)
    assert "long" in repr(cpu)
