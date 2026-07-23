"""Tests for the `remu.arm32` (ARM7TDMI / ARMv4T) bindings.

Run with the extension installed (`maturin develop`): `pytest tests/python`.
"""

import pytest

from remu.arm32 import Cpu, Memory, RunExit

# Hand-assembled instructions (stored little-endian in memory).
MOV_R0_42 = 0xE3A0002A  # MOV r0, #42
MOV_R0_1 = 0xE3A00001  # MOV r0, #1
MOV_R1_2 = 0xE3A01002  # MOV r1, #2
STR_R0_R0_100 = 0xE5800100  # STR r0, [r0, #0x100]
SWI_0 = 0xEF000000  # SWI #0
MRC_P15 = 0xEE110F10  # MRC p15, ... — no coprocessor: undefined trap
T_MOVS_R0_42 = 0x202A  # Thumb MOVS r0, #42


def arm_program(words, org=0x1000):
    """Memory with the ARM instruction `words` at `org`."""
    mem = Memory()
    for i, w in enumerate(words):
        mem.load(org + 4 * i, w.to_bytes(4, "little"))
    return mem


# --- Memory ------------------------------------------------------------------


class TestMemory:
    def test_default_16mib(self):
        mem = Memory()
        assert len(mem) == 16 * 1024 * 1024
        assert mem[0] == 0
        assert mem[-1] == 0

    def test_with_size(self):
        assert len(Memory(0x20000)) == 0x20000
        # Rounded up to a power of two, minimum 64 KiB.
        assert len(Memory(1)) == 0x10000
        assert len(Memory(0x18000)) == 0x20000

    def test_index_and_slice(self):
        mem = Memory()
        mem[0x1234] = 0xAB
        assert mem[0x1234] == 0xAB
        assert mem.read(0x1234) == 0xAB
        mem.write(0x1234, 0xCD)
        assert mem[0x1234] == 0xCD
        mem[0x2000:0x2004] = b"\x01\x02\x03\x04"
        assert mem[0x2000:0x2004] == b"\x01\x02\x03\x04"
        assert mem[0x2000:0x2004:2] == b"\x01\x03"
        with pytest.raises(IndexError):
            mem[16 * 1024 * 1024]
        with pytest.raises(TypeError):
            mem["nope"]

    def test_load_is_little_endian_bytes(self):
        mem = Memory()
        mem.load(0x1000, MOV_R0_42.to_bytes(4, "little"))
        assert mem[0x1000:0x1004] == b"\x2a\x00\xa0\xe3"


# --- Cpu ----------------------------------------------------------------------


class TestCpu:
    def test_power_on_state(self):
        cpu = Cpu()
        assert cpu.pc == 0
        assert cpu.mode == "svc"
        assert not cpu.thumb
        assert cpu.irq_disable and cpu.fiq_disable
        assert cpu.cycles == 0
        assert not cpu.halted
        assert cpu.host_trap is None

    def test_arm_mov(self):
        mem = arm_program([MOV_R0_42])
        cpu = Cpu()
        cpu.pc = 0x1000
        cycles = cpu.step(mem)
        assert cycles >= 1
        assert cpu.r0 == 42
        assert cpu.pc == 0x1004
        assert cpu.cycles == cycles

    def test_thumb_mov(self):
        mem = Memory()
        mem.load(0x1000, T_MOVS_R0_42.to_bytes(2, "little"))  # b"\x2a\x20"
        cpu = Cpu()
        cpu.thumb = True
        cpu.pc = 0x1000
        cpu.step(mem)
        assert cpu.r0 == 42
        assert cpu.thumb
        assert cpu.pc == 0x1002

    def test_run_completes_budget(self):
        mem = arm_program([MOV_R0_1, MOV_R1_2])
        cpu = Cpu()
        cpu.pc = 0x1000
        assert cpu.run(mem, 2) == (2, RunExit.Completed)
        assert (cpu.r0, cpu.r1) == (1, 2)
        assert cpu.pc == 0x1008

    def test_register_aliases(self):
        cpu = Cpu()
        cpu.sp, cpu.lr, cpu.pc = 0x8000, 0x1234, 0x1000
        assert (cpu.r13, cpu.r14, cpu.r15) == (0x8000, 0x1234, 0x1000)
        cpu.r13 = 0x9000
        assert cpu.sp == 0x9000
        with pytest.raises(OverflowError):
            cpu.r0 = 2**32

    def test_halted_setter(self):
        # ARMv4T has no WFI: halting is embedder-controlled state.
        mem = arm_program([MOV_R0_42])
        cpu = Cpu()
        cpu.pc = 0x1000
        cpu.halted = True
        assert cpu.run(mem, 5) == (0, RunExit.Halted)
        assert cpu.halted
        cpu.halted = False
        assert cpu.run(mem, 1) == (1, RunExit.Completed)

    def test_irq_line(self):
        mem = arm_program([MOV_R0_42])
        cpu = Cpu()
        cpu.pc = 0x1000
        cpu.irq_disable = False
        cpu.set_irq(True)
        cpu.step(mem)  # delivers the IRQ instead of executing
        assert cpu.mode == "irq"
        assert cpu.pc == 0x18
        assert cpu.lr == 0x1004  # architected return offset
        assert cpu.irq_disable
        cpu.set_irq(False)

    def test_raise_exception(self):
        cpu = Cpu()
        cpu.pc = 0x1000
        cpu.raise_exception("undefined")
        assert cpu.mode == "und"
        assert cpu.pc == 0x04
        assert cpu.lr == 0x1000
        with pytest.raises(ValueError):
            cpu.raise_exception("reset")

    def test_reset(self):
        cpu = Cpu()
        cpu.r0, cpu.pc = 7, 0x1000
        cpu.mode = "irq"
        cpu.halted = True
        cpu.reset()
        assert (cpu.r0, cpu.pc, cpu.mode) == (0, 0, "svc")
        assert not cpu.halted


# --- Host traps -----------------------------------------------------------------


class TestHostTraps:
    def test_swi_host_trap(self):
        mem = arm_program([SWI_0])
        cpu = Cpu()
        cpu.pc = 0x1000
        cpu.trap_swi = True
        executed, exit = cpu.run(mem, 10)
        assert exit == RunExit.HostTrap
        assert executed == 1
        assert cpu.pc == 0x1004  # already past the SWI
        assert cpu.host_trap == "syscall"  # peek does not consume
        assert cpu.take_host_trap() == "syscall"
        assert cpu.take_host_trap() is None
        assert cpu.host_trap is None

    def test_swi_vectors_without_trap(self):
        mem = arm_program([SWI_0])
        cpu = Cpu()
        cpu.mode = "sys"
        cpu.pc = 0x1000
        cpu.step(mem)
        assert cpu.host_trap is None
        assert cpu.mode == "svc"
        assert cpu.pc == 0x08
        assert cpu.lr == 0x1004
        assert cpu.spsr & 0x1F == 0x1F  # saved CPSR shows System mode

    def test_trap_faults(self):
        mem = arm_program([MRC_P15])
        cpu = Cpu()
        cpu.pc = 0x1000
        cpu.trap_faults = True
        _, exit = cpu.run(mem, 10)
        assert exit == RunExit.HostTrap
        assert cpu.take_host_trap() == ("exception", "undefined")
        assert cpu.pc == 0x1000  # rewound to the faulting instruction


# --- Modes and PSRs --------------------------------------------------------------


class TestModesAndPsr:
    def test_mode_banking(self):
        cpu = Cpu()
        assert cpu.mode == "svc"
        cpu.sp = 0x1111
        cpu.mode = "irq"
        cpu.sp = 0x2222
        cpu.mode = "svc"
        assert cpu.sp == 0x1111
        cpu.mode = "irq"
        assert cpu.sp == 0x2222

    def test_mode_rejects_unknown(self):
        cpu = Cpu()
        with pytest.raises(ValueError):
            cpu.mode = "hyp"

    def test_flag_bits_roundtrip(self):
        cpu = Cpu()
        for name, bit in [
            ("negative", 31),
            ("zero", 30),
            ("carry", 29),
            ("overflow", 28),
            ("irq_disable", 7),
            ("fiq_disable", 6),
            ("thumb", 5),
        ]:
            setattr(cpu, name, True)
            assert cpu.cpsr & (1 << bit), name
            assert getattr(cpu, name), name
            setattr(cpu, name, False)
            assert not cpu.cpsr & (1 << bit), name

    def test_cpsr_write_banks(self):
        cpu = Cpu()  # svc
        cpu.sp = 0x1111
        cpu.cpsr = (cpu.cpsr & ~0x1F) | 0x12  # IRQ mode
        assert cpu.mode == "irq"
        assert cpu.sp != 0x1111  # IRQ's banked stack pointer
        cpu.cpsr = (cpu.cpsr & ~0x1F) | 0x13  # back to Supervisor
        assert cpu.mode == "svc"
        assert cpu.sp == 0x1111

    def test_spsr(self):
        cpu = Cpu()  # svc: has an SPSR
        cpu.spsr = 0xA00000D1
        assert cpu.spsr == 0xA00000D1
        cpu.mode = "sys"  # no SPSR: reads as the CPSR, writes discarded
        assert cpu.spsr == cpu.cpsr
        before = cpu.cpsr
        cpu.spsr = 0x12345678
        assert cpu.spsr == before


# --- Duck-typed Python buses ---------------------------------------------------


class DictBus:
    """A minimal dict-backed pure-Python bus (unmapped bytes read 0)."""

    def __init__(self):
        self.mem = {}

    def read(self, addr):
        return self.mem.get(addr, 0)

    def write(self, addr, value):
        self.mem[addr] = value

    def load(self, addr, data):
        for i, b in enumerate(data):
            self.mem[addr + i] = b


class ExplodingBus:
    def read(self, addr):
        raise RuntimeError("bus fault")

    def write(self, addr, value):
        pass


class TestPythonBus:
    def test_runs_program(self):
        bus = DictBus()
        bus.load(0x1000, MOV_R0_42.to_bytes(4, "little"))
        cpu = Cpu()
        cpu.pc = 0x1000
        cpu.step(bus)
        assert cpu.r0 == 42
        assert cpu.pc == 0x1004

    def test_run_and_writes_visible(self):
        bus = DictBus()
        bus.load(0x1000, MOV_R0_42.to_bytes(4, "little"))
        bus.load(0x1004, STR_R0_R0_100.to_bytes(4, "little"))
        cpu = Cpu()
        cpu.pc = 0x1000
        assert cpu.run(bus, 2) == (2, RunExit.Completed)
        # 42 + 0x100 is unaligned; ARMv4 word stores force-align the address.
        assert bus.mem[(42 + 0x100) & ~3] == 42

    def test_swi_trap_via_python_bus(self):
        bus = DictBus()
        bus.load(0x1000, SWI_0.to_bytes(4, "little"))
        cpu = Cpu()
        cpu.pc = 0x1000
        cpu.trap_swi = True
        _, exit = cpu.run(bus, 10)
        assert exit == RunExit.HostTrap
        assert cpu.take_host_trap() == "syscall"

    def test_callback_exception_propagates(self):
        cpu = Cpu()
        with pytest.raises(RuntimeError, match="bus fault"):
            cpu.step(ExplodingBus())

    def test_run_callback_exception_propagates(self):
        cpu = Cpu()
        with pytest.raises(RuntimeError, match="bus fault"):
            cpu.run(ExplodingBus(), 100)

    def test_rejects_non_bus(self):
        cpu = Cpu()
        with pytest.raises(TypeError, match="bus"):
            cpu.step(42)


# --- repr -----------------------------------------------------------------------


class TestRepr:
    def test_cpu_repr(self):
        cpu = Cpu()
        cpu.pc = 0x1000
        text = repr(cpu)
        assert "remu.arm32.Cpu" in text
        assert "pc=0x00001000" in text
        assert "svc" in text and "arm" in text
        assert "HALTED" not in text
        cpu.thumb = True
        cpu.halted = True
        text = repr(cpu)
        assert "thumb" in text and "HALTED" in text

    def test_memory_repr(self):
        assert repr(Memory()) == "<remu.arm32.Memory 16MiB>"
        assert repr(Memory(0x10000)) == "<remu.arm32.Memory 64KiB>"
