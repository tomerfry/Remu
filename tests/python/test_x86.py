"""Tests for the `remu.x86` (Intel 8086/8088) Python bindings."""

import pytest

from remu.x86 import Cpu, Memory


def real_mode_cpu(mem, code, org=0x1000):
    """CPU with `code` at physical `org`, entered via CS=0, IP=org."""
    mem.load(org, code)
    cpu = Cpu()
    cpu.cs = 0x0000
    cpu.ds = 0x0000
    cpu.ss = 0x0000
    cpu.es = 0x0000
    cpu.ip = org
    return cpu


class DictBus:
    """Duck-typed bus backed by a dict, with optional port registers."""

    def __init__(self, code, org=0x1000):
        self.mem = {org + i: b for i, b in enumerate(code)}
        self.ports = {}

    def read(self, addr):
        return self.mem.get(addr, 0)

    def write(self, addr, value):
        self.mem[addr] = value

    def io_read(self, port):
        return self.ports.get(port, 0xFF)

    def io_write(self, port, value):
        self.ports[port] = value


# --- Memory ------------------------------------------------------------------


class TestMemory:
    def test_size_and_indexing(self):
        mem = Memory()
        assert len(mem) == 0x10_0000
        assert mem[0] == 0
        mem[0xF_FFFF] = 0xAB
        assert mem[0xF_FFFF] == 0xAB
        assert mem[-1] == 0xAB
        with pytest.raises(IndexError):
            mem[0x10_0000]

    def test_slices_and_load(self):
        mem = Memory()
        mem[0x1000:0x1004] = b"\x01\x02\x03\x04"
        assert mem[0x1000:0x1004] == b"\x01\x02\x03\x04"
        mem.load(0x2000, b"\xAA\xBB")
        assert mem[0x2000] == 0xAA
        assert mem[0x2001] == 0xBB


# --- Cpu ---------------------------------------------------------------------


class TestCpu:
    def test_mov_and_hlt(self):
        mem = Memory()
        cpu = real_mode_cpu(mem, b"\xB8\x34\x12\xF4")  # MOV AX,0x1234; HLT
        cpu.step(mem)
        assert cpu.ax == 0x1234
        assert cpu.ip == 0x1003
        cpu.step(mem)
        assert cpu.halted

    def test_run_stops_on_hlt(self):
        mem = Memory()
        cpu = real_mode_cpu(mem, b"\xB8\x34\x12\xF4")
        executed = cpu.run(mem, 100)
        assert executed == 2
        assert cpu.halted
        assert cpu.ax == 0x1234

    def test_segmented_addressing(self):
        mem = Memory()
        mem.load(0x1_0000, b"\xB8\xCD\xAB\xF4")
        cpu = Cpu()
        cpu.cs = 0x1000  # 0x1000 << 4 == 0x1_0000
        cpu.ip = 0x0000
        cpu.step(mem)
        assert cpu.ax == 0xABCD

    def test_eight_bit_halves(self):
        cpu = Cpu()
        cpu.ax = 0x1234
        assert cpu.al == 0x34
        assert cpu.ah == 0x12
        cpu.al = 0xFF
        assert cpu.ax == 0x12FF
        cpu.bh = 0xAA
        assert cpu.bx == 0xAA00

    def test_flags(self):
        mem = Memory()
        # MOV AX,0xFFFF; ADD AX,1 -> AX=0, ZF+CF set
        cpu = real_mode_cpu(mem, b"\xB8\xFF\xFF\x05\x01\x00\xF4")
        cpu.run(mem, 3)
        assert cpu.ax == 0
        assert cpu.zero
        assert cpu.carry
        assert not cpu.sign
        cpu.flags = 0x0000
        assert not cpu.zero and not cpu.carry

    def test_hlt_resumes_on_interrupt(self):
        mem = Memory()
        # IVT vector 0x20 at 0x80 -> handler 0000:2000
        mem[0x80:0x84] = b"\x00\x20\x00\x00"
        cpu = real_mode_cpu(mem, b"\xF4")  # HLT
        cpu.interrupt = True  # IF set
        assert cpu.run(mem, 10) == 1
        assert cpu.halted
        cpu.assert_intr(0x20)
        cpu.step(mem)  # services the interrupt
        assert not cpu.halted
        assert cpu.cs == 0x0000
        assert cpu.ip == 0x2000

    def test_duck_typed_bus(self):
        bus = DictBus(b"\xB8\x34\x12\xF4")
        cpu = Cpu()
        cpu.cs = 0x0000
        cpu.ip = 0x1000
        assert cpu.run(bus, 10) == 2
        assert cpu.ax == 0x1234
        assert cpu.halted

    def test_io_ports(self):
        bus = DictBus(b"\xE4\x10\xE6\x11\xF4")  # IN AL,0x10; OUT 0x11,AL; HLT
        bus.ports[0x10] = 0x5A
        cpu = Cpu()
        cpu.cs = 0x0000
        cpu.ip = 0x1000
        cpu.run(bus, 10)
        assert cpu.al == 0x5A
        assert bus.ports[0x11] == 0x5A

    def test_io_defaults_without_handlers(self):
        class BareBus:
            def __init__(self, code):
                self.mem = {0x1000 + i: b for i, b in enumerate(code)}

            def read(self, addr):
                return self.mem.get(addr, 0)

            def write(self, addr, value):
                self.mem[addr] = value

        bus = BareBus(b"\xE4\x10\xF4")  # IN AL,0x10 -> open bus 0xFF
        cpu = Cpu()
        cpu.cs = 0x0000
        cpu.ip = 0x1000
        cpu.run(bus, 10)
        assert cpu.al == 0xFF

    def test_bus_exception_propagates(self):
        class BoomBus(DictBus):
            def write(self, addr, value):
                raise RuntimeError("boom")

        # MOV [0x2000],AL writes through the bus
        bus = BoomBus(b"\xA2\x00\x20")
        cpu = Cpu()
        cpu.cs = 0x0000
        cpu.ds = 0x0000
        cpu.ip = 0x1000
        with pytest.raises(RuntimeError, match="boom"):
            cpu.step(bus)

    def test_repr(self):
        cpu = Cpu()
        assert "remu.x86.Cpu" in repr(cpu)
