"""Tests for the `remu` Python bindings.

Run with the extension installed (`maturin develop`): `pytest tests/python`.
"""

import pytest

import remu


def make_program(code, org=0x0600):
    """Memory with `code` at `org` and the reset vector pointing at it."""
    mem = remu.Memory()
    mem.load(org, code)
    mem.set_reset_vector(org)
    return mem


def booted_cpu(mem):
    cpu = remu.Cpu()
    cpu.reset(mem)
    return cpu


# --- Memory ------------------------------------------------------------------


class TestMemory:
    def test_starts_zeroed_and_sized(self):
        mem = remu.Memory()
        assert len(mem) == 0x10000
        assert mem[0] == 0
        assert mem[0xFFFF] == 0

    def test_index_read_write(self):
        mem = remu.Memory()
        mem[0x1234] = 0xAB
        assert mem[0x1234] == 0xAB
        assert mem.read(0x1234) == 0xAB
        mem.write(0x1234, 0xCD)
        assert mem[0x1234] == 0xCD

    def test_negative_index(self):
        mem = remu.Memory()
        mem[-1] = 0x99
        assert mem[0xFFFF] == 0x99

    def test_index_out_of_range(self):
        mem = remu.Memory()
        with pytest.raises(IndexError):
            mem[0x10000]
        with pytest.raises(IndexError):
            mem[2**70]  # doesn't fit a machine word, still an IndexError
        with pytest.raises(TypeError):
            mem["nope"]

    def test_slices(self):
        mem = remu.Memory()
        mem[0x0600:0x0604] = b"\x01\x02\x03\x04"
        assert mem[0x0600:0x0604] == b"\x01\x02\x03\x04"
        assert mem[0x0600:0x0604:2] == b"\x01\x03"

    def test_slice_from_list(self):
        mem = remu.Memory()
        mem[0:3] = [1, 2, 3]
        assert mem[0:3] == b"\x01\x02\x03"

    def test_slice_length_mismatch(self):
        mem = remu.Memory()
        with pytest.raises(ValueError):
            mem[0:4] = b"\x01"

    def test_load_and_bytes(self):
        mem = remu.Memory()
        mem.load(0xFFFE, b"\xAA\xBB\xCC")  # wraps to $0000
        assert mem[0xFFFE] == 0xAA
        assert mem[0xFFFF] == 0xBB
        assert mem[0x0000] == 0xCC
        dump = bytes(mem)
        assert len(dump) == 0x10000
        assert dump[0xFFFE] == 0xAA

    def test_reset_vector(self):
        mem = remu.Memory()
        mem.set_reset_vector(0x1234)
        assert mem[0xFFFC] == 0x34
        assert mem[0xFFFD] == 0x12


# --- Cpu ----------------------------------------------------------------------


class TestCpu:
    def test_reset_loads_vector(self):
        mem = make_program(b"", org=0x8000)
        cpu = booted_cpu(mem)
        assert cpu.pc == 0x8000
        assert cpu.sp == 0xFD
        assert cpu.interrupt_disable
        assert cpu.cycles == 7  # RESET takes 7 cycles

    def test_lda_immediate(self):
        mem = make_program(b"\xA9\x42")  # LDA #$42
        cpu = booted_cpu(mem)
        cycles = cpu.step(mem)
        assert cycles == 2
        assert cpu.a == 0x42
        assert not cpu.zero
        assert not cpu.negative

    def test_flags(self):
        mem = make_program(b"\xA9\x00\xA9\x80")  # LDA #$00 ; LDA #$80
        cpu = booted_cpu(mem)
        cpu.step(mem)
        assert cpu.zero and not cpu.negative
        cpu.step(mem)
        assert cpu.negative and not cpu.zero

    def test_register_setters(self):
        cpu = remu.Cpu()
        cpu.a, cpu.x, cpu.y, cpu.sp, cpu.pc = 1, 2, 3, 4, 0x1234
        assert (cpu.a, cpu.x, cpu.y, cpu.sp, cpu.pc) == (1, 2, 3, 4, 0x1234)
        cpu.carry = True
        assert cpu.p & 0x01
        cpu.p = 0x00  # U is forced back on
        assert cpu.p == 0x20
        with pytest.raises(OverflowError):
            cpu.a = 256

    def test_program_loop(self):
        # LDX #$00 ; INX ; CPX #$0A ; BNE -5 ; KIL
        mem = make_program(b"\xA2\x00\xE8\xE0\x0A\xD0\xFB\x02")
        cpu = booted_cpu(mem)
        executed = cpu.run(mem, 1000)
        assert cpu.halted
        assert cpu.x == 0x0A
        assert executed < 1000  # stopped at the jam, not the budget
        # further stepping is a no-op
        assert cpu.step(mem) == 0

    def test_run_is_interruptible(self):
        # An infinite loop: JMP $0600. interrupt_main() sets the same flag as
        # Ctrl-C; the timer thread can only deliver it if run() periodically
        # releases the GIL, and run() must then notice it instead of spinning
        # to the budget. The budget bounds the test if that ever regresses.
        import threading
        import _thread

        mem = make_program(b"\x4C\x00\x06")
        cpu = booted_cpu(mem)
        timer = threading.Timer(0.1, _thread.interrupt_main)
        timer.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                cpu.run(mem, 10**10)
        finally:
            timer.cancel()

    def test_run_respects_instruction_budget(self):
        mem = make_program(b"\xEA" * 32)  # NOPs
        cpu = booted_cpu(mem)
        assert cpu.run(mem, 5) == 5
        assert cpu.pc == 0x0605

    def test_stack_via_memory(self):
        mem = make_program(b"\xA9\x42\x48")  # LDA #$42 ; PHA
        cpu = booted_cpu(mem)
        cpu.run(mem, 2)
        assert mem[0x01FD] == 0x42
        assert cpu.sp == 0xFC

    def test_irq_and_nmi(self):
        mem = make_program(b"\x58\xEA\xEA")  # CLI ; NOP ; NOP
        mem[0xFFFE:0x10000] = b"\x00\x90"  # IRQ vector -> $9000
        cpu = booted_cpu(mem)
        cpu.step(mem)  # CLI
        cpu.set_irq(True)
        cpu.step(mem)  # services the IRQ
        assert cpu.pc == 0x9000
        assert cpu.interrupt_disable

        mem[0xFFFA:0xFFFC] = b"\x00\xA0"  # NMI vector -> $A000
        cpu.trigger_nmi()
        cpu.step(mem)
        assert cpu.pc == 0xA000

    def test_repr(self):
        cpu = remu.Cpu()
        text = repr(cpu)
        assert "pc=$0000" in text and "sp=$FD" in text and "I" in text


# --- Duck-typed Python buses ---------------------------------------------------


class RamBus:
    """A minimal pure-Python bus."""

    def __init__(self):
        self.ram = bytearray(0x10000)

    def read(self, addr):
        return self.ram[addr]

    def write(self, addr, value):
        self.ram[addr] = value


class TestPythonBus:
    def test_runs_program(self):
        bus = RamBus()
        bus.ram[0x0600:0x0602] = b"\xA9\x42"  # LDA #$42
        bus.ram[0xFFFC:0xFFFE] = b"\x00\x06"
        cpu = remu.Cpu()
        cpu.reset(bus)
        assert cpu.pc == 0x0600
        cpu.step(bus)
        assert cpu.a == 0x42

    def test_write_side_effects_visible(self):
        bus = RamBus()
        bus.ram[0x0600:0x0603] = b"\x8D\x00\x20"  # STA $2000
        bus.ram[0xFFFC:0xFFFE] = b"\x00\x06"
        cpu = remu.Cpu()
        cpu.reset(bus)
        cpu.a = 0x77
        cpu.step(bus)
        assert bus.ram[0x2000] == 0x77

    def test_callback_exception_propagates(self):
        class ExplodingBus:
            def read(self, addr):
                raise RuntimeError("bus fault")

            def write(self, addr, value):
                pass

        cpu = remu.Cpu()
        with pytest.raises(RuntimeError, match="bus fault"):
            cpu.reset(ExplodingBus())

    def test_bad_read_value_raises(self):
        class BadBus:
            def read(self, addr):
                return 1000  # not a byte

            def write(self, addr, value):
                pass

        cpu = remu.Cpu()
        with pytest.raises(OverflowError):
            cpu.reset(BadBus())

    def test_rejects_non_bus(self):
        cpu = remu.Cpu()
        with pytest.raises(TypeError, match="bus"):
            cpu.step(42)


# --- Disassembler ---------------------------------------------------------------


class TestDisassemble:
    def test_disassemble(self):
        mem = make_program(b"\xA9\x42\x4C\x00\x06")  # LDA #$42 ; JMP $0600
        text, nxt = remu.disassemble(mem, 0x0600)
        assert text == "0600  LDA #$42"
        assert nxt == 0x0602
        text, nxt = remu.disassemble(mem, nxt)
        assert text == "0602  JMP $0600"
        assert nxt == 0x0605

    def test_disassemble_python_bus(self):
        bus = RamBus()
        bus.ram[0:2] = b"\xA2\x10"  # LDX #$10
        text, _ = remu.disassemble(bus, 0)
        assert text == "0000  LDX #$10"


def test_version():
    assert remu.__version__ == "0.1.0"
