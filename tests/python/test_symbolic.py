"""Tests for the `remu.symbolic` (concolic execution) bindings.

Mirrors the Rust suites in `tests/{x86_32,x86_64,arm32}_symbolic.rs`. The tests
that need an SMT solver skip themselves when none is on PATH (set
`REMU_SMT_SOLVER` to point at one).

Run with the extension installed (`maturin develop`): `pytest tests/python`.
"""

import pytest

import remu

symbolic = pytest.importorskip(
    "remu.symbolic", reason="extension built without the `symbolic` feature"
)

needs_solver = pytest.mark.skipif(
    not symbolic.solver_available(),
    reason="no SMT solver on PATH (set REMU_SMT_SOLVER)",
)

CODE = 0x1100
DATA = 0x2000


# --- Harnesses ----------------------------------------------------------------


def setup386(program):
    """A real-mode 386 with `program` at 0000:CODE and a sane stack."""
    from remu.x86_32 import Cpu, Memory

    mem = Memory()
    mem.load(CODE, program)
    cpu = Cpu()
    cpu.set_cs_ip(0x0000, CODE)
    cpu.esp = 0xFFF0
    return cpu, mem


def setup_arm(words):
    """An ARM CPU with `words` (LE u32 instructions) at 0x100."""
    from remu.arm32 import Cpu, Memory

    org = 0x100
    mem = Memory()
    for i, w in enumerate(words):
        mem.load(org + i * 4, w.to_bytes(4, "little"))
    cpu = Cpu()
    cpu.r15 = org
    cpu.r13 = 0x8000
    return cpu, mem


# ADD EAX, 3 ; XOR EAX, 0xFF ; SUB EAX, 1  (32-bit ops in real mode)
ALU_CHAIN = bytes(
    [
        0x66, 0x05, 0x03, 0x00, 0x00, 0x00,
        0x66, 0x35, 0xFF, 0x00, 0x00, 0x00,
        0x66, 0x2D, 0x01, 0x00, 0x00, 0x00,
    ]
)
# MOV EAX, [0x2000] ; CMP EAX, 0x12345678 ; JNE +7 ; MOV EBX, 0x600D ; HLT ; HLT
# The load is the modrm form (8B /r): the moffs form (A1) is not one of the
# opcodes the overlay models, so it would concretize EAX and record no branch.
MAGIC_CHECK = bytes(
    [
        0x66, 0x8B, 0x06, 0x00, 0x20,
        0x66, 0x3D, 0x78, 0x56, 0x34, 0x12,
        0x75, 0x07,
        0x66, 0xBB, 0x0D, 0x60, 0x00, 0x00,
        0xF4,
        0xF4,
    ]
)


# --- The overlay is opt-in ----------------------------------------------------


class TestInert:
    def test_off_by_default(self):
        cpu, mem = setup386(ALU_CHAIN)
        assert not cpu.sym_enabled
        assert cpu.sym_constraint_count == 0
        assert cpu.sym_check_invariant()

    def test_results_unchanged_when_enabled(self):
        """Enabling the overlay must not perturb concrete execution."""
        plain, mem1 = setup386(ALU_CHAIN)
        plain.eax = 0x12345678
        plain.run(mem1, 3)

        sym, mem2 = setup386(ALU_CHAIN)
        sym.eax = 0x12345678
        sym.sym_init()
        sym.sym_symbolize_reg(0, "eax")
        sym.run(mem2, 3)

        assert sym.eax == plain.eax == ((0x12345678 + 3) ^ 0xFF) - 1
        assert sym.sym_enabled and not plain.sym_enabled

    def test_methods_require_init(self):
        cpu, _ = setup386(ALU_CHAIN)
        with pytest.raises(RuntimeError, match="sym_init"):
            cpu.sym_symbolize_reg(0, "eax")
        with pytest.raises(RuntimeError, match="sym_init"):
            cpu.sym_symbolize_mem(DATA, 32, "m", 0)
        with pytest.raises(RuntimeError, match="sym_init"):
            cpu.sym_smtlib()

    def test_uninitialized_reports_the_real_problem(self):
        """Not having called sym_init() must not surface as "no such branch"."""
        cpu, _ = setup386(ALU_CHAIN)
        with pytest.raises(RuntimeError, match="sym_init"):
            cpu.sym_smtlib(flip=0)
        if hasattr(cpu, "sym_solve"):
            with pytest.raises(RuntimeError, match="sym_init"):
                cpu.sym_solve(flip=0)


# --- Tracking -----------------------------------------------------------------


class TestTracking:
    def test_golden_invariant_alu_chain(self):
        cpu, mem = setup386(ALU_CHAIN)
        cpu.eax = 0x12345678
        cpu.sym_init()
        assert cpu.sym_symbolize_reg(0, "eax") == 0
        for _ in range(3):
            cpu.step(mem)
            assert cpu.sym_check_invariant(), f"invariant broke at eip {cpu.eip:#x}"
        assert cpu.eax == ((0x12345678 + 3) ^ 0xFF) - 1

    def test_inputs_and_seed(self):
        cpu, _ = setup386(ALU_CHAIN)
        cpu.eax = 7
        cpu.sym_init()
        cpu.sym_symbolize_reg(0, "eax")
        cpu.sym_symbolize_mem(DATA, 8, "byte", 0xAB)
        assert cpu.sym_inputs() == [("eax", 32), ("byte", 8)]
        assert cpu.sym_seed() == {"eax": 7, "byte": 0xAB}

    def test_branch_on_symbolic_flag_records_a_constraint(self):
        cpu, mem = setup386(MAGIC_CHECK)
        mem.load(DATA, (0).to_bytes(4, "little"))
        cpu.sym_init()
        cpu.sym_symbolize_mem(DATA, 32, "input", 0)
        cpu.run(mem, 10)
        assert cpu.sym_constraint_count == 1
        assert cpu.ebx != 0x600D  # took the failing side
        assert cpu.sym_check_invariant()

    def test_smtlib_mentions_the_input(self):
        cpu, mem = setup386(MAGIC_CHECK)
        mem.load(DATA, (0).to_bytes(4, "little"))
        cpu.sym_init()
        cpu.sym_symbolize_mem(DATA, 32, "input", 0)
        cpu.run(mem, 10)

        script = cpu.sym_smtlib()
        assert "(set-logic QF_BV)" in script and "declare-const x!0" in script
        # Flipping negates the branch, so the two scripts must differ.
        assert cpu.sym_smtlib(flip=0) != script

    def test_bad_arguments_raise(self):
        cpu, _ = setup386(ALU_CHAIN)
        cpu.sym_init()
        with pytest.raises(ValueError, match="slot out of range"):
            cpu.sym_symbolize_reg(9, "nope")
        with pytest.raises(ValueError, match="width must be"):
            cpu.sym_symbolize_mem(DATA, 12, "nope", 0)
        with pytest.raises(IndexError, match="no branch 0"):
            cpu.sym_smtlib(flip=0)


# --- Solving ------------------------------------------------------------------


@needs_solver
class TestSolve:
    def test_solve_for_the_magic_value(self):
        """The classic query: what input reaches the other side of the branch?"""
        cpu, mem = setup386(MAGIC_CHECK)
        mem.load(DATA, (0).to_bytes(4, "little"))
        cpu.sym_init()
        cpu.sym_symbolize_mem(DATA, 32, "input", 0)
        cpu.run(mem, 10)

        model = cpu.sym_solve(flip=0)
        assert model == {"input": 0x12345678}

    def test_solve_current_path_is_satisfiable(self):
        cpu, mem = setup386(MAGIC_CHECK)
        mem.load(DATA, (1).to_bytes(4, "little"))
        cpu.sym_init()
        cpu.sym_symbolize_mem(DATA, 32, "input", 1)
        cpu.run(mem, 10)

        model = cpu.sym_solve()
        assert model is not None and model["input"] != 0x12345678

    def test_solve_out_of_range_flip_raises(self):
        cpu, _ = setup386(ALU_CHAIN)
        cpu.sym_init()
        with pytest.raises(IndexError):
            cpu.sym_solve(flip=3)


# --- Exploration --------------------------------------------------------------


@needs_solver
class TestFindInput:
    def test_finds_the_password(self):
        """Two byte-wise checks gated behind two branches: the driver has to
        flip both to reach the success marker."""
        # CMP AL, 0x11 ; JNE fail ; CMP AH, 0x22 ; JNE fail
        # MOV EBX, 0x600D ; HLT ; HLT
        program = bytes(
            [
                0x3C, 0x11,
                0x75, 0x09,
                0x80, 0xFC, 0x22,
                0x75, 0x04,
                0x66, 0xBB, 0x0D, 0x60, 0x00, 0x00,
                0xF4,
                0xF4,
            ]
        )
        rounds = []

        def harness(inputs):
            cpu, mem = setup386(program)
            v = inputs.get("pw", 0)
            cpu.eax = v
            cpu.sym_init()
            cpu.sym_symbolize_reg(0, "pw")
            cpu.run(mem, 16)
            rounds.append(v)
            return cpu.ebx == 0x600D, cpu

        found = symbolic.find_input(harness, max_iters=50)
        assert found is not None, f"no solution after rounds {rounds}"
        assert found["pw"] & 0xFF == 0x11
        assert (found["pw"] >> 8) & 0xFF == 0x22

    def test_harness_exception_propagates(self):
        def harness(inputs):
            raise KeyError("boom")

        with pytest.raises(KeyError, match="boom"):
            symbolic.find_input(harness, max_iters=5)

    def test_harness_must_return_a_cpu(self):
        def harness(inputs):
            return False, "not a cpu"

        with pytest.raises(TypeError, match="386, x86-64 or ARM32 Cpu"):
            symbolic.find_input(harness, max_iters=5)

    def test_harness_cpu_must_be_initialized(self):
        def harness(inputs):
            cpu, mem = setup386(ALU_CHAIN)
            return False, cpu  # never called sym_init()

        with pytest.raises(RuntimeError, match="no symbolic overlay"):
            symbolic.find_input(harness, max_iters=5)

    def test_no_solution_returns_none(self):
        def harness(inputs):
            cpu, mem = setup386(ALU_CHAIN)
            cpu.sym_init()
            cpu.sym_symbolize_reg(0, "eax")
            cpu.run(mem, 3)
            return False, cpu  # never succeeds, and records no branches

        assert symbolic.find_input(harness, max_iters=5) is None


# --- The other cores ----------------------------------------------------------


class TestOtherCores:
    def test_x86_64_tracks_and_solves(self):
        from remu.x86_64 import Cpu, Memory

        # CMP EAX, 0x2A ; JNE +1 ; HLT ; HLT  (32-bit operand in long mode)
        mem = Memory()
        code = 0x100000
        mem.load(code, bytes([0x3D, 0x2A, 0x00, 0x00, 0x00, 0x75, 0x01, 0xF4, 0xF4]))
        cpu = Cpu()
        cpu.setup_long_flat(mem, code, 0x200000)
        cpu.rax = 0
        cpu.sym_init()
        cpu.sym_symbolize_reg(0, "rax")
        cpu.run(mem, 4)

        assert cpu.sym_constraint_count == 1
        assert cpu.sym_check_invariant()
        if symbolic.solver_available():
            assert cpu.sym_solve(flip=0)["rax"] == 0x2A

    def test_arm32_tracks_and_solves(self):
        # CMP r0, #0x2A ; BNE +1 ; (fallthrough)
        cpu, mem = setup_arm([0xE350_002A, 0x1A00_0000, 0xE1A0_0000])
        cpu.r0 = 0
        cpu.sym_init()
        cpu.sym_symbolize_reg(0, "r0")
        cpu.run(mem, 2)

        assert cpu.sym_constraint_count == 1
        assert cpu.sym_check_invariant()
        if symbolic.solver_available():
            assert cpu.sym_solve(flip=0)["r0"] == 0x2A


# --- Module surface -----------------------------------------------------------


def test_module_surface():
    assert remu._remu.__symbolic__ is True
    assert isinstance(symbolic.solver_available(), bool)
    assert remu.symbolic is symbolic
