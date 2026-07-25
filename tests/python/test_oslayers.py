"""Tests for the Linux OS-emulation layer bindings (`remu.usermode`,
`remu.os`, `remu.os64`).

The Usermode/Emulator386 tests run the checked-in `tests/data/hello-nolibc`
fixture (a real gcc-built static i386 nolibc binary: prints via write, echoes
argv, allocates with brk, exits with argc), mirroring the expectations of
tests/usermode_tests.rs. The Emulator64 tests synthesize a minimal static
ELF64 with struct.pack, mirroring the builder in tests/os64_x86_64.rs:
write(1, msg) via SYSCALL, then exit_group(7).
"""

import struct
from pathlib import Path

import pytest

from remu.os import Emulator as Emulator386
from remu.os64 import Emulator as Emulator64
from remu.usermode import Usermode

HELLO_NOLIBC = Path(__file__).resolve().parent.parent / "data" / "hello-nolibc"


@pytest.fixture(scope="module")
def nolibc_bytes():
    return HELLO_NOLIBC.read_bytes()


# --- ELF64 builder (mirrors tests/os64_x86_64.rs hello_elf) -------------------

VBASE = 0x0040_0000
EHDR, PHDR = 64, 56
CODE_LEN = 39  # the Rust builder's 36 bytes, +3 for `mov edi, imm32` vs `xor edi, edi`
MSG_VADDR = VBASE + EHDR + PHDR + CODE_LEN


def hello_elf64(msg: bytes, exit_code: int) -> bytes:
    """Static ET_EXEC ELF64, one R+X PT_LOAD at VBASE: write(1, msg, len)
    via SYSCALL, then exit_group(exit_code)."""
    code_off = EHDR + PHDR
    code = b"".join(
        [
            b"\xbf\x01\x00\x00\x00",  # mov edi, 1
            b"\x48\xbe" + struct.pack("<Q", MSG_VADDR),  # movabs rsi, msg
            b"\xba" + struct.pack("<I", len(msg)),  # mov edx, len
            b"\xb8\x01\x00\x00\x00",  # mov eax, 1 (write)
            b"\x0f\x05",  # syscall
            b"\xbf" + struct.pack("<I", exit_code),  # mov edi, exit_code
            b"\xb8\xe7\x00\x00\x00",  # mov eax, 231 (exit_group)
            b"\x0f\x05",  # syscall
        ]
    )
    assert len(code) == CODE_LEN, "code length must match the assumed msg offset"

    filesz = code_off + len(code) + len(msg)
    entry = VBASE + code_off

    f = bytearray(code_off)
    # --- ELF header ---
    f[0:4] = b"\x7fELF"
    f[4] = 2  # ELFCLASS64
    f[5] = 1  # little-endian
    f[6] = 1  # EV_CURRENT
    struct.pack_into("<H", f, 16, 2)  # e_type = ET_EXEC
    struct.pack_into("<H", f, 18, 62)  # e_machine = EM_X86_64
    struct.pack_into("<I", f, 20, 1)  # e_version
    struct.pack_into("<Q", f, 24, entry)  # e_entry
    struct.pack_into("<Q", f, 32, EHDR)  # e_phoff
    struct.pack_into("<H", f, 52, EHDR)  # e_ehsize
    struct.pack_into("<H", f, 54, PHDR)  # e_phentsize
    struct.pack_into("<H", f, 56, 1)  # e_phnum
    # --- Program header (PT_LOAD, R+X) ---
    struct.pack_into("<I", f, EHDR, 1)  # p_type = PT_LOAD
    struct.pack_into("<I", f, EHDR + 4, 5)  # p_flags = R|X
    struct.pack_into("<Q", f, EHDR + 8, 0)  # p_offset
    struct.pack_into("<Q", f, EHDR + 16, VBASE)  # p_vaddr
    struct.pack_into("<Q", f, EHDR + 24, VBASE)  # p_paddr
    struct.pack_into("<Q", f, EHDR + 32, filesz)  # p_filesz
    struct.pack_into("<Q", f, EHDR + 40, filesz)  # p_memsz
    struct.pack_into("<Q", f, EHDR + 48, 0x1000)  # p_align
    return bytes(f) + code + msg


MSG64 = b"hello from ring 3\n"


# --- Usermode (qemu-user style, i386) -----------------------------------------


class TestUsermode:
    def test_hello_nolibc_runs(self, nolibc_bytes):
        # Mirrors nolibc_acceptance_binary_runs in tests/usermode_tests.rs.
        um = Usermode(nolibc_bytes, argv=["hello-nolibc", "world"], envp=["TERM=dumb"])
        um.capture_fd(1)
        code = um.run()
        assert code == 2, "exit status is argc"
        out = um.fd_data(1).decode()
        assert "hello via write" in out
        assert "arg: hello-nolibc" in out
        assert "arg: world" in out
        assert "heap: brk works" in out
        assert um.fd_data(0) is None, "fd 0 is not a sink"

    def test_garbage_image_raises_valueerror(self):
        with pytest.raises(ValueError):
            Usermode(b"this is definitely not an ELF binary")

    def test_registers_and_memory(self, nolibc_bytes):
        um = Usermode(nolibc_bytes)
        assert um.pc > 0
        assert um.reg("eip") == um.pc
        assert um.reg("EIP") == um.pc, "register names are case-insensitive"
        um.set_reg("eax", 0x1234_5678)
        assert um.reg("eax") == 0x1234_5678
        with pytest.raises(ValueError):
            um.reg("xyz")
        with pytest.raises(ValueError):
            um.set_reg("eax", 2**32), "does not fit in 32 bits"
        # The stack region [0xBF800000, 0xC0000000) is mapped at load.
        um.write(0xBFF0_0000, b"remu\x00")
        assert um.read(0xBFF0_0000, 4) == b"remu"
        assert um.read_cstr(0xBFF0_0000) == b"remu"
        with pytest.raises(ValueError):
            um.read(0x1000, 4), "null page is unmapped"

    def test_huge_read_raises(self, nolibc_bytes):
        # An absurd length must raise at the first unmapped chunk — never
        # attempt the full up-front allocation (whose failure would abort the
        # interpreter, not raise).
        um = Usermode(nolibc_bytes)
        with pytest.raises(ValueError):
            um.read(0x1000, 1 << 45)

    def test_strace_and_dump(self, nolibc_bytes):
        um = Usermode(nolibc_bytes)
        assert um.strace is False
        um.strace = True
        assert um.strace is True
        dump = um.dump()
        assert "EAX=" in dump and "EIP=" in dump

    def test_step_and_repr(self, nolibc_bytes):
        um = Usermode(nolibc_bytes)
        um.capture_fd(1)
        assert um.step() is None, "first instruction does not exit"
        r = repr(um)
        assert "remu.usermode.Usermode" in r and "running" in r
        assert um.run() == 1  # exit status is argc; default argv is ["a.out"]
        assert "exited(1)" in repr(um)


# --- Emulator386 (qiling style, i386) -----------------------------------------


class TestEmulator386:
    def test_hello_nolibc_via_path(self):
        # hello-nolibc only uses write/brk/exit_group via int 0x80, all
        # serviced by the os layer; same expectations as the usermode run.
        emu = Emulator386(str(HELLO_NOLIBC), argv=["/hello", "world"])
        emu.capture_fd(1)
        code = emu.run()
        assert code == 2, "exit status is argc"
        assert emu.running is False
        assert emu.exit_code == 2
        out = emu.fd_data(1).decode()
        assert "hello via write" in out
        assert "arg: /hello" in out
        assert "arg: world" in out
        assert "heap: brk works" in out

    def test_cap_and_resume(self):
        emu = Emulator386(str(HELLO_NOLIBC), argv=["/hello", "world"])
        emu.capture_fd(1)
        assert emu.run(max_instructions=10) is None, "budget too small to finish"
        assert emu.running is True
        assert emu.run() == 2, "resumes and completes"
        assert emu.running is False

    def test_capture_fd_validation(self):
        emu = Emulator386(str(HELLO_NOLIBC))
        assert emu.fd_data(1) is None, "no capture enabled yet"
        with pytest.raises(ValueError):
            emu.capture_fd(5), "this layer only captures the standard streams"

    def test_memory_and_regs(self):
        emu = Emulator386(str(HELLO_NOLIBC))
        assert emu.pc > 0
        assert emu.reg("eip") == emu.pc
        assert len(emu.read(emu.pc, 4)) == 4, "entry code is mapped"
        assert emu.reg("cr3") != 0, "paging is on"
        emu.set_reg("eax", 0xDEAD_BEEF)
        assert emu.reg("EAX") == 0xDEAD_BEEF
        with pytest.raises(ValueError):
            emu.reg("rax"), "64-bit names are not i386 registers"
        with pytest.raises(ValueError):
            emu.set_reg("esp", 2**32)
        with pytest.raises(ValueError):
            emu.read(0x10, 4), "null page is unmapped"

    def test_huge_read_raises(self):
        emu = Emulator386(str(HELLO_NOLIBC))
        with pytest.raises(ValueError):
            emu.read(0x10, 1 << 40)

    def test_non_elf_path_raises(self, tmp_path):
        bad = tmp_path / "bad.bin"
        bad.write_bytes(b"this is definitely not an ELF binary")
        with pytest.raises(ValueError):
            Emulator386(str(bad))
        with pytest.raises(ValueError):
            Emulator386(str(tmp_path / "missing.elf"))

    def test_repr(self):
        emu = Emulator386(str(HELLO_NOLIBC))
        r = repr(emu)
        assert "remu.os.Emulator" in r and "running" in r


# --- Emulator64 (qiling style, x86-64) ----------------------------------------


class TestEmulator64:
    def test_hello_from_bytes(self):
        emu = Emulator64(hello_elf64(MSG64, 7), argv=["hello"])
        emu.capture_fd(1)
        code = emu.run()
        assert code == 7, "guest calls exit_group(7)"
        assert emu.running is False
        assert emu.exit_code == 7
        assert emu.fd_data(1) == MSG64, "captured stdout matches the guest's write"
        assert emu.fd_data(0) is None, "fd 0 is not a sink"

    def test_hello_from_path(self, tmp_path):
        path = tmp_path / "hello64.elf"
        path.write_bytes(hello_elf64(MSG64, 7))
        emu = Emulator64(path, argv=["hello"])  # os.PathLike loads via path
        emu.capture_fd(1)
        assert emu.run() == 7
        assert emu.fd_data(1) == MSG64

    def test_cap_memory_and_regs(self):
        emu = Emulator64(hello_elf64(MSG64, 7))
        assert emu.pc == VBASE + EHDR + PHDR, "entry right after the headers"
        assert emu.reg("rip") == emu.pc
        assert emu.read(VBASE, 4) == b"\x7fELF", "the header is mapped at VBASE"
        assert emu.read_cstr(MSG_VADDR) == MSG64, "page zero-fill terminates it"
        assert emu.reg("cr3") != 0, "long-mode paging is on"
        emu.set_reg("r15", 0x1122_3344_5566_7788)
        assert emu.reg("R15") == 0x1122_3344_5566_7788
        with pytest.raises(ValueError):
            emu.reg("eax"), "32-bit names are not x86-64 registers"
        with pytest.raises(ValueError):
            emu.read(0x10, 4), "null page is unmapped"
        # Cap, then resume to completion.
        emu.capture_fd(1)
        assert emu.run(max_instructions=2) is None
        assert emu.running is True
        assert emu.run() == 7
        assert emu.running is False

    def test_huge_read_raises(self):
        emu = Emulator64(hello_elf64(MSG64, 7))
        with pytest.raises(ValueError):
            emu.read(0x1000, 1 << 45)

    def test_bad_program_raises(self):
        with pytest.raises(ValueError):
            Emulator64(b"this is definitely not an ELF binary")
        with pytest.raises(TypeError):
            Emulator64(12345)

    def test_repr(self):
        emu = Emulator64(hello_elf64(MSG64, 7))
        r = repr(emu)
        assert "remu.os64.Emulator" in r and "running" in r
        emu.run()
        assert "exited(7)" in repr(emu)
