# Remu

A CPU emulation framework in Rust with cycle-conscious interpreters for the
MOS 6502, the Intel 8086/8088, the Intel 80386 and x86-64/AMD64. Goals:
realistic emulation, and emulation speed to the MAXIMUM.

## Rust

```rust
use remu::{Cpu, memory::FlatMemory, bus::Bus};

let mut mem = FlatMemory::new();
mem.load(0x0600, &[0xA9, 0x42]); // LDA #$42
mem.set_reset_vector(0x0600);

let mut cpu = Cpu::new();
cpu.reset(&mut mem);
cpu.step(&mut mem);
assert_eq!(cpu.regs.a, 0x42);
```

The 8086 core lives in `remu::x86` — 16-bit real mode, the full documented
instruction set plus the well-known undocumented encodings (`POP CS`, `SALC`,
`SETMO`, …), validated against the
[SingleStepTests 8088](https://github.com/SingleStepTests/8088) suite:

```rust
use remu::x86::{Cpu, LinearMemory};

let mut mem = LinearMemory::new();          // flat 1 MiB real-mode space
mem.load(0x0_0100, &[0xB8, 0x34, 0x12]);    // MOV AX, 0x1234

let mut cpu = Cpu::new();
cpu.regs.cs = 0x0000;
cpu.regs.ip = 0x0100;
cpu.step(&mut mem);
assert_eq!(cpu.regs.ax, 0x1234);
```

The 80386 core lives in `remu::x86_32` — 32-bit registers and addressing,
real mode, protected mode with privilege levels/gates/V86, and paging. The
real-mode instruction set (including the 386's *undefined* flag behavior for
BT/BTS/BTR/BTC, SHLD/SHRD, BSF/BSR, escaped-#DE IDIV quotients, and the
scaled-base SIB quirk) is validated against all 1,758,700 cases of the
[SingleStepTests 80386](https://github.com/SingleStepTests/80386) suite:

```rust
use remu::x86_32::{Cpu, LinearMemory};

let mut mem = LinearMemory::new();          // flat 16 MiB space
mem.load(0x0_1100, &[0x66, 0xB8, 0x78, 0x56, 0x34, 0x12]); // MOV EAX, 0x12345678

let mut cpu = Cpu::new();
cpu.set_cs_ip(0x0000, 0x1100);
cpu.step(&mut mem);
assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
```

The x86-64 core lives in `remu::x86_64` — sixteen 64-bit registers, REX
prefixes and RIP-relative addressing, real/protected/compatibility/long
modes, 4-level paging with NX and large pages, the SYSCALL/SYSRET and MSR
system interface, and the full x86-64 integer instruction set (no x87/SSE
state; SSE encodings raise `#UD` and CPUID says so). `Cpu::setup_long_flat`
drops straight into 64-bit long mode with identity paging:

```rust
use remu::x86_64::{Cpu, LinearMemory};

let mut mem = LinearMemory::new();          // flat 16 MiB space
mem.load(0x1100, &[0x48, 0xC7, 0xC0, 0x78, 0x56, 0x34, 0x12]); // MOV RAX, 0x12345678

let mut cpu = Cpu::new();
cpu.setup_long_flat(&mut mem, 0x1100, 0x8_0000);
cpu.step(&mut mem);
assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
```

## User-mode emulation (qemu-user style)

`remu-user` runs statically linked Linux i386 ELF executables on the 80386
core, on any host: the image is mapped into a sparse 4 GiB address space, the
CPU runs flat ring-3 protected mode, and `INT 0x80` Linux syscalls are
emulated by the host (`write`, `read`, `brk`, `mmap2`, `set_thread_area`
TLS via a real guest GDT, and friends):

```sh
cargo run --release --bin remu-user -- [--strace] [--trace] [--env K=V] \
    <program.elf> [guest args...]
```

The guest's exit status becomes the host exit code; a crash prints a
precise fault report (exception, rewound EIP, register dump). The layer is
written against the small `remu::usermode::UserArch` adapter trait, so
adding another CPU architecture means one new adapter. Post-386 integer
opcodes (`CMPXCHG`, `BSWAP`, `CMOVcc`, `CPUID`, ...) are enabled for
user-mode guests. Current limits (v1): `ET_EXEC` only (no dynamic
linking/PIE), integer-only binaries (no x87), no signals, no threads. See
`tests/data/README.md` for building compatible test programs.

## Tests

```sh
cargo test
```

To run the exhaustive per-opcode hardware suites (data not vendored), point
`REMU_HARTE_DIR` at a `SingleStepTests/65x02` `6502/v1` checkout,
`REMU_HARTE_8088_DIR` at a `SingleStepTests/8088` `v2` checkout and/or
`REMU_HARTE_80386_DIR` at a `SingleStepTests/80386` `v1_ex_real_mode`
checkout, then `cargo test --release`.

## Python

The same core is exposed as a Python extension module (PyO3 + maturin, ships
with type stubs):

```sh
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release
```

```python
import remu

mem = remu.Memory()                 # flat 64 KiB, supports indexing/slicing
mem[0x0600:0x0602] = b"\xA9\x42"    # LDA #$42
mem.set_reset_vector(0x0600)

cpu = remu.Cpu()
cpu.reset(mem)
cpu.step(mem)                       # -> cycles consumed
assert cpu.a == 0x42 and not cpu.zero

cpu.run(mem, 1_000_000)             # hot loop stays in Rust (~300M instr/s)
print(cpu)                          # <remu.Cpu pc=$0602 a=$42 ... p=nv--dIzc cycles=9>
print(remu.disassemble(mem, 0x0600))  # ('0600  LDA #$42', 1538)
```

Registers (`a x y sp pc p`) and flags (`carry`, `zero`, `interrupt_disable`,
`decimal`, `overflow`, `negative`) are plain read/write properties; interrupts
via `set_irq(level)` / `set_nmi(level)` / `trigger_nmi()`.

Any Python object with `read(addr)` and `write(addr, value)` works as a bus
(memory-mapped IO in pure Python):

```python
class MmioBus:
    def __init__(self):
        self.ram = bytearray(0x10000)
    def read(self, addr):
        return self.ram[addr]
    def write(self, addr, value):
        if addr == 0xF001:
            print(chr(value), end="")  # character-out port
        else:
            self.ram[addr] = value

cpu.step(MmioBus())
```

Python tests: `pytest tests/python` (after `maturin develop`).
