# Remu

A CPU emulation framework in Rust, starting with a cycle-conscious MOS 6502
interpreter. Goals: realistic emulation, and emulation speed to the MAXIMUM.

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

```sh
cargo test
```

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
