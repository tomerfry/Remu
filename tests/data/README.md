# User-mode test binaries

Prebuilt, committed guest executables for the `remu-user` acceptance tests, so
running `cargo test` never needs a cross toolchain.

## hello-nolibc

A freestanding (nolibc) Linux i386 program: prints via `write`, echoes its
argv (exercising the initial SysV stack image), grows the heap with `brk`,
and exits with `argc`. Source: `hello.c`.

Rebuild on Linux or WSL (needs only gcc with 32-bit binutils):

```sh
gcc -m32 -march=i386 -O1 -nostdlib -static -fno-pie -no-pie \
    -o hello-nolibc hello.c
```

The user-mode runner enables the core's `extensions` opcodes (`CMPXCHG`,
`XADD`, `BSWAP`, `CMOVcc`, `CPUID`, ...), so integer binaries built for
486+/686 generally run too — `-march=i386` is just the most conservative
choice. There is still no x87 FPU: floating-point instructions are silent
no-ops, so keep test programs integer-only.
