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

`-march=i386` matters: the emulated CPU is a genuine 80386, so 486+
instructions (`CMPXCHG`, `XADD`, `BSWAP`) and later ones (`CMOV`, `CPUID`)
raise #UD. This also rules out stock musl/glibc static binaries for now —
musl's i386 atomics use `lock cmpxchg` unconditionally.
