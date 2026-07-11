#!/usr/bin/env python3
"""Wall-clock throughput of Unicorn Engine (QEMU/TCG) on the byte-identical
workloads that `examples/bench_x86.rs` runs on the Remu x86 cores.

    cargo run --release --example bench_x86
    python examples/bench_unicorn.py

See the Rust file's module comment for the full methodology. Both harnesses
print one TSV row per workload:

    engine  mode  name  instructions  best_s  mips  run_s,...  check

`check` is a final register value (AX/EAX/RAX family) that must match
between the engines. Python overhead is irrelevant here: each timed run is a
single `emu_start()` call executing ~10-40M guest instructions.
"""

import time

from unicorn import Uc, UC_ARCH_X86, UC_MODE_16, UC_MODE_32, UC_MODE_64
from unicorn.x86_const import (
    UC_X86_REG_AX, UC_X86_REG_CX, UC_X86_REG_IP,
    UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP,
    UC_X86_REG_RAX, UC_X86_REG_RCX, UC_X86_REG_RIP,
    UC_X86_REG_CR0, UC_X86_REG_CR3,
)

# Timed runs per workload; best is reported (warm translation cache).
RUNS = 5

# Trip counts — keep in sync with bench_x86.rs.
INNER16 = 0xFFFF
TIGHT16_OUTER, ALU16_OUTER, MEM16_OUTER, CALL16_OUTER = 80, 25, 25, 30
TIGHT_N, ALU_N, MEM_N, CALL_N = 20_000_000, 3_000_000, 4_000_000, 4_000_000


def per16(body, outer):
    return outer * (1 + body * INNER16 + 2)


def pat(n):
    """Source-buffer fill for the mem_rw workloads (same as bench_x86.rs)."""
    return bytes((i * 37 + 11) & 0xFF for i in range(n))


# --- 16-bit real mode, programs at 0000:1000 ---------------------------------

TIGHT16 = bytes([
    0xBA, 0x50, 0x00,        # 1000: MOV DX, 80
    0xB9, 0xFF, 0xFF,        # 1003: MOV CX, 0xFFFF
    0x49,                    # 1006: DEC CX
    0x75, 0xFD,              # 1007: JNZ 1006
    0x4A,                    # 1009: DEC DX
    0x75, 0xF7,              # 100A: JNZ 1003
])                           # 100C: end

ALU16 = bytes([
    0xB8, 0x34, 0x12,        # 1000: MOV AX, 0x1234
    0xBA, 0x19, 0x00,        # 1003: MOV DX, 25
    0xB9, 0xFF, 0xFF,        # 1006: MOV CX, 0xFFFF
    0x05, 0xB9, 0x79,        # 1009: ADD AX, 0x79B9
    0x35, 0x5A, 0x5A,        # 100C: XOR AX, 0x5A5A
    0xD1, 0xC0,              # 100F: ROL AX, 1
    0x29, 0xC8,              # 1011: SUB AX, CX
    0x49,                    # 1013: DEC CX
    0x75, 0xF3,              # 1014: JNZ 1009
    0x4A,                    # 1016: DEC DX
    0x75, 0xED,              # 1017: JNZ 1006
])                           # 1019: end

MEM16 = bytes([
    0xBA, 0x19, 0x00,        # 1000: MOV DX, 25
    0xBE, 0x00, 0x20,        # 1003: MOV SI, 0x2000 (src)
    0xBF, 0x00, 0x30,        # 1006: MOV DI, 0x3000 (dst)
    0xB9, 0xFF, 0xFF,        # 1009: MOV CX, 0xFFFF
    0x89, 0xCB,              # 100C: MOV BX, CX
    0x81, 0xE3, 0xFF, 0x03,  # 100E: AND BX, 0x3FF
    0x8B, 0x00,              # 1012: MOV AX, [BX+SI]
    0x89, 0x01,              # 1014: MOV [BX+DI], AX
    0x49,                    # 1016: DEC CX
    0x75, 0xF3,              # 1017: JNZ 100C
    0x4A,                    # 1019: DEC DX
    0x75, 0xED,              # 101A: JNZ 1009
])                           # 101C: end

CALL16 = bytes([
    0xBC, 0x00, 0xFF,        # 1000: MOV SP, 0xFF00
    0xBA, 0x1E, 0x00,        # 1003: MOV DX, 30
    0xB9, 0xFF, 0xFF,        # 1006: MOV CX, 0xFFFF
    0xE8, 0x08, 0x00,        # 1009: CALL 1014
    0x49,                    # 100C: DEC CX
    0x75, 0xFA,              # 100D: JNZ 1009
    0x4A,                    # 100F: DEC DX
    0x75, 0xF4,              # 1010: JNZ 1006
    0xEB, 0x02,              # 1012: JMP 1016
    0x40,                    # 1014: INC AX
    0xC3,                    # 1015: RET
])                           # 1016: end

# --- 32-bit flat protected mode, programs at 0x1000 --------------------------

TIGHT32 = bytes([
    0xB9, 0x00, 0x2D, 0x31, 0x01,  # 1000: MOV ECX, 20000000
    0x49,                          # 1005: DEC ECX
    0x75, 0xFD,                    # 1006: JNZ 1005
])                                 # 1008: end

ALU32 = bytes([
    0xB9, 0xC0, 0xC6, 0x2D, 0x00,  # 1000: MOV ECX, 3000000
    0xB8, 0x78, 0x56, 0x34, 0x12,  # 1005: MOV EAX, 0x12345678
    0x05, 0xB9, 0x79, 0x37, 0x9E,  # 100A: ADD EAX, 0x9E3779B9
    0x35, 0x5A, 0x5A, 0x5A, 0x5A,  # 100F: XOR EAX, 0x5A5A5A5A
    0xC1, 0xC0, 0x07,              # 1014: ROL EAX, 7
    0x0F, 0xAF, 0xC0,              # 1017: IMUL EAX, EAX
    0x29, 0xC8,                    # 101A: SUB EAX, ECX
    0x49,                          # 101C: DEC ECX
    0x75, 0xEB,                    # 101D: JNZ 100A
])                                 # 101F: end

MEM32 = bytes([
    0xB9, 0x00, 0x09, 0x3D, 0x00,        # 1000: MOV ECX, 4000000
    0xBE, 0x00, 0x00, 0x10, 0x00,        # 1005: MOV ESI, 0x100000 (src)
    0xBF, 0x00, 0x10, 0x10, 0x00,        # 100A: MOV EDI, 0x101000 (dst)
    0x89, 0xCA,                          # 100F: MOV EDX, ECX
    0x81, 0xE2, 0xFF, 0x03, 0x00, 0x00,  # 1011: AND EDX, 0x3FF
    0x8B, 0x04, 0x96,                    # 1017: MOV EAX, [ESI+EDX*4]
    0x89, 0x04, 0x97,                    # 101A: MOV [EDI+EDX*4], EAX
    0x49,                                # 101D: DEC ECX
    0x75, 0xEF,                          # 101E: JNZ 100F
])                                       # 1020: end

CALL32 = bytes([
    0xBC, 0x00, 0x00, 0x20, 0x00,  # 1000: MOV ESP, 0x200000
    0xB9, 0x00, 0x09, 0x3D, 0x00,  # 1005: MOV ECX, 4000000
    0xE8, 0x05, 0x00, 0x00, 0x00,  # 100A: CALL 1014
    0x49,                          # 100F: DEC ECX
    0x75, 0xF8,                    # 1010: JNZ 100A
    0xEB, 0x02,                    # 1012: JMP 1016
    0x40,                          # 1014: INC EAX
    0xC3,                          # 1015: RET
])                                 # 1016: end

# --- 64-bit long mode, programs at 0x10000 -----------------------------------

TIGHT64 = bytes([
    0x48, 0xC7, 0xC1, 0x00, 0x2D, 0x31, 0x01,  # 10000: MOV RCX, 20000000
    0x48, 0xFF, 0xC9,                          # 10007: DEC RCX
    0x75, 0xFB,                                # 1000A: JNZ 10007
])                                             # 1000C: end

ALU64 = bytes([
    0x48, 0xC7, 0xC1, 0xC0, 0xC6, 0x2D, 0x00,                    # 10000: MOV RCX, 3000000
    0x48, 0xB8, 0x78, 0x56, 0x34, 0x12, 0xEF, 0xCD, 0xAB, 0x89,  # 10007: MOV RAX, 0x89ABCDEF12345678
    0x48, 0x05, 0xB9, 0x79, 0x37, 0x9E,                          # 10011: ADD RAX, -0x61C88647
    0x48, 0x35, 0x5A, 0x5A, 0x5A, 0x5A,                          # 10017: XOR RAX, 0x5A5A5A5A
    0x48, 0xC1, 0xC0, 0x07,                                      # 1001D: ROL RAX, 7
    0x48, 0x0F, 0xAF, 0xC0,                                      # 10021: IMUL RAX, RAX
    0x48, 0x29, 0xC8,                                            # 10025: SUB RAX, RCX
    0x48, 0xFF, 0xC9,                                            # 10028: DEC RCX
    0x75, 0xE4,                                                  # 1002B: JNZ 10011
])                                                               # 1002D: end

MEM64 = bytes([
    0x48, 0xC7, 0xC1, 0x00, 0x09, 0x3D, 0x00,  # 10000: MOV RCX, 4000000
    0x48, 0xC7, 0xC6, 0x00, 0x00, 0x10, 0x00,  # 10007: MOV RSI, 0x100000 (src)
    0x48, 0xC7, 0xC7, 0x00, 0x10, 0x10, 0x00,  # 1000E: MOV RDI, 0x101000 (dst)
    0x48, 0x89, 0xCA,                          # 10015: MOV RDX, RCX
    0x48, 0x81, 0xE2, 0xFF, 0x03, 0x00, 0x00,  # 10018: AND RDX, 0x3FF
    0x48, 0x8B, 0x04, 0x96,                    # 1001F: MOV RAX, [RSI+RDX*4]
    0x48, 0x89, 0x04, 0x97,                    # 10023: MOV [RDI+RDX*4], RAX
    0x48, 0xFF, 0xC9,                          # 10027: DEC RCX
    0x75, 0xE9,                                # 1002A: JNZ 10015
])                                             # 1002C: end

CALL64 = bytes([
    0x48, 0xC7, 0xC4, 0x00, 0x00, 0x20, 0x00,  # 10000: MOV RSP, 0x200000
    0x48, 0xC7, 0xC1, 0x00, 0x09, 0x3D, 0x00,  # 10007: MOV RCX, 4000000
    0xE8, 0x07, 0x00, 0x00, 0x00,              # 1000E: CALL 1001A
    0x48, 0xFF, 0xC9,                          # 10013: DEC RCX
    0x75, 0xF6,                                # 10016: JNZ 1000E
    0xEB, 0x04,                                # 10018: JMP 1001E
    0x48, 0xFF, 0xC0,                          # 1001A: INC RAX
    0xC3,                                      # 1001D: RET
])                                             # 1001E: end


def bench(mode_name, mode, pc_reg, check_reg, base, map_size, src_addr, src_len,
          name, code, instructions, paging=False):
    mu = Uc(UC_ARCH_X86, mode)
    mu.mem_map(0, map_size)
    mu.mem_write(base, code)
    mu.mem_write(src_addr, pat(src_len))

    if paging:
        # Identity 4 KiB tables for the low 4 MiB: PD at 3 MiB, one PT,
        # entries P|RW|US|A|D — same tables as bench_x86.rs.
        mu.mem_write(0x300000, (0x301000 | 0x67).to_bytes(4, "little"))
        pt = b"".join(((page << 12) | 0x67).to_bytes(4, "little")
                      for page in range(1024))
        mu.mem_write(0x301000, pt)
        mu.reg_write(UC_X86_REG_CR3, 0x300000)
        mu.reg_write(UC_X86_REG_CR0, mu.reg_read(UC_X86_REG_CR0) | 0x8000_0001)

    end = base + len(code)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        mu.emu_start(base, end, timeout=60_000_000)  # 60s safety timeout
        times.append(time.perf_counter() - t0)
        pc = mu.reg_read(pc_reg)
        assert pc == end, f"{name}: stopped at {pc:#x}, expected {end:#x}"

    best = min(times)
    mips = instructions / best / 1e6
    runs = ",".join(f"{t:.4f}" for t in times)
    check = mu.reg_read(check_reg)
    print(f"unicorn\t{mode_name}\t{name}\t{instructions}\t{best:.4f}"
          f"\t{mips:.1f}\t{runs}\t{check:#x}")


def main():
    print("engine\tmode\tname\tinstructions\tbest_s\tmips\trun_s\tcheck")

    m16 = dict(mode_name="16", mode=UC_MODE_16, pc_reg=UC_X86_REG_IP,
               base=0x1000, map_size=0x10000, src_addr=0x2000, src_len=0x404)
    bench(**m16, check_reg=UC_X86_REG_CX,
          name="tight_loop", code=TIGHT16, instructions=1 + per16(2, TIGHT16_OUTER))
    bench(**m16, check_reg=UC_X86_REG_AX,
          name="alu_mix", code=ALU16, instructions=2 + per16(6, ALU16_OUTER))
    bench(**m16, check_reg=UC_X86_REG_AX,
          name="mem_rw", code=MEM16, instructions=3 + per16(6, MEM16_OUTER))
    bench(**m16, check_reg=UC_X86_REG_AX,
          name="call_ret", code=CALL16, instructions=3 + per16(5, CALL16_OUTER))

    m32 = dict(mode_name="32", mode=UC_MODE_32, pc_reg=UC_X86_REG_EIP,
               base=0x1000, map_size=0x400000, src_addr=0x100000, src_len=0x1008)
    bench(**m32, check_reg=UC_X86_REG_ECX,
          name="tight_loop", code=TIGHT32, instructions=1 + 2 * TIGHT_N)
    bench(**m32, check_reg=UC_X86_REG_EAX,
          name="alu_mix", code=ALU32, instructions=2 + 7 * ALU_N)
    bench(**m32, check_reg=UC_X86_REG_EAX,
          name="mem_rw", code=MEM32, instructions=3 + 6 * MEM_N)
    bench(**m32, check_reg=UC_X86_REG_EAX,
          name="call_ret", code=CALL32, instructions=3 + 5 * CALL_N)
    bench(**m32, check_reg=UC_X86_REG_ECX, paging=True,
          name="tight_loop_pg", code=TIGHT32, instructions=1 + 2 * TIGHT_N)
    bench(**m32, check_reg=UC_X86_REG_EAX, paging=True,
          name="mem_rw_pg", code=MEM32, instructions=3 + 6 * MEM_N)

    m64 = dict(mode_name="64", mode=UC_MODE_64, pc_reg=UC_X86_REG_RIP,
               base=0x10000, map_size=0x400000, src_addr=0x100000, src_len=0x1008)
    bench(**m64, check_reg=UC_X86_REG_RCX,
          name="tight_loop", code=TIGHT64, instructions=1 + 2 * TIGHT_N)
    bench(**m64, check_reg=UC_X86_REG_RAX,
          name="alu_mix", code=ALU64, instructions=2 + 7 * ALU_N)
    bench(**m64, check_reg=UC_X86_REG_RAX,
          name="mem_rw", code=MEM64, instructions=3 + 6 * MEM_N)
    bench(**m64, check_reg=UC_X86_REG_RAX,
          name="call_ret", code=CALL64, instructions=3 + 5 * CALL_N)


if __name__ == "__main__":
    main()
