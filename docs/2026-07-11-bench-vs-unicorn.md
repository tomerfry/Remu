# Remu vs Unicorn Engine — performance report

*2026-07-11 · Windows 11 · AMD Ryzen 7 9800X3D (~5.2 GHz) · rustc 1.93.0,
`--release` with LTO + codegen-units=1 · Unicorn 2.1.2 (pip wheel, QEMU/TCG
JIT) · Python 3.12.3*

This report benchmarks the Remu interpreter cores against Unicorn Engine,
explains every number from the code, and lays out a prioritized improvement
roadmap. Two of the roadmap items were already validated by temporary
experiments with measured gains; their diffs are described below and are
intentionally **not** applied — they are the next development cycle.

## 1. Methodology

Two paired harnesses run **byte-identical hand-assembled guest programs**:

- `examples/bench_x86.rs` — Remu (native, `cargo run --release --example bench_x86`)
- `examples/bench_unicorn.py` — Unicorn (`python examples/bench_unicorn.py`)

Rules that make the numbers comparable:

- Each program initializes every register it uses, loops a statically known
  number of times, and falls through to its end address — the instruction
  count per run is exact. Remu steps exactly that many times and asserts it
  landed on the end address; Unicorn runs one `emu_start(begin, end)`.
- Only the execution loop is timed. Best of 5 runs is reported, so Unicorn
  numbers are for a **warm translation cache** (its favorable case).
- Both harnesses print a final register value per workload; **all 12 matched
  exactly across engines**, doubling as a cross-emulator correctness check.

Matrix: 3 modes × 4 workloads, ~10–40M instructions per run.

| Workload | Body |
|---|---|
| `tight_loop` | `DEC reg / JNZ` counting loop (dispatch + branch cost) |
| `alu_mix` | ADD/XOR/ROL/IMUL/SUB + DEC/JNZ (register ALU throughput) |
| `mem_rw` | masked-index load + store per iteration (memory path) |
| `call_ret` | near CALL/RET + DEC/JNZ (stack traffic, control flow) |

Modes: 16-bit real (8086 core), 32-bit flat protected, paging **off** (386
core), 64-bit long mode, 4-level paging **on** — long mode requires it
(x86-64 core, `setup_long_flat`, 1 GiB pages).

## 2. Results (MIPS, best of 5 — higher is better)

| Mode | Workload | Remu | Unicorn | Unicorn/Remu |
|---|---|---:|---:|---:|
| 16 | tight_loop | 256 | 723 | 2.8× |
| 16 | alu_mix | 198 | 690 | 3.5× |
| 16 | mem_rw | 205 | 131 | **0.64× — Remu wins** |
| 16 | call_ret | 257 | 132 | **0.51× — Remu wins** |
| 32 | tight_loop | 146 | 904 | 6.2× |
| 32 | alu_mix | 85 | 1418 | 16.7× |
| 32 | mem_rw | 85 | 200 | 2.3× |
| 32 | call_ret | 116 | 134 | 1.2× |
| 64 | tight_loop | 60 | 903 | 15.1× |
| 64 | alu_mix | 43 | 1434 | 33.7× |
| 64 | mem_rw | 45 | 55 | 1.2× |
| 64 | call_ret | 55 | 148 | 2.7× |

Same data as approximate **host cycles per emulated instruction** (at 5.2 GHz):

| Core | tight | alu | mem | call |
|---|---:|---:|---:|---:|
| Remu 8086 | 20 | 26 | 25 | 20 |
| Remu 386 | 36 | 61 | 61 | 45 |
| Remu x86-64 | 87 | 122 | 115 | 94 |
| Unicorn (64) | 6 | 3.6 | 95 | 35 |

## 3. Why the numbers look like this

### Unicorn's shape

- **Register ALU is JIT-compiled to native code.** TCG translates each basic
  block once; `alu_mix` then runs at ~1400 MIPS ≈ 3.6 host cycles per guest
  instruction. No interpreter can approach that — this is the 17–34× gap.
- **Every guest load/store pays the softmmu toll.** QEMU generates a TLB
  lookup + possible helper call per memory access. `mem_rw` collapses from
  1400 to 55–200 MIPS the moment memory enters the loop. This is Unicorn's
  structural weakness and Remu's opening.
- **CALL/RET is block-dispatch bound.** RET ends a translation block and
  costs an indirect-target lookup; `call_ret` sits at ~130–150 MIPS in every
  mode, independent of how fast the ALU is.
- **16-bit real mode is a slow path in QEMU** (segment arithmetic in
  helpers): 16-bit `mem_rw`/`call_ret` drop to ~130 MIPS, below Remu.

### Remu's shape — evidence from the code

Remu's per-instruction cost is uniform (no cliffs), but the baseline per
instruction is high and grows with core complexity: 8086 ≈ 20–26 cycles,
386 ≈ 36–61, x86-64 ≈ 87–122. Cost centers, in measured order:

1. **Whole-register-file snapshot every instruction** (`let saved =
   self.regs` in `step()`, for fault rewind — `x86_32/mod.rs`,
   `x86_64/mod.rs`). `size_of::<Registers>()` is **28 B (8086), 204 B (386),
   504 B (x86-64)** — the x86-64 core copies ~8 cache lines per emulated
   instruction. The 8086 core has no snapshot, which is a big part of why it
   is the fastest core. *Measured (experiment A, snapshot removed): 386
   +19–32%, x86-64 +16–26%.*

2. **Instruction fetch is per-byte through the full protection stack.**
   `fetch8` does, per byte: 15-byte-limit check → `fetch_check`
   (canonical/limit) → segment-base add → in long mode a full TLB
   `translate()` per byte → bus read. Immediates fetch via chained
   `fetch16/32/64`, so `MOV RCX, imm32` (7 bytes) performs 7 translations.
   Workload instructions average 1.5 (16-bit) to 3.7 (64-bit) bytes.
   *Measured (experiment B, a one-entry fetch-page cache on top of A —
   still byte-at-a-time!): x86-64 +10–16% more; cumulative vs baseline
   +27–46%.*

3. **Full re-decode every step.** The prefix scan, ModRM/SIB decode, and
   effective-address computation run again on every iteration of a loop the
   guest executes millions of times. Nothing is reused. This is the
   structural difference from Bochs-class interpreters (decoded-instruction
   caches) and the reason the ALU gap to a JIT is double-digit.

4. **Per-step event bookkeeping** — inhibit/NMI/INTR checks, TF test, cycles
   accounting: a handful of well-predicted branches; minor.

Already right (verified, no action needed): generic `Bus` is monomorphized
(memory reads inline to an array index — no virtual dispatch); ALU is
width-specialized by macro and inline with eager flags (parity via
`count_ones` → `popcnt`); wide memory accesses translate once when they
don't cross a page; the data TLB is a 256-entry direct-mapped array with a
dirty-bit fast path — its hit path is already short.

With experiments A+B applied, x86-64 `mem_rw` reached **60.5 MIPS vs
Unicorn's 54.9** — the x86-64 core beats Unicorn on memory-heavy code with
two small diffs.

## 4. Improvement roadmap (next development cycle)

Ordered by measured-or-estimated value per unit of risk. Ground rules per
the project philosophy: one increment at a time, each validated against the
SingleStepTests suites (8088: 3M cases, 386: 1.76M) plus `cargo test`, and
against these benchmarks for the speed delta.

### P1 — Replace the per-step register snapshot *(measured: +16–32%, low risk)*

Fault rewind needs pre-instruction register state, but faults are rare and
most instructions can only fault before writing any register. Options, by
increasing invasiveness:

- **Undo journal**: handlers record `(reg index, old value)` into a small
  fixed array (instructions write ≤ 2–3 GPRs; segment/system registers can
  keep coarse fallback via a flag that forces a full snapshot for the rare
  complex instructions — far-transfers, task switches).
- **Split hot/cold register file**: snapshot only the hot block (GPRs + RIP
  + seg caches, ~200 B for x86-64) and let the rare system-register writers
  snapshot the cold block themselves.
- Applies identically to the 386 and x86-64 cores; the existing
  `commit_on_fault` path (string ops) already shows the pattern.

### P2 — Fetch window: translate once per instruction, not per byte *(measured lower bound: +10–16% on x86-64; estimate +15–30% with full window)*

Translate `CS:RIP` once, grab up to 15 bytes (clamped at the page/limit
boundary) as a slice of guest memory, decode from the slice with one
canonical/limit check per instruction, refill only on page crossing. The
experiment that motivated this cached only the fetch *page* and still ran
byte-at-a-time checks; a real window removes the per-byte `fetch_check`,
`ilen` test, and segment-base add as well. Benefits all three cores (the
386/8086 cores pay the same per-byte checks minus the TLB).

### P3 — Decoded-instruction cache *(estimated 2–5× on hot loops; the strategic interpreter win)*

Bochs-style: a direct-mapped cache keyed by linear (or physical) instruction
address holding the decoded form — handler discriminant/fn pointer, operand
descriptors, length. Loop bodies then skip fetch+decode entirely.

- Self-modifying-code safety: per-page generation counters bumped by any
  write into a page that holds cached entries (page-granular bitmap checked
  in `lin_write*`); cache entries carry the generation they were decoded
  under. This keeps "realistic emulation" intact.
- Natural extension: cache straight-line *traces* and check for pending
  interrupts once per trace instead of once per instruction, amortizing the
  whole `step()` preamble.
- This is how Bochs sustains hundreds of MIPS with full accuracy; it is the
  only interpreter-shaped answer to the 6–34× register-workload gap.

### P4 — Small cleanups *(minor, do opportunistically)*

- Fold NMI/INTR/inhibit/TF into a single `events_pending` flag so the hot
  path is one predictable branch.
- Mark fault/exception paths `#[cold]`; batch `cycles` accounting per block
  once P3 exists.
- Offer `run(&mut bus, n)` on the cores so embedders aren't calling `step()`
  through a loop that can't hoist invariants.

### P5 — Long term: optional JIT/DBT backend

The only route to Unicorn-class ALU throughput (900–1400 MIPS) is dynamic
binary translation. If desired later: template-JIT per basic block (or
Cranelift), gated behind a feature, with the interpreter kept as the
reference for differential testing (JIT vs interpreter lockstep). Large
effort; decide after P1–P3 land, since those already change the competitive
picture on realistic (memory-touching) workloads.

### Benchmark hygiene for the cycle

- Keep `bench_x86.rs` / `bench_unicorn.py` as the regression pair; re-run
  before/after each increment (criterion `benches/cpu.rs` remains for
  fine-grained deltas).
- Gaps worth adding while at it: a paging-ON 386 workload and a ring-3
  variant (both harnesses currently run ring 0), and a `REP MOVS` workload
  (QEMU is unusually good at string ops — honest worst case).

## 5. Targets for the next cycle

With P1 + P2 (both already prototyped): x86-64 core ~55–90 MIPS → beats
Unicorn on `mem_rw` (measured 60.5 vs 54.9) and closes `call_ret` to ~2×.
With P3: hot loops should land in the 150–400 MIPS band across cores, i.e.
faster than Unicorn on every memory-touching workload in every mode, with
the pure-ALU gap reduced to the irreducible JIT-vs-interpreter difference.

Acceptance per increment: all SingleStepTests suites still pass; `cargo
test` green; benchmark table regenerated and appended to this doc with the
delta.
