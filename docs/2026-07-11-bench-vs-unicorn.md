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

---

## 6. Results of the P1 + P2 cycle (2026-07-11, same machine)

The cycle landed as five increments on `performence-upscaling`, each
validated against `cargo test` (171 tests, incl. new fault-rewind and
fetch-boundary pins) and the 80386 MOO suite (1,758,700 cases, debug +
release). New cross-checks: all 12 final-register values still match
Unicorn exactly.

What landed:

- **P1 (both cores)** — the per-step whole-`Registers` snapshot is now a
  GPR-only snapshot (32 B / 128 B); all cold-field writers (15 call sites on
  the 386, 19 on x86-64) call a flag-guarded `prepare_cold_write()` that
  captures the full file at most once per instruction. A permanent
  debug-build differential assert replays the old rewind and compares —
  every debug test run, including a 1.76M-case debug MOO run, validates the
  escalation census bit-exactly. Three new tests pin cold-state rewind on
  mid-instruction faults (IRET/IRETQ outer-SS fault after CS commit, call
  gate + INT gate inner-stack fault after SS commit).
- **P2 (x86-64 only)** — a one-entry *persistent* fetch-translation cache:
  a tag compare replaces the per-byte NX-aware TLB lookup, and a hot loop
  within one page pays zero translates. Only the translation is cached
  (bytes are read live — SMC exact); invalidation rides the existing choke
  points (TLB flushes, INVLPG, `prepare_cold_write` ⊇ every CPL/mode/CS
  change). `fetch16/32/64` additionally issue one wide bus read when the
  field fits in the cached page. Four new tests pin page-straddle #PF/CR2
  (unmapped + NX), SMC-into-next-instruction, and a guest PD-rewrite +
  INVLPG remap.
- **P2 on the 386: measured and rejected.** Three window variants all
  regressed the paging-off benchmarks (criterion: arith −9%, call_ret
  −7.6%): with paging off the per-byte checks are cheaper than window
  bookkeeping at `fetch8`'s many inline sites, and the per-instruction
  refill call dominates 2-byte instructions. The 386 keeps its per-byte
  fetch; its boundary tests were kept as regression pins. The per-page
  *persistent* design would help the 386's paged mode, but no harness
  measures it yet (see the paging-ON workload gap in §4).

### MIPS, best of 5 — before → after

| Mode | Workload | Before | After | Δ | Unicorn | Unicorn/Remu now |
|---|---|---:|---:|---:|---:|---:|
| 16 | tight_loop | 256 | 254 | — | 723 | 2.8× |
| 16 | alu_mix | 198 | 199 | — | 690 | 3.5× |
| 16 | mem_rw | 205 | 202 | — | 131 | **0.65× — Remu wins** |
| 16 | call_ret | 257 | 258 | — | 132 | **0.51× — Remu wins** |
| 32 | tight_loop | 146 | 187 | +28% | 904 | 4.8× |
| 32 | alu_mix | 85 | 95 | +12% | 1418 | 14.9× |
| 32 | mem_rw | 85 | 93 | +9% | 200 | 2.2× |
| 32 | call_ret | 116 | 139 | +20% | 134 | **0.96× — Remu wins** |
| 64 | tight_loop | 60 | 74 | +23% | 903 | 12.3× |
| 64 | alu_mix | 43 | 52 | +21% | 1434 | 27.8× |
| 64 | mem_rw | 45 | 55 | +22% | 55 | **~1.0× — parity** |
| 64 | call_ret | 55 | 71 | +29% | 148 | 2.1× |

(8086 core untouched by design — it has no snapshot and no per-byte fetch
checks. Unicorn column unchanged: same machine, same version, warm cache.)

Targets vs outcome: the x86-64 core reached parity-or-better on `mem_rw`
(55–56 vs 54.9 across runs) as predicted; 32-bit `call_ret` now beats
Unicorn outright, which the roadmap had not predicted. The P2 lesson —
hoisting cheap checks costs more than it saves; only hoisting the TLB
lookup pays, and only when it persists across instructions — narrows P3's
design space usefully: a decoded-instruction cache must amortize *decode*,
not fetch checks, to clear the bar.

---

## 7. Results of the P3 cycle (2026-07-11, same machine)

The cycle landed as seven increments on `performence-upscaling-phase2`,
each validated against `cargo test` (207 tests, incl. 24 new decode/icache
pins) and the 80386 MOO suite (1,758,700 cases, debug + release). Debug
bench runs additionally re-validate *every cache hit* against a fresh
decode via a permanent differential assert — several hundred million hit
validations per run, zero mismatches. All 13 paired final-register
cross-checks (12 original + `rep_movs`) still match Unicorn exactly.

What landed:

- **P3 (both cores)** — a decoded-instruction cache: direct-mapped
  16384-entry table (32 B/entry on the 386, 40 B on x86-64) keyed on the
  physical address of the first instruction byte with the decode context
  (CS.D, CR0.PE, EFLAGS.VM, `extensions`; plus `mode64` on x86-64) folded
  into the key's high bits. A hit skips the fetch, prefix scan and opcode
  dispatch, re-evaluates the cached EA *formula* from live registers into
  the same `Operand` the fused path builds, and runs a verbatim copy of the
  fused handler body — cycle formulas included. Misses decode through a new
  `try_decode` whose failure path rewinds and re-runs the fused dispatch,
  so exception ordering is bit-identical.
- **SMC exactness** — per-physical-page write stamps against a monotonic
  u64 store clock, bumped by every guest store (first *and* last byte's
  page for unsplit wide writes) and by page-walker A/D write-backs (a guest
  can execute from its own page tables — a test pins it). Physical keying
  makes entries CR3-independent: a remap test proves re-keying with no
  flush and entry retention across the switch. Host-side writes are covered
  by an O(1) `invalidate_icache()` (generation bump), called after every
  serviced trap in the usermode/os/os64 layers. `prepare_cold_write`
  deliberately does *not* flush — SYSCALL/SYSRET-heavy guests keep their
  cache; a per-hit CS-limit (legacy) / canonicality (64-bit) compare
  replaces the per-byte `fetch_check`.
- **Not cached, by design** — page-crossing instructions (the stamp covers
  one page and physical contiguity is not stable under remapping), LOCK-
  and REP-prefixed forms, segment loads/STI (interrupt shadow), 67-prefixed
  instructions in 64-bit mode (RIP-relative truncation corner), and >7
  prefixes. All run fused, unchanged.
- **Coverage** — the full ALU/MOV/INC/DEC/PUSH/POP/Jcc/JMP/CALL/RET/LEA/
  TEST/shift/IMUL matrix (the bench histograms showed `alu_mix` is 29%
  shift+IMUL, which the roadmap's initial subset had deferred), plus the
  real-code families: PUSH imm, IMUL r,r/m,imm, XCHG/NOP, MOVZX/MOVSX,
  MOVSXD, SETcc, CMOVcc.
- **Codegen lessons (hard-won, worth recording):** `exec_decoded` must be
  `#[inline(always)]` and the cache arrays fixed-size `Box<[T; N]>`
  (bounds-check elision) or the entire win evaporates; passing the decoded
  struct via out-parameter instead of enum payloads was worth 20+ MIPS; and
  the two cores want opposite match shapes (386: all arms inlined; x86-64:
  extended arms out of line — 4–5% either way). The per-case `Cpu::new()`
  in the MOO harness had to become per-file reuse (a fresh 640 KiB icache
  1.76M times tripled suite wall time).

### MIPS, best of 5 — before → after

| Mode | Workload | Before | After | Δ | Unicorn | Unicorn/Remu now |
|---|---|---:|---:|---:|---:|---:|
| 16 | tight_loop | 255 | 256 | — | 703 | 2.7× |
| 16 | alu_mix | 199 | 198 | — | 674 | 3.4× |
| 16 | mem_rw | 205 | 200 | — | 131 | **0.65× — Remu wins** |
| 16 | call_ret | 256 | 256 | — | 131 | **0.51× — Remu wins** |
| 32 | tight_loop | 181 | 184 | +2% | 889 | 4.8× |
| 32 | alu_mix | 96 | 122 | +27% | 1407 | 11.5× |
| 32 | mem_rw | 99 | 137 | +38% | 199 | 1.5× |
| 32 | call_ret | 139 | 152 | +9% | 133 | **0.87× — Remu wins** |
| 64 | tight_loop | 75 | 102 | +36% | 891 | 8.8× |
| 64 | alu_mix | 51 | 88 | +72% | 1519 | 17.3× |
| 64 | mem_rw | 56 | 89 | +60% | 55 | **0.61× — Remu wins** |
| 64 | call_ret | 69 | 94 | +36% | 147 | 1.6× |

New workloads (the §4/§6 benchmark-hygiene gaps, all paired + cross-checked):

| Mode | Workload | Remu | Unicorn | Standing |
|---|---|---:|---:|---|
| 32 | tight_loop_pg | 152 | 896 | paging costs Remu 17% (TLB-peek probe + fetch translate) |
| 32 | mem_rw_pg | 97 | 200 | data-side TLB per access; QEMU folds paging into its TLB |
| 32 | tight_loop_r3 | 183 | 950 | ring 3 is free — protection is per-access compares |
| 32 | mem_rw_r3 | 137 | 198 | ditto |
| 32 | rep_movs | 0.36 s/run | 1.51 s/run | **Remu 4.2× faster** — the expected worst case is a win |

Targets vs outcome: `mem_rw` and `call_ret` improved beyond their targets —
Remu now beats Unicorn on **every memory-touching workload in every mode**
except 32-bit paged `mem_rw`, exactly the strategic goal of §7's roadmap.
The pure-ALU stretch targets (386 tight 350+/alu 200+; x86-64 150+/100+)
were not reached: ablation measurements put the per-step floor at ~25 host
cycles (step preamble ≈ 3.7, probe ≈ 5, state install + dispatch + arm ≈
16) — the icache's win scales with decode complexity, and `tight_loop`'s
1–2-byte instructions have almost no decode to save. The stage-5
micro-trace criterion (step preamble ≥ 20% of hit-path time AND tight <
400) evaluates **NO-GO** for this cycle: the preamble is only ~13%.
Closing the remaining pure-ALU gap needs per-*trace* amortization of the
probe + snapshot + event checks (the P4 `events_pending` fold and
`run(&mut bus, n)` entry point remain open alongside it), and beyond that
the P5 JIT.

---

## 8. P4 + JIT cycle (2026-07-12)

Cycle developed on branch `enhancing-performance`. Two efforts: **P4**, the
interpreter-tier cleanups §4 named (events fold, `run(n)`, `#[cold]` paths),
applied to *both* the 386 and x86-64 cores; and a **feature-gated template
JIT** (`--features jit`, `src/x86_64/jit/`) for the x86-64 core — the §4 P5
item, off by default with the interpreter kept as its reference and fallback.

Correction to §7's framing: 32-bit **non-paged** `mem_rw` is also a Unicorn
win (135 vs 199), not just the paged variant. "Beats Unicorn on every memory
workload" holds for 16-bit and 64-bit modes and REP MOVS; in 32-bit, Remu
wins `call_ret` and `rep_movs` but not `mem_rw`.

### 8.1 P4 — events fold + run(n)

The per-step preamble (5 event branches + an unconditional inhibit clear + a
TF read) collapses to one `boundary_pending()` predicate over an `events: u8`
bitmask; `step()` splits into an inline fast path and a `#[cold]` slow path,
with the fault rewind in a shared `#[cold]` epilogue. `run(&mut bus, n)`
retires a batch and returns `RunExit::{Completed,HostTrap,Halted,Shutdown}`;
os/os64 drive it (usermode stays on `step_one` — its segv latch is bus-side).
A per-iteration `host_trap` probe cost 2–5%, so a recorded trap now rings an
`EVT_HOST_TRAP` doorbell bit and the run loop tests one predicate.

Criterion (Melem/s = MIPS), interpreter, vs the post-P3 baseline:

| Core | Workload | Before | After | Δ |
|---|---|---:|---:|---:|
| 386 | tight_loop | 184 | 193 | +5.0% |
| 386 | alu (arith) | 129 | 133 | +2.9% |
| 386 | call_ret | 154 | 160 | +3.6% |
| x64 | tight_loop | 102 | 105 | +2.6% |
| x64 | alu (arith) | 95 | 96.6 | +1.6% |
| x64 | call_ret | 95.5 | 99.5 | +4.3% |

Short-circuit `||` in `boundary_pending()` is load-bearing — a fused bitwise
`|` regressed 386 tight_loop 8% (the branch stalls on all four loads). The
cycles-batching experiment (§4 commit 4) was measured and **rejected**: no
gain on tight/call, −4–5% on alu_run (extra local-accumulator bookkeeping),
and it would quantize guest RDTSC to block boundaries.

### 8.2 JIT — stages A + B (x86-64, `--features jit`)

dynasm-rs template JIT reached through `run()`. Guest regs stay memory-
resident (`[rbp+off]`); guest flags ride the host EFLAGS and materialize into
`regs.rflags` only at block exits. Blocks are single-page, keyed
`phys|icache_ctx()`, and revalidate in their own prologue against the icache
write-stamp + O(1) inval clock — so guest code writes self-invalidate them.
A self-looping block bounds iterations with `loop` (flag-preserving) and
carries the `run(n)` budget in a host register, retiring exactly `n`. Cycle
weights mirror `exec_decoded`, keeping `Cpu::cycles` in lockstep.

- **Stage A**: MOV reg,imm; INC/DEC reg; Jcc; JMP rel; self-loop detection.
- **Stage B**: register ALU (ADD/OR/ADC/SBB/AND/SUB/XOR/CMP r/r + r/imm),
  immediate shifts/rotates, two-operand IMUL — gated by a per-flag exactness
  state machine. Host and guest disagree on *undefined* flags (AND/OR/XOR's
  AF, multi-bit shift/rotate OF, IMUL's SF/ZF/PF); the translator tracks each
  flag as exact/const/garbage and takes the longest block prefix ending with
  no garbage flag where every consumer reads only exact flags.

x86-64 MIPS (best of 5), interpreter (P4) vs JIT-on, vs Unicorn:

| Workload | Interp | JIT | Unicorn | JIT standing |
|---|---:|---:|---:|---|
| tight_loop | 105 | **10406** | 903 | **11.5× Unicorn** |
| alu_mix | 91 | **5205** | 1415 | **3.7× Unicorn** |
| mem_rw | 94 | 97.5 | 53 | **1.8× Unicorn** (not yet JIT'd) |
| call_ret | 95 | 97.7 | 148 | Unicorn 1.5× (not yet JIT'd) |

The two workloads Unicorn dominated (its TCG JIT vs an interpreter) are now
decisive Remu wins. `mem_rw`/`call_ret` are not yet translated (their blocks
contain memory and call/ret); they no longer *regress* under the feature
because untranslatable heads are marked with a version-tagged COLD sentinel
in the block table (no per-instruction hash lookup on the fallback path).

Differential net (`tests/x86_64_jit_tests.rs`): chunked lockstep of `run()`
(JIT) vs `step()` (interpreter) over prime chunk sizes asserts identical
registers, **cycles** and RAM; covers the exact alu_mix body, an ALU/shift
sweep, 32/64-bit zero-extension, and SMC invalidation. Feature off = byte-
identical to before; all suites green both ways incl. MOO 80386 (1.76M).

### 8.3 Open (next cycle)

- **JIT stage C — block chaining**: patch direct-branch exits to jump
  straight to the successor block instead of returning to the dispatcher
  (translation-time chaining for constant targets first; backpatching via
  `Assembler::alter` after an API spike).
- **JIT stage D — memory + stack ops**: `MOV`/ALU with memory operands,
  PUSH/POP, CALL/RET via `extern "win64"` helpers wrapping the interpreter's
  `read*/write*/push*/pop*` (per-`Bus` monomorphization, `TypeId` flush);
  faults side-exit and re-execute in the interpreter. Unlocks JIT for
  `mem_rw` and `call_ret` (the last Unicorn loss).
- ~~**386 JIT port** (same infra)~~ — **done, see §9.** **usermode segv →
  CPU-visible exception** (so usermode can adopt `run(n)`) remains open.

---

## 9. 386 JIT port — stages A + B (2026-07-12, same machine)

The x86-64 template JIT was ported to the 80386 core on branch
`enhance-remaining-cpus`, reusing the whole infrastructure: the direct-mapped
block table keyed `phys | icache_ctx()`, the hot-counter/`HOT_THRESHOLD`
warm-up, the SMC write-stamp + O(1) invalidation-clock prologue revalidation,
the `loop`-bounded self-loop with the rel8 bounce trampoline, and the flag
materialization via `LAHF`/`SETO`. The host is still x86-64 (the gate is
`all(feature = "jit", target_arch = "x86_64")`), so a 32-bit guest instruction
maps to the equivalent 32-bit host instruction. Lives in `src/x86_32/jit/`;
reached through `Cpu::run`, interpreter kept as the reference and fallback.

Deltas from the x86-64 backend, all mechanical:

- **32-bit only.** Guest GPRs are `[u32; 8]` (stride 4, no high-dword zeroing);
  every operand is `DWORD`. The `o64` paths are dropped.
- **Gated on CS.D = 1.** Only 32-bit code segments are translated, so EIP is a
  full 32-bit offset and near-branch targets wrap mod 2^32 to match the emitted
  host arithmetic; 16-bit segments and V86 fall back to the interpreter (the
  block key already folds CS.D/CR0.PE/EFLAGS.VM/`extensions`).
- **Cycle-exact by construction.** The 386's register-form `exec_decoded`
  weights (MOV/INC/DEC/ALU = 2, shift = 3, IMUL = 20, Jcc 7/3, JMP = 7) equal
  the x86-64 core's, so the `CYC_*` table transfers verbatim. Any legacy prefix
  costs the interpreter one cycle/byte, which the block does not model, so
  prefixed forms are refused (`npfx() == 0`) — trivially satisfied by the
  prefix-free 32-bit hot set.
- **Flag exactness unchanged.** The 386 interpreter clears AF on logic ops and
  shifts, clears OF on SAR, and (unlike the host) *computes* SF/ZF/PF for
  two-operand IMUL — but the host leaves those undefined, so marking them
  `garbage` is both correct and identical to the x86-64 table. `TEST` (AND
  without write-back) and `CMP` emit host `test`/`cmp` (no destination write).

Translated set (stage A + B, register operands only): `MOV r,imm`,
`MOV r/m,imm` (reg form), `INC`/`DEC` (both the `40`–`4F` short forms and the
`FF /0,/1` group form), register ALU (`ADD/OR/ADC/SBB/AND/SUB/XOR/CMP` r,r and
r,imm; `TEST`), immediate shifts/rotates (SHL/SHR/SAR/ROL/ROR; RCL/RCR
excluded), two-operand `IMUL`, `Jcc`, `JMP rel`, and self-loop detection.

**Correctness.** `tests/x86_32_jit_tests.rs` — 10 differential tests chunk
`run()` (JIT) against `step()` (interpreter) at prime chunk sizes and assert
identical registers, **cycles**, and RAM: tight loops (both DEC forms), the
`alu_mix` body, an ALU/shift sweep, shift-by-1/SAR (OF-exact + OF-cleared
paths), TEST/CMP no-write-back, a >127-byte self-loop (bounce trampoline), and
SMC invalidation. Feature off = byte-identical to before. Full suite green both
ways (`cargo test` and `cargo test --features jit`, incl. the existing 58 386
core tests and 9 x86-64 JIT tests); the interpreter path is untouched, so the
80386 MOO conformance suite (which drives `step()`) is unaffected.

### MIPS, best of 5 — 386 interpreter vs JIT (same machine, ring 0, paging off)

| Workload | Interp | JIT | Δ | Unicorn | JIT standing |
|---|---:|---:|---:|---:|---|
| tight_loop | 188 | **10176** | **54×** | 889 | **11.4× Unicorn** |
| alu_mix | 126 | **5233** | **42×** | 1407 | **3.7× Unicorn** |
| mem_rw | 140 | 108 | −22% | 199 | Unicorn 1.8× (not yet JIT'd) |
| call_ret | 157 | 104 | −34% | 133 | Unicorn 1.3× (not yet JIT'd) |

The two compute-bound workloads reach x86-64-JIT parity (x86-64 measured
10406 / 5205 on the same rig) and decisively beat Unicorn's TCG. The 386 core
now stands exactly where the x86-64 core does after stage B: pure-register
loops are the JIT's, memory/call loops are still the interpreter's.

**The `mem_rw`/`call_ret` regression is real and understood.** Those loops are
dominated by memory and CALL/RET operands, which stage A+B cannot translate, so
they run on the interpreter *through* the `run_jit` fallback — which pays a
per-instruction dispatch probe (CS.D test, `icache_phys`, key, direct-mapped
table lookup) that plain `run_interp` does not. On the x86-64 core this probe
is hidden under a slower interpreter (its `mem_rw`/`call_ret` even improved
slightly under the feature); the 386 interpreter retires those workloads ~1.5×
faster per instruction, so the same fixed probe cost surfaces as a 22–34% loss.
Marking untranslatable heads COLD (a version-tagged sentinel — already ported)
removes the *retranslation* cost but not the probe itself. This is inherent to
stage A+B and is exactly what **stage D (memory + stack ops)** removes: once the
memory MOVs and CALL/RET are translated, those loops stop hitting the fallback
at all. The JIT is off by default, so nothing regresses unless `--features jit`
is set on memory-heavy 32-bit code.

### Open (386-specific, next cycle)

- **Stage D on the 386** (memory + stack ops via `extern "win64"` helpers
  wrapping `read*/write*/push*/pop*`) — turns `mem_rw`/`call_ret` from a
  fallback regression into a translated win, as on the x86-64 roadmap.
- **Register-register `MOV`** (`MovRmRW`/`MovRRmW`) is a cheap stage-C-adjacent
  add that would shrink the `mem_rw` fallback fragment.
- **Stage C block chaining** and the shared **usermode `run(n)`** adoption apply
  to both cores.
