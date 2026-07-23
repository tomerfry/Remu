//! Feature-gated template JIT (dynamic binary translation) for the 80386
//! core, reached through [`Cpu::run`]. It mirrors the x86-64 core's JIT: the
//! interpreter stays the reference, JIT'd blocks reproduce its architectural
//! effects exactly, and anything a block cannot represent falls back to a
//! single interpreter step.
//!
//! ## Model
//!
//! - Guest registers stay memory-resident in `regs.gpr` (32-bit); emitted
//!   code addresses them as `[rbp + OFF_GPR + 4*i]` (rbp pinned to `*mut
//!   Cpu`). No register allocator — every side-exit is trivially
//!   state-correct.
//! - Guest status flags live in the host EFLAGS within a block and are
//!   materialized into `regs.eflags` only at block exits, using the last
//!   flag-writer's *defined-flags* mask (so `INC`/`DEC`, which preserve CF,
//!   leave the guest CF untouched).
//! - Only **32-bit code segments** (CS.D = 1) are translated. EIP is then a
//!   full 32-bit offset and near-branch targets wrap mod 2^32, matching the
//!   emitted host arithmetic; 16-bit segments and V86 fall back to the
//!   interpreter. The block key folds in the icache context (CS.D, CR0.PE,
//!   EFLAGS.VM, `extensions`), so a block never outlives its mode.
//! - Blocks are single physical page, keyed `phys | icache_ctx()`, and
//!   revalidated in their own prologue against the icache write-stamp and
//!   the O(1) invalidation clock — the same SMC machinery the decoded-insn
//!   cache uses. A stale block self-detects and yields to the dispatcher.
//! - The budget (`Cpu::run`'s `n`) rides in a host register; a self-looping
//!   block bounds its iterations with `loop` (flag-preserving) so `run(n)`
//!   still retires exactly `n` instructions.

#![cfg(all(feature = "jit", target_arch = "x86_64"))]

use core::mem::offset_of;

use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynasmApi};

use super::icache::{ICache, STAMP_SLOTS};
use super::{Bus, Cpu, Registers, RunExit, RunResult, reg};

mod emit;

/// Table `idx` sentinel marking a head known to be untranslatable.
const COLD: u32 = u32::MAX;

/// Result of a direct-mapped block-table probe.
enum Lookup {
    /// A valid translated block at this index.
    Hit(usize),
    /// A head recorded as untranslatable (interpret it).
    Cold,
    /// No valid entry — evaluate/translate.
    Miss,
}

// --- Guest-state field offsets from the `*mut Cpu` base (rbp) ---------------
// Computed from the real layout so emitted code never bakes an absolute
// address (a `Cpu` may move). `regs`/`icache` are private to the parent
// module, which this submodule may name.

/// Offset of `regs.gpr[0]` (the register file is `[u32; 8]`, stride 4).
pub(crate) const OFF_GPR: i32 = (offset_of!(Cpu, regs) + offset_of!(Registers, gpr)) as i32;
/// Offset of `regs.eip` (a `u32`).
pub(crate) const OFF_EIP: i32 = (offset_of!(Cpu, regs) + offset_of!(Registers, eip)) as i32;
/// Offset of `regs.eflags` (an `EFlags` newtype over `u32`).
pub(crate) const OFF_EFLAGS: i32 = (offset_of!(Cpu, regs) + offset_of!(Registers, eflags)) as i32;
/// Offset of `cycles`.
pub(crate) const OFF_CYCLES: i32 = offset_of!(Cpu, cycles) as i32;
/// Offset of the `icache.stamps` boxed-array pointer.
pub(crate) const OFF_STAMPS: i32 = (offset_of!(Cpu, icache) + offset_of!(ICache, stamps)) as i32;
/// Offset of `icache.inval`.
pub(crate) const OFF_INVAL: i32 = (offset_of!(Cpu, icache) + offset_of!(ICache, inval)) as i32;

// --- Cache sizing -----------------------------------------------------------

/// Direct-mapped block table entries (keyed like the decoded-insn cache).
const JIT_SLOTS: usize = 8192;
/// Untagged saturating hot counters; aliasing only over/under-warms.
const HOT_SLOTS: usize = 4096;
/// Executions of a guest key before it is translated.
const HOT_THRESHOLD: u8 = 8;
/// Rebuild the whole cache once this many blocks accumulate.
const MAX_BLOCKS: usize = 4096;
/// …or this many bytes of emitted code.
const MAX_CODE_BYTES: usize = 8 << 20;

/// One direct-mapped table slot; `key == 0` is empty.
#[derive(Clone, Copy)]
struct Slot {
    key: u64,
    version: u64,
    idx: u32,
}

/// A translated block. `key`/`version`/`stamp_slot` are retained for
/// cross-block chaining and debugging in later stages; the base stage
/// validates through the table [`Slot`], the block's own prologue, and the
/// dispatcher's `start_eip` guard.
#[allow(dead_code)]
struct BlockMeta {
    /// `phys | icache_ctx()` of the first instruction.
    key: u64,
    /// `icache.clock` at translation time (SMC/inval validity floor).
    version: u64,
    /// Physical-page write-stamp slot the prologue checks.
    stamp_slot: usize,
    /// Entry point in the executable buffer.
    entry: AssemblyOffset,
    /// Guest EIP the block starts at.
    start_eip: u32,
    /// Instructions retired by one full traversal (the run() budget unit).
    ninsns: u32,
}

/// Per-CPU JIT state. Cloning yields a fresh empty cache (the cache is a pure
/// accelerator keyed by the icache clocks, so a clone starts cold correctly).
pub(crate) struct JitState {
    asm: Assembler,
    blocks: Vec<BlockMeta>,
    table: Box<[Slot; JIT_SLOTS]>,
    hot: Box<[u8; HOT_SLOTS]>,
}

impl JitState {
    pub(crate) fn new() -> Self {
        JitState {
            asm: Assembler::new().expect("allocate JIT assembler"),
            blocks: Vec::new(),
            table: Box::new(
                [Slot {
                    key: 0,
                    version: 0,
                    idx: 0,
                }; JIT_SLOTS],
            ),
            hot: Box::new([0u8; HOT_SLOTS]),
        }
    }

    /// Drop every block and reset to an empty cache.
    pub(super) fn flush(&mut self) {
        *self = JitState::new();
    }
}

impl Clone for JitState {
    fn clone(&self) -> Self {
        JitState::new()
    }
}

impl core::fmt::Debug for JitState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("JitState")
            .field("blocks", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

impl Cpu {
    /// Flush the JIT cache. Public counterpart to [`Cpu::invalidate_icache`]
    /// for hosts that write guest code memory behind the CPU's back; the
    /// per-page stamps already cover CPU-performed stores.
    pub fn invalidate_jit(&mut self) {
        self.jit.flush();
    }

    /// [`Cpu::run`] with the JIT fast path. Falls back to a single
    /// interpreter step for every boundary, non-32-bit segment, cold code,
    /// and anything a block cannot yet represent — so its observable behavior
    /// is identical to the interpreter `run`.
    pub(crate) fn run_jit<B: Bus>(&mut self, bus: &mut B, n: u64) -> RunResult {
        if self.host_trap.is_some() {
            return RunResult {
                executed: 0,
                exit: RunExit::HostTrap,
            };
        }
        let mut executed = 0u64;
        while executed < n {
            if self.boundary_pending() {
                if let Some(exit) = self.run_boundary(bus) {
                    return RunResult { executed, exit };
                }
                executed += 1;
                continue;
            }
            // The template JIT only translates 32-bit code segments (CS.D =
            // 1); everything else interprets. See the module docs.
            if !self.regs.seg[reg::CS as usize].db() {
                self.step_fast(bus);
                executed += 1;
                continue;
            }
            let Some(phys) = self.icache_phys() else {
                // Fetch translation cold — one step warms it.
                self.step_fast(bus);
                executed += 1;
                continue;
            };
            let key = phys as u64 | self.icache_ctx();
            let budget = n - executed;
            match self.jit_lookup(key, phys) {
                // The `start_eip` guard: the key is physical, but exits bake
                // absolute EIP constants, so the same physical code reached at
                // a different EIP (a CS-base change, or two linear pages
                // mapped to one frame) must not enter this block.
                Lookup::Hit(idx)
                    if self.jit.blocks[idx].start_eip == self.regs.eip
                        && self.jit.blocks[idx].ninsns as u64 <= budget =>
                {
                    let retired = self.jit_enter(idx, budget);
                    if retired == 0 {
                        // Stale prologue exit: drop the block and step once so
                        // the loop makes progress (a fresh block re-translates).
                        self.jit_evict(phys);
                        self.step_fast(bus);
                        executed += 1;
                    } else {
                        executed += retired;
                    }
                }
                // A known-untranslatable head, or a block too large for the
                // remaining budget: interpret one instruction.
                Lookup::Cold | Lookup::Hit(_) => {
                    self.step_fast(bus);
                    executed += 1;
                }
                Lookup::Miss => {
                    if self.jit_should_translate(bus, key, phys) {
                        // Re-enter the loop; the freshly filled block hits next.
                        continue;
                    }
                    self.step_fast(bus);
                    executed += 1;
                }
            }
        }
        RunResult {
            executed,
            exit: RunExit::Completed,
        }
    }

    /// Probe the direct-mapped table for `key`, revalidating against the
    /// icache write-stamp and invalidation clock. A [`COLD`] sentinel marks a
    /// head known to be untranslatable, so cold code needs no hash lookup on
    /// the fallback path.
    fn jit_lookup(&self, key: u64, phys: u32) -> Lookup {
        let slot = &self.jit.table[phys as usize & (JIT_SLOTS - 1)];
        if slot.key != key {
            return Lookup::Miss;
        }
        let stamp = self.icache_stamp(phys);
        if slot.version < stamp || slot.version < self.icache.inval {
            return Lookup::Miss; // stale: re-evaluate (code may have changed)
        }
        if slot.idx == COLD {
            Lookup::Cold
        } else {
            Lookup::Hit(slot.idx as usize)
        }
    }

    /// Enter block `idx` with `budget` instructions available; returns the
    /// number retired (0 means the prologue found the block stale).
    fn jit_enter(&mut self, idx: usize, budget: u64) -> u64 {
        let entry = self.jit.blocks[idx].entry;
        // The executable buffer is not modified during the call (no
        // reentrancy), so the raw pointer stays valid after the lock drops.
        let ptr = {
            let reader = self.jit.asm.reader();
            let buf = reader.lock();
            buf.ptr(entry) as usize
        };
        let f: extern "win64" fn(*mut Cpu, u64) -> u64 = unsafe { core::mem::transmute(ptr) };
        let cpu = self as *mut Cpu;
        f(cpu, budget)
    }

    /// Warm `key`'s hot counter and, once hot, translate it. Returns whether a
    /// block was installed (so the caller re-probes instead of stepping).
    fn jit_should_translate<B: Bus>(&mut self, bus: &mut B, key: u64, phys: u32) -> bool {
        let h = &mut self.jit.hot[phys as usize & (HOT_SLOTS - 1)];
        *h = h.saturating_add(1);
        if *h < HOT_THRESHOLD {
            return false;
        }
        if self.jit.blocks.len() >= MAX_BLOCKS || self.jit.asm.offset().0 >= MAX_CODE_BYTES {
            self.jit.flush();
        }
        self.jit_translate(bus, key, phys)
    }

    /// Mark `key` as untranslatable in the direct-mapped table so the fallback
    /// path recognizes it without a hash lookup. Version-tagged, so a later
    /// guest write to the page (bumping the stamp) re-opens translation.
    pub(super) fn jit_mark_cold(&mut self, key: u64, phys: u32) {
        let version = self.icache.clock;
        self.jit.table[phys as usize & (JIT_SLOTS - 1)] = Slot {
            key,
            version,
            idx: COLD,
        };
    }

    /// Drop the block at `phys`'s direct-mapped slot (its code stays in the
    /// buffer until the next flush). O(1): a physical address maps to exactly
    /// one slot.
    fn jit_evict(&mut self, phys: u32) {
        self.jit.table[phys as usize & (JIT_SLOTS - 1)].key = 0;
    }

    /// Read the per-page write-stamp the way the icache does.
    #[inline]
    fn icache_stamp(&self, phys: u32) -> u64 {
        self.icache.stamps[(phys >> 12) as usize & (STAMP_SLOTS - 1)]
    }
}
