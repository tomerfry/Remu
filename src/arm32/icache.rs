//! Direct-mapped decoded-instruction cache — the ARM32 port of
//! `x86_32/icache.rs`; see that file for the design. ARM's fixed-width,
//! aligned instructions simplify every rule:
//!
//! - The probe address is the (physical) PC itself — this core has no MMU,
//!   so there is no fetch-translation layer to consult and a probe can
//!   never miss for translation reasons.
//! - The context key is a single bit: ARM vs Thumb state (bit 63; the
//!   decode of a word depends on nothing else).
//! - Instructions can never cross a 4 KiB page (they are 4- or 2-byte
//!   aligned), so everything is cacheable and one write stamp always
//!   covers the whole instruction.
//! - Entries are indexed by `pc >> 1` (the finest instruction alignment),
//!   giving a 32 KiB contiguous code window.
//!
//! Write stamps make self-modifying code exact: every guest store bumps a
//! monotonic clock into a per-page slot, and an entry only hits while its
//! fill-time clock is at least as new. Host-side writes to guest memory
//! bypass the CPU — hosts must call [`Cpu::invalidate_icache`] (O(1))
//! before stepping again, exactly as on the x86 cores.

use std::alloc::{Layout, alloc_zeroed, handle_alloc_error};
use std::fmt;

use super::Cpu;
use super::decode::DecodedInsn;

/// Number of direct-mapped entries, indexed by `pc >> 1`
/// (16384 × 32 B = 512 KiB, a 32 KiB contiguous code window).
pub(crate) const ICACHE_ENTRIES: usize = 16384;

/// Per-page write-stamp slots ((addr >> 12) & mask); pages 64 MiB apart
/// share a slot — aliasing only causes spurious re-decodes, never a stale
/// hit.
pub(crate) const STAMP_SLOTS: usize = 16384;

/// One cache slot; the all-zero pattern is a dead entry (version 0 is
/// below the initial invalidation clock).
#[derive(Clone, Copy)]
pub(crate) struct Entry {
    /// Instruction address | Thumb bit (bit 63).
    pub key: u64,
    /// Write-clock value at fill time; live while `>= stamps[page]` and
    /// `>= inval`.
    pub version: u64,
    /// The decode products.
    pub insn: DecodedInsn,
}

/// The cache proper, hanging off [`Cpu`] (never the bus: memory may be
/// shared or reseeded while the CPU is rebuilt).
#[derive(Clone)]
pub(crate) struct ICache {
    /// Direct-mapped entries. Fixed-size so a masked index is provably in
    /// bounds.
    pub entries: Box<[Entry; ICACHE_ENTRIES]>,
    /// Per-page write stamps: clock value of the last store into any
    /// aliasing page.
    pub stamps: Box<[u64; STAMP_SLOTS]>,
    /// Monotonic store clock; starts at 1 so zeroed entries are dead.
    pub clock: u64,
    /// Entries with `version < inval` are dead (O(1) invalidate-all).
    pub inval: u64,

    // Probe/fill accounting for the exactness tests (compiled out of
    // release library builds).
    #[cfg(any(test, debug_assertions))]
    pub hits: u64,
    #[cfg(any(test, debug_assertions))]
    pub fills: u64,
}

/// Zeroed boxed array from the allocator (lazily-zeroed pages; safety:
/// only for types valid when all-zero — `Entry` and `u64`).
fn zeroed_array<T, const N: usize>() -> Box<[T; N]> {
    let layout = Layout::new::<[T; N]>();
    unsafe {
        let p = alloc_zeroed(layout) as *mut [T; N];
        if p.is_null() {
            handle_alloc_error(layout);
        }
        Box::from_raw(p)
    }
}

impl ICache {
    pub fn new() -> Self {
        ICache {
            entries: zeroed_array(),
            stamps: zeroed_array(),
            clock: 1,
            inval: 1,
            #[cfg(any(test, debug_assertions))]
            hits: 0,
            #[cfg(any(test, debug_assertions))]
            fills: 0,
        }
    }

    /// Record a guest store to `addr`.
    #[inline(always)]
    pub fn stamp_write(&mut self, addr: u32) {
        self.clock += 1;
        self.stamps[(addr >> 12) as usize & (STAMP_SLOTS - 1)] = self.clock;
    }

    /// Kill every current entry in O(1).
    #[inline]
    pub fn invalidate_all(&mut self) {
        self.clock += 1;
        self.inval = self.clock;
    }
}

impl fmt::Debug for ICache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ICache")
            .field("clock", &self.clock)
            .field("inval", &self.inval)
            .finish_non_exhaustive()
    }
}

impl Cpu {
    /// Install a freshly decoded instruction at `pc` (every ARM32
    /// instruction is cacheable — none can cross a page).
    #[inline]
    pub(crate) fn icache_fill(&mut self, key: u64, slot: usize, d: &DecodedInsn) {
        self.icache.entries[slot] = Entry {
            key,
            version: self.icache.clock,
            insn: *d,
        };
        #[cfg(any(test, debug_assertions))]
        {
            self.icache.fills += 1;
        }
        self.stat(|s| s.icache_fills += 1);
    }

    /// Invalidate the decoded-instruction cache (O(1)). Required after any
    /// host-side write to guest memory that may contain code — the CPU
    /// only stamps stores it performs itself.
    #[inline]
    pub fn invalidate_icache(&mut self) {
        self.icache.invalidate_all();
    }

    /// Debug builds re-decode every cache hit from the bus and compare —
    /// a mismatch means a store went unstamped or a key bit is missing.
    #[cfg(debug_assertions)]
    pub(crate) fn icache_differential<B: super::Bus>(&mut self, bus: &mut B, cached: &DecodedInsn) {
        let fresh = if self.thumb {
            super::decode::decode_thumb(bus.read16(self.start_pc))
        } else {
            super::decode::decode_arm(bus.read32(self.start_pc))
        };
        assert_eq!(
            fresh, *cached,
            "icache hit does not match a fresh decode at {:#x}",
            self.start_pc
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Cpu, LinearMemory};

    fn run_until_swi(cpu: &mut Cpu, mem: &mut LinearMemory, cap: u32) {
        cpu.trap_swi = true;
        for _ in 0..cap {
            if cpu.host_trap.is_some() {
                return;
            }
            cpu.step(mem);
        }
        panic!("did not reach SWI");
    }

    /// Same-page SMC: a loop patches its ADD into a SUB between
    /// iterations; the stamps must force a re-decode.
    #[test]
    fn smc_same_page_repatches() {
        #[rustfmt::skip]
        let program: [u32; 9] = [
            0xE3A0_0002, // 00: MOV r0, #2
            0xE3A0_1002, // 04: MOV r1, #2
            0xE280_0001, // 08: ADD r0, r0, #1        <- patched to SUB
            0xE59F_200C, // 0C: LDR r2, [pc, #12]     ; pool at 0x0C+8+12 = 0x20
            0xE50F_2010, // 10: STR r2, [pc, #-16]    ; patches 0x10+8-16 = 0x08
            0xE251_1001, // 14: SUBS r1, r1, #1
            0x1AFF_FFFA, // 18: BNE 0x08              ; 0x08 - (0x18+8) = -24
            0xEF00_0000, // 1C: SWI 0
            0xE240_0001, // 20: .word SUB r0, r0, #1
        ];
        let mut mem = LinearMemory::new();
        for (i, w) in program.iter().enumerate() {
            mem.load(i as u32 * 4, &w.to_le_bytes());
        }
        let mut cpu = Cpu::new();
        run_until_swi(&mut cpu, &mut mem, 100);
        // Pass 1: ADD (3), pass 2: SUB (2). A stale hit would give 4.
        assert_eq!(cpu.regs.gpr[0], 2);
    }

    /// The public O(1) invalidate-all covers host-side writes.
    #[test]
    fn invalidate_icache_after_host_write() {
        let mut mem = LinearMemory::new();
        mem.load(0, &0xE3A0_0001u32.to_le_bytes()); // MOV r0, #1
        mem.load(4, &0xEF00_0000u32.to_le_bytes()); // SWI 0
        let mut cpu = Cpu::new();
        run_until_swi(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.gpr[0], 1);

        mem.load(0, &0xE3A0_0002u32.to_le_bytes()); // MOV r0, #2
        cpu.invalidate_icache();
        cpu.host_trap = None;
        cpu.regs.gpr[15] = 0;
        run_until_swi(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.gpr[0], 2);
    }

    /// ARM and Thumb entries for the same address must not alias: the key
    /// carries the state bit.
    #[test]
    fn arm_thumb_context_keyed() {
        let mut mem = LinearMemory::new();
        // At 0x100: as ARM, MOV r0, #1; as Thumb halves, the same bytes
        // decode differently — run ARM first, then jump here in Thumb.
        mem.load(0x100, &0xE3A0_0001u32.to_le_bytes()); // ARM: MOV r0, #1
        mem.load(0x104, &0xEF00_0000u32.to_le_bytes()); // ARM: SWI
        let mut cpu = Cpu::new();
        cpu.trap_swi = true;
        cpu.regs.gpr[15] = 0x100;
        cpu.step(&mut mem);
        assert_eq!(cpu.regs.gpr[0], 1);
        cpu.step(&mut mem);
        assert!(cpu.host_trap.take().is_some());

        // Same address, Thumb state: 0x0001 = MOVS r1, r0 (format 1 LSL#0),
        // must decode as Thumb, not replay the cached ARM MOV.
        use crate::arm32::registers::psr;
        cpu.regs.cpsr.set(psr::T, true);
        cpu.regs.gpr[15] = 0x100;
        cpu.regs.gpr[0] = 7;
        cpu.regs.gpr[1] = 0;
        cpu.step(&mut mem);
        assert_eq!(cpu.regs.gpr[1], 7, "Thumb decode of the same bytes");
    }

    /// Cycle parity: pass 1 (fills) and pass 2 (hits) of the same program
    /// return identical per-step cycle counts.
    #[test]
    fn cycle_parity_two_pass() {
        #[rustfmt::skip]
        let program: [u32; 7] = [
            0xE3A0_0005, // 40: MOV r0, #5
            0xE3A0_2A02, // 44: MOV r2, #0x2000 (off the code page, so the
                         //     store stamps do not evict the loop itself)
            0xE582_0000, // 48: STR r0, [r2]
            0xE592_1000, // 4C: LDR r1, [r2]
            0xE250_0001, // 50: SUBS r0, r0, #1
            0x1AFF_FFFB, // 54: BNE 0x48
            0xEF00_0000, // 58: SWI 0
        ];
        let mut mem = LinearMemory::new();
        for (i, w) in program.iter().enumerate() {
            mem.load(0x40 + i as u32 * 4, &w.to_le_bytes());
        }
        let mut cpu = Cpu::new();
        cpu.trap_swi = true;

        let run = |cpu: &mut Cpu, mem: &mut LinearMemory| {
            cpu.regs.gpr[15] = 0x40;
            cpu.host_trap = None;
            let mut cycles = Vec::new();
            for _ in 0..300 {
                if cpu.host_trap.is_some() {
                    break;
                }
                cycles.push(cpu.step(mem));
            }
            cycles
        };
        let pass1 = run(&mut cpu, &mut mem);
        let pass2 = run(&mut cpu, &mut mem);
        assert!(cpu.icache.hits > 0, "second pass must hit");
        assert_eq!(pass1, pass2, "cold and warm cycle counts must match");
    }
}
