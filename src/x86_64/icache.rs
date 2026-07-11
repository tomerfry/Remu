//! Direct-mapped decoded-instruction cache — the x86-64 port of
//! `x86_32/icache.rs`; see that file for the design. Differences:
//!
//! - The probe's physical address comes from the P2 fetch-translation
//!   cache (`fetch_tag`/`fetch_page`) when paging is on, inheriting its
//!   NX/permission discipline and its invalidation (every TLB flush,
//!   `INVLPG` and `prepare_cold_write` clears it); paging off is identity.
//! - The context key bits are CR0.PE, RFLAGS.VM, `mode64()` and CS.D
//!   (bits 60–63; physical addresses are ≤ 2^52).
//! - Per-hit revalidation is canonicality in 64-bit mode (page-aligned
//!   boundary, so the first byte covers the page-local entry) and the CS
//!   limit in legacy/compat modes.

use std::alloc::{Layout, alloc_zeroed, handle_alloc_error};
use std::fmt;

use super::Cpu;
use super::decode::DecodedInsn;
use super::registers::{RFlags, cr0, reg};

/// Number of direct-mapped entries, byte-indexed by physical address
/// (16384 × 40 B = 640 KiB, a 16 KiB contiguous code window).
pub(crate) const ICACHE_ENTRIES: usize = 16384;

/// Per-page write-stamp slots ((phys >> 12) & mask); aliasing only causes
/// spurious re-decodes, never a stale hit.
pub(crate) const STAMP_SLOTS: usize = 16384;

/// One cache slot; the all-zero pattern is a dead entry (version 0 is
/// below the initial invalidation clock).
#[derive(Clone, Copy)]
pub(crate) struct Entry {
    /// Physical address of the first instruction byte | context bits.
    pub key: u64,
    /// Write-clock value at fill time.
    pub version: u64,
    /// The decode products.
    pub insn: DecodedInsn,
}

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

    #[cfg(any(test, debug_assertions))]
    pub hits: u64,
    #[cfg(any(test, debug_assertions))]
    pub fills: u64,
    #[cfg(any(test, debug_assertions))]
    pub uncacheable: u64,
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
            #[cfg(any(test, debug_assertions))]
            uncacheable: 0,
        }
    }

    /// Record a guest store to physical address `phys`.
    #[inline(always)]
    pub fn stamp_write(&mut self, phys: u64) {
        self.clock += 1;
        self.stamps[(phys >> 12) as usize & (STAMP_SLOTS - 1)] = self.clock;
    }

    /// Record a store spanning `first..=last` (physical).
    #[inline(always)]
    pub fn stamp_write_span(&mut self, first: u64, last: u64) {
        self.stamp_write(first);
        if first >> 12 != last >> 12 {
            self.stamp_write(last);
        }
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
    /// Decode-context bits folded into an entry's key (bits 60–63): pure
    /// bit moves of CR0.PE (bit 0 → 60), RFLAGS.VM (bit 17 → 61),
    /// `mode64()` (→ 62) and CS.D (attrs bit 10 → 63).
    /// The caller (`exec_one`) refreshes `self.m64` before probing.
    #[inline(always)]
    pub(crate) fn icache_ctx(&self) -> u64 {
        ((self.regs.cr0 & cr0::PE as u64) << 60)
            | ((self.regs.rflags.bits() as u64 & RFlags::VM.bits() as u64) << 44)
            | ((self.m64 as u64) << 62)
            | ((self.regs.seg[reg::CS as usize].attrs as u64 & 0x400) << 53)
    }

    /// Physical address of the byte at CS:RIP, if it can be determined
    /// without faulting or bus access: identity with paging off, the P2
    /// fetch-translation cache otherwise (a stale or empty fetch tag
    /// returns `None`; the fused fetch will translate and refill it).
    #[inline(always)]
    pub(crate) fn icache_phys(&self) -> Option<u64> {
        let lin = self.regs.seg[reg::CS as usize]
            .base
            .wrapping_add(self.regs.rip);
        let lin = if self.m64 { lin } else { lin & 0xFFFF_FFFF };
        if !self.paging() {
            return Some(lin);
        }
        if lin >> 12 == self.fetch_tag {
            Some(self.fetch_page | (lin & 0xFFF))
        } else {
            None
        }
    }

    /// Install a freshly decoded instruction at `phys` (its first byte).
    pub(crate) fn icache_fill(&mut self, phys: u64, d: &DecodedInsn) {
        // Never cache a page-crosser (see x86_32/icache.rs module docs).
        if (phys & 0xFFF) + d.len as u64 > 0x1000 {
            #[cfg(any(test, debug_assertions))]
            {
                self.icache.uncacheable += 1;
            }
            self.stat(|s| s.icache_uncacheable += 1);
            return;
        }
        let slot = phys as usize & (ICACHE_ENTRIES - 1);
        self.icache.entries[slot] = Entry {
            key: phys | self.icache_ctx(),
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
    /// host-side write to guest memory that may contain code — the CPU only
    /// stamps stores it performs itself.
    #[inline]
    pub fn invalidate_icache(&mut self) {
        self.icache.invalidate_all();
    }
}

#[cfg(test)]
mod tests {
    use super::super::registers::reg;
    use super::super::{Cpu, LinearMemory};

    fn run_until_halt(cpu: &mut Cpu, mem: &mut LinearMemory, cap: u32) {
        for _ in 0..cap {
            if cpu.halted || cpu.host_trap.is_some() {
                return;
            }
            cpu.step(mem);
        }
        panic!("did not halt");
    }

    /// Build a 4-level 4 KiB-page table set at `root` mapping exactly the
    /// page containing linear `va` to physical `pa` (present/RW/user, A+D
    /// preset).
    fn map_one(mem: &mut LinearMemory, root: u64, va: u64, pa: u64) {
        let (pdpt, pd, pt) = (root + 0x1000, root + 0x2000, root + 0x3000);
        mem.load(root + ((va >> 39) & 0x1FF) * 8, &(pdpt | 0x67).to_le_bytes());
        mem.load(pdpt + ((va >> 30) & 0x1FF) * 8, &(pd | 0x67).to_le_bytes());
        mem.load(pd + ((va >> 21) & 0x1FF) * 8, &(pt | 0x67).to_le_bytes());
        mem.load(pt + ((va >> 12) & 0x1FF) * 8, &((pa & !0xFFF) | 0x67).to_le_bytes());
    }

    /// (c) CR3 remap, same linear page to different physical frames: the
    /// physically-keyed entries must simply re-key — no flush anywhere —
    /// and the first mapping's entry must still hit when switching back.
    #[test]
    fn cr3_remap_rekeys_without_flush() {
        let mut mem = LinearMemory::new();
        let mut cpu = Cpu::new();
        cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);

        const VA: u64 = 0x20_0000;
        const P1: u64 = 0x30_0000;
        // Not 16 KiB-congruent with P1, so the two frames use different
        // direct-mapped slots and retention is observable.
        const P2: u64 = 0x31_1000;
        map_one(&mut mem, 0x8_0000, VA, P1);
        map_one(&mut mem, 0x9_0000, VA, P2);
        mem.load(P1, &[0x48, 0xFF, 0xC0, 0xF4]); // INC RAX; HLT
        mem.load(P2, &[0x48, 0xFF, 0xC8, 0xF4]); // DEC RAX; HLT
        cpu.invalidate_icache(); // host writes above

        // After a CR3 switch the fetch-translation cache is cold, so the
        // first run never probes; each phase runs the snippet twice (warm
        // the fetch tag, then fill or hit).
        let mut go = |cpu: &mut Cpu, mem: &mut LinearMemory| {
            cpu.regs.rip = VA;
            cpu.halted = false;
            run_until_halt(cpu, mem, 10);
        };

        cpu.regs.cr3 = 0x8_0000;
        cpu.invalidate_tlb();
        go(&mut cpu, &mut mem);
        go(&mut cpu, &mut mem); // fills the P1 entry
        assert_eq!(cpu.regs.gpr[0], 2, "first mapping runs INC twice");

        cpu.regs.cr3 = 0x9_0000;
        cpu.invalidate_tlb();
        go(&mut cpu, &mut mem);
        go(&mut cpu, &mut mem); // fills the P2 entry; a stale P1 hit would INC
        assert_eq!(cpu.regs.gpr[0], 0, "remapped page runs DEC, not stale INC");

        // Back to the first mapping: the physically-keyed P1 entry is
        // intact — the second run hits without a new fill.
        cpu.regs.cr3 = 0x8_0000;
        cpu.invalidate_tlb();
        go(&mut cpu, &mut mem);
        let (fills, hits) = (cpu.icache.fills, cpu.icache.hits);
        go(&mut cpu, &mut mem);
        assert_eq!(cpu.regs.gpr[0], 2);
        assert_eq!(cpu.icache.fills, fills, "P1 entry survived the remap");
        assert!(cpu.icache.hits > hits, "P1 entry hits after switching back");
    }

    /// (a) Same-page SMC through a RIP-relative guest store: a loop patches
    /// its INC into a DEC between iterations.
    #[test]
    fn smc_same_page_repatches_64() {
        #[rustfmt::skip]
        let program: &[u8] = &[
            0x48, 0xC7, 0xC0, 0x02, 0x00, 0x00, 0x00,       // 10000: MOV RAX, 2
            0x48, 0xC7, 0xC1, 0x02, 0x00, 0x00, 0x00,       // 10007: MOV RCX, 2
            0x48, 0xFF, 0xC0,                               // 1000E: INC RAX  <- patched
            0xC6, 0x05, 0xF8, 0xFF, 0xFF, 0xFF, 0xC8,       // 10011: MOV byte [RIP-8], 0xC8
            0x48, 0xFF, 0xC9,                               // 10018: DEC RCX
            0x75, 0xF1,                                     // 1001B: JNZ 0x1000E
            0xF4,                                           // 1001D: HLT
        ];
        let mut mem = LinearMemory::new();
        mem.load(0x1_0000, program);
        let mut cpu = Cpu::new();
        cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);
        run_until_halt(&mut cpu, &mut mem, 100);
        // Pass 1: INC (3), pass 2: DEC (2). A stale hit would give 4.
        assert_eq!(cpu.regs.gpr[0], 2);
    }

    /// (m) 67-prefixed instructions in 64-bit mode are never cached.
    #[test]
    fn asize_override_not_cached_in_64bit() {
        #[rustfmt::skip]
        let program: &[u8] = &[
            0x48, 0xC7, 0xC7, 0x00, 0x00, 0x02, 0x00, // MOV RDI, 0x20000
            0x67, 0x8B, 0x07,                         // MOV EAX, [EDI] (67-prefixed)
            0xF4,                                     // HLT
        ];
        let mut mem = LinearMemory::new();
        mem.load(0x1_0000, program);
        mem.load(0x2_0000, &[0x78, 0x56, 0x34, 0x12]);
        let mut cpu = Cpu::new();
        cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);
        cpu.step(&mut mem); // MOV RDI (fills)
        let fills = cpu.icache.fills;
        cpu.step(&mut mem); // the 67-prefixed MOV
        assert_eq!(cpu.regs.reg32(0), 0x1234_5678);
        assert_eq!(cpu.icache.fills, fills, "67-in-64-bit must not fill");
    }

    /// (g) Page-crossing instructions are never cached (paging off).
    #[test]
    fn page_crossing_instruction_not_cached_64() {
        let mut mem = LinearMemory::new();
        mem.load(0x1FFE, &[0xB8, 0x34, 0x12, 0xF4]); // real mode: MOV AX crosses 0x2000
        let mut cpu = Cpu::new();
        cpu.set_cs_ip(0x0000, 0x1FFE);
        let fills0 = cpu.icache.fills;
        let unc0 = cpu.icache.uncacheable;
        cpu.step(&mut mem);
        assert_eq!(cpu.regs.reg16(0), 0x1234);
        assert_eq!(cpu.icache.fills, fills0, "page-crosser must not fill");
        assert_eq!(cpu.icache.uncacheable, unc0 + 1);
    }

    /// (f) The public O(1) invalidate-all covers host-side writes.
    #[test]
    fn invalidate_icache_after_host_write_64() {
        let mut mem = LinearMemory::new();
        mem.load(0x1_0000, &[0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, 0xF4]);
        let mut cpu = Cpu::new();
        cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.gpr[0], 1);

        mem.load(0x1_0000, &[0x48, 0xC7, 0xC0, 0x02, 0x00, 0x00, 0x00, 0xF4]);
        cpu.invalidate_icache();
        cpu.regs.rip = 0x1_0000;
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.gpr[0], 2);
    }

    /// Cycle parity: pass 1 (fills) and pass 2 (hits) of the same program
    /// return identical per-step cycle counts.
    #[test]
    fn cycle_parity_two_pass() {
        #[rustfmt::skip]
        let program: &[u8] = &[
            0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00, // MOV RAX, 16
            0x48, 0xC7, 0xC3, 0x00, 0x00, 0x03, 0x00, // MOV RBX, 0x30000
            0x48, 0x89, 0x03,                         // MOV [RBX], RAX
            0x48, 0x03, 0x03,                         // ADD RAX, [RBX]
            0x48, 0xFF, 0xC8,                         // DEC RAX
            0x75, 0xFB,                               // JNZ (taken until zero)
            0xF4,                                     // HLT
        ];
        let mut mem = LinearMemory::new();
        mem.load(0x1_0000, program);
        let mut cpu = Cpu::new();
        cpu.setup_long_flat(&mut mem, 0x1_0000, 0x20_0000);

        let mut run = |cpu: &mut Cpu, mem: &mut LinearMemory| {
            cpu.regs.rip = 0x1_0000;
            cpu.halted = false;
            let mut cycles = Vec::new();
            for _ in 0..300 {
                if cpu.halted {
                    break;
                }
                cycles.push(cpu.step(mem));
            }
            cycles
        };
        let pass1 = run(&mut cpu, &mut mem);
        let pass2 = run(&mut cpu, &mut mem);
        assert_eq!(pass1, pass2, "cold and warm cycle counts must match");
    }
}
