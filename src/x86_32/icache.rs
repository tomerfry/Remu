//! Direct-mapped decoded-instruction cache, physically keyed.
//!
//! Each entry holds the register-independent decode products of one
//! instruction ([`DecodedInsn`]); a hit skips the fetch, prefix scan and
//! opcode dispatch entirely and re-evaluates only the EA formula. Safety
//! comes from three checks on the probe:
//!
//! - **key**: physical address of the first instruction byte, with the
//!   decode context (CS.D, CR0.PE, EFLAGS.VM, `extensions`) folded into the
//!   high bits — a mode change simply misses, nothing is flushed;
//! - **write stamps**: every guest store (and page-walk A/D write-back)
//!   bumps a monotonic clock into a per-physical-page slot; an entry is
//!   live only while its fill-time clock is newer, so self-modifying code
//!   re-decodes exactly like the fused interpreter, which re-fetches every
//!   byte (no prefetch queue is modeled);
//! - **CS limit**: re-checked per hit, replacing the per-byte `fetch_check`
//!   (a limit shrink without a mode change misses and runs fused → #GP).
//!
//! Instructions whose bytes cross a 4 KiB physical page are never cached:
//! the stamp covers only the first page, and physical contiguity across
//! pages is not stable under remapping. Host-side writes to guest memory
//! bypass the CPU entirely — hosts must call [`Cpu::invalidate_icache`]
//! (O(1)) before stepping again.

use std::alloc::{Layout, alloc_zeroed, handle_alloc_error};
use std::fmt;

use super::Cpu;
use super::decode::DecodedInsn;
use super::registers::{EFlags, cr0, reg};

/// Number of direct-mapped entries, byte-indexed by physical address
/// (16384 × 32 B = 512 KiB, a 16 KiB contiguous code window).
pub(crate) const ICACHE_ENTRIES: usize = 16384;

/// Per-page write-stamp slots, indexed by `(phys >> 12) & (SLOTS-1)`.
/// Pages 64 MiB apart share a slot; aliasing only causes spurious
/// re-decodes, never a stale hit.
pub(crate) const STAMP_SLOTS: usize = 16384;

/// One cache slot. The all-zero pattern is a dead entry (`version` 0 is
/// below the initial invalidation clock), so the table can be allocated
/// as zeroed pages instead of being written entry by entry.
#[derive(Clone, Copy)]
pub(crate) struct Entry {
    /// Physical address of the first instruction byte | context bits.
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
    /// Direct-mapped entries. Fixed-size so a masked index is provably
    /// in bounds (no bounds check on the probe path).
    pub entries: Box<[Entry; ICACHE_ENTRIES]>,
    /// Per-page write stamps, holding the clock value of the last store
    /// into any aliasing page.
    pub stamps: Box<[u64; STAMP_SLOTS]>,
    /// Monotonic store clock. Starts at 1 so zeroed entries are dead; a
    /// u64 cannot realistically wrap.
    pub clock: u64,
    /// Entries with `version < inval` are dead (O(1) invalidate-all).
    pub inval: u64,

    // Probe/fill accounting for the exactness tests (compiled out of
    // release library builds).
    #[cfg(any(test, debug_assertions))]
    pub hits: u64,
    #[cfg(any(test, debug_assertions))]
    pub fills: u64,
    #[cfg(any(test, debug_assertions))]
    pub uncacheable: u64,
}

/// Allocate a zeroed boxed array without writing the pattern element by
/// element — the OS hands back lazily-zeroed pages, so an icache that is
/// never (or sparsely) used costs almost nothing to construct.
///
/// Safety: only used for types whose all-zero bit pattern is a valid value
/// (`Entry` — a dead slot — and `u64`).
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
    pub fn stamp_write(&mut self, phys: u32) {
        self.clock += 1;
        self.stamps[(phys >> 12) as usize & (STAMP_SLOTS - 1)] = self.clock;
    }

    /// Record a store spanning `first..=last` (physical): stamps the first
    /// byte's page and, when the span crosses into it, the last byte's.
    #[inline(always)]
    pub fn stamp_write_span(&mut self, first: u32, last: u32) {
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
    /// The decode-context bits folded into an entry's key (bits 32–35):
    /// everything besides the instruction bytes themselves that changes
    /// what they decode *to*. CPL, segment bases/limits and paging state
    /// are execution-time state, checked live by the reused handlers.
    #[inline(always)]
    pub(crate) fn icache_ctx(&self) -> u64 {
        // Pure bit moves, no boolean materialization: CS.D (attrs bit 10)
        // to bit 32, CR0.PE (bit 0) to bit 33, EFLAGS.VM (bit 17) to bit
        // 34, `extensions` to bit 35.
        ((self.regs.seg[reg::CS as usize].attrs as u64 & 0x400) << 22)
            | ((self.regs.cr0 as u64 & cr0::PE as u64) << 33)
            | ((self.regs.eflags.bits() as u64 & EFlags::VM.bits() as u64) << 17)
            | ((self.extensions as u64) << 35)
    }

    /// Physical address of the byte at CS:EIP, if it can be determined
    /// without faulting and without bus access: identity when paging is
    /// off, a read-permission TLB peek otherwise (a TLB miss returns
    /// `None`; the fused path's fetch will walk and fill the TLB).
    ///
    /// Instruction fetch is never a `supervisor_override` access, so the
    /// privilege used here is plain CPL — matching what `fetch8`'s
    /// translate sees after `exec_one` resets the override flag.
    #[inline(always)]
    pub(crate) fn icache_phys(&self) -> Option<u32> {
        let lin = self.regs.seg[reg::CS as usize]
            .base
            .wrapping_add(self.regs.eip);
        if !self.paging() {
            return Some(lin);
        }
        self.tlb.peek(lin, self.cpl() == 3)
    }

    /// Install a freshly decoded instruction at `phys` (its first byte).
    pub(crate) fn icache_fill(&mut self, phys: u32, d: &DecodedInsn) {
        // Never cache a page-crosser (see module docs). `len` is at least 1.
        if (phys & 0xFFF) + d.len as u32 > 0x1000 {
            #[cfg(any(test, debug_assertions))]
            {
                self.icache.uncacheable += 1;
            }
            self.stat(|s| s.icache_uncacheable += 1);
            return;
        }
        let slot = phys as usize & (ICACHE_ENTRIES - 1);
        // The clock has not moved since decode began (decode only fetches;
        // A/D write-backs during those fetches bump it *before* the bytes
        // are read), so `clock` is the fill-time version and any later
        // store — including one by this very instruction — invalidates.
        self.icache.entries[slot] = Entry {
            key: phys as u64 | self.icache_ctx(),
            version: self.icache.clock,
            insn: *d,
        };
        #[cfg(any(test, debug_assertions))]
        {
            self.icache.fills += 1;
        }
        self.stat(|s| s.icache_fills += 1);
    }

    /// Invalidate the decoded-instruction cache (O(1)).
    ///
    /// The CPU stamps every store it performs itself, but host-side writes
    /// to guest memory (loaders, syscall emulation writing guest buffers,
    /// direct `LinearMemory` pokes) bypass it entirely — call this before
    /// the next [`Cpu::step`] after any such write.
    #[inline]
    pub fn invalidate_icache(&mut self) {
        self.icache.invalidate_all();
    }
}

#[cfg(test)]
mod tests {
    use super::super::registers::{EFlags, SegReg, cr0, reg};
    use super::super::{Cpu, HostTrap, LinearMemory};

    /// Real-mode CPU at 0000:1000 with `bytes` loaded there.
    fn real_cpu(bytes: &[u8]) -> (Cpu, LinearMemory) {
        let mut mem = LinearMemory::new();
        mem.load(0x1000, bytes);
        let mut cpu = Cpu::new();
        cpu.set_cs_ip(0x0000, 0x1000);
        (cpu, mem)
    }

    /// Flat ring-0 protected-mode CPU (descriptor caches loaded directly,
    /// same trick as the benchmark), 32-bit code unless `db32` is false.
    fn prot_cpu(db32: bool) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.regs.cr0 |= cr0::PE;
        cpu.regs.seg[reg::CS as usize] = SegReg {
            sel: 0x08,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: if db32 { 0x0C9B } else { 0x009B },
        };
        for s in [reg::SS, reg::DS, reg::ES, reg::FS, reg::GS] {
            cpu.regs.seg[s as usize] = SegReg {
                sel: 0x10,
                base: 0,
                limit: 0xFFFF_FFFF,
                attrs: 0x0C93,
            };
        }
        cpu
    }

    fn run_until_halt(cpu: &mut Cpu, mem: &mut LinearMemory, cap: u32) {
        for _ in 0..cap {
            if cpu.halted || cpu.host_trap.is_some() {
                return;
            }
            cpu.step(mem);
        }
        panic!("did not halt");
    }

    /// (a) Same-page SMC through a guest store: a loop patches its own
    /// INC into a DEC between iterations; the write stamp must kill the
    /// cached entry so the second pass re-decodes.
    #[test]
    fn smc_same_page_repatches() {
        #[rustfmt::skip]
        let program: &[u8] = &[
            0xB8, 0x02, 0x00,             // 1000: MOV AX, 2
            0xB9, 0x02, 0x00,             // 1003: MOV CX, 2
            0x40,                         // 1006: INC AX      <- patched to DEC
            0xC6, 0x06, 0x06, 0x10, 0x48, // 1007: MOV byte [0x1006], 0x48
            0x49,                         // 100C: DEC CX
            0x75, 0xF7,                   // 100D: JNZ 0x1006
            0xF4,                         // 100F: HLT
        ];
        let (mut cpu, mut mem) = real_cpu(program);
        run_until_halt(&mut cpu, &mut mem, 100);
        // Pass 1: INC (3), pass 2: DEC (2). A stale hit would give 4.
        assert_eq!(cpu.regs.reg16(0), 2);
    }

    /// (b) A store into the executing instruction's *own* bytes: the
    /// current iteration completes with the old decode (the fused
    /// interpreter has no prefetch queue but has already consumed the
    /// bytes); the next iteration sees the new bytes — here an invalid
    /// ModRM that must fault #UD instead of replaying the stale decode.
    #[test]
    fn smc_store_to_own_bytes() {
        #[rustfmt::skip]
        let program: &[u8] = &[
            0xB9, 0x02, 0x00,             // 1000: MOV CX, 2
            0xC6, 0x06, 0x04, 0x10, 0x48, // 1003: MOV byte [0x1004], 0x48 (own ModRM)
            0x49,                         // 1008: DEC CX
            0x75, 0xF8,                   // 1009: JNZ 0x1003
            0xF4,                         // 100B: HLT
        ];
        let (mut cpu, mut mem) = real_cpu(program);
        cpu.trap_faults = true;
        run_until_halt(&mut cpu, &mut mem, 100);
        // First visit executed fully (CX 2->1, patch landed), the second
        // decodes `C6 48 ..` = C6/1 -> #UD.
        assert_eq!(cpu.regs.reg16(1), 1);
        assert!(
            matches!(cpu.host_trap, Some(HostTrap::Exception(e)) if e.vector == 6),
            "expected #UD from the patched bytes, got {:?}",
            cpu.host_trap
        );
        assert_eq!(cpu.regs.eip, 0x1003, "faulting EIP must rewind to start");
    }

    /// (g) Instructions whose bytes cross a 4 KiB physical page are never
    /// cached.
    #[test]
    fn page_crossing_instruction_not_cached() {
        let mut mem = LinearMemory::new();
        mem.load(0x1FFE, &[0xB8, 0x34, 0x12, 0xF4]); // MOV AX,0x1234 crosses 0x2000
        let mut cpu = Cpu::new();
        cpu.set_cs_ip(0x0000, 0x1FFE);
        let fills0 = cpu.icache.fills;
        let unc0 = cpu.icache.uncacheable;
        cpu.step(&mut mem);
        assert_eq!(cpu.regs.reg16(0), 0x1234);
        assert_eq!(cpu.icache.fills, fills0, "page-crosser must not fill");
        assert_eq!(cpu.icache.uncacheable, unc0 + 1);
        // The HLT at 0x2001 is page-local and caches fine.
    }

    /// (h) LOCK- and REP-prefixed instructions never enter the decoded
    /// path (and so never fill).
    #[test]
    fn lock_and_rep_forms_not_cached() {
        #[rustfmt::skip]
        let program: &[u8] = &[
            0xBB, 0x00, 0x30,       // MOV BX, 0x3000
            0xF0, 0x01, 0x07,       // LOCK ADD [BX], AX
            0xBF, 0x00, 0x31,       // MOV DI, 0x3100
            0xB9, 0x04, 0x00,       // MOV CX, 4
            0xF3, 0xAA,             // REP STOSB
            0xF4,                   // HLT
        ];
        let (mut cpu, mut mem) = real_cpu(program);
        run_until_halt(&mut cpu, &mut mem, 100);
        // Only the three MOV r,imm and (not) HLT are cacheable: HLT is not
        // in the subset either, so exactly 3 fills.
        assert_eq!(cpu.icache.fills, 3);
    }

    /// Context bits: the same physical bytes must re-decode (miss) after
    /// CR0.PE, EFLAGS.VM or `extensions` change, without any flush call.
    #[test]
    fn context_change_rekeys() {
        let (mut cpu, mut mem) = real_cpu(&[0xB8, 0x34, 0x12, 0xF4]); // MOV AX; HLT
        run_until_halt(&mut cpu, &mut mem, 10);
        let fills = cpu.icache.fills;
        let hits = cpu.icache.hits;

        // Same context: pure hits, no new fills.
        cpu.set_cs_ip(0x0000, 0x1000);
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.icache.fills, fills);
        assert!(cpu.icache.hits > hits);

        // Flip `extensions` (a bare pub field — no setter to flush from):
        // the key changes, so the old entry misses and a new fill happens.
        cpu.extensions = true;
        cpu.set_cs_ip(0x0000, 0x1000);
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.icache.fills, fills + 1, "extensions must re-key");

        // Flip EFLAGS.VM (host-style poke): re-keys again.
        cpu.regs.eflags.insert(EFlags::VM);
        cpu.regs.cr0 |= cr0::PE; // V86 requires PE; CS stays real-style
        cpu.set_cs_ip(0x0000, 0x1000);
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.icache.fills, fills + 2, "PE/VM must re-key");
    }

    /// (d) CS limit shrink without any segment reload: the per-hit limit
    /// revalidation must miss and the fused path must raise #GP(0) with
    /// EIP rewound.
    #[test]
    fn cs_limit_shrink_faults() {
        let mut mem = LinearMemory::new();
        mem.load(0x1000, &[0x40, 0xF4]); // INC EAX; HLT
        let mut cpu = prot_cpu(true);
        cpu.regs.eip = 0x1000;
        cpu.step(&mut mem); // fills + executes INC
        assert_eq!(cpu.regs.gpr[0], 1);

        cpu.regs.seg[reg::CS as usize].limit = 0xFFF; // host-style shrink
        cpu.trap_faults = true;
        cpu.regs.eip = 0x1000;
        let hits = cpu.icache.hits;
        cpu.step(&mut mem);
        assert_eq!(cpu.icache.hits, hits, "shrunken limit must not hit");
        assert!(
            matches!(cpu.host_trap, Some(HostTrap::Exception(e)) if e.vector == 13),
            "expected #GP, got {:?}",
            cpu.host_trap
        );
        assert_eq!(cpu.regs.eip, 0x1000);
        assert_eq!(cpu.regs.gpr[0], 1, "faulting INC must not execute");
    }

    /// (e) The same physical bytes under 16-bit and 32-bit code segments
    /// decode differently; CS.D is in the key, so both decodes coexist.
    #[test]
    fn cs_d_context_coexists() {
        let mut mem = LinearMemory::new();
        // As 32-bit code: MOV EAX,0x12345678; HLT.
        // As 16-bit code: MOV AX,0x5678; XOR AL,0x12; HLT.
        mem.load(0x1000, &[0xB8, 0x78, 0x56, 0x34, 0x12, 0xF4]);

        let mut cpu = prot_cpu(true);
        cpu.regs.eip = 0x1000;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.gpr[0], 0x1234_5678);

        // Same CPU, same bytes, 16-bit CS (host-style attribute poke).
        cpu.regs.seg[reg::CS as usize].attrs = 0x009B;
        cpu.regs.gpr[0] = 0;
        cpu.regs.eip = 0x1000;
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.reg16(0), 0x566A); // 0x5678 with AL ^= 0x12

        // And back to 32-bit: still decodes correctly. (The 16-bit fill
        // evicted the 32-bit entry — same direct-mapped slot — so this
        // re-fills; the guarantee is correctness, not retention.)
        cpu.regs.seg[reg::CS as usize].attrs = 0x0C9B;
        cpu.regs.gpr[0] = 0;
        cpu.regs.eip = 0x1000;
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
    }

    /// (k) A wide store crossing into the cached instruction's page from
    /// the page before (paging off: the write reaches the bus unsplit)
    /// must still invalidate via the span stamp.
    #[test]
    fn split_write_stamps_both_pages() {
        let mut mem = LinearMemory::new();
        // Victim at 0x2000: INC AX; HLT.
        mem.load(0x2000, &[0x40, 0xF4]);
        // Writer at 0x1000: MOV word [0x1FFF], 0x9048 (writes 0x48 to
        // 0x1FFF and 0x90/NOP over the victim's first byte); HLT.
        mem.load(0x1000, &[0xC7, 0x06, 0xFF, 0x1F, 0x48, 0x90, 0xF4]);
        let mut cpu = Cpu::new();

        cpu.set_cs_ip(0x0000, 0x2000);
        run_until_halt(&mut cpu, &mut mem, 10); // INC AX (fills), HLT
        assert_eq!(cpu.regs.reg16(0), 1);

        cpu.set_cs_ip(0x0000, 0x1000);
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10); // the crossing store

        cpu.set_cs_ip(0x0000, 0x2000);
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10); // now NOP; HLT
        assert_eq!(cpu.regs.reg16(0), 1, "patched NOP must not increment");
    }

    /// (k, paging on) Same shape with CR0.PG set and identity page tables:
    /// the split write path translates both pages and must stamp both.
    #[test]
    fn split_write_stamps_both_pages_paging() {
        let mut mem = LinearMemory::new();
        // Identity tables: PDE[0] at 0x8000 -> PT at 0x9000; PT maps the
        // first 4 MiB identity, supervisor, R/W, accessed+dirty preset (so
        // no A/D write-backs muddy the stamp assertions).
        mem.load(0x8000, &0x9067u32.to_le_bytes());
        for page in 0..1024u32 {
            let pte = (page << 12) | 0x67;
            mem.load(0x9000 + page * 4, &pte.to_le_bytes());
        }
        mem.load(0x2000, &[0x40, 0xF4]);
        mem.load(0x1000, &[0xC7, 0x06, 0xFF, 0x1F, 0x48, 0x90, 0xF4]);

        let mut cpu = prot_cpu(false); // 16-bit flat, matching the code
        cpu.regs.cr3 = 0x8000;
        cpu.regs.cr0 |= cr0::PG;

        cpu.regs.eip = 0x2000;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.reg16(0), 1);

        cpu.regs.eip = 0x1000;
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);

        cpu.regs.eip = 0x2000;
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.reg16(0), 1, "patched NOP must not increment");
    }

    /// (l) Page-walker A/D write-backs are guest-visible stores; a guest
    /// executing from its own page table must see its cached instruction
    /// invalidated when a walk sets an accessed bit inside those bytes.
    #[test]
    fn ad_writeback_invalidates_code_in_page_table() {
        let mut mem = LinearMemory::new();
        mem.load(0x8000, &0x9067u32.to_le_bytes()); // PDE[0] -> PT at 0x9000
        for page in 0..1024u32 {
            let pte = (page << 12) | 0x67; // present, RW, user, A+D set
            mem.load(0x9000 + page * 4, &pte.to_le_bytes());
        }
        // PTE for page 0x40 (lin 0x40000), at 0x9000 + 0x40*4 = 0x9100, is
        // also executed as code: 41 F4 00 00 = INC CX; HLT — and as a PTE
        // it is present (bit 0 of 0x41) with the accessed bit CLEAR, frame
        // 0xF000. Reading lin 0x40000 walks and sets A, rewriting these
        // very bytes (0x41 -> 0x61 = POPA).
        mem.load(0x9100, &[0x41, 0xF4, 0x00, 0x00]);

        let mut cpu = prot_cpu(false);
        cpu.regs.cr3 = 0x8000;
        cpu.regs.cr0 |= cr0::PG;
        cpu.regs.gpr[reg::ESP as usize] = 0x7000;

        // Execute the PTE-as-code: INC CX, fills the icache.
        cpu.regs.eip = 0x9100;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.reg16(1), 1);

        // Touch lin 0x40000 (a1 = MOV AX, moffs16) so the walk sets the
        // accessed bit inside the cached instruction's bytes.
        mem.load(0x1000, &[0xA1, 0x00, 0x00, 0xF4]); // MOV AX,[0]; HLT
        cpu.regs.seg[reg::DS as usize].base = 0x40000;
        cpu.regs.eip = 0x1000;
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);

        // Re-execute: the bytes are now 61 F4 = POPA; HLT. POPA pops eight
        // zero words from the (zeroed) stack, so CX becomes 0 — a stale
        // hit would instead increment it to 2.
        cpu.regs.eip = 0x9100;
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.reg16(1), 0, "A-bit write-back must invalidate");
    }

    /// (f) The public O(1) invalidate-all covers host-side writes.
    #[test]
    fn invalidate_icache_after_host_write() {
        let (mut cpu, mut mem) = real_cpu(&[0xB8, 0x01, 0x00, 0xF4]); // MOV AX,1
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.reg16(0), 1);

        mem.load(0x1000, &[0xB8, 0x02, 0x00, 0xF4]); // host rewrite: MOV AX,2
        cpu.invalidate_icache();
        cpu.set_cs_ip(0x0000, 0x1000);
        cpu.halted = false;
        run_until_halt(&mut cpu, &mut mem, 10);
        assert_eq!(cpu.regs.reg16(0), 2);
    }
}
