//! 386 paging: two-level page-table walk with a small software TLB.
//!
//! All linear-address memory access funnels through the `lin_*` helpers,
//! which translate when `CR0.PG` is set and split accesses that cross a page
//! boundary. The 386 has no `WP` bit: supervisor writes ignore page-level
//! write protection, and no `INVLPG` — the TLB flushes on `CR3`/`CR0` loads.

use super::registers::cr0;
use super::{Bus, Cpu, Exception, Exec};

/// Number of direct-mapped TLB entries (must be a power of two).
const TLB_SIZE: usize = 256;

/// Tag value marking an empty slot (no linear page number is that large).
const EMPTY: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
struct TlbEntry {
    /// Linear page number (`lin >> 12`), or [`EMPTY`].
    tag: u32,
    /// Physical page base.
    phys: u32,
    /// Combined PDE & PTE user bit.
    user: bool,
    /// Combined PDE & PTE writable bit.
    writable: bool,
    /// The PTE dirty bit is known set (writes may skip the walk).
    dirty: bool,
}

/// A small direct-mapped translation cache.
#[derive(Debug, Clone)]
pub(crate) struct Tlb {
    entries: Box<[TlbEntry; TLB_SIZE]>,
}

impl Tlb {
    pub fn new() -> Self {
        Tlb {
            entries: Box::new(
                [TlbEntry {
                    tag: EMPTY,
                    phys: 0,
                    user: false,
                    writable: false,
                    dirty: false,
                }; TLB_SIZE],
            ),
        }
    }

    pub fn flush(&mut self) {
        for e in self.entries.iter_mut() {
            e.tag = EMPTY;
        }
    }

    /// Non-faulting read probe: the cached physical address for `lin` if a
    /// matching entry exists and permits a read at the given privilege
    /// (mirrors `translate`'s read rule). No walk, no A/D updates.
    #[inline(always)]
    pub fn peek(&self, lin: u32, user: bool) -> Option<u32> {
        let page = lin >> 12;
        let e = &self.entries[(page as usize) & (TLB_SIZE - 1)];
        if e.tag == page && (!user || e.user) {
            Some(e.phys | (lin & 0xFFF))
        } else {
            None
        }
    }
}

/// Page-table entry bits shared by PDEs and PTEs.
mod pte {
    pub const P: u32 = 1 << 0;
    pub const RW: u32 = 1 << 1;
    pub const US: u32 = 1 << 2;
    pub const A: u32 = 1 << 5;
    pub const D: u32 = 1 << 6;
}

impl Cpu {
    /// Flush the TLB (CR3 load, CR0 paging/protection changes).
    pub(crate) fn flush_tlb(&mut self) {
        self.tlb.flush();
    }

    /// Privilege of the current access for page protection.
    ///
    /// Explicit accesses use CPL, but the 386's *implicit* accesses — reading
    /// the descriptor tables, the IDT and the TSS, writing accessed/busy
    /// bits, and pushing a frame onto an inner-privilege stack while CPL is
    /// still 3 — are always supervisor accesses.
    #[inline]
    fn user_access(&self) -> bool {
        self.cpl() == 3 && !self.supervisor_override
    }

    /// Translate linear address `lin` for a read (`write == false`) or write.
    /// Returns the physical address; raises #PF with `CR2 = lin` on failure.
    ///
    /// The 386 has no CR0.WP, so supervisor writes ignore the writable bit.
    fn translate<B: Bus>(&mut self, bus: &mut B, lin: u32, write: bool) -> Exec<u32> {
        let page = lin >> 12;
        let slot = (page as usize) & (TLB_SIZE - 1);
        let e = self.tlb.entries[slot];
        if e.tag == page {
            let user = self.user_access();
            if (!user || (e.user && (!write || e.writable))) && (!write || e.dirty) {
                return Ok(e.phys | (lin & 0xFFF));
            }
        }
        self.walk(bus, lin, write)
    }

    /// Full two-level walk, updating accessed/dirty bits and the TLB.
    #[cold]
    fn walk<B: Bus>(&mut self, bus: &mut B, lin: u32, write: bool) -> Exec<u32> {
        let user = self.user_access();
        let fault = |present: bool| {
            let code = (present as u16) | (write as u16) << 1 | (user as u16) << 2;
            Exception::pf(code)
        };

        let pde_addr = (self.regs.cr3 & 0xFFFF_F000) + ((lin >> 22) << 2);
        let mut pde = bus.read32(pde_addr);
        if pde & pte::P == 0 {
            self.regs.cr2 = lin;
            return Err(fault(false));
        }
        let pte_addr = (pde & 0xFFFF_F000) + (((lin >> 12) & 0x3FF) << 2);
        let mut ptev = bus.read32(pte_addr);
        if ptev & pte::P == 0 {
            self.regs.cr2 = lin;
            return Err(fault(false));
        }

        // Combined protection: user needs U/S in both levels; user writes
        // additionally need R/W in both. Supervisor accesses always pass.
        let comb_user = pde & ptev & pte::US != 0;
        let comb_rw = pde & ptev & pte::RW != 0;
        if user && (!comb_user || (write && !comb_rw)) {
            self.regs.cr2 = lin;
            return Err(fault(true));
        }

        if pde & pte::A == 0 {
            pde |= pte::A;
            // A guest can execute from its own page tables, so even the
            // A/D write-backs stamp the icache (cold path, costs nothing).
            self.icache.stamp_write(pde_addr);
            bus.write32(pde_addr, pde);
        }
        let need_dirty = write && ptev & pte::D == 0;
        if ptev & pte::A == 0 || need_dirty {
            ptev |= pte::A;
            if write {
                ptev |= pte::D;
            }
            self.icache.stamp_write(pte_addr);
            bus.write32(pte_addr, ptev);
        }

        let page = lin >> 12;
        let slot = (page as usize) & (TLB_SIZE - 1);
        self.tlb.entries[slot] = TlbEntry {
            tag: page,
            phys: ptev & 0xFFFF_F000,
            user: comb_user,
            writable: comb_rw,
            dirty: ptev & pte::D != 0,
        };
        Ok((ptev & 0xFFFF_F000) | (lin & 0xFFF))
    }

    /// Whether paging is active.
    #[inline]
    pub(crate) fn paging(&self) -> bool {
        self.regs.cr0 & cr0::PG != 0
    }

    // --- Linear-address access (post-segmentation) ---------------------------

    #[inline]
    pub(crate) fn lin_read8<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Exec<u8> {
        if !self.paging() {
            return Ok(bus.read(lin));
        }
        let phys = self.translate(bus, lin, false)?;
        Ok(bus.read(phys))
    }

    #[inline]
    pub(crate) fn lin_read16<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Exec<u16> {
        if !self.paging() {
            return Ok(bus.read16(lin));
        }
        if lin & 0xFFF < 0xFFF {
            let phys = self.translate(bus, lin, false)?;
            Ok(bus.read16(phys))
        } else {
            let lo = self.lin_read8(bus, lin)? as u16;
            let hi = self.lin_read8(bus, lin.wrapping_add(1))? as u16;
            Ok(lo | hi << 8)
        }
    }

    #[inline]
    pub(crate) fn lin_read32<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Exec<u32> {
        if !self.paging() {
            return Ok(bus.read32(lin));
        }
        if lin & 0xFFF < 0xFFD {
            let phys = self.translate(bus, lin, false)?;
            Ok(bus.read32(phys))
        } else {
            let lo = self.lin_read16(bus, lin)? as u32;
            let hi = self.lin_read16(bus, lin.wrapping_add(2))? as u32;
            Ok(lo | hi << 16)
        }
    }

    // --- Implicit (always-supervisor) system-structure access ----------------

    /// Read a byte from a system structure (descriptor table, TSS, IDT).
    #[inline]
    pub(crate) fn sys_read8<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Exec<u8> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_read8(bus, lin);
        self.supervisor_override = sup;
        r
    }

    /// Read a word from a system structure.
    #[inline]
    pub(crate) fn sys_read16<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Exec<u16> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_read16(bus, lin);
        self.supervisor_override = sup;
        r
    }

    /// Read a double-word from a system structure.
    #[inline]
    pub(crate) fn sys_read32<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Exec<u32> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_read32(bus, lin);
        self.supervisor_override = sup;
        r
    }

    /// Write a byte to a system structure (accessed/busy bit writeback).
    #[inline]
    pub(crate) fn sys_write8<B: Bus>(&mut self, bus: &mut B, lin: u32, v: u8) -> Exec<()> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_write8(bus, lin, v);
        self.supervisor_override = sup;
        r
    }

    #[inline]
    pub(crate) fn lin_write8<B: Bus>(&mut self, bus: &mut B, lin: u32, v: u8) -> Exec<()> {
        if !self.paging() {
            self.icache.stamp_write(lin);
            bus.write(lin, v);
            return Ok(());
        }
        let phys = self.translate(bus, lin, true)?;
        self.icache.stamp_write(phys);
        bus.write(phys, v);
        Ok(())
    }

    #[inline]
    pub(crate) fn lin_write16<B: Bus>(&mut self, bus: &mut B, lin: u32, v: u16) -> Exec<()> {
        if !self.paging() {
            self.icache.stamp_write_span(lin, lin.wrapping_add(1));
            bus.write16(lin, v);
            return Ok(());
        }
        if lin & 0xFFF < 0xFFF {
            let phys = self.translate(bus, lin, true)?;
            self.icache.stamp_write(phys);
            bus.write16(phys, v);
            Ok(())
        } else {
            // Translate both pages before either byte lands so a fault on
            // the second page leaves the first untouched.
            let p0 = self.translate(bus, lin, true)?;
            let p1 = self.translate(bus, lin.wrapping_add(1), true)?;
            self.icache.stamp_write_span(p0, p1);
            bus.write(p0, v as u8);
            bus.write(p1, (v >> 8) as u8);
            Ok(())
        }
    }

    #[inline]
    pub(crate) fn lin_write32<B: Bus>(&mut self, bus: &mut B, lin: u32, v: u32) -> Exec<()> {
        if !self.paging() {
            self.icache.stamp_write_span(lin, lin.wrapping_add(3));
            bus.write32(lin, v);
            return Ok(());
        }
        if lin & 0xFFF < 0xFFD {
            let phys = self.translate(bus, lin, true)?;
            self.icache.stamp_write(phys);
            bus.write32(phys, v);
            Ok(())
        } else {
            // Byte-wise with all pages translated up front (see lin_write16).
            let mut phys = [0u32; 4];
            for (i, p) in phys.iter_mut().enumerate() {
                *p = self.translate(bus, lin.wrapping_add(i as u32), true)?;
            }
            self.icache.stamp_write_span(phys[0], phys[3]);
            for (i, p) in phys.iter().enumerate() {
                bus.write(*p, (v >> (8 * i)) as u8);
            }
            Ok(())
        }
    }

    // --- Public host accessors (for OS-emulation syscall marshaling) ----------
    // A host syscall layer must read/write the guest's *virtual* buffers the
    // same way the CPU does — through segmentation-flat linear addresses and
    // the page tables. These expose the tested `lin_*` path with the public
    // `Exception` error type; a fault (an inaccessible buffer) surfaces as
    // `Err`, which the caller maps to `-EFAULT`.

    /// Read a byte from guest linear address `lin`, honoring paging.
    #[inline]
    pub fn read_linear8<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Result<u8, Exception> {
        self.lin_read8(bus, lin)
    }

    /// Read a little-endian word from guest linear address `lin`.
    #[inline]
    pub fn read_linear16<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Result<u16, Exception> {
        self.lin_read16(bus, lin)
    }

    /// Read a little-endian double-word from guest linear address `lin`.
    #[inline]
    pub fn read_linear32<B: Bus>(&mut self, bus: &mut B, lin: u32) -> Result<u32, Exception> {
        self.lin_read32(bus, lin)
    }

    /// Write a byte to guest linear address `lin`, honoring paging.
    #[inline]
    pub fn write_linear8<B: Bus>(&mut self, bus: &mut B, lin: u32, v: u8) -> Result<(), Exception> {
        self.lin_write8(bus, lin, v)
    }

    /// Write a little-endian word to guest linear address `lin`.
    #[inline]
    pub fn write_linear16<B: Bus>(
        &mut self,
        bus: &mut B,
        lin: u32,
        v: u16,
    ) -> Result<(), Exception> {
        self.lin_write16(bus, lin, v)
    }

    /// Write a little-endian double-word to guest linear address `lin`.
    #[inline]
    pub fn write_linear32<B: Bus>(
        &mut self,
        bus: &mut B,
        lin: u32,
        v: u32,
    ) -> Result<(), Exception> {
        self.lin_write32(bus, lin, v)
    }

    /// Invalidate the whole translation cache. A host (OS-emulation) layer that
    /// edits page tables directly and then reloads `CR3` as a field — rather
    /// than via `MOV CR3` — must call this so stale entries do not linger (the
    /// 386 has no `INVLPG`, so a full flush is the only option anyway).
    #[inline]
    pub fn invalidate_tlb(&mut self) {
        self.flush_tlb();
    }

    /// Translate guest linear address `lin` to a physical address using the
    /// current page tables (`write` selects the write-permission rule).
    /// Returns `Err` (a `#PF`, with `CR2` set) if it is not accessible.
    #[inline]
    pub fn translate_linear<B: Bus>(
        &mut self,
        bus: &mut B,
        lin: u32,
        write: bool,
    ) -> Result<u32, Exception> {
        if !self.paging() {
            return Ok(lin);
        }
        self.translate(bus, lin, write)
    }
}
