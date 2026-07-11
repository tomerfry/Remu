//! Paging: legacy 32-bit (2-level), PAE (3-level) and long-mode (4-level)
//! page-table walks with a small software TLB.
//!
//! All linear-address memory access funnels through the `lin_*` helpers,
//! which translate when `CR0.PG` is set and split accesses that cross a page
//! boundary. Supported page sizes: 4 KiB everywhere, 4 MiB (legacy + CR4.PSE),
//! 2 MiB (PAE/long), 1 GiB (long). `CR0.WP` write protection and the
//! `EFER.NXE` no-execute bit are honored; PCIDs and protection keys are not
//! implemented (the corresponding CR4 bits reject).

use super::registers::{cr0, cr4, efer};
use super::{Bus, Cpu, Exception, Exec};

/// Number of direct-mapped TLB entries (must be a power of two).
const TLB_SIZE: usize = 256;

/// Tag value marking an empty slot (canonical linear page numbers never
/// reach it).
const EMPTY: u64 = u64::MAX;

/// Mask of the physical frame bits in a 64-bit page-table entry
/// (MAXPHYADDR = 52).
const PHYS_MASK: u64 = 0x000F_FFFF_FFFF_F000;

/// The access kind of a translation, selecting the protection rule and the
/// fault error-code bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Access {
    Read,
    Write,
    Fetch,
}

#[derive(Debug, Clone, Copy)]
struct TlbEntry {
    /// Linear page number (`lin >> 12`), or [`EMPTY`].
    tag: u64,
    /// Physical page base.
    phys: u64,
    /// Combined user bit of the whole walk.
    user: bool,
    /// Combined writable bit of the whole walk.
    writable: bool,
    /// Combined no-execute bit of the whole walk.
    nx: bool,
    /// The leaf dirty bit is known set (writes may skip the walk).
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
                    nx: false,
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

    /// Drop the entry covering linear address `lin` (INVLPG).
    pub fn invalidate(&mut self, lin: u64) {
        let page = lin >> 12;
        let e = &mut self.entries[(page as usize) & (TLB_SIZE - 1)];
        if e.tag == page {
            e.tag = EMPTY;
        }
    }
}

/// Page-table entry bits shared by all levels and formats.
mod pte {
    pub const P: u64 = 1 << 0;
    pub const RW: u64 = 1 << 1;
    pub const US: u64 = 1 << 2;
    pub const A: u64 = 1 << 5;
    pub const D: u64 = 1 << 6;
    /// Page size (non-leaf levels map a large page when set).
    pub const PS: u64 = 1 << 7;
    /// No-execute (64-bit entry formats; reserved unless EFER.NXE).
    pub const NX: u64 = 1 << 63;
}

impl Cpu {
    /// Flush the TLB (CR3 load, CR0/CR4/EFER paging-relevant changes).
    pub(crate) fn flush_tlb(&mut self) {
        self.fetch_invalidate();
        self.tlb.flush();
    }

    /// Drop the TLB entry for one page (INVLPG).
    pub(crate) fn invlpg(&mut self, lin: u64) {
        self.fetch_invalidate();
        self.tlb.invalidate(lin);
    }

    /// Privilege of the current access for page protection.
    ///
    /// Explicit accesses use CPL, but *implicit* accesses — reading the
    /// descriptor tables, the IDT and the TSS, writing accessed/busy bits,
    /// and pushing a frame onto an inner-privilege stack while CPL is still
    /// 3 — are always supervisor accesses.
    #[inline]
    fn user_access(&self) -> bool {
        self.cpl() == 3 && !self.supervisor_override
    }

    /// Whether the TLB entry `e` permits the access; misses fall back to a
    /// full walk (which re-derives protection and sets A/D bits).
    #[inline]
    fn tlb_permits(&self, e: &TlbEntry, access: Access) -> bool {
        let user = self.user_access();
        match access {
            Access::Read => !user || e.user,
            Access::Fetch => (!user || e.user) && !e.nx,
            Access::Write => {
                let prot = if user {
                    e.user && e.writable
                } else {
                    e.writable || self.regs.cr0 & cr0::WP == 0
                };
                prot && e.dirty
            }
        }
    }

    /// Translate linear address `lin`. Returns the physical address; raises
    /// #PF with `CR2 = lin` on failure.
    fn translate<B: Bus>(&mut self, bus: &mut B, lin: u64, access: Access) -> Exec<u64> {
        let page = lin >> 12;
        let slot = (page as usize) & (TLB_SIZE - 1);
        let e = self.tlb.entries[slot];
        if e.tag == page && self.tlb_permits(&e, access) {
            return Ok(e.phys | (lin & 0xFFF));
        }
        self.walk(bus, lin, access)
    }

    /// Build the #PF error code and record CR2.
    fn page_fault(&mut self, lin: u64, access: Access, present: bool, rsvd: bool) -> Exception {
        self.regs.cr2 = lin;
        let code = (present as u16)
            | ((access == Access::Write) as u16) << 1
            | (self.user_access() as u16) << 2
            | (rsvd as u16) << 3
            | ((access == Access::Fetch && self.regs.msr.efer & efer::NXE != 0) as u16) << 4;
        Exception::pf(code)
    }

    /// Whether the protection accumulated over a walk permits the access.
    fn walk_permits(&self, access: Access, user_ok: bool, writable: bool, nx: bool) -> bool {
        let user = self.user_access();
        match access {
            Access::Read => !user || user_ok,
            Access::Fetch => (!user || user_ok) && !nx,
            Access::Write => {
                if user {
                    user_ok && writable
                } else {
                    writable || self.regs.cr0 & cr0::WP == 0
                }
            }
        }
    }

    /// Full page-table walk in the format the mode selects, updating
    /// accessed/dirty bits and the TLB.
    #[cold]
    fn walk<B: Bus>(&mut self, bus: &mut B, lin: u64, access: Access) -> Exec<u64> {
        if self.regs.cr4 & cr4::PAE == 0 {
            self.walk_legacy(bus, lin, access)
        } else {
            self.walk_wide(bus, lin, access)
        }
    }

    /// Legacy 2-level walk: 32-bit entries, optional 4 MiB pages (CR4.PSE).
    fn walk_legacy<B: Bus>(&mut self, bus: &mut B, lin: u64, access: Access) -> Exec<u64> {
        let lin32 = lin as u32 as u64;
        let pde_addr = (self.regs.cr3 & 0xFFFF_F000) + ((lin32 >> 22) << 2);
        let mut pde = bus.read32(pde_addr) as u64;
        if pde & pte::P == 0 {
            return Err(self.page_fault(lin, access, false, false));
        }

        let large = pde & pte::PS != 0 && self.regs.cr4 & cr4::PSE != 0;
        let (leaf_addr, mut leaf, user_ok, writable, phys) = if large {
            let user_ok = pde & pte::US != 0;
            let writable = pde & pte::RW != 0;
            let phys = (pde & 0xFFC0_0000) | (lin32 & 0x003F_F000);
            (pde_addr, pde, user_ok, writable, phys)
        } else {
            let pte_addr = (pde & 0xFFFF_F000) + (((lin32 >> 12) & 0x3FF) << 2);
            let ptev = bus.read32(pte_addr) as u64;
            if ptev & pte::P == 0 {
                return Err(self.page_fault(lin, access, false, false));
            }
            let user_ok = pde & ptev & pte::US != 0;
            let writable = pde & ptev & pte::RW != 0;
            (pte_addr, ptev, user_ok, writable, ptev & 0xFFFF_F000)
        };

        if !self.walk_permits(access, user_ok, writable, false) {
            return Err(self.page_fault(lin, access, true, false));
        }

        if !large && pde & pte::A == 0 {
            bus.write32(pde_addr, (pde | pte::A) as u32);
            pde |= pte::A;
        }
        let _ = pde;
        let write = access == Access::Write;
        if leaf & pte::A == 0 || (write && leaf & pte::D == 0) {
            leaf |= pte::A;
            if write {
                leaf |= pte::D;
            }
            bus.write32(leaf_addr, leaf as u32);
        }

        self.tlb_fill(lin, phys, user_ok, writable, false, leaf & pte::D != 0);
        Ok(phys | (lin32 & 0xFFF))
    }

    /// PAE (3-level) and long-mode (4-level) walk: 64-bit entries, NX, and
    /// 2 MiB / 1 GiB pages.
    fn walk_wide<B: Bus>(&mut self, bus: &mut B, lin: u64, access: Access) -> Exec<u64> {
        let long = self.long_mode();
        let nxe = self.regs.msr.efer & efer::NXE != 0;

        // Collect (entry address, entry) from the top level down to the leaf.
        let mut levels: [(u64, u64); 4] = [(0, 0); 4];
        let mut n = 0usize;

        let mut table = if long {
            self.regs.cr3 & PHYS_MASK
        } else {
            // PAE: CR3 points at the 4-entry page-directory-pointer table.
            let pdpte_addr = (self.regs.cr3 & 0xFFFF_FFE0) + (((lin >> 30) & 3) << 3);
            let pdpte = bus.read64(pdpte_addr);
            if pdpte & pte::P == 0 {
                return Err(self.page_fault(lin, access, false, false));
            }
            // PAE PDPTEs define no RW/US/PS bits; several would-be flag bits
            // are reserved and fault when set, as is NX (in any mode: the
            // PAE PDPTE has no NX bit at all).
            if pdpte & (pte::RW | pte::US | pte::PS | pte::NX | 0x1E0) != 0 {
                return Err(self.page_fault(lin, access, true, true));
            }
            pdpte & PHYS_MASK
        };

        // Remaining levels: long mode walks PML4→PDPT→PD→PT (indices from
        // bits 47:12); PAE walks PD→PT (bits 29:12).
        let shifts: &[u32] = if long { &[39, 30, 21, 12] } else { &[21, 12] };
        let mut phys = 0u64;
        let mut page_mask = 0u64;
        for (i, &shift) in shifts.iter().enumerate() {
            let addr = table + (((lin >> shift) & 0x1FF) << 3);
            let e = bus.read64(addr);
            if e & pte::P == 0 {
                return Err(self.page_fault(lin, access, false, false));
            }
            // Reserved bits: NX without EFER.NXE, bits 62:52 above
            // MAXPHYADDR, and PS where no large page exists (PML4, PT).
            let leaf_level = i == shifts.len() - 1;
            let large_ok = shift == 21 || (long && shift == 30);
            if (!nxe && e & pte::NX != 0)
                || e & 0x7FF0_0000_0000_0000 != 0
                || (!leaf_level && !large_ok && e & pte::PS != 0)
            {
                return Err(self.page_fault(lin, access, true, true));
            }
            levels[n] = (addr, e);
            n += 1;
            if leaf_level || e & pte::PS != 0 {
                // Large-page leaves: low frame bits below the page size must
                // be zero (PAT bit 12 excepted — not modeled, must be clear).
                page_mask = if leaf_level { 0xFFF } else { (1 << shift) - 1 };
                if e & PHYS_MASK & page_mask != 0 {
                    return Err(self.page_fault(lin, access, true, true));
                }
                phys = (e & PHYS_MASK & !page_mask) | (lin & page_mask & !0xFFF);
                break;
            }
            table = e & PHYS_MASK;
        }

        // Combined protection across the walked levels.
        let mut user_ok = true;
        let mut writable = true;
        let mut nx = false;
        for &(_, e) in &levels[..n] {
            user_ok &= e & pte::US != 0;
            writable &= e & pte::RW != 0;
            nx |= nxe && e & pte::NX != 0;
        }
        if !self.walk_permits(access, user_ok, writable, nx) {
            return Err(self.page_fault(lin, access, true, false));
        }

        // Accessed on every level; dirty on the leaf for writes.
        let write = access == Access::Write;
        for (i, &(addr, e)) in levels[..n].iter().enumerate() {
            let leaf = i == n - 1;
            let mut v = e | pte::A;
            if leaf && write {
                v |= pte::D;
            }
            if v != e {
                bus.write64(addr, v);
            }
        }
        let dirty = levels[n - 1].1 & pte::D != 0 || write;

        self.tlb_fill(
            lin,
            phys | (lin & page_mask & !0xFFF),
            user_ok,
            writable,
            nx,
            dirty,
        );
        Ok(phys | (lin & 0xFFF))
    }

    /// Install a 4 KiB translation in the TLB (large pages are cached one
    /// 4 KiB chunk at a time).
    fn tlb_fill(&mut self, lin: u64, phys: u64, user: bool, writable: bool, nx: bool, dirty: bool) {
        let page = lin >> 12;
        let slot = (page as usize) & (TLB_SIZE - 1);
        self.tlb.entries[slot] = TlbEntry {
            tag: page,
            phys: phys & !0xFFF,
            user,
            writable,
            nx,
            dirty,
        };
    }

    /// Whether paging is active.
    #[inline]
    pub(crate) fn paging(&self) -> bool {
        self.regs.cr0 & cr0::PG != 0
    }

    // --- Linear-address access (post-segmentation) ---------------------------

    /// Physical address of the code byte at linear `lin` (NX-aware `Fetch`
    /// access, so a #PF error code carries the instruction-fetch bit). Used
    /// by the fetch window to translate once per page instead of per byte.
    #[inline]
    pub(crate) fn fetch_translate<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u64> {
        self.translate(bus, lin, Access::Fetch)
    }

    #[inline]
    pub(crate) fn lin_read8<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u8> {
        if !self.paging() {
            return Ok(bus.read(lin));
        }
        let phys = self.translate(bus, lin, Access::Read)?;
        Ok(bus.read(phys))
    }

    #[inline]
    pub(crate) fn lin_read16<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u16> {
        if !self.paging() {
            return Ok(bus.read16(lin));
        }
        if lin & 0xFFF < 0xFFF {
            let phys = self.translate(bus, lin, Access::Read)?;
            Ok(bus.read16(phys))
        } else {
            let lo = self.lin_read8(bus, lin)? as u16;
            let hi = self.lin_read8(bus, lin.wrapping_add(1))? as u16;
            Ok(lo | hi << 8)
        }
    }

    #[inline]
    pub(crate) fn lin_read32<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u32> {
        if !self.paging() {
            return Ok(bus.read32(lin));
        }
        if lin & 0xFFF < 0xFFD {
            let phys = self.translate(bus, lin, Access::Read)?;
            Ok(bus.read32(phys))
        } else {
            let lo = self.lin_read16(bus, lin)? as u32;
            let hi = self.lin_read16(bus, lin.wrapping_add(2))? as u32;
            Ok(lo | hi << 16)
        }
    }

    #[inline]
    pub(crate) fn lin_read64<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u64> {
        if !self.paging() {
            return Ok(bus.read64(lin));
        }
        if lin & 0xFFF < 0xFF9 {
            let phys = self.translate(bus, lin, Access::Read)?;
            Ok(bus.read64(phys))
        } else {
            let lo = self.lin_read32(bus, lin)? as u64;
            let hi = self.lin_read32(bus, lin.wrapping_add(4))? as u64;
            Ok(lo | hi << 32)
        }
    }

    #[inline]
    pub(crate) fn lin_write8<B: Bus>(&mut self, bus: &mut B, lin: u64, v: u8) -> Exec<()> {
        if !self.paging() {
            bus.write(lin, v);
            return Ok(());
        }
        let phys = self.translate(bus, lin, Access::Write)?;
        bus.write(phys, v);
        Ok(())
    }

    #[inline]
    pub(crate) fn lin_write16<B: Bus>(&mut self, bus: &mut B, lin: u64, v: u16) -> Exec<()> {
        if !self.paging() {
            bus.write16(lin, v);
            return Ok(());
        }
        if lin & 0xFFF < 0xFFF {
            let phys = self.translate(bus, lin, Access::Write)?;
            bus.write16(phys, v);
            Ok(())
        } else {
            // Translate both pages before either byte lands so a fault on
            // the second page leaves the first untouched.
            let p0 = self.translate(bus, lin, Access::Write)?;
            let p1 = self.translate(bus, lin.wrapping_add(1), Access::Write)?;
            bus.write(p0, v as u8);
            bus.write(p1, (v >> 8) as u8);
            Ok(())
        }
    }

    #[inline]
    pub(crate) fn lin_write32<B: Bus>(&mut self, bus: &mut B, lin: u64, v: u32) -> Exec<()> {
        if !self.paging() {
            bus.write32(lin, v);
            return Ok(());
        }
        if lin & 0xFFF < 0xFFD {
            let phys = self.translate(bus, lin, Access::Write)?;
            bus.write32(phys, v);
            Ok(())
        } else {
            // Byte-wise with all pages translated up front (see lin_write16).
            let mut phys = [0u64; 4];
            for (i, p) in phys.iter_mut().enumerate() {
                *p = self.translate(bus, lin.wrapping_add(i as u64), Access::Write)?;
            }
            for (i, p) in phys.iter().enumerate() {
                bus.write(*p, (v >> (8 * i)) as u8);
            }
            Ok(())
        }
    }

    #[inline]
    pub(crate) fn lin_write64<B: Bus>(&mut self, bus: &mut B, lin: u64, v: u64) -> Exec<()> {
        if !self.paging() {
            bus.write64(lin, v);
            return Ok(());
        }
        if lin & 0xFFF < 0xFF9 {
            let phys = self.translate(bus, lin, Access::Write)?;
            bus.write64(phys, v);
            Ok(())
        } else {
            let mut phys = [0u64; 8];
            for (i, p) in phys.iter_mut().enumerate() {
                *p = self.translate(bus, lin.wrapping_add(i as u64), Access::Write)?;
            }
            for (i, p) in phys.iter().enumerate() {
                bus.write(*p, (v >> (8 * i)) as u8);
            }
            Ok(())
        }
    }

    // --- Implicit (always-supervisor) system-structure access ----------------

    /// Read a byte from a system structure (descriptor table, TSS, IDT).
    #[inline]
    pub(crate) fn sys_read8<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u8> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_read8(bus, lin);
        self.supervisor_override = sup;
        r
    }

    /// Read a word from a system structure.
    #[inline]
    pub(crate) fn sys_read16<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u16> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_read16(bus, lin);
        self.supervisor_override = sup;
        r
    }

    /// Read a double-word from a system structure.
    #[inline]
    pub(crate) fn sys_read32<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u32> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_read32(bus, lin);
        self.supervisor_override = sup;
        r
    }

    /// Read a quad-word from a system structure.
    #[inline]
    pub(crate) fn sys_read64<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Exec<u64> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_read64(bus, lin);
        self.supervisor_override = sup;
        r
    }

    /// Write a byte to a system structure (accessed/busy bit writeback).
    #[inline]
    pub(crate) fn sys_write8<B: Bus>(&mut self, bus: &mut B, lin: u64, v: u8) -> Exec<()> {
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let r = self.lin_write8(bus, lin, v);
        self.supervisor_override = sup;
        r
    }

    // --- Public host accessors (for OS-emulation syscall marshaling) ----------
    // A host syscall layer must read/write the guest's *virtual* buffers the
    // same way the CPU does — through the page tables. These expose the
    // `lin_*` path with the public `Exception` error type; a fault (an
    // inaccessible buffer) surfaces as `Err`, which the caller maps to
    // `-EFAULT`.

    /// Read a byte from guest linear address `lin`, honoring paging.
    #[inline]
    pub fn read_linear8<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Result<u8, Exception> {
        self.lin_read8(bus, lin)
    }

    /// Read a little-endian word from guest linear address `lin`.
    #[inline]
    pub fn read_linear16<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Result<u16, Exception> {
        self.lin_read16(bus, lin)
    }

    /// Read a little-endian double-word from guest linear address `lin`.
    #[inline]
    pub fn read_linear32<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Result<u32, Exception> {
        self.lin_read32(bus, lin)
    }

    /// Read a little-endian quad-word from guest linear address `lin`.
    #[inline]
    pub fn read_linear64<B: Bus>(&mut self, bus: &mut B, lin: u64) -> Result<u64, Exception> {
        self.lin_read64(bus, lin)
    }

    /// Write a byte to guest linear address `lin`, honoring paging.
    #[inline]
    pub fn write_linear8<B: Bus>(&mut self, bus: &mut B, lin: u64, v: u8) -> Result<(), Exception> {
        self.lin_write8(bus, lin, v)
    }

    /// Write a little-endian word to guest linear address `lin`.
    #[inline]
    pub fn write_linear16<B: Bus>(
        &mut self,
        bus: &mut B,
        lin: u64,
        v: u16,
    ) -> Result<(), Exception> {
        self.lin_write16(bus, lin, v)
    }

    /// Write a little-endian double-word to guest linear address `lin`.
    #[inline]
    pub fn write_linear32<B: Bus>(
        &mut self,
        bus: &mut B,
        lin: u64,
        v: u32,
    ) -> Result<(), Exception> {
        self.lin_write32(bus, lin, v)
    }

    /// Write a little-endian quad-word to guest linear address `lin`.
    #[inline]
    pub fn write_linear64<B: Bus>(
        &mut self,
        bus: &mut B,
        lin: u64,
        v: u64,
    ) -> Result<(), Exception> {
        self.lin_write64(bus, lin, v)
    }

    /// Invalidate the whole translation cache. A host (OS-emulation) layer
    /// that edits page tables directly and then reloads `CR3` as a field —
    /// rather than via `MOV CR3` — must call this so stale entries do not
    /// linger.
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
        lin: u64,
        write: bool,
    ) -> Result<u64, Exception> {
        if !self.paging() {
            return Ok(lin);
        }
        let access = if write { Access::Write } else { Access::Read };
        self.translate(bus, lin, access)
    }
}
