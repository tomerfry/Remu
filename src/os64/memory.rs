//! Guest process address space for the x86-64 OS layer.
//!
//! The process runs under real long-mode paging (`CR0.PG = 1`, `CR4.PAE = 1`,
//! `EFER.LMA = 1`, flat ring-3 segments), so the CPU's validated 4-level
//! page-table walker enforces permissions and raises `#PF` — which the OS layer
//! traps to demand-grow the stack or deliver `SIGSEGV`. Three pieces cooperate,
//! mirroring the 32-bit [`crate::os::memory`] design widened to 64 bits:
//!
//! * [`PhysMem`] — a sparse, page-granular guest-*physical* frame store; the
//!   [`Bus`] the CPU accesses after translation.
//! * page tables — a four-level tree (PML4→PDPT→PD→PT) built in physical memory
//!   (`CR3` points at the PML4); the source of truth for linear→physical and
//!   permissions.
//! * [`AddressSpace`] — a frame allocator, the page tables, and a VMA list,
//!   exposing `map`/`mmap`/`brk` and host-side guest-memory access that bypasses
//!   user permissions (the kernel's privilege).

use std::collections::HashMap;

use crate::x86_64::Bus;

/// Page size (and physical frame size).
pub const PAGE_SIZE: u64 = 4096;
const FRAME: usize = 4096;
const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// Physical frame-address bits in a 64-bit page-table entry (MAXPHYADDR = 52),
/// matching the CPU core's paging unit.
const PHYS_MASK: u64 = 0x000F_FFFF_FFFF_F000;

/// Round `addr` up to the next page boundary. Saturating, so a malformed
/// (near-`u64::MAX`) length can never overflow the rounding itself.
#[inline]
fn page_up(addr: u64) -> u64 {
    addr.saturating_add(PAGE_MASK) & !PAGE_MASK
}

/// Round `addr` down to its page base.
#[inline]
fn page_down(addr: u64) -> u64 {
    addr & !PAGE_MASK
}

/// Page-rounded exclusive end of `[start, start+len)`, saturating so a
/// guest- or ELF-supplied length can never overflow the address computation
/// (a debug-build panic / release wraparound on hostile input).
#[inline]
fn range_end(start: u64, len: u64) -> u64 {
    page_up(start.saturating_add(len))
}

// --- Protection flags (match Linux PROT_*) -----------------------------------
pub const PROT_READ: u32 = 1;
pub const PROT_WRITE: u32 = 2;
pub const PROT_EXEC: u32 = 4;

// --- Page-table entry bits ---------------------------------------------------
mod pte {
    pub const P: u64 = 1 << 0;
    pub const RW: u64 = 1 << 1;
    pub const US: u64 = 1 << 2;
}

// --- Canonical process layout ------------------------------------------------
/// Exclusive top of the initial stack (one page below the first non-canonical
/// lower-half address, `0x0000_8000_0000_0000`).
pub const STACK_TOP: u64 = 0x0000_7FFF_FFFF_F000;
/// Lowest address the stack may auto-grow to (8 MiB below the top, like the
/// default `RLIMIT_STACK`).
pub const STACK_LIMIT: u64 = STACK_TOP - 0x0080_0000;
/// Base of the `mmap` arena (grows up toward the stack). Any base works: the
/// dynamic linker uses whatever `mmap` returns.
pub const MMAP_BASE: u64 = 0x0000_7F00_0000_0000;
/// Exclusive top of the canonical lower half — the highest address a user
/// mapping may reach. The loader rejects segments that would cross it.
pub const USER_END: u64 = 0x0000_8000_0000_0000;

/// Sparse guest-physical memory: 4 KiB frames created on first touch. Reads of
/// never-written frames return 0 (open-bus-like), matching the infallible x86
/// [`Bus`] contract.
#[derive(Default)]
pub struct PhysMem {
    frames: HashMap<u64, Box<[u8; FRAME]>>,
}

impl PhysMem {
    pub fn new() -> Self {
        PhysMem {
            frames: HashMap::new(),
        }
    }

    #[inline]
    fn frame(&self, page: u64) -> Option<&[u8; FRAME]> {
        self.frames.get(&page).map(|b| &**b)
    }

    #[inline]
    fn frame_mut(&mut self, page: u64) -> &mut [u8; FRAME] {
        self.frames
            .entry(page)
            .or_insert_with(|| Box::new([0u8; FRAME]))
    }

    /// Read a physical byte (0 if the frame was never written).
    #[inline]
    pub fn load8(&self, addr: u64) -> u8 {
        self.frame(addr >> 12)
            .map_or(0, |f| f[(addr & PAGE_MASK) as usize])
    }

    /// Write a physical byte, allocating the frame if needed.
    #[inline]
    pub fn store8(&mut self, addr: u64, v: u8) {
        self.frame_mut(addr >> 12)[(addr & PAGE_MASK) as usize] = v;
    }

    /// Read a little-endian physical quad-word (used for page-table entries; a
    /// PTE never straddles a frame because tables are frame-aligned).
    #[inline]
    pub fn load64(&self, addr: u64) -> u64 {
        let off = (addr & PAGE_MASK) as usize;
        match self.frame(addr >> 12) {
            Some(f) if off + 8 <= FRAME => u64::from_le_bytes(f[off..off + 8].try_into().unwrap()),
            _ => (0..8).fold(0u64, |acc, i| acc | (self.load8(addr + i) as u64) << (8 * i)),
        }
    }

    /// Write a little-endian physical quad-word.
    #[inline]
    pub fn store64(&mut self, addr: u64, v: u64) {
        let off = (addr & PAGE_MASK) as usize;
        if off + 8 <= FRAME {
            self.frame_mut(addr >> 12)[off..off + 8].copy_from_slice(&v.to_le_bytes());
        } else {
            for i in 0..8 {
                self.store8(addr + i, (v >> (8 * i)) as u8);
            }
        }
    }

    /// Copy `buf.len()` bytes out of physical memory starting at `addr`; the
    /// caller guarantees the span stays within one frame.
    #[inline]
    fn copy_out(&self, addr: u64, buf: &mut [u8]) {
        let off = (addr & PAGE_MASK) as usize;
        match self.frame(addr >> 12) {
            Some(f) => buf.copy_from_slice(&f[off..off + buf.len()]),
            None => buf.fill(0),
        }
    }

    /// Copy `data` into physical memory starting at `addr`; the caller
    /// guarantees the span stays within one frame.
    #[inline]
    fn copy_in(&mut self, addr: u64, data: &[u8]) {
        let off = (addr & PAGE_MASK) as usize;
        self.frame_mut(addr >> 12)[off..off + data.len()].copy_from_slice(data);
    }
}

impl Bus for PhysMem {
    #[inline]
    fn read(&mut self, addr: u64) -> u8 {
        self.load8(addr)
    }

    #[inline]
    fn write(&mut self, addr: u64, value: u8) {
        self.store8(addr, value);
    }

    // The CPU only issues a wide access that stays within one page, i.e. one
    // frame here — so these hit a single frame's slice.
    #[inline]
    fn read16(&mut self, addr: u64) -> u16 {
        let off = (addr & PAGE_MASK) as usize;
        match self.frame(addr >> 12) {
            Some(f) if off + 2 <= FRAME => u16::from_le_bytes([f[off], f[off + 1]]),
            _ => self.load8(addr) as u16 | (self.load8(addr.wrapping_add(1)) as u16) << 8,
        }
    }

    #[inline]
    fn read32(&mut self, addr: u64) -> u32 {
        let off = (addr & PAGE_MASK) as usize;
        match self.frame(addr >> 12) {
            Some(f) if off + 4 <= FRAME => u32::from_le_bytes(f[off..off + 4].try_into().unwrap()),
            _ => (0..4).fold(0u32, |acc, i| acc | (self.load8(addr + i) as u32) << (8 * i)),
        }
    }

    #[inline]
    fn read64(&mut self, addr: u64) -> u64 {
        self.load64(addr)
    }

    #[inline]
    fn write16(&mut self, addr: u64, value: u16) {
        let off = (addr & PAGE_MASK) as usize;
        if off + 2 <= FRAME {
            self.frame_mut(addr >> 12)[off..off + 2].copy_from_slice(&value.to_le_bytes());
        } else {
            self.store8(addr, value as u8);
            self.store8(addr.wrapping_add(1), (value >> 8) as u8);
        }
    }

    #[inline]
    fn write32(&mut self, addr: u64, value: u32) {
        let off = (addr & PAGE_MASK) as usize;
        if off + 4 <= FRAME {
            self.frame_mut(addr >> 12)[off..off + 4].copy_from_slice(&value.to_le_bytes());
        } else {
            for i in 0..4 {
                self.store8(addr + i, (value >> (8 * i)) as u8);
            }
        }
    }

    #[inline]
    fn write64(&mut self, addr: u64, value: u64) {
        self.store64(addr, value);
    }
}

/// What a VMA is, for `/proc/self/maps` and fault handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmaKind {
    /// ELF image (executable or interpreter).
    Image,
    /// The initial stack (grows down).
    Stack,
    /// `brk` heap.
    Heap,
    /// Anonymous or file-backed `mmap`.
    Mapping,
}

/// A mapped region of the guest linear address space.
#[derive(Debug, Clone, Copy)]
pub struct Vma {
    pub start: u64,
    pub end: u64,
    pub prot: u32,
    pub kind: VmaKind,
}

/// The process address space: page tables plus VMA bookkeeping.
pub struct AddressSpace {
    /// Physical base of the PML4 (the value loaded into `CR3`).
    pub cr3: u64,
    /// Next free physical frame (bump allocator).
    next_phys: u64,
    /// Mapped regions, for `/proc/self/maps` and fault classification.
    pub vmas: Vec<Vma>,
    /// Current program break (top of the heap).
    pub brk: u64,
    /// Initial program break (start of the heap region).
    pub brk_base: u64,
    /// Next free `mmap` address (bump allocator, grows up).
    pub mmap_top: u64,
}

impl AddressSpace {
    /// Create an address space with an empty PML4.
    pub fn new(mem: &mut PhysMem) -> Self {
        let mut a = AddressSpace {
            cr3: 0,
            next_phys: PAGE_SIZE, // leave physical frame 0 unused
            vmas: Vec::new(),
            brk: 0,
            brk_base: 0,
            mmap_top: MMAP_BASE,
        };
        a.cr3 = a.alloc_frame(mem);
        a
    }

    /// Allocate the next zeroed physical frame.
    fn alloc_frame(&mut self, mem: &mut PhysMem) -> u64 {
        let f = self.next_phys;
        self.next_phys += PAGE_SIZE;
        mem.frame_mut(f >> 12); // vivify (zeroed)
        f
    }

    /// The four 9-bit table indices of `lin` (PML4, PDPT, PD, PT).
    #[inline]
    fn indices(lin: u64) -> [u64; 4] {
        [
            (lin >> 39) & 0x1FF,
            (lin >> 30) & 0x1FF,
            (lin >> 21) & 0x1FF,
            (lin >> 12) & 0x1FF,
        ]
    }

    /// Walk the page tables for linear `lin`; returns its physical address, or
    /// `None` if any level is not present.
    pub fn resolve(&self, mem: &PhysMem, lin: u64) -> Option<u64> {
        let idx = Self::indices(lin);
        let mut table = self.cr3 & PHYS_MASK;
        for level in idx {
            let e = mem.load64(table + level * 8);
            if e & pte::P == 0 {
                return None;
            }
            table = e & PHYS_MASK;
        }
        Some(table | (lin & PAGE_MASK))
    }

    /// Read the raw leaf page-table entry for `lin` (0 if any level is
    /// unmapped) — carries the flag bits, unlike [`AddressSpace::resolve`].
    fn pte_entry(&self, mem: &PhysMem, lin: u64) -> u64 {
        let idx = Self::indices(lin);
        let mut table = self.cr3 & PHYS_MASK;
        for &level in &idx[..3] {
            let e = mem.load64(table + level * 8);
            if e & pte::P == 0 {
                return 0;
            }
            table = e & PHYS_MASK;
        }
        mem.load64(table + idx[3] * 8)
    }

    /// Install a leaf entry mapping linear page `lin` to physical frame `phys`
    /// with `prot` (and the `user` bit for the U/S check), allocating any
    /// missing interior tables. Interior entries are maximally permissive; the
    /// leaf decides protection.
    fn set_pte(&mut self, mem: &mut PhysMem, lin: u64, phys: u64, prot: u32, user: bool) {
        let idx = Self::indices(lin);
        let mut table = self.cr3 & PHYS_MASK;
        for &level in &idx[..3] {
            let addr = table + level * 8;
            let mut e = mem.load64(addr);
            if e & pte::P == 0 {
                let child = self.alloc_frame(mem);
                e = child | pte::P | pte::RW | pte::US;
                mem.store64(addr, e);
            }
            table = e & PHYS_MASK;
        }
        let mut flags = pte::P;
        if user {
            flags |= pte::US;
        }
        if prot & PROT_WRITE != 0 {
            flags |= pte::RW;
        }
        mem.store64(table + idx[3] * 8, (phys & PHYS_MASK) | flags);
    }

    /// Map `[start, start+len)` (page-rounded) with `prot`, allocating fresh
    /// frames but *not* recording a VMA. Pages already mapped keep their frame
    /// and *widen* their protection — the union of old and new — so adjacent
    /// ELF segments sharing a boundary page never lose write access regardless
    /// of mapping order. `user` sets the U/S bit.
    fn map_range(&mut self, mem: &mut PhysMem, start: u64, len: u64, prot: u32, user: bool) {
        let e = range_end(start, len);
        let mut lin = page_down(start);
        while lin < e {
            let (phys, prot) = match self.resolve(mem, lin) {
                None => (self.alloc_frame(mem), prot),
                Some(p) => {
                    let widened = if self.pte_entry(mem, lin) & pte::RW != 0 {
                        prot | PROT_WRITE
                    } else {
                        prot
                    };
                    (p & !PAGE_MASK, widened)
                }
            };
            self.set_pte(mem, lin, phys, prot, user);
            lin += PAGE_SIZE;
        }
    }

    /// Map `[start, start+len)` as user memory with `prot` and record a VMA.
    pub fn map(&mut self, mem: &mut PhysMem, start: u64, len: u64, prot: u32, kind: VmaKind) {
        self.map_range(mem, start, len, prot, true);
        self.vmas.push(Vma {
            start: page_down(start),
            end: range_end(start, len),
            prot,
            kind,
        });
    }

    /// Copy `data` into guest memory at linear `lin` (kernel privilege: bypasses
    /// user page protection, so read-only segments can be filled). Walks the
    /// page tables once per page. Returns `false` if any target page is
    /// unmapped.
    pub fn write_bytes(&self, mem: &mut PhysMem, lin: u64, data: &[u8]) -> bool {
        let mut done = 0usize;
        while done < data.len() {
            let addr = lin.wrapping_add(done as u64);
            let Some(phys) = self.resolve(mem, addr) else {
                return false;
            };
            let n = (PAGE_SIZE - (addr & PAGE_MASK)).min((data.len() - done) as u64) as usize;
            mem.copy_in(phys, &data[done..done + n]);
            done += n;
        }
        true
    }

    /// Read `buf.len()` bytes from guest memory at linear `lin`. Walks the page
    /// tables once per page. Returns `false` if any source page is unmapped.
    pub fn read_bytes(&self, mem: &PhysMem, lin: u64, buf: &mut [u8]) -> bool {
        let mut done = 0usize;
        while done < buf.len() {
            let addr = lin.wrapping_add(done as u64);
            let Some(phys) = self.resolve(mem, addr) else {
                return false;
            };
            let n = (PAGE_SIZE - (addr & PAGE_MASK)).min((buf.len() - done) as u64) as usize;
            mem.copy_out(phys, &mut buf[done..done + n]);
            done += n;
        }
        true
    }

    /// Read a NUL-terminated string starting at linear `lin` (without the NUL),
    /// bounded by `max` bytes.
    pub fn read_cstr(&self, mem: &PhysMem, lin: u64, max: usize) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for i in 0..max as u64 {
            let p = self.resolve(mem, lin.wrapping_add(i))?;
            let b = mem.load8(p);
            if b == 0 {
                return Some(out);
            }
            out.push(b);
        }
        Some(out)
    }

    /// `mmap` an anonymous (or to-be-filled) region of `len` bytes with `prot`.
    /// Bump-allocates from the arena; returns the base address, or 0 if the
    /// arena would collide with the stack region (the caller reports `-ENOMEM`).
    pub fn mmap(&mut self, mem: &mut PhysMem, len: u64, prot: u32) -> u64 {
        let addr = self.mmap_top;
        let size = page_up(len);
        if addr.saturating_add(size) > STACK_LIMIT {
            return 0;
        }
        self.map(mem, addr, size, prot, VmaKind::Mapping);
        self.mmap_top += size;
        addr
    }

    /// `mmap` at a fixed address (`MAP_FIXED`), for file-backed segments.
    pub fn mmap_fixed(&mut self, mem: &mut PhysMem, addr: u64, len: u64, prot: u32) {
        self.map(mem, addr, len, prot, VmaKind::Mapping);
        self.mmap_top = self.mmap_top.max(range_end(addr, len));
    }

    /// `mprotect`: change the protection of already-mapped pages in the range.
    /// Unmapped pages are skipped. The caller must invalidate the CPU TLB.
    pub fn protect(&mut self, mem: &mut PhysMem, start: u64, len: u64, prot: u32) {
        let e = range_end(start, len);
        let mut lin = page_down(start);
        while lin < e {
            if let Some(phys) = self.resolve(mem, lin) {
                self.set_pte(mem, lin, phys & !PAGE_MASK, prot, true);
            }
            lin += PAGE_SIZE;
        }
    }

    /// `munmap`: mark leaf entries in the range not-present. The caller must
    /// invalidate the CPU TLB.
    pub fn unmap(&mut self, mem: &mut PhysMem, start: u64, len: u64) {
        let s = page_down(start);
        let e = range_end(start, len);
        let mut lin = s;
        while lin < e {
            let idx = Self::indices(lin);
            let mut table = self.cr3 & PHYS_MASK;
            let mut present = true;
            for &level in &idx[..3] {
                let ent = mem.load64(table + level * 8);
                if ent & pte::P == 0 {
                    present = false;
                    break;
                }
                table = ent & PHYS_MASK;
            }
            if present {
                mem.store64(table + idx[3] * 8, 0);
            }
            lin += PAGE_SIZE;
        }
        self.vmas.retain(|v| !(v.start >= s && v.end <= e));
    }

    /// Set the initial and current `brk` to `addr` (called by the loader).
    pub fn init_brk(&mut self, addr: u64) {
        let b = page_up(addr);
        self.brk = b;
        self.brk_base = b;
    }

    /// The `brk` syscall: query (arg 0) or move the program break. Returns the
    /// resulting break.
    pub fn set_brk(&mut self, mem: &mut PhysMem, new: u64) -> u64 {
        if new < self.brk_base || new > MMAP_BASE {
            return self.brk;
        }
        let old = page_up(self.brk);
        let want = page_up(new);
        if want > old {
            self.map(mem, old, want - old, PROT_READ | PROT_WRITE, VmaKind::Heap);
        }
        self.brk = new;
        new
    }

    /// Whether `lin` sits just below the stack region and within the growth
    /// limit — a stack-expansion fault rather than a wild access. Returns the
    /// page base to grow down to.
    pub fn is_stack_growth(&self, lin: u64) -> Option<u64> {
        let stack = self.vmas.iter().find(|v| v.kind == VmaKind::Stack)?;
        if lin >= STACK_LIMIT && lin < stack.start {
            Some(page_down(lin))
        } else {
            None
        }
    }

    /// Grow the stack down to include `new_bottom`, extending the stack VMA.
    pub fn grow_stack(&mut self, mem: &mut PhysMem, new_bottom: u64) {
        let old = self
            .vmas
            .iter()
            .find(|v| v.kind == VmaKind::Stack)
            .map(|v| v.start)
            .unwrap_or(STACK_TOP);
        let bottom = page_down(new_bottom);
        if bottom < old {
            self.map_range(mem, bottom, old - bottom, PROT_READ | PROT_WRITE, true);
            if let Some(v) = self.vmas.iter_mut().find(|v| v.kind == VmaKind::Stack) {
                v.start = bottom;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_resolve_roundtrip() {
        let mut mem = PhysMem::new();
        let mut a = AddressSpace::new(&mut mem);
        // A high canonical address exercises all four table levels.
        let lin = 0x0000_5555_5555_6000;
        a.map(&mut mem, lin, 0x2000, PROT_READ | PROT_WRITE, VmaKind::Image);
        assert!(a.resolve(&mem, lin).is_some());
        assert!(a.write_bytes(&mut mem, lin + 0x123, b"hello"));
        let mut buf = [0u8; 5];
        assert!(a.read_bytes(&mem, lin + 0x123, &mut buf));
        assert_eq!(&buf, b"hello");
        // A distinct page in the same 2 MiB region shares interior tables.
        assert!(a.resolve(&mem, lin + 0x1000).is_some());
        // An unmapped neighbor resolves to nothing.
        assert!(a.resolve(&mem, lin + 0x100000).is_none());
    }

    #[test]
    fn brk_and_mmap() {
        let mut mem = PhysMem::new();
        let mut a = AddressSpace::new(&mut mem);
        a.init_brk(0x40_0000);
        assert_eq!(a.set_brk(&mut mem, 0), 0x40_0000, "query");
        let nb = a.set_brk(&mut mem, 0x40_5000);
        assert_eq!(nb, 0x40_5000);
        assert!(a.resolve(&mem, 0x40_4FFF).is_some());

        let m = a.mmap(&mut mem, 0x3000, PROT_READ | PROT_WRITE);
        assert_eq!(m, MMAP_BASE);
        assert!(a.resolve(&mem, m + 0x2FFF).is_some());
        let m2 = a.mmap(&mut mem, 1, PROT_READ);
        assert_eq!(m2, MMAP_BASE + 0x3000, "page-rounded bump");
    }

    #[test]
    fn huge_sizes_saturate_without_panic() {
        // Hostile lengths must not overflow the page-rounding arithmetic (a
        // debug-build panic). These run in debug, so a regression would panic.
        let mut mem = PhysMem::new();
        let mut a = AddressSpace::new(&mut mem);
        assert_eq!(a.mmap(&mut mem, u64::MAX, PROT_READ), 0, "absurd mmap is rejected");
        a.init_brk(0x40_0000);
        assert_eq!(a.set_brk(&mut mem, u64::MAX), 0x40_0000, "absurd brk is a no-op");
        // A map whose start+len wraps must not panic; it maps a saturated range.
        a.map(&mut mem, u64::MAX - 0x100, 0x1000, PROT_READ, VmaKind::Mapping);
    }

    #[test]
    fn stack_growth_classification() {
        let mut mem = PhysMem::new();
        let mut a = AddressSpace::new(&mut mem);
        a.map(
            &mut mem,
            STACK_TOP - 0x1000,
            0x1000,
            PROT_READ | PROT_WRITE,
            VmaKind::Stack,
        );
        // A touch just below the stack is a growth; far below is not.
        assert!(a.is_stack_growth(STACK_TOP - 0x2000).is_some());
        assert!(a.is_stack_growth(STACK_LIMIT - 1).is_none());
        a.grow_stack(&mut mem, STACK_TOP - 0x2000);
        assert!(a.resolve(&mem, STACK_TOP - 0x2000).is_some());
    }
}
