//! Guest process address space.
//!
//! The process runs under real 386 paging (`CR0.PG = 1`, flat ring-3
//! segments), so the CPU's validated page-table walker enforces permissions
//! and raises `#PF` — which the OS layer traps to demand-grow the stack or
//! deliver `SIGSEGV`. Three pieces cooperate:
//!
//! * [`PhysMem`] — a sparse, page-granular guest-*physical* frame store; the
//!   [`Bus`] the CPU accesses after translation.
//! * page tables — a two-level directory built in physical memory (`CR3` points
//!   at it); the source of truth for linear→physical and permissions.
//! * [`AddressSpace`] — a frame allocator, the page tables, and a VMA list,
//!   exposing `map`/`mmap`/`brk` and host-side guest-memory access that bypasses
//!   user permissions (the kernel's privilege).

use std::collections::HashMap;

use crate::x86_32::Bus;

/// Page size (and physical frame size).
pub const PAGE_SIZE: u32 = 4096;
const FRAME: usize = 4096;
const PAGE_MASK: u32 = PAGE_SIZE - 1;

/// Round `addr` up to the next page boundary (saturating at 4 GiB).
#[inline]
fn page_up(addr: u64) -> u32 {
    ((addr + PAGE_MASK as u64) & !(PAGE_MASK as u64)).min(0xFFFF_F000) as u32
}

/// Round `addr` down to its page base.
#[inline]
fn page_down(addr: u32) -> u32 {
    addr & !PAGE_MASK
}

// --- Protection flags (match Linux PROT_*) -----------------------------------
pub const PROT_READ: u32 = 1;
pub const PROT_WRITE: u32 = 2;
pub const PROT_EXEC: u32 = 4;

// --- Page-table entry bits ---------------------------------------------------
mod pte {
    pub const P: u32 = 1 << 0;
    pub const RW: u32 = 1 << 1;
    pub const US: u32 = 1 << 2;
}

/// Sparse guest-physical memory: 4 KiB frames created on first touch. Reads of
/// never-written frames return 0 (open-bus-like), matching the infallible 386
/// [`Bus`] contract.
#[derive(Default)]
pub struct PhysMem {
    frames: HashMap<u32, Box<[u8; FRAME]>>,
}

impl PhysMem {
    pub fn new() -> Self {
        PhysMem {
            frames: HashMap::new(),
        }
    }

    #[inline]
    fn frame(&self, page: u32) -> Option<&[u8; FRAME]> {
        self.frames.get(&page).map(|b| &**b)
    }

    #[inline]
    fn frame_mut(&mut self, page: u32) -> &mut [u8; FRAME] {
        self.frames
            .entry(page)
            .or_insert_with(|| Box::new([0u8; FRAME]))
    }

    /// Read a physical byte (0 if the frame was never written).
    #[inline]
    pub fn load8(&self, addr: u32) -> u8 {
        self.frame(addr >> 12)
            .map_or(0, |f| f[(addr & PAGE_MASK) as usize])
    }

    /// Write a physical byte, allocating the frame if needed.
    #[inline]
    pub fn store8(&mut self, addr: u32, v: u8) {
        self.frame_mut(addr >> 12)[(addr & PAGE_MASK) as usize] = v;
    }

    /// Read a little-endian physical dword (used for page-table entries; a PTE
    /// never straddles a frame because tables are frame-aligned).
    #[inline]
    pub fn load32(&self, addr: u32) -> u32 {
        let off = (addr & PAGE_MASK) as usize;
        match self.frame(addr >> 12) {
            Some(f) if off + 4 <= FRAME => {
                u32::from_le_bytes([f[off], f[off + 1], f[off + 2], f[off + 3]])
            }
            _ => (0..4).fold(0u32, |acc, i| {
                acc | (self.load8(addr + i) as u32) << (8 * i)
            }),
        }
    }

    /// Write a little-endian physical dword.
    #[inline]
    pub fn store32(&mut self, addr: u32, v: u32) {
        let off = (addr & PAGE_MASK) as usize;
        if off + 4 <= FRAME {
            self.frame_mut(addr >> 12)[off..off + 4].copy_from_slice(&v.to_le_bytes());
        } else {
            for i in 0..4 {
                self.store8(addr + i, (v >> (8 * i)) as u8);
            }
        }
    }

    /// Copy `buf.len()` bytes out of physical memory starting at `addr`; the
    /// caller guarantees the span stays within one frame.
    #[inline]
    fn copy_out(&self, addr: u32, buf: &mut [u8]) {
        let off = (addr & PAGE_MASK) as usize;
        match self.frame(addr >> 12) {
            Some(f) => buf.copy_from_slice(&f[off..off + buf.len()]),
            None => buf.fill(0),
        }
    }

    /// Copy `data` into physical memory starting at `addr`; the caller
    /// guarantees the span stays within one frame.
    #[inline]
    fn copy_in(&mut self, addr: u32, data: &[u8]) {
        let off = (addr & PAGE_MASK) as usize;
        self.frame_mut(addr >> 12)[off..off + data.len()].copy_from_slice(data);
    }
}

impl Bus for PhysMem {
    #[inline]
    fn read(&mut self, addr: u32) -> u8 {
        self.load8(addr)
    }

    #[inline]
    fn write(&mut self, addr: u32, value: u8) {
        self.store8(addr, value);
    }

    // The CPU only issues a wide access that stays within one page, i.e. one
    // frame here — so these hit a single frame's slice.
    #[inline]
    fn read16(&mut self, addr: u32) -> u16 {
        let off = (addr & PAGE_MASK) as usize;
        match self.frame(addr >> 12) {
            Some(f) if off + 2 <= FRAME => u16::from_le_bytes([f[off], f[off + 1]]),
            _ => self.load8(addr) as u16 | (self.load8(addr.wrapping_add(1)) as u16) << 8,
        }
    }

    #[inline]
    fn read32(&mut self, addr: u32) -> u32 {
        self.load32(addr)
    }

    #[inline]
    fn write16(&mut self, addr: u32, value: u16) {
        let off = (addr & PAGE_MASK) as usize;
        if off + 2 <= FRAME {
            self.frame_mut(addr >> 12)[off..off + 2].copy_from_slice(&value.to_le_bytes());
        } else {
            self.store8(addr, value as u8);
            self.store8(addr.wrapping_add(1), (value >> 8) as u8);
        }
    }

    #[inline]
    fn write32(&mut self, addr: u32, value: u32) {
        self.store32(addr, value);
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
    /// Kernel structures (GDT); not user-accessible.
    System,
}

/// A mapped region of the guest linear address space.
#[derive(Debug, Clone, Copy)]
pub struct Vma {
    pub start: u32,
    pub end: u32,
    pub prot: u32,
    pub kind: VmaKind,
}

/// Base of the `mmap` arena (grows up). Not a realistic i386 layout, but any
/// base works: `ld.so` uses whatever address `mmap` returns.
pub const MMAP_BASE: u32 = 0x4000_0000;
/// Top of the initial stack.
pub const STACK_TOP: u32 = 0xC000_0000;
/// Lowest address the stack may auto-grow to (8 MiB, like the default rlimit).
pub const STACK_LIMIT: u32 = STACK_TOP - 0x0080_0000;

/// The process address space: page tables plus VMA bookkeeping.
pub struct AddressSpace {
    /// Physical base of the page directory (the value loaded into `CR3`).
    pub cr3: u32,
    /// Next free physical frame (bump allocator).
    next_phys: u32,
    /// Mapped regions, for `/proc/self/maps` and fault classification.
    pub vmas: Vec<Vma>,
    /// Current program break (top of the heap).
    pub brk: u32,
    /// Initial program break (start of the heap region).
    pub brk_base: u32,
    /// Next free `mmap` address (bump allocator, grows up).
    pub mmap_top: u32,
}

impl AddressSpace {
    /// Create an address space with an empty page directory.
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
    fn alloc_frame(&mut self, mem: &mut PhysMem) -> u32 {
        let f = self.next_phys;
        self.next_phys += PAGE_SIZE;
        mem.frame_mut(f >> 12); // vivify (zeroed)
        f
    }

    /// Walk the page tables for linear `lin`; returns its physical address, or
    /// `None` if unmapped.
    pub fn resolve(&self, mem: &PhysMem, lin: u32) -> Option<u32> {
        let pde = mem.load32(self.cr3 + (lin >> 22) * 4);
        if pde & pte::P == 0 {
            return None;
        }
        let pt = pde & !PAGE_MASK;
        let entry = mem.load32(pt + ((lin >> 12) & 0x3FF) * 4);
        if entry & pte::P == 0 {
            return None;
        }
        Some((entry & !PAGE_MASK) | (lin & PAGE_MASK))
    }

    /// Install a page-table entry mapping linear page `lin` to physical frame
    /// `phys` with `prot` (and the `user` bit for the U/S check).
    fn set_pte(&mut self, mem: &mut PhysMem, lin: u32, phys: u32, prot: u32, user: bool) {
        let pde_addr = self.cr3 + (lin >> 22) * 4;
        let mut pde = mem.load32(pde_addr);
        if pde & pte::P == 0 {
            let pt = self.alloc_frame(mem);
            // Directory entries are maximally permissive; the leaf PTE decides.
            pde = pt | pte::P | pte::RW | pte::US;
            mem.store32(pde_addr, pde);
        }
        let pt = pde & !PAGE_MASK;
        let mut flags = pte::P;
        if user {
            flags |= pte::US;
        }
        if prot & PROT_WRITE != 0 {
            flags |= pte::RW;
        }
        mem.store32(pt + ((lin >> 12) & 0x3FF) * 4, phys | flags);
    }

    /// Read the raw page-table entry for `lin` (0 if unmapped).
    fn pte_entry(&self, mem: &PhysMem, lin: u32) -> u32 {
        let pde = mem.load32(self.cr3 + (lin >> 22) * 4);
        if pde & pte::P == 0 {
            return 0;
        }
        mem.load32((pde & !PAGE_MASK) + ((lin >> 12) & 0x3FF) * 4)
    }

    /// Map `[start, start+len)` (page-rounded) with `prot`, allocating fresh
    /// frames but *not* recording a VMA. Pages already mapped keep their frame
    /// and *widen* their protection — the union of old and new — so adjacent
    /// ELF segments sharing a boundary page never lose write access regardless
    /// of mapping order. `user` sets the U/S bit.
    fn map_range(&mut self, mem: &mut PhysMem, start: u32, len: u32, prot: u32, user: bool) {
        let s = page_down(start);
        let e = page_up(start as u64 + len as u64);
        let mut lin = s;
        while lin < e {
            let (phys, prot) = match self.resolve(mem, lin) {
                None => (self.alloc_frame(mem), prot),
                Some(p) => {
                    // Union the existing write permission into the new prot.
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
    pub fn map(&mut self, mem: &mut PhysMem, start: u32, len: u32, prot: u32, kind: VmaKind) {
        self.map_range(mem, start, len, prot, true);
        self.vmas.push(Vma {
            start: page_down(start),
            end: page_up(start as u64 + len as u64),
            prot,
            kind,
        });
    }

    /// Map a page-aligned kernel region (U/S clear): the GDT lives here, so a
    /// ring-3 access faults but the CPU's implicit (supervisor) reads succeed.
    pub fn map_kernel(&mut self, mem: &mut PhysMem, start: u32, len: u32) {
        self.map_range(mem, start, len, PROT_READ | PROT_WRITE, false);
        self.vmas.push(Vma {
            start: page_down(start),
            end: page_up(start as u64 + len as u64),
            prot: PROT_READ | PROT_WRITE,
            kind: VmaKind::System,
        });
    }

    /// Copy `data` into guest memory at linear `lin` (kernel privilege: bypasses
    /// user page protection, so read-only segments can be filled). Walks the
    /// page tables once per page, not per byte. Returns `false` if any target
    /// page is unmapped.
    pub fn write_bytes(&self, mem: &mut PhysMem, lin: u32, data: &[u8]) -> bool {
        let mut done = 0usize;
        while done < data.len() {
            let addr = lin.wrapping_add(done as u32);
            let Some(phys) = self.resolve(mem, addr) else {
                return false;
            };
            let n = (PAGE_SIZE - (addr & PAGE_MASK)).min((data.len() - done) as u32) as usize;
            mem.copy_in(phys, &data[done..done + n]);
            done += n;
        }
        true
    }

    /// Read `buf.len()` bytes from guest memory at linear `lin`. Walks the page
    /// tables once per page. Returns `false` if any source page is unmapped.
    pub fn read_bytes(&self, mem: &PhysMem, lin: u32, buf: &mut [u8]) -> bool {
        let mut done = 0usize;
        while done < buf.len() {
            let addr = lin.wrapping_add(done as u32);
            let Some(phys) = self.resolve(mem, addr) else {
                return false;
            };
            let n = (PAGE_SIZE - (addr & PAGE_MASK)).min((buf.len() - done) as u32) as usize;
            mem.copy_out(phys, &mut buf[done..done + n]);
            done += n;
        }
        true
    }

    /// Read a NUL-terminated string starting at linear `lin` (without the NUL),
    /// bounded by `max` bytes.
    pub fn read_cstr(&self, mem: &PhysMem, lin: u32, max: usize) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for i in 0..max as u32 {
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
    pub fn mmap(&mut self, mem: &mut PhysMem, len: u32, prot: u32) -> u32 {
        let addr = self.mmap_top;
        let size = page_up(len as u64);
        if addr as u64 + size as u64 > STACK_LIMIT as u64 {
            return 0;
        }
        self.map(mem, addr, size, prot, VmaKind::Mapping);
        self.mmap_top = self.mmap_top.wrapping_add(size);
        addr
    }

    /// `mprotect`: change the protection of already-mapped pages in the range.
    /// Pages that are not mapped are skipped. The caller must invalidate the CPU
    /// TLB afterward.
    pub fn protect(&mut self, mem: &mut PhysMem, start: u32, len: u32, prot: u32) {
        let s = page_down(start);
        let e = page_up(start as u64 + len as u64);
        let mut lin = s;
        while lin < e {
            if let Some(phys) = self.resolve(mem, lin) {
                self.set_pte(mem, lin, phys & !PAGE_MASK, prot, true);
            }
            lin += PAGE_SIZE;
        }
    }

    /// `munmap`: mark pages in the range not-present. The caller must invalidate
    /// the CPU TLB afterward.
    pub fn unmap(&mut self, mem: &mut PhysMem, start: u32, len: u32) {
        let s = page_down(start);
        let e = page_up(start as u64 + len as u64);
        let mut lin = s;
        while lin < e {
            let pde = mem.load32(self.cr3 + (lin >> 22) * 4);
            if pde & pte::P != 0 {
                let pt = pde & !PAGE_MASK;
                mem.store32(pt + ((lin >> 12) & 0x3FF) * 4, 0);
            }
            lin += PAGE_SIZE;
        }
        self.vmas.retain(|v| !(v.start >= s && v.end <= e));
    }

    /// `mmap` at a fixed address (`MAP_FIXED`), for the ELF interpreter and
    /// file-backed segments.
    pub fn mmap_fixed(&mut self, mem: &mut PhysMem, addr: u32, len: u32, prot: u32) {
        self.map(mem, addr, len, prot, VmaKind::Mapping);
        let end = page_up(addr as u64 + len as u64);
        self.mmap_top = self.mmap_top.max(end);
    }

    /// Set the initial and current `brk` to `addr` (called by the loader).
    pub fn init_brk(&mut self, addr: u32) {
        let b = page_up(addr as u64);
        self.brk = b;
        self.brk_base = b;
    }

    /// The `brk` syscall: query (arg 0) or move the program break. Returns the
    /// resulting break.
    pub fn set_brk(&mut self, mem: &mut PhysMem, new: u32) -> u32 {
        if new < self.brk_base || new > MMAP_BASE {
            return self.brk;
        }
        let old = page_up(self.brk as u64);
        let want = page_up(new as u64);
        if want > old {
            self.map(mem, old, want - old, PROT_READ | PROT_WRITE, VmaKind::Heap);
        }
        self.brk = new;
        new
    }

    /// Whether `lin` sits just below the stack region and within the growth
    /// limit — i.e. a stack-expansion fault rather than a wild access.
    pub fn is_stack_growth(&self, lin: u32) -> Option<u32> {
        let stack = self.vmas.iter().find(|v| v.kind == VmaKind::Stack)?;
        if lin >= STACK_LIMIT && lin < stack.start {
            Some(page_down(lin))
        } else {
            None
        }
    }

    /// Grow the stack down to include `new_bottom`, extending the stack VMA.
    pub fn grow_stack(&mut self, mem: &mut PhysMem, new_bottom: u32) {
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
