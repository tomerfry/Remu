//! Sparse 4 GiB guest address space.
//!
//! Memory is a single-level table of 64 KiB chunks allocated on `map`, so a
//! full process image (text, data, heap, mmap area, stack) costs only the
//! pages it touches plus a 512 KiB pointer table. Byte access is one shift,
//! one table load and one bounds-free chunk load — this sits on the CPU's
//! per-instruction fetch path.
//!
//! The address space is little-endian and infallible on the CPU-facing
//! accessors: reads of unmapped memory return 0, writes are dropped, and the
//! first such access is latched (see [`AddressSpace::take_segv`]) so a runner
//! can terminate the guest SIGSEGV-style after the instruction completes.
//!
//! Host-side helpers return `Result<_, ()>` — the only failure is "touched
//! unmapped guest memory" (`EFAULT`-shaped), which carries no further data.
#![allow(clippy::result_unit_err)]

const CHUNK_SHIFT: u32 = 16;
const CHUNK_SIZE: usize = 1 << CHUNK_SHIFT; // 64 KiB
const CHUNK_COUNT: usize = 1 << (32 - CHUNK_SHIFT);
const CHUNK_MASK: u32 = CHUNK_SIZE as u32 - 1;

/// Allocation granularity reported to and assumed by the guest.
pub const PAGE_SIZE: u32 = 4096;
/// Exclusive top of the stack region (classic i386 `TASK_SIZE`).
pub const STACK_TOP: u32 = 0xC000_0000;
/// Reserved stack size below [`STACK_TOP`].
pub const STACK_SIZE: u32 = 8 * 1024 * 1024;
/// Where anonymous `mmap` allocations start (classic i386 mmap base).
pub const MMAP_BASE: u32 = 0x4000_0000;

/// An access to unmapped memory, latched for the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segv {
    /// The unmapped address.
    pub addr: u32,
    /// `true` for a write, `false` for a read.
    pub write: bool,
}

/// A sparse little-endian 4 GiB guest address space.
pub struct AddressSpace {
    /// 64 KiB chunks, allocated on map (`Option<Box<_>>` is pointer-sized).
    chunks: Vec<Option<Box<[u8; CHUNK_SIZE]>>>,
    /// Current program break (moved by `brk(2)`).
    pub brk: u32,
    /// Lowest legal program break (end of the loaded image).
    pub brk_base: u32,
    /// Bump cursor for anonymous `mmap` allocations.
    pub mmap_next: u32,
    /// First access to an unmapped address since the last `take_segv`.
    segv: Option<Segv>,
}

fn new_chunk() -> Box<[u8; CHUNK_SIZE]> {
    // Via a zeroed Vec so the chunk never lands on the stack.
    vec![0u8; CHUNK_SIZE].into_boxed_slice().try_into().unwrap()
}

/// Round `v` up to a multiple of `align` (a power of two); `None` on overflow.
pub fn align_up(v: u32, align: u32) -> Option<u32> {
    debug_assert!(align.is_power_of_two());
    v.checked_add(align - 1).map(|x| x & !(align - 1))
}

impl AddressSpace {
    /// Create an empty (fully unmapped) address space.
    pub fn new() -> Self {
        AddressSpace {
            chunks: std::iter::repeat_with(|| None).take(CHUNK_COUNT).collect(),
            brk: 0,
            brk_base: 0,
            mmap_next: MMAP_BASE,
            segv: None,
        }
    }

    // --- CPU-facing accessors (bus hot path) --------------------------------
    // Wide accessors require the caller not to cross a 4 KiB page boundary
    // (the x86_32 bus contract); every 4 KiB page lies inside one chunk.

    /// Read one byte; unmapped reads return 0 and latch a [`Segv`].
    #[inline]
    pub fn read8(&mut self, addr: u32) -> u8 {
        match &self.chunks[(addr >> CHUNK_SHIFT) as usize] {
            Some(c) => c[(addr & CHUNK_MASK) as usize],
            None => {
                self.note_segv(addr, false);
                0
            }
        }
    }

    /// Write one byte; unmapped writes are dropped and latch a [`Segv`].
    #[inline]
    pub fn write8(&mut self, addr: u32, value: u8) {
        match &mut self.chunks[(addr >> CHUNK_SHIFT) as usize] {
            Some(c) => c[(addr & CHUNK_MASK) as usize] = value,
            None => self.note_segv(addr, true),
        }
    }

    /// Read a little-endian word (must not cross a 4 KiB page).
    #[inline]
    pub fn read16(&mut self, addr: u32) -> u16 {
        debug_assert!(addr & 0xFFF <= 0xFFE, "page-crossing read16");
        let off = (addr & CHUNK_MASK) as usize;
        match &self.chunks[(addr >> CHUNK_SHIFT) as usize] {
            Some(c) => u16::from_le_bytes(c[off..off + 2].try_into().unwrap()),
            None => {
                self.note_segv(addr, false);
                0
            }
        }
    }

    /// Read a little-endian dword (must not cross a 4 KiB page).
    #[inline]
    pub fn read32(&mut self, addr: u32) -> u32 {
        debug_assert!(addr & 0xFFF <= 0xFFC, "page-crossing read32");
        let off = (addr & CHUNK_MASK) as usize;
        match &self.chunks[(addr >> CHUNK_SHIFT) as usize] {
            Some(c) => u32::from_le_bytes(c[off..off + 4].try_into().unwrap()),
            None => {
                self.note_segv(addr, false);
                0
            }
        }
    }

    /// Write a little-endian word (must not cross a 4 KiB page).
    #[inline]
    pub fn write16(&mut self, addr: u32, value: u16) {
        debug_assert!(addr & 0xFFF <= 0xFFE, "page-crossing write16");
        let off = (addr & CHUNK_MASK) as usize;
        match &mut self.chunks[(addr >> CHUNK_SHIFT) as usize] {
            Some(c) => c[off..off + 2].copy_from_slice(&value.to_le_bytes()),
            None => self.note_segv(addr, true),
        }
    }

    /// Write a little-endian dword (must not cross a 4 KiB page).
    #[inline]
    pub fn write32(&mut self, addr: u32, value: u32) {
        debug_assert!(addr & 0xFFF <= 0xFFC, "page-crossing write32");
        let off = (addr & CHUNK_MASK) as usize;
        match &mut self.chunks[(addr >> CHUNK_SHIFT) as usize] {
            Some(c) => c[off..off + 4].copy_from_slice(&value.to_le_bytes()),
            None => self.note_segv(addr, true),
        }
    }

    #[inline(never)]
    fn note_segv(&mut self, addr: u32, write: bool) {
        if self.segv.is_none() {
            self.segv = Some(Segv { addr, write });
        }
    }

    /// Take the latched unmapped access, if any (clears the latch).
    pub fn take_segv(&mut self) -> Option<Segv> {
        self.segv.take()
    }

    // --- Mapping -------------------------------------------------------------

    /// Ensure `[addr, addr+len)` is mapped (chunk-granular, zero-filled).
    pub fn map(&mut self, addr: u32, len: u32) {
        if len == 0 {
            return;
        }
        let first = (addr >> CHUNK_SHIFT) as usize;
        let last = (((addr as u64 + len as u64 - 1).min(u32::MAX as u64)) >> CHUNK_SHIFT) as usize;
        for chunk in &mut self.chunks[first..=last] {
            if chunk.is_none() {
                *chunk = Some(new_chunk());
            }
        }
    }

    /// Unmap every chunk *fully contained* in `[addr, addr+len)`; partially
    /// covered chunks stay mapped (allocation is chunk-granular).
    pub fn unmap(&mut self, addr: u32, len: u32) {
        let start = addr as u64;
        let end = start + len as u64;
        let first = start.div_ceil(CHUNK_SIZE as u64);
        let last = (end.min(1 << 32)) / CHUNK_SIZE as u64;
        for i in first..last {
            self.chunks[i as usize] = None;
        }
    }

    /// Whether the chunk containing `addr` is mapped.
    pub fn is_mapped(&self, addr: u32) -> bool {
        self.chunks[(addr >> CHUNK_SHIFT) as usize].is_some()
    }

    /// `brk(2)` semantics: 0 or an unsatisfiable value returns the current
    /// break; otherwise the break moves (mapping any new range) and the new
    /// value is returned. Shrinking moves the break without unmapping.
    pub fn set_brk(&mut self, new: u32) -> u32 {
        if new < self.brk_base || new > MMAP_BASE {
            return self.brk;
        }
        if new > self.brk {
            self.map(self.brk, new - self.brk);
        }
        self.brk = new;
        self.brk
    }

    /// Anonymous `mmap`: page-aligns `len`, honors a fixed `addr`, otherwise
    /// bump-allocates from [`MMAP_BASE`]. Fails (`ENOMEM`-style) when the
    /// bump cursor would run into the stack region.
    pub fn mmap(&mut self, addr: u32, len: u32, fixed: bool) -> Result<u32, ()> {
        if len == 0 {
            return Err(());
        }
        let len = align_up(len, PAGE_SIZE).ok_or(())?;
        if fixed {
            if addr & (PAGE_SIZE - 1) != 0 || addr.checked_add(len - 1).is_none() {
                return Err(());
            }
            self.map(addr, len);
            return Ok(addr);
        }
        let base = self.mmap_next;
        let end = base as u64 + len as u64;
        if end > (STACK_TOP - STACK_SIZE) as u64 {
            return Err(());
        }
        self.map(base, len);
        self.mmap_next = end as u32;
        Ok(base)
    }

    // --- Host-side helpers (loader, syscall marshalling) ---------------------
    // These handle any alignment and length; `Err` means some byte fell on
    // unmapped memory (`EFAULT`-style). They never latch a `Segv` — they are
    // host accesses, not guest execution.

    /// Copy `data` into guest memory. On `Err` a prefix may have been written.
    pub fn write_bytes(&mut self, addr: u32, data: &[u8]) -> Result<(), ()> {
        let mut addr = addr as u64;
        let mut data = data;
        while !data.is_empty() {
            if addr >= 1 << 32 {
                return Err(());
            }
            let off = (addr as u32 & CHUNK_MASK) as usize;
            let n = data.len().min(CHUNK_SIZE - off);
            match &mut self.chunks[(addr as u32 >> CHUNK_SHIFT) as usize] {
                Some(c) => c[off..off + n].copy_from_slice(&data[..n]),
                None => return Err(()),
            }
            addr += n as u64;
            data = &data[n..];
        }
        Ok(())
    }

    /// Copy guest memory into `buf`. On `Err` a prefix may have been read.
    pub fn read_bytes(&mut self, addr: u32, buf: &mut [u8]) -> Result<(), ()> {
        let mut addr = addr as u64;
        let mut buf = buf;
        while !buf.is_empty() {
            if addr >= 1 << 32 {
                return Err(());
            }
            let off = (addr as u32 & CHUNK_MASK) as usize;
            let n = buf.len().min(CHUNK_SIZE - off);
            match &self.chunks[(addr as u32 >> CHUNK_SHIFT) as usize] {
                Some(c) => buf[..n].copy_from_slice(&c[off..off + n]),
                None => return Err(()),
            }
            addr += n as u64;
            buf = &mut buf[n..];
        }
        Ok(())
    }

    /// Read a little-endian dword at any alignment (0 if unmapped).
    pub fn read_u32(&mut self, addr: u32) -> u32 {
        let mut b = [0u8; 4];
        let _ = self.read_bytes(addr, &mut b);
        u32::from_le_bytes(b)
    }

    /// Write a little-endian dword at any alignment (dropped if unmapped).
    pub fn write_u32(&mut self, addr: u32, value: u32) {
        let _ = self.write_bytes(addr, &value.to_le_bytes());
    }

    /// Read a NUL-terminated string (without the NUL), capped at 4096 bytes.
    /// `Err` on unmapped memory or an unterminated string.
    pub fn read_cstr(&mut self, addr: u32) -> Result<Vec<u8>, ()> {
        const MAX: usize = 4096;
        let mut out = Vec::new();
        let mut a = addr as u64;
        loop {
            if a >= 1 << 32 || out.len() >= MAX {
                return Err(());
            }
            let b = match &self.chunks[(a as u32 >> CHUNK_SHIFT) as usize] {
                Some(c) => c[(a as u32 & CHUNK_MASK) as usize],
                None => return Err(()),
            };
            if b == 0 {
                return Ok(out);
            }
            out.push(b);
            a += 1;
        }
    }
}

impl Default for AddressSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_zero_fills_and_roundtrips() {
        let mut m = AddressSpace::new();
        m.map(0x0804_8000, 0x2000);
        assert!(m.is_mapped(0x0804_8000));
        assert_eq!(m.read8(0x0804_8123), 0, "fresh chunks are zeroed");
        m.write8(0x0804_8123, 0xAB);
        assert_eq!(m.read8(0x0804_8123), 0xAB);
        m.write32(0x0804_9000, 0x1234_5678);
        assert_eq!(m.read32(0x0804_9000), 0x1234_5678);
        assert_eq!(m.read16(0x0804_9002), 0x1234);
        assert!(m.take_segv().is_none());
    }

    #[test]
    fn wide_access_at_chunk_boundary_pages() {
        let mut m = AddressSpace::new();
        m.map(0x0001_0000, 0x2_0000);
        // Highest non-crossing dword of a 4 KiB page inside a chunk.
        m.write32(0x0001_FFFC, 0xDEAD_BEEF);
        assert_eq!(m.read32(0x0001_FFFC), 0xDEAD_BEEF);
        // And bytes straddling the chunk boundary via the generic helpers.
        m.write_bytes(0x0001_FFFE, &[1, 2, 3, 4]).unwrap();
        let mut b = [0u8; 4];
        m.read_bytes(0x0001_FFFE, &mut b).unwrap();
        assert_eq!(b, [1, 2, 3, 4]);
    }

    #[test]
    fn unmapped_access_latches_first_segv() {
        let mut m = AddressSpace::new();
        assert_eq!(m.read8(0x1000), 0);
        m.write8(0x2000, 0xFF);
        assert_eq!(
            m.take_segv(),
            Some(Segv {
                addr: 0x1000,
                write: false
            })
        );
        assert!(m.take_segv().is_none(), "take clears the latch");
        m.write8(0x3000, 1);
        assert_eq!(
            m.take_segv(),
            Some(Segv {
                addr: 0x3000,
                write: true
            })
        );
    }

    #[test]
    fn brk_grows_maps_and_rejects_bad_values() {
        let mut m = AddressSpace::new();
        m.brk_base = 0x0805_0000;
        m.brk = 0x0805_0000;
        assert_eq!(m.set_brk(0), 0x0805_0000, "query form");
        assert_eq!(m.set_brk(0x0805_4000), 0x0805_4000);
        assert!(m.is_mapped(0x0805_3FFF));
        assert_eq!(m.set_brk(0x0805_0000 - 1), 0x0805_4000, "below base");
        assert_eq!(m.set_brk(MMAP_BASE + 1), 0x0805_4000, "into mmap area");
        assert_eq!(m.set_brk(0x0805_1000), 0x0805_1000, "shrink moves break");
    }

    #[test]
    fn mmap_bump_fixed_and_munmap() {
        let mut m = AddressSpace::new();
        let a = m.mmap(0, 100, false).unwrap();
        assert_eq!(a, MMAP_BASE);
        let b = m.mmap(0, 0x1000, false).unwrap();
        assert_eq!(b, MMAP_BASE + 0x1000, "len was page-rounded");
        assert!(m.is_mapped(a) && m.is_mapped(b));

        let f = m.mmap(0x5000_0000, 0x1_0000, true).unwrap();
        assert_eq!(f, 0x5000_0000);
        assert!(
            m.mmap(0x5000_0001, 0x1000, true).is_err(),
            "unaligned fixed"
        );
        assert!(m.mmap(0, 0, false).is_err(), "zero length");
        assert!(m.mmap(0, u32::MAX, false).is_err(), "collides with stack");

        m.unmap(0x5000_0000, 0x1_0000);
        assert!(!m.is_mapped(0x5000_0000));
        // Partial chunk coverage keeps the chunk.
        m.map(0x6000_0000, CHUNK_SIZE as u32);
        m.unmap(0x6000_0000, 0x1000);
        assert!(m.is_mapped(0x6000_0000));
    }

    #[test]
    fn cstr_and_u32_helpers() {
        let mut m = AddressSpace::new();
        m.map(0x1_0000, 0x1000);
        m.write_bytes(0x1_0000, b"hello\0").unwrap();
        assert_eq!(m.read_cstr(0x1_0000).unwrap(), b"hello");
        assert_eq!(m.read_cstr(0x1_0005).unwrap(), b"");
        assert!(m.read_cstr(0x9000_0000).is_err(), "unmapped");
        m.write_u32(0x1_0001, 0xCAFE_F00D); // unaligned is fine host-side
        assert_eq!(m.read_u32(0x1_0001), 0xCAFE_F00D);
        assert!(m.write_bytes(0x9000_0000, &[1]).is_err());
        assert!(m.take_segv().is_none(), "host helpers never latch");
    }
}
