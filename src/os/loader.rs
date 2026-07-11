//! Minimal ELF32 (`EM_386`) loader: maps `PT_LOAD` segments, records what the
//! auxiliary vector needs, and reports `PT_INTERP` so the caller can load the
//! dynamic linker. Only the subset needed to run Linux i386 binaries is parsed.

use crate::os::abi::elf;
use crate::os::memory::{AddressSpace, PROT_EXEC, PROT_READ, PROT_WRITE, PhysMem};

/// Base address for a PIE / `ET_DYN` main executable.
pub const EXE_PIE_BASE: u32 = 0x5655_5000;
/// Base address for the ELF interpreter (`ld-linux.so.2`).
pub const INTERP_BASE: u32 = 0xF7C0_0000;

/// The result of loading one ELF image.
#[derive(Debug, Clone)]
pub struct ElfImage {
    /// Entry point (with load bias applied).
    pub entry: u32,
    /// Linear address of the program headers in the loaded image (`AT_PHDR`).
    pub phdr: u32,
    /// Program-header entry size and count (`AT_PHENT` / `AT_PHNUM`).
    pub phent: u16,
    pub phnum: u16,
    /// `PT_INTERP` path, if the image is dynamically linked.
    pub interp: Option<String>,
    /// Highest mapped address (rounded), used to seed `brk`.
    pub load_end: u32,
    /// The load bias applied (0 for `ET_EXEC`).
    pub base: u32,
}

#[inline]
fn rd_u16(d: &[u8], off: usize) -> Option<u16> {
    d.get(off..off + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
}

#[inline]
fn rd_u32(d: &[u8], off: usize) -> Option<u32> {
    d.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn prot_of(p_flags: u32) -> u32 {
    let mut p = 0;
    if p_flags & elf::PF_R != 0 {
        p |= PROT_READ;
    }
    if p_flags & elf::PF_W != 0 {
        p |= PROT_WRITE;
    }
    if p_flags & elf::PF_X != 0 {
        p |= PROT_EXEC;
    }
    p
}

/// Validate and load an ELF32 image into `aspace`. `bias` is added to every
/// address for `ET_DYN` images (ignored for `ET_EXEC`).
pub fn load(
    aspace: &mut AddressSpace,
    mem: &mut PhysMem,
    data: &[u8],
    bias: u32,
) -> Result<ElfImage, String> {
    if data.get(0..4) != Some(&elf::MAGIC) {
        return Err("not an ELF file".into());
    }
    if data.get(4).copied() != Some(elf::CLASS32) {
        return Err("not a 32-bit (ELFCLASS32) ELF".into());
    }
    if data.get(5).copied() != Some(elf::DATA_LE) {
        return Err("not little-endian".into());
    }
    let e_type = rd_u16(data, 16).ok_or("truncated header")?;
    let e_machine = rd_u16(data, 18).ok_or("truncated header")?;
    if e_machine != elf::EM_386 {
        return Err(format!("unsupported e_machine {e_machine} (need EM_386)"));
    }
    if e_type != elf::ET_EXEC && e_type != elf::ET_DYN {
        return Err(format!("unsupported e_type {e_type}"));
    }
    let bias = if e_type == elf::ET_DYN { bias } else { 0 };

    let e_entry = rd_u32(data, 24).ok_or("truncated header")?;
    let e_phoff = rd_u32(data, 28).ok_or("truncated header")?;
    let e_phentsize = rd_u16(data, 42).ok_or("truncated header")?;
    let e_phnum = rd_u16(data, 44).ok_or("truncated header")?;

    let mut interp = None;
    let mut load_end = 0u32;
    let mut phdr_from_pt = None;
    let mut phdr_in_load = None;

    for i in 0..e_phnum as usize {
        let base = e_phoff as usize + i * e_phentsize as usize;
        let p_type = rd_u32(data, base).ok_or("truncated phdr")?;
        let p_offset = rd_u32(data, base + 4).ok_or("truncated phdr")?;
        let p_vaddr = rd_u32(data, base + 8).ok_or("truncated phdr")?;
        let p_filesz = rd_u32(data, base + 16).ok_or("truncated phdr")?;
        let p_memsz = rd_u32(data, base + 20).ok_or("truncated phdr")?;
        let p_flags = rd_u32(data, base + 24).ok_or("truncated phdr")?;

        match p_type {
            elf::PT_LOAD => {
                let vaddr = p_vaddr.wrapping_add(bias);
                aspace.map(
                    mem,
                    vaddr,
                    p_memsz,
                    prot_of(p_flags),
                    crate::os::memory::VmaKind::Image,
                );
                let file = data
                    .get(p_offset as usize..(p_offset + p_filesz) as usize)
                    .ok_or("PT_LOAD file range out of bounds")?;
                if !aspace.write_bytes(mem, vaddr, file) {
                    return Err("PT_LOAD target unmapped".into());
                }
                load_end = load_end.max(vaddr.wrapping_add(p_memsz));
                // Program headers may lie inside this segment's file range.
                if e_phoff >= p_offset && e_phoff < p_offset + p_filesz {
                    phdr_in_load = Some(vaddr + (e_phoff - p_offset));
                }
            }
            elf::PT_INTERP => {
                let s = data
                    .get(p_offset as usize..(p_offset + p_filesz) as usize)
                    .ok_or("PT_INTERP out of bounds")?;
                let s = s.split(|&b| b == 0).next().unwrap_or(s);
                interp = Some(String::from_utf8_lossy(s).into_owned());
            }
            elf::PT_PHDR => {
                phdr_from_pt = Some(p_vaddr.wrapping_add(bias));
            }
            _ => {}
        }
    }

    let phdr = phdr_from_pt.or(phdr_in_load).unwrap_or(bias + e_phoff);

    Ok(ElfImage {
        entry: e_entry.wrapping_add(bias),
        phdr,
        phent: e_phentsize,
        phnum: e_phnum,
        interp,
        load_end,
        base: bias,
    })
}
