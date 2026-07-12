//! Minimal ELF64 (`EM_X86_64`) loader: maps `PT_LOAD` segments, records what
//! the auxiliary vector needs, and reports `PT_INTERP` so the caller can load
//! the dynamic linker. Only the subset needed to run Linux x86-64 binaries is
//! parsed; hand-rolled, since the framework carries no parsing crates.
//!
//! Note the ELF64 program-header layout differs from ELF32: `p_flags` sits
//! right after `p_type` (offset 4), and the 64-bit fields follow.

use crate::os64::abi::elf;
use crate::os64::memory::{
    AddressSpace, PROT_EXEC, PROT_READ, PROT_WRITE, PhysMem, USER_END, VmaKind,
};

/// Load base for a PIE / `ET_DYN` main executable (Linux's usual PIE base).
pub const EXE_PIE_BASE: u64 = 0x5555_5555_5000;
/// Load base for the ELF interpreter (`ld-linux-x86-64.so.2`).
pub const INTERP_BASE: u64 = 0x7FFF_F7C0_0000;

/// The result of loading one ELF image.
#[derive(Debug, Clone)]
pub struct ElfImage {
    /// Entry point (with load bias applied).
    pub entry: u64,
    /// Linear address of the program headers in the loaded image (`AT_PHDR`).
    pub phdr: u64,
    /// Program-header entry size and count (`AT_PHENT` / `AT_PHNUM`).
    pub phent: u16,
    pub phnum: u16,
    /// `PT_INTERP` path, if the image is dynamically linked.
    pub interp: Option<String>,
    /// Highest mapped address (rounded), used to seed `brk`.
    pub load_end: u64,
    /// The load bias applied (0 for `ET_EXEC`).
    pub base: u64,
}

#[inline]
fn rd_u16(d: &[u8], off: usize) -> Option<u16> {
    d.get(off..off + 2)
        .map(|s| u16::from_le_bytes(s.try_into().unwrap()))
}

#[inline]
fn rd_u32(d: &[u8], off: usize) -> Option<u32> {
    d.get(off..off + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

#[inline]
fn rd_u64(d: &[u8], off: usize) -> Option<u64> {
    d.get(off..off + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
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

/// Validate and load an ELF64 image into `aspace`. `bias` is added to every
/// address for `ET_DYN` images (ignored for `ET_EXEC`).
pub fn load(
    aspace: &mut AddressSpace,
    mem: &mut PhysMem,
    data: &[u8],
    bias: u64,
) -> Result<ElfImage, String> {
    if data.get(0..4) != Some(&elf::MAGIC) {
        return Err("not an ELF file".into());
    }
    if data.get(4).copied() != Some(elf::CLASS64) {
        return Err("not a 64-bit (ELFCLASS64) ELF".into());
    }
    if data.get(5).copied() != Some(elf::DATA_LE) {
        return Err("not little-endian".into());
    }
    let e_type = rd_u16(data, 16).ok_or("truncated header")?;
    let e_machine = rd_u16(data, 18).ok_or("truncated header")?;
    if e_machine != elf::EM_X86_64 {
        return Err(format!(
            "unsupported e_machine {e_machine} (need EM_X86_64)"
        ));
    }
    if e_type != elf::ET_EXEC && e_type != elf::ET_DYN {
        return Err(format!("unsupported e_type {e_type}"));
    }
    let bias = if e_type == elf::ET_DYN { bias } else { 0 };

    let e_entry = rd_u64(data, 24).ok_or("truncated header")?;
    let e_phoff = rd_u64(data, 32).ok_or("truncated header")?;
    let e_phentsize = rd_u16(data, 54).ok_or("truncated header")?;
    let e_phnum = rd_u16(data, 56).ok_or("truncated header")?;

    let mut interp = None;
    let mut load_end = 0u64;
    let mut phdr_from_pt = None;
    let mut phdr_in_load = None;

    for i in 0..e_phnum as usize {
        let base = e_phoff as usize + i * e_phentsize as usize;
        let p_type = rd_u32(data, base).ok_or("truncated phdr")?;
        let p_flags = rd_u32(data, base + 4).ok_or("truncated phdr")?;
        let p_offset = rd_u64(data, base + 8).ok_or("truncated phdr")?;
        let p_vaddr = rd_u64(data, base + 16).ok_or("truncated phdr")?;
        let p_filesz = rd_u64(data, base + 32).ok_or("truncated phdr")?;
        let p_memsz = rd_u64(data, base + 40).ok_or("truncated phdr")?;

        match p_type {
            elf::PT_LOAD => {
                let vaddr = p_vaddr.wrapping_add(bias);
                // Reject malformed geometry before it reaches the memory
                // manager: filesz must fit within memsz, and the mapped range
                // must stay inside the canonical lower half (this also rules
                // out the vaddr+memsz overflow that would otherwise wrap).
                if p_filesz > p_memsz {
                    return Err("PT_LOAD filesz exceeds memsz".into());
                }
                let mem_end = vaddr
                    .checked_add(p_memsz)
                    .filter(|&e| e <= USER_END)
                    .ok_or("PT_LOAD maps outside the user address space")?;
                aspace.map(mem, vaddr, p_memsz, prot_of(p_flags), VmaKind::Image);
                let end = p_offset.checked_add(p_filesz).ok_or("PT_LOAD overflow")?;
                let file = data
                    .get(p_offset as usize..end as usize)
                    .ok_or("PT_LOAD file range out of bounds")?;
                if !aspace.write_bytes(mem, vaddr, file) {
                    return Err("PT_LOAD target unmapped".into());
                }
                load_end = load_end.max(mem_end);
                if e_phoff >= p_offset && e_phoff < p_offset + p_filesz {
                    phdr_in_load = Some(vaddr + (e_phoff - p_offset));
                }
            }
            elf::PT_INTERP => {
                let end = (p_offset + p_filesz) as usize;
                let s = data
                    .get(p_offset as usize..end)
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
