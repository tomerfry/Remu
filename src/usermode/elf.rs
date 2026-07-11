//! Minimal ELF32 executable loading: just enough to map a statically linked
//! `ET_EXEC` image (ehdr + program headers; sections are irrelevant at run
//! time). Hand-rolled on purpose — the framework carries no parsing crates.

use super::addr_space::{AddressSpace, PAGE_SIZE, align_up};

/// `e_machine` for Intel 80386.
pub const EM_386: u16 = 3;
/// Loadable program header type.
pub const PT_LOAD: u32 = 1;

/// A program header (the fields user-mode loading needs).
#[derive(Debug, Clone, Copy)]
pub struct Phdr {
    pub p_type: u32,
    pub offset: u32,
    pub vaddr: u32,
    pub filesz: u32,
    pub memsz: u32,
    pub flags: u32,
}

/// A parsed ELF32 executable header + program header table.
#[derive(Debug)]
pub struct Elf32 {
    pub entry: u32,
    pub phoff: u32,
    pub phnum: u16,
    pub machine: u16,
    pub phdrs: Vec<Phdr>,
}

/// Why an image was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    NotElf,
    Not32Bit,
    NotLittleEndian,
    /// Not `ET_EXEC` — dynamic/PIE executables are not supported.
    NotExec,
    /// `e_machine` does not match the emulated CPU.
    WrongMachine(u16),
    /// A header or segment reaches outside the file.
    Truncated,
    NoLoadSegments,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::NotElf => write!(f, "not an ELF file"),
            LoadError::Not32Bit => write!(f, "not a 32-bit (ELFCLASS32) ELF"),
            LoadError::NotLittleEndian => write!(f, "not a little-endian ELF"),
            LoadError::NotExec => {
                write!(
                    f,
                    "not an ET_EXEC image (dynamic/PIE executables are unsupported)"
                )
            }
            LoadError::WrongMachine(m) => write!(f, "wrong e_machine {m} for this CPU"),
            LoadError::Truncated => write!(f, "header or segment reaches outside the file"),
            LoadError::NoLoadSegments => write!(f, "no PT_LOAD segments"),
        }
    }
}

fn u16le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}

fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

/// Parse and validate an ELF32 `ET_EXEC` image. `e_machine` is reported, not
/// checked — the caller compares it against the emulated CPU's.
pub fn parse(bytes: &[u8]) -> Result<Elf32, LoadError> {
    if bytes.len() < 4 || &bytes[..4] != b"\x7fELF" {
        return Err(LoadError::NotElf);
    }
    if bytes.len() < 52 {
        return Err(LoadError::Truncated);
    }
    if bytes[4] != 1 {
        return Err(LoadError::Not32Bit);
    }
    if bytes[5] != 1 {
        return Err(LoadError::NotLittleEndian);
    }
    if u16le(bytes, 16) != 2 {
        return Err(LoadError::NotExec);
    }
    let machine = u16le(bytes, 18);
    let entry = u32le(bytes, 24);
    let phoff = u32le(bytes, 28);
    let phentsize = u16le(bytes, 42);
    let phnum = u16le(bytes, 44);
    if phentsize != 32 {
        return Err(LoadError::Truncated);
    }
    let table = phoff as u64 + phnum as u64 * 32;
    if table > bytes.len() as u64 {
        return Err(LoadError::Truncated);
    }

    let mut phdrs = Vec::with_capacity(phnum as usize);
    for i in 0..phnum as usize {
        let p = phoff as usize + i * 32;
        let ph = Phdr {
            p_type: u32le(bytes, p),
            offset: u32le(bytes, p + 4),
            vaddr: u32le(bytes, p + 8),
            filesz: u32le(bytes, p + 16),
            memsz: u32le(bytes, p + 20),
            flags: u32le(bytes, p + 24),
        };
        if ph.p_type == PT_LOAD {
            let file_end = ph.offset as u64 + ph.filesz as u64;
            let mem_end = ph.vaddr as u64 + ph.memsz as u64;
            if file_end > bytes.len() as u64 || ph.filesz > ph.memsz || mem_end > 1 << 32 {
                return Err(LoadError::Truncated);
            }
        }
        phdrs.push(ph);
    }
    if !phdrs.iter().any(|p| p.p_type == PT_LOAD) {
        return Err(LoadError::NoLoadSegments);
    }
    Ok(Elf32 {
        entry,
        phoff,
        phnum,
        machine,
        phdrs,
    })
}

/// Map every `PT_LOAD` segment of a [`parse`]d image into `mem` (fresh chunks
/// are zeroed, so BSS needs no explicit fill). Returns
/// `(entry, brk_start, phdr_vaddr)` where `brk_start` is the page-aligned end
/// of the image and `phdr_vaddr` is the program header table's guest address
/// if some `PT_LOAD` covers it (for `AT_PHDR`).
pub fn load(elf: &Elf32, bytes: &[u8], mem: &mut AddressSpace) -> (u32, u32, Option<u32>) {
    let mut image_end = 0u32;
    let mut phdr_vaddr = None;
    let table_end = elf.phoff as u64 + elf.phnum as u64 * 32;
    for ph in elf.phdrs.iter().filter(|p| p.p_type == PT_LOAD) {
        let page_base = ph.vaddr & !(PAGE_SIZE - 1);
        mem.map(page_base, ph.vaddr - page_base + ph.memsz);
        mem.write_bytes(
            ph.vaddr,
            &bytes[ph.offset as usize..(ph.offset + ph.filesz) as usize],
        )
        .expect("segment range was just mapped");
        image_end = image_end.max(ph.vaddr + ph.memsz);
        if ph.offset as u64 <= elf.phoff as u64 && table_end <= ph.offset as u64 + ph.filesz as u64
        {
            phdr_vaddr = Some(ph.vaddr + (elf.phoff - ph.offset));
        }
    }
    let brk_start = align_up(image_end, PAGE_SIZE).unwrap_or(!(PAGE_SIZE - 1));
    (elf.entry, brk_start, phdr_vaddr)
}
