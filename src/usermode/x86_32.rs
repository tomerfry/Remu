//! The x86-32 [`UserArch`] adapter: runs Linux i386 user code on the
//! [`crate::x86_32`] core in flat ring-3 protected mode with paging off
//! (linear == physical, straight to the [`AddressSpace`]).
//!
//! A real GDT lives in guest memory at [`GDT_BASE`] so that segment loads the
//! guest performs itself (most importantly `MOV GS` after `set_thread_area`)
//! resolve architecturally. The initial segment registers are loaded by
//! writing the descriptor caches directly, no descriptor fetch needed.

use super::UserArch;
use super::addr_space::{AddressSpace, PAGE_SIZE};
use super::linux::abi::{self, EFAULT, EINVAL, ESRCH, NR_SET_THREAD_AREA};
use crate::x86_32::{Bus, Cpu, DescTable, EFlags, HostTrap, SegReg, cr0, reg};

/// Guest address of the GDT page (top page, outside every user range).
pub const GDT_BASE: u32 = 0xFFFF_F000;
/// Flat ring-3 code selector (GDT index 4, RPL 3).
pub const USER_CS: u16 = 0x23;
/// Flat ring-3 data/stack selector (GDT index 5, RPL 3).
pub const USER_DS: u16 = 0x2B;
/// First of the three TLS GDT slots (Linux `GDT_ENTRY_TLS_MIN`).
const TLS_ENTRY_MIN: u32 = 6;
const TLS_ENTRY_MAX: u32 = 8;
/// GDT covers entries 0..=8.
const GDT_LIMIT: u16 = 9 * 8 - 1;

/// Segment-cache attrs for the flat ring-3 segments (G=1, D/B=1, P=1, DPL=3;
/// the values the descriptors below decode to).
const CODE_ATTRS: u16 = 0x0CFA;
const DATA_ATTRS: u16 = 0x0CF2;

/// Descriptor bytes: base 0, limit 0xFFFFF pages (4 GiB), ring 3, 32-bit.
const CODE_DESC: [u8; 8] = [0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFA, 0xCF, 0x00];
const DATA_DESC: [u8; 8] = [0xFF, 0xFF, 0x00, 0x00, 0x00, 0xF2, 0xCF, 0x00];

/// The Linux syscall vector.
const SYSCALL_VECTOR: u8 = 0x80;

impl Bus for AddressSpace {
    #[inline]
    fn read(&mut self, addr: u32) -> u8 {
        self.read8(addr)
    }

    #[inline]
    fn write(&mut self, addr: u32, value: u8) {
        self.write8(addr, value)
    }

    #[inline]
    fn read16(&mut self, addr: u32) -> u16 {
        AddressSpace::read16(self, addr)
    }

    #[inline]
    fn read32(&mut self, addr: u32) -> u32 {
        AddressSpace::read32(self, addr)
    }

    #[inline]
    fn write16(&mut self, addr: u32, value: u16) {
        AddressSpace::write16(self, addr, value)
    }

    #[inline]
    fn write32(&mut self, addr: u32, value: u32) {
        AddressSpace::write32(self, addr, value)
    }
}

impl UserArch for Cpu {
    const ELF_MACHINE: u16 = super::elf::EM_386;

    fn new() -> Self {
        Cpu::new()
    }

    fn setup(&mut self, mem: &mut AddressSpace, entry: u32, sp: u32) {
        // A real GDT in guest memory: null, unused, unused, unused, user
        // code, user data, three empty TLS slots.
        mem.map(GDT_BASE, PAGE_SIZE);
        mem.write_bytes(GDT_BASE + 4 * 8, &CODE_DESC).unwrap();
        mem.write_bytes(GDT_BASE + 5 * 8, &DATA_DESC).unwrap();
        self.regs.gdtr = DescTable {
            base: GDT_BASE,
            limit: GDT_LIMIT,
        };
        // An empty IDT: any delivered exception #GPs its way to a clean
        // triple fault (`shutdown`) without touching guest memory.
        self.regs.idtr = DescTable { base: 0, limit: 0 };

        // Flat ring-3 protected mode, entered by loading the caches directly.
        self.regs.cr0 |= cr0::PE;
        let code = SegReg {
            sel: USER_CS,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: CODE_ATTRS,
        };
        let data = SegReg {
            sel: USER_DS,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: DATA_ATTRS,
        };
        self.regs.seg[reg::CS as usize] = code;
        for s in [reg::SS, reg::DS, reg::ES, reg::FS] {
            self.regs.seg[s as usize] = data;
        }
        // GS stays null until the guest loads it (glibc/musl TLS).
        self.regs.seg[reg::GS as usize] = SegReg {
            sel: 0,
            base: 0,
            limit: 0,
            attrs: 0,
        };

        self.regs.gpr = [0; 8];
        self.regs.gpr[reg::ESP as usize] = sp;
        self.regs.eip = entry;
        self.regs.eflags.insert(EFlags::IF);
        self.syscall_int = Some(SYSCALL_VECTOR);
    }

    #[inline]
    fn step(&mut self, mem: &mut AddressSpace) -> u32 {
        Cpu::step(self, mem)
    }

    #[inline]
    fn take_syscall(&mut self) -> bool {
        if self.host_trap == Some(HostTrap::Syscall) {
            self.host_trap = None;
            return true;
        }
        false
    }

    #[inline]
    fn syscall_number(&self) -> u32 {
        self.regs.gpr[reg::EAX as usize]
    }

    #[inline]
    fn syscall_args(&self) -> [u32; 6] {
        let g = &self.regs.gpr;
        [
            g[reg::EBX as usize],
            g[reg::ECX as usize],
            g[reg::EDX as usize],
            g[reg::ESI as usize],
            g[reg::EDI as usize],
            g[reg::EBP as usize],
        ]
    }

    #[inline]
    fn set_syscall_ret(&mut self, value: u32) {
        self.regs.gpr[reg::EAX as usize] = value;
    }

    fn arch_syscall(&mut self, mem: &mut AddressSpace, nr: u32, args: &[u32; 6]) -> Option<u32> {
        match nr {
            NR_SET_THREAD_AREA => Some(set_thread_area(mem, args[0])),
            _ => None,
        }
    }

    fn pc(&self) -> u32 {
        self.regs.eip
    }

    fn dead(&self) -> bool {
        self.shutdown
    }

    fn dump(&self) -> String {
        let g = &self.regs.gpr;
        format!(
            "EAX={:08x} EBX={:08x} ECX={:08x} EDX={:08x}\n\
             ESI={:08x} EDI={:08x} EBP={:08x} ESP={:08x}\n\
             EIP={:08x} EFLAGS={:08x} CS={:04x} SS={:04x} DS={:04x} ES={:04x} FS={:04x} GS={:04x}",
            g[reg::EAX as usize],
            g[reg::EBX as usize],
            g[reg::ECX as usize],
            g[reg::EDX as usize],
            g[reg::ESI as usize],
            g[reg::EDI as usize],
            g[reg::EBP as usize],
            g[reg::ESP as usize],
            self.regs.eip,
            self.regs.eflags.image32(),
            self.regs.seg[reg::CS as usize].sel,
            self.regs.seg[reg::SS as usize].sel,
            self.regs.seg[reg::DS as usize].sel,
            self.regs.seg[reg::ES as usize].sel,
            self.regs.seg[reg::FS as usize].sel,
            self.regs.seg[reg::GS as usize].sel,
        )
    }
}

/// `set_thread_area(2)`: install a TLS descriptor into one of the three TLS
/// GDT slots in guest memory and report the chosen entry back through the
/// guest's `user_desc`. The guest then loads `GS` itself, which resolves
/// through this GDT architecturally.
fn set_thread_area(mem: &mut AddressSpace, uinfo: u32) -> u32 {
    let mut raw = [0u8; 16];
    if mem.read_bytes(uinfo, &mut raw).is_err() {
        return abi::err(EFAULT);
    }
    let entry_number = u32::from_le_bytes(raw[0..4].try_into().unwrap());
    let base = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    let limit = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let flags = u32::from_le_bytes(raw[12..16].try_into().unwrap());
    let seg_32bit = flags & 1 != 0;
    let read_exec_only = flags & 0x8 != 0;
    let limit_in_pages = flags & 0x10 != 0;
    let seg_not_present = flags & 0x20 != 0;
    let useable = flags & 0x40 != 0;

    let entry = if entry_number == u32::MAX {
        // Allocate the first free slot (its descriptor is still all zeros).
        let mut free = None;
        for e in TLS_ENTRY_MIN..=TLS_ENTRY_MAX {
            let mut d = [0u8; 8];
            if mem.read_bytes(GDT_BASE + e * 8, &mut d).is_err() {
                return abi::err(EFAULT); // guest unmapped its own GDT
            }
            if d == [0; 8] {
                free = Some(e);
                break;
            }
        }
        match free {
            Some(e) => e,
            None => return abi::err(ESRCH),
        }
    } else if (TLS_ENTRY_MIN..=TLS_ENTRY_MAX).contains(&entry_number) {
        entry_number
    } else {
        return abi::err(EINVAL);
    };

    if limit > 0xF_FFFF {
        return abi::err(EINVAL);
    }
    // Data segment, DPL 3, accessed; writable unless read_exec_only.
    let access: u8 = 0xF1 | if read_exec_only { 0 } else { 0x02 };
    let access = if seg_not_present {
        access & !0x80
    } else {
        access
    };
    let flags_nibble: u8 = (limit_in_pages as u8) << 3 | (seg_32bit as u8) << 2 | (useable as u8);
    let desc = [
        (limit & 0xFF) as u8,
        (limit >> 8 & 0xFF) as u8,
        (base & 0xFF) as u8,
        (base >> 8 & 0xFF) as u8,
        (base >> 16 & 0xFF) as u8,
        access,
        (flags_nibble << 4) | (limit >> 16 & 0xF) as u8,
        (base >> 24) as u8,
    ];
    if mem.write_bytes(GDT_BASE + entry * 8, &desc).is_err() {
        return abi::err(EFAULT); // guest unmapped its own GDT
    }
    mem.write_u32(uinfo, entry);
    0
}
