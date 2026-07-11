//! Model-specific registers: the RDMSR/WRMSR address space.
//!
//! Implemented MSRs are the long-mode essentials (EFER, the SYSCALL block,
//! the FS/GS/KernelGS bases), the TSC, and plain storage for a few registers
//! every OS pokes (PAT, APIC base, SYSENTER block). Unknown addresses raise
//! #GP, as on hardware.

use super::registers::{efer, reg};
use super::{Cpu, Exception, Exec};

/// MSR addresses.
mod addr {
    pub const TSC: u32 = 0x0000_0010;
    pub const APIC_BASE: u32 = 0x0000_001B;
    pub const SYSENTER_CS: u32 = 0x0000_0174;
    pub const SYSENTER_ESP: u32 = 0x0000_0175;
    pub const SYSENTER_EIP: u32 = 0x0000_0176;
    pub const PAT: u32 = 0x0000_0277;
    pub const EFER: u32 = 0xC000_0080;
    pub const STAR: u32 = 0xC000_0081;
    pub const LSTAR: u32 = 0xC000_0082;
    pub const CSTAR: u32 = 0xC000_0083;
    pub const SFMASK: u32 = 0xC000_0084;
    pub const FS_BASE: u32 = 0xC000_0100;
    pub const GS_BASE: u32 = 0xC000_0101;
    pub const KERNEL_GS_BASE: u32 = 0xC000_0102;
    pub const TSC_AUX: u32 = 0xC000_0103;
}

impl Cpu {
    /// RDMSR: read the MSR selected by ECX. #GP(0) on unknown addresses.
    pub(crate) fn rdmsr(&mut self, index: u32) -> Exec<u64> {
        let m = &self.regs.msr;
        Ok(match index {
            addr::TSC => self.cycles,
            addr::APIC_BASE => m.apic_base,
            addr::SYSENTER_CS => m.sysenter_cs,
            addr::SYSENTER_ESP => m.sysenter_esp,
            addr::SYSENTER_EIP => m.sysenter_eip,
            addr::PAT => m.pat,
            addr::EFER => m.efer,
            addr::STAR => m.star,
            addr::LSTAR => m.lstar,
            addr::CSTAR => m.cstar,
            addr::SFMASK => m.sfmask,
            addr::FS_BASE => self.regs.seg[reg::FS as usize].base,
            addr::GS_BASE => self.regs.seg[reg::GS as usize].base,
            addr::KERNEL_GS_BASE => m.kernel_gs_base,
            addr::TSC_AUX => m.tsc_aux as u64,
            _ => return Err(Exception::gp(0)),
        })
    }

    /// WRMSR: write the MSR selected by ECX. #GP(0) on unknown addresses,
    /// reserved bits, and illegal EFER transitions.
    pub(crate) fn wrmsr(&mut self, index: u32, v: u64) -> Exec<()> {
        match index {
            addr::TSC => self.cycles = v,
            addr::APIC_BASE => self.regs.msr.apic_base = v,
            addr::SYSENTER_CS => self.regs.msr.sysenter_cs = v,
            addr::SYSENTER_ESP => self.regs.msr.sysenter_esp = v,
            addr::SYSENTER_EIP => self.regs.msr.sysenter_eip = v,
            addr::PAT => self.regs.msr.pat = v,
            addr::EFER => self.write_efer(v)?,
            addr::STAR => self.regs.msr.star = v,
            addr::LSTAR => {
                if !Self::canonical(v) {
                    return Err(Exception::gp(0));
                }
                self.regs.msr.lstar = v;
            }
            addr::CSTAR => {
                if !Self::canonical(v) {
                    return Err(Exception::gp(0));
                }
                self.regs.msr.cstar = v;
            }
            addr::SFMASK => self.regs.msr.sfmask = v,
            addr::FS_BASE => {
                if !Self::canonical(v) {
                    return Err(Exception::gp(0));
                }
                self.regs.seg[reg::FS as usize].base = v;
            }
            addr::GS_BASE => {
                if !Self::canonical(v) {
                    return Err(Exception::gp(0));
                }
                self.regs.seg[reg::GS as usize].base = v;
            }
            addr::KERNEL_GS_BASE => {
                if !Self::canonical(v) {
                    return Err(Exception::gp(0));
                }
                self.regs.msr.kernel_gs_base = v;
            }
            addr::TSC_AUX => self.regs.msr.tsc_aux = v as u32,
            _ => return Err(Exception::gp(0)),
        }
        Ok(())
    }

    /// EFER write: reserved bits and the LMA/LME rules.
    fn write_efer(&mut self, v: u64) -> Exec<()> {
        if v & !efer::SUPPORTED != 0 {
            return Err(Exception::gp(0));
        }
        let old = self.regs.msr.efer;
        // LMA is read-only: software cannot flip it directly.
        if (v ^ old) & efer::LMA != 0 {
            return Err(Exception::gp(0));
        }
        // LME cannot change while paging is on (long mode is entered/left
        // only by toggling CR0.PG).
        if (v ^ old) & efer::LME != 0 && self.regs.cr0 & super::cr0::PG != 0 {
            return Err(Exception::gp(0));
        }
        self.regs.msr.efer = v;
        // NXE participates in page-permission caching.
        self.flush_tlb();
        Ok(())
    }
}
