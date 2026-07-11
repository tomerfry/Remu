//! x86-64 register file: sixteen 64-bit general registers, segment registers
//! with descriptor caches (64-bit bases for FS/GS), RFLAGS, control
//! registers, and the model-specific registers the core implements.

use bitflags::bitflags;

bitflags! {
    /// The RFLAGS register (only the architecturally defined low 22 bits are
    /// used; bits 63:22 are reserved-zero).
    ///
    /// Reserved bits have fixed values when the register is materialized
    /// (`PUSHF`, exception frames): bit 1 reads as 1, bits 3, 5 and 15 read
    /// as 0. [`RFlags::image16`]/[`RFlags::image`] centralize that.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RFlags: u32 {
        /// Carry.
        const CF = 1 << 0;
        /// Parity (of the low byte of the result).
        const PF = 1 << 2;
        /// Auxiliary carry (out of bit 3, for BCD).
        const AF = 1 << 4;
        /// Zero.
        const ZF = 1 << 6;
        /// Sign.
        const SF = 1 << 7;
        /// Trap (single-step exception after each instruction).
        const TF = 1 << 8;
        /// Interrupt enable.
        const IF = 1 << 9;
        /// Direction (string operations decrement when set).
        const DF = 1 << 10;
        /// Overflow.
        const OF = 1 << 11;
        /// I/O privilege level (two bits).
        const IOPL = 3 << 12;
        /// Nested task.
        const NT = 1 << 14;
        /// Resume (suppresses instruction breakpoints for one instruction).
        const RF = 1 << 16;
        /// Virtual-8086 mode (legacy mode only; always 0 in long mode).
        const VM = 1 << 17;
        /// Alignment check / access control.
        const AC = 1 << 18;
        /// Virtual interrupt flag (storage only; VME is not implemented).
        const VIF = 1 << 19;
        /// Virtual interrupt pending (storage only).
        const VIP = 1 << 20;
        /// CPUID-available toggle bit.
        const ID = 1 << 21;
    }
}

impl RFlags {
    /// The arithmetic/status flags (what `SAHF`/`CMPS` and friends touch).
    pub const STATUS: u32 = 0x0000_08D5;

    /// FLAGS as pushed by 16-bit `PUSHF` and real-mode exception frames:
    /// bit 1 forced to 1, bits 3/5/15 to 0.
    #[inline]
    pub fn image16(self) -> u16 {
        (self.bits() as u16) | 0x0002
    }

    /// (R/E)FLAGS as pushed by `PUSHFD`/`PUSHFQ`: like [`RFlags::image16`]
    /// but VM and RF read as 0 in the pushed image.
    #[inline]
    pub fn image(self) -> u64 {
        ((self.bits() | 0x0002) & !(RFlags::VM | RFlags::RF).bits()) as u64
    }

    /// Replace the bits selected by `mask` with those from `value`,
    /// discarding undefined bits. Used by `POPF`/`IRET`, whose writable set
    /// depends on operand size and privilege.
    #[inline]
    pub fn load(&mut self, value: u64, mask: u32) {
        let keep = self.bits() & !mask;
        *self = RFlags::from_bits_truncate(keep | (value as u32 & mask));
    }

    /// Current I/O privilege level (0–3).
    #[inline]
    pub fn iopl(self) -> u8 {
        ((self.bits() >> 12) & 3) as u8
    }

    /// Set `SF`, `ZF` and `PF` from an 8-bit result.
    #[inline]
    pub fn set_szp8(&mut self, v: u8) {
        self.set(RFlags::SF, v & 0x80 != 0);
        self.set(RFlags::ZF, v == 0);
        self.set(RFlags::PF, v.count_ones().is_multiple_of(2));
    }

    /// Set `SF`, `ZF` and `PF` from a 16-bit result (`PF` reflects the low byte).
    #[inline]
    pub fn set_szp16(&mut self, v: u16) {
        self.set(RFlags::SF, v & 0x8000 != 0);
        self.set(RFlags::ZF, v == 0);
        self.set(RFlags::PF, (v as u8).count_ones().is_multiple_of(2));
    }

    /// Set `SF`, `ZF` and `PF` from a 32-bit result (`PF` reflects the low byte).
    #[inline]
    pub fn set_szp32(&mut self, v: u32) {
        self.set(RFlags::SF, v & 0x8000_0000 != 0);
        self.set(RFlags::ZF, v == 0);
        self.set(RFlags::PF, (v as u8).count_ones().is_multiple_of(2));
    }

    /// Set `SF`, `ZF` and `PF` from a 64-bit result (`PF` reflects the low byte).
    #[inline]
    pub fn set_szp64(&mut self, v: u64) {
        self.set(RFlags::SF, v & 0x8000_0000_0000_0000 != 0);
        self.set(RFlags::ZF, v == 0);
        self.set(RFlags::PF, (v as u8).count_ones().is_multiple_of(2));
    }
}

/// Register indices as encoded in instructions (REX extends each field by
/// one bit, selecting R8–R15).
///
/// 64-bit order: RAX RCX RDX RBX RSP RBP RSI RDI R8–R15 (narrower widths use
/// the low bytes). Without a REX prefix, 8-bit indices 4–7 select AH CH DH BH
/// instead. Segment order: ES CS SS DS FS GS.
pub mod reg {
    pub const RAX: u8 = 0;
    pub const RCX: u8 = 1;
    pub const RDX: u8 = 2;
    pub const RBX: u8 = 3;
    pub const RSP: u8 = 4;
    pub const RBP: u8 = 5;
    pub const RSI: u8 = 6;
    pub const RDI: u8 = 7;
    pub const R8: u8 = 8;
    pub const R9: u8 = 9;
    pub const R10: u8 = 10;
    pub const R11: u8 = 11;
    pub const R12: u8 = 12;
    pub const R13: u8 = 13;
    pub const R14: u8 = 14;
    pub const R15: u8 = 15;
    pub const ES: u8 = 0;
    pub const CS: u8 = 1;
    pub const SS: u8 = 2;
    pub const DS: u8 = 3;
    pub const FS: u8 = 4;
    pub const GS: u8 = 5;
}

/// A segment register: the visible selector plus the descriptor cache the
/// CPU actually uses for every access.
///
/// In real mode, loading a segment register sets `base = selector * 16` and
/// leaves `limit`/`attrs` untouched. In 64-bit mode the CS/DS/ES/SS bases are
/// treated as zero and limits are not checked; only the FS/GS bases (which
/// extend to 64 bits via MSRs) still apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegReg {
    /// The visible selector.
    pub sel: u16,
    /// Cached segment base linear address (64-bit for FS/GS in long mode).
    pub base: u64,
    /// Cached effective limit in bytes (already scaled by granularity).
    pub limit: u32,
    /// Cached access rights: descriptor access byte in bits 0–7 and the
    /// flags nibble (G/D/L/AVL) in bits 8–11.
    pub attrs: u16,
}

/// `attrs` value of a real-mode segment: present, writable data, D/B clear.
pub const REAL_MODE_ATTRS: u16 = 0x0093;

impl SegReg {
    /// A real-mode segment for selector `sel` (base `sel*16`, 64 KiB limit).
    pub fn real(sel: u16) -> Self {
        SegReg {
            sel,
            base: (sel as u64) << 4,
            limit: 0xFFFF,
            attrs: REAL_MODE_ATTRS,
        }
    }

    /// The D/B bit: default operand/address size for code segments, stack
    /// pointer width for stack segments.
    #[inline]
    pub fn db(self) -> bool {
        self.attrs & 0x0400 != 0
    }

    /// The L bit: 64-bit code segment (meaningful only when EFER.LMA is set).
    #[inline]
    pub fn l(self) -> bool {
        self.attrs & 0x0200 != 0
    }

    /// Descriptor privilege level.
    #[inline]
    pub fn dpl(self) -> u8 {
        ((self.attrs >> 5) & 3) as u8
    }

    /// Expand-down data segment (valid offsets are *above* the limit).
    #[inline]
    pub fn expand_down(self) -> bool {
        self.attrs & 0x0018 == 0x0010 && self.attrs & 0x0004 != 0
    }
}

/// A descriptor-table register (GDTR or IDTR); the base is 64-bit in long
/// mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescTable {
    pub base: u64,
    pub limit: u16,
}

/// CR0 bits used by the core.
pub mod cr0 {
    /// Protected-mode enable.
    pub const PE: u64 = 1 << 0;
    /// Monitor coprocessor (`WAIT` faults when set with TS).
    pub const MP: u64 = 1 << 1;
    /// FPU emulation (ESC opcodes raise `#NM` when set).
    pub const EM: u64 = 1 << 2;
    /// Task switched (ESC/`WAIT` raise `#NM` when set as configured).
    pub const TS: u64 = 1 << 3;
    /// Extension type (hardwired to 1 on post-486 processors).
    pub const ET: u64 = 1 << 4;
    /// Numeric error (x87 error reporting; storage only).
    pub const NE: u64 = 1 << 5;
    /// Write protect: supervisor writes honor page-level write protection.
    pub const WP: u64 = 1 << 16;
    /// Alignment mask (storage only; #AC is not implemented).
    pub const AM: u64 = 1 << 18;
    /// Not write-through / cache disable (no cache is modeled; storage only).
    pub const NW: u64 = 1 << 29;
    pub const CD: u64 = 1 << 30;
    /// Paging enable.
    pub const PG: u64 = 1 << 31;
}

/// CR4 bits used by the core (unsupported bits raise #GP when set).
pub mod cr4 {
    /// Virtual-8086 extensions (storage only; VME interrupt redirection is
    /// not implemented).
    pub const VME: u64 = 1 << 0;
    /// Protected-mode virtual interrupts (storage only).
    pub const PVI: u64 = 1 << 1;
    /// Time-stamp disable (`RDTSC` is privileged when set).
    pub const TSD: u64 = 1 << 2;
    /// Debugging extensions (storage only).
    pub const DE: u64 = 1 << 3;
    /// Page-size extensions: 4 MiB pages in legacy (non-PAE) paging.
    pub const PSE: u64 = 1 << 4;
    /// Physical-address extension: 64-bit page-table entries. Required to
    /// activate long mode.
    pub const PAE: u64 = 1 << 5;
    /// Machine-check enable (storage only).
    pub const MCE: u64 = 1 << 6;
    /// Global-page enable (accepted; global pages are flushed like any other
    /// entry by this core's whole-TLB flushes).
    pub const PGE: u64 = 1 << 7;
    /// Performance-counter enable (`RDPMC` at CPL 3).
    pub const PCE: u64 = 1 << 8;
    /// OS FXSAVE/FXRSTOR support (storage only; no SSE state exists).
    pub const OSFXSR: u64 = 1 << 9;
    /// OS unmasked-SIMD-exception support (storage only).
    pub const OSXMMEXCPT: u64 = 1 << 10;
    /// All bits this core accepts in CR4.
    pub const SUPPORTED: u64 =
        VME | PVI | TSD | DE | PSE | PAE | MCE | PGE | PCE | OSFXSR | OSXMMEXCPT;
}

/// EFER bits.
pub mod efer {
    /// System-call extensions: enables SYSCALL/SYSRET.
    pub const SCE: u64 = 1 << 0;
    /// Long-mode enable (armed; becomes active when paging turns on).
    pub const LME: u64 = 1 << 8;
    /// Long-mode active (read-only to software; set/cleared with CR0.PG).
    pub const LMA: u64 = 1 << 10;
    /// No-execute enable (honors the NX bit in 64-bit page-table entries).
    pub const NXE: u64 = 1 << 11;
    /// All bits this core accepts in EFER.
    pub const SUPPORTED: u64 = SCE | LME | LMA | NXE;
}

/// The model-specific registers the core implements beyond plain storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Msrs {
    /// Extended-feature-enable register (see [`efer`]).
    pub efer: u64,
    /// SYSCALL target selectors: CS/SS bases for SYSCALL (47:32) and SYSRET
    /// (63:48).
    pub star: u64,
    /// SYSCALL target RIP in 64-bit mode.
    pub lstar: u64,
    /// SYSCALL target RIP for compatibility mode (stored; compat SYSCALL
    /// itself is #UD as on Intel hardware).
    pub cstar: u64,
    /// SYSCALL RFLAGS clear mask.
    pub sfmask: u64,
    /// The hidden GS base swapped in by SWAPGS.
    pub kernel_gs_base: u64,
    /// IA32_TSC_AUX, read by RDTSCP.
    pub tsc_aux: u32,
    /// SYSENTER MSRs (storage only; SYSENTER/SYSEXIT are not implemented).
    pub sysenter_cs: u64,
    pub sysenter_esp: u64,
    pub sysenter_eip: u64,
    /// IA32_PAT (storage only; memory types are not modeled).
    pub pat: u64,
    /// IA32_APIC_BASE (storage only; no local APIC is modeled).
    pub apic_base: u64,
}

impl Msrs {
    fn new() -> Self {
        Msrs {
            efer: 0,
            star: 0,
            lstar: 0,
            cstar: 0,
            sfmask: 0,
            kernel_gs_base: 0,
            tsc_aux: 0,
            sysenter_cs: 0,
            sysenter_esp: 0,
            sysenter_eip: 0,
            pat: 0x0007_0406_0007_0406, // architectural PAT reset value
            apic_base: 0xFEE0_0900,     // enabled, BSP, default base
        }
    }
}

/// The x86-64 register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registers {
    /// General registers in instruction-encoding order
    /// (RAX RCX RDX RBX RSP RBP RSI RDI R8–R15) — see [`reg`].
    pub gpr: [u64; 16],
    /// Segment registers in encoding order (ES CS SS DS FS GS) — see [`reg`].
    pub seg: [SegReg; 6],
    /// Instruction pointer.
    pub rip: u64,
    /// Processor flags.
    pub rflags: RFlags,
    /// Control registers. CR1 does not exist; CR2 holds the page-fault
    /// linear address; CR8 is the task-priority register (storage only — no
    /// local APIC is modeled).
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub cr8: u64,
    /// Debug registers (DR4/DR5 alias DR6/DR7 on access; storage only).
    pub dr: [u64; 8],
    /// Model-specific registers.
    pub msr: Msrs,
    /// Global and interrupt descriptor table registers. In real mode the
    /// IDTR describes the interrupt vector table (reset: base 0, limit 3FF).
    pub gdtr: DescTable,
    pub idtr: DescTable,
    /// Local descriptor table register and task register (selector + cache).
    pub ldtr: SegReg,
    pub tr: SegReg,
}

impl Registers {
    /// Power-on register state: real mode, execution starts at `F000:FFF0`
    /// with the CS base at `FFFF0000`; flags = 2; IDTR covers the real-mode
    /// IVT; caches disabled per the architectural CR0 reset value.
    pub fn new() -> Self {
        let mut seg = [SegReg::real(0); 6];
        seg[reg::CS as usize] = SegReg {
            sel: 0xF000,
            base: 0xFFFF_0000,
            limit: 0xFFFF,
            attrs: REAL_MODE_ATTRS,
        };
        Registers {
            gpr: [0; 16],
            seg,
            rip: 0x0000_FFF0,
            rflags: RFlags::empty(),
            cr0: cr0::CD | cr0::NW | cr0::ET,
            cr2: 0,
            cr3: 0,
            cr4: 0,
            cr8: 0,
            dr: [0; 8],
            msr: Msrs::new(),
            gdtr: DescTable {
                base: 0,
                limit: 0xFFFF,
            },
            idtr: DescTable {
                base: 0,
                limit: 0x03FF,
            },
            ldtr: SegReg {
                sel: 0,
                base: 0,
                limit: 0xFFFF,
                attrs: 0x0082,
            },
            tr: SegReg {
                sel: 0,
                base: 0,
                limit: 0xFFFF,
                attrs: 0x008B,
            },
        }
    }

    /// Read a 64-bit register by its instruction encoding.
    #[inline]
    pub fn reg64(&self, i: u8) -> u64 {
        self.gpr[(i & 15) as usize]
    }

    /// Write a 64-bit register by its instruction encoding.
    #[inline]
    pub fn set_reg64(&mut self, i: u8, v: u64) {
        self.gpr[(i & 15) as usize] = v;
    }

    /// Read a 32-bit register by its instruction encoding.
    #[inline]
    pub fn reg32(&self, i: u8) -> u32 {
        self.gpr[(i & 15) as usize] as u32
    }

    /// Write a 32-bit register: bits 63:32 of the destination are zeroed, as
    /// for every 32-bit result in 64-bit mode.
    #[inline]
    pub fn set_reg32(&mut self, i: u8, v: u32) {
        self.gpr[(i & 15) as usize] = v as u64;
    }

    /// Read a 16-bit register by its instruction encoding.
    #[inline]
    pub fn reg16(&self, i: u8) -> u16 {
        self.gpr[(i & 15) as usize] as u16
    }

    /// Write a 16-bit register, preserving the high bits.
    #[inline]
    pub fn set_reg16(&mut self, i: u8, v: u16) {
        let r = &mut self.gpr[(i & 15) as usize];
        *r = (*r & !0xFFFF) | v as u64;
    }

    /// Read an 8-bit register by its instruction encoding. With a REX prefix
    /// (`rex = true`) indices 4–7 select SPL BPL SIL DIL (and 8–15 the R8B
    /// group); without, they select the legacy high bytes AH CH DH BH.
    #[inline]
    pub fn reg8(&self, i: u8, rex: bool) -> u8 {
        if !rex && i & 0xC == 4 {
            (self.gpr[(i & 3) as usize] >> 8) as u8
        } else {
            self.gpr[(i & 15) as usize] as u8
        }
    }

    /// Write an 8-bit register (see [`Registers::reg8`] for the encoding).
    #[inline]
    pub fn set_reg8(&mut self, i: u8, rex: bool, v: u8) {
        if !rex && i & 0xC == 4 {
            let r = &mut self.gpr[(i & 3) as usize];
            *r = (*r & !0xFF00) | ((v as u64) << 8);
        } else {
            let r = &mut self.gpr[(i & 15) as usize];
            *r = (*r & !0xFF) | v as u64;
        }
    }

    /// Read a segment selector by its instruction encoding (ES CS SS DS FS GS).
    #[inline]
    pub fn seg_sel(&self, i: u8) -> u16 {
        self.seg[(i % 6) as usize].sel
    }
}

impl Default for Registers {
    fn default() -> Self {
        Registers::new()
    }
}
