//! 80386 register file: 32-bit general registers, segment registers with
//! their descriptor caches, EFLAGS, and the system registers.

use bitflags::bitflags;

bitflags! {
    /// The 386 EFLAGS register.
    ///
    /// Reserved bits have fixed values when the register is materialized
    /// (`PUSHF`, exception frames): bit 1 reads as 1, bits 3, 5 and 15 read
    /// as 0. [`EFlags::image16`]/[`EFlags::image32`] centralize that.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EFlags: u32 {
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
        /// Virtual-8086 mode.
        const VM = 1 << 17;
    }
}

impl EFlags {
    /// All architecturally defined 386 flag bits.
    pub const DEFINED: u32 = 0x0003_7FD5;
    /// The arithmetic/status flags (what `SAHF`/`CMPS` and friends touch).
    pub const STATUS: u32 = 0x0000_08D5;

    /// FLAGS as pushed by 16-bit `PUSHF` and real-mode exception frames:
    /// bit 1 forced to 1, bits 3/5/15 to 0.
    #[inline]
    pub fn image16(self) -> u16 {
        (self.bits() as u16) | 0x0002
    }

    /// EFLAGS as pushed by `PUSHFD`: like [`EFlags::image16`] but VM and RF
    /// read as 0 in the pushed image.
    #[inline]
    pub fn image32(self) -> u32 {
        (self.bits() | 0x0002) & !(EFlags::VM | EFlags::RF).bits()
    }

    /// Replace the bits selected by `mask` with those from `value`,
    /// discarding undefined bits. Used by `POPF`/`IRET`, whose writable set
    /// depends on operand size and privilege.
    #[inline]
    pub fn load(&mut self, value: u32, mask: u32) {
        let keep = self.bits() & !mask;
        *self = EFlags::from_bits_truncate(keep | (value & mask));
    }

    /// Current I/O privilege level (0–3).
    #[inline]
    pub fn iopl(self) -> u8 {
        ((self.bits() >> 12) & 3) as u8
    }

    /// Set `SF`, `ZF` and `PF` from an 8-bit result.
    #[inline]
    pub fn set_szp8(&mut self, v: u8) {
        self.set(EFlags::SF, v & 0x80 != 0);
        self.set(EFlags::ZF, v == 0);
        self.set(EFlags::PF, v.count_ones().is_multiple_of(2));
    }

    /// Set `SF`, `ZF` and `PF` from a 16-bit result (`PF` reflects the low byte).
    #[inline]
    pub fn set_szp16(&mut self, v: u16) {
        self.set(EFlags::SF, v & 0x8000 != 0);
        self.set(EFlags::ZF, v == 0);
        self.set(EFlags::PF, (v as u8).count_ones().is_multiple_of(2));
    }

    /// Set `SF`, `ZF` and `PF` from a 32-bit result (`PF` reflects the low byte).
    #[inline]
    pub fn set_szp32(&mut self, v: u32) {
        self.set(EFlags::SF, v & 0x8000_0000 != 0);
        self.set(EFlags::ZF, v == 0);
        self.set(EFlags::PF, (v as u8).count_ones().is_multiple_of(2));
    }
}

/// Register indices as encoded in instructions.
///
/// 32-bit order: EAX ECX EDX EBX ESP EBP ESI EDI (16-bit uses the low words).
/// 8-bit order: AL CL DL BL AH CH DH BH. Segment order: ES CS SS DS FS GS.
pub mod reg {
    pub const EAX: u8 = 0;
    pub const ECX: u8 = 1;
    pub const EDX: u8 = 2;
    pub const EBX: u8 = 3;
    pub const ESP: u8 = 4;
    pub const EBP: u8 = 5;
    pub const ESI: u8 = 6;
    pub const EDI: u8 = 7;
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
/// leaves `limit`/`attrs` untouched (which is what makes "unreal mode" work).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegReg {
    /// The visible selector.
    pub sel: u16,
    /// Cached segment base linear address.
    pub base: u32,
    /// Cached effective limit in bytes (already scaled by granularity).
    pub limit: u32,
    /// Cached access rights: descriptor access byte in bits 0–7 and the
    /// flags nibble (G/D/B/AVL) in bits 8–11.
    pub attrs: u16,
}

/// `attrs` value of a real-mode segment: present, writable data, D/B clear.
pub const REAL_MODE_ATTRS: u16 = 0x0093;

impl SegReg {
    /// A real-mode segment for selector `sel` (base `sel*16`, 64 KiB limit).
    pub fn real(sel: u16) -> Self {
        SegReg {
            sel,
            base: (sel as u32) << 4,
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

/// A descriptor-table register (GDTR or IDTR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescTable {
    pub base: u32,
    pub limit: u16,
}

/// The 386 register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Registers {
    /// General registers in instruction-encoding order
    /// (EAX ECX EDX EBX ESP EBP ESI EDI) — see [`reg`].
    pub gpr: [u32; 8],
    /// Segment registers in encoding order (ES CS SS DS FS GS) — see [`reg`].
    pub seg: [SegReg; 6],
    /// Instruction pointer.
    pub eip: u32,
    /// Processor flags.
    pub eflags: EFlags,
    /// Control registers. CR1 does not exist; CR2 holds the page-fault
    /// linear address.
    pub cr0: u32,
    pub cr2: u32,
    pub cr3: u32,
    /// Debug registers (DR4/DR5 alias DR6/DR7 on access).
    pub dr: [u32; 8],
    /// Test registers (386-specific TLB test interface; storage only).
    pub tr6: u32,
    pub tr7: u32,
    /// Global and interrupt descriptor table registers. In real mode the
    /// IDTR describes the interrupt vector table (reset: base 0, limit 3FF).
    pub gdtr: DescTable,
    pub idtr: DescTable,
    /// Local descriptor table register and task register (selector + cache).
    pub ldtr: SegReg,
    pub tr: SegReg,
}

/// CR0 bits used by the core.
pub mod cr0 {
    /// Protected-mode enable.
    pub const PE: u32 = 1 << 0;
    /// Monitor coprocessor (`WAIT` faults when set with TS).
    pub const MP: u32 = 1 << 1;
    /// FPU emulation (ESC opcodes raise `#NM` when set).
    pub const EM: u32 = 1 << 2;
    /// Task switched (ESC/`WAIT` raise `#NM` when set as configured).
    pub const TS: u32 = 1 << 3;
    /// Extension type (386: 387 protocol; reads as set).
    pub const ET: u32 = 1 << 4;
    /// Paging enable.
    pub const PG: u32 = 1 << 31;
}

impl Registers {
    /// Power-on register state: execution starts at `F000:FFF0` with the CS
    /// base at `FFFF0000` (the 386 reset quirk); flags = 2; IDTR covers the
    /// real-mode IVT.
    pub fn new() -> Self {
        let mut seg = [SegReg::real(0); 6];
        seg[reg::CS as usize] = SegReg {
            sel: 0xF000,
            base: 0xFFFF_0000,
            limit: 0xFFFF,
            attrs: REAL_MODE_ATTRS,
        };
        Registers {
            gpr: [0; 8],
            seg,
            eip: 0x0000_FFF0,
            eflags: EFlags::empty(),
            cr0: cr0::ET,
            cr2: 0,
            cr3: 0,
            dr: [0; 8],
            tr6: 0,
            tr7: 0,
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
                attrs: 0x0083,
            },
        }
    }

    /// Read a 32-bit register by its instruction encoding.
    #[inline]
    pub fn reg32(&self, i: u8) -> u32 {
        self.gpr[(i & 7) as usize]
    }

    /// Write a 32-bit register by its instruction encoding.
    #[inline]
    pub fn set_reg32(&mut self, i: u8, v: u32) {
        self.gpr[(i & 7) as usize] = v;
    }

    /// Read a 16-bit register by its instruction encoding (AX CX DX BX SP BP SI DI).
    #[inline]
    pub fn reg16(&self, i: u8) -> u16 {
        self.gpr[(i & 7) as usize] as u16
    }

    /// Write a 16-bit register, preserving the high word.
    #[inline]
    pub fn set_reg16(&mut self, i: u8, v: u16) {
        let r = &mut self.gpr[(i & 7) as usize];
        *r = (*r & 0xFFFF_0000) | v as u32;
    }

    /// Read an 8-bit register by its instruction encoding (AL CL DL BL AH CH DH BH).
    #[inline]
    pub fn reg8(&self, i: u8) -> u8 {
        let i = i & 7;
        let w = self.gpr[(i & 3) as usize];
        if i & 4 != 0 { (w >> 8) as u8 } else { w as u8 }
    }

    /// Write an 8-bit register.
    #[inline]
    pub fn set_reg8(&mut self, i: u8, v: u8) {
        let i = i & 7;
        let r = &mut self.gpr[(i & 3) as usize];
        if i & 4 != 0 {
            *r = (*r & !0xFF00) | ((v as u32) << 8);
        } else {
            *r = (*r & !0xFF) | v as u32;
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
