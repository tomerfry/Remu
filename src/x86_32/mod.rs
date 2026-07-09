//! The Intel 80386 CPU core: 32-bit registers and addressing, real mode and
//! protected mode with paging, the full documented 386 integer instruction
//! set, instruction-atomic timing.
//!
//! Like the other cores, it is deliberately self-contained (own [`Bus`]
//! trait, own register file) so promoting each core to its own crate later
//! stays mechanical.
//!
//! ```
//! use remu::x86_32::{Cpu, Bus, LinearMemory};
//!
//! let mut mem = LinearMemory::new();
//! mem.load(0x0_1100, &[0x66, 0xB8, 0x78, 0x56, 0x34, 0x12]); // MOV EAX, 0x12345678
//! let mut cpu = Cpu::new();
//! cpu.set_cs_ip(0x0000, 0x1100);
//! cpu.step(&mut mem);
//! assert_eq!(cpu.regs.gpr[0], 0x1234_5678);
//! ```

mod alu;
mod execute;
mod execute_0f;
mod modrm;
mod paging;
mod protected;
pub mod registers;

pub use registers::{DescTable, EFlags, Registers, SegReg, cr0, reg};

/// The 386 memory and I/O bus.
///
/// Memory addresses are 32-bit physical. The wide accessors have
/// byte-composed defaults so simple devices only implement `read`/`write`;
/// RAM-backed buses should override them for speed. The CPU only issues a
/// wide access when it does not cross a page boundary, so overrides may
/// assume contiguity.
pub trait Bus {
    /// Read one byte from physical address `addr`.
    fn read(&mut self, addr: u32) -> u8;

    /// Write `value` to physical address `addr`.
    fn write(&mut self, addr: u32, value: u8);

    /// Read a little-endian word from `addr`.
    #[inline]
    fn read16(&mut self, addr: u32) -> u16 {
        self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
    }

    /// Read a little-endian double-word from `addr`.
    #[inline]
    fn read32(&mut self, addr: u32) -> u32 {
        self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
    }

    /// Write a little-endian word to `addr`.
    #[inline]
    fn write16(&mut self, addr: u32, value: u16) {
        self.write(addr, value as u8);
        self.write(addr.wrapping_add(1), (value >> 8) as u8);
    }

    /// Write a little-endian double-word to `addr`.
    #[inline]
    fn write32(&mut self, addr: u32, value: u32) {
        self.write16(addr, value as u16);
        self.write16(addr.wrapping_add(2), (value >> 16) as u16);
    }

    /// Read one byte from I/O port `port`.
    fn io_read(&mut self, port: u16) -> u8 {
        let _ = port;
        0xFF
    }

    /// Write `value` to I/O port `port`.
    fn io_write(&mut self, port: u16, value: u8) {
        let _ = (port, value);
    }
}

/// A flat 16 MiB RAM (the address space of the SingleStepTests 386EX rig) —
/// the simplest possible 386 machine, for tests and raw programs. Addresses
/// wrap at 16 MiB.
pub struct LinearMemory {
    /// 24-bit physical address space.
    pub ram: Box<[u8]>,
}

/// Address mask for [`LinearMemory`] (16 MiB).
const LINEAR_MASK: usize = 0xFF_FFFF;

impl LinearMemory {
    /// Create a zero-initialized 16 MiB memory.
    pub fn new() -> Self {
        LinearMemory {
            ram: vec![0u8; LINEAR_MASK + 1].into_boxed_slice(),
        }
    }

    /// Load `data` into memory starting at physical address `addr`.
    pub fn load(&mut self, addr: u32, data: &[u8]) {
        for (i, &byte) in data.iter().enumerate() {
            self.ram[(addr as usize + i) & LINEAR_MASK] = byte;
        }
    }
}

impl Default for LinearMemory {
    fn default() -> Self {
        LinearMemory::new()
    }
}

impl Bus for LinearMemory {
    fn read(&mut self, addr: u32) -> u8 {
        self.ram[addr as usize & LINEAR_MASK]
    }

    fn write(&mut self, addr: u32, value: u8) {
        self.ram[addr as usize & LINEAR_MASK] = value;
    }

    fn read16(&mut self, addr: u32) -> u16 {
        let a = addr as usize & LINEAR_MASK;
        if a < LINEAR_MASK {
            u16::from_le_bytes(self.ram[a..a + 2].try_into().unwrap())
        } else {
            self.read(addr) as u16 | (self.read(addr.wrapping_add(1)) as u16) << 8
        }
    }

    fn read32(&mut self, addr: u32) -> u32 {
        let a = addr as usize & LINEAR_MASK;
        if a + 3 <= LINEAR_MASK {
            u32::from_le_bytes(self.ram[a..a + 4].try_into().unwrap())
        } else {
            self.read16(addr) as u32 | (self.read16(addr.wrapping_add(2)) as u32) << 16
        }
    }

    fn write16(&mut self, addr: u32, value: u16) {
        let a = addr as usize & LINEAR_MASK;
        if a < LINEAR_MASK {
            self.ram[a..a + 2].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write(addr, value as u8);
            self.write(addr.wrapping_add(1), (value >> 8) as u8);
        }
    }

    fn write32(&mut self, addr: u32, value: u32) {
        let a = addr as usize & LINEAR_MASK;
        if a + 3 <= LINEAR_MASK {
            self.ram[a..a + 4].copy_from_slice(&value.to_le_bytes());
        } else {
            self.write16(addr, value as u16);
            self.write16(addr.wrapping_add(2), (value >> 16) as u16);
        }
    }
}

/// A CPU exception (or software interrupt turned fault) in flight.
///
/// `error` carries the error code pushed by protected-mode gates for the
/// faults that define one; it is ignored in real mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exception {
    pub vector: u8,
    pub error: Option<u16>,
}

impl Exception {
    /// #DE — divide error.
    pub(crate) fn de() -> Self {
        Exception {
            vector: 0,
            error: None,
        }
    }
    /// #BR — BOUND range exceeded.
    pub(crate) fn br() -> Self {
        Exception {
            vector: 5,
            error: None,
        }
    }
    /// #UD — invalid opcode.
    pub(crate) fn ud() -> Self {
        Exception {
            vector: 6,
            error: None,
        }
    }
    /// #NM — no math coprocessor.
    pub(crate) fn nm() -> Self {
        Exception {
            vector: 7,
            error: None,
        }
    }
    /// #TS — invalid TSS.
    pub(crate) fn ts(sel: u16) -> Self {
        Exception {
            vector: 10,
            error: Some(sel),
        }
    }
    /// #NP — segment not present.
    pub(crate) fn np(sel: u16) -> Self {
        Exception {
            vector: 11,
            error: Some(sel),
        }
    }
    /// #SS — stack fault.
    pub(crate) fn ss(sel: u16) -> Self {
        Exception {
            vector: 12,
            error: Some(sel),
        }
    }
    /// #GP — general protection fault.
    pub(crate) fn gp(sel: u16) -> Self {
        Exception {
            vector: 13,
            error: Some(sel),
        }
    }
    /// #PF — page fault (CR2 is set by the paging unit).
    pub(crate) fn pf(code: u16) -> Self {
        Exception {
            vector: 14,
            error: Some(code),
        }
    }
}

/// Result type for anything that can raise a CPU exception.
pub(crate) type Exec<T> = Result<T, Exception>;

/// Cycles consumed by servicing a hardware interrupt (real-mode figure).
const INTERRUPT_CYCLES: u32 = 37;

/// One-byte opcodes that may legally carry a LOCK prefix (given a memory
/// destination, validated per-handler): the ALU r/m,r forms, the immediate
/// ALU groups, XCHG, and the NOT/NEG/INC/DEC groups. `0F` two-byte opcodes
/// are screened in `dispatch_0f`.
const LOCK_CANDIDATE: [bool; 256] = {
    let mut t = [false; 256];
    let mut op = 0x00;
    while op <= 0x31 {
        // 00/01, 08/09, ... 30/31 (rm,r forms of ADD OR ADC SBB AND SUB XOR).
        t[op] = true;
        t[op + 1] = true;
        op += 8;
    }
    t[0x80] = true;
    t[0x81] = true;
    t[0x82] = true;
    t[0x83] = true;
    t[0x86] = true;
    t[0x87] = true;
    t[0xF6] = true;
    t[0xF7] = true;
    t[0xFE] = true;
    t[0xFF] = true;
    t
};

/// An 80386 processor.
///
/// As with the other cores, the CPU does not own its bus — call [`Cpu::step`]
/// with a `&mut B: Bus`.
#[derive(Debug, Clone)]
pub struct Cpu {
    /// The register file.
    pub regs: Registers,
    /// Total cycles elapsed since construction/reset. Timing is
    /// instruction-atomic using documented 386 base timings; the prefetch
    /// queue and dynamic bus sizing are not modeled, so counts are
    /// approximate.
    pub cycles: u64,
    /// Set while the processor is stopped by `HLT` (an interrupt resumes it).
    pub halted: bool,
    /// Set on triple fault; only RESET leaves this state.
    pub shutdown: bool,

    /// Latched pending non-maskable interrupt (edge-triggered, vector 2).
    nmi_pending: bool,
    /// Pending maskable interrupt request with its vector (as supplied by a
    /// PIC during the INTA cycle).
    intr: Option<u8>,
    /// Interrupts (and traps) are inhibited for one instruction after
    /// `MOV SS`/`POP SS`/`STI`.
    inhibit_interrupts: bool,

    // --- Per-instruction decode state ---------------------------------------
    /// Segment-override prefix (segment register index).
    seg_override: Option<u8>,
    /// Repeat prefix: `true` for `REP`/`REPE` (F3), `false` for `REPNE` (F2).
    rep: Option<bool>,
    /// A LOCK prefix was decoded.
    lock: bool,
    /// The current instruction legally used its LOCK prefix (set by handlers
    /// of lockable instructions with a memory destination).
    lock_ok: bool,
    /// Effective operand size is 32-bit (CS.D xor `66` prefix).
    osize32: bool,
    /// Effective address size is 32-bit (CS.D xor `67` prefix).
    asize32: bool,
    /// EIP at the start of the current instruction (fault reporting).
    start_eip: u32,
    /// A fault should keep the current register state and only rewind EIP —
    /// set by string instructions, whose per-iteration progress is
    /// architecturally visible when an iteration faults.
    commit_on_fault: bool,
    /// Bytes consumed by the current instruction (15-byte limit).
    ilen: u8,
    /// TLB for paged address translation (see `paging.rs`).
    tlb: paging::Tlb,
}

impl Cpu {
    /// Create a CPU in its power-on state: execution begins at `F000:FFF0`
    /// (CS base `FFFF0000`).
    pub fn new() -> Self {
        Cpu {
            regs: Registers::new(),
            cycles: 0,
            halted: false,
            shutdown: false,
            nmi_pending: false,
            intr: None,
            inhibit_interrupts: false,
            seg_override: None,
            rep: None,
            lock: false,
            lock_ok: false,
            osize32: false,
            asize32: false,
            start_eip: 0,
            commit_on_fault: false,
            ilen: 0,
            tlb: paging::Tlb::new(),
        }
    }

    /// Perform a RESET: registers to power-on state, pending interrupts and
    /// shutdown cleared.
    pub fn reset(&mut self) {
        self.regs = Registers::new();
        self.halted = false;
        self.shutdown = false;
        self.nmi_pending = false;
        self.intr = None;
        self.inhibit_interrupts = false;
        self.tlb.flush();
    }

    /// Convenience for tests and loaders: set `CS:EIP` with real-mode
    /// semantics (CS base = `sel * 16`).
    pub fn set_cs_ip(&mut self, sel: u16, eip: u32) {
        self.regs.seg[reg::CS as usize] = SegReg::real(sel);
        self.regs.eip = eip;
    }

    /// True once `CR0.PE` is set and the CPU is not in Virtual-8086 mode.
    #[inline]
    pub fn protected_mode(&self) -> bool {
        self.regs.cr0 & cr0::PE != 0 && !self.regs.eflags.contains(EFlags::VM)
    }

    /// Current privilege level (CPL). Real mode and V86 report their fixed
    /// levels through the CS cache attributes maintained by segment loads.
    #[inline]
    pub fn cpl(&self) -> u8 {
        if self.regs.eflags.contains(EFlags::VM) {
            3
        } else if self.protected_mode() {
            self.regs.seg[reg::CS as usize].dpl()
        } else {
            0
        }
    }

    /// Execute one instruction (or service a pending interrupt). Returns the
    /// number of cycles consumed, and adds them to [`Cpu::cycles`].
    pub fn step<B: Bus>(&mut self, bus: &mut B) -> u32 {
        if self.shutdown {
            self.cycles += 1;
            return 1;
        }

        // Interrupts are recognized at instruction boundaries, except for the
        // one-instruction shadow after MOV SS / POP SS / STI.
        let inhibited = self.inhibit_interrupts;
        self.inhibit_interrupts = false;
        if !inhibited {
            if self.nmi_pending {
                self.nmi_pending = false;
                self.halted = false;
                self.deliver(
                    bus,
                    Exception {
                        vector: 2,
                        error: None,
                    },
                    false,
                );
                self.cycles += INTERRUPT_CYCLES as u64;
                return INTERRUPT_CYCLES;
            }
            if let Some(vector) = self.intr
                && self.regs.eflags.contains(EFlags::IF)
            {
                self.intr = None;
                self.halted = false;
                self.deliver(
                    bus,
                    Exception {
                        vector,
                        error: None,
                    },
                    false,
                );
                self.cycles += INTERRUPT_CYCLES as u64;
                return INTERRUPT_CYCLES;
            }
        }

        if self.halted {
            self.cycles += 1;
            return 1;
        }

        // Trap flag: a single-step exception fires after this instruction if
        // TF was set when it started.
        let trap = self.regs.eflags.contains(EFlags::TF);

        // Faults restore register state so the instruction can restart.
        // EFLAGS keeps whatever the faulting computation left behind (the
        // divide-fault frame pushes those flags, and LOCK-#UD is decided at
        // decode time before anything runs), and CR2 keeps the page-fault
        // address. The snapshot is a plain copy.
        let saved = self.regs;
        let mut cycles = match self.exec_one(bus) {
            Ok(c) => c,
            Err(e) => {
                if self.commit_on_fault {
                    // String/PUSHA-style progress stays; only EIP rewinds.
                    self.regs.eip = self.start_eip;
                } else {
                    let (eflags, cr2) = (self.regs.eflags, self.regs.cr2);
                    self.regs = saved;
                    self.regs.eflags = eflags;
                    self.regs.cr2 = cr2;
                }
                self.deliver(bus, e, false);
                INTERRUPT_CYCLES
            }
        };

        if trap && self.regs.eflags.contains(EFlags::TF) && !self.inhibit_interrupts {
            self.deliver(
                bus,
                Exception {
                    vector: 1,
                    error: None,
                },
                false,
            );
            cycles += INTERRUPT_CYCLES;
        }

        self.cycles += cycles as u64;
        cycles
    }

    /// Decode prefixes and execute the instruction at `CS:EIP`.
    fn exec_one<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        self.start_eip = self.regs.eip;
        self.ilen = 0;
        self.seg_override = None;
        self.rep = None;
        self.lock = false;
        self.lock_ok = false;
        self.commit_on_fault = false;
        let db = self.regs.seg[reg::CS as usize].db();
        self.osize32 = db;
        self.asize32 = db;

        let mut cycles = 0u32;
        let opcode = loop {
            let b = self.fetch8(bus)?;
            match b {
                0x26 => self.seg_override = Some(reg::ES),
                0x2E => self.seg_override = Some(reg::CS),
                0x36 => self.seg_override = Some(reg::SS),
                0x3E => self.seg_override = Some(reg::DS),
                0x64 => self.seg_override = Some(reg::FS),
                0x65 => self.seg_override = Some(reg::GS),
                0x66 => self.osize32 = !db,
                0x67 => self.asize32 = !db,
                0xF0 => self.lock = true,
                0xF2 => self.rep = Some(false),
                0xF3 => self.rep = Some(true),
                _ => break b,
            }
            cycles += 1;
        };

        // LOCK legality is a decode-time check: an opcode that can never
        // lock raises #UD before any side effect. Candidate opcodes verify
        // their operand form (memory destination, group sub-opcode) right
        // after ModRM decode via `lock_check`.
        if self.lock && opcode != 0x0F && !LOCK_CANDIDATE[opcode as usize] {
            return Err(Exception::ud());
        }

        cycles += self.dispatch(bus, opcode)?;

        // Backstop: a candidate whose handler never validated the prefix.
        if self.lock && !self.lock_ok {
            return Err(Exception::ud());
        }
        Ok(cycles)
    }

    /// Validate the LOCK prefix for a candidate instruction once its operand
    /// is known: legal only with a memory destination (`ok`).
    #[inline]
    pub(crate) fn lock_check(&mut self, ok: bool) -> Exec<()> {
        if self.lock {
            if !ok {
                return Err(Exception::ud());
            }
            self.lock_ok = true;
        }
        Ok(())
    }

    // --- Interrupt line control ---------------------------------------------

    /// Latch a pending NMI (vector 2, not maskable by `IF`).
    pub fn trigger_nmi(&mut self) {
        self.nmi_pending = true;
    }

    /// Assert the INTR line with `vector` (as a PIC would supply during the
    /// interrupt-acknowledge cycle). Serviced at the next instruction
    /// boundary with `IF` set.
    pub fn assert_intr(&mut self, vector: u8) {
        self.intr = Some(vector);
    }

    /// Deassert the INTR line.
    pub fn clear_intr(&mut self) {
        self.intr = None;
    }

    // --- Segment-relative memory access -------------------------------------
    // Every instruction memory access funnels through these: segment limit
    // check, base + offset to a linear address, paging, then the bus.

    /// Check that `off .. off+size-1` lies within `seg`'s limit and that the
    /// segment permits the access, raising #GP(0) — or #SS(0) for the stack
    /// segment — otherwise. `write` selects the write-permission rule; reads
    /// only reject execute-only code segments (instruction fetch bypasses
    /// this via [`Cpu::fetch8`]).
    #[inline]
    fn ea_check(&self, seg: u8, off: u32, size: u32, write: bool) -> Exec<()> {
        let s = &self.regs.seg[seg as usize];
        let fault = || {
            if seg == reg::SS {
                Err(Exception::ss(0))
            } else {
                Err(Exception::gp(0))
            }
        };

        // Present/usable (null-loaded segments have attrs == 0).
        if s.attrs & 0x80 == 0 {
            return fault();
        }
        // Type: writes need a writable data segment; reads reject
        // execute-only code. (Real/V86 attrs are writable data.)
        if write {
            if s.attrs & 0x1A != 0x12 {
                return fault();
            }
        } else if s.attrs & 0x0A == 0x08 {
            return fault();
        }

        let last = off as u64 + (size - 1) as u64;
        let ok = if s.expand_down() {
            // Valid offsets are (limit, top]; top is FFFF or FFFFFFFF by D/B.
            let top = if s.db() { u32::MAX as u64 } else { 0xFFFF };
            off as u64 > s.limit as u64 && last <= top
        } else {
            last <= s.limit as u64
        };
        if ok { Ok(()) } else { fault() }
    }

    /// Instruction-fetch check: limit only (execute-only segments fetch fine).
    #[inline]
    fn fetch_check(&self, off: u32) -> Exec<()> {
        let cs = &self.regs.seg[reg::CS as usize];
        if off as u64 <= cs.limit as u64 {
            Ok(())
        } else {
            Err(Exception::gp(0))
        }
    }

    /// Offset wrap mask for data accesses: multi-byte operands wrap at 64 KiB
    /// under a 16-bit address size (8086-compatible), at 4 GiB otherwise.
    #[inline]
    pub(crate) fn data_wrap(&self) -> u32 {
        if self.asize32 { u32::MAX } else { 0xFFFF }
    }

    /// Offset wrap mask for stack accesses (SS.B selects SP vs ESP width).
    #[inline]
    pub(crate) fn stack_wrap(&self) -> u32 {
        if self.regs.seg[reg::SS as usize].db() {
            u32::MAX
        } else {
            0xFFFF
        }
    }

    /// Read one byte at `seg:off`.
    #[inline]
    pub(crate) fn read8<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32) -> Exec<u8> {
        self.ea_check(seg, off, 1, false)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_read8(bus, lin)
    }

    /// Read a little-endian word at `seg:off`, component offsets wrapping at
    /// the data-address width.
    #[inline]
    pub(crate) fn read16<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32) -> Exec<u16> {
        let wrap = self.data_wrap();
        self.read16w(bus, seg, off, wrap)
    }

    /// Read a little-endian double-word at `seg:off` (data-width wrap).
    #[inline]
    pub(crate) fn read32<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32) -> Exec<u32> {
        let wrap = self.data_wrap();
        self.read32w(bus, seg, off, wrap)
    }

    /// Word read with an explicit component-offset wrap mask. A wrapped
    /// access straddling the limit faults (the 386 does not split operands
    /// across a 64 KiB wrap; only *separate* component accesses wrap).
    #[inline]
    pub(crate) fn read16w<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
        wrap: u32,
    ) -> Exec<u16> {
        let _ = wrap;
        self.ea_check(seg, off, 2, false)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_read16(bus, lin)
    }

    /// Double-word read (see [`Cpu::read16w`]).
    #[inline]
    pub(crate) fn read32w<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
        wrap: u32,
    ) -> Exec<u32> {
        let _ = wrap;
        self.ea_check(seg, off, 4, false)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_read32(bus, lin)
    }

    /// Write one byte at `seg:off`.
    #[inline]
    pub(crate) fn write8<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32, v: u8) -> Exec<()> {
        self.ea_check(seg, off, 1, true)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_write8(bus, lin, v)
    }

    /// Write a little-endian word at `seg:off` (data-width wrap).
    #[inline]
    pub(crate) fn write16<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32, v: u16) -> Exec<()> {
        let wrap = self.data_wrap();
        self.write16w(bus, seg, off, v, wrap)
    }

    /// Write a little-endian double-word at `seg:off` (data-width wrap).
    #[inline]
    pub(crate) fn write32<B: Bus>(&mut self, bus: &mut B, seg: u8, off: u32, v: u32) -> Exec<()> {
        let wrap = self.data_wrap();
        self.write32w(bus, seg, off, v, wrap)
    }

    /// Word write with an explicit component-offset wrap mask (straddles
    /// fault; see [`Cpu::read16w`]).
    #[inline]
    pub(crate) fn write16w<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
        v: u16,
        wrap: u32,
    ) -> Exec<()> {
        let _ = wrap;
        self.ea_check(seg, off, 2, true)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_write16(bus, lin, v)
    }

    /// Double-word write (see [`Cpu::read16w`]).
    #[inline]
    pub(crate) fn write32w<B: Bus>(
        &mut self,
        bus: &mut B,
        seg: u8,
        off: u32,
        v: u32,
        wrap: u32,
    ) -> Exec<()> {
        let _ = wrap;
        self.ea_check(seg, off, 4, true)?;
        let lin = self.regs.seg[seg as usize].base.wrapping_add(off);
        self.lin_write32(bus, lin, v)
    }

    /// The effective segment for a data access whose default segment is
    /// `default`, honoring any override prefix.
    #[inline]
    pub(crate) fn seg_or(&self, default: u8) -> u8 {
        self.seg_override.unwrap_or(default)
    }

    // --- Instruction fetch ----------------------------------------------------

    /// Fetch the byte at `CS:EIP` and advance `EIP`, enforcing the 15-byte
    /// instruction length limit.
    #[inline]
    pub(crate) fn fetch8<B: Bus>(&mut self, bus: &mut B) -> Exec<u8> {
        if self.ilen >= 15 {
            return Err(Exception::ud());
        }
        self.ilen += 1;
        self.fetch_check(self.regs.eip)?;
        let lin = self.regs.seg[reg::CS as usize]
            .base
            .wrapping_add(self.regs.eip);
        let b = self.lin_read8(bus, lin)?;
        self.regs.eip = self.regs.eip.wrapping_add(1);
        Ok(b)
    }

    /// Fetch a little-endian word at `CS:EIP`.
    #[inline]
    pub(crate) fn fetch16<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        let lo = self.fetch8(bus)? as u16;
        let hi = self.fetch8(bus)? as u16;
        Ok(lo | hi << 8)
    }

    /// Fetch a little-endian double-word at `CS:EIP`.
    #[inline]
    pub(crate) fn fetch32<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        let lo = self.fetch16(bus)? as u32;
        let hi = self.fetch16(bus)? as u32;
        Ok(lo | hi << 16)
    }

    /// Fetch an immediate of the current operand size, sign-extending a
    /// 16-bit immediate to 32 bits.
    #[inline]
    pub(crate) fn fetch_imm<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        if self.osize32 {
            self.fetch32(bus)
        } else {
            Ok(self.fetch16(bus)? as u32)
        }
    }

    // --- Stack ----------------------------------------------------------------
    // SS.B selects whether SP or ESP is the stack pointer; the operand size
    // of the push/pop selects the datum width independently.

    /// Current stack pointer honoring SS.B.
    #[inline]
    pub(crate) fn stack_ptr(&self) -> u32 {
        let sp = self.regs.gpr[reg::ESP as usize];
        if self.regs.seg[reg::SS as usize].db() {
            sp
        } else {
            sp & 0xFFFF
        }
    }

    /// Adjust the stack pointer by `delta` (SP or ESP per SS.B).
    #[inline]
    pub(crate) fn adjust_sp(&mut self, delta: i32) {
        let r = &mut self.regs.gpr[reg::ESP as usize];
        if self.regs.seg[reg::SS as usize].db() {
            *r = r.wrapping_add(delta as u32);
        } else {
            *r = (*r & 0xFFFF_0000) | (*r as u16).wrapping_add(delta as u16) as u32;
        }
    }

    /// Push a word onto the stack.
    #[inline]
    pub(crate) fn push16<B: Bus>(&mut self, bus: &mut B, v: u16) -> Exec<()> {
        let wrap = self.stack_wrap();
        let sp = self.stack_ptr().wrapping_sub(2) & wrap;
        self.write16w(bus, reg::SS, sp, v, wrap)?;
        self.adjust_sp(-2);
        Ok(())
    }

    /// Push a double-word onto the stack.
    #[inline]
    pub(crate) fn push32<B: Bus>(&mut self, bus: &mut B, v: u32) -> Exec<()> {
        let wrap = self.stack_wrap();
        let sp = self.stack_ptr().wrapping_sub(4) & wrap;
        self.write32w(bus, reg::SS, sp, v, wrap)?;
        self.adjust_sp(-4);
        Ok(())
    }

    /// Push an operand-sized value.
    #[inline]
    pub(crate) fn push<B: Bus>(&mut self, bus: &mut B, v: u32) -> Exec<()> {
        if self.osize32 {
            self.push32(bus, v)
        } else {
            self.push16(bus, v as u16)
        }
    }

    /// Pop a word off the stack.
    #[inline]
    pub(crate) fn pop16<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        let wrap = self.stack_wrap();
        let v = self.read16w(bus, reg::SS, self.stack_ptr(), wrap)?;
        self.adjust_sp(2);
        Ok(v)
    }

    /// Pop a double-word off the stack.
    #[inline]
    pub(crate) fn pop32<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        let wrap = self.stack_wrap();
        let v = self.read32w(bus, reg::SS, self.stack_ptr(), wrap)?;
        self.adjust_sp(4);
        Ok(v)
    }

    /// Pop an operand-sized value (zero-extended).
    #[inline]
    pub(crate) fn pop<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        if self.osize32 {
            self.pop32(bus)
        } else {
            Ok(self.pop16(bus)? as u32)
        }
    }

    /// Pop a segment selector: a 32-bit operand size still performs only a
    /// 16-bit read (no fault at SP = FFFE on a 64 KiB stack) but releases
    /// four bytes — 386 behavior asserted by the hardware test suite.
    pub(crate) fn pop_sreg<B: Bus>(&mut self, bus: &mut B) -> Exec<u16> {
        let wrap = self.stack_wrap();
        let v = self.read16w(bus, reg::SS, self.stack_ptr(), wrap)?;
        self.adjust_sp(if self.osize32 { 4 } else { 2 });
        Ok(v)
    }

    // --- Interrupt and exception delivery --------------------------------------

    /// Deliver exception/interrupt `e` at the current `CS:EIP`. `sw_int` marks
    /// `INT n`-family traps (affects protected-mode privilege checks). Nested
    /// delivery failure escalates to double fault, then shutdown.
    pub(crate) fn deliver<B: Bus>(&mut self, bus: &mut B, e: Exception, sw_int: bool) {
        match self.raise(bus, e, sw_int) {
            Ok(()) => (),
            Err(e2) => {
                // Double fault, then triple fault → shutdown.
                let df = Exception {
                    vector: 8,
                    error: Some(0),
                };
                let _ = e2;
                if self.raise(bus, df, false).is_err() {
                    self.shutdown = true;
                    self.halted = true;
                }
            }
        }
    }

    /// The fallible part of delivery: real mode uses the IVT at `IDTR.base`;
    /// protected mode — including V86 — goes through the IDT gates (see
    /// `protected.rs`).
    fn raise<B: Bus>(&mut self, bus: &mut B, e: Exception, sw_int: bool) -> Exec<()> {
        if self.regs.cr0 & cr0::PE != 0 {
            self.interrupt_protected(bus, e, sw_int)
        } else {
            self.interrupt_real(bus, e.vector)
        }
    }

    /// Real-mode interrupt: push FLAGS/CS/IP (16-bit), clear IF/TF, and load
    /// `CS:IP` from the vector table described by IDTR.
    fn interrupt_real<B: Bus>(&mut self, bus: &mut B, vector: u8) -> Exec<()> {
        let entry = vector as u32 * 4;
        if entry + 3 > self.regs.idtr.limit as u32 {
            return Err(Exception::gp(vector as u16 * 8 + 2));
        }
        let base = self.regs.idtr.base;
        let ip = self.lin_read16(bus, base + entry)?;
        let cs = self.lin_read16(bus, base + entry + 2)?;

        let flags = self.regs.eflags.image16();
        self.push16(bus, flags)?;
        let (old_cs, old_ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.eip);
        self.push16(bus, old_cs)?;
        self.push16(bus, old_ip as u16)?;
        self.regs.eflags.remove(EFlags::IF | EFlags::TF);
        self.regs.seg[reg::CS as usize].sel = cs;
        self.regs.seg[reg::CS as usize].base = (cs as u32) << 4;
        self.regs.eip = ip as u32;
        Ok(())
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu::new()
    }
}
