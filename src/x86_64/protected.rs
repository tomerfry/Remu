//! Protected- and long-mode segmentation: descriptor loading (16-byte system
//! descriptors in long mode), privilege checks, far control transfers
//! (including call gates), interrupt/trap gates with stack switching, IST,
//! Virtual-8086 entry/exit (legacy mode), the SYSCALL/SYSRET fast system
//! calls, and the I/O permission bitmap.
//!
//! Hardware task switching (task gates, `JMP/CALL TSS`, `IRET` with NT) is
//! not implemented: those transfers raise #GP — matching long mode, which
//! removed task switching from the architecture entirely.

use super::registers::{RFlags, Registers, SegReg, cr0, efer, reg};
use super::{Bus, Cpu, Event, Exception, Exec};
use super::{O16, O32};

/// A parsed segment/gate descriptor.
///
/// Long-mode system descriptors and gates occupy 16 bytes; `hi2` carries
/// their upper half (zero for 8-byte descriptors).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Descriptor {
    /// Linear address of the descriptor (to write the accessed bit back).
    addr: u64,
    pub base: u64,
    /// Byte-granular effective limit.
    pub limit: u32,
    /// Access byte (bits 0–7) plus flags nibble (bits 8–11).
    pub attrs: u16,
    /// Raw low/high dwords of the low half (gates reuse the fields).
    lo: u32,
    hi: u32,
    /// Upper 8 bytes of a 16-byte descriptor.
    hi2: u64,
}

impl Descriptor {
    /// Descriptor type field (bits 0–3 of the access byte) for system
    /// descriptors and gates (S == 0).
    #[inline]
    fn sys_type(&self) -> u8 {
        (self.attrs & 0x0F) as u8
    }

    /// Code/data (S) descriptor?
    #[inline]
    fn is_code_data(&self) -> bool {
        self.attrs & 0x10 != 0
    }

    #[inline]
    fn is_code(&self) -> bool {
        self.attrs & 0x18 == 0x18
    }

    #[inline]
    fn is_writable_data(&self) -> bool {
        self.attrs & 0x1A == 0x12
    }

    #[inline]
    fn is_readable(&self) -> bool {
        // Data segments are always readable; code needs the R bit.
        self.attrs & 0x08 == 0 || self.attrs & 0x02 != 0
    }

    #[inline]
    fn is_conforming(&self) -> bool {
        self.attrs & 0x1C == 0x1C
    }

    #[inline]
    pub(crate) fn present(&self) -> bool {
        self.attrs & 0x80 != 0
    }

    /// The D/B flag (default size / stack width).
    #[inline]
    fn db(&self) -> bool {
        self.attrs & 0x0400 != 0
    }

    /// The L flag (64-bit code segment).
    #[inline]
    fn l(&self) -> bool {
        self.attrs & 0x0200 != 0
    }

    #[inline]
    pub fn dpl(&self) -> u8 {
        ((self.attrs >> 5) & 3) as u8
    }

    /// Gate target selector.
    #[inline]
    fn gate_sel(&self) -> u16 {
        (self.lo >> 16) as u16
    }

    /// Gate target offset: 16-bit gates use the low word, 32-bit gates the
    /// high word too, 64-bit gates the upper half as well.
    #[inline]
    fn gate_off(&self, wide: bool, long: bool) -> u64 {
        let mut off = (self.lo & 0xFFFF) as u64;
        if wide {
            off |= (self.hi & 0xFFFF_0000) as u64;
        }
        if long {
            off |= (self.hi2 as u32 as u64) << 32;
        }
        off
    }

    /// Call-gate parameter count (legacy gates only; long-mode gates have
    /// none).
    #[inline]
    fn gate_params(&self) -> u32 {
        self.hi & 0x1F
    }

    /// Interrupt-stack-table index of a long-mode interrupt gate (0 = none).
    #[inline]
    fn gate_ist(&self) -> u8 {
        (self.hi & 7) as u8
    }
}

impl Cpu {
    // --- Descriptor tables -----------------------------------------------------

    /// The `(base, limit)` of the descriptor table a selector refers to. The
    /// LDT limit is a full 32-bit byte limit (its descriptor may be page
    /// granular), unlike the 16-bit GDTR/IDTR limits.
    fn table(&self, sel: u16) -> (u64, u32) {
        if sel & 4 != 0 {
            (self.regs.ldtr.base, self.regs.ldtr.limit)
        } else {
            (self.regs.gdtr.base, self.regs.gdtr.limit as u32)
        }
    }

    /// Read the descriptor for `sel`, raising `#GP(sel)` if it lies outside
    /// its table. In long mode a system descriptor (S == 0) is 16 bytes; its
    /// upper half is fetched too (with the extended limit check) and its
    /// must-be-zero type field validated. Descriptor-table reads are
    /// implicit supervisor accesses, so they bypass page-level
    /// user/supervisor checks.
    pub(crate) fn read_descriptor<B: Bus>(&mut self, bus: &mut B, sel: u16) -> Exec<Descriptor> {
        self.read_descriptor_err(bus, sel, Exception::gp)
    }

    /// [`Cpu::read_descriptor`] with a caller-chosen fault for an
    /// out-of-range selector — a selector taken from the TSS must raise `#TS`
    /// rather than `#GP`.
    fn read_descriptor_err<B: Bus>(
        &mut self,
        bus: &mut B,
        sel: u16,
        err: fn(u16) -> Exception,
    ) -> Exec<Descriptor> {
        let (base, limit) = self.table(sel);
        let index = (sel & 0xFFF8) as u32;
        if index + 7 > limit {
            return Err(err(sel & 0xFFFC));
        }
        let addr = base.wrapping_add(index as u64);
        let lo = self.sys_read32(bus, addr)?;
        let hi = self.sys_read32(bus, addr.wrapping_add(4))?;
        let mut d = Self::parse_descriptor(addr, lo, hi);
        if self.long_mode() && !d.is_code_data() {
            // 16-byte system descriptor: base 63:32 plus a must-be-zero
            // type field that keeps the upper half from aliasing a valid
            // descriptor.
            if index + 15 > limit {
                return Err(err(sel & 0xFFFC));
            }
            let hi2 = self.sys_read64(bus, addr.wrapping_add(8))?;
            if hi2 & 0x0000_1F00 != 0 {
                return Err(err(sel & 0xFFFC));
            }
            d.hi2 = hi2;
            d.base |= (hi2 as u32 as u64) << 32;
        }
        Ok(d)
    }

    fn parse_descriptor(addr: u64, lo: u32, hi: u32) -> Descriptor {
        let base = ((lo >> 16) | ((hi & 0xFF) << 16) | (hi & 0xFF00_0000)) as u64;
        let mut limit = (lo & 0xFFFF) | (hi & 0x000F_0000);
        if hi & 0x0080_0000 != 0 {
            limit = (limit << 12) | 0xFFF; // 4 KiB granularity
        }
        // Access byte (descriptor bits 8–15) into attrs bits 0–7, and the
        // flags nibble G/D-B/L/AVL (descriptor bits 20–23) into attrs bits
        // 8–11 — a `>> 12`, not `>> 8`, so D/B and L land where `db()`/`l()`
        // read them.
        let attrs = (((hi >> 12) & 0x0F00) | ((hi >> 8) & 0xFF)) as u16;
        Descriptor {
            addr,
            base,
            limit,
            attrs,
            lo,
            hi,
            hi2: 0,
        }
    }

    /// Set the accessed bit of a loaded segment descriptor (a real memory
    /// write, as on hardware).
    fn mark_accessed<B: Bus>(&mut self, bus: &mut B, d: &Descriptor) -> Exec<()> {
        if d.is_code_data() && d.attrs & 1 == 0 {
            let byte = ((d.hi >> 8) & 0xFF) as u8 | 1;
            self.sys_write8(bus, d.addr.wrapping_add(5), byte)?;
        }
        Ok(())
    }

    /// Install descriptor `d` into segment register `idx`.
    fn commit_seg(&mut self, idx: u8, sel: u16, d: &Descriptor) {
        self.regs.seg[idx as usize] = SegReg {
            sel,
            base: d.base,
            limit: d.limit,
            attrs: d.attrs | 0x80,
        };
    }

    // --- Segment register loads ---------------------------------------------------

    /// Load segment register `idx` (not CS) with `sel`, per the current mode.
    pub(crate) fn load_seg<B: Bus>(&mut self, bus: &mut B, idx: u8, sel: u16) -> Exec<()> {
        if self.regs.cr0 & cr0::PE == 0 {
            // Real mode: base tracks the selector; limit/attrs are sticky.
            let s = &mut self.regs.seg[idx as usize];
            s.sel = sel;
            s.base = (sel as u64) << 4;
            Ok(())
        } else if self.regs.rflags.contains(RFlags::VM) {
            self.regs.seg[idx as usize] = SegReg {
                sel,
                base: (sel as u64) << 4,
                limit: 0xFFFF,
                attrs: 0x00F3,
            };
            Ok(())
        } else {
            self.load_seg_protected(bus, idx, sel)
        }
    }

    fn load_seg_protected<B: Bus>(&mut self, bus: &mut B, idx: u8, sel: u16) -> Exec<()> {
        if idx == reg::SS {
            return self.load_ss(bus, sel);
        }
        // Null selector: legal to load, faults on use (legacy modes; 64-bit
        // mode does not check data segments on use at all).
        if sel & 0xFFFC == 0 {
            // Loading FS/GS with null keeps their (MSR-visible) base — real
            // hardware only clears the descriptor-derived attributes.
            let base = if idx == reg::FS || idx == reg::GS {
                self.regs.seg[idx as usize].base
            } else {
                0
            };
            self.regs.seg[idx as usize] = SegReg {
                sel,
                base,
                limit: 0,
                attrs: 0,
            };
            return Ok(());
        }
        let d = self.read_descriptor(bus, sel)?;
        // Must be data or readable code.
        if !d.is_code_data() || (d.is_code() && !d.is_readable()) {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        // Privilege: DPL >= max(CPL, RPL) unless conforming code.
        if !d.is_conforming() {
            let rpl = (sel & 3) as u8;
            if d.dpl() < self.cpl() || d.dpl() < rpl {
                return Err(Exception::gp(sel & 0xFFFC));
            }
        }
        if !d.present() {
            return Err(Exception::np(sel & 0xFFFC));
        }
        self.mark_accessed(bus, &d)?;
        self.commit_seg(idx, sel, &d);
        Ok(())
    }

    fn load_ss<B: Bus>(&mut self, bus: &mut B, sel: u16) -> Exec<()> {
        let cpl = self.cpl();
        if sel & 0xFFFC == 0 {
            // 64-bit mode allows a null SS below CPL 3 (the handler-entry
            // convention); everywhere else it faults.
            if self.mode64() && cpl != 3 && (sel & 3) as u8 == cpl {
                self.regs.seg[reg::SS as usize] = SegReg {
                    sel,
                    base: 0,
                    limit: 0,
                    attrs: 0,
                };
                return Ok(());
            }
            return Err(Exception::gp(0));
        }
        let rpl = (sel & 3) as u8;
        if rpl != cpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        let d = self.read_descriptor(bus, sel)?;
        if !d.is_writable_data() || d.dpl() != cpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::ss(sel & 0xFFFC));
        }
        self.mark_accessed(bus, &d)?;
        self.commit_seg(reg::SS, sel, &d);
        Ok(())
    }

    /// Load CS for a same-privilege transfer (far JMP/CALL target, or the
    /// code segment reached through a gate). Forces CS.RPL = CPL.
    fn load_cs<B: Bus>(&mut self, bus: &mut B, sel: u16, d: &Descriptor, cpl: u8) -> Exec<()> {
        self.mark_accessed(bus, d)?;
        self.commit_seg(reg::CS, (sel & 0xFFFC) | cpl as u16, d);
        Ok(())
    }

    /// Validate a code descriptor's L/D combination for the current mode and
    /// whether `off` is a legal target within it. In long mode L=1 requires
    /// D=0 and a canonical offset; L=0 (compat) and legacy segments check
    /// the limit.
    fn check_code_target(&self, d: &Descriptor, sel: u16, off: u64) -> Exec<()> {
        if self.long_mode() {
            if d.l() && d.db() {
                return Err(Exception::gp(sel & 0xFFFC));
            }
            if d.l() {
                if !Self::canonical(off) {
                    return Err(Exception::gp(0));
                }
                return Ok(());
            }
        }
        if off > d.limit as u64 {
            return Err(Exception::gp(0));
        }
        Ok(())
    }

    // --- Far control transfers -------------------------------------------------------

    /// Far JMP to `sel:off`.
    pub(crate) fn jump_far<B: Bus>(&mut self, bus: &mut B, sel: u16, off: u64) -> Exec<()> {
        if !self.protected_mode() {
            return self.far_real(sel, off);
        }
        if sel & 0xFFFC == 0 {
            return Err(Exception::gp(0));
        }
        let d = self.read_descriptor(bus, sel)?;
        if d.is_code_data() {
            if !d.is_code() {
                return Err(Exception::gp(sel & 0xFFFC));
            }
            self.check_code_reachable(sel, &d)?;
            if !d.present() {
                return Err(Exception::np(sel & 0xFFFC));
            }
            self.check_code_target(&d, sel, off)?;
            self.load_cs(bus, sel, &d, self.cpl())?;
            self.regs.rip = off;
            Ok(())
        } else {
            match d.sys_type() {
                12 if self.long_mode() => {
                    // Long-mode call gate used by JMP.
                    let (gsel, goff, _) = self.check_gate(sel, &d, true)?;
                    let gd = self.read_descriptor(bus, gsel)?;
                    self.check_gate_target(gsel, &gd)?;
                    if !gd.is_conforming() && gd.dpl() != self.cpl() {
                        return Err(Exception::gp(gsel & 0xFFFC));
                    }
                    self.check_code_target(&gd, gsel, goff)?;
                    self.load_cs(bus, gsel, &gd, self.cpl())?;
                    self.regs.rip = goff;
                    Ok(())
                }
                4 | 12 if !self.long_mode() => {
                    // Legacy call gate used by JMP: same checks as CALL but
                    // the target must not be more privileged.
                    let wide = d.sys_type() == 12;
                    let (gsel, goff, _) = self.check_gate(sel, &d, false)?;
                    let gd = self.read_descriptor(bus, gsel)?;
                    self.check_gate_target(gsel, &gd)?;
                    if !gd.is_conforming() && gd.dpl() != self.cpl() {
                        return Err(Exception::gp(gsel & 0xFFFC));
                    }
                    let off = if wide { goff } else { goff & 0xFFFF };
                    if off > gd.limit as u64 {
                        return Err(Exception::gp(0));
                    }
                    self.load_cs(bus, gsel, &gd, self.cpl())?;
                    self.regs.rip = off;
                    Ok(())
                }
                // Task gates / TSS switching are not implemented.
                _ => Err(Exception::gp(sel & 0xFFFC)),
            }
        }
    }

    /// Far CALL to `sel:off` (pushes the return address).
    pub(crate) fn call_far<B: Bus>(&mut self, bus: &mut B, sel: u16, off: u64) -> Exec<()> {
        if !self.protected_mode() {
            let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.rip);
            self.push_op(bus, cs as u64)?;
            self.push_op(bus, ip)?;
            return self.far_real(sel, off);
        }
        if sel & 0xFFFC == 0 {
            return Err(Exception::gp(0));
        }
        let d = self.read_descriptor(bus, sel)?;
        if d.is_code_data() {
            if !d.is_code() {
                return Err(Exception::gp(sel & 0xFFFC));
            }
            self.check_code_reachable(sel, &d)?;
            if !d.present() {
                return Err(Exception::np(sel & 0xFFFC));
            }
            self.check_code_target(&d, sel, off)?;
            let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.rip);
            self.push_op(bus, cs as u64)?;
            self.push_op(bus, ip)?;
            self.load_cs(bus, sel, &d, self.cpl())?;
            self.regs.rip = off;
            Ok(())
        } else {
            match d.sys_type() {
                12 if self.long_mode() => self.call_gate_long(bus, sel, &d),
                4 | 12 if !self.long_mode() => self.call_gate_legacy(bus, sel, &d),
                _ => Err(Exception::gp(sel & 0xFFFC)),
            }
        }
    }

    /// Real/V86 far transfer: selector reloads the base, offset checked
    /// against the (sticky) limit.
    fn far_real(&mut self, sel: u16, off: u64) -> Exec<()> {
        let cs = &mut self.regs.seg[reg::CS as usize];
        if off > cs.limit as u64 {
            return Err(Exception::gp(0));
        }
        cs.sel = sel;
        cs.base = (sel as u64) << 4;
        self.regs.rip = off;
        Ok(())
    }

    /// Direct far-transfer code-segment privilege rules (no gate).
    fn check_code_reachable(&self, sel: u16, d: &Descriptor) -> Exec<()> {
        let cpl = self.cpl();
        let rpl = (sel & 3) as u8;
        if d.is_conforming() {
            if d.dpl() > cpl {
                return Err(Exception::gp(sel & 0xFFFC));
            }
        } else if rpl > cpl || d.dpl() != cpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        Ok(())
    }

    /// Validate a call/interrupt gate itself: DPL and presence. Returns
    /// (target selector, offset, param count).
    fn check_gate(&self, sel: u16, d: &Descriptor, long: bool) -> Exec<(u16, u64, u32)> {
        let cpl = self.cpl();
        let rpl = (sel & 3) as u8;
        if d.dpl() < cpl || d.dpl() < rpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::np(sel & 0xFFFC));
        }
        let wide = long || d.sys_type() == 12;
        Ok((
            d.gate_sel(),
            d.gate_off(wide, long),
            if long { 0 } else { d.gate_params() },
        ))
    }

    /// Validate the code segment a gate points to.
    fn check_gate_target(&self, gsel: u16, gd: &Descriptor) -> Exec<()> {
        if gsel & 0xFFFC == 0 {
            return Err(Exception::gp(0));
        }
        if !gd.is_code() {
            return Err(Exception::gp(gsel & 0xFFFC));
        }
        if gd.dpl() > self.cpl() {
            return Err(Exception::gp(gsel & 0xFFFC));
        }
        if !gd.present() {
            return Err(Exception::np(gsel & 0xFFFC));
        }
        // A long-mode gate must target 64-bit code.
        if self.long_mode() && (!gd.l() || gd.db()) {
            return Err(Exception::gp(gsel & 0xFFFC));
        }
        Ok(())
    }

    /// CALL through a legacy call gate, possibly switching to an inner stack.
    fn call_gate_legacy<B: Bus>(&mut self, bus: &mut B, sel: u16, gate: &Descriptor) -> Exec<()> {
        let wide = gate.sys_type() == 12;
        let (gsel, goff, params) = self.check_gate(sel, gate, false)?;
        let gd = self.read_descriptor(bus, gsel)?;
        self.check_gate_target(gsel, &gd)?;

        let cpl = self.cpl();
        if !gd.is_conforming() && gd.dpl() < cpl {
            // More-privileged: switch to the inner stack from the TSS.
            let new_cpl = gd.dpl();
            let (new_ss, new_sp) = self.tss_stack(bus, new_cpl)?;
            let (old_ss, old_sp) = (self.regs.seg[reg::SS as usize].sel, self.stack_ptr());

            // Copy the parameters from the old stack before switching.
            let mut args = [0u32; 32];
            for i in 0..params {
                let off = old_sp.wrapping_add(i as u64 * if wide { 4 } else { 2 });
                args[i as usize] = if wide {
                    self.read32(bus, reg::SS, off)?
                } else {
                    self.read16(bus, reg::SS, off)? as u32
                };
            }

            self.switch_stack(bus, new_ss, new_sp, new_cpl)?;
            // CS is still the outer segment, so CPL would misclassify these
            // pushes to the inner stack as user accesses under paging.
            let sup = self.supervisor_override;
            self.supervisor_override = true;
            let pushed = self.push_gate_frame(bus, wide, old_ss, old_sp, &args, params);
            self.supervisor_override = sup;
            pushed?;
            self.load_cs(bus, gsel, &gd, new_cpl)?;
            self.regs.rip = if wide { goff } else { goff & 0xFFFF };
            Ok(())
        } else {
            // Same privilege through the gate.
            let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.rip);
            if wide {
                self.push32(bus, cs as u32)?;
                self.push32(bus, ip as u32)?;
            } else {
                self.push16(bus, cs)?;
                self.push16(bus, ip as u16)?;
            }
            self.load_cs(bus, gsel, &gd, cpl)?;
            self.regs.rip = if wide { goff } else { goff & 0xFFFF };
            Ok(())
        }
    }

    /// CALL through a long-mode (16-byte) call gate: pushes are always
    /// 64-bit and there are no parameters; an inner transition loads a null
    /// SS with the new RPL and an RSP from the 64-bit TSS.
    fn call_gate_long<B: Bus>(&mut self, bus: &mut B, sel: u16, gate: &Descriptor) -> Exec<()> {
        let (gsel, goff, _) = self.check_gate(sel, gate, true)?;
        let gd = self.read_descriptor(bus, gsel)?;
        self.check_gate_target(gsel, &gd)?;
        if !Self::canonical(goff) {
            return Err(Exception::gp(0));
        }

        let cpl = self.cpl();
        let (old_ss, old_sp) = (self.regs.seg[reg::SS as usize].sel, self.stack_ptr());
        let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.rip);

        if !gd.is_conforming() && gd.dpl() < cpl {
            let new_cpl = gd.dpl();
            let mut rsp = self.tss_rsp(bus, new_cpl)?;
            self.set_null_ss(new_cpl);
            let sup = self.supervisor_override;
            self.supervisor_override = true;
            let pushed = (|| -> Exec<()> {
                self.push64_at(bus, &mut rsp, old_ss as u64)?;
                self.push64_at(bus, &mut rsp, old_sp)?;
                self.push64_at(bus, &mut rsp, cs as u64)?;
                self.push64_at(bus, &mut rsp, ip)
            })();
            self.supervisor_override = sup;
            pushed?;
            self.regs.gpr[reg::RSP as usize] = rsp;
            self.load_cs(bus, gsel, &gd, new_cpl)?;
        } else {
            self.push64(bus, cs as u64)?;
            self.push64(bus, ip)?;
            self.load_cs(bus, gsel, &gd, cpl)?;
        }
        self.regs.rip = goff;
        Ok(())
    }

    /// Push a call-gate return frame (outer SS:SP, copied parameters, then
    /// CS:IP) onto the freshly switched inner stack (legacy gates).
    fn push_gate_frame<B: Bus>(
        &mut self,
        bus: &mut B,
        wide: bool,
        old_ss: u16,
        old_sp: u64,
        args: &[u32; 32],
        params: u32,
    ) -> Exec<()> {
        let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.rip);
        if wide {
            self.push32(bus, old_ss as u32)?;
            self.push32(bus, old_sp as u32)?;
            for i in (0..params).rev() {
                self.push32(bus, args[i as usize])?;
            }
            self.push32(bus, cs as u32)?;
            self.push32(bus, ip as u32)?;
        } else {
            self.push16(bus, old_ss)?;
            self.push16(bus, old_sp as u16)?;
            for i in (0..params).rev() {
                self.push16(bus, args[i as usize] as u16)?;
            }
            self.push16(bus, cs)?;
            self.push16(bus, ip as u16)?;
        }
        Ok(())
    }

    /// Write a quad-word at the flat 64-bit stack address `*rsp - 8` and
    /// decrement it — the push primitive of long-mode delivery, which is
    /// 64-bit flat regardless of the interrupted code's submode.
    fn push64_at<B: Bus>(&mut self, bus: &mut B, rsp: &mut u64, v: u64) -> Exec<()> {
        let a = rsp.wrapping_sub(8);
        if !Self::canonical(a) || !Self::canonical(a.wrapping_add(7)) {
            return Err(Exception::ss(0));
        }
        self.lin_write64(bus, a, v)?;
        *rsp = a;
        Ok(())
    }

    /// Load SS with a null selector carrying `rpl` (long-mode inner
    /// transitions).
    fn set_null_ss(&mut self, rpl: u8) {
        self.regs.seg[reg::SS as usize] = SegReg {
            sel: rpl as u16,
            base: 0,
            limit: 0,
            attrs: 0,
        };
    }

    /// Fetch the inner-stack pointer for privilege level `level` from the
    /// current legacy TSS.
    fn tss_stack<B: Bus>(&mut self, bus: &mut B, level: u8) -> Exec<(u16, u64)> {
        let tr = self.regs.tr;
        let tss_err = Exception::ts(tr.sel & 0xFFFC);
        let wide = tr.attrs & 0x08 != 0; // type 9/B = 32-bit, 1/3 = 16-bit
        if wide {
            // ESP at `off`, SS at `off+4`: the last byte read is `off+5`,
            // which must lie within the limit.
            let off = 4 + level as u32 * 8;
            if off + 5 > tr.limit {
                return Err(tss_err);
            }
            let sp = self.sys_read32(bus, tr.base.wrapping_add(off as u64))?;
            let ss = self.sys_read16(bus, tr.base.wrapping_add(off as u64 + 4))?;
            Ok((ss, sp as u64))
        } else {
            let off = 2 + level as u32 * 4;
            if off + 3 > tr.limit {
                return Err(tss_err);
            }
            let sp = self.sys_read16(bus, tr.base.wrapping_add(off as u64))? as u64;
            let ss = self.sys_read16(bus, tr.base.wrapping_add(off as u64 + 2))?;
            Ok((ss, sp))
        }
    }

    /// Fetch RSPn from the 64-bit TSS (long-mode inner transitions).
    fn tss_rsp<B: Bus>(&mut self, bus: &mut B, level: u8) -> Exec<u64> {
        let tr = self.regs.tr;
        let off = 4 + level as u32 * 8;
        if off + 7 > tr.limit {
            return Err(Exception::ts(tr.sel & 0xFFFC));
        }
        self.sys_read64(bus, tr.base.wrapping_add(off as u64))
    }

    /// Fetch ISTn (1–7) from the 64-bit TSS.
    fn tss_ist<B: Bus>(&mut self, bus: &mut B, n: u8) -> Exec<u64> {
        let tr = self.regs.tr;
        let off = 0x24 + (n as u32 - 1) * 8;
        if off + 7 > tr.limit {
            return Err(Exception::ts(tr.sel & 0xFFFC));
        }
        self.sys_read64(bus, tr.base.wrapping_add(off as u64))
    }

    /// Load SS:eSP for an inner-privilege transition (legacy). The SS
    /// descriptor must be a writable data segment at the new privilege level.
    fn switch_stack<B: Bus>(&mut self, bus: &mut B, ss: u16, sp: u64, new_cpl: u8) -> Exec<()> {
        if ss & 0xFFFC == 0 {
            return Err(Exception::ts(0));
        }
        if (ss & 3) as u8 != new_cpl {
            return Err(Exception::ts(ss & 0xFFFC));
        }
        let d = self.read_descriptor_err(bus, ss, Exception::ts)?;
        if !d.is_writable_data() || d.dpl() != new_cpl {
            return Err(Exception::ts(ss & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::ss(ss & 0xFFFC));
        }
        self.mark_accessed(bus, &d)?;
        self.commit_seg(reg::SS, ss, &d);
        if d.db() {
            self.regs.gpr[reg::RSP as usize] = sp as u32 as u64;
        } else {
            self.regs.set_reg16(reg::RSP, sp as u16);
        }
        Ok(())
    }

    // --- RETF / IRET --------------------------------------------------------------

    /// Far return, releasing `n` extra bytes of stack.
    pub(crate) fn retf<B: Bus>(&mut self, bus: &mut B, n: u64) -> Exec<u32> {
        if !self.protected_mode() {
            let ip = self.pop_op(bus)?;
            let cs = self.pop_op(bus)? as u16;
            self.adjust_sp(n as i64);
            self.far_real(cs, if self.osize == O16 { ip & 0xFFFF } else { ip })?;
            return Ok(10);
        }

        let ip = self.pop_op(bus)?;
        let sel = self.pop_op(bus)? as u16;
        if sel & 0xFFFC == 0 {
            return Err(Exception::gp(0));
        }
        let cpl = self.cpl();
        let rpl = (sel & 3) as u8;
        if rpl < cpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        let d = self.read_descriptor(bus, sel)?;
        if !d.is_code() {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if d.is_conforming() {
            if d.dpl() > rpl {
                return Err(Exception::gp(sel & 0xFFFC));
            }
        } else if d.dpl() != rpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::np(sel & 0xFFFC));
        }
        self.check_code_target(&d, sel, ip)?;

        self.adjust_sp(n as i64);
        if rpl > cpl {
            // Outer return: restore the caller's stack too.
            let sp = self.pop_op(bus)?;
            let ss = self.pop_op(bus)? as u16;
            self.load_cs(bus, sel, &d, rpl)?;
            self.regs.rip = ip;
            self.load_outer_ss(bus, ss, sp, rpl)?;
            self.adjust_sp(n as i64);
            self.validate_data_segs(rpl);
        } else {
            self.load_cs(bus, sel, &d, rpl)?;
            self.regs.rip = ip;
        }
        Ok(18)
    }

    /// SS load during an outer return (RETF/IRET to lower privilege).
    fn load_outer_ss<B: Bus>(&mut self, bus: &mut B, ss: u16, sp: u64, rpl: u8) -> Exec<()> {
        if ss & 0xFFFC == 0 {
            // Long mode: a null SS is legal when not returning to CPL 3.
            if self.long_mode() && rpl != 3 && (ss & 3) as u8 == rpl {
                self.set_null_ss(rpl);
                self.regs.gpr[reg::RSP as usize] = sp;
                return Ok(());
            }
            return Err(Exception::gp(0));
        }
        if (ss & 3) as u8 != rpl {
            return Err(Exception::gp(ss & 0xFFFC));
        }
        let d = self.read_descriptor(bus, ss)?;
        if !d.is_writable_data() || d.dpl() != rpl {
            return Err(Exception::gp(ss & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::ss(ss & 0xFFFC));
        }
        self.mark_accessed(bus, &d)?;
        self.commit_seg(reg::SS, ss, &d);
        if self.long_mode() && self.regs.seg[reg::CS as usize].l() {
            self.regs.gpr[reg::RSP as usize] = sp;
        } else if d.db() {
            self.regs.gpr[reg::RSP as usize] = sp as u32 as u64;
        } else {
            self.regs.set_reg16(reg::RSP, sp as u16);
        }
        Ok(())
    }

    /// After dropping privilege, data segment registers that are no longer
    /// reachable are silently nulled.
    fn validate_data_segs(&mut self, cpl: u8) {
        for idx in [reg::ES, reg::DS, reg::FS, reg::GS] {
            let s = self.regs.seg[idx as usize];
            let present = s.attrs & 0x80 != 0;
            let conforming_code = s.attrs & 0x1C == 0x1C;
            let dpl = ((s.attrs >> 5) & 3) as u8;
            if present && !conforming_code && dpl < cpl {
                self.regs.seg[idx as usize] = SegReg {
                    sel: 0,
                    base: 0,
                    limit: 0,
                    attrs: 0,
                };
            }
        }
    }

    /// IRET at the current operand size.
    pub(crate) fn iret<B: Bus>(&mut self, bus: &mut B) -> Exec<u32> {
        if self.regs.cr0 & cr0::PE == 0 {
            // Real mode.
            let ip = self.pop_op(bus)?;
            let cs = self.pop_op(bus)? as u16;
            let fl = self.pop_op(bus)?;
            self.far_real(cs, if self.osize == O16 { ip & 0xFFFF } else { ip })?;
            let mask = self.popf_mask();
            self.regs.rflags.load(fl, mask);
            return Ok(22);
        }
        if self.regs.rflags.contains(RFlags::VM) {
            // V86: IOPL-sensitive.
            if self.regs.rflags.iopl() < 3 {
                return Err(Exception::gp(0));
            }
            let ip = self.pop_op(bus)?;
            let cs = self.pop_op(bus)? as u16;
            let fl = self.pop_op(bus)?;
            self.far_real(cs, if self.osize == O16 { ip & 0xFFFF } else { ip })?;
            let mask = self.popf_mask() & !(RFlags::IOPL | RFlags::VM).bits();
            self.regs.rflags.load(fl, mask);
            return Ok(22);
        }

        if self.regs.rflags.contains(RFlags::NT) && !self.long_mode() {
            // Task return — hardware task switching is not implemented.
            return Err(Exception::gp(0));
        }

        let ip = self.pop_op(bus)?;
        let sel = self.pop_op(bus)? as u16;
        let fl = self.pop_op(bus)?;

        // IRETD from CPL 0 with VM set in the popped image → return to V86
        // (legacy mode only; the VM bit is ignored in long mode).
        if !self.long_mode()
            && self.osize == O32
            && fl & RFlags::VM.bits() as u64 != 0
            && self.cpl() == 0
        {
            return self.iret_to_v86(bus, ip, sel, fl);
        }

        // In 64-bit mode IRET always pops SS:RSP, even without a privilege
        // change.
        let pop_stack_always = self.m64;
        let (mut new_sp, mut new_ss) = (0u64, 0u16);
        if pop_stack_always {
            new_sp = self.pop_op(bus)?;
            new_ss = self.pop_op(bus)? as u16;
        }

        if sel & 0xFFFC == 0 {
            return Err(Exception::gp(0));
        }
        let cpl = self.cpl();
        let rpl = (sel & 3) as u8;
        if rpl < cpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        let d = self.read_descriptor(bus, sel)?;
        if !d.is_code() {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if d.is_conforming() {
            if d.dpl() > rpl {
                return Err(Exception::gp(sel & 0xFFFC));
            }
        } else if d.dpl() != rpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::np(sel & 0xFFFC));
        }
        self.check_code_target(&d, sel, ip)?;

        let flag_mask = self.iret_flag_mask(cpl);
        if rpl > cpl {
            if !pop_stack_always {
                new_sp = self.pop_op(bus)?;
                new_ss = self.pop_op(bus)? as u16;
            }
            self.load_cs(bus, sel, &d, rpl)?;
            // `load_outer_ss` can still fault (#GP/#SS on a bad user stack
            // selector); commit the popped flag image only once it cannot,
            // or a fault would enter the handler with the *returned-to*
            // RFLAGS — the register the fault path deliberately preserves.
            self.load_outer_ss(bus, new_ss, new_sp, rpl)?;
            self.regs.rip = ip;
            self.regs.rflags.load(fl, flag_mask);
            self.validate_data_segs(rpl);
        } else {
            self.load_cs(bus, sel, &d, rpl)?;
            if pop_stack_always {
                self.load_outer_ss(bus, new_ss, new_sp, rpl)?;
            }
            self.regs.rip = ip;
            self.regs.rflags.load(fl, flag_mask);
        }
        Ok(22)
    }

    /// RFLAGS bits IRET may write at privilege `cpl`.
    fn iret_flag_mask(&self, cpl: u8) -> u32 {
        let mut mask = 0x0024_7FD5u32; // defined 16-bit flags + AC + ID
        if self.osize != O16 {
            mask |= RFlags::RF.bits();
        }
        if cpl > 0 {
            mask &= !RFlags::IOPL.bits();
            if cpl > self.regs.rflags.iopl() {
                mask &= !RFlags::IF.bits();
            }
        }
        mask
    }

    /// IRETD with VM=1: restore the V86 frame (EIP CS EFLAGS ESP SS ES DS FS GS).
    fn iret_to_v86<B: Bus>(&mut self, bus: &mut B, ip: u64, cs: u16, fl: u64) -> Exec<u32> {
        let sp = self.pop32(bus)?;
        let ss = self.pop32(bus)? as u16;
        let es = self.pop32(bus)? as u16;
        let ds = self.pop32(bus)? as u16;
        let fs = self.pop32(bus)? as u16;
        let gs = self.pop32(bus)? as u16;

        // All defined bits (including VM and IOPL) come from the image.
        self.regs.rflags = RFlags::from_bits_truncate(fl as u32);
        for (idx, sel) in [
            (reg::CS, cs),
            (reg::SS, ss),
            (reg::ES, es),
            (reg::DS, ds),
            (reg::FS, fs),
            (reg::GS, gs),
        ] {
            self.regs.seg[idx as usize] = SegReg {
                sel,
                base: (sel as u64) << 4,
                limit: 0xFFFF,
                attrs: 0x00F3,
            };
        }
        self.regs.rip = ip & 0xFFFF;
        self.regs.gpr[reg::RSP as usize] = sp as u64;
        Ok(60)
    }

    // --- Interrupt delivery through the IDT ------------------------------------------

    /// Protected-mode interrupt/exception delivery: long mode (16-byte
    /// gates, 64-bit frames, IST) when EFER.LMA is set, the legacy path
    /// (with V86 support) otherwise.
    pub(crate) fn interrupt_protected<B: Bus>(
        &mut self,
        bus: &mut B,
        e: Exception,
        class: Event,
    ) -> Exec<()> {
        if self.long_mode() {
            self.interrupt_long(bus, e, class)
        } else {
            self.interrupt_legacy(bus, e, class)
        }
    }

    /// Long-mode delivery: the IDT holds 16-byte gates, the frame is five
    /// (or six, with an error code) quad-words on a 16-byte-aligned stack,
    /// and the handler always runs in 64-bit code.
    fn interrupt_long<B: Bus>(&mut self, bus: &mut B, e: Exception, class: Event) -> Exec<()> {
        let sw = class == Event::SoftInt;
        let ext = (class == Event::External) as u16;
        let sel_err = |sel: u16| sel & 0xFFFC | ext;

        let vector = e.vector;
        let idt_err = (vector as u16) * 8 + 2 + ext;
        let entry = vector as u32 * 16;
        if entry + 15 > self.regs.idtr.limit as u32 {
            return Err(Exception::gp(idt_err));
        }
        let addr = self.regs.idtr.base.wrapping_add(entry as u64);
        let lo = self.sys_read32(bus, addr)?;
        let hi = self.sys_read32(bus, addr.wrapping_add(4))?;
        let hi2 = self.sys_read64(bus, addr.wrapping_add(8))?;
        let mut gate = Self::parse_descriptor(addr, lo, hi);
        gate.hi2 = hi2;

        let trap = match gate.sys_type() {
            14 => false,
            15 => true,
            _ => return Err(Exception::gp(idt_err)),
        };
        if sw && gate.dpl() < self.cpl() {
            return Err(Exception::gp(idt_err));
        }
        if !gate.present() {
            return Err(Exception::np(idt_err));
        }

        let gsel = gate.gate_sel();
        if gsel & 0xFFFC == 0 {
            return Err(Exception::gp(ext));
        }
        let gd = self.read_descriptor(bus, gsel)?;
        if !gd.is_code() || gd.dpl() > self.cpl() {
            return Err(Exception::gp(sel_err(gsel)));
        }
        if !gd.present() {
            return Err(Exception::np(sel_err(gsel)));
        }
        if !gd.l() || gd.db() {
            return Err(Exception::gp(sel_err(gsel)));
        }
        let goff = gate.gate_off(true, true);
        if !Self::canonical(goff) {
            return Err(Exception::gp(ext));
        }

        let cpl = self.cpl();
        let new_cpl = if gd.is_conforming() { cpl } else { gd.dpl() };
        let cpl_change = new_cpl < cpl;

        // Stack selection: IST wins; otherwise TSS.RSPn on privilege change;
        // otherwise stay on the current stack. The new RSP is then aligned
        // down to 16 bytes, as the architecture specifies for 64-bit frames.
        let ist = gate.gate_ist();
        let mut rsp = if ist != 0 {
            self.tss_ist(bus, ist)?
        } else if cpl_change {
            self.tss_rsp(bus, new_cpl)?
        } else {
            self.stack_ptr()
        };
        rsp &= !0xF;

        let old = self.regs;
        let old_rsp = self.stack_ptr();
        if cpl_change {
            self.set_null_ss(new_cpl);
        }

        // The frame is pushed with supervisor privilege: CS still holds the
        // interrupted (possibly user) code segment while these land on the
        // handler stack.
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let pushed = (|| -> Exec<()> {
            self.push64_at(bus, &mut rsp, old.seg[reg::SS as usize].sel as u64)?;
            self.push64_at(bus, &mut rsp, old_rsp)?;
            self.push64_at(bus, &mut rsp, old.rflags.bits() as u64 | 2)?;
            self.push64_at(bus, &mut rsp, old.seg[reg::CS as usize].sel as u64)?;
            self.push64_at(bus, &mut rsp, old.rip)?;
            if let Some(err) = e.error {
                self.push64_at(bus, &mut rsp, err as u64)?;
            }
            Ok(())
        })();
        self.supervisor_override = sup;
        pushed?;

        self.regs.gpr[reg::RSP as usize] = rsp;
        self.load_cs(bus, gsel, &gd, new_cpl)?;
        self.regs.rip = goff;
        self.regs
            .rflags
            .remove(RFlags::TF | RFlags::NT | RFlags::RF);
        if !trap {
            self.regs.rflags.remove(RFlags::IF);
        }
        Ok(())
    }

    /// Legacy protected-mode (and V86) delivery through 8-byte gates.
    fn interrupt_legacy<B: Bus>(&mut self, bus: &mut B, e: Exception, class: Event) -> Exec<()> {
        let sw = class == Event::SoftInt;
        // EXT: the fault was raised while delivering an external event.
        let ext = (class == Event::External) as u16;
        let sel_err = |sel: u16| sel & 0xFFFC | ext;

        let vector = e.vector;
        let idt_err = (vector as u16) * 8 + 2 + ext;
        let entry = vector as u32 * 8;
        if entry + 7 > self.regs.idtr.limit as u32 {
            return Err(Exception::gp(idt_err));
        }
        let addr = self.regs.idtr.base.wrapping_add(entry as u64);
        let lo = self.sys_read32(bus, addr)?;
        let hi = self.sys_read32(bus, addr.wrapping_add(4))?;
        let gate = Self::parse_descriptor(addr, lo, hi);

        let (trap, wide) = match gate.sys_type() {
            6 => (false, false),
            7 => (true, false),
            14 => (false, true),
            15 => (true, true),
            // Task gates are not implemented.
            _ => return Err(Exception::gp(idt_err)),
        };
        if sw && gate.dpl() < self.cpl() {
            return Err(Exception::gp(idt_err));
        }
        if !gate.present() {
            return Err(Exception::np(idt_err));
        }

        let gsel = gate.gate_sel();
        if gsel & 0xFFFC == 0 {
            return Err(Exception::gp(ext));
        }
        let gd = self.read_descriptor(bus, gsel)?;
        if !gd.is_code() || gd.dpl() > self.cpl() {
            return Err(Exception::gp(sel_err(gsel)));
        }
        if !gd.present() {
            return Err(Exception::np(sel_err(gsel)));
        }
        let goff = gate.gate_off(wide, false);
        if goff > gd.limit as u64 {
            return Err(Exception::gp(ext));
        }

        if self.regs.rflags.contains(RFlags::VM) {
            return self.v86_interrupt(bus, e, &gd, gsel, goff, trap, wide, ext);
        }

        let cpl = self.cpl();
        if !gd.is_conforming() && gd.dpl() < cpl {
            // Inner transition: switch stacks and push SS:ESP too.
            let new_cpl = gd.dpl();
            let (nss, nsp) = self.tss_stack(bus, new_cpl)?;
            let (oss, osp) = (self.regs.seg[reg::SS as usize].sel, self.stack_ptr());
            self.switch_stack(bus, nss, nsp, new_cpl)?;
            // CS still holds the outer segment, so these pushes onto the
            // inner stack must not be page-checked against the outer CPL.
            let sup = self.supervisor_override;
            self.supervisor_override = true;
            let pushed = self.push_int_frame(bus, wide, Some((oss, osp)), e.error);
            self.supervisor_override = sup;
            pushed?;
            self.load_cs(bus, gsel, &gd, new_cpl)?;
        } else if gd.is_conforming() || gd.dpl() == cpl {
            self.push_int_frame(bus, wide, None, e.error)?;
            self.load_cs(bus, gsel, &gd, cpl)?;
        } else {
            return Err(Exception::gp(sel_err(gsel)));
        }
        self.regs.rip = if wide { goff } else { goff & 0xFFFF };
        self.regs
            .rflags
            .remove(RFlags::TF | RFlags::NT | RFlags::RF);
        if !trap {
            self.regs.rflags.remove(RFlags::IF);
        }
        Ok(())
    }

    /// Push FLAGS/CS/IP (+ optional outer SS:SP, + optional error code) at
    /// the gate's width (legacy delivery).
    fn push_int_frame<B: Bus>(
        &mut self,
        bus: &mut B,
        wide: bool,
        outer: Option<(u16, u64)>,
        error: Option<u16>,
    ) -> Exec<()> {
        let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.rip);
        if wide {
            if let Some((ss, sp)) = outer {
                self.push32(bus, ss as u32)?;
                self.push32(bus, sp as u32)?;
            }
            let fl = self.regs.rflags.bits() | 2;
            self.push32(bus, fl)?;
            self.push32(bus, cs as u32)?;
            self.push32(bus, ip as u32)?;
            if let Some(err) = error {
                self.push32(bus, err as u32)?;
            }
        } else {
            if let Some((ss, sp)) = outer {
                self.push16(bus, ss)?;
                self.push16(bus, sp as u16)?;
            }
            let fl = self.regs.rflags.image16();
            self.push16(bus, fl)?;
            self.push16(bus, cs)?;
            self.push16(bus, ip as u16)?;
            if let Some(err) = error {
                self.push16(bus, err)?;
            }
        }
        Ok(())
    }

    /// Push the V86 interrupt frame: the four V86 data selectors, then the
    /// interrupted SS:SP, EFLAGS, CS:IP and any error code.
    fn push_v86_frame<B: Bus>(
        &mut self,
        bus: &mut B,
        wide: bool,
        old: &Registers,
        osp: u64,
        error: Option<u16>,
    ) -> Exec<()> {
        if wide {
            for idx in [reg::GS, reg::FS, reg::DS, reg::ES] {
                self.push32(bus, old.seg[idx as usize].sel as u32)?;
            }
            self.push32(bus, old.seg[reg::SS as usize].sel as u32)?;
            self.push32(bus, osp as u32)?;
            self.push32(bus, old.rflags.bits() | 2)?;
            self.push32(bus, old.seg[reg::CS as usize].sel as u32)?;
            self.push32(bus, old.rip as u32)?;
            if let Some(err) = error {
                self.push32(bus, err as u32)?;
            }
        } else {
            for idx in [reg::GS, reg::FS, reg::DS, reg::ES] {
                self.push16(bus, old.seg[idx as usize].sel)?;
            }
            self.push16(bus, old.seg[reg::SS as usize].sel)?;
            self.push16(bus, osp as u16)?;
            self.push16(bus, old.rflags.image16())?;
            self.push16(bus, old.seg[reg::CS as usize].sel)?;
            self.push16(bus, old.rip as u16)?;
            if let Some(err) = error {
                self.push16(bus, err)?;
            }
        }
        Ok(())
    }

    /// Interrupt while in V86 mode: switch to the ring-0 handler, pushing the
    /// V86 segment registers first.
    #[allow(clippy::too_many_arguments)]
    fn v86_interrupt<B: Bus>(
        &mut self,
        bus: &mut B,
        e: Exception,
        gd: &Descriptor,
        gsel: u16,
        goff: u64,
        trap: bool,
        wide: bool,
        ext: u16,
    ) -> Exec<()> {
        // The handler must run at ring 0.
        if gd.dpl() != 0 || gd.is_conforming() {
            return Err(Exception::gp(gsel & 0xFFFC | ext));
        }
        let (nss, nsp) = self.tss_stack(bus, 0)?;
        let old = self.regs;
        let osp = self.stack_ptr();

        // Leaving V86 mode and switching stacks happens before the pushes,
        // which can still fault; `Cpu::raise` rolls the whole delivery back
        // in that case, so the nested fault sees VM and the V86 stack intact.
        self.regs
            .rflags
            .remove(RFlags::VM | RFlags::TF | RFlags::RF | RFlags::NT);
        if !trap {
            self.regs.rflags.remove(RFlags::IF);
        }
        self.switch_stack(bus, nss, nsp, 0)?;

        // The ring-0 frame is pushed with supervisor privilege (CS still
        // holds the V86 code segment, whose CPL is 3).
        let sup = self.supervisor_override;
        self.supervisor_override = true;
        let pushed = self.push_v86_frame(bus, wide, &old, osp, e.error);
        self.supervisor_override = sup;
        pushed?;

        // The V86 data segments are unusable in the handler.
        for idx in [reg::ES, reg::DS, reg::FS, reg::GS] {
            self.regs.seg[idx as usize] = SegReg {
                sel: 0,
                base: 0,
                limit: 0,
                attrs: 0,
            };
        }
        self.mark_accessed(bus, gd)?;
        self.commit_seg(reg::CS, gsel & 0xFFFC, gd);
        self.regs.rip = if wide { goff } else { goff & 0xFFFF };
        Ok(())
    }

    /// Software interrupt (INT n / INT3 / INTO / ICEBP): traps report the
    /// *next* instruction, which RIP already points to.
    pub(crate) fn software_int<B: Bus>(&mut self, bus: &mut B, vector: u8) -> Exec<()> {
        // OS-emulation hook: the designated syscall vector (e.g. int 0x80) is
        // caught here — before any IDT/gate/TSS logic — so it works at CPL 3
        // with no IDT installed. RIP already points past the INT instruction.
        if self.syscall_int == Some(vector) {
            self.host_trap = Some(super::HostTrap::Syscall);
            return Ok(());
        }
        self.raise(
            bus,
            Exception {
                vector,
                error: None,
            },
            Event::SoftInt,
        )
    }

    // --- SYSCALL / SYSRET / SWAPGS ---------------------------------------------------

    /// SYSCALL (64-bit mode): RCX ← next RIP, R11 ← RFLAGS, then jump to
    /// LSTAR at CPL 0 with the STAR-derived flat selectors. Requires
    /// EFER.SCE; #UD outside 64-bit mode (Intel behavior — AMD also allows
    /// compatibility-mode SYSCALL, which this core does not).
    pub(crate) fn syscall(&mut self) -> Exec<u32> {
        if !self.m64 || self.regs.msr.efer & efer::SCE == 0 {
            return Err(Exception::ud());
        }
        self.regs.gpr[reg::RCX as usize] = self.regs.rip;
        self.regs.gpr[reg::R11 as usize] =
            (self.regs.rflags.bits() & !RFlags::RF.bits()) as u64 | 2;

        // OS-emulation hook: hand the call to the host instead of LSTAR.
        if self.trap_syscall {
            self.host_trap = Some(super::HostTrap::Syscall);
            return Ok(2);
        }

        let star = self.regs.msr.star;
        let sel = (star >> 32) as u16 & 0xFFFC;
        self.regs.seg[reg::CS as usize] = SegReg {
            sel,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: 0x0A9B, // 64-bit ring-0 code
        };
        self.regs.seg[reg::SS as usize] = SegReg {
            sel: sel.wrapping_add(8),
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: 0x0C93, // flat ring-0 data
        };
        let mask = self.regs.msr.sfmask as u32 | RFlags::RF.bits();
        self.regs.rflags = RFlags::from_bits_truncate(self.regs.rflags.bits() & !mask);
        self.regs.rip = self.regs.msr.lstar;
        Ok(2)
    }

    /// SYSRET: return to CPL 3 at RCX (64-bit with REX.W, compatibility mode
    /// without), restoring RFLAGS from R11.
    pub(crate) fn sysret(&mut self) -> Exec<u32> {
        if !self.m64 || self.regs.msr.efer & efer::SCE == 0 {
            return Err(Exception::ud());
        }
        if self.cpl() != 0 {
            return Err(Exception::gp(0));
        }
        let star = self.regs.msr.star;
        let base = (star >> 48) as u16;
        let rip = self.regs.gpr[reg::RCX as usize];
        let (sel, attrs, new_rip) = if self.rex_w() {
            if !Self::canonical(rip) {
                return Err(Exception::gp(0));
            }
            (base.wrapping_add(16) | 3, 0x0A_FBu16, rip) // 64-bit ring-3 code
        } else {
            (base | 3, 0x0C_FBu16, rip as u32 as u64) // compat ring-3 code
        };
        self.regs.seg[reg::CS as usize] = SegReg {
            sel,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs,
        };
        self.regs.seg[reg::SS as usize] = SegReg {
            sel: base.wrapping_add(8) | 3,
            base: 0,
            limit: 0xFFFF_FFFF,
            attrs: 0x0CF3, // flat ring-3 data
        };
        let r11 = self.regs.gpr[reg::R11 as usize];
        self.regs.rflags = RFlags::from_bits_truncate((r11 as u32 & 0x003C_7FD7) | 2);
        self.regs.rip = new_rip;
        Ok(2)
    }

    /// SWAPGS (64-bit mode, ring 0): exchange GS.base with KernelGSBase.
    pub(crate) fn swapgs(&mut self) -> Exec<u32> {
        if !self.m64 {
            return Err(Exception::ud());
        }
        if self.cpl() != 0 {
            return Err(Exception::gp(0));
        }
        core::mem::swap(
            &mut self.regs.seg[reg::GS as usize].base,
            &mut self.regs.msr.kernel_gs_base,
        );
        Ok(2)
    }

    // --- I/O permission ------------------------------------------------------------------

    /// Check I/O access legality for `size` bytes at `port`. In protected
    /// mode with CPL > IOPL (and always in V86 mode) the TSS I/O permission
    /// bitmap arbitrates.
    pub(crate) fn io_check<B: Bus>(&mut self, bus: &mut B, port: u16, size: u8) -> Exec<()> {
        if self.regs.cr0 & cr0::PE == 0 {
            return Ok(());
        }
        if !self.regs.rflags.contains(RFlags::VM) && self.cpl() <= self.regs.rflags.iopl() {
            return Ok(());
        }
        // Consult the TSS I/O permission bitmap (32/64-bit TSS only).
        let tr = self.regs.tr;
        if tr.attrs & 0x08 == 0 {
            return Err(Exception::gp(0));
        }
        if 0x67 > tr.limit {
            return Err(Exception::gp(0));
        }
        let iobase = self.sys_read16(bus, tr.base.wrapping_add(0x66))? as u32;
        let first = iobase + (port as u32 >> 3);
        let last = iobase + ((port as u32 + size as u32 - 1) >> 3);
        if last > tr.limit {
            return Err(Exception::gp(0));
        }
        let mut bits = 0u32;
        for (i, a) in (first..=last).enumerate() {
            bits |= (self.sys_read8(bus, tr.base.wrapping_add(a as u64))? as u32) << (8 * i);
        }
        let shift = port as u32 & 7;
        let mask = ((1u32 << size) - 1) << shift;
        if bits & mask != 0 {
            return Err(Exception::gp(0));
        }
        Ok(())
    }
}
