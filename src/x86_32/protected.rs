//! Protected-mode segmentation: descriptor loading, privilege checks, far
//! control transfers (including call gates), interrupt/trap gates with stack
//! switching and Virtual-8086 entry/exit, and the I/O permission bitmap.
//!
//! Hardware task switching (task gates, `JMP/CALL TSS`, `IRET` with NT) is
//! not implemented: those transfers raise #GP. Everything else — including
//! entering and leaving V86 mode — is modeled.

use super::registers::{EFlags, Registers, SegReg, cr0, reg};
use super::{Bus, Cpu, Event, Exception, Exec};

/// A parsed segment/gate descriptor.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Descriptor {
    /// Linear address of the descriptor (to write the accessed bit back).
    addr: u32,
    pub base: u32,
    /// Byte-granular effective limit.
    pub limit: u32,
    /// Access byte (bits 0–7) plus flags nibble (bits 8–11).
    pub attrs: u16,
    /// Raw low/high dwords (gates reuse the fields differently).
    lo: u32,
    hi: u32,
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

    #[inline]
    pub fn dpl(&self) -> u8 {
        ((self.attrs >> 5) & 3) as u8
    }

    /// Gate target selector.
    #[inline]
    fn gate_sel(&self) -> u16 {
        (self.lo >> 16) as u16
    }

    /// Gate target offset (32-bit gates use the high word too).
    #[inline]
    fn gate_off(&self, wide: bool) -> u32 {
        let lo = self.lo & 0xFFFF;
        if wide {
            (self.hi & 0xFFFF_0000) | lo
        } else {
            lo
        }
    }

    /// Call-gate parameter count.
    #[inline]
    fn gate_params(&self) -> u32 {
        self.hi & 0x1F
    }
}

impl Cpu {
    // --- Descriptor tables -----------------------------------------------------

    /// The `(base, limit)` of the descriptor table a selector refers to. The
    /// LDT limit is a full 32-bit byte limit (its descriptor may be page
    /// granular), unlike the 16-bit GDTR/IDTR limits.
    fn table(&self, sel: u16) -> (u32, u32) {
        if sel & 4 != 0 {
            (self.regs.ldtr.base, self.regs.ldtr.limit)
        } else {
            (self.regs.gdtr.base, self.regs.gdtr.limit as u32)
        }
    }

    /// Read the 8-byte descriptor for `sel`, raising `#GP(sel)` if it lies
    /// outside its table. Descriptor-table reads are implicit supervisor
    /// accesses, so they bypass page-level user/supervisor checks.
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
        let addr = base.wrapping_add(index);
        let lo = self.sys_read32(bus, addr)?;
        let hi = self.sys_read32(bus, addr.wrapping_add(4))?;
        Ok(Self::parse_descriptor(addr, lo, hi))
    }

    fn parse_descriptor(addr: u32, lo: u32, hi: u32) -> Descriptor {
        let base = (lo >> 16) | ((hi & 0xFF) << 16) | (hi & 0xFF00_0000);
        let mut limit = (lo & 0xFFFF) | (hi & 0x000F_0000);
        if hi & 0x0080_0000 != 0 {
            limit = (limit << 12) | 0xFFF; // 4 KiB granularity
        }
        let attrs = (((hi >> 8) & 0x0F00) | ((hi >> 8) & 0xFF)) as u16;
        Descriptor {
            addr,
            base,
            limit,
            attrs,
            lo,
            hi,
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
            s.base = (sel as u32) << 4;
            Ok(())
        } else if self.regs.eflags.contains(EFlags::VM) {
            self.regs.seg[idx as usize] = SegReg {
                sel,
                base: (sel as u32) << 4,
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
        // Null selector: legal to load, faults on use.
        if sel & 0xFFFC == 0 {
            self.regs.seg[idx as usize] = SegReg {
                sel,
                base: 0,
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
        if sel & 0xFFFC == 0 {
            return Err(Exception::gp(0));
        }
        let rpl = (sel & 3) as u8;
        let cpl = self.cpl();
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

    // --- Far control transfers -------------------------------------------------------

    /// Far JMP to `sel:off`.
    pub(crate) fn jump_far<B: Bus>(&mut self, bus: &mut B, sel: u16, off: u32) -> Exec<()> {
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
            if off > d.limit {
                return Err(Exception::gp(0));
            }
            self.load_cs(bus, sel, &d, self.cpl())?;
            self.regs.eip = off;
            Ok(())
        } else {
            match d.sys_type() {
                4 | 12 => {
                    // Call gate used by JMP: same checks as CALL but the
                    // target must not be more privileged.
                    let (gsel, goff, _, wide) = self.check_gate(sel, &d)?;
                    let gd = self.read_descriptor(bus, gsel)?;
                    self.check_gate_target(gsel, &gd)?;
                    if !gd.is_conforming() && gd.dpl() != self.cpl() {
                        return Err(Exception::gp(gsel & 0xFFFC));
                    }
                    let off = if wide { goff } else { goff & 0xFFFF };
                    if off > gd.limit {
                        return Err(Exception::gp(0));
                    }
                    self.load_cs(bus, gsel, &gd, self.cpl())?;
                    self.regs.eip = off;
                    Ok(())
                }
                // Task gates / TSS switching are not implemented.
                _ => Err(Exception::gp(sel & 0xFFFC)),
            }
        }
    }

    /// Far CALL to `sel:off` (pushes the return address).
    pub(crate) fn call_far<B: Bus>(&mut self, bus: &mut B, sel: u16, off: u32) -> Exec<()> {
        if !self.protected_mode() {
            let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.eip);
            self.push(bus, cs as u32)?;
            self.push(bus, ip)?;
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
            if off > d.limit {
                return Err(Exception::gp(0));
            }
            let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.eip);
            self.push(bus, cs as u32)?;
            self.push(bus, ip)?;
            self.load_cs(bus, sel, &d, self.cpl())?;
            self.regs.eip = off;
            Ok(())
        } else {
            match d.sys_type() {
                4 | 12 => self.call_gate(bus, sel, &d),
                _ => Err(Exception::gp(sel & 0xFFFC)),
            }
        }
    }

    /// Real/V86 far transfer: selector reloads the base, offset checked
    /// against the (sticky) limit.
    fn far_real(&mut self, sel: u16, off: u32) -> Exec<()> {
        let cs = &mut self.regs.seg[reg::CS as usize];
        if off > cs.limit {
            return Err(Exception::gp(0));
        }
        cs.sel = sel;
        cs.base = (sel as u32) << 4;
        self.regs.eip = off;
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
    /// (target selector, offset, param count, 32-bit?).
    fn check_gate(&self, sel: u16, d: &Descriptor) -> Exec<(u16, u32, u32, bool)> {
        let cpl = self.cpl();
        let rpl = (sel & 3) as u8;
        if d.dpl() < cpl || d.dpl() < rpl {
            return Err(Exception::gp(sel & 0xFFFC));
        }
        if !d.present() {
            return Err(Exception::np(sel & 0xFFFC));
        }
        let wide = d.sys_type() == 12;
        Ok((d.gate_sel(), d.gate_off(wide), d.gate_params(), wide))
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
        Ok(())
    }

    /// CALL through a call gate, possibly switching to an inner stack.
    fn call_gate<B: Bus>(&mut self, bus: &mut B, sel: u16, gate: &Descriptor) -> Exec<()> {
        let (gsel, goff, params, wide) = self.check_gate(sel, gate)?;
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
                let off = old_sp.wrapping_add(i * if wide { 4 } else { 2 });
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
            self.regs.eip = if wide { goff } else { goff & 0xFFFF };
            Ok(())
        } else {
            // Same privilege through the gate.
            let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.eip);
            if wide {
                self.push32(bus, cs as u32)?;
                self.push32(bus, ip)?;
            } else {
                self.push16(bus, cs)?;
                self.push16(bus, ip as u16)?;
            }
            self.load_cs(bus, gsel, &gd, cpl)?;
            self.regs.eip = if wide { goff } else { goff & 0xFFFF };
            Ok(())
        }
    }

    /// Push a call-gate return frame (outer SS:SP, copied parameters, then
    /// CS:IP) onto the freshly switched inner stack.
    fn push_gate_frame<B: Bus>(
        &mut self,
        bus: &mut B,
        wide: bool,
        old_ss: u16,
        old_sp: u32,
        args: &[u32; 32],
        params: u32,
    ) -> Exec<()> {
        let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.eip);
        if wide {
            self.push32(bus, old_ss as u32)?;
            self.push32(bus, old_sp)?;
            for i in (0..params).rev() {
                self.push32(bus, args[i as usize])?;
            }
            self.push32(bus, cs as u32)?;
            self.push32(bus, ip)?;
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

    /// Fetch the inner-stack pointer for privilege level `level` from the
    /// current TSS.
    fn tss_stack<B: Bus>(&mut self, bus: &mut B, level: u8) -> Exec<(u16, u32)> {
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
            let sp = self.sys_read32(bus, tr.base.wrapping_add(off))?;
            let ss = self.sys_read16(bus, tr.base.wrapping_add(off + 4))?;
            Ok((ss, sp))
        } else {
            let off = 2 + level as u32 * 4;
            if off + 3 > tr.limit {
                return Err(tss_err);
            }
            let sp = self.sys_read16(bus, tr.base.wrapping_add(off))? as u32;
            let ss = self.sys_read16(bus, tr.base.wrapping_add(off + 2))?;
            Ok((ss, sp))
        }
    }

    /// Load SS:eSP for an inner-privilege transition. The SS descriptor must
    /// be a writable data segment at the new privilege level.
    fn switch_stack<B: Bus>(&mut self, bus: &mut B, ss: u16, sp: u32, new_cpl: u8) -> Exec<()> {
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
            self.regs.gpr[reg::ESP as usize] = sp;
        } else {
            self.regs.set_reg16(reg::ESP, sp as u16);
        }
        Ok(())
    }

    // --- RETF / IRET --------------------------------------------------------------

    /// Far return, releasing `n` extra bytes of stack.
    pub(crate) fn retf<B: Bus>(&mut self, bus: &mut B, n: u32) -> Exec<u32> {
        if !self.protected_mode() {
            let ip = self.pop(bus)?;
            let cs = self.pop(bus)? as u16;
            self.adjust_sp(n as i32);
            self.far_real(cs, if self.osize32 { ip } else { ip & 0xFFFF })?;
            return Ok(10);
        }

        let ip = self.pop(bus)?;
        let sel = self.pop(bus)? as u16;
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
        if ip > d.limit {
            return Err(Exception::gp(0));
        }

        self.adjust_sp(n as i32);
        if rpl > cpl {
            // Outer return: restore the caller's stack too.
            let sp = self.pop(bus)?;
            let ss = self.pop(bus)? as u16;
            self.load_cs(bus, sel, &d, rpl)?;
            self.regs.eip = ip;
            self.load_outer_ss(bus, ss, sp, rpl)?;
            self.adjust_sp(n as i32);
            self.validate_data_segs(rpl);
        } else {
            self.load_cs(bus, sel, &d, rpl)?;
            self.regs.eip = ip;
        }
        Ok(18)
    }

    /// SS load during an outer return (RETF/IRET to lower privilege).
    fn load_outer_ss<B: Bus>(&mut self, bus: &mut B, ss: u16, sp: u32, rpl: u8) -> Exec<()> {
        if ss & 0xFFFC == 0 {
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
        if d.db() {
            self.regs.gpr[reg::ESP as usize] = sp;
        } else {
            self.regs.set_reg16(reg::ESP, sp as u16);
        }
        Ok(())
    }

    /// After dropping privilege, data segment registers that are no longer
    /// reachable are silently nulled (386 behavior).
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
            let ip = self.pop(bus)?;
            let cs = self.pop(bus)? as u16;
            let fl = self.pop(bus)?;
            self.far_real(cs, if self.osize32 { ip } else { ip & 0xFFFF })?;
            let mask = self.popf_mask();
            self.regs.eflags.load(fl, mask);
            return Ok(22);
        }
        if self.regs.eflags.contains(EFlags::VM) {
            // V86: IOPL-sensitive.
            if self.regs.eflags.iopl() < 3 {
                return Err(Exception::gp(0));
            }
            let ip = self.pop(bus)?;
            let cs = self.pop(bus)? as u16;
            let fl = self.pop(bus)?;
            self.far_real(cs, if self.osize32 { ip } else { ip & 0xFFFF })?;
            let mask = self.popf_mask() & !(EFlags::IOPL | EFlags::VM).bits();
            self.regs.eflags.load(fl, mask);
            return Ok(22);
        }

        if self.regs.eflags.contains(EFlags::NT) {
            // Task return — hardware task switching is not implemented.
            return Err(Exception::gp(0));
        }

        let ip = self.pop(bus)?;
        let sel = self.pop(bus)? as u16;
        let fl = self.pop(bus)?;

        // IRETD from CPL 0 with VM set in the popped image → return to V86.
        // At CPL > 0 the VM bit is simply not writable (it is absent from
        // `iret_flag_mask`), so the return proceeds as an ordinary one.
        if self.osize32 && fl & EFlags::VM.bits() != 0 && self.cpl() == 0 {
            return self.iret_to_v86(bus, ip, sel, fl);
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
        if ip > d.limit {
            return Err(Exception::gp(0));
        }

        let flag_mask = self.iret_flag_mask(cpl);
        if rpl > cpl {
            let sp = self.pop(bus)?;
            let ss = self.pop(bus)? as u16;
            self.load_cs(bus, sel, &d, rpl)?;
            // `load_outer_ss` can still fault (#GP/#SS on a bad user stack
            // selector); commit the popped flag image only once it cannot,
            // or a fault would enter the handler with the *returned-to*
            // EFLAGS — the register the fault path deliberately preserves.
            self.load_outer_ss(bus, ss, sp, rpl)?;
            self.regs.eip = ip;
            self.regs.eflags.load(fl, flag_mask);
            self.validate_data_segs(rpl);
        } else {
            self.load_cs(bus, sel, &d, rpl)?;
            self.regs.eip = ip;
            self.regs.eflags.load(fl, flag_mask);
        }
        Ok(22)
    }

    /// EFLAGS bits IRET may write at privilege `cpl`.
    fn iret_flag_mask(&self, cpl: u8) -> u32 {
        let mut mask = 0x0000_7FD5u32;
        if self.osize32 {
            mask |= EFlags::RF.bits();
        }
        if cpl > 0 {
            mask &= !EFlags::IOPL.bits();
            if cpl > self.regs.eflags.iopl() {
                mask &= !EFlags::IF.bits();
            }
        }
        mask
    }

    /// IRETD with VM=1: restore the V86 frame (EIP CS EFLAGS ESP SS ES DS FS GS).
    fn iret_to_v86<B: Bus>(&mut self, bus: &mut B, ip: u32, cs: u16, fl: u32) -> Exec<u32> {
        let sp = self.pop32(bus)?;
        let ss = self.pop32(bus)? as u16;
        let es = self.pop32(bus)? as u16;
        let ds = self.pop32(bus)? as u16;
        let fs = self.pop32(bus)? as u16;
        let gs = self.pop32(bus)? as u16;

        // All defined bits (including VM and IOPL) come from the image.
        self.regs.eflags = EFlags::from_bits_truncate(fl);
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
                base: (sel as u32) << 4,
                limit: 0xFFFF,
                attrs: 0x00F3,
            };
        }
        self.regs.eip = ip & 0xFFFF;
        self.regs.gpr[reg::ESP as usize] = sp;
        Ok(60)
    }

    // --- Interrupt delivery through the IDT ------------------------------------------

    /// Protected-mode (and V86) interrupt/exception delivery.
    ///
    /// `class` selects the gate DPL check (`INT n` needs gate DPL >= CPL) and
    /// the `EXT` bit of any error code raised while delivering an external
    /// interrupt.
    pub(crate) fn interrupt_protected<B: Bus>(
        &mut self,
        bus: &mut B,
        e: Exception,
        class: Event,
    ) -> Exec<()> {
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
        let addr = self.regs.idtr.base.wrapping_add(entry);
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
        let goff = gate.gate_off(wide);
        if goff > gd.limit {
            return Err(Exception::gp(ext));
        }

        if self.regs.eflags.contains(EFlags::VM) {
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
        self.regs.eip = if wide { goff } else { goff & 0xFFFF };
        self.regs
            .eflags
            .remove(EFlags::TF | EFlags::NT | EFlags::RF);
        if !trap {
            self.regs.eflags.remove(EFlags::IF);
        }
        Ok(())
    }

    /// Push FLAGS/CS/IP (+ optional outer SS:SP, + optional error code) at
    /// the gate's width.
    fn push_int_frame<B: Bus>(
        &mut self,
        bus: &mut B,
        wide: bool,
        outer: Option<(u16, u32)>,
        error: Option<u16>,
    ) -> Exec<()> {
        let (cs, ip) = (self.regs.seg[reg::CS as usize].sel, self.regs.eip);
        if wide {
            if let Some((ss, sp)) = outer {
                self.push32(bus, ss as u32)?;
                self.push32(bus, sp)?;
            }
            let fl = self.regs.eflags.bits() | 2;
            self.push32(bus, fl)?;
            self.push32(bus, cs as u32)?;
            self.push32(bus, ip)?;
            if let Some(err) = error {
                self.push32(bus, err as u32)?;
            }
        } else {
            if let Some((ss, sp)) = outer {
                self.push16(bus, ss)?;
                self.push16(bus, sp as u16)?;
            }
            let fl = self.regs.eflags.image16();
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
        osp: u32,
        error: Option<u16>,
    ) -> Exec<()> {
        if wide {
            for idx in [reg::GS, reg::FS, reg::DS, reg::ES] {
                self.push32(bus, old.seg[idx as usize].sel as u32)?;
            }
            self.push32(bus, old.seg[reg::SS as usize].sel as u32)?;
            self.push32(bus, osp)?;
            self.push32(bus, old.eflags.bits() | 2)?;
            self.push32(bus, old.seg[reg::CS as usize].sel as u32)?;
            self.push32(bus, old.eip)?;
            if let Some(err) = error {
                self.push32(bus, err as u32)?;
            }
        } else {
            for idx in [reg::GS, reg::FS, reg::DS, reg::ES] {
                self.push16(bus, old.seg[idx as usize].sel)?;
            }
            self.push16(bus, old.seg[reg::SS as usize].sel)?;
            self.push16(bus, osp as u16)?;
            self.push16(bus, old.eflags.image16())?;
            self.push16(bus, old.seg[reg::CS as usize].sel)?;
            self.push16(bus, old.eip as u16)?;
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
        goff: u32,
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
            .eflags
            .remove(EFlags::VM | EFlags::TF | EFlags::RF | EFlags::NT);
        if !trap {
            self.regs.eflags.remove(EFlags::IF);
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
        self.regs.eip = if wide { goff } else { goff & 0xFFFF };
        Ok(())
    }

    /// Software interrupt (INT n / INT3 / INTO / ICEBP): traps report the
    /// *next* instruction, which EIP already points to.
    pub(crate) fn software_int<B: Bus>(&mut self, bus: &mut B, vector: u8) -> Exec<()> {
        self.raise(
            bus,
            Exception {
                vector,
                error: None,
            },
            Event::SoftInt,
        )
    }

    // --- I/O permission ------------------------------------------------------------------

    /// Check I/O access legality for `size` bytes at `port`. In protected
    /// mode with CPL > IOPL (and always in V86 mode) the TSS I/O permission
    /// bitmap arbitrates.
    pub(crate) fn io_check<B: Bus>(&mut self, bus: &mut B, port: u16, size: u8) -> Exec<()> {
        if self.regs.cr0 & cr0::PE == 0 {
            return Ok(());
        }
        if !self.regs.eflags.contains(EFlags::VM) && self.cpl() <= self.regs.eflags.iopl() {
            return Ok(());
        }
        // Consult the TSS I/O permission bitmap (32-bit TSS only).
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
            bits |= (self.sys_read8(bus, tr.base.wrapping_add(a))? as u32) << (8 * i);
        }
        let shift = port as u32 & 7;
        let mask = ((1u32 << size) - 1) << shift;
        if bits & mask != 0 {
            return Err(Exception::gp(0));
        }
        Ok(())
    }
}
