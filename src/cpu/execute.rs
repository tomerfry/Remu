//! Instruction execution: the dispatch table and one handler per [`Operation`].
//!
//! Handlers read and write through the resolved [`Operand`] (and via
//! [`Cpu::read`]/[`Cpu::write`]), so each operation is implemented exactly once
//! regardless of addressing mode. [`Cpu::execute`] returns *extra* cycles beyond
//! the table's base count — currently only taken branches contribute.

use crate::bus::Bus;
use crate::cpu::addressing::{AddressingMode, Operand};
use crate::cpu::opcodes::{OpInfo, Operation};
use crate::cpu::registers::Status;
use crate::cpu::Cpu;
use crate::interrupt::IRQ_VECTOR;

impl Cpu {
    /// Dispatch one decoded instruction. Returns extra cycles (branch penalties).
    pub(crate) fn execute<B: Bus>(&mut self, bus: &mut B, info: &OpInfo, m: &Operand) -> u8 {
        use Operation::*;
        match info.operation {
            // Loads
            LDA => { let v = self.read(bus, m.addr); self.regs.a = v; self.regs.p.set_zn(v); 0 }
            LDX => { let v = self.read(bus, m.addr); self.regs.x = v; self.regs.p.set_zn(v); 0 }
            LDY => { let v = self.read(bus, m.addr); self.regs.y = v; self.regs.p.set_zn(v); 0 }

            // Stores
            STA => { self.write(bus, m.addr, self.regs.a); 0 }
            STX => { self.write(bus, m.addr, self.regs.x); 0 }
            STY => { self.write(bus, m.addr, self.regs.y); 0 }

            // Register transfers
            TAX => { self.regs.x = self.regs.a; self.regs.p.set_zn(self.regs.x); 0 }
            TAY => { self.regs.y = self.regs.a; self.regs.p.set_zn(self.regs.y); 0 }
            TXA => { self.regs.a = self.regs.x; self.regs.p.set_zn(self.regs.a); 0 }
            TYA => { self.regs.a = self.regs.y; self.regs.p.set_zn(self.regs.a); 0 }
            TSX => { self.regs.x = self.regs.sp; self.regs.p.set_zn(self.regs.x); 0 }
            TXS => { self.regs.sp = self.regs.x; 0 } // TXS does not affect flags

            // Stack
            PHA => { self.push(bus, self.regs.a); 0 }
            PHP => { self.push_status(bus, true); 0 }
            PLA => { let v = self.pull(bus); self.regs.a = v; self.regs.p.set_zn(v); 0 }
            PLP => { self.pull_status(bus); 0 }

            // Logical
            AND => { let v = self.read(bus, m.addr); self.regs.a &= v; self.regs.p.set_zn(self.regs.a); 0 }
            ORA => { let v = self.read(bus, m.addr); self.regs.a |= v; self.regs.p.set_zn(self.regs.a); 0 }
            EOR => { let v = self.read(bus, m.addr); self.regs.a ^= v; self.regs.p.set_zn(self.regs.a); 0 }
            BIT => { self.bit(bus, m); 0 }

            // Arithmetic
            ADC => { let v = self.read(bus, m.addr); self.adc(v); 0 }
            SBC => { let v = self.read(bus, m.addr); self.sbc(v); 0 }
            CMP => { let v = self.read(bus, m.addr); self.compare(self.regs.a, v); 0 }
            CPX => { let v = self.read(bus, m.addr); self.compare(self.regs.x, v); 0 }
            CPY => { let v = self.read(bus, m.addr); self.compare(self.regs.y, v); 0 }

            // Increment / decrement
            INC => { self.rmw(bus, m, |c, v| { let r = v.wrapping_add(1); c.regs.p.set_zn(r); r }); 0 }
            DEC => { self.rmw(bus, m, |c, v| { let r = v.wrapping_sub(1); c.regs.p.set_zn(r); r }); 0 }
            INX => { self.regs.x = self.regs.x.wrapping_add(1); self.regs.p.set_zn(self.regs.x); 0 }
            INY => { self.regs.y = self.regs.y.wrapping_add(1); self.regs.p.set_zn(self.regs.y); 0 }
            DEX => { self.regs.x = self.regs.x.wrapping_sub(1); self.regs.p.set_zn(self.regs.x); 0 }
            DEY => { self.regs.y = self.regs.y.wrapping_sub(1); self.regs.p.set_zn(self.regs.y); 0 }

            // Shifts / rotates
            ASL => { self.shift(bus, info.mode, m, Cpu::asl_val); 0 }
            LSR => { self.shift(bus, info.mode, m, Cpu::lsr_val); 0 }
            ROL => { self.shift(bus, info.mode, m, Cpu::rol_val); 0 }
            ROR => { self.shift(bus, info.mode, m, Cpu::ror_val); 0 }

            // Flag operations
            CLC => { self.regs.p.remove(Status::C); 0 }
            SEC => { self.regs.p.insert(Status::C); 0 }
            CLD => { self.regs.p.remove(Status::D); 0 }
            SED => { self.regs.p.insert(Status::D); 0 }
            CLI => { self.regs.p.remove(Status::I); 0 }
            SEI => { self.regs.p.insert(Status::I); 0 }
            CLV => { self.regs.p.remove(Status::V); 0 }

            // Branches
            BCC => self.branch(m, !self.regs.p.contains(Status::C)),
            BCS => self.branch(m, self.regs.p.contains(Status::C)),
            BNE => self.branch(m, !self.regs.p.contains(Status::Z)),
            BEQ => self.branch(m, self.regs.p.contains(Status::Z)),
            BPL => self.branch(m, !self.regs.p.contains(Status::N)),
            BMI => self.branch(m, self.regs.p.contains(Status::N)),
            BVC => self.branch(m, !self.regs.p.contains(Status::V)),
            BVS => self.branch(m, self.regs.p.contains(Status::V)),

            // Jumps / subroutines
            JMP => { self.regs.pc = m.addr; 0 }
            JSR => { self.jsr(bus); 0 }
            RTS => { self.rts(bus); 0 }

            // System
            BRK => { self.brk(bus); 0 }
            RTI => { self.rti(bus); 0 }
            NOP => 0,
            KIL => { self.halted = true; 0 }
        }
    }

    // --- Helpers ----------------------------------------------------------

    fn bit<B: Bus>(&mut self, bus: &mut B, m: &Operand) {
        let v = self.read(bus, m.addr);
        self.regs.p.set(Status::Z, (self.regs.a & v) == 0);
        self.regs.p.set(Status::V, v & 0x40 != 0);
        self.regs.p.set(Status::N, v & 0x80 != 0);
    }

    /// Compare `reg` against `v`, setting C/Z/N (no register is modified).
    fn compare(&mut self, reg: u8, v: u8) {
        let r = reg.wrapping_sub(v);
        self.regs.p.set(Status::C, reg >= v);
        self.regs.p.set_zn(r);
    }

    /// Add with carry, honoring decimal (BCD) mode. On the NMOS 6502 the N/V/Z
    /// flags in decimal mode follow the documented quirky behavior.
    pub(crate) fn adc(&mut self, v: u8) {
        let a = self.regs.a;
        let cin = self.regs.p.contains(Status::C) as u16;

        if self.regs.p.contains(Status::D) {
            // Z reflects the plain binary sum.
            let binsum = a as u16 + v as u16 + cin;
            self.regs.p.set(Status::Z, (binsum & 0xFF) == 0);

            let mut lo = (a & 0x0F) as u16 + (v & 0x0F) as u16 + cin;
            if lo >= 0x0A {
                lo = ((lo + 0x06) & 0x0F) + 0x10;
            }
            let mut sum = (a & 0xF0) as u16 + (v & 0xF0) as u16 + lo;

            // N and V are taken from this intermediate, before the high-nibble fix.
            self.regs.p.set(Status::N, sum & 0x80 != 0);
            self.regs.p
                .set(Status::V, (a as u16 ^ sum) & (v as u16 ^ sum) & 0x80 != 0);

            if sum >= 0xA0 {
                sum += 0x60;
            }
            self.regs.p.set(Status::C, sum >= 0x100);
            self.regs.a = (sum & 0xFF) as u8;
        } else {
            let sum = a as u16 + v as u16 + cin;
            let result = (sum & 0xFF) as u8;
            self.regs.p.set(Status::C, sum > 0xFF);
            self.regs.p
                .set(Status::V, (a as u16 ^ sum) & (v as u16 ^ sum) & 0x80 != 0);
            self.regs.a = result;
            self.regs.p.set_zn(result);
        }
    }

    /// Subtract with carry (borrow). On the NMOS 6502 the C/Z/V/N flags are
    /// identical to binary mode even when D is set; only the accumulator result
    /// differs.
    pub(crate) fn sbc(&mut self, v: u8) {
        let a = self.regs.a;
        let cin = self.regs.p.contains(Status::C) as i16;

        // Binary result drives all flags.
        let r = a as i16 - v as i16 - (1 - cin);
        let result = r as u8;
        self.regs.p.set(Status::C, r >= 0);
        self.regs.p.set(Status::V, (a ^ v) & (a ^ result) & 0x80 != 0);
        self.regs.p.set_zn(result);

        if self.regs.p.contains(Status::D) {
            let mut lo = (a & 0x0F) as i16 - (v & 0x0F) as i16 + cin - 1;
            if lo < 0 {
                lo = ((lo - 0x06) & 0x0F) - 0x10;
            }
            let mut hi = (a & 0xF0) as i16 - (v & 0xF0) as i16 + lo;
            if hi < 0 {
                hi -= 0x60;
            }
            self.regs.a = (hi & 0xFF) as u8;
        } else {
            self.regs.a = result;
        }
    }

    /// Read-modify-write helper: read the operand, transform it, write it back.
    fn rmw<B: Bus>(&mut self, bus: &mut B, m: &Operand, f: impl Fn(&mut Cpu, u8) -> u8) {
        let v = self.read(bus, m.addr);
        let r = f(self, v);
        self.write(bus, m.addr, r);
    }

    /// Shift/rotate on either the accumulator or memory, depending on `mode`.
    fn shift<B: Bus>(
        &mut self,
        bus: &mut B,
        mode: AddressingMode,
        m: &Operand,
        f: fn(&mut Cpu, u8) -> u8,
    ) {
        if mode == AddressingMode::Accumulator {
            self.regs.a = f(self, self.regs.a);
        } else {
            let v = self.read(bus, m.addr);
            let r = f(self, v);
            self.write(bus, m.addr, r);
        }
    }

    fn asl_val(&mut self, v: u8) -> u8 {
        self.regs.p.set(Status::C, v & 0x80 != 0);
        let r = v << 1;
        self.regs.p.set_zn(r);
        r
    }

    fn lsr_val(&mut self, v: u8) -> u8 {
        self.regs.p.set(Status::C, v & 0x01 != 0);
        let r = v >> 1;
        self.regs.p.set_zn(r);
        r
    }

    fn rol_val(&mut self, v: u8) -> u8 {
        let carry_in = self.regs.p.contains(Status::C) as u8;
        self.regs.p.set(Status::C, v & 0x80 != 0);
        let r = (v << 1) | carry_in;
        self.regs.p.set_zn(r);
        r
    }

    fn ror_val(&mut self, v: u8) -> u8 {
        let carry_in = self.regs.p.contains(Status::C) as u8;
        self.regs.p.set(Status::C, v & 0x01 != 0);
        let r = (v >> 1) | (carry_in << 7);
        self.regs.p.set_zn(r);
        r
    }

    /// Take a branch if `cond`: `+1` cycle for the branch, `+1` more if the
    /// target is on a different page. The target is precomputed in `m.addr`.
    fn branch(&mut self, m: &Operand, cond: bool) -> u8 {
        if !cond {
            return 0;
        }
        self.regs.pc = m.addr;
        if m.page_crossed { 2 } else { 1 }
    }

    /// JSR interleaves its operand fetch with the stack pushes: it fetches the
    /// target low byte, pushes the return address, and only *then* fetches the
    /// high byte. If the operand lives in the stack page, the push overwrites the
    /// high byte before it's read — a real hardware quirk the Tom Harte suite
    /// exercises.
    fn jsr<B: Bus>(&mut self, bus: &mut B) {
        let lo = self.fetch_byte(bus) as u16; // PC now points at the high byte
        let ret = self.regs.pc; // address of the last JSR byte (RTS adds 1)
        self.push(bus, (ret >> 8) as u8);
        self.push(bus, ret as u8);
        let hi = self.read(bus, self.regs.pc) as u16;
        self.regs.pc = (hi << 8) | lo;
    }

    fn rts<B: Bus>(&mut self, bus: &mut B) {
        let lo = self.pull(bus) as u16;
        let hi = self.pull(bus) as u16;
        self.regs.pc = (lo | (hi << 8)).wrapping_add(1);
    }

    fn brk<B: Bus>(&mut self, bus: &mut B) {
        // BRK has a padding byte: the pushed PC skips it.
        self.regs.pc = self.regs.pc.wrapping_add(1);
        self.service_interrupt(bus, IRQ_VECTOR, true);
    }

    fn rti<B: Bus>(&mut self, bus: &mut B) {
        self.pull_status(bus);
        let lo = self.pull(bus) as u16;
        let hi = self.pull(bus) as u16;
        self.regs.pc = lo | (hi << 8);
    }
}
