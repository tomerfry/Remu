//! Instruction dispatch and execution: one `match` over the opcode byte.
//!
//! Prefixes never reach [`Cpu::dispatch`] — the step loop consumes them and
//! records their effect (`seg_override`, `rep`). Every remaining byte executes
//! as *something*, as on real silicon: the 8086 has no invalid-opcode fault,
//! so the well-understood undocumented encodings (`POP CS`, the 60–6F jump
//! aliases, `SALC`, `SETMO`, the C0/C1/C8/C9 RET aliases, `TEST /1`) are
//! implemented rather than trapped.
//!
//! Returned cycle counts are documented 8086 base timings plus the
//! effective-address penalty; the prefetch queue is not modeled.

use super::modrm::Operand;
use super::registers::{reg, Flags};
use super::{Bus, Cpu};

/// The eight ALU operations selected by the `reg` field of group 80–83, in
/// encoding order. `CMP` (index 7) is handled by the caller not writing back.
macro_rules! alu_table {
    ($($f:ident),+) => { [$(Cpu::$f),+] };
}
const ALU8: [fn(&mut Cpu, u8, u8) -> u8; 8] =
    alu_table!(add8, or8, adc8, sbb8, and8, sub8, xor8, sub8);
const ALU16: [fn(&mut Cpu, u16, u16) -> u16; 8] =
    alu_table!(add16, or16, adc16, sbb16, and16, sub16, xor16, sub16);

impl Cpu {
    /// Execute the instruction whose (non-prefix) opcode byte is `opcode`.
    /// Returns the cycles consumed.
    pub(crate) fn dispatch<B: Bus>(&mut self, bus: &mut B, opcode: u8) -> u32 {
        match opcode {
            // --- ALU: ADD OR ADC SBB AND SUB XOR CMP, five forms each ------
            // The three low bits select the form; bits 5–3 select the op.
            0x00 | 0x08 | 0x10 | 0x18 | 0x20 | 0x28 | 0x30 | 0x38 => {
                let f = ALU8[(opcode >> 3) as usize];
                self.op_rm_r8(bus, f, opcode != 0x38)
            }
            0x01 | 0x09 | 0x11 | 0x19 | 0x21 | 0x29 | 0x31 | 0x39 => {
                let f = ALU16[(opcode >> 3) as usize];
                self.op_rm_r16(bus, f, opcode != 0x39)
            }
            0x02 | 0x0A | 0x12 | 0x1A | 0x22 | 0x2A | 0x32 | 0x3A => {
                let f = ALU8[(opcode >> 3) as usize];
                self.op_r_rm8(bus, f, opcode != 0x3A)
            }
            0x03 | 0x0B | 0x13 | 0x1B | 0x23 | 0x2B | 0x33 | 0x3B => {
                let f = ALU16[(opcode >> 3) as usize];
                self.op_r_rm16(bus, f, opcode != 0x3B)
            }
            0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 | 0x3C => {
                let f = ALU8[(opcode >> 3) as usize];
                let b = self.fetch_byte(bus);
                let r = f(self, self.regs.reg8(0), b);
                if opcode != 0x3C { self.regs.set_reg8(0, r); }
                4
            }
            0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D => {
                let f = ALU16[(opcode >> 3) as usize];
                let b = self.fetch_word(bus);
                let r = f(self, self.regs.ax, b);
                if opcode != 0x3D { self.regs.ax = r; }
                4
            }

            // Immediate group: 80/82 = r/m8,imm8; 81 = r/m16,imm16; 83 = r/m16,imm8 (sign-extended)
            0x80 | 0x82 => {
                let (m, op) = self.modrm(bus);
                let a = self.read_op8(bus, op);
                let b = self.fetch_byte(bus);
                let r = ALU8[m.reg() as usize](self, a, b);
                let wb = m.reg() != 7;
                if wb { self.write_op8(bus, op, r); }
                if op.is_mem() { (if wb { 17 } else { 10 }) + self.ea_cycles } else { 4 }
            }
            0x81 | 0x83 => {
                let (m, op) = self.modrm(bus);
                let a = self.read_op16(bus, op);
                let b = if opcode == 0x81 { self.fetch_word(bus) } else { self.fetch_byte(bus) as i8 as u16 };
                let r = ALU16[m.reg() as usize](self, a, b);
                let wb = m.reg() != 7;
                if wb { self.write_op16(bus, op, r); }
                if op.is_mem() { (if wb { 17 } else { 10 }) + self.ea_cycles } else { 4 }
            }

            // --- Stack: PUSH/POP -------------------------------------------
            0x06 | 0x0E | 0x16 | 0x1E => { let v = self.regs.seg(opcode >> 3); self.push16(bus, v); 10 }
            0x07 | 0x17 | 0x1F => {
                let v = self.pop16(bus);
                self.regs.set_seg(opcode >> 3, v);
                if opcode == 0x17 { self.inhibit_interrupts = true; } // POP SS
                8
            }
            0x0F => { self.regs.cs = self.pop16(bus); 8 } // undocumented 8086: POP CS
            0x50..=0x57 => {
                // 8086 quirk: PUSH SP pushes the already-decremented value.
                let r = opcode & 7;
                let v = if r == 4 { self.regs.sp.wrapping_sub(2) } else { self.regs.reg16(r) };
                self.push16(bus, v);
                11
            }
            0x58..=0x5F => { let v = self.pop16(bus); self.regs.set_reg16(opcode & 7, v); 8 }
            0x8F => {
                let (_, op) = self.modrm(bus);
                let v = self.pop16(bus);
                self.write_op16(bus, op, v);
                if op.is_mem() { 17 + self.ea_cycles } else { 8 }
            }
            0x9C => { let v = self.regs.flags.to_word(); self.push16(bus, v); 10 }
            0x9D => { let v = self.pop16(bus); self.regs.flags = Flags::from_word(v); 8 }

            // --- INC/DEC r16 -------------------------------------------------
            0x40..=0x47 => { let v = self.regs.reg16(opcode & 7); let r = self.inc16(v); self.regs.set_reg16(opcode & 7, r); 2 }
            0x48..=0x4F => { let v = self.regs.reg16(opcode & 7); let r = self.dec16(v); self.regs.set_reg16(opcode & 7, r); 2 }

            // --- Conditional jumps (60–6F are undocumented aliases) ----------
            0x60..=0x7F => {
                let rel = self.fetch_byte(bus) as i8 as u16;
                if self.cond(opcode & 0xF) { self.regs.ip = self.regs.ip.wrapping_add(rel); 16 } else { 4 }
            }

            // --- TEST / XCHG -------------------------------------------------
            0x84 => self.op_rm_r8(bus, Cpu::and8, false),
            0x85 => self.op_rm_r16(bus, Cpu::and16, false),
            0x86 => {
                let (m, op) = self.modrm(bus);
                let a = self.read_op8(bus, op);
                let b = self.regs.reg8(m.reg());
                self.write_op8(bus, op, b);
                self.regs.set_reg8(m.reg(), a);
                if op.is_mem() { 17 + self.ea_cycles } else { 4 }
            }
            0x87 => {
                let (m, op) = self.modrm(bus);
                let a = self.read_op16(bus, op);
                let b = self.regs.reg16(m.reg());
                self.write_op16(bus, op, b);
                self.regs.set_reg16(m.reg(), a);
                if op.is_mem() { 17 + self.ea_cycles } else { 4 }
            }
            0x90..=0x97 => { // XCHG AX, r16 (90 = NOP)
                let r = opcode & 7;
                let t = self.regs.ax;
                self.regs.ax = self.regs.reg16(r);
                self.regs.set_reg16(r, t);
                3
            }
            0xA8 => { let b = self.fetch_byte(bus); let a = self.regs.reg8(0); self.and8(a, b); 4 }
            0xA9 => { let b = self.fetch_word(bus); let a = self.regs.ax; self.and16(a, b); 4 }

            // --- MOV ---------------------------------------------------------
            0x88 => { let (m, op) = self.modrm(bus); let v = self.regs.reg8(m.reg()); self.write_op8(bus, op, v); if op.is_mem() { 9 + self.ea_cycles } else { 2 } }
            0x89 => { let (m, op) = self.modrm(bus); let v = self.regs.reg16(m.reg()); self.write_op16(bus, op, v); if op.is_mem() { 9 + self.ea_cycles } else { 2 } }
            0x8A => { let (m, op) = self.modrm(bus); let v = self.read_op8(bus, op); self.regs.set_reg8(m.reg(), v); if op.is_mem() { 8 + self.ea_cycles } else { 2 } }
            0x8B => { let (m, op) = self.modrm(bus); let v = self.read_op16(bus, op); self.regs.set_reg16(m.reg(), v); if op.is_mem() { 8 + self.ea_cycles } else { 2 } }
            0x8C => { let (m, op) = self.modrm(bus); let v = self.regs.seg(m.reg()); self.write_op16(bus, op, v); if op.is_mem() { 9 + self.ea_cycles } else { 2 } }
            0x8E => {
                let (m, op) = self.modrm(bus);
                let v = self.read_op16(bus, op);
                self.regs.set_seg(m.reg(), v);
                if m.reg() & 3 == reg::SS { self.inhibit_interrupts = true; }
                if op.is_mem() { 8 + self.ea_cycles } else { 2 }
            }
            0x8D => { // LEA (register form is undefined; leave the register alone)
                let (m, op) = self.modrm(bus);
                if let Operand::Mem { off, .. } = op { self.regs.set_reg16(m.reg(), off); }
                2 + self.ea_cycles
            }
            0xA0 => { let off = self.fetch_word(bus); let seg = self.seg_or(reg::DS); let v = self.read8(bus, seg, off); self.regs.set_reg8(0, v); 10 }
            0xA1 => { let off = self.fetch_word(bus); let seg = self.seg_or(reg::DS); self.regs.ax = self.read16(bus, seg, off); 10 }
            0xA2 => { let off = self.fetch_word(bus); let seg = self.seg_or(reg::DS); let v = self.regs.reg8(0); self.write8(bus, seg, off, v); 10 }
            0xA3 => { let off = self.fetch_word(bus); let seg = self.seg_or(reg::DS); let v = self.regs.ax; self.write16(bus, seg, off, v); 10 }
            0xB0..=0xB7 => { let v = self.fetch_byte(bus); self.regs.set_reg8(opcode & 7, v); 4 }
            0xB8..=0xBF => { let v = self.fetch_word(bus); self.regs.set_reg16(opcode & 7, v); 4 }
            0xC6 => { let (_, op) = self.modrm(bus); let v = self.fetch_byte(bus); self.write_op8(bus, op, v); if op.is_mem() { 10 + self.ea_cycles } else { 4 } }
            0xC7 => { let (_, op) = self.modrm(bus); let v = self.fetch_word(bus); self.write_op16(bus, op, v); if op.is_mem() { 10 + self.ea_cycles } else { 4 } }

            // --- Wide loads: LES / LDS ---------------------------------------
            0xC4 | 0xC5 => {
                let (m, op) = self.modrm(bus);
                if let Operand::Mem { seg, off } = op {
                    let v = self.read16(bus, seg, off);
                    let s = self.read16(bus, seg, off.wrapping_add(2));
                    self.regs.set_reg16(m.reg(), v);
                    if opcode == 0xC4 { self.regs.es = s; } else { self.regs.ds = s; }
                }
                16 + self.ea_cycles
            }

            // --- Conversions / flags transfers -------------------------------
            0x98 => { self.regs.ax = self.regs.ax as u8 as i8 as i16 as u16; 2 }
            0x99 => { self.regs.dx = if self.regs.ax & 0x8000 != 0 { 0xFFFF } else { 0 }; 5 }
            0x9E => { // SAHF
                let ah = (self.regs.ax >> 8) as u16;
                let keep = self.regs.flags.bits() & 0xFF00;
                self.regs.flags = Flags::from_bits_truncate(keep | ah);
                4
            }
            0x9F => { // LAHF: bit 1 reads as 1, bits 3/5 as 0
                let lo = (self.regs.flags.bits() as u8 & 0xD5) | 0x02;
                self.regs.ax = self.regs.ax & 0x00FF | (lo as u16) << 8;
                4
            }

            // --- BCD adjustments ----------------------------------------------
            0x27 => { self.daa(); 4 }
            0x2F => { self.das(); 4 }
            0x37 => { self.aaa(); 8 }
            0x3F => { self.aas(); 8 }
            0xD4 => { let base = self.fetch_byte(bus); if !self.aam(base) { self.interrupt(bus, 0); } 83 }
            0xD5 => { let base = self.fetch_byte(bus); self.aad(base); 60 }

            // --- String operations --------------------------------------------
            0xA4 => self.rep_str(bus, Cpu::movs8, false, 18, 17),
            0xA5 => self.rep_str(bus, Cpu::movs16, false, 18, 17),
            0xA6 => self.rep_str(bus, Cpu::cmps8, true, 22, 22),
            0xA7 => self.rep_str(bus, Cpu::cmps16, true, 22, 22),
            0xAA => self.rep_str(bus, Cpu::stos8, false, 11, 10),
            0xAB => self.rep_str(bus, Cpu::stos16, false, 11, 10),
            0xAC => self.rep_str(bus, Cpu::lods8, false, 12, 13),
            0xAD => self.rep_str(bus, Cpu::lods16, false, 12, 13),
            0xAE => self.rep_str(bus, Cpu::scas8, true, 15, 15),
            0xAF => self.rep_str(bus, Cpu::scas16, true, 15, 15),

            // --- Shift/rotate groups ------------------------------------------
            0xD0 => self.shift_grp8(bus, None),
            0xD1 => self.shift_grp16(bus, None),
            0xD2 => { let c = self.regs.cx as u8 as u32; self.shift_grp8(bus, Some(c)) }
            0xD3 => { let c = self.regs.cx as u8 as u32; self.shift_grp16(bus, Some(c)) }

            // --- RET / RETF (C0/C1/C8/C9 are undocumented aliases) ------------
            0xC3 | 0xC1 => { self.regs.ip = self.pop16(bus); 16 }
            0xC2 | 0xC0 => {
                let n = self.fetch_word(bus);
                self.regs.ip = self.pop16(bus);
                self.regs.sp = self.regs.sp.wrapping_add(n);
                20
            }
            0xCB | 0xC9 => { self.regs.ip = self.pop16(bus); self.regs.cs = self.pop16(bus); 26 }
            0xCA | 0xC8 => {
                let n = self.fetch_word(bus);
                self.regs.ip = self.pop16(bus);
                self.regs.cs = self.pop16(bus);
                self.regs.sp = self.regs.sp.wrapping_add(n);
                25
            }

            // --- Software interrupts -------------------------------------------
            0xCC => { self.interrupt(bus, 3); 52 }
            0xCD => { let v = self.fetch_byte(bus); self.interrupt(bus, v); 51 }
            0xCE => { if self.regs.flags.contains(Flags::OF) { self.interrupt(bus, 4); 53 } else { 4 } }
            0xCF => { // IRET
                self.regs.ip = self.pop16(bus);
                self.regs.cs = self.pop16(bus);
                let f = self.pop16(bus);
                self.regs.flags = Flags::from_word(f);
                24
            }

            // --- SALC / XLAT / ESC ----------------------------------------------
            0xD6 => { // undocumented SALC: AL = CF ? FF : 00
                let v = if self.regs.flags.contains(Flags::CF) { 0xFF } else { 0 };
                self.regs.set_reg8(0, v);
                3
            }
            0xD7 => {
                let seg = self.seg_or(reg::DS);
                let off = self.regs.bx.wrapping_add(self.regs.ax & 0xFF);
                let v = self.read8(bus, seg, off);
                self.regs.set_reg8(0, v);
                11
            }
            0xD8..=0xDF => { // ESC: 8087 operand fetch; no coprocessor attached
                let (_, op) = self.modrm(bus);
                if let Operand::Mem { seg, off } = op { let _ = self.read8(bus, seg, off); }
                if op.is_mem() { 8 + self.ea_cycles } else { 2 }
            }

            // --- Loops / IN / OUT -------------------------------------------------
            0xE0 => { // LOOPNZ
                let rel = self.fetch_byte(bus) as i8 as u16;
                self.regs.cx = self.regs.cx.wrapping_sub(1);
                if self.regs.cx != 0 && !self.regs.flags.contains(Flags::ZF) { self.regs.ip = self.regs.ip.wrapping_add(rel); 19 } else { 5 }
            }
            0xE1 => { // LOOPZ
                let rel = self.fetch_byte(bus) as i8 as u16;
                self.regs.cx = self.regs.cx.wrapping_sub(1);
                if self.regs.cx != 0 && self.regs.flags.contains(Flags::ZF) { self.regs.ip = self.regs.ip.wrapping_add(rel); 18 } else { 6 }
            }
            0xE2 => { // LOOP
                let rel = self.fetch_byte(bus) as i8 as u16;
                self.regs.cx = self.regs.cx.wrapping_sub(1);
                if self.regs.cx != 0 { self.regs.ip = self.regs.ip.wrapping_add(rel); 17 } else { 5 }
            }
            0xE3 => { // JCXZ
                let rel = self.fetch_byte(bus) as i8 as u16;
                if self.regs.cx == 0 { self.regs.ip = self.regs.ip.wrapping_add(rel); 18 } else { 6 }
            }
            0xE4 => { let p = self.fetch_byte(bus) as u16; let v = bus.io_read(p); self.regs.set_reg8(0, v); 10 }
            0xE5 => { let p = self.fetch_byte(bus) as u16; self.regs.ax = self.io_read16(bus, p); 10 }
            0xE6 => { let p = self.fetch_byte(bus) as u16; bus.io_write(p, self.regs.reg8(0)); 10 }
            0xE7 => { let p = self.fetch_byte(bus) as u16; let v = self.regs.ax; self.io_write16(bus, p, v); 10 }
            0xEC => { let v = bus.io_read(self.regs.dx); self.regs.set_reg8(0, v); 8 }
            0xED => { let p = self.regs.dx; self.regs.ax = self.io_read16(bus, p); 8 }
            0xEE => { bus.io_write(self.regs.dx, self.regs.reg8(0)); 8 }
            0xEF => { let (p, v) = (self.regs.dx, self.regs.ax); self.io_write16(bus, p, v); 8 }

            // --- CALL / JMP -------------------------------------------------------
            0x9A => { // CALL far ptr16:16
                let off = self.fetch_word(bus);
                let seg = self.fetch_word(bus);
                let (cs, ip) = (self.regs.cs, self.regs.ip);
                self.push16(bus, cs);
                self.push16(bus, ip);
                self.regs.cs = seg;
                self.regs.ip = off;
                28
            }
            0xE8 => { // CALL rel16
                let rel = self.fetch_word(bus);
                let ip = self.regs.ip;
                self.push16(bus, ip);
                self.regs.ip = ip.wrapping_add(rel);
                19
            }
            0xE9 => { let rel = self.fetch_word(bus); self.regs.ip = self.regs.ip.wrapping_add(rel); 15 }
            0xEA => { let off = self.fetch_word(bus); let seg = self.fetch_word(bus); self.regs.ip = off; self.regs.cs = seg; 15 }
            0xEB => { let rel = self.fetch_byte(bus) as i8 as u16; self.regs.ip = self.regs.ip.wrapping_add(rel); 15 }

            // --- Processor control ---------------------------------------------
            0x9B => 3, // WAIT: no 8087, TEST# assumed asserted
            0xF4 => { self.halted = true; 2 }
            0xF5 => { self.regs.flags.toggle(Flags::CF); 2 }
            0xF8 => { self.regs.flags.remove(Flags::CF); 2 }
            0xF9 => { self.regs.flags.insert(Flags::CF); 2 }
            0xFA => { self.regs.flags.remove(Flags::IF); 2 }
            0xFB => { self.regs.flags.insert(Flags::IF); self.inhibit_interrupts = true; 2 }
            0xFC => { self.regs.flags.remove(Flags::DF); 2 }
            0xFD => { self.regs.flags.insert(Flags::DF); 2 }

            // --- Group F6/F7: TEST NOT NEG MUL IMUL DIV IDIV ---------------------
            0xF6 => {
                let (m, op) = self.modrm(bus);
                let v = self.read_op8(bus, op);
                let mem = op.is_mem();
                let ea = self.ea_cycles;
                match m.reg() {
                    0 | 1 => { let b = self.fetch_byte(bus); self.and8(v, b); if mem { 11 + ea } else { 5 } }
                    2 => { self.write_op8(bus, op, !v); if mem { 16 + ea } else { 3 } }
                    3 => { let r = self.neg8(v); self.write_op8(bus, op, r); if mem { 16 + ea } else { 3 } }
                    4 => { self.mul8(v); 70 + if mem { 6 + ea } else { 0 } }
                    5 => { self.imul8(v); 90 + if mem { 6 + ea } else { 0 } }
                    6 => { if !self.div8(v, false) { self.interrupt(bus, 0); } 85 + if mem { 6 + ea } else { 0 } }
                    _ => { if !self.div8(v, true) { self.interrupt(bus, 0); } 110 + if mem { 6 + ea } else { 0 } }
                }
            }
            0xF7 => {
                let (m, op) = self.modrm(bus);
                let v = self.read_op16(bus, op);
                let mem = op.is_mem();
                let ea = self.ea_cycles;
                match m.reg() {
                    0 | 1 => { let b = self.fetch_word(bus); self.and16(v, b); if mem { 11 + ea } else { 5 } }
                    2 => { self.write_op16(bus, op, !v); if mem { 16 + ea } else { 3 } }
                    3 => { let r = self.neg16(v); self.write_op16(bus, op, r); if mem { 16 + ea } else { 3 } }
                    4 => { self.mul16(v); 118 + if mem { 6 + ea } else { 0 } }
                    5 => { self.imul16(v); 144 + if mem { 6 + ea } else { 0 } }
                    6 => { if !self.div16(v, false) { self.interrupt(bus, 0); } 155 + if mem { 6 + ea } else { 0 } }
                    _ => { if !self.div16(v, true) { self.interrupt(bus, 0); } 176 + if mem { 6 + ea } else { 0 } }
                }
            }

            // --- Group FE: INC/DEC r/m8 ------------------------------------------
            // The /2../7 forms are undocumented byte-sized control transfers;
            // they decode here as INC/DEC on the low bit like the hardware's
            // partial decode of bit 0, which is close enough for /0 and /1.
            0xFE => {
                let (m, op) = self.modrm(bus);
                let v = self.read_op8(bus, op);
                let r = if m.reg() & 1 == 0 { self.inc8(v) } else { self.dec8(v) };
                self.write_op8(bus, op, r);
                if op.is_mem() { 15 + self.ea_cycles } else { 3 }
            }

            // --- Group FF: INC DEC CALL CALL-far JMP JMP-far PUSH ------------------
            0xFF => {
                let (m, op) = self.modrm(bus);
                let mem = op.is_mem();
                let ea = self.ea_cycles;
                match m.reg() {
                    0 => { let v = self.read_op16(bus, op); let r = self.inc16(v); self.write_op16(bus, op, r); if mem { 15 + ea } else { 3 } }
                    1 => { let v = self.read_op16(bus, op); let r = self.dec16(v); self.write_op16(bus, op, r); if mem { 15 + ea } else { 3 } }
                    2 => { // CALL near indirect
                        let v = self.read_op16(bus, op);
                        let ip = self.regs.ip;
                        self.push16(bus, ip);
                        self.regs.ip = v;
                        if mem { 21 + ea } else { 16 }
                    }
                    3 => { // CALL far indirect (register form is undefined; keeps CS)
                        let (off, seg) = self.far_operand(bus, op);
                        let (cs, ip) = (self.regs.cs, self.regs.ip);
                        self.push16(bus, cs);
                        self.push16(bus, ip);
                        self.regs.cs = seg;
                        self.regs.ip = off;
                        37 + ea
                    }
                    4 => { self.regs.ip = self.read_op16(bus, op); if mem { 18 + ea } else { 11 } }
                    5 => { // JMP far indirect
                        let (off, seg) = self.far_operand(bus, op);
                        self.regs.cs = seg;
                        self.regs.ip = off;
                        24 + ea
                    }
                    _ => { // PUSH r/m16 (/7 is an undocumented alias of /6)
                        let v = match op {
                            Operand::Reg(4) => self.regs.sp.wrapping_sub(2), // PUSH SP quirk
                            _ => self.read_op16(bus, op),
                        };
                        self.push16(bus, v);
                        if mem { 16 + ea } else { 11 }
                    }
                }
            }

            // Prefixes (26/2E/36/3E, F0/F1/F2/F3) are consumed by the step loop.
            _ => unreachable!("prefix byte {opcode:#04X} reached dispatch"),
        }
    }

    // --- Dispatch helpers ----------------------------------------------------

    /// Evaluate condition code `n` (the low nibble of a Jcc opcode).
    fn cond(&self, n: u8) -> bool {
        let f = self.regs.flags;
        let r = match n >> 1 {
            0 => f.contains(Flags::OF),
            1 => f.contains(Flags::CF),
            2 => f.contains(Flags::ZF),
            3 => f.contains(Flags::CF) || f.contains(Flags::ZF),
            4 => f.contains(Flags::SF),
            5 => f.contains(Flags::PF),
            6 => f.contains(Flags::SF) != f.contains(Flags::OF),
            _ => f.contains(Flags::ZF) || (f.contains(Flags::SF) != f.contains(Flags::OF)),
        };
        r != (n & 1 != 0)
    }

    /// `op r/m8, r8` — destination is the r/m operand. `wb == false` for CMP/TEST.
    #[inline(always)]
    fn op_rm_r8<B: Bus>(&mut self, bus: &mut B, f: fn(&mut Cpu, u8, u8) -> u8, wb: bool) -> u32 {
        let (m, op) = self.modrm(bus);
        let a = self.read_op8(bus, op);
        let b = self.regs.reg8(m.reg());
        let r = f(self, a, b);
        if wb { self.write_op8(bus, op, r); }
        if op.is_mem() { (if wb { 16 } else { 9 }) + self.ea_cycles } else { 3 }
    }

    #[inline(always)]
    fn op_rm_r16<B: Bus>(&mut self, bus: &mut B, f: fn(&mut Cpu, u16, u16) -> u16, wb: bool) -> u32 {
        let (m, op) = self.modrm(bus);
        let a = self.read_op16(bus, op);
        let b = self.regs.reg16(m.reg());
        let r = f(self, a, b);
        if wb { self.write_op16(bus, op, r); }
        if op.is_mem() { (if wb { 16 } else { 9 }) + self.ea_cycles } else { 3 }
    }

    /// `op r8, r/m8` — destination is the register operand.
    #[inline(always)]
    fn op_r_rm8<B: Bus>(&mut self, bus: &mut B, f: fn(&mut Cpu, u8, u8) -> u8, wb: bool) -> u32 {
        let (m, op) = self.modrm(bus);
        let a = self.regs.reg8(m.reg());
        let b = self.read_op8(bus, op);
        let r = f(self, a, b);
        if wb { self.regs.set_reg8(m.reg(), r); }
        if op.is_mem() { 9 + self.ea_cycles } else { 3 }
    }

    #[inline(always)]
    fn op_r_rm16<B: Bus>(&mut self, bus: &mut B, f: fn(&mut Cpu, u16, u16) -> u16, wb: bool) -> u32 {
        let (m, op) = self.modrm(bus);
        let a = self.regs.reg16(m.reg());
        let b = self.read_op16(bus, op);
        let r = f(self, a, b);
        if wb { self.regs.set_reg16(m.reg(), r); }
        if op.is_mem() { 9 + self.ea_cycles } else { 3 }
    }

    /// Load an `offset:segment` pair for far indirect CALL/JMP. The register
    /// form is architecturally undefined; we use the register as the offset
    /// and keep the current CS.
    fn far_operand<B: Bus>(&mut self, bus: &mut B, op: Operand) -> (u16, u16) {
        match op {
            Operand::Mem { seg, off } => {
                (self.read16(bus, seg, off), self.read16(bus, seg, off.wrapping_add(2)))
            }
            Operand::Reg(i) => (self.regs.reg16(i), self.regs.cs),
        }
    }

    /// Shift/rotate group on a byte operand. `count` is `None` for the 1-bit
    /// forms (D0), `Some(CL)` for the CL forms (D2).
    fn shift_grp8<B: Bus>(&mut self, bus: &mut B, count: Option<u32>) -> u32 {
        let (m, op) = self.modrm(bus);
        let n = count.unwrap_or(1);
        let v = self.read_op8(bus, op);
        let r = match m.reg() {
            0 => self.rol8(v, n),
            1 => self.ror8(v, n),
            2 => self.rcl8(v, n),
            3 => self.rcr8(v, n),
            4 => self.shl8(v, n),
            5 => self.shr8(v, n),
            6 => self.setmo8(v, n), // undocumented SETMO/SETMOC
            _ => self.sar8(v, n),
        };
        self.write_op8(bus, op, r);
        self.shift_cycles(op.is_mem(), count)
    }

    fn shift_grp16<B: Bus>(&mut self, bus: &mut B, count: Option<u32>) -> u32 {
        let (m, op) = self.modrm(bus);
        let n = count.unwrap_or(1);
        let v = self.read_op16(bus, op);
        let r = match m.reg() {
            0 => self.rol16(v, n),
            1 => self.ror16(v, n),
            2 => self.rcl16(v, n),
            3 => self.rcr16(v, n),
            4 => self.shl16(v, n),
            5 => self.shr16(v, n),
            6 => self.setmo16(v, n),
            _ => self.sar16(v, n),
        };
        self.write_op16(bus, op, r);
        self.shift_cycles(op.is_mem(), count)
    }

    fn shift_cycles(&self, mem: bool, count: Option<u32>) -> u32 {
        match (count, mem) {
            (None, false) => 2,
            (None, true) => 15 + self.ea_cycles,
            (Some(n), false) => 8 + 4 * n,
            (Some(n), true) => 20 + self.ea_cycles + 4 * n,
        }
    }

    /// Undocumented shift-group `/6` (SETMO): sets the operand to all-ones;
    /// with a CL count of zero it is a no-op. Result flags are undefined on
    /// hardware (and masked by the test suite); we set them like a logic op.
    fn setmo8(&mut self, v: u8, count: u32) -> u8 {
        if count == 0 { return v; }
        self.or8(0xFF, 0xFF)
    }

    fn setmo16(&mut self, v: u16, count: u32) -> u16 {
        if count == 0 { return v; }
        self.or16(0xFFFF, 0xFFFF)
    }

    // --- Port I/O (16-bit access composed from the byte-wide bus) -------------

    fn io_read16<B: Bus>(&mut self, bus: &mut B, port: u16) -> u16 {
        let lo = bus.io_read(port) as u16;
        let hi = bus.io_read(port.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    fn io_write16<B: Bus>(&mut self, bus: &mut B, port: u16, v: u16) {
        bus.io_write(port, v as u8);
        bus.io_write(port.wrapping_add(1), (v >> 8) as u8);
    }

    // --- String primitives (one iteration each; see `rep_str`) ----------------

    /// ±element-size depending on DF.
    #[inline]
    fn delta(&self, n: u16) -> u16 {
        if self.regs.flags.contains(Flags::DF) { n.wrapping_neg() } else { n }
    }

    fn movs8<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(1);
        let seg = self.seg_or(reg::DS);
        let v = self.read8(bus, seg, self.regs.si);
        let (es, di) = (self.regs.es, self.regs.di);
        self.write8(bus, es, di, v);
        self.regs.si = self.regs.si.wrapping_add(d);
        self.regs.di = self.regs.di.wrapping_add(d);
    }

    fn movs16<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(2);
        let seg = self.seg_or(reg::DS);
        let v = self.read16(bus, seg, self.regs.si);
        let (es, di) = (self.regs.es, self.regs.di);
        self.write16(bus, es, di, v);
        self.regs.si = self.regs.si.wrapping_add(d);
        self.regs.di = self.regs.di.wrapping_add(d);
    }

    fn cmps8<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(1);
        let seg = self.seg_or(reg::DS);
        let a = self.read8(bus, seg, self.regs.si);
        let (es, di) = (self.regs.es, self.regs.di);
        let b = self.read8(bus, es, di);
        self.sub8(a, b);
        self.regs.si = self.regs.si.wrapping_add(d);
        self.regs.di = self.regs.di.wrapping_add(d);
    }

    fn cmps16<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(2);
        let seg = self.seg_or(reg::DS);
        let a = self.read16(bus, seg, self.regs.si);
        let (es, di) = (self.regs.es, self.regs.di);
        let b = self.read16(bus, es, di);
        self.sub16(a, b);
        self.regs.si = self.regs.si.wrapping_add(d);
        self.regs.di = self.regs.di.wrapping_add(d);
    }

    fn stos8<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(1);
        let (es, di, v) = (self.regs.es, self.regs.di, self.regs.reg8(0));
        self.write8(bus, es, di, v);
        self.regs.di = self.regs.di.wrapping_add(d);
    }

    fn stos16<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(2);
        let (es, di, v) = (self.regs.es, self.regs.di, self.regs.ax);
        self.write16(bus, es, di, v);
        self.regs.di = self.regs.di.wrapping_add(d);
    }

    fn lods8<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(1);
        let seg = self.seg_or(reg::DS);
        let v = self.read8(bus, seg, self.regs.si);
        self.regs.set_reg8(0, v);
        self.regs.si = self.regs.si.wrapping_add(d);
    }

    fn lods16<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(2);
        let seg = self.seg_or(reg::DS);
        self.regs.ax = self.read16(bus, seg, self.regs.si);
        self.regs.si = self.regs.si.wrapping_add(d);
    }

    fn scas8<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(1);
        let (es, di) = (self.regs.es, self.regs.di);
        let b = self.read8(bus, es, di);
        let a = self.regs.reg8(0);
        self.sub8(a, b);
        self.regs.di = self.regs.di.wrapping_add(d);
    }

    fn scas16<B: Bus>(&mut self, bus: &mut B) {
        let d = self.delta(2);
        let (es, di) = (self.regs.es, self.regs.di);
        let b = self.read16(bus, es, di);
        let a = self.regs.ax;
        self.sub16(a, b);
        self.regs.di = self.regs.di.wrapping_add(d);
    }

    /// Run a string primitive, honoring an active REP/REPE/REPNE prefix.
    /// `cmp` marks CMPS/SCAS, whose repetition also terminates on the ZF
    /// condition. Interrupts are not serviced mid-REP in this tier-2 model
    /// (the longest possible run is 64 Ki iterations).
    fn rep_str<B: Bus>(&mut self, bus: &mut B, one: fn(&mut Cpu, &mut B), cmp: bool, base: u32, per: u32) -> u32 {
        let Some(cont_on_zf) = self.rep else {
            one(self, bus);
            return base;
        };
        let mut cycles = 9;
        while self.regs.cx != 0 {
            one(self, bus);
            self.regs.cx = self.regs.cx.wrapping_sub(1);
            cycles += per;
            if cmp && self.regs.flags.contains(Flags::ZF) != cont_on_zf { break; }
        }
        cycles
    }
}
