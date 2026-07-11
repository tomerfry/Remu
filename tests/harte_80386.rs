//! SingleStepTests harness for the 80386 core (the `80386` v1 suite,
//! captured from a real 386EX by Daniel Balsom).
//!
//! Each gzipped `MOO` file is named after an opcode (`01.MOO.gz`), a
//! group-opcode/reg pair (`80.4.MOO.gz`), or a prefixed variant
//! (`6601.MOO.gz`) and contains up to 2,500 cases with an initial CPU+RAM
//! state and the expected *changes*. We set the CPU to `initial`, step until
//! the terminating `HLT`, and assert the final registers and RAM.
//!
//! `MOO` is a simple chunked binary format (see
//! <https://github.com/dbalsom/moo>); the small parser below reads only the
//! chunks the harness needs and skips the rest (cycles, prefetch queue, EA
//! info). Undefined register/flag bits are masked via the suite's `RM32`
//! chunks (set bit = defined, compare). Cycle counts are NOT validated: the
//! suite captures 386EX bus cycles including prefetch, which this core does
//! not model.
//!
//! The data is large and is **not** vendored. Point the test at a checkout:
//!
//! ```text
//! git clone --depth 1 https://github.com/SingleStepTests/80386.git
//! REMU_HARTE_80386_DIR=/path/to/80386/v1_ex_real_mode cargo test --release --test harte_80386
//! ```
//!
//! When the variable is unset the test prints a notice and passes, so CI
//! without the data stays green.

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};

use remu::x86_32::{Bus, Cpu, EFlags, LinearMemory, SegReg, reg};

/// The ArduinoX86 386EX rig: 16 MiB RAM plus the 386EX's always-visible
/// configuration ports (22h/23h), which read as 7Fh/42h; all other
/// unmapped ports read FFh.
struct RigBus {
    mem: LinearMemory,
}

impl Bus for RigBus {
    fn read(&mut self, addr: u32) -> u8 {
        self.mem.read(addr)
    }
    fn write(&mut self, addr: u32, value: u8) {
        self.mem.write(addr, value);
    }
    fn read16(&mut self, addr: u32) -> u16 {
        self.mem.read16(addr)
    }
    fn read32(&mut self, addr: u32) -> u32 {
        self.mem.read32(addr)
    }
    fn write16(&mut self, addr: u32, value: u16) {
        self.mem.write16(addr, value);
    }
    fn write32(&mut self, addr: u32, value: u32) {
        self.mem.write32(addr, value);
    }
    fn io_read(&mut self, port: u16) -> u8 {
        match port {
            0x22 => 0x7F,
            0x23 => 0x42,
            _ => 0xFF,
        }
    }
}

/// Register order of the `RG32`/`RM32` chunks.
const NREGS: usize = 20;

/// EFLAGS bits that are architecturally comparable on the 386 (the SMM dump
/// sets bits 31:18, which have no architectural meaning).
const EFLAGS_CMP: u32 = 0x0003_FFFF;

#[derive(Default, Clone)]
struct State {
    regs: [u32; NREGS],
    present: u32,
    ram: Vec<(u32, u8)>,
}

#[derive(Default, Clone)]
struct MooTest {
    name: String,
    bytes: Vec<u8>,
    init: State,
    fina: State,
    /// Per-register defined-bit masks for this test (top-level & per-test).
    mask: [u32; NREGS],
    /// Exception vector and address of the FLAGS image pushed for it.
    exception: Option<(u8, u32)>,
    hash: [u8; 20],
}

/// Minimal MOO chunk cursor.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8], pos: usize, end: usize) -> Self {
        Cursor { data, pos, end }
    }

    fn next_chunk(&mut self) -> Option<(&'a [u8; 4], &'a [u8])> {
        if self.pos + 8 > self.end {
            return None;
        }
        let tag: &[u8; 4] = self.data[self.pos..self.pos + 4].try_into().unwrap();
        let len =
            u32::from_le_bytes(self.data[self.pos + 4..self.pos + 8].try_into().unwrap()) as usize;
        let body = &self.data[self.pos + 8..(self.pos + 8 + len).min(self.end)];
        self.pos += 8 + len;
        Some((tag, body))
    }
}

fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

/// Parse an `RG32`/`RM32` payload: bitmask + one dword per set bit.
fn parse_rg32(body: &[u8]) -> ([u32; NREGS], u32) {
    let present = u32le(body, 0);
    let mut regs = [0u32; NREGS];
    let mut off = 4;
    for (i, r) in regs.iter_mut().enumerate() {
        if present & (1 << i) != 0 {
            *r = u32le(body, off);
            off += 4;
        }
    }
    (regs, present)
}

/// Parse a `RAM ` payload: count + 5-byte (addr, value) entries.
fn parse_ram(body: &[u8]) -> Vec<(u32, u8)> {
    let n = u32le(body, 0) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 4 + i * 5;
        out.push((u32le(body, off), body[off + 4]));
    }
    out
}

fn parse_state(body: &[u8]) -> (State, [u32; NREGS], u32) {
    let mut st = State::default();
    let mut mask = [u32::MAX; NREGS];
    let mut mask_present = 0u32;
    let mut c = Cursor::new(body, 0, body.len());
    while let Some((tag, chunk)) = c.next_chunk() {
        match tag {
            b"RG32" => (st.regs, st.present) = parse_rg32(chunk),
            b"RM32" => {
                let (m, p) = parse_rg32(chunk);
                for i in 0..NREGS {
                    if p & (1 << i) != 0 {
                        mask[i] = m[i];
                    }
                }
                mask_present = p;
            }
            b"RAM " => st.ram = parse_ram(chunk),
            _ => (), // QUEU, EA32, ...
        }
    }
    (st, mask, mask_present)
}

/// Parse one MOO file into tests plus the file-wide register masks.
fn parse_moo(data: &[u8]) -> (Vec<MooTest>, [u32; NREGS]) {
    let mut file_mask = [u32::MAX; NREGS];
    let mut tests = Vec::new();
    let mut c = Cursor::new(data, 0, data.len());
    while let Some((tag, body)) = c.next_chunk() {
        match tag {
            b"RM32" => {
                let (m, p) = parse_rg32(body);
                for (i, fm) in file_mask.iter_mut().enumerate() {
                    if p & (1 << i) != 0 {
                        *fm = m[i];
                    }
                }
            }
            b"TEST" => {
                let mut t = MooTest {
                    mask: file_mask,
                    ..Default::default()
                };
                // Skip the 4-byte test index, then walk subchunks.
                let mut tc = Cursor::new(body, 4, body.len());
                while let Some((tag2, chunk)) = tc.next_chunk() {
                    match tag2 {
                        b"NAME" => {
                            let n = u32le(chunk, 0) as usize;
                            t.name = String::from_utf8_lossy(&chunk[4..4 + n]).into_owned();
                        }
                        b"BYTS" => {
                            let n = u32le(chunk, 0) as usize;
                            t.bytes = chunk[4..4 + n].to_vec();
                        }
                        b"INIT" => (t.init, _, _) = parse_state(chunk),
                        b"FINA" => {
                            let (st, mask, p) = parse_state(chunk);
                            t.fina = st;
                            for (i, tm) in t.mask.iter_mut().enumerate() {
                                if p & (1 << i) != 0 {
                                    *tm &= mask[i];
                                }
                            }
                        }
                        b"EXCP" => t.exception = Some((chunk[0], u32le(chunk, 1))),
                        b"HASH" => t.hash = chunk[..20].try_into().unwrap(),
                        _ => (), // BYTS, CYCL, GMET, ...
                    }
                }
                tests.push(t);
            }
            _ => (), // MOO header, META
        }
    }
    (tests, file_mask)
}

/// RG32 register index → CPU state, as a value for comparison.
fn reg_value(cpu: &Cpu, i: usize) -> u32 {
    match i {
        0 => cpu.regs.cr0,
        1 => cpu.regs.cr3,
        2 => cpu.regs.gpr[reg::EAX as usize],
        3 => cpu.regs.gpr[reg::EBX as usize],
        4 => cpu.regs.gpr[reg::ECX as usize],
        5 => cpu.regs.gpr[reg::EDX as usize],
        6 => cpu.regs.gpr[reg::ESI as usize],
        7 => cpu.regs.gpr[reg::EDI as usize],
        8 => cpu.regs.gpr[reg::EBP as usize],
        9 => cpu.regs.gpr[reg::ESP as usize],
        10 => cpu.regs.seg[reg::CS as usize].sel as u32,
        11 => cpu.regs.seg[reg::DS as usize].sel as u32,
        12 => cpu.regs.seg[reg::ES as usize].sel as u32,
        13 => cpu.regs.seg[reg::FS as usize].sel as u32,
        14 => cpu.regs.seg[reg::GS as usize].sel as u32,
        15 => cpu.regs.seg[reg::SS as usize].sel as u32,
        16 => cpu.regs.eip,
        17 => cpu.regs.eflags.bits() | 2,
        18 => cpu.regs.dr[6],
        19 => cpu.regs.dr[7],
        _ => unreachable!(),
    }
}

const REG_NAMES: [&str; NREGS] = [
    "cr0", "cr3", "eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "cs", "ds", "es", "fs",
    "gs", "ss", "eip", "eflags", "dr6", "dr7",
];

fn set_initial(cpu: &mut Cpu, st: &State) {
    cpu.regs.cr0 = st.regs[0];
    cpu.regs.cr3 = st.regs[1];
    cpu.regs.gpr[reg::EAX as usize] = st.regs[2];
    cpu.regs.gpr[reg::EBX as usize] = st.regs[3];
    cpu.regs.gpr[reg::ECX as usize] = st.regs[4];
    cpu.regs.gpr[reg::EDX as usize] = st.regs[5];
    cpu.regs.gpr[reg::ESI as usize] = st.regs[6];
    cpu.regs.gpr[reg::EDI as usize] = st.regs[7];
    cpu.regs.gpr[reg::EBP as usize] = st.regs[8];
    cpu.regs.gpr[reg::ESP as usize] = st.regs[9];
    for (i, idx) in [
        (10, reg::CS),
        (11, reg::DS),
        (12, reg::ES),
        (13, reg::FS),
        (14, reg::GS),
        (15, reg::SS),
    ] {
        cpu.regs.seg[idx as usize] = SegReg::real(st.regs[i] as u16);
    }
    cpu.regs.eip = st.regs[16];
    cpu.regs.eflags = EFlags::from_bits_truncate(st.regs[17]);
    cpu.regs.dr[6] = st.regs[18];
    cpu.regs.dr[7] = st.regs[19];
}

/// Compare mask for register `i` (segment selectors compare 16 bits, EFLAGS
/// only its architectural bits).
fn cmp_mask(i: usize, mask: [u32; NREGS]) -> u32 {
    match i {
        10..=15 => mask[i] & 0xFFFF,
        17 => mask[i] & EFLAGS_CMP,
        _ => mask[i],
    }
}

/// Run every case in one opcode file. Returns (failures, cases).
fn run_file(path: &PathBuf, revoked: &HashSet<String>) -> (usize, usize) {
    let raw =
        std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut data = Vec::new();
    flate2::read::GzDecoder::new(&raw[..])
        .read_to_end(&mut data)
        .unwrap_or_else(|e| panic!("failed to gunzip {}: {e}", path.display()));
    let (mut tests, _) = parse_moo(&data);

    // The v1 MOO files for two-operand IMUL predate the suite's metadata
    // fix (SingleStepTests/80386#6) marking SF/ZF/AF/PF undefined
    // (f_umask 0xFF2B); apply that mask here.
    let stem = path.file_name().unwrap().to_string_lossy();
    if stem.contains("0FAF") {
        for t in &mut tests {
            t.mask[17] &= 0xFFFF_FF2B;
        }
    }

    let mut failures = 0;
    let mut skipped = 0;
    // One 16 MiB memory reused across cases: each case seeds every address
    // it touches up front and we zero initial∪final addresses afterwards.
    let mut bus = RigBus {
        mem: LinearMemory::new(),
    };
    // One CPU reused across cases: `reset()` restores power-on state and
    // invalidates the caches (the reseeded RAM is a host-side write), and
    // `set_initial` overwrites the whole register file. Constructing a CPU
    // per case would allocate a fresh icache 1.76M times.
    let mut cpu = Cpu::new();
    for t in &tests {
        let hash_hex: String = t.hash.iter().map(|b| format!("{b:02x}")).collect();
        if revoked.contains(&hash_hex) {
            continue;
        }
        // A handful of tests overwrite their own instruction bytes (BYTS
        // includes the terminating HALT): the real CPU then runs stale bytes
        // from its prefetch queue, which this core does not model. Skip them.
        let code_start = (t.init.regs[10] << 4).wrapping_add(t.init.regs[16]);
        let self_modifying = t
            .fina
            .ram
            .iter()
            .any(|&(addr, _)| addr.wrapping_sub(code_start) < t.bytes.len() as u32);
        if self_modifying {
            skipped += 1;
            continue;
        }
        for &(addr, val) in &t.init.ram {
            bus.mem.ram[addr as usize & 0xFF_FFFF] = val;
        }

        cpu.reset();
        set_initial(&mut cpu, &t.init);
        let mut steps = 0;
        while !cpu.halted && steps < 1000 {
            cpu.step(&mut bus);
            steps += 1;
        }

        let mut errors: Vec<String> = Vec::new();
        if steps >= 1000 {
            errors.push("did not reach HLT".into());
        }

        for (i, name) in REG_NAMES.iter().enumerate() {
            let exp = if t.fina.present & (1 << i) != 0 {
                t.fina.regs[i]
            } else {
                t.init.regs[i]
            };
            let got = reg_value(&cpu, i);
            let m = cmp_mask(i, t.mask);
            if (got ^ exp) & m != 0 {
                errors.push(format!(
                    "{name}: exp {exp:08X} got {got:08X} (mask {m:08X})"
                ));
            }
        }

        for &(addr, val) in &t.fina.ram {
            let got = bus.mem.ram[addr as usize & 0xFF_FFFF];
            // The FLAGS image pushed by an exception may contain undefined
            // flags; mask them via the file's eflags mask.
            let m = match t.exception {
                Some((_, fa)) if addr == fa => t.mask[17] as u8,
                Some((_, fa)) if addr == fa + 1 => (t.mask[17] >> 8) as u8,
                _ => 0xFF,
            };
            if (got ^ val) & m != 0 {
                errors.push(format!("ram[{addr:06X}]: exp {val:02X} got {got:02X}"));
            }
        }

        if !errors.is_empty() {
            if failures < 5 {
                let exc = match t.exception {
                    Some((v, fa)) => format!(" exc={v}@{fa:06X}"),
                    None => String::new(),
                };
                eprintln!(
                    "FAIL {} [{}]{}: {} || init eip={:04X} sp={:08X} eax={:08X} ebx={:08X} ecx={:08X} flags={:08X}",
                    path.file_name().unwrap().to_string_lossy(),
                    t.name,
                    exc,
                    errors.join("; "),
                    t.init.regs[16],
                    t.init.regs[9],
                    t.init.regs[2],
                    t.init.regs[3],
                    t.init.regs[4],
                    t.init.regs[17],
                );
            }
            failures += 1;
        }

        for &(addr, _) in &t.init.ram {
            bus.mem.ram[addr as usize & 0xFF_FFFF] = 0;
        }
        for &(addr, _) in &t.fina.ram {
            bus.mem.ram[addr as usize & 0xFF_FFFF] = 0;
        }
    }
    if skipped > 0 {
        eprintln!(
            "{}: skipped {skipped} prefetch-dependent (self-modifying) tests",
            path.file_name().unwrap().to_string_lossy()
        );
    }
    (failures, tests.len())
}

fn load_revocations(dir: &Path) -> HashSet<String> {
    let mut set = HashSet::new();
    // The revocation list sits next to the test directory.
    for candidate in [
        dir.join("revocation_list.txt"),
        dir.join("../revocation_list.txt"),
    ] {
        if let Ok(data) = std::fs::read_to_string(&candidate) {
            set.extend(
                data.lines()
                    .map(|l| l.trim().to_lowercase())
                    .filter(|l| !l.is_empty()),
            );
            break;
        }
    }
    set
}

#[test]
fn harte_80386_suite() {
    let Ok(dir) = std::env::var("REMU_HARTE_80386_DIR") else {
        eprintln!("REMU_HARTE_80386_DIR not set — skipping SingleStepTests 80386 suite.");
        return;
    };
    let dir = PathBuf::from(dir);
    let revoked = load_revocations(&dir);

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().ends_with(".MOO.gz"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();

    let mut total_failures = 0;
    let mut total_cases = 0;
    let mut failed_opcodes = Vec::new();
    for path in &entries {
        let stem = path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .trim_end_matches(".MOO.gz")
            .to_string();
        let (f, n) = run_file(path, &revoked);
        total_cases += n;
        if f > 0 {
            total_failures += f;
            failed_opcodes.push(format!("{stem} ({f}/{n})"));
        }
    }

    if total_failures > 0 {
        panic!(
            "{total_failures} failing cases across {} opcode files: {}",
            failed_opcodes.len(),
            failed_opcodes.join(", ")
        );
    }
    eprintln!(
        "SingleStepTests 80386 suite passed: {} files, {total_cases} cases.",
        entries.len()
    );
}
