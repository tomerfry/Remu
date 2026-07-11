//! SingleStepTests harness for the 8086/8088 core (the `8088` v2 JSON suite,
//! captured from a real NMOS 8088 by Daniel Balsom).
//!
//! Each gzipped JSON file is named after an opcode (`00.json.gz`) or a
//! group-opcode/reg pair (`80.3.json.gz`) and contains up to 10,000 cases,
//! each with an initial CPU+RAM state and the expected *changes* (final state
//! lists only registers and memory that differ). We set the CPU to `initial`,
//! run one `step()`, and assert the final registers, flags and RAM.
//!
//! Flags are compared under the per-opcode `flags-mask` from the suite's
//! `metadata.json`, which excludes flags the hardware leaves undefined.
//! Cycle counts are NOT validated: the suite counts real bus cycles including
//! prefetch-queue effects, which this tier-2 core does not model.
//!
//! The data is large and is **not** vendored. Point the test at a checkout:
//!
//! ```text
//! git clone --depth 1 --filter=blob:none --sparse https://github.com/SingleStepTests/8088.git
//! cd 8088 && git sparse-checkout set v2
//! REMU_HARTE_8088_DIR=/path/to/8088/v2 cargo test --release --test harte_8088
//! ```
//!
//! When the variable is unset the test prints a notice and passes, so CI
//! without the data stays green.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use remu::x86::{Cpu, Flags, LinearMemory};
use serde::Deserialize;

#[derive(Deserialize, Default, Clone, Copy)]
struct Regs {
    ax: Option<u16>,
    bx: Option<u16>,
    cx: Option<u16>,
    dx: Option<u16>,
    cs: Option<u16>,
    ss: Option<u16>,
    ds: Option<u16>,
    es: Option<u16>,
    sp: Option<u16>,
    bp: Option<u16>,
    si: Option<u16>,
    di: Option<u16>,
    ip: Option<u16>,
    flags: Option<u16>,
}

#[derive(Deserialize)]
struct State {
    regs: Regs,
    ram: Vec<(u32, u8)>,
}

#[derive(Deserialize)]
struct TestCase {
    name: String,
    initial: State,
    #[serde(rename = "final")]
    final_state: State,
}

/// Load `<opcode>[.<reg>]` → flags-mask entries from the suite's metadata.
fn load_flag_masks(dir: &Path) -> HashMap<String, u16> {
    let mut masks = HashMap::new();
    let Ok(data) = std::fs::read_to_string(dir.join("metadata.json")) else {
        return masks;
    };
    let meta: serde_json::Value = serde_json::from_str(&data).expect("bad metadata.json");
    let Some(opcodes) = meta.get("opcodes").and_then(|v| v.as_object()) else {
        return masks;
    };
    for (op, entry) in opcodes {
        if let Some(regs) = entry.get("reg").and_then(|v| v.as_object()) {
            for (r, sub) in regs {
                if let Some(m) = sub.get("flags-mask").and_then(|v| v.as_u64()) {
                    masks.insert(format!("{op}.{r}"), m as u16);
                }
            }
        } else if let Some(m) = entry.get("flags-mask").and_then(|v| v.as_u64()) {
            masks.insert(op.clone(), m as u16);
        }
    }
    masks
}

fn set_initial(cpu: &mut Cpu, r: &Regs) {
    cpu.regs.ax = r.ax.unwrap_or(0);
    cpu.regs.bx = r.bx.unwrap_or(0);
    cpu.regs.cx = r.cx.unwrap_or(0);
    cpu.regs.dx = r.dx.unwrap_or(0);
    cpu.regs.cs = r.cs.unwrap_or(0);
    cpu.regs.ss = r.ss.unwrap_or(0);
    cpu.regs.ds = r.ds.unwrap_or(0);
    cpu.regs.es = r.es.unwrap_or(0);
    cpu.regs.sp = r.sp.unwrap_or(0);
    cpu.regs.bp = r.bp.unwrap_or(0);
    cpu.regs.si = r.si.unwrap_or(0);
    cpu.regs.di = r.di.unwrap_or(0);
    cpu.regs.ip = r.ip.unwrap_or(0);
    cpu.regs.flags = Flags::from_word(r.flags.unwrap_or(0));
}

/// Run every case in one opcode file. Returns (failures, cases).
fn run_file(path: &PathBuf, flags_mask: u16) -> (usize, usize) {
    let raw =
        std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut data = String::new();
    flate2::read::GzDecoder::new(&raw[..])
        .read_to_string(&mut data)
        .unwrap_or_else(|e| panic!("failed to gunzip {}: {e}", path.display()));
    let cases: Vec<TestCase> = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));

    let mut failures = 0;
    // One 1 MiB memory reused across cases: each case seeds every address it
    // touches up front and we zero initial∪final addresses afterwards.
    let mut mem = LinearMemory::new();
    for case in &cases {
        for &(addr, val) in &case.initial.ram {
            mem.ram[addr as usize & 0xF_FFFF] = val;
        }

        let mut cpu = Cpu::new();
        set_initial(&mut cpu, &case.initial.regs);
        cpu.step(&mut mem);

        let i = &case.initial.regs;
        let f = &case.final_state.regs;
        let exp = |fin: Option<u16>, ini: Option<u16>| fin.or(ini).unwrap_or(0);
        let exp_flags = exp(f.flags, i.flags);
        let got = &cpu.regs;
        let mut ok = got.ax == exp(f.ax, i.ax)
            && got.bx == exp(f.bx, i.bx)
            && got.cx == exp(f.cx, i.cx)
            && got.dx == exp(f.dx, i.dx)
            && got.cs == exp(f.cs, i.cs)
            && got.ss == exp(f.ss, i.ss)
            && got.ds == exp(f.ds, i.ds)
            && got.es == exp(f.es, i.es)
            && got.sp == exp(f.sp, i.sp)
            && got.bp == exp(f.bp, i.bp)
            && got.si == exp(f.si, i.si)
            && got.di == exp(f.di, i.di)
            && got.ip == exp(f.ip, i.ip)
            && got.flags.to_word() & flags_mask == exp_flags & flags_mask;

        if ok {
            for &(addr, val) in &case.final_state.ram {
                if mem.ram[addr as usize & 0xF_FFFF] != val {
                    ok = false;
                    break;
                }
            }
        }

        if !ok {
            if failures < 5 {
                eprintln!(
                    "FAIL {} [{}]:\n  exp ip:{:04X} flags:{:04X} ax:{:04X} bx:{:04X} cx:{:04X} dx:{:04X} sp:{:04X} bp:{:04X} si:{:04X} di:{:04X} cs:{:04X} ss:{:04X} ds:{:04X} es:{:04X}\n  got ip:{:04X} flags:{:04X} ax:{:04X} bx:{:04X} cx:{:04X} dx:{:04X} sp:{:04X} bp:{:04X} si:{:04X} di:{:04X} cs:{:04X} ss:{:04X} ds:{:04X} es:{:04X} (mask {:04X})",
                    path.file_name().unwrap().to_string_lossy(),
                    case.name,
                    exp(f.ip, i.ip),
                    exp_flags,
                    exp(f.ax, i.ax),
                    exp(f.bx, i.bx),
                    exp(f.cx, i.cx),
                    exp(f.dx, i.dx),
                    exp(f.sp, i.sp),
                    exp(f.bp, i.bp),
                    exp(f.si, i.si),
                    exp(f.di, i.di),
                    exp(f.cs, i.cs),
                    exp(f.ss, i.ss),
                    exp(f.ds, i.ds),
                    exp(f.es, i.es),
                    got.ip,
                    got.flags.to_word(),
                    got.ax,
                    got.bx,
                    got.cx,
                    got.dx,
                    got.sp,
                    got.bp,
                    got.si,
                    got.di,
                    got.cs,
                    got.ss,
                    got.ds,
                    got.es,
                    flags_mask,
                );
            }
            failures += 1;
        }

        for &(addr, _) in &case.initial.ram {
            mem.ram[addr as usize & 0xF_FFFF] = 0;
        }
        for &(addr, _) in &case.final_state.ram {
            mem.ram[addr as usize & 0xF_FFFF] = 0;
        }
    }
    (failures, cases.len())
}

#[test]
fn harte_8088_suite() {
    let Ok(dir) = std::env::var("REMU_HARTE_8088_DIR") else {
        eprintln!("REMU_HARTE_8088_DIR not set — skipping SingleStepTests 8088 suite.");
        return;
    };
    let dir = PathBuf::from(dir);
    let masks = load_flag_masks(&dir);

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().ends_with(".json.gz"))
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
            .trim_end_matches(".json.gz")
            .to_string();
        let mask = masks.get(&stem).copied().unwrap_or(0xFFFF);
        let (f, n) = run_file(path, mask);
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
        "SingleStepTests 8088 suite passed: {} files, {total_cases} cases.",
        entries.len()
    );
}
