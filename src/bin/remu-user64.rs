//! User-mode frontend for Linux x86-64 ELF binaries: run a program on the
//! emulated x86-64 core under long-mode paging, servicing its syscalls on the
//! host (qiling / qemu-user style).
//!
//! Usage:
//!   remu-user64 [--trace] [--rootfs DIR] [--env KEY=VAL]... <program.elf> [guest args...]
//!
//! The guest's exit status becomes the host exit code; a guest crash prints a
//! diagnostic (with `--trace`) and exits 139 (as after SIGSEGV). Only
//! statically linked images run today; a `--rootfs` is required for
//! dynamically linked ones (dynamic-linker support is still in progress).

use std::path::PathBuf;
use std::process::ExitCode;

use remu::os64::Emulator;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut trace = false;
    let mut rootfs: Option<PathBuf> = None;
    let mut env: Vec<String> = ["PATH=/usr/bin:/bin", "HOME=/", "TERM=dumb"]
        .map(String::from)
        .into();
    let mut program: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--trace" => trace = true,
            "--rootfs" => {
                i += 1;
                let Some(dir) = args.get(i) else {
                    eprintln!("--rootfs requires a directory");
                    return ExitCode::FAILURE;
                };
                rootfs = Some(PathBuf::from(dir));
            }
            "--env" => {
                i += 1;
                let Some(kv) = args.get(i) else {
                    eprintln!("--env requires KEY=VAL");
                    return ExitCode::FAILURE;
                };
                let Some(key) = kv.split_once('=').map(|(k, _)| k) else {
                    eprintln!("--env requires KEY=VAL, got `{kv}`");
                    return ExitCode::FAILURE;
                };
                env.retain(|e| e.split_once('=').is_none_or(|(k, _)| k != key));
                env.push(kv.clone());
            }
            _ => {
                program = Some(i);
                break; // everything from here on belongs to the guest
            }
        }
        i += 1;
    }
    let Some(prog_idx) = program else {
        eprintln!(
            "usage: remu-user64 [--trace] [--rootfs DIR] [--env KEY=VAL]... <program.elf> [args...]"
        );
        return ExitCode::FAILURE;
    };

    let path = PathBuf::from(&args[prog_idx]);
    let argv: Vec<String> = args[prog_idx..].to_vec();

    let mut emu = match Emulator::load(&path, &argv, &env, rootfs) {
        Ok(emu) => emu,
        Err(e) => {
            eprintln!("{}: {e}", args[prog_idx]);
            return ExitCode::FAILURE;
        }
    };
    emu.trace = trace;

    let code = emu.run();
    ExitCode::from((code & 0xFF) as u8)
}
