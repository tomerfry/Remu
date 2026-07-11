//! User-mode frontend: run a statically linked Linux i386 ELF executable,
//! emulating its syscalls on the host (qemu-user style).
//!
//! Usage:
//!   remu-user [--strace] [--trace] [--env KEY=VAL]... <program.elf> [guest args...]
//!
//! The guest's exit status becomes the host exit code; a guest crash prints a
//! register dump and exits 139 (as after SIGSEGV).

use std::process::ExitCode;

use remu::usermode::{Exit, UserArch, Usermode};
use remu::x86_32::Cpu;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut strace = false;
    let mut trace = false;
    let mut env: Vec<String> = ["PATH=/usr/bin:/bin", "HOME=/", "TERM=dumb"]
        .map(String::from)
        .into();
    let mut program: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--strace" => strace = true,
            "--trace" => trace = true,
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
            "usage: remu-user [--strace] [--trace] [--env KEY=VAL]... <program.elf> [args...]"
        );
        return ExitCode::FAILURE;
    };

    let path = &args[prog_idx];
    let image = match std::fs::read(path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let argv: Vec<&str> = args[prog_idx..].iter().map(String::as_str).collect();
    let envp: Vec<&str> = env.iter().map(String::as_str).collect();
    let mut um: Usermode<Cpu> = match Usermode::load(&image, &argv, &envp) {
        Ok(um) => um,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    um.process.strace = strace;

    let exit = if trace {
        loop {
            eprintln!("[remu] {}", um.cpu.dump().replace('\n', " | "));
            if let Some(exit) = um.step_one() {
                break exit;
            }
        }
    } else {
        um.run()
    };

    match exit {
        Exit::Exited(code) => ExitCode::from((code & 0xFF) as u8),
        Exit::Fault(msg) => {
            eprintln!("{msg}");
            ExitCode::from(139)
        }
    }
}
