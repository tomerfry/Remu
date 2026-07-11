//! Run a 32-bit Linux ELF binary under the Remu OS-emulation layer.
//!
//! ```text
//! cargo run --example linux -- [--rootfs DIR] [--trace] <elf> [args...]
//! ```
//!
//! With no rootfs, statically-linked binaries (and the bundled test fixtures)
//! run directly. Dynamically-linked binaries need `--rootfs` pointing at an
//! i386 root containing their `ld-linux.so.2` and shared libraries.

use std::path::{Path, PathBuf};

use remu::os::Emulator;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut rootfs: Option<PathBuf> = None;
    let mut trace = false;
    let mut elf: Option<String> = None;
    let mut prog_args: Vec<String> = Vec::new();

    while let Some(a) = args.next() {
        match a.as_str() {
            "--rootfs" => rootfs = args.next().map(PathBuf::from),
            "--trace" => trace = true,
            "-h" | "--help" => {
                usage();
                return;
            }
            _ => {
                elf = Some(a);
                prog_args.extend(args.by_ref());
                break;
            }
        }
    }

    let Some(elf) = elf else {
        usage();
        std::process::exit(2);
    };

    let mut argv = vec![elf.clone()];
    argv.extend(prog_args);
    let envp = vec![
        "PATH=/bin:/usr/bin".to_string(),
        "HOME=/root".to_string(),
        "TERM=xterm".to_string(),
    ];

    match Emulator::load(Path::new(&elf), &argv, &envp, rootfs) {
        Ok(mut emu) => {
            emu.trace = trace;
            let code = emu.run();
            std::process::exit(code);
        }
        Err(e) => {
            eprintln!("remu: load error: {e}");
            std::process::exit(1);
        }
    }
}

fn usage() {
    eprintln!("usage: linux [--rootfs DIR] [--trace] <elf> [args...]");
}
