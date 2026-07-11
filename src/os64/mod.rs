//! A Linux (x86-64) userspace operating-system emulation layer: loads 64-bit
//! ELF binaries and services their syscalls against a host-backed virtual
//! filesystem, in the spirit of qiling. Built on the [`crate::x86_64`] core.
//!
//! This is the 64-bit sibling of [`crate::os`] (i386). The guest runs under
//! real long-mode paging in ring 3; its `SYSCALL` instructions and CPU faults
//! are trapped back to this layer via the core's host hooks. Statically linked
//! `ET_EXEC` (and non-PIE) images run today; dynamic linking (loading
//! `ld-linux-x86-64.so.2` from a rootfs) is scaffolded but not yet complete.
//!
//! ```no_run
//! use remu::os64::Emulator;
//! use std::path::Path;
//!
//! let mut emu = Emulator::load(Path::new("hello"), &["hello".into()], &[], None).unwrap();
//! let code = emu.run();
//! std::process::exit(code);
//! ```

pub mod abi;
pub mod arch;
pub mod fs;
pub mod loader;
pub mod memory;
pub mod process;
mod syscall;

use std::path::{Path, PathBuf};

use crate::x86_64::{Cpu, Exception, HostTrap};

use fs::Vfs;
use memory::{AddressSpace, PROT_READ, PROT_WRITE, PhysMem, STACK_TOP, VmaKind};

/// Initial stack reservation (grows down on demand beyond this).
const INIT_STACK: u64 = 0x0004_0000; // 256 KiB

/// A running Linux x86-64 process.
pub struct Emulator {
    pub cpu: Cpu,
    pub mem: PhysMem,
    pub aspace: AddressSpace,
    pub vfs: Vfs,
    /// Whether the process is still running (cleared by `exit`).
    pub running: bool,
    /// Exit status (low byte of `exit`, or 128+signal on a fatal fault).
    pub exit_code: i32,
    /// Guest thread-pointer clear address from `set_tid_address` (stored only).
    pub clear_tid: u64,
    /// Print diagnostics for unimplemented syscalls and fatal faults.
    pub trace: bool,
}

impl Emulator {
    /// Load an ELF program from a file and prepare it to run.
    pub fn load(
        elf_path: &Path,
        argv: &[String],
        envp: &[String],
        rootfs: Option<PathBuf>,
    ) -> Result<Emulator, String> {
        let data = std::fs::read(elf_path).map_err(|e| format!("read {elf_path:?}: {e}"))?;
        Self::load_image(&data, argv, envp, rootfs)
    }

    /// Load an ELF program from an in-memory image and prepare it to run: build
    /// the address space, map segments (and the interpreter, if dynamically
    /// linked), construct the initial stack, and enter ring-3 long mode.
    pub fn load_image(
        data: &[u8],
        argv: &[String],
        envp: &[String],
        rootfs: Option<PathBuf>,
    ) -> Result<Emulator, String> {
        let mut mem = PhysMem::new();
        let mut aspace = AddressSpace::new(&mut mem);

        // Load the main image (PIE loads at EXE_PIE_BASE; ET_EXEC at its vaddrs).
        let image = loader::load(&mut aspace, &mut mem, data, loader::EXE_PIE_BASE)?;

        // Load the dynamic linker if requested by PT_INTERP.
        let (entry, interp_base) = if let Some(interp) = &image.interp {
            let interp_data = Self::read_interp(rootfs.as_ref(), interp)?;
            let interp_img = loader::load(&mut aspace, &mut mem, &interp_data, loader::INTERP_BASE)?;
            (interp_img.entry, Some(interp_img.base))
        } else {
            (image.entry, None)
        };

        // Heap starts above the executable image.
        aspace.init_brk(image.load_end);

        // Reserve and map the initial stack.
        aspace.map(
            &mut mem,
            STACK_TOP - INIT_STACK,
            INIT_STACK,
            PROT_READ | PROT_WRITE,
            VmaKind::Stack,
        );

        let argv: Vec<Vec<u8>> = if argv.is_empty() {
            vec![b"/a.out".to_vec()]
        } else {
            argv.iter().map(|s| s.as_bytes().to_vec()).collect()
        };
        let envp: Vec<Vec<u8>> = envp.iter().map(|s| s.as_bytes().to_vec()).collect();

        let rsp = process::build_stack(&aspace, &mut mem, &image, interp_base, &argv, &envp, &argv[0]);

        let mut cpu = Cpu::new();
        arch::enter_user(&mut cpu, &aspace, entry, rsp);

        Ok(Emulator {
            cpu,
            mem,
            aspace,
            vfs: Vfs::new(),
            running: true,
            exit_code: 0,
            clear_tid: 0,
            trace: false,
        })
    }

    /// Read the interpreter (`ld-linux-x86-64.so.2`) from the rootfs.
    fn read_interp(rootfs: Option<&PathBuf>, interp: &str) -> Result<Vec<u8>, String> {
        let root = rootfs
            .ok_or_else(|| format!("binary needs interpreter {interp} but no rootfs was provided"))?;
        let host = root.join(interp.trim_start_matches('/'));
        std::fs::read(&host).map_err(|e| format!("read interpreter {host:?}: {e}"))
    }

    /// Run the process to completion, returning its exit status.
    pub fn run(&mut self) -> i32 {
        self.run_capped(u64::MAX)
    }

    /// Run for at most `max` instructions (a runaway guard for tests). Returns
    /// the exit status; if the cap is hit, returns 125.
    pub fn run_capped(&mut self, max: u64) -> i32 {
        let mut steps = 0u64;
        while self.running {
            if steps >= max {
                if self.trace {
                    eprintln!("[remu] instruction cap reached at rip={:#018x}", self.cpu.regs.rip);
                }
                return 125;
            }
            self.cpu.step(&mut self.mem);
            steps += 1;
            if let Some(trap) = self.cpu.host_trap.take() {
                match trap {
                    HostTrap::Syscall => self.dispatch_syscall(),
                    HostTrap::Exception(e) => self.handle_fault(e),
                }
                // Trap service writes guest memory host-side (read buffers,
                // mmap, stack growth) — stale decoded code must not survive.
                self.cpu.invalidate_icache();
            } else if self.cpu.shutdown {
                self.exit_code = 139;
                break;
            }
        }
        self.exit_code
    }

    /// Handle a CPU fault the guest kernel would normally take: demand-grow the
    /// stack, or terminate the process with the equivalent signal.
    fn handle_fault(&mut self, e: Exception) {
        if e.vector == 14 {
            let cr2 = self.cpu.regs.cr2;
            if let Some(bottom) = self.aspace.is_stack_growth(cr2) {
                self.aspace.grow_stack(&mut self.mem, bottom);
                return; // restart the faulting instruction
            }
        }
        let (sig, name) = match e.vector {
            0 => (8, "SIGFPE"),
            6 => (4, "SIGILL"),
            _ => (11, "SIGSEGV"),
        };
        if self.trace {
            eprintln!(
                "[remu] fatal fault vector={} ({name}) rip={:#018x} cr2={:#018x}",
                e.vector, self.cpu.regs.rip, self.cpu.regs.cr2
            );
        }
        self.running = false;
        self.exit_code = 128 + sig;
    }
}
