//! User-mode ("qemu-user" style) emulation: load a guest program into a
//! process-shaped address space, run it on an emulated CPU, and service its
//! OS system calls on the host.
//!
//! The layer is written against the small [`UserArch`] adapter trait rather
//! than any concrete core, so supporting another architecture means one new
//! adapter — the cores themselves stay self-contained. The only supported
//! guest ABI today is statically linked Linux i386 ELF executables on the
//! [`crate::x86_32`] core.

pub mod addr_space;
pub mod elf;
pub mod linux;
pub mod x86_32;

pub use addr_space::{AddressSpace, Segv};

/// What the user-mode runner needs from a CPU core: flat user-privilege
/// setup, single-stepping, and the guest's syscall register convention. One
/// small adapter per core; the cores themselves stay self-contained.
///
/// The trait is deliberately 32-bit/little-endian scoped, matching
/// [`AddressSpace`]; widening it is a future refactor, not a seam kept open
/// speculatively.
pub trait UserArch {
    /// The ELF `e_machine` this CPU executes (`EM_386` = 3).
    const ELF_MACHINE: u16;
    /// Page size reported to the guest.
    const PAGE_SIZE: u32 = 4096;

    /// A power-on CPU.
    fn new() -> Self;

    /// One-time user-mode setup: flat user-privilege machine state with the
    /// syscall trap armed, PC at `entry` and the stack pointer at `sp`. May
    /// map arch-private pages (the x86-32 adapter keeps a GDT in guest
    /// memory).
    fn setup(&mut self, mem: &mut AddressSpace, entry: u32, sp: u32);

    /// Execute one instruction; returns cycles (speed accounting only).
    fn step(&mut self, mem: &mut AddressSpace) -> u32;

    /// True exactly once per trapped syscall instruction (clears the latch).
    /// The CPU state already points at the instruction after the syscall.
    fn take_syscall(&mut self) -> bool;

    /// The guest's syscall number register (i386: EAX).
    fn syscall_number(&self) -> u32;

    /// The six syscall argument registers (i386: EBX ECX EDX ESI EDI EBP).
    fn syscall_args(&self) -> [u32; 6];

    /// Write the syscall return value (i386: EAX).
    fn set_syscall_ret(&mut self, value: u32);

    /// First claim on arch-specific syscalls before the generic table (i386
    /// claims `set_thread_area`). `Some(ret)` handles the call.
    fn arch_syscall(&mut self, mem: &mut AddressSpace, nr: u32, args: &[u32; 6]) -> Option<u32> {
        let _ = (mem, nr, args);
        None
    }

    /// Current program counter (for diagnostics).
    fn pc(&self) -> u32;

    /// Unrecoverable CPU state (386: triple-fault shutdown).
    fn dead(&self) -> bool;

    /// Multi-line register dump for fault reports and tracing.
    fn dump(&self) -> String;
}

/// An emulated user-mode process: CPU, address space and host-side state.
pub struct Usermode<A: UserArch> {
    pub cpu: A,
    pub mem: AddressSpace,
    pub process: linux::Process,
}

impl<A: UserArch> Usermode<A> {
    /// Load a statically linked Linux ELF executable and prepare it to run:
    /// image mapped, brk placed, initial stack (argv/envp/auxv) built, CPU in
    /// flat user mode at the entry point.
    pub fn load(prog: &[u8], argv: &[&str], envp: &[&str]) -> Result<Self, elf::LoadError> {
        let image = elf::parse(prog)?;
        if image.machine != A::ELF_MACHINE {
            return Err(elf::LoadError::WrongMachine(image.machine));
        }
        let mut mem = AddressSpace::new();
        let (entry, brk_start, phdr_vaddr) = elf::load(&image, prog, &mut mem);
        mem.brk_base = brk_start;
        mem.brk = brk_start;

        let mut process = linux::Process::new();
        let table = image.phoff as usize..image.phoff as usize + image.phnum as usize * 32;
        let sp = linux::build_stack(
            &mut mem,
            &mut process,
            argv,
            envp,
            entry,
            phdr_vaddr,
            &prog[table],
            image.phnum,
        );

        let mut cpu = A::new();
        cpu.setup(&mut mem, entry, sp);
        Ok(Usermode { cpu, mem, process })
    }

    /// Advance one instruction, servicing any syscall it raised. `Some` once
    /// the process has ended (tracing frontends drive this directly).
    #[inline]
    pub fn step_one(&mut self) -> Option<Exit> {
        self.cpu.step(&mut self.mem);
        if self.cpu.take_syscall() {
            let nr = self.cpu.syscall_number();
            let args = self.cpu.syscall_args();
            match linux::dispatch(&mut self.cpu, &mut self.mem, &mut self.process, nr, args) {
                linux::Control::Ret(v) => self.cpu.set_syscall_ret(v),
                linux::Control::Exit(code) => return Some(Exit::Exited(code)),
            }
        }
        if let Some(s) = self.mem.take_segv() {
            return Some(Exit::Fault(format!(
                "segmentation fault: {} unmapped address {:#010x}\n{}",
                if s.write { "write to" } else { "read of" },
                s.addr,
                self.cpu.dump()
            )));
        }
        if self.cpu.dead() {
            return Some(Exit::Fault(format!(
                "unhandled CPU exception (triple fault)\n{}",
                self.cpu.dump()
            )));
        }
        None
    }

    /// Run the process to completion.
    pub fn run(&mut self) -> Exit {
        loop {
            if let Some(exit) = self.step_one() {
                return exit;
            }
        }
    }
}

/// How an emulated process ended.
#[derive(Debug)]
pub enum Exit {
    /// The guest called `exit`/`exit_group` with this status.
    Exited(i32),
    /// The guest died (unmapped access or unhandled CPU exception); the
    /// message includes a register dump.
    Fault(String),
}
