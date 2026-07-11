//! Linux i386 process emulation: per-process state, the initial stack image,
//! and (in [`dispatch`]) the syscall table.

pub mod abi;
pub mod fs;

use super::UserArch;
use super::addr_space::{AddressSpace, PAGE_SIZE, STACK_SIZE, STACK_TOP};
use abi::err;

/// Host-side state of the emulated process.
pub struct Process {
    /// The open-file table (0-2 are the host standard streams).
    pub fds: fs::FdTable,
    /// Deterministic-per-run PRNG for `AT_RANDOM` and `getrandom(2)`.
    prng: u64,
    /// Process start, for `CLOCK_MONOTONIC`.
    start: std::time::Instant,
    /// Log one line per syscall to stderr.
    pub strace: bool,
}

impl Process {
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Process {
            fds: fs::FdTable::new(),
            prng: seed | 1,
            start: std::time::Instant::now(),
            strace: false,
        }
    }

    /// Fill `buf` with pseudo-random bytes (xorshift64*; not cryptographic,
    /// which matches what an emulated guest can expect).
    pub fn rand_bytes(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            self.prng ^= self.prng << 13;
            self.prng ^= self.prng >> 7;
            self.prng ^= self.prng << 17;
            let v = self.prng.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
    }
}

impl Default for Process {
    fn default() -> Self {
        Self::new()
    }
}

/// Push `data` onto the downward-growing stack; returns its guest address.
fn push(mem: &mut AddressSpace, top: &mut u32, data: &[u8]) -> u32 {
    *top -= data.len() as u32;
    mem.write_bytes(*top, data).expect("stack region is mapped");
    *top
}

/// Push a NUL-terminated string (NUL first — the stack grows down, so the
/// string then reads forward in memory).
fn push_cstr(mem: &mut AddressSpace, top: &mut u32, s: &[u8]) -> u32 {
    push(mem, top, &[0]);
    push(mem, top, s)
}

/// Map the stack region and build the i386 SysV process-entry image:
///
/// ```text
/// STACK_TOP  [argv/envp string bytes]
///            [16 bytes of AT_RANDOM data]
///            [phdr table copy, only when no PT_LOAD covers it]
///            [auxv pairs, AT_NULL]  [envp ptrs, 0]  [argv ptrs, 0]
/// ESP ->     [argc]                              (ESP 16-byte aligned)
/// ```
///
/// `phdr_vaddr`/`phdr_bytes` come from the ELF loader: the guest address of
/// the program header table if a `PT_LOAD` maps it, else the raw bytes to
/// copy onto the stack (musl/glibc need `AT_PHDR` for static TLS init).
/// Returns the initial stack pointer.
#[allow(clippy::too_many_arguments)] // it assembles exactly these eight inputs
pub fn build_stack(
    mem: &mut AddressSpace,
    process: &mut Process,
    argv: &[&str],
    envp: &[&str],
    entry: u32,
    phdr_vaddr: Option<u32>,
    phdr_bytes: &[u8],
    phnum: u16,
) -> u32 {
    mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE);
    let mut top = STACK_TOP;

    let argv_ptrs: Vec<u32> = argv
        .iter()
        .map(|s| push_cstr(mem, &mut top, s.as_bytes()))
        .collect();
    let envp_ptrs: Vec<u32> = envp
        .iter()
        .map(|s| push_cstr(mem, &mut top, s.as_bytes()))
        .collect();

    let mut rand = [0u8; 16];
    process.rand_bytes(&mut rand);
    let rand_ptr = push(mem, &mut top, &rand);

    let phdr_ptr = match phdr_vaddr {
        Some(v) => Some(v),
        None if !phdr_bytes.is_empty() => {
            top &= !3; // keep the table copy word-aligned
            Some(push(mem, &mut top, phdr_bytes))
        }
        None => None,
    };

    let mut auxv: Vec<[u32; 2]> = Vec::new();
    if let Some(p) = phdr_ptr {
        auxv.push([abi::AT_PHDR, p]);
        auxv.push([abi::AT_PHENT, 32]);
        auxv.push([abi::AT_PHNUM, phnum as u32]);
    }
    auxv.push([abi::AT_PAGESZ, PAGE_SIZE]);
    auxv.push([abi::AT_BASE, 0]);
    auxv.push([abi::AT_FLAGS, 0]);
    auxv.push([abi::AT_ENTRY, entry]);
    auxv.push([abi::AT_UID, abi::GUEST_ID]);
    auxv.push([abi::AT_EUID, abi::GUEST_ID]);
    auxv.push([abi::AT_GID, abi::GUEST_ID]);
    auxv.push([abi::AT_EGID, abi::GUEST_ID]);
    auxv.push([abi::AT_HWCAP, 0]);
    auxv.push([abi::AT_CLKTCK, 100]);
    auxv.push([abi::AT_SECURE, 0]);
    auxv.push([abi::AT_RANDOM, rand_ptr]);
    auxv.push([abi::AT_NULL, 0]);

    let words = 1 + argv_ptrs.len() + 1 + envp_ptrs.len() + 1 + 2 * auxv.len();
    let esp = (top - words as u32 * 4) & !15;

    let mut p = esp;
    let mut put = |v: u32| {
        mem.write_u32(p, v);
        p += 4;
    };
    put(argv_ptrs.len() as u32); // argc
    for a in &argv_ptrs {
        put(*a);
    }
    put(0);
    for e in &envp_ptrs {
        put(*e);
    }
    put(0);
    for [tag, val] in &auxv {
        put(*tag);
        put(*val);
    }
    esp
}

// --- Syscall dispatch ----------------------------------------------------------

/// What a serviced syscall means for the run loop.
pub enum Control {
    /// Write this value to the return register and continue.
    Ret(u32),
    /// The process exited with this status.
    Exit(i32),
}

/// Service one syscall. The CPU adapter gets first claim (arch-specific
/// calls like i386 `set_thread_area`); everything else goes through the
/// generic table. Unknown numbers return `-ENOSYS`.
pub fn dispatch<A: UserArch>(
    cpu: &mut A,
    mem: &mut AddressSpace,
    process: &mut Process,
    nr: u32,
    args: [u32; 6],
) -> Control {
    let ctl = dispatch_inner(cpu, mem, process, nr, args);
    if process.strace {
        let name = syscall_name(nr);
        match &ctl {
            Control::Ret(v) => eprintln!(
                "[remu] {name}({:#x}, {:#x}, {:#x}) = {}",
                args[0], args[1], args[2], *v as i32
            ),
            Control::Exit(c) => eprintln!("[remu] {name}({c})"),
        }
    }
    ctl
}

fn dispatch_inner<A: UserArch>(
    cpu: &mut A,
    mem: &mut AddressSpace,
    process: &mut Process,
    nr: u32,
    args: [u32; 6],
) -> Control {
    if let Some(ret) = cpu.arch_syscall(mem, nr, &args) {
        return Control::Ret(ret);
    }
    let fds = &mut process.fds;
    let ret = match nr {
        abi::NR_EXIT | abi::NR_EXIT_GROUP => return Control::Exit(args[0] as i32),

        // File / IO.
        abi::NR_READ => fs::sys_read(mem, fds, args[0], args[1], args[2]),
        abi::NR_WRITE => fs::sys_write(mem, fds, args[0], args[1], args[2]),
        abi::NR_READV => fs::sys_readv(mem, fds, args[0], args[1], args[2]),
        abi::NR_WRITEV => fs::sys_writev(mem, fds, args[0], args[1], args[2]),
        abi::NR_OPEN => fs::sys_open(mem, fds, args[0], args[1]),
        abi::NR_OPENAT => fs::sys_openat(mem, fds, args[0], args[1], args[2]),
        abi::NR_CLOSE => fs::sys_close(fds, args[0]),
        abi::NR_LSEEK => fs::sys_lseek(fds, args[0], args[1], args[2]),
        abi::NR_FSTAT64 => fs::sys_fstat64(mem, fds, args[0], args[1]),
        abi::NR_IOCTL => err(abi::ENOTTY),

        // Memory.
        abi::NR_BRK => mem.set_brk(args[0]),
        abi::NR_MMAP2 => sys_mmap2(mem, &args),
        abi::NR_MUNMAP => {
            mem.unmap(args[0], args[1]);
            0
        }
        abi::NR_MPROTECT | abi::NR_MADVISE => 0,

        // Identity.
        abi::NR_GETPID | abi::NR_GETTID => abi::GUEST_ID,
        abi::NR_GETUID32 | abi::NR_GETEUID32 | abi::NR_GETGID32 | abi::NR_GETEGID32 => {
            abi::GUEST_ID
        }
        abi::NR_SET_TID_ADDRESS => abi::GUEST_ID,
        abi::NR_UNAME => sys_uname(mem, args[0]),

        // Time.
        abi::NR_TIME => sys_time(mem, args[0]),
        abi::NR_GETTIMEOFDAY => sys_gettimeofday(mem, args[0]),
        abi::NR_CLOCK_GETTIME => sys_clock_gettime(mem, process, args[0], args[1]),

        // Odds and ends real programs poke at startup.
        abi::NR_GETRANDOM => sys_getrandom(mem, process, args[0], args[1]),
        abi::NR_RT_SIGACTION | abi::NR_RT_SIGPROCMASK => 0, // no signal delivery
        abi::NR_FUTEX => match args[1] & 0x7F {
            abi::FUTEX_WAKE => 0,
            abi::FUTEX_WAIT => err(abi::EAGAIN), // single-threaded: never blocks
            _ => err(abi::ENOSYS),
        },
        abi::NR_UGETRLIMIT => sys_ugetrlimit(mem, args[1]),

        _ => err(abi::ENOSYS),
    };
    Control::Ret(ret)
}

fn sys_mmap2(mem: &mut AddressSpace, args: &[u32; 6]) -> u32 {
    let (addr, len, flags, fd) = (args[0], args[1], args[3], args[4]);
    if flags & abi::MAP_ANONYMOUS == 0 || fd != u32::MAX {
        return err(abi::ENODEV); // no file-backed mappings in v1
    }
    match mem.mmap(addr, len, flags & abi::MAP_FIXED != 0) {
        Ok(a) => a,
        Err(()) => err(abi::ENOMEM),
    }
}

fn sys_uname(mem: &mut AddressSpace, buf: u32) -> u32 {
    // struct utsname: six fixed 65-byte NUL-padded fields.
    let fields = ["Linux", "remu", "5.10.0", "#1", "i386", ""];
    let mut out = [0u8; 6 * 65];
    for (i, f) in fields.iter().enumerate() {
        out[i * 65..i * 65 + f.len()].copy_from_slice(f.as_bytes());
    }
    if mem.write_bytes(buf, &out).is_err() {
        return err(abi::EFAULT);
    }
    0
}

fn unix_now() -> (u32, u32) {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => (d.as_secs() as u32, d.subsec_nanos()),
        Err(_) => (0, 0),
    }
}

fn sys_time(mem: &mut AddressSpace, tloc: u32) -> u32 {
    let (secs, _) = unix_now();
    if tloc != 0 {
        mem.write_u32(tloc, secs);
    }
    secs
}

fn sys_gettimeofday(mem: &mut AddressSpace, tv: u32) -> u32 {
    if tv != 0 {
        let (secs, nanos) = unix_now();
        mem.write_u32(tv, secs);
        mem.write_u32(tv + 4, nanos / 1000);
    }
    0
}

fn sys_clock_gettime(mem: &mut AddressSpace, process: &Process, clock: u32, ts: u32) -> u32 {
    const CLOCK_MONOTONIC: u32 = 1;
    let (secs, nanos) = if clock == CLOCK_MONOTONIC {
        let d = process.start.elapsed();
        (d.as_secs() as u32, d.subsec_nanos())
    } else {
        unix_now()
    };
    mem.write_u32(ts, secs);
    mem.write_u32(ts + 4, nanos);
    0
}

fn sys_getrandom(mem: &mut AddressSpace, process: &mut Process, buf: u32, count: u32) -> u32 {
    let n = count.min(256); // getrandom(2) may return fewer bytes than asked
    let mut bytes = vec![0u8; n as usize];
    process.rand_bytes(&mut bytes);
    if mem.write_bytes(buf, &bytes).is_err() {
        return err(abi::EFAULT);
    }
    n
}

fn sys_ugetrlimit(mem: &mut AddressSpace, rlim: u32) -> u32 {
    // { rlim_cur, rlim_max } = RLIM_INFINITY for every resource.
    mem.write_u32(rlim, u32::MAX);
    mem.write_u32(rlim + 4, u32::MAX);
    0
}

fn syscall_name(nr: u32) -> &'static str {
    match nr {
        abi::NR_EXIT => "exit",
        abi::NR_READ => "read",
        abi::NR_WRITE => "write",
        abi::NR_OPEN => "open",
        abi::NR_CLOSE => "close",
        abi::NR_TIME => "time",
        abi::NR_LSEEK => "lseek",
        abi::NR_GETPID => "getpid",
        abi::NR_BRK => "brk",
        abi::NR_IOCTL => "ioctl",
        abi::NR_GETTIMEOFDAY => "gettimeofday",
        abi::NR_MUNMAP => "munmap",
        abi::NR_UNAME => "uname",
        abi::NR_MPROTECT => "mprotect",
        abi::NR_READV => "readv",
        abi::NR_WRITEV => "writev",
        abi::NR_RT_SIGACTION => "rt_sigaction",
        abi::NR_RT_SIGPROCMASK => "rt_sigprocmask",
        abi::NR_UGETRLIMIT => "ugetrlimit",
        abi::NR_MMAP2 => "mmap2",
        abi::NR_FSTAT64 => "fstat64",
        abi::NR_GETUID32 => "getuid32",
        abi::NR_GETGID32 => "getgid32",
        abi::NR_GETEUID32 => "geteuid32",
        abi::NR_GETEGID32 => "getegid32",
        abi::NR_MADVISE => "madvise",
        abi::NR_GETTID => "gettid",
        abi::NR_FUTEX => "futex",
        abi::NR_SET_THREAD_AREA => "set_thread_area",
        abi::NR_EXIT_GROUP => "exit_group",
        abi::NR_SET_TID_ADDRESS => "set_tid_address",
        abi::NR_CLOCK_GETTIME => "clock_gettime",
        abi::NR_OPENAT => "openat",
        abi::NR_GETRANDOM => "getrandom",
        _ => "sys_?",
    }
}
