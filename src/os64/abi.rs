//! Linux x86-64 ABI constants: `SYSCALL` numbers (`asm/unistd_64.h`), `errno`
//! values (`asm-generic/errno.h`), ELF64 and auxiliary-vector tags, and
//! `mmap`/`open`/`arch_prctl` flags. Values are the architectural x86-64 ones,
//! defined here rather than taken from any host libc — the host is not Linux.

// --- Syscall numbers (x86-64 unistd_64, the subset we service) ---------------
pub mod sys {
    pub const READ: u64 = 0;
    pub const WRITE: u64 = 1;
    pub const OPEN: u64 = 2;
    pub const CLOSE: u64 = 3;
    pub const STAT: u64 = 4;
    pub const FSTAT: u64 = 5;
    pub const LSTAT: u64 = 6;
    pub const LSEEK: u64 = 8;
    pub const MMAP: u64 = 9;
    pub const MPROTECT: u64 = 10;
    pub const MUNMAP: u64 = 11;
    pub const BRK: u64 = 12;
    pub const RT_SIGACTION: u64 = 13;
    pub const RT_SIGPROCMASK: u64 = 14;
    pub const IOCTL: u64 = 16;
    pub const PREAD64: u64 = 17;
    pub const READV: u64 = 19;
    pub const WRITEV: u64 = 20;
    pub const ACCESS: u64 = 21;
    pub const SCHED_YIELD: u64 = 24;
    pub const MADVISE: u64 = 28;
    pub const DUP: u64 = 32;
    pub const NANOSLEEP: u64 = 35;
    pub const GETPID: u64 = 39;
    pub const EXIT: u64 = 60;
    pub const UNAME: u64 = 63;
    pub const FCNTL: u64 = 72;
    pub const GETCWD: u64 = 79;
    pub const READLINK: u64 = 89;
    pub const GETUID: u64 = 102;
    pub const GETGID: u64 = 104;
    pub const GETEUID: u64 = 107;
    pub const GETEGID: u64 = 108;
    pub const SIGALTSTACK: u64 = 131;
    pub const ARCH_PRCTL: u64 = 158;
    pub const GETTID: u64 = 186;
    pub const FUTEX: u64 = 202;
    pub const SET_TID_ADDRESS: u64 = 218;
    pub const CLOCK_GETTIME: u64 = 228;
    pub const EXIT_GROUP: u64 = 231;
    pub const OPENAT: u64 = 257;
    pub const NEWFSTATAT: u64 = 262;
    pub const SET_ROBUST_LIST: u64 = 273;
    pub const PRLIMIT64: u64 = 302;
    pub const GETRANDOM: u64 = 318;
    pub const RSEQ: u64 = 334;
}

// --- errno (returned as -errno in RAX) ---------------------------------------
pub mod errno {
    pub const EPERM: i64 = 1;
    pub const ENOENT: i64 = 2;
    pub const EIO: i64 = 5;
    pub const EBADF: i64 = 9;
    pub const EAGAIN: i64 = 11;
    pub const ENOMEM: i64 = 12;
    pub const EACCES: i64 = 13;
    pub const EFAULT: i64 = 14;
    pub const EEXIST: i64 = 17;
    pub const ENOTDIR: i64 = 20;
    pub const EINVAL: i64 = 22;
    pub const ENOTTY: i64 = 25;
    pub const ESPIPE: i64 = 29;
    pub const ERANGE: i64 = 34;
    pub const ENAMETOOLONG: i64 = 36;
    pub const ENOSYS: i64 = 38;
}

// --- ELF64 -------------------------------------------------------------------
pub mod elf {
    pub const MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
    pub const CLASS64: u8 = 2;
    pub const DATA_LE: u8 = 1;
    pub const ET_EXEC: u16 = 2;
    pub const ET_DYN: u16 = 3;
    pub const EM_X86_64: u16 = 62;

    pub const PT_LOAD: u32 = 1;
    pub const PT_INTERP: u32 = 3;
    pub const PT_PHDR: u32 = 6;
    pub const PT_GNU_STACK: u32 = 0x6474_E551;

    pub const PF_X: u32 = 1;
    pub const PF_W: u32 = 2;
    pub const PF_R: u32 = 4;
}

// --- Auxiliary vector tags ---------------------------------------------------
pub mod auxv {
    pub const AT_NULL: u64 = 0;
    pub const AT_PHDR: u64 = 3;
    pub const AT_PHENT: u64 = 4;
    pub const AT_PHNUM: u64 = 5;
    pub const AT_PAGESZ: u64 = 6;
    pub const AT_BASE: u64 = 7;
    pub const AT_FLAGS: u64 = 8;
    pub const AT_ENTRY: u64 = 9;
    pub const AT_UID: u64 = 11;
    pub const AT_EUID: u64 = 12;
    pub const AT_GID: u64 = 13;
    pub const AT_EGID: u64 = 14;
    pub const AT_PLATFORM: u64 = 15;
    pub const AT_HWCAP: u64 = 16;
    pub const AT_CLKTCK: u64 = 17;
    pub const AT_SECURE: u64 = 23;
    pub const AT_RANDOM: u64 = 25;
    pub const AT_EXECFN: u64 = 31;
}

// --- mmap / open flags -------------------------------------------------------
pub mod mmap {
    pub const PROT_READ: u64 = 1;
    pub const PROT_WRITE: u64 = 2;
    pub const PROT_EXEC: u64 = 4;
    pub const MAP_FIXED: u64 = 0x10;
    pub const MAP_ANONYMOUS: u64 = 0x20;
}

pub mod open {
    pub const O_WRONLY: u64 = 0o1;
    pub const O_RDWR: u64 = 0o2;
    /// `openat` special "relative to cwd" dirfd.
    pub const AT_FDCWD: i64 = -100;
}

// --- arch_prctl subcodes -----------------------------------------------------
pub mod prctl {
    pub const ARCH_SET_GS: u64 = 0x1001;
    pub const ARCH_SET_FS: u64 = 0x1002;
    pub const ARCH_GET_FS: u64 = 0x1003;
    pub const ARCH_GET_GS: u64 = 0x1004;
}
