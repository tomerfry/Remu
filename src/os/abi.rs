//! Linux i386 ABI constants: `int 0x80` syscall numbers, `errno` values, ELF
//! and auxiliary-vector tags, and `mmap`/`open` flags. Values are the
//! architectural i386 ones (`asm/unistd_32.h`, `asm-generic/errno.h`).

// --- Syscall numbers (i386 unistd_32, the subset we service) -----------------
pub mod sys {
    pub const EXIT: u32 = 1;
    pub const READ: u32 = 3;
    pub const WRITE: u32 = 4;
    pub const OPEN: u32 = 5;
    pub const CLOSE: u32 = 6;
    pub const UNLINK: u32 = 10;
    pub const LSEEK: u32 = 19;
    pub const GETPID: u32 = 20;
    pub const ACCESS: u32 = 33;
    pub const BRK: u32 = 45;
    pub const IOCTL: u32 = 54;
    pub const GETPPID: u32 = 64;
    pub const MUNMAP: u32 = 91;
    pub const READLINK: u32 = 85;
    pub const MMAP: u32 = 90; // old mmap (struct arg); we mainly use MMAP2
    pub const MPROTECT: u32 = 125;
    pub const WRITEV: u32 = 146;
    pub const _LLSEEK: u32 = 140;
    pub const GETCWD: u32 = 183;
    pub const MMAP2: u32 = 192;
    pub const STAT64: u32 = 195;
    pub const LSTAT64: u32 = 196;
    pub const FSTAT64: u32 = 197;
    pub const GETUID32: u32 = 199;
    pub const GETGID32: u32 = 200;
    pub const GETEUID32: u32 = 201;
    pub const GETEGID32: u32 = 202;
    pub const GETDENTS64: u32 = 220;
    pub const FCNTL64: u32 = 221;
    pub const GETTID: u32 = 224;
    pub const SET_THREAD_AREA: u32 = 243;
    pub const EXIT_GROUP: u32 = 252;
    pub const SET_TID_ADDRESS: u32 = 258;
    pub const CLOCK_GETTIME: u32 = 265;
    pub const OPENAT: u32 = 295;
    pub const GETRANDOM: u32 = 355;
    pub const UNAME: u32 = 122;
    pub const RT_SIGACTION: u32 = 174;
    pub const RT_SIGPROCMASK: u32 = 175;
    pub const ARCH_PRCTL: u32 = 384; // not on i386 in practice; stubbed
}

// --- errno (returned as -errno in EAX) ---------------------------------------
pub mod errno {
    pub const EPERM: i32 = 1;
    pub const ENOENT: i32 = 2;
    pub const EIO: i32 = 5;
    pub const EBADF: i32 = 9;
    pub const ENOSPC: i32 = 28;
    pub const ESPIPE: i32 = 29;
    pub const ENOMEM: i32 = 12;
    pub const EACCES: i32 = 13;
    pub const EFAULT: i32 = 14;
    pub const EEXIST: i32 = 17;
    pub const ENOTDIR: i32 = 20;
    pub const EISDIR: i32 = 21;
    pub const EINVAL: i32 = 22;
    pub const ENOSYS: i32 = 38;
    pub const ENOTTY: i32 = 25;
    pub const ERANGE: i32 = 34;
    pub const ENAMETOOLONG: i32 = 36;
}

// --- ELF32 -------------------------------------------------------------------
pub mod elf {
    pub const MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
    pub const CLASS32: u8 = 1;
    pub const DATA_LE: u8 = 1;
    pub const ET_EXEC: u16 = 2;
    pub const ET_DYN: u16 = 3;
    pub const EM_386: u16 = 3;

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
    pub const AT_NULL: u32 = 0;
    pub const AT_PHDR: u32 = 3;
    pub const AT_PHENT: u32 = 4;
    pub const AT_PHNUM: u32 = 5;
    pub const AT_PAGESZ: u32 = 6;
    pub const AT_BASE: u32 = 7;
    pub const AT_FLAGS: u32 = 8;
    pub const AT_ENTRY: u32 = 9;
    pub const AT_UID: u32 = 11;
    pub const AT_EUID: u32 = 12;
    pub const AT_GID: u32 = 13;
    pub const AT_EGID: u32 = 14;
    pub const AT_PLATFORM: u32 = 15;
    pub const AT_HWCAP: u32 = 16;
    pub const AT_CLKTCK: u32 = 17;
    pub const AT_SECURE: u32 = 23;
    pub const AT_RANDOM: u32 = 25;
    pub const AT_EXECFN: u32 = 31;
    pub const AT_SYSINFO: u32 = 32;
    pub const AT_SYSINFO_EHDR: u32 = 33;
}

// --- mmap / open flags -------------------------------------------------------
pub mod mmap {
    pub const PROT_READ: u32 = 1;
    pub const PROT_WRITE: u32 = 2;
    pub const PROT_EXEC: u32 = 4;
    pub const MAP_FIXED: u32 = 0x10;
    pub const MAP_ANONYMOUS: u32 = 0x20;
}

pub mod open {
    pub const O_WRONLY: u32 = 0o1;
    pub const O_RDWR: u32 = 0o2;
    pub const O_CREAT: u32 = 0o100;
    pub const O_TRUNC: u32 = 0o1000;
    pub const O_APPEND: u32 = 0o2000;
    pub const O_DIRECTORY: u32 = 0o200000;
    /// `openat` special "relative to cwd" dirfd.
    pub const AT_FDCWD: i32 = -100;
}

// --- GDT layout (see `arch::i386`) -------------------------------------------
pub mod gdt {
    /// Ring-3 flat code selector (`CS`).
    pub const USER_CS: u16 = 0x18 | 3;
    /// Ring-3 flat data selector (`DS`/`ES`/`SS`/`FS`/`GS`).
    pub const USER_DS: u16 = 0x20 | 3;
    /// First GDT entry available to `set_thread_area` for TLS.
    pub const TLS_MIN: u16 = 6;
    /// Number of TLS entries.
    pub const TLS_COUNT: u16 = 3;
    /// Total GDT entries (null, 2 kernel, 2 user, then TLS).
    pub const ENTRIES: u16 = TLS_MIN + TLS_COUNT;
    /// Linear base of the GDT (a supervisor page near the top of the space).
    pub const BASE: u32 = 0xFFFF_E000;
}
