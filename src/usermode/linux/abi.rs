//! Linux i386 ABI constants: syscall numbers (from
//! `arch/x86/entry/syscalls/syscall_32.tbl`), errno values, and auxiliary
//! vector tags. Defined here rather than taken from any host libc — the host
//! is not Linux.

// --- Syscall numbers ---------------------------------------------------------

pub const NR_EXIT: u32 = 1;
pub const NR_READ: u32 = 3;
pub const NR_WRITE: u32 = 4;
pub const NR_OPEN: u32 = 5;
pub const NR_CLOSE: u32 = 6;
pub const NR_TIME: u32 = 13;
pub const NR_LSEEK: u32 = 19;
pub const NR_GETPID: u32 = 20;
pub const NR_BRK: u32 = 45;
pub const NR_IOCTL: u32 = 54;
pub const NR_GETTIMEOFDAY: u32 = 78;
pub const NR_MUNMAP: u32 = 91;
pub const NR_UNAME: u32 = 122;
pub const NR_MPROTECT: u32 = 125;
pub const NR_READV: u32 = 145;
pub const NR_WRITEV: u32 = 146;
pub const NR_RT_SIGACTION: u32 = 174;
pub const NR_RT_SIGPROCMASK: u32 = 175;
pub const NR_UGETRLIMIT: u32 = 191;
pub const NR_MMAP2: u32 = 192;
pub const NR_FSTAT64: u32 = 197;
pub const NR_GETUID32: u32 = 199;
pub const NR_GETGID32: u32 = 200;
pub const NR_GETEUID32: u32 = 201;
pub const NR_GETEGID32: u32 = 202;
pub const NR_MADVISE: u32 = 219;
pub const NR_GETTID: u32 = 224;
pub const NR_FUTEX: u32 = 240;
pub const NR_SET_THREAD_AREA: u32 = 243;
pub const NR_EXIT_GROUP: u32 = 252;
pub const NR_SET_TID_ADDRESS: u32 = 258;
pub const NR_CLOCK_GETTIME: u32 = 265;
pub const NR_OPENAT: u32 = 295;
pub const NR_GETRANDOM: u32 = 355;

// --- Errno values --------------------------------------------------------------

pub const EPERM: u32 = 1;
pub const ENOENT: u32 = 2;
pub const ESRCH: u32 = 3;
pub const EINTR: u32 = 4;
pub const EIO: u32 = 5;
pub const EBADF: u32 = 9;
pub const EAGAIN: u32 = 11;
pub const ENOMEM: u32 = 12;
pub const EACCES: u32 = 13;
pub const EFAULT: u32 = 14;
pub const EEXIST: u32 = 17;
pub const ENODEV: u32 = 19;
pub const EINVAL: u32 = 22;
pub const ENOTTY: u32 = 25;
pub const ESPIPE: u32 = 29;
pub const ENOSYS: u32 = 38;

/// Encode a syscall error return (`-errno` in EAX).
#[inline]
pub fn err(errno: u32) -> u32 {
    errno.wrapping_neg()
}

// --- Auxiliary vector tags ------------------------------------------------------

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
pub const AT_HWCAP: u32 = 16;
pub const AT_CLKTCK: u32 = 17;
pub const AT_SECURE: u32 = 23;
pub const AT_RANDOM: u32 = 25;

// --- Misc ABI values -------------------------------------------------------------

/// `openat(2)` "relative to the current directory" sentinel.
pub const AT_FDCWD: u32 = (-100i32) as u32;
/// `mmap` flags used by the v1 implementation.
pub const MAP_FIXED: u32 = 0x10;
pub const MAP_ANONYMOUS: u32 = 0x20;
/// `futex(2)` ops (low bits, private flag masked off).
pub const FUTEX_WAIT: u32 = 0;
pub const FUTEX_WAKE: u32 = 1;
/// `lseek(2)` whence values.
pub const SEEK_SET: u32 = 0;
pub const SEEK_CUR: u32 = 1;
pub const SEEK_END: u32 = 2;
/// The identity every guest process reports (pid/tid/uid/gid).
pub const GUEST_ID: u32 = 1000;
