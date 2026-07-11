//! Linux x86-64 syscall dispatch. The host hook delivers a trapped `SYSCALL`
//! here; we read the number/arguments from the registers (RAX; RDI RSI RDX R10
//! R8 R9), service the call against memory + the VFS, and write the result (or
//! `-errno`) to RAX.

use super::Emulator;
use crate::os64::abi::{errno, mmap, prctl, sys};
use crate::os64::memory::{PROT_EXEC, PROT_READ, PROT_WRITE};
use crate::x86_64::reg;

/// Bytes moved per host-side buffer chunk, so a guest-supplied length never
/// drives an unbounded host allocation.
const XFER_CHUNK: usize = 64 * 1024;

/// Syscall argument registers, in ABI order.
const ARG_REGS: [u8; 6] = [reg::RDI, reg::RSI, reg::RDX, reg::R10, reg::R8, reg::R9];

impl Emulator {
    // --- register / memory marshaling helpers -------------------------------

    fn arg(&self, i: usize) -> u64 {
        self.cpu.regs.gpr[ARG_REGS[i] as usize]
    }

    fn set_ret(&mut self, v: i64) {
        self.cpu.regs.gpr[reg::RAX as usize] = v as u64;
    }

    /// Read guest bytes; `false` on an inaccessible page.
    fn g_read(&self, lin: u64, buf: &mut [u8]) -> bool {
        self.aspace.read_bytes(&self.mem, lin, buf)
    }

    /// Write guest bytes; `false` on an inaccessible page.
    fn g_write(&mut self, lin: u64, data: &[u8]) -> bool {
        self.aspace.write_bytes(&mut self.mem, lin, data)
    }

    fn g_u64(&self, lin: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.g_read(lin, &mut b).then(|| u64::from_le_bytes(b))
    }

    fn g_string(&self, lin: u64) -> Option<String> {
        self.aspace
            .read_cstr(&self.mem, lin, 4096)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    }

    /// Copy up to `len` bytes from `fd` into guest memory at `lin`, in bounded
    /// chunks. Returns bytes moved or `-errno`.
    fn transfer_read(&mut self, fd: i64, lin: u64, len: usize) -> i64 {
        let mut buf = vec![0u8; XFER_CHUNK.min(len.max(1))];
        let (mut total, mut off, mut left) = (0i64, 0u64, len);
        while left > 0 {
            let n = left.min(XFER_CHUNK);
            let got = self.vfs.read(fd, &mut buf[..n]);
            if got < 0 {
                return if total > 0 { total } else { got };
            }
            if got == 0 {
                break;
            }
            if !self.g_write(lin + off, &buf[..got as usize]) {
                return if total > 0 { total } else { -errno::EFAULT };
            }
            total += got;
            off += got as u64;
            left -= got as usize;
            if (got as usize) < n {
                break;
            }
        }
        total
    }

    /// Copy up to `len` bytes from guest memory at `lin` to `fd`, in bounded
    /// chunks. Returns bytes moved or `-errno`.
    fn transfer_write(&mut self, fd: i64, lin: u64, len: usize) -> i64 {
        let mut buf = vec![0u8; XFER_CHUNK.min(len.max(1))];
        let (mut total, mut off, mut left) = (0i64, 0u64, len);
        while left > 0 {
            let n = left.min(XFER_CHUNK);
            if !self.g_read(lin + off, &mut buf[..n]) {
                return if total > 0 { total } else { -errno::EFAULT };
            }
            let w = self.vfs.write(fd, &buf[..n]);
            if w < 0 {
                return if total > 0 { total } else { w };
            }
            total += w;
            off += w as u64;
            left -= w as usize;
            if (w as usize) < n {
                break; // short write: stop here
            }
        }
        total
    }

    // --- dispatch -----------------------------------------------------------

    /// Service the pending trapped `SYSCALL`.
    pub(crate) fn dispatch_syscall(&mut self) {
        let nr = self.cpu.regs.gpr[reg::RAX as usize];
        let ret = self.syscall(nr);
        self.set_ret(ret);
    }

    fn syscall(&mut self, nr: u64) -> i64 {
        match nr {
            sys::EXIT | sys::EXIT_GROUP => {
                self.running = false;
                self.exit_code = (self.arg(0) & 0xFF) as i32;
                0
            }
            sys::READ => {
                let (fd, ptr, len) = (self.arg(0) as i64, self.arg(1), self.arg(2) as usize);
                self.transfer_read(fd, ptr, len)
            }
            sys::WRITE => {
                let (fd, ptr, len) = (self.arg(0) as i64, self.arg(1), self.arg(2) as usize);
                self.transfer_write(fd, ptr, len)
            }
            sys::READV => self.sys_iov(false),
            sys::WRITEV => self.sys_iov(true),
            sys::OPEN => {
                let (path, flags) = (self.arg(0), self.arg(1));
                self.do_open(path, flags)
            }
            sys::OPENAT => {
                // dirfd (arg0) honored only as AT_FDCWD; workloads use relative
                // or cwd paths.
                let (path, flags) = (self.arg(1), self.arg(2));
                self.do_open(path, flags)
            }
            sys::CLOSE => self.vfs.close(self.arg(0) as i64),
            sys::LSEEK => {
                let (fd, off, whence) = (self.arg(0) as i64, self.arg(1) as i64, self.arg(2));
                self.vfs.lseek(fd, off, whence)
            }
            sys::READLINK => {
                let (path, buf, size) = (self.arg(0), self.arg(1), self.arg(2));
                self.sys_readlink(path, buf, size)
            }
            sys::ACCESS => -errno::ENOENT, // no guest filesystem root yet
            sys::BRK => {
                let want = self.arg(0);
                self.aspace.set_brk(&mut self.mem, want) as i64
            }
            sys::MMAP => self.sys_mmap(),
            sys::MUNMAP => {
                let (addr, len) = (self.arg(0), self.arg(1));
                self.aspace.unmap(&mut self.mem, addr, len);
                self.cpu.invalidate_tlb();
                0
            }
            sys::MPROTECT => {
                let (addr, len, prot_in) = (self.arg(0), self.arg(1), self.arg(2));
                self.aspace.protect(&mut self.mem, addr, len, prot_bits(prot_in));
                self.cpu.invalidate_tlb();
                0
            }
            sys::MADVISE => 0,
            sys::FSTAT | sys::NEWFSTATAT => {
                // newfstatat(dirfd, path, statbuf, flags): stat the fd in arg0
                // when AT_EMPTY_PATH is used; otherwise fall through to the fd.
                let (fd, buf) = if nr == sys::FSTAT {
                    (self.arg(0) as i64, self.arg(1))
                } else {
                    (self.arg(0) as i64, self.arg(2))
                };
                self.sys_fstat(fd, buf)
            }
            sys::ARCH_PRCTL => self.sys_arch_prctl(),
            sys::UNAME => self.sys_uname(self.arg(0)),
            sys::GETRANDOM => self.sys_getrandom(),
            sys::GETCWD => self.sys_getcwd(),
            sys::IOCTL => self.sys_ioctl(),
            sys::FCNTL => 0,
            sys::DUP => -errno::EBADF,
            sys::SCHED_YIELD => 0,
            sys::NANOSLEEP => 0,

            // --- identity / signal / misc stubs ----------------------------
            sys::GETPID | sys::GETTID => 1,
            sys::GETUID | sys::GETEUID | sys::GETGID | sys::GETEGID => 0,
            sys::SET_TID_ADDRESS => {
                self.clear_tid = self.arg(0);
                1
            }
            sys::SET_ROBUST_LIST => 0,
            sys::RSEQ => -errno::ENOSYS, // glibc treats absence as "no rseq"
            sys::RT_SIGACTION | sys::RT_SIGPROCMASK | sys::SIGALTSTACK => 0,
            sys::PRLIMIT64 => self.sys_prlimit64(),
            sys::FUTEX => match self.arg(1) & 0x7F {
                1 => 0,             // FUTEX_WAKE
                0 => -errno::EAGAIN, // FUTEX_WAIT: single-threaded, never blocks
                _ => -errno::ENOSYS,
            },
            sys::CLOCK_GETTIME => {
                let ts = self.arg(1);
                self.g_write(ts, &[0u8; 16]); // zeroed timespec
                0
            }

            _ => {
                if self.trace {
                    eprintln!("[remu] unimplemented syscall {nr}");
                }
                -errno::ENOSYS
            }
        }
    }

    // --- individual syscalls ------------------------------------------------

    /// `readv`/`writev`: gather/scatter over an `iovec[]` (each entry is a
    /// 16-byte `{ base: u64, len: u64 }`).
    fn sys_iov(&mut self, write: bool) -> i64 {
        let (fd, iov, cnt) = (self.arg(0) as i64, self.arg(1), self.arg(2));
        let mut total = 0i64;
        for i in 0..cnt {
            let ent = iov + i * 16;
            let (Some(ptr), Some(len)) = (self.g_u64(ent), self.g_u64(ent + 8)) else {
                return if total > 0 { total } else { -errno::EFAULT };
            };
            if len == 0 {
                continue;
            }
            let n = if write {
                self.transfer_write(fd, ptr, len as usize)
            } else {
                self.transfer_read(fd, ptr, len as usize)
            };
            if n < 0 {
                return if total > 0 { total } else { n };
            }
            total += n;
            if (n as u64) < len {
                break; // short transfer: stop the gather/scatter
            }
        }
        total
    }

    fn do_open(&mut self, path: u64, flags: u64) -> i64 {
        match self.g_string(path) {
            Some(p) => self.vfs.open(&p, flags),
            None => -errno::EFAULT,
        }
    }

    fn sys_readlink(&mut self, path: u64, buf: u64, size: u64) -> i64 {
        let Some(_p) = self.g_string(path) else {
            return -errno::EFAULT;
        };
        // No procfs yet: report the link as absent.
        let _ = (buf, size);
        -errno::ENOENT
    }

    fn sys_mmap(&mut self) -> i64 {
        let (addr, len, prot_in, flags) = (self.arg(0), self.arg(1), self.arg(2), self.arg(3));
        let (fd, file_off) = (self.arg(4) as i64, self.arg(5));
        let mut prot = prot_bits(prot_in);
        if prot == 0 {
            prot = PROT_READ;
        }

        let base = if flags & mmap::MAP_FIXED != 0 && addr != 0 {
            self.aspace.mmap_fixed(&mut self.mem, addr, len, prot);
            addr
        } else {
            let b = self.aspace.mmap(&mut self.mem, len, prot);
            if b == 0 {
                return -errno::ENOMEM;
            }
            b
        };

        // File-backed: populate from the fd (needed for dynamic linking). The
        // transfer is chunked, so a large mapping does not allocate a big host
        // buffer.
        if flags & mmap::MAP_ANONYMOUS == 0 && fd >= 0 {
            let saved = self.vfs.lseek(fd, 0, 1);
            self.vfs.lseek(fd, file_off as i64, 0);
            self.transfer_read(fd, base, len as usize);
            if saved >= 0 {
                self.vfs.lseek(fd, saved, 0);
            }
        }
        base as i64
    }

    fn sys_fstat(&mut self, fd: i64, buf: u64) -> i64 {
        const S_IFREG: u32 = 0o100000;
        const S_IFCHR: u32 = 0o020000;
        let (mode, size) = if self.vfs.is_tty(fd) {
            (S_IFCHR | 0o620, 0)
        } else if let Some(len) = self.vfs.file_len(fd) {
            (S_IFREG | 0o444, len)
        } else if self.vfs.is_open(fd) {
            (S_IFREG | 0o444, 0)
        } else {
            return -errno::EBADF;
        };
        let st = build_stat(mode, size, fd as u64 + 1);
        if !self.g_write(buf, &st) {
            return -errno::EFAULT;
        }
        0
    }

    fn sys_arch_prctl(&mut self) -> i64 {
        let (code, addr) = (self.arg(0), self.arg(1));
        match code {
            prctl::ARCH_SET_FS => {
                self.cpu.regs.seg[reg::FS as usize].base = addr;
                0
            }
            prctl::ARCH_SET_GS => {
                self.cpu.regs.seg[reg::GS as usize].base = addr;
                0
            }
            prctl::ARCH_GET_FS => {
                let v = self.cpu.regs.seg[reg::FS as usize].base;
                if self.g_write(addr, &v.to_le_bytes()) { 0 } else { -errno::EFAULT }
            }
            prctl::ARCH_GET_GS => {
                let v = self.cpu.regs.seg[reg::GS as usize].base;
                if self.g_write(addr, &v.to_le_bytes()) { 0 } else { -errno::EFAULT }
            }
            _ => -errno::EINVAL,
        }
    }

    fn sys_uname(&mut self, buf: u64) -> i64 {
        // struct utsname: 6 fields × 65 bytes.
        let fields = ["Linux", "remu", "6.1.0-remu", "#1 SMP", "x86_64", "(none)"];
        let mut out = vec![0u8; 65 * 6];
        for (i, f) in fields.iter().enumerate() {
            let b = f.as_bytes();
            out[i * 65..i * 65 + b.len()].copy_from_slice(b);
        }
        if self.g_write(buf, &out) { 0 } else { -errno::EFAULT }
    }

    fn sys_getrandom(&mut self) -> i64 {
        let (buf, len) = (self.arg(0), self.arg(1) as usize);
        // Deterministic PRNG (not cryptographic), emitted in bounded chunks.
        let mut x = 0x1234_5678_9abc_def0u64 ^ self.cpu.cycles;
        let mut chunk = vec![0u8; XFER_CHUNK.min(len.max(1))];
        let mut off = 0usize;
        while off < len {
            let n = (len - off).min(XFER_CHUNK);
            for b in chunk[..n].iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *b = x as u8;
            }
            if !self.g_write(buf + off as u64, &chunk[..n]) {
                return -errno::EFAULT;
            }
            off += n;
        }
        len as i64
    }

    fn sys_getcwd(&mut self) -> i64 {
        let (buf, size) = (self.arg(0), self.arg(1) as usize);
        let cwd = b"/\0";
        if cwd.len() > size {
            return -errno::ERANGE;
        }
        if !self.g_write(buf, cwd) {
            return -errno::EFAULT;
        }
        cwd.len() as i64
    }

    fn sys_ioctl(&mut self) -> i64 {
        let (fd, req) = (self.arg(0) as i64, self.arg(1));
        const TCGETS: u64 = 0x5401;
        const TIOCGWINSZ: u64 = 0x5413;
        if self.vfs.is_tty(fd) && matches!(req, TCGETS | TIOCGWINSZ) {
            0
        } else if self.vfs.is_open(fd) {
            -errno::ENOTTY
        } else {
            -errno::EBADF
        }
    }

    fn sys_prlimit64(&mut self) -> i64 {
        // prlimit64(pid, resource, new_limit, old_limit): report RLIM_INFINITY.
        let old = self.arg(3);
        if old != 0 {
            let inf = u64::MAX;
            let mut buf = [0u8; 16];
            buf[0..8].copy_from_slice(&inf.to_le_bytes()); // rlim_cur
            buf[8..16].copy_from_slice(&inf.to_le_bytes()); // rlim_max
            if !self.g_write(old, &buf) {
                return -errno::EFAULT;
            }
        }
        0
    }
}

/// Translate Linux `PROT_*` mmap bits into the memory layer's flags.
fn prot_bits(p: u64) -> u32 {
    let mut prot = 0;
    if p & mmap::PROT_READ != 0 {
        prot |= PROT_READ;
    }
    if p & mmap::PROT_WRITE != 0 {
        prot |= PROT_WRITE;
    }
    if p & mmap::PROT_EXEC != 0 {
        prot |= PROT_EXEC;
    }
    prot
}

/// Build an x86-64 `struct stat` (144 bytes) with the fields that matter.
fn build_stat(mode: u32, size: u64, ino: u64) -> [u8; 144] {
    let mut s = [0u8; 144];
    let put32 = |s: &mut [u8; 144], off: usize, v: u32| s[off..off + 4].copy_from_slice(&v.to_le_bytes());
    let put64 = |s: &mut [u8; 144], off: usize, v: u64| s[off..off + 8].copy_from_slice(&v.to_le_bytes());
    put64(&mut s, 8, ino); // st_ino
    put64(&mut s, 16, 1); // st_nlink
    put32(&mut s, 24, mode); // st_mode
    put64(&mut s, 48, size); // st_size
    put64(&mut s, 56, 4096); // st_blksize
    put64(&mut s, 64, size.div_ceil(512)); // st_blocks
    s
}
