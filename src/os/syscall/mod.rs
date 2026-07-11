//! Linux i386 syscall dispatch. The A1 host hook delivers `int 0x80` here; we
//! read the number/arguments from the registers (via the arch seam), service
//! the call against memory + the VFS, and write the result (or `-errno`) to
//! `EAX`.

use super::Emulator;
use crate::os::abi::{errno, gdt, mmap, sys};
use crate::os::arch::{I386, TargetArch};
use crate::os::memory::{PROT_EXEC, PROT_READ, PROT_WRITE};

/// Bytes moved per host-side buffer chunk. Transfers loop over this so a
/// guest-supplied length never drives an unbounded host allocation.
const XFER_CHUNK: usize = 64 * 1024;

/// `S_IF*` file-type bits.
mod mode {
    pub const REG: u32 = 0o100000;
    pub const DIR: u32 = 0o040000;
    pub const CHR: u32 = 0o020000;
}

impl Emulator {
    // --- register / memory marshaling helpers -------------------------------

    fn arg(&self, i: usize) -> u32 {
        I386::syscall_arg(&self.cpu, i)
    }

    fn set_ret(&mut self, v: i32) {
        I386::set_syscall_ret(&mut self.cpu, v as u32);
    }

    /// Read guest bytes; `false` on an inaccessible page.
    fn g_read(&self, lin: u32, buf: &mut [u8]) -> bool {
        self.aspace.read_bytes(&self.mem, lin, buf)
    }

    /// Write guest bytes; `false` on an inaccessible page.
    fn g_write(&mut self, lin: u32, data: &[u8]) -> bool {
        self.aspace.write_bytes(&mut self.mem, lin, data)
    }

    fn g_u32(&self, lin: u32) -> Option<u32> {
        let mut b = [0u8; 4];
        self.g_read(lin, &mut b).then(|| u32::from_le_bytes(b))
    }

    fn g_cstr(&self, lin: u32, max: usize) -> Option<Vec<u8>> {
        self.aspace.read_cstr(&self.mem, lin, max)
    }

    fn g_string(&self, lin: u32) -> Option<String> {
        self.g_cstr(lin, 4096)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
    }

    /// Copy up to `len` bytes from `fd` into guest memory at `lin`, in bounded
    /// chunks. Returns bytes moved or `-errno`.
    fn transfer_read(&mut self, fd: i32, lin: u32, len: usize) -> i32 {
        let mut buf = vec![0u8; XFER_CHUNK.min(len.max(1))];
        let (mut total, mut off, mut left) = (0i32, 0u32, len);
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
            off += got as u32;
            left -= got as usize;
            if (got as usize) < n {
                break;
            }
        }
        total
    }

    /// Copy up to `len` bytes from guest memory at `lin` to `fd`, in bounded
    /// chunks. Returns bytes moved or `-errno`.
    fn transfer_write(&mut self, fd: i32, lin: u32, len: usize) -> i32 {
        let mut buf = vec![0u8; XFER_CHUNK.min(len.max(1))];
        let (mut total, mut off, mut left) = (0i32, 0u32, len);
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
            off += w as u32;
            left -= w as usize;
            if (w as usize) < n {
                break; // short write: stop here
            }
        }
        total
    }

    // --- dispatch -----------------------------------------------------------

    /// Service the pending `int 0x80` syscall.
    pub(crate) fn dispatch_syscall(&mut self) {
        let nr = I386::syscall_nr(&self.cpu);
        let ret = self.syscall(nr);
        self.set_ret(ret);
    }

    fn syscall(&mut self, nr: u32) -> i32 {
        match nr {
            sys::EXIT | sys::EXIT_GROUP => {
                self.running = false;
                self.exit_code = (self.arg(0) & 0xFF) as i32;
                0
            }
            sys::READ => self.sys_read(),
            sys::WRITE => self.sys_write(),
            sys::WRITEV => self.sys_writev(),
            sys::OPEN => {
                let path = self.arg(0);
                let flags = self.arg(1);
                let mode = self.arg(2);
                self.do_open(path, flags, mode)
            }
            sys::OPENAT => {
                // dirfd in arg0 ignored except AT_FDCWD (paths are absolute or
                // cwd-relative for the workloads we target).
                let path = self.arg(1);
                let flags = self.arg(2);
                let mode = self.arg(3);
                self.do_open(path, flags, mode)
            }
            sys::CLOSE => self.vfs.close(self.arg(0) as i32),
            sys::LSEEK => {
                let fd = self.arg(0) as i32;
                let off = self.arg(1) as i32 as i64;
                let whence = self.arg(2);
                self.vfs.lseek(fd, off, whence) as i32
            }
            sys::_LLSEEK => self.sys_llseek(),
            sys::READLINK => {
                let path = self.arg(0);
                let buf = self.arg(1);
                let size = self.arg(2);
                self.sys_readlink(path, buf, size)
            }
            sys::ACCESS => match self.g_string(self.arg(0)) {
                Some(p) => self.vfs.access(&p),
                None => -errno::EFAULT,
            },
            sys::BRK => {
                let want = self.arg(0);
                self.aspace.set_brk(&mut self.mem, want) as i32
            }
            sys::MMAP2 => self.sys_mmap(true),
            sys::MMAP => self.sys_mmap(false),
            sys::MUNMAP => {
                let addr = self.arg(0);
                let len = self.arg(1);
                self.aspace.unmap(&mut self.mem, addr, len);
                self.cpu.invalidate_tlb();
                0
            }
            sys::MPROTECT => {
                let addr = self.arg(0);
                let len = self.arg(1);
                let prot = self.arg(2);
                self.aspace.protect(&mut self.mem, addr, len, prot);
                self.cpu.invalidate_tlb();
                0
            }
            sys::FSTAT64 => {
                let fd = self.arg(0) as i32;
                let buf = self.arg(1);
                self.sys_fstat64(fd, buf)
            }
            sys::STAT64 | sys::LSTAT64 => {
                let path = self.arg(0);
                let buf = self.arg(1);
                self.sys_stat64(path, buf)
            }
            sys::GETDENTS64 => self.sys_getdents64(),
            sys::IOCTL => self.sys_ioctl(),
            sys::FCNTL64 => 0, // F_GETFD/F_SETFD etc. → success (no real semantics yet)
            sys::UNAME => self.sys_uname(self.arg(0)),
            sys::SET_THREAD_AREA => self.sys_set_thread_area(),
            sys::GETRANDOM => self.sys_getrandom(),
            sys::GETCWD => self.sys_getcwd(),

            // --- identity / misc stubs -------------------------------------
            sys::GETPID | sys::GETTID => 1,
            sys::GETPPID => 0,
            sys::GETUID32 | sys::GETEUID32 | sys::GETGID32 | sys::GETEGID32 => 0,
            sys::SET_TID_ADDRESS => 1,
            sys::RT_SIGACTION | sys::RT_SIGPROCMASK => 0,
            sys::CLOCK_GETTIME => {
                // Zeroed timespec (arg1) → time 0; enough to not fault.
                let ts = self.arg(1);
                self.g_write(ts, &[0u8; 8]);
                0
            }
            sys::ARCH_PRCTL => -errno::ENOSYS,

            _ => {
                if self.trace {
                    eprintln!("[remu] unimplemented syscall {nr}");
                }
                -errno::ENOSYS
            }
        }
    }

    // --- individual syscalls ------------------------------------------------

    fn sys_read(&mut self) -> i32 {
        let fd = self.arg(0) as i32;
        let ptr = self.arg(1);
        let len = self.arg(2) as usize;
        self.transfer_read(fd, ptr, len)
    }

    fn sys_write(&mut self) -> i32 {
        let fd = self.arg(0) as i32;
        let ptr = self.arg(1);
        let len = self.arg(2) as usize;
        self.transfer_write(fd, ptr, len)
    }

    fn sys_writev(&mut self) -> i32 {
        let fd = self.arg(0) as i32;
        let iov = self.arg(1);
        let cnt = self.arg(2);
        let mut total = 0i32;
        for i in 0..cnt {
            let base = iov + i * 8;
            let Some(ptr) = self.g_u32(base) else {
                return if total > 0 { total } else { -errno::EFAULT };
            };
            let Some(len) = self.g_u32(base + 4) else {
                return if total > 0 { total } else { -errno::EFAULT };
            };
            if len == 0 {
                continue;
            }
            let n = self.transfer_write(fd, ptr, len as usize);
            if n < 0 {
                return if total > 0 { total } else { n };
            }
            total += n;
            if (n as u32) < len {
                break; // short write: stop the gather
            }
        }
        total
    }

    fn do_open(&mut self, path: u32, flags: u32, mode: u32) -> i32 {
        match self.g_string(path) {
            Some(p) => self.vfs.open(&p, flags, mode),
            None => -errno::EFAULT,
        }
    }

    fn sys_llseek(&mut self) -> i32 {
        // _llseek(fd, off_hi, off_lo, result*, whence)
        let fd = self.arg(0) as i32;
        let off = ((self.arg(1) as u64) << 32 | self.arg(2) as u64) as i64;
        let result = self.arg(3);
        let whence = self.arg(4);
        let pos = self.vfs.lseek(fd, off, whence);
        if pos < 0 {
            return pos as i32;
        }
        if !self.g_write(result, &(pos as u64).to_le_bytes()) {
            return -errno::EFAULT;
        }
        0
    }

    fn sys_readlink(&mut self, path: u32, buf: u32, size: u32) -> i32 {
        let Some(p) = self.g_string(path) else {
            return -errno::EFAULT;
        };
        let Some(target) = self.vfs.readlink(&p) else {
            return -errno::ENOENT;
        };
        let bytes = target.as_bytes();
        let n = bytes.len().min(size as usize);
        if !self.g_write(buf, &bytes[..n]) {
            return -errno::EFAULT;
        }
        n as i32
    }

    fn sys_mmap(&mut self, mmap2: bool) -> i32 {
        let addr = self.arg(0);
        let len = self.arg(1);
        let prot_in = self.arg(2);
        let flags = self.arg(3);
        let fd = self.arg(4) as i32;
        let off_units = self.arg(5);
        let file_off = if mmap2 {
            off_units as u64 * 4096
        } else {
            off_units as u64
        };

        let mut prot = 0;
        if prot_in & mmap::PROT_READ != 0 {
            prot |= PROT_READ;
        }
        if prot_in & mmap::PROT_WRITE != 0 {
            prot |= PROT_WRITE;
        }
        if prot_in & mmap::PROT_EXEC != 0 {
            prot |= PROT_EXEC;
        }
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
        base as i32
    }

    fn sys_fstat64(&mut self, fd: i32, buf: u32) -> i32 {
        let (m, size) = if self.vfs.is_dir_fd(fd) {
            (mode::DIR | 0o755, 0)
        } else if self.vfs.is_tty(fd) {
            (mode::CHR | 0o666, 0)
        } else if let Some(len) = self.vfs.file_len(fd) {
            (mode::REG | 0o644, len)
        } else {
            return -errno::EBADF;
        };
        let st = build_stat64(m, size, 1);
        if !self.g_write(buf, &st) {
            return -errno::EFAULT;
        }
        0
    }

    fn sys_stat64(&mut self, path: u32, buf: u32) -> i32 {
        let Some(p) = self.g_string(path) else {
            return -errno::EFAULT;
        };
        // Open (read-only) to probe, then stat that fd.
        let fd = self.vfs.open(&p, 0, 0);
        if fd < 0 {
            return fd;
        }
        let r = self.sys_fstat64(fd, buf);
        self.vfs.close(fd);
        r
    }

    fn sys_getdents64(&mut self) -> i32 {
        let fd = self.arg(0) as i32;
        let buf = self.arg(1);
        let cap = self.arg(2) as usize;
        // struct linux_dirent64 { u64 d_ino; s64 d_off; u16 d_reclen; u8 d_type;
        //                         char d_name[]; } — name NUL-terminated.
        let mut block: Vec<u8> = Vec::new();
        let ok = self.vfs.next_dents(fd, |name, is_dir| {
            let reclen = (19 + name.len() + 1 + 7) & !7; // align to 8
            if block.len() + reclen > cap {
                return false;
            }
            let start = block.len();
            block.resize(start + reclen, 0);
            let rec = &mut block[start..start + reclen];
            rec[0..8].copy_from_slice(&1u64.to_le_bytes()); // d_ino
            rec[8..16].copy_from_slice(&((start + reclen) as u64).to_le_bytes()); // d_off cookie
            rec[16..18].copy_from_slice(&(reclen as u16).to_le_bytes()); // d_reclen
            rec[18] = if is_dir { 4 } else { 8 }; // DT_DIR / DT_REG
            rec[19..19 + name.len()].copy_from_slice(name.as_bytes());
            true
        });
        if ok.is_none() {
            return -errno::ENOTDIR;
        }
        if block.is_empty() {
            return 0;
        }
        if !self.g_write(buf, &block) {
            return -errno::EFAULT;
        }
        block.len() as i32
    }

    fn sys_ioctl(&mut self) -> i32 {
        let fd = self.arg(0) as i32;
        let req = self.arg(1);
        // TCGETS / TIOCGWINSZ on a tty → success (isatty probes this).
        const TCGETS: u32 = 0x5401;
        const TIOCGWINSZ: u32 = 0x5413;
        if self.vfs.is_tty(fd) && matches!(req, TCGETS | TIOCGWINSZ) {
            0
        } else if self.vfs.get(fd).is_none() {
            -errno::EBADF
        } else {
            -errno::ENOTTY
        }
    }

    fn sys_uname(&mut self, buf: u32) -> i32 {
        // struct utsname: 6 fields × 65 bytes.
        let fields = ["Linux", "remu", "5.15.0-remu", "#1 SMP", "i686", "(none)"];
        let mut out = vec![0u8; 65 * 6];
        for (i, f) in fields.iter().enumerate() {
            let b = f.as_bytes();
            out[i * 65..i * 65 + b.len()].copy_from_slice(b);
        }
        if !self.g_write(buf, &out) {
            return -errno::EFAULT;
        }
        0
    }

    fn sys_set_thread_area(&mut self) -> i32 {
        let uinfo = self.arg(0);
        let Some(entry_number) = self.g_u32(uinfo) else {
            return -errno::EFAULT;
        };
        let base = self.g_u32(uinfo + 4).unwrap_or(0);
        let limit = self.g_u32(uinfo + 8).unwrap_or(0);
        let flags = self.g_u32(uinfo + 12).unwrap_or(0);

        let entry = if entry_number == 0xFFFF_FFFF {
            if self.tls_next >= gdt::TLS_MIN + gdt::TLS_COUNT {
                return -errno::EINVAL; // out of TLS slots
            }
            let e = self.tls_next;
            self.tls_next += 1;
            e
        } else if (entry_number as u16) < gdt::ENTRIES {
            entry_number as u16
        } else {
            return -errno::EINVAL;
        };
        let limit_in_pages = flags & (1 << 4) != 0;
        let writable = flags & (1 << 3) == 0; // read_exec_only clear → writable
        crate::os::arch::i386::write_tls(
            &self.aspace,
            &mut self.mem,
            entry,
            base,
            limit,
            limit_in_pages,
            writable,
        );
        // Return the chosen entry number to the guest.
        if !self.g_write(uinfo, &(entry as u32).to_le_bytes()) {
            return -errno::EFAULT;
        }
        0
    }

    fn sys_getrandom(&mut self) -> i32 {
        let buf = self.arg(0);
        let len = self.arg(1) as usize;
        // Deterministic PRNG (not cryptographic), emitted in bounded chunks.
        let mut x = 0x1234_5678u32 ^ (self.cpu.cycles as u32);
        let mut chunk = vec![0u8; XFER_CHUNK.min(len.max(1))];
        let mut off = 0usize;
        while off < len {
            let n = (len - off).min(XFER_CHUNK);
            for b in chunk[..n].iter_mut() {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                *b = x as u8;
            }
            if !self.g_write(buf + off as u32, &chunk[..n]) {
                return -errno::EFAULT;
            }
            off += n;
        }
        len as i32
    }

    fn sys_getcwd(&mut self) -> i32 {
        let buf = self.arg(0);
        let size = self.arg(1) as usize;
        let mut cwd = self.vfs.cwd().as_bytes().to_vec();
        cwd.push(0);
        if cwd.len() > size {
            return -errno::ERANGE;
        }
        if !self.g_write(buf, &cwd) {
            return -errno::EFAULT;
        }
        cwd.len() as i32
    }
}

/// Build an i386 `struct stat64` (96 bytes) with the fields that matter.
fn build_stat64(mode: u32, size: u64, ino: u64) -> [u8; 96] {
    let mut s = [0u8; 96];
    let put32 = |s: &mut [u8; 96], off: usize, v: u32| s[off..off + 4].copy_from_slice(&v.to_le_bytes());
    let put64 = |s: &mut [u8; 96], off: usize, v: u64| s[off..off + 8].copy_from_slice(&v.to_le_bytes());
    put32(&mut s, 12, ino as u32); // __st_ino
    put32(&mut s, 16, mode); // st_mode
    put32(&mut s, 20, 1); // st_nlink
    put64(&mut s, 44, size); // st_size (unaligned 8 bytes, per the i386 layout)
    put32(&mut s, 52, 4096); // st_blksize
    put64(&mut s, 56, size.div_ceil(512)); // st_blocks
    put64(&mut s, 88, ino); // st_ino
    s
}
