//! File descriptors and the file/IO syscalls.
//!
//! Fds 0-2 map to the host's standard streams (raw bytes, no newline
//! translation). `open`/`openat` serve read-only host files resolved against
//! the host working directory. [`Fd::Sink`] captures output in memory so
//! tests never touch the real console.

use std::io::{Read, Seek, SeekFrom, Write};

use super::super::addr_space::AddressSpace;
use super::abi::{self, err};

const IO_CHUNK: usize = 64 * 1024;
/// Linux `UIO_MAXIOV`.
const MAX_IOV: u32 = 1024;

/// One open file description.
pub enum Fd {
    Stdin,
    Stdout,
    Stderr,
    File(std::fs::File),
    /// In-memory output capture (tests).
    Sink(Vec<u8>),
}

impl Fd {
    /// Write, mapping host errors to guest errnos.
    fn write(&mut self, buf: &[u8]) -> Result<usize, u32> {
        match self {
            Fd::Stdin => Err(abi::EBADF),
            Fd::Stdout => std::io::stdout().write(buf).map_err(|_| abi::EIO),
            Fd::Stderr => std::io::stderr().write(buf).map_err(|_| abi::EIO),
            Fd::File(f) => f.write(buf).map_err(|_| abi::EIO),
            Fd::Sink(v) => {
                v.extend_from_slice(buf);
                Ok(buf.len())
            }
        }
    }

    /// Read, mapping host errors to guest errnos.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, u32> {
        match self {
            Fd::Stdin => std::io::stdin().read(buf).map_err(|_| abi::EIO),
            Fd::Stdout | Fd::Stderr => Err(abi::EBADF),
            Fd::File(f) => f.read(buf).map_err(|_| abi::EIO),
            Fd::Sink(_) => Ok(0),
        }
    }
}

/// The process fd table; slots 0-2 start as the standard streams.
pub struct FdTable {
    fds: Vec<Option<Fd>>,
}

impl FdTable {
    pub fn new() -> Self {
        FdTable {
            fds: vec![Some(Fd::Stdin), Some(Fd::Stdout), Some(Fd::Stderr)],
        }
    }

    pub fn get(&mut self, fd: u32) -> Option<&mut Fd> {
        self.fds.get_mut(fd as usize)?.as_mut()
    }

    /// Place `f` in the lowest free slot; returns its number.
    pub fn insert(&mut self, f: Fd) -> u32 {
        for (i, slot) in self.fds.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(f);
                return i as u32;
            }
        }
        self.fds.push(Some(f));
        (self.fds.len() - 1) as u32
    }

    /// Replace whatever occupies `fd` (tests: redirect stdout to a Sink).
    pub fn install(&mut self, fd: u32, f: Fd) {
        let i = fd as usize;
        if i >= self.fds.len() {
            self.fds.resize_with(i + 1, || None);
        }
        self.fds[i] = Some(f);
    }

    pub fn close(&mut self, fd: u32) -> bool {
        match self.fds.get_mut(fd as usize) {
            Some(slot @ Some(_)) => {
                *slot = None;
                true
            }
            _ => false,
        }
    }

    /// The bytes captured by a [`Fd::Sink`] at `fd`, if that's what lives there.
    pub fn sink_data(&self, fd: u32) -> Option<&[u8]> {
        match self.fds.get(fd as usize)? {
            Some(Fd::Sink(v)) => Some(v),
            _ => None,
        }
    }
}

impl Default for FdTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Write `count` guest bytes at `buf` to `f`, chunked. `Err` is an errno;
/// short writes return the partial count like the kernel does.
fn write_from_guest(mem: &mut AddressSpace, f: &mut Fd, buf: u32, count: u32) -> Result<u32, u32> {
    let mut tmp = [0u8; IO_CHUNK];
    let mut total = 0u32;
    while total < count {
        let n = ((count - total) as usize).min(IO_CHUNK);
        if mem
            .read_bytes(buf.wrapping_add(total), &mut tmp[..n])
            .is_err()
        {
            return if total > 0 {
                Ok(total)
            } else {
                Err(abi::EFAULT)
            };
        }
        match f.write(&tmp[..n]) {
            Ok(w) => {
                total += w as u32;
                if w < n {
                    break;
                }
            }
            Err(e) => return if total > 0 { Ok(total) } else { Err(e) },
        }
    }
    Ok(total)
}

pub fn sys_write(mem: &mut AddressSpace, fds: &mut FdTable, fd: u32, buf: u32, count: u32) -> u32 {
    let Some(f) = fds.get(fd) else {
        return err(abi::EBADF);
    };
    match write_from_guest(mem, f, buf, count) {
        Ok(n) => n,
        Err(e) => err(e),
    }
}

pub fn sys_read(mem: &mut AddressSpace, fds: &mut FdTable, fd: u32, buf: u32, count: u32) -> u32 {
    let Some(f) = fds.get(fd) else {
        return err(abi::EBADF);
    };
    let mut tmp = [0u8; IO_CHUNK];
    let n = (count as usize).min(IO_CHUNK);
    match f.read(&mut tmp[..n]) {
        Ok(r) => {
            if mem.write_bytes(buf, &tmp[..r]).is_err() {
                return err(abi::EFAULT);
            }
            r as u32 // a short read is legal
        }
        Err(e) => err(e),
    }
}

pub fn sys_writev(mem: &mut AddressSpace, fds: &mut FdTable, fd: u32, iov: u32, cnt: u32) -> u32 {
    if cnt > MAX_IOV {
        return err(abi::EINVAL);
    }
    if fds.get(fd).is_none() {
        return err(abi::EBADF);
    }
    let mut total = 0u32;
    for i in 0..cnt {
        let base = mem.read_u32(iov + i * 8);
        let len = mem.read_u32(iov + i * 8 + 4);
        let f = fds.get(fd).expect("checked above");
        match write_from_guest(mem, f, base, len) {
            Ok(n) => {
                total += n;
                if n < len {
                    break;
                }
            }
            Err(e) => return if total > 0 { total } else { err(e) },
        }
    }
    total
}

pub fn sys_readv(mem: &mut AddressSpace, fds: &mut FdTable, fd: u32, iov: u32, cnt: u32) -> u32 {
    if cnt > MAX_IOV {
        return err(abi::EINVAL);
    }
    let mut total = 0u32;
    for i in 0..cnt {
        let base = mem.read_u32(iov + i * 8);
        let len = mem.read_u32(iov + i * 8 + 4);
        let r = sys_read(mem, fds, fd, base, len);
        if (r as i32) < 0 {
            return if total > 0 { total } else { r };
        }
        total += r;
        if r < len {
            break; // EOF or short read
        }
    }
    total
}

/// `open(2)`, read-only: relative guest paths resolve against the host
/// working directory; absolute guest paths and write flags are refused.
pub fn sys_open(mem: &mut AddressSpace, fds: &mut FdTable, path: u32, flags: u32) -> u32 {
    let Ok(bytes) = mem.read_cstr(path) else {
        return err(abi::EFAULT);
    };
    if flags & 0x3 != 0 {
        return err(abi::EACCES); // O_WRONLY / O_RDWR
    }
    let name = String::from_utf8_lossy(&bytes).into_owned();
    if name.starts_with('/') {
        return err(abi::ENOENT); // no guest filesystem root in v1
    }
    match std::fs::File::open(&name) {
        Ok(f) => fds.insert(Fd::File(f)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => err(abi::ENOENT),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => err(abi::EACCES),
        Err(_) => err(abi::EIO),
    }
}

pub fn sys_openat(
    mem: &mut AddressSpace,
    fds: &mut FdTable,
    dirfd: u32,
    path: u32,
    flags: u32,
) -> u32 {
    if dirfd != abi::AT_FDCWD {
        return err(abi::ENOSYS); // only AT_FDCWD in v1
    }
    sys_open(mem, fds, path, flags)
}

pub fn sys_close(fds: &mut FdTable, fd: u32) -> u32 {
    if fds.close(fd) { 0 } else { err(abi::EBADF) }
}

pub fn sys_lseek(fds: &mut FdTable, fd: u32, offset: u32, whence: u32) -> u32 {
    let Some(f) = fds.get(fd) else {
        return err(abi::EBADF);
    };
    let Fd::File(file) = f else {
        return err(abi::ESPIPE);
    };
    let pos = match whence {
        abi::SEEK_SET => SeekFrom::Start(offset as u64),
        abi::SEEK_CUR => SeekFrom::Current(offset as i32 as i64),
        abi::SEEK_END => SeekFrom::End(offset as i32 as i64),
        _ => return err(abi::EINVAL),
    };
    match file.seek(pos) {
        Ok(p) if p <= u32::MAX as u64 => p as u32,
        Ok(_) => err(abi::EINVAL), // beyond what a 32-bit lseek can report
        Err(_) => err(abi::EINVAL),
    }
}

/// `fstat64(2)`: a zeroed i386 `stat64` (96 bytes, packed layout) with just
/// enough filled in — mode, nlink, ids, size, blksize, inode.
pub fn sys_fstat64(mem: &mut AddressSpace, fds: &mut FdTable, fd: u32, statbuf: u32) -> u32 {
    const S_IFCHR: u32 = 0o020000;
    const S_IFREG: u32 = 0o100000;
    let Some(f) = fds.get(fd) else {
        return err(abi::EBADF);
    };
    let (mode, size) = match f {
        Fd::Stdin | Fd::Stdout | Fd::Stderr | Fd::Sink(_) => (S_IFCHR | 0o620, 0),
        Fd::File(file) => (
            S_IFREG | 0o444,
            file.metadata().map(|m| m.len()).unwrap_or(0),
        ),
    };
    let mut st = [0u8; 96];
    st[16..20].copy_from_slice(&mode.to_le_bytes());
    st[20..24].copy_from_slice(&1u32.to_le_bytes()); // st_nlink
    st[24..28].copy_from_slice(&abi::GUEST_ID.to_le_bytes()); // st_uid
    st[28..32].copy_from_slice(&abi::GUEST_ID.to_le_bytes()); // st_gid
    st[44..52].copy_from_slice(&size.to_le_bytes()); // st_size (long long)
    st[52..56].copy_from_slice(&1024u32.to_le_bytes()); // st_blksize
    st[88..96].copy_from_slice(&(fd as u64 + 1).to_le_bytes()); // st_ino
    if mem.write_bytes(statbuf, &st).is_err() {
        return err(abi::EFAULT);
    }
    0
}
