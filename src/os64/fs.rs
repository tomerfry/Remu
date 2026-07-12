//! A small file-descriptor table and virtual filesystem for the x86-64 OS
//! layer.
//!
//! Fds 0-2 map to the host's standard streams (raw bytes, no translation).
//! `open`/`openat` serve read-only host files resolved against the host working
//! directory; write flags and absolute guest paths are refused (there is no
//! guest root yet). [`Fd::Sink`] captures output in memory so tests never touch
//! the real console. All methods return non-negative counts or `-errno`.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::os64::abi::{errno, open};

/// One open file description.
pub enum Fd {
    Stdin,
    Stdout,
    Stderr,
    File(std::fs::File),
    /// In-memory output capture (tests).
    Sink(Vec<u8>),
}

/// The process fd table plus VFS policy.
pub struct Vfs {
    fds: Vec<Option<Fd>>,
}

impl Vfs {
    pub fn new() -> Self {
        Vfs {
            fds: vec![Some(Fd::Stdin), Some(Fd::Stdout), Some(Fd::Stderr)],
        }
    }

    fn get(&mut self, fd: i64) -> Option<&mut Fd> {
        usize::try_from(fd)
            .ok()
            .and_then(|i| self.fds.get_mut(i))?
            .as_mut()
    }

    /// Place `f` in the lowest free slot; returns its number.
    fn insert(&mut self, f: Fd) -> i64 {
        for (i, slot) in self.fds.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(f);
                return i as i64;
            }
        }
        self.fds.push(Some(f));
        (self.fds.len() - 1) as i64
    }

    /// Replace whatever occupies `fd` (tests: redirect stdout to a Sink).
    pub fn install(&mut self, fd: usize, f: Fd) {
        if fd >= self.fds.len() {
            self.fds.resize_with(fd + 1, || None);
        }
        self.fds[fd] = Some(f);
    }

    /// The bytes captured by a [`Fd::Sink`] at `fd`, if that's what lives there.
    pub fn sink_data(&self, fd: usize) -> Option<&[u8]> {
        match self.fds.get(fd)? {
            Some(Fd::Sink(v)) => Some(v),
            _ => None,
        }
    }

    /// `open(2)`, read-only. Relative guest paths resolve against the host cwd;
    /// absolute paths and write flags are refused.
    pub fn open(&mut self, path: &str, flags: u64) -> i64 {
        if flags & (open::O_WRONLY | open::O_RDWR) != 0 {
            return -errno::EACCES;
        }
        if path.starts_with('/') {
            return -errno::ENOENT; // no guest filesystem root yet
        }
        match std::fs::File::open(path) {
            Ok(f) => self.insert(Fd::File(f)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => -errno::ENOENT,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => -errno::EACCES,
            Err(_) => -errno::EIO,
        }
    }

    pub fn close(&mut self, fd: i64) -> i64 {
        match usize::try_from(fd).ok().and_then(|i| self.fds.get_mut(i)) {
            Some(slot @ Some(_)) => {
                *slot = None;
                0
            }
            _ => -errno::EBADF,
        }
    }

    /// Read into `buf`; returns bytes read (0 = EOF) or `-errno`.
    pub fn read(&mut self, fd: i64, buf: &mut [u8]) -> i64 {
        let Some(f) = self.get(fd) else {
            return -errno::EBADF;
        };
        let r = match f {
            Fd::Stdin => std::io::stdin().read(buf),
            Fd::Stdout | Fd::Stderr => return -errno::EBADF,
            Fd::File(file) => file.read(buf),
            Fd::Sink(_) => Ok(0),
        };
        r.map_or(-errno::EIO, |n| n as i64)
    }

    /// Write `buf`; returns bytes written or `-errno`.
    pub fn write(&mut self, fd: i64, buf: &[u8]) -> i64 {
        let Some(f) = self.get(fd) else {
            return -errno::EBADF;
        };
        let r = match f {
            Fd::Stdin => return -errno::EBADF,
            Fd::Stdout => std::io::stdout().write(buf),
            Fd::Stderr => std::io::stderr().write(buf),
            Fd::File(_) => return -errno::EBADF, // opened read-only
            Fd::Sink(v) => {
                v.extend_from_slice(buf);
                Ok(buf.len())
            }
        };
        r.map_or(-errno::EIO, |n| n as i64)
    }

    pub fn lseek(&mut self, fd: i64, offset: i64, whence: u64) -> i64 {
        let Some(Fd::File(file)) = self.get(fd) else {
            return -errno::ESPIPE;
        };
        let pos = match whence {
            0 => SeekFrom::Start(offset as u64),
            1 => SeekFrom::Current(offset),
            2 => SeekFrom::End(offset),
            _ => return -errno::EINVAL,
        };
        file.seek(pos).map_or(-errno::EINVAL, |p| p as i64)
    }

    /// File length for `fstat`, or `None` for non-regular fds.
    pub fn file_len(&mut self, fd: i64) -> Option<u64> {
        match self.get(fd)? {
            Fd::File(file) => file.metadata().ok().map(|m| m.len()),
            _ => None,
        }
    }

    /// Whether `fd` is a character device (a standard stream), for `fstat`.
    pub fn is_tty(&mut self, fd: i64) -> bool {
        matches!(self.get(fd), Some(Fd::Stdin | Fd::Stdout | Fd::Stderr))
    }

    /// Whether `fd` refers to an open description at all.
    pub fn is_open(&mut self, fd: i64) -> bool {
        self.get(fd).is_some()
    }
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}
