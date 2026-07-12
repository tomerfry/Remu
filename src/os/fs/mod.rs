//! Virtual filesystem: a host rootfs directory as guest `/`, path-sandboxed,
//! with virtual `/dev` and `/proc` overlays and a per-process fd table.
//!
//! The VFS deals only in host byte buffers; the syscall layer marshals those to
//! and from guest memory. Paths are canonicalized and confined to the rootfs so
//! a guest cannot escape via `..` or absolute symlinks.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use crate::os::abi::{errno, open};

/// A directory entry snapshot (for `getdents64`).
#[derive(Clone)]
pub struct DirEnt {
    pub name: String,
    pub is_dir: bool,
}

/// Kinds of virtual device/proc file.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Virt {
    Null,
    Zero,
    Full,
    Random,
    /// A read-only file whose bytes were generated at open time.
    Generated,
}

/// An open file descriptor.
pub enum Fd {
    /// Standard streams wired to the host (0/1/2).
    Std(u8),
    /// A host-backed regular file.
    File { file: File, path: String },
    /// A host-backed directory opened for `getdents64`.
    Dir {
        entries: Vec<DirEnt>,
        pos: usize,
        path: String,
    },
    /// A virtual device or generated file.
    Virtual {
        kind: Virt,
        data: Vec<u8>,
        pos: usize,
    },
}

/// The filesystem state for one process.
pub struct Vfs {
    /// Host directory serving as guest `/` (None → only virtual + std fds).
    rootfs: Option<PathBuf>,
    /// Canonicalized rootfs, for the symlink-escape containment check.
    canon_root: Option<PathBuf>,
    /// Guest current working directory (absolute, normalized).
    cwd: String,
    /// Descriptor table (indexed by fd).
    fds: Vec<Option<Fd>>,
    /// Guest path of the executed program (for `/proc/self/exe`, cmdline).
    pub exec_path: String,
    /// `argv` bytes, for `/proc/self/cmdline`.
    pub cmdline: Vec<u8>,
    /// `/proc/self/maps` text, refreshed by the emulator.
    pub maps: String,
    /// When `Some`, guest writes to stdout/stderr are captured here instead of
    /// going to the host (for tests).
    capture: Option<Vec<u8>>,
}

impl Vfs {
    pub fn new(rootfs: Option<PathBuf>) -> Self {
        let canon_root = rootfs.as_ref().and_then(|r| r.canonicalize().ok());
        Vfs {
            rootfs,
            canon_root,
            cwd: "/".into(),
            fds: vec![Some(Fd::Std(0)), Some(Fd::Std(1)), Some(Fd::Std(2))],
            exec_path: "/a.out".into(),
            cmdline: Vec::new(),
            maps: String::new(),
            capture: None,
        }
    }

    /// Redirect guest stdout/stderr into an in-memory buffer (for tests).
    pub fn capture_output(&mut self) {
        self.capture = Some(Vec::new());
    }

    /// The captured stdout/stderr bytes (empty if capture was not enabled).
    pub fn captured(&self) -> &[u8] {
        self.capture.as_deref().unwrap_or(&[])
    }

    // --- fd table -----------------------------------------------------------

    fn install(&mut self, fd: Fd) -> i32 {
        if let Some(i) = self.fds.iter().position(|s| s.is_none()) {
            self.fds[i] = Some(fd);
            i as i32
        } else {
            self.fds.push(Some(fd));
            (self.fds.len() - 1) as i32
        }
    }

    pub fn get(&self, fd: i32) -> Option<&Fd> {
        self.fds.get(fd as usize).and_then(|s| s.as_ref())
    }

    pub fn close(&mut self, fd: i32) -> i32 {
        match self.fds.get_mut(fd as usize) {
            Some(slot @ Some(_)) => {
                *slot = None;
                0
            }
            _ => -errno::EBADF,
        }
    }

    // --- path handling ------------------------------------------------------

    /// Normalize a guest path to an absolute, `.`/`..`-collapsed form.
    fn normalize(&self, path: &str) -> String {
        let start = if path.starts_with('/') {
            String::new()
        } else {
            self.cwd.clone()
        };
        let joined = format!("{start}/{path}");
        let mut parts: Vec<&str> = Vec::new();
        for comp in joined.split('/') {
            match comp {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                c => parts.push(c),
            }
        }
        format!("/{}", parts.join("/"))
    }

    /// Map a normalized guest path to a host path under the rootfs (sandboxed).
    ///
    /// `normalize` already collapses guest `..`, so the guest cannot escape by
    /// path. This additionally guards against host symlinks inside the rootfs
    /// that point outside it: the canonicalized target (or, for a not-yet-created
    /// file, its nearest existing ancestor) must remain under the rootfs.
    fn host_path(&self, guest: &str) -> Option<PathBuf> {
        let root = self.rootfs.as_ref()?;
        let cand = root.join(guest.trim_start_matches('/'));
        if let Some(canon_root) = &self.canon_root {
            let resolved = cand
                .canonicalize()
                .ok()
                .or_else(|| cand.parent().and_then(|p| p.canonicalize().ok()));
            if let Some(r) = resolved
                && !r.starts_with(canon_root)
            {
                return None; // symlink escape → treat as absent
            }
        }
        Some(cand)
    }

    /// Generate the content of a virtual `/proc` file, if `guest` names one.
    fn generated(&self, guest: &str) -> Option<Vec<u8>> {
        let body = match guest {
            "/proc/self/cmdline" => self.cmdline.clone(),
            "/proc/self/maps" => self.maps.clone().into_bytes(),
            "/proc/self/exe" => self.exec_path.clone().into_bytes(),
            "/proc/cpuinfo" => b"processor\t: 0\nvendor_id\t: GenuineIntel\n".to_vec(),
            _ => return None,
        };
        Some(body)
    }

    /// Resolve a virtual device path to its kind, if any.
    fn virt_dev(guest: &str) -> Option<Virt> {
        match guest {
            "/dev/null" => Some(Virt::Null),
            "/dev/zero" => Some(Virt::Zero),
            "/dev/full" => Some(Virt::Full),
            "/dev/random" | "/dev/urandom" => Some(Virt::Random),
            _ => None,
        }
    }

    // --- syscall surface ----------------------------------------------------

    /// `open`/`openat` (dirfd handling is done by the caller building `path`).
    pub fn open(&mut self, path: &str, flags: u32, _mode: u32) -> i32 {
        let guest = self.normalize(path);

        if let Some(kind) = Self::virt_dev(&guest) {
            return self.install(Fd::Virtual {
                kind,
                data: Vec::new(),
                pos: 0,
            });
        }
        if let Some(data) = self.generated(&guest) {
            return self.install(Fd::Virtual {
                kind: Virt::Generated,
                data,
                pos: 0,
            });
        }

        let Some(host) = self.host_path(&guest) else {
            return -errno::ENOENT;
        };
        if host.is_dir() {
            let mut entries = vec![
                DirEnt {
                    name: ".".into(),
                    is_dir: true,
                },
                DirEnt {
                    name: "..".into(),
                    is_dir: true,
                },
            ];
            if let Ok(rd) = std::fs::read_dir(&host) {
                for e in rd.flatten() {
                    entries.push(DirEnt {
                        name: e.file_name().to_string_lossy().into_owned(),
                        is_dir: e.path().is_dir(),
                    });
                }
            }
            return self.install(Fd::Dir {
                entries,
                pos: 0,
                path: guest,
            });
        }

        let mut opts = OpenOptions::new();
        let write = flags & (open::O_WRONLY | open::O_RDWR) != 0;
        opts.read(flags & open::O_WRONLY == 0);
        if write {
            opts.write(true);
        }
        if flags & open::O_CREAT != 0 {
            opts.create(true);
        }
        if flags & open::O_TRUNC != 0 {
            opts.truncate(true);
        }
        if flags & open::O_APPEND != 0 {
            opts.append(true);
        }
        match opts.open(&host) {
            Ok(file) => self.install(Fd::File { file, path: guest }),
            Err(e) => -errno_from_io(&e),
        }
    }

    /// Read up to `buf.len()` bytes; returns count or `-errno`.
    pub fn read(&mut self, fd: i32, buf: &mut [u8]) -> i32 {
        match self.fds.get_mut(fd as usize).and_then(|s| s.as_mut()) {
            Some(Fd::Std(0)) => std::io::stdin()
                .read(buf)
                .map(|n| n as i32)
                .unwrap_or(-errno::EIO),
            Some(Fd::Std(_)) => -errno::EBADF,
            Some(Fd::File { file, .. }) => file.read(buf).map(|n| n as i32).unwrap_or(-errno::EIO),
            Some(Fd::Dir { .. }) => -errno::EISDIR,
            Some(Fd::Virtual { kind, data, pos }) => match kind {
                Virt::Null => 0,
                Virt::Zero => {
                    buf.iter_mut().for_each(|b| *b = 0);
                    buf.len() as i32
                }
                Virt::Full => {
                    buf.iter_mut().for_each(|b| *b = 0);
                    buf.len() as i32
                }
                Virt::Random => {
                    // Deterministic PRNG (xorshift seeded by position).
                    let mut x = 0x2545_F491u32.wrapping_add(*pos as u32).max(1);
                    for b in buf.iter_mut() {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        *b = x as u8;
                    }
                    *pos += buf.len();
                    buf.len() as i32
                }
                Virt::Generated => {
                    let n = (data.len() - (*pos).min(data.len())).min(buf.len());
                    buf[..n].copy_from_slice(&data[*pos..*pos + n]);
                    *pos += n;
                    n as i32
                }
            },
            None => -errno::EBADF,
        }
    }

    /// Write `data`; returns count or `-errno`.
    pub fn write(&mut self, fd: i32, data: &[u8]) -> i32 {
        match self.fds.get_mut(fd as usize).and_then(|s| s.as_mut()) {
            Some(Fd::Std(1)) => {
                if let Some(cap) = self.capture.as_mut() {
                    cap.extend_from_slice(data);
                } else {
                    let _ = std::io::stdout().write_all(data);
                    let _ = std::io::stdout().flush();
                }
                data.len() as i32
            }
            Some(Fd::Std(2)) => {
                if let Some(cap) = self.capture.as_mut() {
                    cap.extend_from_slice(data);
                } else {
                    let _ = std::io::stderr().write_all(data);
                }
                data.len() as i32
            }
            Some(Fd::Std(_)) => -errno::EBADF,
            Some(Fd::File { file, .. }) => {
                file.write(data).map(|n| n as i32).unwrap_or(-errno::EIO)
            }
            Some(Fd::Virtual { kind, .. }) => match kind {
                Virt::Null | Virt::Zero | Virt::Random => data.len() as i32,
                Virt::Full => -errno::ENOSPC,
                Virt::Generated => -errno::EBADF,
            },
            _ => -errno::EBADF,
        }
    }

    /// `lseek` (whence: 0=SET, 1=CUR, 2=END). Returns the new offset or `-errno`.
    pub fn lseek(&mut self, fd: i32, off: i64, whence: u32) -> i64 {
        match self.fds.get_mut(fd as usize).and_then(|s| s.as_mut()) {
            Some(Fd::File { file, .. }) => {
                let sk = match whence {
                    0 => SeekFrom::Start(off as u64),
                    1 => SeekFrom::Current(off),
                    2 => SeekFrom::End(off),
                    _ => return -(errno::EINVAL as i64),
                };
                file.seek(sk)
                    .map(|p| p as i64)
                    .unwrap_or(-(errno::EIO as i64))
            }
            Some(Fd::Virtual { pos, data, .. }) => {
                let base = match whence {
                    0 => 0i64,
                    1 => *pos as i64,
                    2 => data.len() as i64,
                    _ => return -(errno::EINVAL as i64),
                };
                let np = (base + off).max(0) as usize;
                *pos = np;
                np as i64
            }
            Some(_) => -(errno::ESPIPE as i64),
            None => -(errno::EBADF as i64),
        }
    }

    /// Size of a regular-file fd (for `fstat64`), or `None`.
    pub fn file_len(&self, fd: i32) -> Option<u64> {
        match self.get(fd)? {
            Fd::File { file, .. } => file.metadata().ok().map(|m| m.len()),
            Fd::Virtual { data, .. } => Some(data.len() as u64),
            _ => None,
        }
    }

    /// Whether an fd is a directory.
    pub fn is_dir_fd(&self, fd: i32) -> bool {
        matches!(self.get(fd), Some(Fd::Dir { .. }))
    }

    /// Whether an fd is a tty (only the std streams here).
    pub fn is_tty(&self, fd: i32) -> bool {
        matches!(self.get(fd), Some(Fd::Std(_)))
    }

    /// Read the next batch of directory entries for `getdents64`, invoking
    /// `emit(name, is_dir)` and advancing the cursor; returns entries consumed.
    pub fn next_dents(
        &mut self,
        fd: i32,
        mut emit: impl FnMut(&str, bool) -> bool,
    ) -> Option<usize> {
        let Some(Fd::Dir { entries, pos, .. }) =
            self.fds.get_mut(fd as usize).and_then(|s| s.as_mut())
        else {
            return None;
        };
        let mut n = 0;
        while *pos < entries.len() {
            let e = &entries[*pos];
            if !emit(&e.name, e.is_dir) {
                break;
            }
            *pos += 1;
            n += 1;
        }
        Some(n)
    }

    /// Resolve `readlink` targets we synthesize (`/proc/self/exe`).
    pub fn readlink(&self, path: &str) -> Option<String> {
        let guest = self.normalize(path);
        match guest.as_str() {
            "/proc/self/exe" => Some(self.exec_path.clone()),
            _ => {
                let host = self.host_path(&guest)?;
                std::fs::read_link(host)
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned())
            }
        }
    }

    /// `access`: does the guest path exist / is it a known virtual?
    pub fn access(&self, path: &str) -> i32 {
        let guest = self.normalize(path);
        if Self::virt_dev(&guest).is_some() || self.generated(&guest).is_some() {
            return 0;
        }
        match self.host_path(&guest) {
            Some(h) if h.exists() => 0,
            _ => -errno::ENOENT,
        }
    }

    pub fn cwd(&self) -> &str {
        &self.cwd
    }
}

/// Translate a std::io error to a Linux errno (coarse).
fn errno_from_io(e: &std::io::Error) -> i32 {
    use std::io::ErrorKind::*;
    match e.kind() {
        NotFound => errno::ENOENT,
        PermissionDenied => errno::EACCES,
        AlreadyExists => errno::EEXIST,
        _ => errno::EINVAL,
    }
}
