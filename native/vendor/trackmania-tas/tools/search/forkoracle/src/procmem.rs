//! Read another process's memory via /proc/<pid>/mem, list maps, find needles.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Clone, Debug)]
pub struct Region {
    pub start: u64,
    pub end: u64,
    pub perms: String,
    pub off: u64,
    pub path: String,
}

pub fn maps(pid: i32) -> Vec<Region> {
    let s = std::fs::read_to_string(format!("/proc/{}/maps", pid)).unwrap_or_default();
    let mut v = Vec::new();
    for l in s.lines() {
        let mut it = l.split_whitespace();
        let range = match it.next() {
            Some(x) => x,
            None => continue,
        };
        let perms = it.next().unwrap_or("").to_string();
        let off = u64::from_str_radix(it.next().unwrap_or("0"), 16).unwrap_or(0);
        let _dev = it.next();
        let _ino = it.next();
        let path = it.next().unwrap_or("").to_string();
        let (a, b) = range.split_once('-').unwrap();
        v.push(Region {
            start: u64::from_str_radix(a, 16).unwrap(),
            end: u64::from_str_radix(b, 16).unwrap(),
            perms,
            off,
            path,
        });
    }
    v
}

pub fn read_region(f: &mut File, r: &Region) -> Option<Vec<u8>> {
    let len = (r.end - r.start) as usize;
    if len > 1 << 30 {
        return None;
    }
    let mut buf = vec![0u8; len];
    f.seek(SeekFrom::Start(r.start)).ok()?;
    match f.read_exact(&mut buf) {
        Ok(_) => Some(buf),
        Err(_) => None,
    }
}

pub fn read_at(pid: i32, addr: u64, len: usize) -> Option<Vec<u8>> {
    let mut f = File::open(format!("/proc/{}/mem", pid)).ok()?;
    let mut buf = vec![0u8; len];
    f.seek(SeekFrom::Start(addr)).ok()?;
    f.read_exact(&mut buf).ok()?;
    Some(buf)
}

/// All addresses where `needle` occurs in writable/anonymous+file regions.
pub fn find(pid: i32, needle: &[u8], want_write: bool) -> Vec<(u64, Region)> {
    let mut out = Vec::new();
    let mut f = match File::open(format!("/proc/{}/mem", pid)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open mem: {}", e);
            return out;
        }
    };
    for r in maps(pid) {
        if !r.perms.starts_with('r') {
            continue;
        }
        if want_write && !r.perms.contains('w') {
            continue;
        }
        if r.path.starts_with("/dev") || r.path == "[vvar]" || r.path.starts_with("[vsyscall") {
            continue;
        }
        let buf = match read_region(&mut f, &r) {
            Some(b) => b,
            None => continue,
        };
        let mut i = 0usize;
        while i + needle.len() <= buf.len() {
            match memmem(&buf[i..], needle) {
                Some(p) => {
                    out.push((r.start + (i + p) as u64, r.clone()));
                    i += p + 1;
                }
                None => break,
            }
        }
    }
    out
}

/// Boyer–Moore–Horspool substring search.
///
/// The naive "scan for the first byte, then compare" loop degrades badly on the
/// heap of a game engine, where the needle's first byte (often 0x00) matches
/// millions of times. Horspool skips by a whole shift table entry per mismatch,
/// so a 3965-byte needle strides through the haystack in ~len/needle steps in
/// the common case, and never does worse than the naive scan.
pub struct Horspool {
    needle: Vec<u8>,
    shift: [usize; 256],
}

impl Horspool {
    pub fn new(needle: &[u8]) -> Horspool {
        let n = needle.len();
        let mut shift = [n; 256];
        for (i, &b) in needle.iter().enumerate().take(n.saturating_sub(1)) {
            shift[b as usize] = n - 1 - i;
        }
        Horspool {
            needle: needle.to_vec(),
            shift,
        }
    }

    /// First occurrence at or after `from`.
    pub fn find_from(&self, hay: &[u8], from: usize) -> Option<usize> {
        let n = self.needle.len();
        if n == 0 || hay.len() < n {
            return None;
        }
        let last = n - 1;
        let mut i = from;
        while i + n <= hay.len() {
            let c = hay[i + last];
            if c == self.needle[last] && hay[i..i + n] == self.needle[..] {
                return Some(i);
            }
            i += self.shift[c as usize];
        }
        None
    }

    pub fn find_all(&self, hay: &[u8]) -> Vec<usize> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while let Some(p) = self.find_from(hay, i) {
            out.push(p);
            i = p + 1;
        }
        out
    }
}

pub fn memmem(hay: &[u8], needle: &[u8]) -> Option<usize> {
    Horspool::new(needle).find_from(hay, 0)
}


/* `write_at` was here: writing another process's memory through
   /proc/<pid>/mem. Its only caller was `fk poke`, which overwrote every
   in-memory copy of a bitstream to test whether the engine re-reads it. It
   does not -- the raw stream is decoded once at load and never read again --
   so the question is answered and the capability is not needed. */
