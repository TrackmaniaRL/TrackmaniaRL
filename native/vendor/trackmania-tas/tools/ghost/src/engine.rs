//! Finding the car in the engine BY TYPE, not by looking for floats that move
//! like a car.
//!
//! WHY THIS EXISTS
//! ---------------
//! The state reader used to locate the vehicle by scanning memory for a
//! self-consistent (position, quaternion, velocity) triple. That is a
//! DESCRIPTION of a car, not an identification of one: the engine holds several
//! objects that satisfy it, a frozen slot satisfies it trivially (a constant
//! position has a consistent zero velocity), and which one the scan lands on
//! varies between runs. Measured on one map: 8 identical runs, 1 found the car.
//! Every fix in that direction is another adjective on the description.
//!
//! The engine already knows what its objects ARE. `TrackmaniaServer` is
//! stripped, but the ManiaPlanet engine registers every class by NAME and by
//! its 32-bit class id, and the strings are right there in the binary:
//! `CSceneVehicleVis`, `CSceneVehicleVisState`, `CGameCtnApp`. This module
//! reads that registry out of the binary, so a class can be named rather than
//! guessed at.
//!
//! What the registry gives us, per class:
//!   * the class id (`CSceneVehicleVis` is `0x0A018000` -- the same id the
//!     ghost's own telemetry entity carries, which is how we know it is right);
//!   * the address of the descriptor, which is what an instance points at.

use crate::cli::{die, flag, need};

/// A LOAD segment of the ELF: file offset, virtual address, size.
struct Seg {
    off: u64,
    vaddr: u64,
    filesz: u64,
}

pub struct Image {
    pub bytes: Vec<u8>,
    segs: Vec<Seg>,
}

impl Image {
    pub fn load(path: &str) -> Result<Image, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        if bytes.len() < 64 || &bytes[0..4] != b"\x7fELF" {
            return Err(format!("{} is not an ELF file", path));
        }
        let u16at = |o: usize| u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap());
        let u64at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let phoff = u64at(0x20) as usize;
        let phentsize = u16at(0x36) as usize;
        let phnum = u16at(0x38) as usize;
        let mut segs = Vec::new();
        for i in 0..phnum {
            let p = phoff + i * phentsize;
            if u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) != 1 {
                continue; // PT_LOAD
            }
            segs.push(Seg {
                off: u64at(p + 0x08),
                vaddr: u64at(p + 0x10),
                filesz: u64at(p + 0x20),
            });
        }
        Ok(Image { bytes, segs })
    }

    pub fn off_to_vaddr(&self, off: u64) -> Option<u64> {
        self.segs
            .iter()
            .find(|s| off >= s.off && off < s.off + s.filesz)
            .map(|s| s.vaddr + (off - s.off))
    }
    pub fn vaddr_to_off(&self, va: u64) -> Option<u64> {
        self.segs
            .iter()
            .find(|s| va >= s.vaddr && va < s.vaddr + s.filesz)
            .map(|s| s.off + (va - s.vaddr))
    }

    /// Every NUL-terminated occurrence of `name` as a standalone C string.
    pub fn find_cstring(&self, name: &str) -> Vec<u64> {
        let n = name.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i + n.len() + 1 <= self.bytes.len() {
            if &self.bytes[i..i + n.len()] == n
                && self.bytes[i + n.len()] == 0
                && (i == 0 || self.bytes[i - 1] == 0)
            {
                if let Some(va) = self.off_to_vaddr(i as u64) {
                    out.push(va);
                }
                i += n.len();
                continue;
            }
            i += 1;
        }
        out
    }

    /// Every 8-byte-aligned qword in the image equal to `va`.
    pub fn find_pointers_to(&self, va: u64) -> Vec<u64> {
        let key = va.to_le_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i + 8 <= self.bytes.len() {
            if self.bytes[i..i + 8] == key {
                if let Some(at) = self.off_to_vaddr(i as u64) {
                    out.push(at);
                }
                i += 8;
                continue;
            }
            i += 1;
        }
        out
    }

    pub fn u32_at_vaddr(&self, va: u64) -> Option<u32> {
        let o = self.vaddr_to_off(va)? as usize;
        if o + 4 > self.bytes.len() {
            return None;
        }
        Some(u32::from_le_bytes(self.bytes[o..o + 4].try_into().unwrap()))
    }
    pub fn u64_at_vaddr(&self, va: u64) -> Option<u64> {
        let o = self.vaddr_to_off(va)? as usize;
        if o + 8 > self.bytes.len() {
            return None;
        }
        Some(u64::from_le_bytes(self.bytes[o..o + 8].try_into().unwrap()))
    }
}

/// A GBX class id is `0xEEEECCCC` with a small engine number: every id this
/// project has ever handled is `0x0?......` or `0x2?......` with a low 12 bits
/// of 0 for the class itself.
fn looks_like_class_id(v: u32) -> bool {
    let top = v >> 24;
    (0x01..=0x30).contains(&top) && (v & 0xFFF) == 0
}

pub struct ClassInfo {
    pub name: String,
    pub name_vaddr: u64,
    pub desc_vaddr: u64,
    pub class_id: u32,
    /// Offset of the name pointer within the descriptor.
    pub name_off: i64,
}

/// Read the engine's class registry entry for one class name.
///
/// The entry is found structurally: a pointer to the name string, with a
/// plausible GBX class id in the same small struct. Nothing here is a magic
/// offset -- the offset the id sits at is DISCOVERED and reported, and the
/// caller can check it is the same for every class it asks about, which is the
/// control that says the layout was read and not guessed.
pub fn class_info(img: &Image, name: &str) -> Result<ClassInfo, String> {
    let strs = img.find_cstring(name);
    if strs.is_empty() {
        return Err(format!("no class name string {:?} in this binary", name));
    }
    let mut best: Option<ClassInfo> = None;
    for sv in &strs {
        for p in img.find_pointers_to(*sv) {
            // look for a class id within +-64 bytes of the name pointer
            for d in (-64i64..=64).step_by(4) {
                let at = (p as i64 + d) as u64;
                let Some(v) = img.u32_at_vaddr(at) else { continue };
                if looks_like_class_id(v) {
                    let ci = ClassInfo {
                        name: name.to_string(),
                        name_vaddr: *sv,
                        desc_vaddr: p,
                        class_id: v,
                        name_off: -d,
                    };
                    if best.is_none() {
                        best = Some(ci);
                    }
                }
            }
        }
    }
    best.ok_or_else(|| {
        format!(
            "found {} copies of the name {:?} but no class id near a pointer to any of them",
            strs.len(),
            name
        )
    })
}

pub fn cmd(a: &[String]) {
    let what = a.first().map(|s| s.as_str()).unwrap_or_else(|| die("ghost engine <classinfo|classes>"));
    let rest = &a[1..];
    let bin = flag(rest, "--binary")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            crate::oracle::server_dir(flag(rest, "--server"))
                .join("TrackmaniaServer")
                .to_string_lossy()
                .to_string()
        });
    let img = Image::load(&bin).unwrap_or_else(|e| die(e));
    match what {
        "classinfo" => {
            let name = need(rest, "--class");
            match class_info(&img, name) {
                Err(e) => die(e),
                Ok(ci) => {
                    println!("class            {}", ci.name);
                    println!("name string at   {:#x}", ci.name_vaddr);
                    println!("descriptor at    {:#x}  (name pointer)", ci.desc_vaddr);
                    println!("class id         {:#010x}", ci.class_id);
                    println!("id sits at       name_ptr {:+}", -ci.name_off);
                }
            }
        }
        "classes" => {
            // The control: read several classes whose ids this project already
            // knows from the FILE FORMAT, and check the registry agrees. If the
            // registry says CSceneVehicleVis is 0x0A018000 -- the id the ghost's
            // own telemetry entity carries -- then the table was read, not
            // invented.
            let known: [(&str, u32); 4] = [
                ("CSceneVehicleVis", 0x0A018000),
                ("CPlugEntRecordData", 0x0911F000),
                ("CGameCtnGhost", 0x03092000),
                ("CGameCtnChallenge", 0x03043000),
            ];
            let mut agree = 0;
            for (n, want) in known {
                match class_info(&img, n) {
                    Err(e) => println!("  {:<22} {}", n, e),
                    Ok(ci) => {
                        let ok = ci.class_id == want;
                        if ok {
                            agree += 1;
                        }
                        println!(
                            "  {:<22} id {:#010x} {} (the file format says {:#010x}), descriptor {:#x}, id at name_ptr {:+}",
                            n,
                            ci.class_id,
                            if ok { "==" } else { "!=" },
                            want,
                            ci.desc_vaddr,
                            -ci.name_off
                        );
                    }
                }
            }
            println!("\n{} of {} class ids match the ids the FILE FORMAT already told us.", agree, known.len());
            if agree < known.len() {
                std::process::exit(1);
            }
        }
        "idsites" => {
            let id = flag(rest, "--class-id")
                .map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0))
                .unwrap_or(0x0A018000);
            let sites = id_sites(&img, id);
            println!("{} occurrences of {:#010x}", sites.len(), id);
            let lim: usize = flag(rest, "--limit").and_then(|v| v.parse().ok()).unwrap_or(12);
            for va in sites.iter().take(lim) {
                let o = img.vaddr_to_off(*va).unwrap() as usize;
                let lo = o.saturating_sub(32);
                let hi = (o + 40).min(img.bytes.len());
                let hex: String = img.bytes[lo..hi]
                    .iter()
                    .enumerate()
                    .map(|(k, b)| if lo + k == o { format!("[{:02x}", b) } else if lo + k == o + 3 { format!("{:02x}] ", b) } else { format!("{:02x} ", b) })
                    .collect();
                println!("  {:#x} (aligned {}): {}", va, va % 8 == 0, hex);
            }
        }
        "vtable" => {
            let id = flag(rest, "--class-id")
                .map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0))
                .unwrap_or(0x0A018000);
            let fns = class_id_getters(&img, id);
            println!("class id {:#010x}", id);
            println!("  GetClassId candidates (mov eax, imm32; ret): {}", fns.len());
            for f in &fns {
                println!("    {:#x}", f);
            }
            let vts = vtables_containing(&img, &fns);
            println!("  vtable slots pointing at them: {}", vts.len());
            for (slot, f) in &vts {
                let start = vtable_start(&img, *slot);
                println!(
                    "    slot {:#x} -> fn {:#x};  vtable starts {:#x}, this is slot #{}",
                    slot,
                    f,
                    start,
                    (slot - start) / 8
                );
            }
        }
        o => die(format!("unknown `ghost engine` operation {:?}", o)),
    }
}

// ---------------------------------------------------------------------------
// Finding a class's vtable from its class id
// ---------------------------------------------------------------------------
//
// Every `CMwNod` in this engine answers `GetClassId()` with a constant, and the
// compiler emits that as `mov eax, <imm32>; ret`. So the class id -- which we
// already know from the FILE FORMAT, independently of any binary -- leads to
// the function, the function leads to the vtable that contains it, and the
// vtable leads to the instances: an object of that class is a qword pointing at
// it. No floats, no heuristics, no "looks like a car".

/// Every `mov eax, imm32; ret` in the image whose immediate is `id`.
/// `c3` is `ret`; some are `mov eax, imm32; ret` preceded by `endbr64`.
pub fn class_id_getters(img: &Image, id: u32) -> Vec<u64> {
    let mut pat = vec![0xB8u8];
    pat.extend_from_slice(&id.to_le_bytes());
    pat.push(0xC3);
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + pat.len() <= img.bytes.len() {
        if img.bytes[i..i + pat.len()] == pat[..] {
            if let Some(va) = img.off_to_vaddr(i as u64) {
                out.push(va);
            }
            i += pat.len();
            continue;
        }
        i += 1;
    }
    out
}

/// Vtable slots that hold a pointer to one of `fns`, i.e. the vtables of every
/// class whose `GetClassId` is that function.
pub fn vtables_containing(img: &Image, fns: &[u64]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for f in fns {
        for slot in img.find_pointers_to(*f) {
            out.push((slot, *f));
        }
    }
    out
}

/// Walk back from a vtable slot to the start of the vtable: the run of
/// plausible code pointers ends at a non-pointer (an RTTI slot, an offset-to-top
/// of 0, or padding).
pub fn vtable_start(img: &Image, slot: u64) -> u64 {
    let mut at = slot;
    while at >= 8 {
        let prev = at - 8;
        match img.u64_at_vaddr(prev) {
            Some(v) if img.vaddr_to_off(v).is_some() && v > 0x1000 => at = prev,
            _ => break,
        }
    }
    at
}

/// Every 4-byte-aligned occurrence of a class id, with its segment kind and a
/// dump of what surrounds it. This is how the registry's actual shape gets
/// read instead of assumed.
pub fn id_sites(img: &Image, id: u32) -> Vec<u64> {
    let key = id.to_le_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 4 <= img.bytes.len() {
        if img.bytes[i..i + 4] == key {
            if let Some(va) = img.off_to_vaddr(i as u64) {
                out.push(va);
            }
        }
        i += 1;
    }
    out
}
