//! `ghost static` -- a PARKED ghost: one car standing still at a chosen world
//! position for a chosen span, wearing a chosen skin. The instrument of the
//! skin-scenery study (2026-09-12): the MediaTracker draws a ghost block's car
//! wherever its record says, and a 3D car skin can carry any mesh -- so a
//! parked ghost with a scenery skin is scenery the map file does not embed.
//!
//! Built on a DONOR container the client is known to import (a certified lap):
//! the record is laid out fresh on a 50 ms grid out to `--span` (every sample a
//! copy of the donor's first vehicle sample, `record::rebuild_to`), then every
//! sample's transform is overwritten with the requested position, a yaw about
//! +Y and zero velocity. The skin path and the PackDesc locator URL are edited
//! in place (the 32-byte skin checksum is zeroed, as `identity set --skin`
//! does). Nothing else in the donor changes: tape, splits and declared time are
//! irrelevant to a MediaTracker ghost block and are left alone, and the file
//! still names the donor's map uid.
//!
//! Control: the written file is decoded again and every vehicle sample must
//! read back at the requested position (to the encoding's 1e-4 m), and the
//! identity scan must find the skin and locator strings as written.

use gbx::container::{body_strings_in, replace_strings, write_gbx, Container};
use gbx::recwrite::{rewrite_ghost, write_transform, Xform};

use crate::cli::{die, flag, has};

pub struct Params {
    pub pos: [f32; 3],
    pub yaw_deg: f64,
    pub span_ms: i64,
    pub skin: Option<String>,
    pub locator: Option<String>,
    /// The PackDesc checksum = SHA-256 of the skin zip (measured 2026-09-12 on a
    /// leaderboard ghost: its 32 bytes are exactly sha256 of the zip Nadeo serves).
    /// A zero checksum with a local zip makes the game try to "update" the file
    /// from its (empty) locator at the menu: the "Updating data… $<$> (???)" modal.
    pub checksum: Option<[u8; 32]>,
}

/// Yaw about +Y (radians, 0 = facing +Z, right-handed) as an (x, y, z, w)
/// quaternion.
fn yaw_quat(yaw_rad: f64) -> [f64; 4] {
    let h = yaw_rad / 2.0;
    [0.0, h.sin(), 0.0, h.cos()]
}

pub fn build(inp: &str, out: &str, p: &Params) -> Result<String, String> {
    if p.span_ms <= 0 {
        return Err("--span must be positive milliseconds".into());
    }
    let tmp1 = format!("{out}.static-grid.tmp");
    let tmp2 = format!("{out}.static-xf.tmp");
    // 1. a fresh 50 ms grid of template samples
    let note = crate::record::rebuild_to(inp, &tmp1, p.span_ms, None, 50, false)?;
    // 2. every sample parked at the requested transform
    let xf = Xform { pos: p.pos, quat: yaw_quat(p.yaw_deg.to_radians()), vel: [0.0; 3] };
    let mut written = 0usize;
    rewrite_ghost(&tmp1, &tmp2, |rd| {
        let vi = rd
            .ents
            .iter()
            .enumerate()
            .filter(|(_, e)| e.sample_size >= 100 && !e.times.is_empty())
            .max_by_key(|(_, e)| e.times.len())
            .map(|(i, _)| i)
            .ok_or("no vehicle entity in the rebuilt record")?;
        let e = &mut rd.ents[vi];
        let ss = e.sample_size;
        let n = e.raw.len() / ss;
        for k in 0..n {
            write_transform(&mut e.raw[k * ss..(k + 1) * ss], 47, &xf);
            written += 1;
        }
        Ok(())
    })?;
    let _ = std::fs::remove_file(&tmp1);
    // 3. identity: the skin path and the locator URL that follows it in the
    //    PackDesc (`u8 version | 32-byte checksum | string path | string url`)
    let c = Container::load(&tmp2)?;
    let body = c.body().to_vec();
    let fields = crate::ident::scan(&c);
    let skin_field = fields
        .iter()
        .find(|f| f.role == crate::ident::Role::Skin)
        .ok_or("the donor carries no skin PackDesc to edit")?;
    let mut edits: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    let mut pre = body.clone();
    if let Some(s) = &p.skin {
        if s.as_bytes() != skin_field.s.as_bytes() {
            edits.push((skin_field.at, skin_field.len, s.as_bytes().to_vec()));
        }
        // the checksum: SHA-256 of the zip when given (--skin-file / --checksum), else
        // zero as `identity set` writes it (the game then wants to update the file)
        match p.checksum {
            Some(h) => pre[skin_field.at - 32..skin_field.at].copy_from_slice(&h),
            None => {
                for b in pre[skin_field.at - 32..skin_field.at].iter_mut() {
                    *b = 0;
                }
            }
        }
    }
    if let Some(url) = &p.locator {
        // the locator is the string right after the path: its length word sits
        // at path.at + 4 + path.len (an EMPTY string on every donor here, so the
        // string walk never lists it -- it is addressed structurally)
        let at = skin_field.at + 4 + skin_field.len;
        let old_len = u32::from_le_bytes(body[at..at + 4].try_into().unwrap()) as usize;
        if old_len > 1024 {
            return Err(format!("the word after the skin path reads {old_len}, not a plausible locator length"));
        }
        // only an http(s) URL or an empty string is a locator the game understands
        if !(url.is_empty() || url.starts_with("http://") || url.starts_with("https://")) {
            return Err(format!("--locator {url:?}: not an http(s) URL"));
        }
        edits.push((at, old_len, url.as_bytes().to_vec()));
    }
    edits.sort_by_key(|e| e.0);
    let new_body = if edits.is_empty() { pre } else { replace_strings(&pre, &edits, None)? };
    let unframed = gbx::container::unframed_edits();
    if !unframed.is_empty() {
        return Err(format!("an identity edit fell outside every skippable chunk: {}", unframed.join(", ")));
    }
    write_gbx(&c.gbx, new_body, out)?;
    let _ = std::fs::remove_file(&tmp2);
    // ---- controls
    let back = gbx::record::decode_ghost(out)?;
    let n = back.samples.len();
    if n == 0 {
        let _ = std::fs::remove_file(out);
        return Err("the written file decodes to no vehicle samples".into());
    }
    for (i, s) in back.samples.iter().enumerate() {
        let d = ((s.x as f64 - p.pos[0] as f64).powi(2) + (s.y as f64 - p.pos[1] as f64).powi(2) + (s.z as f64 - p.pos[2] as f64).powi(2)).sqrt();
        if d > 1e-3 {
            let _ = std::fs::remove_file(out);
            return Err(format!("sample {i} reads back at ({:.3}, {:.3}, {:.3}), {d:.4} m from the requested position", s.x, s.y, s.z));
        }
    }
    let c2 = Container::load(out)?;
    let f2 = crate::ident::scan(&c2);
    let skin_now = f2.iter().find(|f| f.role == crate::ident::Role::Skin).map(|f| f.s.clone()).unwrap_or_default();
    if let Some(s) = &p.skin {
        if &skin_now != s {
            let _ = std::fs::remove_file(out);
            return Err(format!("skin reads back as {skin_now:?}, asked for {s:?}"));
        }
    }
    let loc_now = f2.iter().find(|f| f.role == crate::ident::Role::Locator).map(|f| f.s.clone());
    if let Some(url) = &p.locator {
        if !url.is_empty() && loc_now.as_deref() != Some(url.as_str()) {
            let _ = std::fs::remove_file(out);
            return Err(format!("locator reads back as {loc_now:?}, asked for {url:?}"));
        }
    }
    // the PackDesc bytes, for the report: version byte + checksum
    let body2 = c2.body();
    let sk2 = f2.iter().find(|f| f.role == crate::ident::Role::Skin).unwrap();
    let ver = body2[sk2.at - 33];
    let cks: Vec<String> = body2[sk2.at - 32..sk2.at].iter().map(|b| format!("{b:02x}")).collect();
    // the display-name string, to confirm the head of 0x03092000 survived
    let head = body_strings_in(body2, 0, sk2.at + 4 + sk2.len + 64);
    let nick = head.iter().find(|b| b.at > sk2.at).map(|b| b.s.clone()).unwrap_or_default();
    Ok(format!(
        "{n} samples on a 50 ms grid ({} .. {}) parked at ({:.3}, {:.3}, {:.3}) yaw {:.1} deg, {written} transforms written; rebuild: {note}; skin {skin_now:?} locator {loc_now:?} (PackDesc v{ver}, checksum {}); first string after the skin: {nick:?}",
        gbx::container::secs(back.samples.first().map(|s| s.time_ms as i64).unwrap_or(0)),
        gbx::container::secs(back.samples.last().map(|s| s.time_ms as i64).unwrap_or(0)),
        p.pos[0], p.pos[1], p.pos[2], p.yaw_deg,
        cks.join("")
    ))
}

pub fn cmd(a: &[String]) {
    let usage = "ghost static IN OUT --pos X,Y,Z [--yaw DEG] [--span MS] [--skin PATH] [--locator URL] [--skin-file ZIP | --checksum HEX64]";
    let inp = a.first().unwrap_or_else(|| die(usage));
    let out = a.get(1).unwrap_or_else(|| die(usage));
    let pos_s = flag(a, "--pos").unwrap_or_else(|| die(usage));
    let v: Vec<f32> = pos_s.split(',').map(|x| x.trim().parse::<f32>().unwrap_or_else(|_| die(format!("--pos {pos_s:?}: three numbers")))).collect();
    if v.len() != 3 {
        die(format!("--pos {pos_s:?}: three numbers"));
    }
    let yaw_deg: f64 = flag(a, "--yaw").map(|s| s.parse().unwrap_or_else(|_| die("--yaw DEG"))).unwrap_or(0.0);
    let span_ms: i64 = flag(a, "--span").map(|s| s.parse().unwrap_or_else(|_| die("--span MS"))).unwrap_or(60_000);
    let p = Params {
        pos: [v[0], v[1], v[2]],
        yaw_deg,
        span_ms,
        skin: flag(a, "--skin").map(|s| s.to_string()),
        locator: flag(a, "--locator").map(|s| s.to_string()),
        checksum: match (flag(a, "--skin-file"), flag(a, "--checksum")) {
            (Some(f), _) => Some(gbx::sha::sha256_file(std::path::Path::new(f)).unwrap_or_else(|e| die(&format!("--skin-file {f}: {e}")))),
            (None, Some(h)) => {
                let h = h.trim();
                if h.len() != 64 { die("--checksum wants 64 hex chars (sha256 of the zip)"); }
                let mut out = [0u8; 32];
                for i in 0..32 { out[i] = u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).unwrap_or_else(|_| die("--checksum: not hex")); }
                Some(out)
            }
            _ => None,
        },
    };
    if has(a, "--help") {
        die(usage);
    }
    match build(inp, out, &p) {
        Ok(m) => println!("wrote {out}: {m}"),
        Err(e) => die(e),
    }
}
