//! Writing `CPlugEntRecordData` back into a ghost.
//!
//! The decoder in `entrec.rs` consumes the record blob to its exact last byte,
//! which is what makes an encoder possible: everything in the grammar is
//! represented in `RecordData`, so re-emitting it must reproduce the blob
//! byte-for-byte. `tmtraj rec roundtrip` asserts exactly that on a real file,
//! and it is the control that licenses every later edit -- if the encoder can
//! reproduce an untouched record bit-for-bit, then a file that differs from the
//! original differs only where we changed it.

use crate::record::{Ent, RecordData, Res};
use crate::container::Gbx;

fn put_i32(o: &mut Vec<u8>, v: i32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}
fn put_data(o: &mut Vec<u8>, d: &[u8]) {
    put_i32(o, d.len() as i32);
    o.extend_from_slice(d);
}

/// Re-emit the decompressed record payload. Inverse of `parse_record_data`.
pub fn encode_record_data(rd: &RecordData) -> Vec<u8> {
    let v = rd.version;
    let mut o: Vec<u8> = Vec::with_capacity(rd.bytes_total.max(1024));
    put_i32(&mut o, rd.start_ms);
    put_i32(&mut o, rd.end_ms);
    put_i32(&mut o, rd.descs.len() as i32);
    for d in &rd.descs {
        put_u32(&mut o, d.class_id);
        put_i32(&mut o, d.u01);
        put_i32(&mut o, d.u02);
        put_i32(&mut o, d.u03);
        put_data(&mut o, &d.u04);
        put_i32(&mut o, d.u05);
    }
    if v >= 2 {
        put_i32(&mut o, rd.notices.len() as i32);
        for (a, b, c) in &rd.notices {
            put_i32(&mut o, *a);
            put_i32(&mut o, *b);
            if v >= 4 {
                put_u32(&mut o, c.unwrap_or(0));
            }
        }
    }
    // THE ENTITY LIST IS INTERLEAVED, and getting it wrong is the one mistake
    // that still produces a plausible-looking blob. The reader is:
    //     hasNext; while hasNext { fields; deltas; hasNext = u8; deltas2 }
    // so the flag for the NEXT entity sits BETWEEN this entity's samples and
    // its own deltas2 list -- not after it.
    if rd.ents.is_empty() {
        o.push(0);
    } else {
        o.push(1);
    }
    for (i, e) in rd.ents.iter().enumerate() {
        put_i32(&mut o, e.type_);
        put_i32(&mut o, e.u01);
        put_i32(&mut o, e.u02);
        put_i32(&mut o, e.u03);
        if v >= 6 {
            put_i32(&mut o, e.u04);
        }
        if v >= 11 {
            encode_deltas(&mut o, e);
        } else {
            for (k, t) in e.times.iter().enumerate() {
                o.push(1);
                put_i32(&mut o, *t);
                put_data(&mut o, &e.raw[k * e.sample_size..(k + 1) * e.sample_size]);
            }
            o.push(0);
        }
        o.push(if i + 1 < rd.ents.len() { 1 } else { 0 });
        if v >= 2 {
            for (a, b, d) in &e.deltas2 {
                o.push(1);
                put_i32(&mut o, *a);
                put_i32(&mut o, *b);
                put_data(&mut o, d);
            }
            o.push(0);
        }
    }
    if v >= 3 {
        for (a, b, d) in &rd.bulk_notices {
            o.push(1);
            put_i32(&mut o, *a);
            put_i32(&mut o, *b);
            put_data(&mut o, d);
        }
        o.push(0);
        if v >= 8 {
            put_i32(&mut o, rd.custom_modules.len() as i32);
        }
        for cm in &rd.custom_modules {
            for (u01, d, d2) in &cm.deltas {
                o.push(1);
                put_i32(&mut o, *u01);
                put_data(&mut o, d);
                if v >= 9 {
                    put_data(&mut o, d2);
                }
            }
            o.push(0);
            if v >= 10 {
                put_i32(&mut o, cm.period.unwrap_or(0));
            }
        }
    }
    o
}

/// The v>=11 columnar delta coding, inverse of `read_encoded_deltas`.
fn encode_deltas(o: &mut Vec<u8>, e: &Ent) {
    let n = e.times.len();
    put_i32(o, n as i32);
    if n == 0 {
        return;
    }
    let ss = e.sample_size;
    put_i32(o, ss as i32);
    let mut prev = 0i32;
    for t in &e.times {
        put_i32(o, t.wrapping_sub(prev));
        prev = *t;
    }
    // column-major: one running u8 accumulator per byte index
    let mut cols = vec![0u8; ss * n];
    for i in 0..ss {
        let mut acc: u8 = 0;
        for b in 0..n {
            let val = e.raw[b * ss + i];
            cols[i * n + b] = val.wrapping_sub(acc);
            acc = val;
        }
    }
    o.extend_from_slice(&cols);
}

/// Where the record node sits in the body, and how it is framed.
pub struct RecSite {
    /// offset of the `version` u32 (the class id is the 4 bytes before it)
    pub hdr: usize,
    pub version: u32,
    pub usize_: usize,
    pub csize: usize,
    /// the enclosing skippable chunk, if the blob sits inside one
    pub skip_chunk: Option<(u32, usize, usize, usize)>,
}

/// Locate the record node in a decompressed body (same walk as
/// `find_entrecord_blob`, but returning the site rather than the payload).
pub fn find_rec_site(body: &[u8]) -> Res<RecSite> {
    let needle = crate::record::CLASS_CPLUGENTRECORDDATA.to_le_bytes();
    let mut off = 0usize;
    loop {
        let Some(rel) = body[off..]
            .windows(4)
            .position(|w| w == needle)
        else {
            return Err("CPlugEntRecordData (0x0911F000) chunk not found".into());
        };
        let hit = off + rel;
        off = hit + 1;
        let mut q = hit;
        while q + 8 <= body.len() && body[q + 4..q + 8] == needle {
            q += 4;
        }
        let p = q + 4;
        if p + 12 > body.len() {
            continue;
        }
        let version = u32::from_le_bytes(body[p..p + 4].try_into().unwrap());
        let usize_ = u32::from_le_bytes(body[p + 4..p + 8].try_into().unwrap()) as usize;
        let csize = u32::from_le_bytes(body[p + 8..p + 12].try_into().unwrap()) as usize;
        if !(1..=20).contains(&version)
            || csize == 0
            || usize_ == 0
            || p + 12 + csize > body.len()
            || body[p + 12..p + 14] != [0x78, 0x9c]
        {
            continue;
        }
        let skip = crate::container::all_skip_chunks(body)
            .into_iter()
            .find(|(_, _, po, sz)| p >= *po && p + 12 + csize <= *po + *sz);
        return Ok(RecSite {
            hdr: p,
            version,
            usize_,
            csize,
            skip_chunk: skip,
        });
    }
}

fn zlib_compress(raw: &[u8]) -> Vec<u8> {
    // miniz_oxide, not flate2: the rest of the crate already decompresses with
    // miniz_oxide and this keeps the dependency list at one entry. Level 6 is
    // what the game itself writes.
    miniz_oxide::deflate::compress_to_vec_zlib(raw, 6)
}

/// Splice a new record payload into a body, fixing the enclosing chunk's size.
///
/// **When the record node has no enclosing skippable chunk, this ATTEMPTS the
/// write and then PROVES it, rather than refusing outright.**
///
/// The refusal existed because a skippable chunk carries a byte count that has
/// to absorb a payload-size change, and with no such chunk there is no count to
/// fix. But that reasoning only covers the case where something downstream
/// needs the count. A record node reached STRUCTURALLY -- version, uncompressed
/// size, compressed size, then that many bytes -- needs no outer count at all,
/// because every length it depends on is written inside the node itself, and
/// this function writes all three.
///
/// Refusing unconditionally therefore blocked a class of container the tools
/// could in fact write: search-created ghosts, whose record sits outside the
/// five class-id prefixes `all_skip_chunks` recognises. Those files could be
/// read, decoded and validated, but never spliced -- so `regen`, `split-car`
/// and every repair path died on them with a framing error.
///
/// The honest version is to write it and check: re-parse the body we just
/// produced, and require that the record we find is the record we intended. If
/// it does not read back, we refuse exactly as before and say so. That follows
/// the rule the rest of this toolchain runs on -- a command that writes a file
/// runs a control first -- instead of declining to try.
pub fn splice_record(body: &[u8], site: &RecSite, new_raw: &[u8]) -> Res<Vec<u8>> {
    let comp = zlib_compress(new_raw);
    let mut out = Vec::with_capacity(body.len() + comp.len());
    out.extend_from_slice(&body[..site.hdr]);
    out.extend_from_slice(&site.version.to_le_bytes());
    out.extend_from_slice(&(new_raw.len() as u32).to_le_bytes());
    out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
    out.extend_from_slice(&comp);
    out.extend_from_slice(&body[site.hdr + 12 + site.csize..]);
    // the skippable chunk that frames the node carries its own byte count
    if let Some((cid, coff, poff, sz)) = site.skip_chunk {
        let delta = comp.len() as i64 - site.csize as i64;
        let nsz = (sz as i64 + delta) as u32;
        out[coff + 8..coff + 12].copy_from_slice(&nsz.to_le_bytes());
        let _ = (cid, poff);
    } else if comp.len() != site.csize {
        // No enclosing count to fix. Prove the body still reads rather than
        // assuming it cannot.
        match find_rec_site(&out) {
            Ok(back) if back.hdr == site.hdr && back.csize == comp.len() && back.usize_ == new_raw.len() => {}
            Ok(back) => {
                return Err(format!(
                    "record node is not inside a skippable chunk, and after writing the new \
                     payload the node no longer reads back as itself (found version {} at {} \
                     with {} compressed bytes, expected {} at {} with {}): refusing to write a \
                     body whose framing would be wrong",
                    back.version, back.hdr, back.csize, site.version, site.hdr, comp.len()
                )
                .into());
            }
            Err(e) => {
                return Err(format!(
                    "record node is not inside a skippable chunk, and the body no longer parses \
                     after writing the new payload ({e}): refusing to write a body whose framing \
                     would be wrong"
                )
                .into());
            }
        }
    }
    Ok(out)
}

/// Read a ghost, hand its record to `f`, write the result to `out`.
///
/// The body is written UNCOMPRESSED (`'U'`), which the dedicated server accepts
/// (documented in the oracle notes) and which keeps this free of an LZO
/// compressor.
pub fn rewrite_ghost<F>(path: &str, out: &str, f: F) -> Res<(usize, usize)>
where
    F: FnOnce(&mut RecordData) -> Res<()>,
{
    let data = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let g = Gbx::parse(&data);
    let site = find_rec_site(&g.body)?;
    let blob = {
        miniz_oxide::inflate::decompress_to_vec_zlib(
            &g.body[site.hdr + 12..site.hdr + 12 + site.csize],
        )
        .map_err(|e| format!("zlib: {:?}", e))?
    };
    let mut rd = crate::record::parse_record_data(&blob, site.version)?;
    f(&mut rd)?;
    let raw = encode_record_data(&rd);
    let nb = splice_record(&g.body, &site, &raw)?;
    let mut file = g.header_bytes_u();
    file.extend_from_slice(&nb);
    std::fs::write(out, &file).map_err(|e| format!("{}: {}", out, e))?;
    Ok((blob.len(), raw.len()))
}

/// Encode a complete inline `CPlugEntRecordData` node from parsed fields.
///
/// The caller writes the surrounding node-reference index. This function owns
/// the node itself: class id, its v11 compressed chunk, and the node terminator.
/// Every byte is derived from [`RecordData`] plus GBX framing constants; no donor
/// chunk participates.
pub fn encode_record_node(rd: &RecordData) -> Vec<u8> {
    let raw = encode_record_data(rd);
    let comp = zlib_compress(&raw);
    let mut out = Vec::with_capacity(24 + comp.len());
    out.extend_from_slice(&crate::record::CLASS_CPLUGENTRECORDDATA.to_le_bytes());
    out.extend_from_slice(&crate::record::CLASS_CPLUGENTRECORDDATA.to_le_bytes());
    out.extend_from_slice(&rd.version.to_le_bytes());
    out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
    out.extend_from_slice(&comp);
    out.extend_from_slice(&0xFACADE01u32.to_le_bytes());
    out
}

// ---------------------------------------------------------------------------
// Writing one CSceneVehicleVis sample
// ---------------------------------------------------------------------------

/// The engine-side state a sample's transform is built from.
#[derive(Clone, Copy, Debug, Default)]
pub struct Xform {
    pub pos: [f32; 3],
    /// (x, y, z, w), unit norm
    pub quat: [f64; 4],
    pub vel: [f64; 3],
}

/// Inverse of `read_transform`: 22 bytes at `o`.
///
/// The recorded encoding is lossy by construction -- orientation is an angle in
/// u16 plus two axis angles in i16, speed is `exp(i16/1000)` and the velocity
/// DIRECTION is two i8s -- so a regenerated sample can only ever be as good as
/// that grid. `tmtraj rec reencode` measures the encoder against a real
/// ghost's own bytes: decode its samples, re-encode them, and count how many
/// come back identical. Anything less than "all of them" is a bug here, not a
/// property of the format.
pub fn write_transform(d: &mut [u8], o: usize, t: &Xform) {
    let pi = std::f64::consts::PI;
    for k in 0..3 {
        d[o + k * 4..o + k * 4 + 4].copy_from_slice(&t.pos[k].to_le_bytes());
    }
    // NO sign normalisation. q and -q are the same rotation, and the two
    // encodings (ang, ah, ap) and (pi-ang, ah-pi, -ap) both decode to it -- but
    // the game writes the quaternion it holds, sign and all: 143 of lqpzz's 474
    // samples carry qw < 0, i.e. an angle past pi/2. Forcing qw >= 0 here made
    // those 143 samples re-encode to different (still correct) bytes, which is
    // exactly the kind of "equivalent but not identical" that hides a real
    // error later. Encode what the engine holds.
    let q = t.quat;
    let ang = q[3].clamp(-1.0, 1.0).acos();
    let sa = ang.sin();
    let (ah, ap) = if sa.abs() < 1e-12 {
        (0.0, 0.0)
    } else {
        (
            q[1].atan2(q[0]),
            (q[2] / sa).clamp(-1.0, 1.0).asin(),
        )
    };
    let angu = (ang * 65535.0 / pi).round().clamp(0.0, 65535.0) as u16;
    let ahi = (ah * 32767.0 / pi).round().clamp(-32768.0, 32767.0) as i16;
    let api = (ap / (pi / 2.0) * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
    let speed = (t.vel[0] * t.vel[0] + t.vel[1] * t.vel[1] + t.vel[2] * t.vel[2]).sqrt();
    let spi = if speed > 0.0 {
        (1000.0 * speed.ln()).round().clamp(-32768.0, 32767.0) as i16
    } else {
        -32768i16
    };
    let (vh, vp) = if speed > 0.0 {
        (
            t.vel[1].atan2(t.vel[0]),
            (t.vel[2] / speed).clamp(-1.0, 1.0).asin(),
        )
    } else {
        (0.0, 0.0)
    };
    let vhi = (vh * 127.0 / pi).round().clamp(-128.0, 127.0) as i8;
    let vpi = (vp / (pi / 2.0) * 127.0).round().clamp(-128.0, 127.0) as i8;
    d[o + 12..o + 14].copy_from_slice(&angu.to_le_bytes());
    d[o + 14..o + 16].copy_from_slice(&ahi.to_le_bytes());
    d[o + 16..o + 18].copy_from_slice(&api.to_le_bytes());
    d[o + 18..o + 20].copy_from_slice(&spi.to_le_bytes());
    d[o + 20] = vhi as u8;
    d[o + 21] = vpi as u8;
}

/// A 3x3 rotation matrix (row-major) as an (x, y, z, w) quaternion.
pub fn mat_to_quat_pub(m: &[f64; 9]) -> [f64; 4] {
    let tr = m[0] + m[4] + m[8];
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [(m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s, 0.25 * s]
    } else if m[0] > m[4] && m[0] > m[8] {
        let s = (1.0 + m[0] - m[4] - m[8]).sqrt() * 2.0;
        [0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s, (m[7] - m[5]) / s]
    } else if m[4] > m[8] {
        let s = (1.0 + m[4] - m[0] - m[8]).sqrt() * 2.0;
        [(m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s, (m[2] - m[6]) / s]
    } else {
        let s = (1.0 + m[8] - m[0] - m[4]).sqrt() * 2.0;
        [(m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s, (m[3] - m[1]) / s]
    }
}
