//! Deterministic, machine-readable structural manifests for TM2020 ghosts.
//!
//! This is deliberately a view of parsed fields, not a byte diff. It records the
//! GBX header, every structurally framed chunk we can prove, the decoded input
//! packet shapes, and the complete `CPlugEntRecordData` grammar down to the first
//! vehicle sample. A file without a record is represented by `record: null`; it
//! is not an error and cannot be confused with a record containing zero entities.

use crate::container::{all_skip_chunks, Gbx};
use crate::record::{
    decode_vehicle_sample, find_entrecord_blob, parse_record_data, CLASS_CPLUGENTRECORDDATA,
    CLASS_CSCENEVEHICLEVIS,
};
use crate::recwrite::find_rec_site;
use crate::tape::{StateEnc, Tape};
use std::collections::BTreeMap;
use std::fmt::Write as _;

fn q(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c <= '\u{1f}' => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn hex(b: &[u8]) -> String {
    let mut out = String::with_capacity(2 * b.len());
    for x in b {
        write!(out, "{x:02x}").unwrap();
    }
    out
}

fn opt_i32(v: Option<i32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".into())
}

fn opt_u32(v: Option<u32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".into())
}

pub fn validation_manifest(body: &[u8]) -> Option<String> {
    let (_, _, mut o, n) = all_skip_chunks(body)
        .into_iter()
        .find(|c| c.0 == 0x0309_202D)?;
    let end = o + n;
    let u32_at = |o: &mut usize| -> Option<u32> {
        let v = u32::from_le_bytes(body.get(*o..*o + 4)?.try_into().ok()?);
        *o += 4;
        Some(v)
    };
    let string_at = |o: &mut usize| -> Option<String> {
        let n = u32::from_le_bytes(body.get(*o..*o + 4)?.try_into().ok()?) as usize;
        *o += 4;
        let s = String::from_utf8_lossy(body.get(*o..*o + n)?).into_owned();
        *o += n;
        Some(s)
    };
    let flag = u32_at(&mut o)?;
    if flag != 0 {
        return Some(format!(
            "{{\"u01\":{},\"unsupported_embedded_inputs\":true}}",
            flag
        ));
    }
    let exe = string_at(&mut o)?;
    let checksum = u32_at(&mut o)?;
    let os = u32_at(&mut o)? as i32;
    let cpu = u32_at(&mut o)? as i32;
    let wall_start = u32_at(&mut o)? as i32;
    let wall_end = u32_at(&mut o)? as i32;
    let title = string_at(&mut o)?;
    if o + 32 > end {
        return None;
    }
    let title_checksum = hex(&body[o..o + 32]);
    o += 32;
    let u02 = u32_at(&mut o)? as i32;
    let u03 = u32_at(&mut o)? as i32;
    let seed = u32_at(&mut o)? as i32;
    let u04 = u32_at(&mut o)? as i32;
    let settings = string_at(&mut o)?;
    Some(format!(
        "{{\"u01\":{},\"exe_version\":{},\"exe_checksum\":{},\"os_kind\":{},\"cpu_kind\":{},\"walltime_start\":{},\"walltime_end\":{},\"title_id\":{},\"title_checksum_hex\":{},\"settings_flags\":{},\"start_checkpoint_index\":{},\"u03\":{},\"validation_seed\":{},\"u04\":{},\"race_settings\":{},\"bytes_consumed\":{},\"bytes_total\":{}}}",
        flag, q(&exe), checksum, os, cpu, wall_start, wall_end, q(&title), q(&title_checksum),
        u02, u03, u03, seed, u04, q(&settings), o - (end - n), n
    ))
}

/// Build a deterministic JSON manifest from one complete GBX file.
///
/// The result contains no filesystem path or timestamp, so the same bytes produce
/// the same manifest wherever and whenever they are inspected.
pub fn manifest_bytes(data: &[u8]) -> Result<String, String> {
    if data.len() < 16 || data.get(..3) != Some(b"GBX") {
        return Err("not a GBX file".into());
    }
    let g = Gbx::parse(data);
    let mut out = String::new();
    out.push_str("{\"schema\":\"tm2020-ghost-structure-v1\"");
    write!(
        out,
        ",\"file_bytes\":{},\"header\":{{\"gbx_version\":{},\"format\":{},\"reference_compression\":{},\"body_compression\":{},\"unknown\":{},\"class_id\":\"0x{:08X}\",\"user_data_bytes\":{},\"num_nodes\":{},\"reference_table_bytes\":{},\"body_bytes\":{}}}",
        data.len(),
        g.version,
        data[5],
        data[6],
        data[7],
        g.unknown.map(|x| x.to_string()).unwrap_or_else(|| "null".into()),
        g.class_id,
        g.user_data.len(),
        g.num_nodes,
        g.ref_table.len(),
        g.body.len(),
    )
    .unwrap();

    #[derive(Clone)]
    struct Site {
        id: u32,
        off: usize,
        poff: usize,
        size: usize,
        framing: &'static str,
    }
    let mut sites: Vec<Site> = all_skip_chunks(&g.body)
        .into_iter()
        .map(|(id, off, poff, size)| Site {
            id,
            off,
            poff,
            size,
            framing: "skippable",
        })
        .collect();
    if let Ok(site) = find_rec_site(&g.body) {
        let off = site.hdr.saturating_sub(4);
        sites.push(Site {
            id: CLASS_CPLUGENTRECORDDATA,
            off,
            poff: site.hdr,
            size: 12 + site.csize,
            framing: "compressed-node",
        });
    }
    // These two chunks are known inline chunks in CGameCtnGhost. They have no
    // size word, so only record sites not already framed as a skippable chunk.
    for id in [0x0309_200F_u32, 0x0309_2010_u32] {
        let pat = id.to_le_bytes();
        for off in 0..g.body.len().saturating_sub(3) {
            if g.body[off..off + 4] == pat
                && !sites.iter().any(|s| s.off == off)
                && !sites
                    .iter()
                    .any(|s| s.framing == "skippable" && off > s.off && off < s.poff + s.size)
            {
                sites.push(Site {
                    id,
                    off,
                    poff: off + 4,
                    size: 0,
                    framing: "inline",
                });
            }
        }
    }
    sites.sort_by_key(|s| (s.off, s.id));
    out.push_str(",\"chunks\":[");
    for (i, s) in sites.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write!(
            out,
            "{{\"id\":\"0x{:08X}\",\"offset\":{},\"payload_offset\":{},\"payload_bytes\":{},\"framing\":{}}}",
            s.id, s.off, s.poff, s.size, q(s.framing)
        )
        .unwrap();
    }
    out.push(']');
    match validation_manifest(&g.body) {
        Some(v) => write!(out, ",\"validation\":{v}").unwrap(),
        None => out.push_str(",\"validation\":null"),
    }

    match Tape::from_body(&g.body) {
        Err(e) => write!(out, ",\"tape\":null,\"tape_error\":{}", q(&e)).unwrap(),
        Ok(t) => {
            write!(
                out,
                ",\"tape\":{{\"chunk_version\":{},\"archives\":[",
                t.chunk_version
            )
            .unwrap();
            for (ai, a) in t.archives.iter().enumerate() {
                if ai > 0 {
                    out.push(',');
                }
                let mut modes: BTreeMap<u32, usize> = BTreeMap::new();
                let (mut lit, mut prev, mut prev2) = (0usize, 0usize, 0usize);
                for p in &a.packets {
                    *modes.entry(p.mode).or_default() += 1;
                    match p.state {
                        StateEnc::Lit(_) => lit += 1,
                        StateEnc::Prev => prev += 1,
                        StateEnc::Prev2(_, _) => prev2 += 1,
                    }
                }
                write!(
                    out,
                    "{{\"format_version\":{},\"field0\":{},\"start_offset_ms\":{},\"packets\":{},\"bitstream_bytes\":{},\"bits_used\":{},\"tail_bytes\":{},\"mode_histogram\":{{",
                    a.format_version, a.field0, a.start_offset_ms, a.packets.len(), a.orig_bitstream_len,
                    a.orig_bits_used, a.tail.len()
                )
                .unwrap();
                for (mi, (mode, n)) in modes.iter().enumerate() {
                    if mi > 0 {
                        out.push(',');
                    }
                    write!(out, "{}:{}", q(&mode.to_string()), n).unwrap();
                }
                let first_packet = match a.packets.first() {
                    None => "null".to_string(),
                    Some(p) => {
                        let state = match p.state {
                            StateEnc::Lit(x) => q(&format!("literal:0x{x:09X}")),
                            StateEnc::Prev => q("previous"),
                            StateEnc::Prev2(x, y) => q(&format!("previous2:{x},{y}")),
                        };
                        format!(
                            "{{\"word0\":{},\"flags\":{},\"mode\":{},\"state\":{},\"vehicle_same\":{},\"mouse\":{},\"triggers\":{},\"steer_raw\":{},\"steer_i8\":{},\"accel\":{},\"brake\":{},\"respawn\":{}}}",
                            p.word0,
                            p.flags,
                            p.mode,
                            state,
                            p.vsame,
                            p.mouse
                                .map(|(x, y)| format!("[{x},{y}]"))
                                .unwrap_or_else(|| "null".into()),
                            p.tri
                                .map(|x| format!("[{},{},{},{}]", x[0], x[1], x[2], x[3]))
                                .unwrap_or_else(|| "null".into()),
                            p.steer,
                            p.steer_i8(),
                            p.accel,
                            p.brake,
                            p.respawn(),
                        )
                    }
                };
                write!(
                    out,
                    "}},\"payload_shapes\":{{\"state_literal\":{},\"state_prev\":{},\"state_prev2\":{},\"vehicle_same\":{},\"mouse\":{},\"triggers\":{},\"respawn\":{},\"accel_on\":{},\"brake_on\":{},\"steer32\":{}}},\"first_packet\":{}}}",
                    lit,
                    prev,
                    prev2,
                    a.packets.iter().filter(|p| p.vsame).count(),
                    a.packets.iter().filter(|p| p.mouse.is_some()).count(),
                    a.packets.iter().filter(|p| p.tri.is_some()).count(),
                    a.packets.iter().filter(|p| p.respawn()).count(),
                    a.packets.iter().filter(|p| p.accel != 0).count(),
                    a.packets.iter().filter(|p| p.brake != 0).count(),
                    a.packets.iter().filter(|p| matches!(p.mode, 12 | 13)).count(),
                    first_packet,
                )
                .unwrap();
            }
            out.push_str("]}");
        }
    }

    match find_entrecord_blob(&g.body) {
        Err(e) => write!(out, ",\"record\":null,\"record_error\":{}", q(&e)).unwrap(),
        Ok((version, blob)) => {
            let rd = parse_record_data(&blob, version)?;
            write!(
                out,
                ",\"record\":{{\"version\":{},\"start_ms\":{},\"end_ms\":{},\"bytes_consumed\":{},\"bytes_total\":{},\"descriptors\":[",
                rd.version, rd.start_ms, rd.end_ms, rd.bytes_consumed, rd.bytes_total
            )
            .unwrap();
            for (i, d) in rd.descs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write!(
                    out,
                    "{{\"class_id\":\"0x{:08X}\",\"u01\":{},\"u02\":{},\"u03\":{},\"u04_bytes\":{},\"u04_hex\":{},\"u05\":{}}}",
                    d.class_id, d.u01, d.u02, d.u03, d.u04.len(), q(&hex(&d.u04)), d.u05
                )
                .unwrap();
            }
            out.push_str("],\"notices\":[");
            for (i, (a, b, c)) in rd.notices.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write!(out, "[{}, {}, {}]", a, b, opt_u32(*c)).unwrap();
            }
            out.push_str("],\"entities\":[");
            for (i, e) in rd.ents.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let class_id = if e.type_ >= 0 {
                    rd.descs.get(e.type_ as usize).map(|d| d.class_id)
                } else {
                    None
                };
                write!(
                    out,
                    "{{\"type\":{},\"class_id\":{},\"u01\":{},\"u02\":{},\"u03\":{},\"u04\":{},\"samples\":{},\"sample_bytes\":{},\"first_ms\":{},\"last_ms\":{},\"deltas2\":[",
                    e.type_,
                    class_id.map(|x| q(&format!("0x{x:08X}"))).unwrap_or_else(|| "null".into()),
                    e.u01, e.u02, e.u03, e.u04, e.times.len(), e.sample_size,
                    opt_i32(e.times.first().copied()), opt_i32(e.times.last().copied())
                )
                .unwrap();
                for (j, (ty, time, payload)) in e.deltas2.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    write!(
                        out,
                        "{{\"type\":{},\"time_ms\":{},\"bytes\":{}}}",
                        ty,
                        time,
                        payload.len()
                    )
                    .unwrap();
                }
                out.push_str("]}");
            }
            out.push_str("],\"bulk_notices\":[");
            for (i, (a, b, d)) in rd.bulk_notices.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write!(out, "{{\"u01\":{},\"u02\":{},\"bytes\":{}}}", a, b, d.len()).unwrap();
            }
            out.push_str("],\"custom_modules\":[");
            for (i, cm) in rd.custom_modules.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write!(out, "{{\"period\":{},\"deltas\":[", opt_i32(cm.period)).unwrap();
                for (j, (a, d, d2)) in cm.deltas.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    write!(
                        out,
                        "{{\"u01\":{},\"bytes\":{},\"bytes2\":{}}}",
                        a,
                        d.len(),
                        d2.len()
                    )
                    .unwrap();
                }
                out.push_str("]}");
            }
            out.push(']');

            let first = rd
                .ents
                .iter()
                .filter(|e| {
                    e.type_ >= 0
                        && rd.descs.get(e.type_ as usize).map(|d| d.class_id)
                            == Some(CLASS_CSCENEVEHICLEVIS)
                        && e.sample_size >= 103
                        && e.raw.len() >= e.sample_size
                })
                .max_by_key(|e| e.times.len());
            match first {
                None => out.push_str(",\"first_vehicle_sample\":null"),
                Some(e) => {
                    let raw = &e.raw[..e.sample_size];
                    let s = decode_vehicle_sample(raw);
                    write!(
                        out,
                        ",\"first_vehicle_sample\":{{\"time_ms\":{},\"position\":[{:.9},{:.9},{:.9}],\"quaternion_xyzw\":[{:.12},{:.12},{:.12},{:.12}],\"velocity\":[{:.9},{:.9},{:.9}],\"speed_mps\":{:.9},\"gear_raw\":{},\"rpm_raw\":{},\"ground_mode_raw\":{},\"wetness_raw\":{},\"raw_hex\":{}}}",
                        e.times.first().copied().unwrap_or(0),
                        s.x, s.y, s.z, s.qx, s.qy, s.qz, s.qw, s.vx, s.vy, s.vz, s.speed_ms,
                        s.gear_raw, s.rpm_raw, s.ground_mode_raw, raw.get(101).copied().unwrap_or(0), q(&hex(raw))
                    )
                    .unwrap();
                }
            }
            out.push('}');
        }
    }

    out.push('}');
    Ok(out)
}

pub fn manifest_file(path: &str) -> Result<String, String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    manifest_bytes(&data)
}

/// A machine-readable differential. Both parsed manifests are embedded rather
/// than reduced to a lossy list, so a downstream consumer can ask any structural
/// question without reopening either GBX file.
pub fn diff_files(left: &str, right: &str) -> Result<String, String> {
    let a = manifest_file(left)?;
    let b = manifest_file(right)?;
    Ok(format!(
        "{{\"schema\":\"tm2020-ghost-struct-diff-v1\",\"equal\":{},\"left\":{},\"right\":{}}}",
        a == b,
        a,
        b
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> String {
        format!(
            "{}/../testdata/human_22730.Ghost.Gbx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    #[test]
    fn a_real_recording_exposes_the_record_and_packet_histogram() {
        let m = manifest_file(&fixture()).expect("fixture manifest");
        assert!(m.starts_with("{\"schema\":\"tm2020-ghost-structure-v1\""));
        assert!(m.contains("\"record\":{\"version\":11"));
        assert!(m.contains("\"class_id\":\"0x0A018000\""));
        assert!(m.contains("\"sample_bytes\":116"));
        assert!(m.contains("\"mode_histogram\""));
        assert!(m.contains("\"first_vehicle_sample\":{\"time_ms\":0"));
    }

    #[test]
    fn the_diff_is_symmetric_data_not_a_textual_byte_diff() {
        let p = fixture();
        let d = diff_files(&p, &p).expect("diff");
        assert!(d.contains("\"equal\":true"));
        assert_eq!(d.matches("tm2020-ghost-structure-v1").count(), 2);
    }
}
