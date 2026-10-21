//! `ghost record` -- operations on the telemetry record itself, as a container
//! for samples, independent of what the samples say.
//!
//! `ghost regen` writes engine state into the record a file ALREADY HAS. That
//! is the right shape when the carrier's record covers the run, and 34 of the
//! project's 171 published ghosts show why it usually does: a transplanted tape
//! inherits a carrier that is a LONGER recording than our run.
//!
//! It is the wrong shape when the carrier's record does not cover the run, and
//! two of this project's headline results are exactly that case:
//!
//!   * **227654** -- the carrier is a 27-player multiplayer replay whose
//!     largest vehicle entity holds 365 samples spanning race 1.310 .. 19.480,
//!     for a run that finishes at 57.482. There is no grid to write 38 seconds
//!     of car onto, so `regen` hard-errors on the missing in-race samples --
//!     correctly, and with no way forward. It also carries 26 OTHER PEOPLE'S
//!     CARS, which a render would draw.
//!   * **186935** -- `BEST_793893` has a record node with no `CSceneVehicleVis`
//!     entity at all. Nothing to render, and nothing to write into.
//!
//! `rebuild` makes the grid: it throws the vehicle entities away and lays down
//! a fresh one, `0, period, 2*period, ...` out to the span, every sample a copy
//! of one template sample. That file is not yet a run -- every sample is
//! identical, so it is a car frozen at one point, and `regen`'s own G5 refuses
//! exactly that. It is scaffolding, and the next step is
//! `ghost regen --neutralise --inputs`, which writes the transform from engine
//! state, the input echo from the tape, and zeros every other per-run byte.
//!
//! **Which is what makes the template harmless.** After a neutralised regen the
//! only bytes still carrying the template's values are the format constants
//! that are identical in every ghost of every driver. So the template is taken
//! from the file's own record by default, and the two steps together produce a
//! record with no per-run byte from anybody.
//!
//! What is NOT touched: the input chunk, the declared time, the checkpoint
//! chunk, the uid, the identity strings. The oracle reads the input archive, so
//! rebuilding the record cannot change what the file simulates to -- and
//! `ghost verify` re-simulates the written file to prove that rather than
//! assuming it.

use gbx::container::secs;
use gbx::record::{Ent, RecordData};
use gbx::recwrite::rewrite_ghost;
use crate::cli::{has, die, flag, num};

/// The vehicle entity to rebuild from: the one with the most samples among
/// those whose samples are big enough to be a `CSceneVehicleVis` (>= 100 B).
/// Same rule as `fk`'s reader, so the two cannot disagree about which entity is
/// the car.
fn pick_vehicle(rd: &RecordData) -> Option<usize> {
    rd.ents
        .iter()
        .enumerate()
        .filter(|(_, e)| e.sample_size >= 100 && !e.times.is_empty())
        .max_by_key(|(_, e)| e.times.len())
        .map(|(i, _)| i)
}

pub fn cmd(a: &[String]) {
    match a.first().map(String::as_str) {
        Some("seed-x") => {
            let inp = a.get(1).unwrap_or_else(|| die("ghost record seed-x IN OUT --dx METRES"));
            let out = a.get(2).unwrap_or_else(|| die("ghost record seed-x IN OUT --dx METRES"));
            let dx: f32 = flag(a, "--dx")
                .unwrap_or_else(|| die("--dx METRES"))
                .parse()
                .unwrap_or_else(|_| die("--dx wants metres"));
            match seed_x(inp, out, dx) {
                Ok(msg) => println!("{out}: {msg}"),
                Err(e) => die(e),
            }
        }
        Some("chain") => {
            let p = a.get(1).unwrap_or_else(|| die("ghost record chain FILE"));
            crate::splice::print_chain(p);
        }
        Some("countdown") => {
            let inp = a.get(1).unwrap_or_else(|| die("ghost record countdown IN OUT"));
            let out = a.get(2).unwrap_or_else(|| die("ghost record countdown IN OUT"));
            let check = has(a, "--check");
            match countdown(inp, out, check) {
                Ok(m) => println!("{m}"),
                Err(e) => die(e),
            }
        }
        Some("dumpbytes") => {
            let p = a
                .get(1)
                .unwrap_or_else(|| die("ghost record dumpbytes FILE --bytes LIST [--stride N]"));
            let bytes = parse_byte_ranges(
                &flag(a, "--bytes").unwrap_or_else(|| die("--bytes N,N,A..B")),
            );
            let stride: usize =
                flag(a, "--stride").map(|s| s.parse().unwrap_or(1)).unwrap_or(1).max(1);
            let ent: Option<usize> = flag(a, "--ent").map(|s| s.parse().unwrap_or(0));
            match dumpbytes(p, &bytes, stride, ent) {
                Ok(m) => print!("{m}"),
                Err(e) => die(e),
            }
        }
        Some("bytediff") => {
            let l = a.get(1).unwrap_or_else(|| die("ghost record bytediff LEFT RIGHT"));
            let r = a.get(2).unwrap_or_else(|| die("ghost record bytediff LEFT RIGHT"));
            match bytediff(l, r) {
                Ok(m) => print!("{m}"),
                Err(e) => die(e),
            }
        }
        Some("life") => {
            let p = a.get(1).unwrap_or_else(|| die("ghost record life FILE --life N [--csv OUT]"));
            let n: usize = flag(a, "--life")
                .unwrap_or_else(|| die("--life N"))
                .parse()
                .unwrap_or_else(|_| die("--life wants a number"));
            crate::splice::life_csv(p, n, flag(a, "--csv"), flag(a, "--shift-ms").map(|s| s.parse().unwrap_or_else(|_| die("--shift-ms wants a number"))).unwrap_or(0));
        }
        Some("rebuild") => rebuild(&a[1..]),
        Some("resegment") => {
            let inp = a.get(1).unwrap_or_else(|| die("ghost record resegment IN OUT --like DONOR"));
            let out = a.get(2).unwrap_or_else(|| die("ghost record resegment IN OUT --like DONOR"));
            let donor = flag(a, "--like")
                .unwrap_or_else(|| die("--like DONOR.Ghost.Gbx: the container whose segment boundaries to copy"));
            match resegment(inp, out, &donor) {
                Ok(m) => println!("{out}: {m}"),
                Err(e) => die(e),
            }
        }
        Some("shorten") => {
            let inp = a.get(1).unwrap_or_else(|| die("ghost record shorten IN OUT"));
            let out = a.get(2).unwrap_or_else(|| die("ghost record shorten IN OUT"));
            match shorten_scene(inp, out) {
                Ok(m) => println!("{out}: {m}"),
                Err(e) => die(e),
            }
        }
        Some("channels") => {
            let inp = a.get(1).unwrap_or_else(|| {
                die("ghost record channels IN OUT [--from DONOR --bytes N,N] [--set N=V,N=V]")
            });
            let out = a.get(2).unwrap_or_else(|| {
                die("ghost record channels IN OUT [--from DONOR --bytes N,N] [--set N=V,N=V]")
            });
            if let Some(set) = flag(a, "--set") {
                let pairs: Vec<(usize, u8)> = set
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        let (b, v) = s.split_once('=').unwrap_or_else(|| {
                            die(format!("--set wants N=V pairs, got {s:?}"))
                        });
                        (
                            b.trim().parse().unwrap_or_else(|_| die(format!("--set: {b} is not a byte index"))),
                            v.trim().parse().unwrap_or_else(|_| die(format!("--set: {v} is not a 0..255 value"))),
                        )
                    })
                    .collect();
                match set_channels(inp, out, &pairs) {
                    Ok(m) => println!("{out}: {m}"),
                    Err(e) => die(e),
                }
            } else {
                let donor = flag(a, "--from").unwrap_or_else(|| {
                    die("--from DONOR.Ghost.Gbx: the file to take the named sample bytes from")
                });
                // RANGES, the same `19,47..69,91` spelling `resample --bytes`
                // already takes. A bisect over the 94 bytes `regen` does NOT
                // write is a range and a range's complement; spelling those out
                // as 94 comma-separated integers on a command line is how an
                // off-by-one gets into a measurement unseen.
                let list = flag(a, "--bytes")
                    .unwrap_or_else(|| die("--bytes N,N,A..B : which of the 116 sample bytes to copy"));
                let bytes: Vec<usize> = parse_byte_ranges(&list);
                match copy_channels(inp, out, donor, &bytes) {
                    Ok(m) => println!("{out}: {m}"),
                    Err(e) => die(e),
                }
            }
        }
        Some("blob") => {
            // The DECOMPRESSED CPlugEntRecordData payload, straight to a file.
            // A rewrite that changes the container's bytes may be changing
            // nothing but the zlib stream, or it may be changing the record.
            // Those are very different facts and only this tells them apart.
            let inp = a.get(1).unwrap_or_else(|| die("ghost record blob FILE OUT"));
            let out = a.get(2).unwrap_or_else(|| die("ghost record blob FILE OUT"));
            let body = gbx::record::load_body(inp).unwrap_or_else(|e| die(e));
            let (ver, blob) =
                gbx::record::find_entrecord_blob(&body).unwrap_or_else(|e| die(e));
            std::fs::write(out, &blob).unwrap_or_else(|e| die(e.to_string()));
            println!("{out}: version {ver}, {} bytes", blob.len());
        }
        Some("ents") => {
            // Keep or drop whole entities, changing nothing else. This is the
            // bisect between a record the client ACCEPTS and one it dies on:
            // 227654's container imports with 29 entities and our rebuild dies
            // with 1, and the only honest way to find which property matters is
            // to walk one file toward the other one entity at a time.
            let inp = a
                .get(1)
                .unwrap_or_else(|| die("ghost record ents IN OUT --keep LIST | --drop LIST"));
            let out = a
                .get(2)
                .unwrap_or_else(|| die("ghost record ents IN OUT --keep LIST | --drop LIST"));
            let keep = flag(a, "--keep").map(|s| parse_byte_ranges(&s));
            let drop = flag(a, "--drop").map(|s| parse_byte_ranges(&s));
            let dup = flag(a, "--dup").map(|s| parse_byte_ranges(&s));
            let pad: Option<usize> = flag(a, "--pad").map(|s| s.parse().expect("--pad N"));
            if keep.is_some() == drop.is_some() && dup.is_none() && pad.is_none() {
                die("ghost record ents needs --keep LIST, --drop LIST or --dup LIST \
                     (indices as `ghost record show` prints them, e.g. 2,3 or 2..29)");
            }
            let mut note = String::new();
            match rewrite_ghost(inp, out, |rd| {
                let n = rd.ents.len();
                let survive: Vec<usize> = match (&keep, &drop) {
                    (Some(k), _) => k.iter().copied().filter(|i| *i < n).collect(),
                    (_, Some(d)) => (0..n).filter(|i| !d.contains(i)).collect(),
                    _ => (0..n).collect(),
                };
                if survive.is_empty() {
                    return Err("that would leave the record with no entity at all".into());
                }
                let mut kept = Vec::new();
                for i in &survive {
                    kept.push(rd.ents[*i].clone());
                    // --dup puts a second copy of an entity right after it, so
                    // that "the count changed" can be tested in the direction
                    // that removes nothing.
                    if dup.as_ref().is_some_and(|d| d.contains(i)) {
                        kept.push(rd.ents[*i].clone());
                    }
                }
                // --pad N appends copies of the last entity until the record
                // holds N of them. On 227654 the client's threshold is a COUNT:
                // 28 entities crashes and 29 imports, whichever 29 they are.
                if let Some(p) = pad {
                    while kept.len() < p {
                        kept.push(kept[kept.len() - 1].clone());
                    }
                }
                note = format!(
                    "{} of {} entities kept: {:?}{}",
                    kept.len(),
                    n,
                    survive,
                    dup.as_ref()
                        .map(|d| format!(", duplicated {d:?}"))
                        .unwrap_or_default()
                );
                rd.ents = kept;
                Ok(())
            }) {
                Ok(_) => println!("{out}: {note}"),
                Err(e) => die(e),
            }
        }
        Some("show") => show(&a[1..]),
        Some("graft-scene") => {
            let inp = a.get(1).unwrap_or_else(|| {
                die("ghost record graft-scene IN OUT --from DONOR.Ghost.Gbx")
            });
            let out = a.get(2).unwrap_or_else(|| {
                die("ghost record graft-scene IN OUT --from DONOR.Ghost.Gbx")
            });
            let donor = flag(a, "--from")
                .unwrap_or_else(|| die("--from DONOR.Ghost.Gbx: the container the car was rebuilt out of"));
            match graft_scene(inp, out, donor, crate::cli::has(a, "--car-deltas")) {
                Ok(m) => println!("{out}: {m}"),
                Err(e) => die(e),
            }
        }
        Some("from-csv") => {
            let inp = a.get(1).unwrap_or_else(|| {
                die("ghost record from-csv IN OUT --csv TRACE.csv [--tol MS]")
            });
            let out = a.get(2).unwrap_or_else(|| {
                die("ghost record from-csv IN OUT --csv TRACE.csv [--tol MS]")
            });
            let csv = flag(a, "--csv").unwrap_or_else(|| {
                die("ghost record from-csv needs --csv TRACE.csv -- a trajectory CSV with \
                     time_ms,x,y,z,vx,vy,vz,qx,qy,qz,qw, the shape `fk trace` and \
                     `tmtraj export` write")
            });
            let tol: i32 = flag(a, "--tol").map_or(25, |s| s.parse().expect("--tol MS"));
            match from_csv(inp, out, &csv, tol) {
                Ok(m) => println!("{out}: {m}"),
                Err(e) => die(e),
            }
        }
        Some("resample") => {
            let inp = a.get(1).unwrap_or_else(|| {
                die("ghost record resample IN OUT --from SRC.Ghost.Gbx")
            });
            let out = a.get(2).unwrap_or_else(|| {
                die("ghost record resample IN OUT --from SRC.Ghost.Gbx")
            });
            let src = flag(a, "--from").unwrap_or_else(|| {
                die("ghost record resample needs --from SRC.Ghost.Gbx -- the file whose CAR \
                     SAMPLES you want, written into IN's record")
            });
            let opt = ResampleOpts {
                all_cars: crate::cli::has(a, "--all-cars"),
                report: crate::cli::has(a, "--report"),
                mixed_run: crate::cli::has(a, "--mixed-run"),
                bytes: flag(a, "--bytes").map(|s| parse_byte_ranges(&s)),
                fill_tol_ms: flag(a, "--fill-tol").map_or(0, |s| s.parse().expect("--fill-tol MS")),
                hold_last: crate::cli::has(a, "--hold-last"),
            };
            match resample_opt(inp, out, &src, &opt) {
                Ok(m) => println!("{out}: {m}"),
                Err(e) => die(e),
            }
        }
        Some("entorder") => {
            let inp = a.get(1).unwrap_or_else(|| {
                die("ghost record entorder IN OUT --car-first | --car-last")
            });
            let out = a.get(2).unwrap_or_else(|| {
                die("ghost record entorder IN OUT --car-first | --car-last")
            });
            let first = crate::cli::has(a, "--car-first");
            let last = crate::cli::has(a, "--car-last");
            if first == last {
                die("exactly one of --car-first or --car-last -- the two directions of the same \
                     experiment, and a run that does neither, or both, measures nothing");
            }
            match set_ent_order(inp, out, first) {
                Ok(m) => println!("{out}: {m}"),
                Err(e) => die(e),
            }
        }
        Some("notices") => {
            let inp = a.get(1).unwrap_or_else(|| {
                die("ghost record notices IN OUT --from DONOR.Ghost.Gbx | --strip")
            });
            let out = a.get(2).unwrap_or_else(|| {
                die("ghost record notices IN OUT --from DONOR.Ghost.Gbx | --strip")
            });
            let from = flag(a, "--from");
            let strip = crate::cli::has(a, "--strip");
            if from.is_some() == strip {
                die("exactly one of --from DONOR (restore) or --strip (remove) -- the two \
                     directions of the same experiment, and a run that does neither, or both, \
                     measures nothing");
            }
            match set_notices(inp, out, from, strip) {
                Ok(m) => println!("{out}: {m}"),
                Err(e) => die(e),
            }
        }
        Some("entfields") => {
            let inp = a.get(1).unwrap_or_else(|| {
                die("ghost record entfields IN OUT [--u01 N] [--u02 N] [--u04 N]")
            });
            let out = a.get(2).unwrap_or_else(|| {
                die("ghost record entfields IN OUT [--u01 N] [--u02 N] [--u04 N]")
            });
            let g = |k: &str| num(a, k).map(|v| v as i32);
            match set_ent_fields(inp, out, g("--u01"), g("--u02"), g("--u04")) {
                Ok(m) => println!("{out}: {m}"),
                Err(e) => die(e),
            }
        }
        _ => die(
            "ghost record seed-x IN OUT --dx METRES\n\
             \x20                            -- causal probe: move only the first vehicle sample's x\n\
             ghost record rebuild IN OUT --span MS [--period MS] [--template N] [--hold-last]\n\
             \x20                            -- --hold-last extends a record the car already\n\
             \x20                               holds by repeating its last sample: the car\n\
             \x20                               parked where it finished\n\
             ghost record resegment IN OUT --like DONOR\n\
             \x20                            -- re-cut ONE rebuilt car entity into the segment\n\
             \x20                               boundaries a donor container uses, keeping every\n\
             \x20                               sample byte. Some maps change the vehicle and\n\
             \x20                               record one entity per segment\n\
             ghost record shorten IN OUT   -- make the scene end when the car does,\n\
             \x20                                without touching the car's samples\n\
             ghost record entfields IN OUT [--u01 N] [--u02 N] [--u04 N]\n\
             \x20                            -- the vehicle entity's UNIDENTIFIED header\n\
             \x20                               fields, for the client-import bisect\n\
             ghost record graft-scene IN OUT --from DONOR\n\
             \x20                            -- put the container's non-vehicle entities back\n\
             ghost record notices IN OUT --from DONOR | --strip\n\
             \x20                            -- restore or remove the record's notice lists:\n\
             \x20                               the two directions of one experiment\n\
             ghost record entorder IN OUT --car-first | --car-last\n\
             \x20                            -- move the car to the front or the back: in every\n\
             \x20                               ghost the game wrote it is entity 0\n\
             ghost record ents IN OUT --keep LIST | --drop LIST | --dup LIST | --pad N\n\
             \x20                            -- keep, drop, duplicate or PAD the entity list and\n\
             \x20                               change nothing else. 227654's client needs 29 of\n\
             \x20                               them and does not care which: our own 1-entity\n\
             \x20                               record imports at `--pad 29` and dies at 28\n\
             ghost record resample IN OUT --from SRC [--all-cars] [--mixed-run]\n\
             \x20                         [--fill-tol MS] [--hold-last] [--bytes LIST] [--report]\n\
             \x20                            -- put SRC's car samples into IN's record, instant\n\
             \x20                               for instant, changing nothing else. The way to\n\
             \x20                               film a run whose own record the client refuses\n\
             ghost record from-csv IN OUT --csv TRACE.csv [--tol MS]\n\
             \x20                            -- write a TRAJECTORY CSV's position, orientation\n\
             \x20                               and velocity into IN's record, instant for\n\
             \x20                               instant. The way to film a run whose engine\n\
             \x20                               trace exists as a CSV and whose own container\n\
             \x20                               the client refuses: rebuild a container the\n\
             \x20                               client accepts to the run's span, then write\n\
             \x20                               the run into it. Instants outside the CSV's own\n\
             \x20                               span HOLD its first/last row, and the count is\n\
             \x20                               reported rather than hidden\n\
             ghost record blob FILE OUT   -- the DECOMPRESSED record payload, for telling a\n\
             \x20                               changed zlib stream from a changed record\n\
             ghost record show FILE",
        ),
    }
}

fn seed_x(inp: &str, out: &str, dx: f32) -> Result<String, String> {
    let mut before = 0.0f32;
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity")?;
        let e = &mut rd.ents[vi];
        if e.sample_size < 59 || e.raw.len() < e.sample_size {
            return Err("the vehicle has no complete first transform".into());
        }
        before = f32::from_le_bytes(e.raw[47..51].try_into().unwrap());
        e.raw[47..51].copy_from_slice(&(before + dx).to_le_bytes());
        Ok(())
    })?;
    let back = gbx::record::decode_ghost(out)?;
    let got = back.samples.first().ok_or("written file has no first sample")?.x;
    if (got as f64 - (before + dx) as f64).abs() > 1e-5 {
        let _ = std::fs::remove_file(out);
        return Err("first-sample x did not read back at the requested value".into());
    }
    Ok(format!(
        "first vehicle sample x {before:.6} -> {:.6}; every other field untouched",
        before + dx
    ))
}

/// Write a trajectory CSV into a container's car record, instant for instant.
///
/// This exists because a run can have a TRUE trajectory and no container the
/// client will import. Haystack 6's 278-move route is the case: the tape's own
/// synthesised container carries 6 skippable chunks where a game-recorded one
/// carries 24, and the client dies at `Trackmania.exe+0xd3788a` on import
/// whatever the record holds -- measured with the map and a game-recorded ghost
/// as controls, so it is the container and not the map, the length, the sample
/// count or the entity count. `ghost record rebuild` a container the client
/// DOES accept out to the run's span, write the run into it with this, and the
/// scene is the run.
///
/// It is not a regeneration and it does not pretend to be: nothing here runs an
/// engine. The CSV is the answer somebody else measured, and this command's
/// whole job is to put it in the file without changing it. So it reads back the
/// file it wrote and refuses unless the first and last written samples decode
/// to the positions the CSV asked for -- the encoding is lossy by construction
/// (`write_transform` says how), and a claim that the record IS the trace has to
/// be checked against the bytes on disk rather than against the intent.
pub fn from_csv(inp: &str, out: &str, csv: &str, tol_ms: i32) -> Result<String, String> {
    let text = std::fs::read_to_string(csv).map_err(|e| format!("{csv}: {e}"))?;
    let mut lines = text.lines();
    let hdr = lines.next().ok_or_else(|| format!("{csv} is empty"))?;
    let cols: Vec<&str> = hdr.split(',').map(str::trim).collect();
    let idx = |n: &str| {
        cols.iter()
            .position(|c| *c == n)
            .ok_or_else(|| format!("{csv} has no column {n:?}; it has: {}", cols.join(",")))
    };
    let (it, ix, iy, iz) = (idx("time_ms")?, idx("x")?, idx("y")?, idx("z")?);
    let (ivx, ivy, ivz) = (idx("vx")?, idx("vy")?, idx("vz")?);
    let (iqx, iqy, iqz, iqw) = (idx("qx")?, idx("qy")?, idx("qz")?, idx("qw")?);
    let need = *[it, ix, iy, iz, ivx, ivy, ivz, iqx, iqy, iqz, iqw]
        .iter()
        .max()
        .unwrap();

    let mut rows: Vec<(i32, gbx::recwrite::Xform)> = Vec::new();
    for (n, l) in lines.enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split(',').collect();
        if f.len() <= need {
            return Err(format!("{csv}:{}: {} fields, need {}", n + 2, f.len(), need + 1));
        }
        let g = |i: usize| -> Result<f64, String> {
            f[i].trim()
                .parse::<f64>()
                .map_err(|_| format!("{csv}:{}: {:?} is not a number", n + 2, f[i].trim()))
        };
        let q = [g(iqx)?, g(iqy)?, g(iqz)?, g(iqw)?];
        let nq = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if !nq.is_finite() || nq < 1e-6 {
            return Err(format!("{csv}:{}: the quaternion is not a rotation", n + 2));
        }
        let p = [g(ix)? as f32, g(iy)? as f32, g(iz)? as f32];
        if p.iter().any(|v| !v.is_finite()) {
            return Err(format!("{csv}:{}: a non-finite position", n + 2));
        }
        rows.push((
            g(it)?.round() as i32,
            gbx::recwrite::Xform {
                pos: p,
                quat: [q[0] / nq, q[1] / nq, q[2] / nq, q[3] / nq],
                vel: [g(ivx)?, g(ivy)?, g(ivz)?],
            },
        ));
    }
    if rows.is_empty() {
        return Err(format!("{csv} has no rows"));
    }
    rows.sort_by_key(|r| r.0);
    let times: Vec<i32> = rows.iter().map(|r| r.0).collect();
    let (lo, hi) = (times[0], times[times.len() - 1]);

    let mut wrote = 0usize;
    let mut held = 0usize;
    let mut worst_dt = 0i32;
    let mut nsamples = 0usize;
    let mut first_last: Vec<(usize, usize)> = Vec::new(); // (sample index, row index)
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity to write into")?;
        let e = &mut rd.ents[vi];
        let ss = e.sample_size;
        // `write_transform` owns 22 bytes at 47.
        if ss < 69 {
            return Err(format!("the vehicle's samples are {ss} B; the transform needs 69"));
        }
        nsamples = e.times.len();
        for k in 0..e.times.len() {
            let t = e.times[k];
            let j = times.partition_point(|x| *x < t);
            let cand = [j.checked_sub(1), (j < times.len()).then_some(j)];
            let best = cand
                .iter()
                .flatten()
                .map(|i| (*i, (times[*i] - t).abs()))
                .min_by_key(|(_, d)| *d);
            let (ri, dt) = match best {
                Some((i, d)) if d <= tol_ms => {
                    wrote += 1;
                    (i, d)
                }
                // Outside the CSV's own span the car HOLDS the nearest end.
                // The alternative is leaving the grid's template sample there,
                // which is a different car in a different place -- a visible
                // teleport at both ends of a clip.
                _ if t < lo => {
                    held += 1;
                    (0, 0)
                }
                _ if t > hi => {
                    held += 1;
                    (times.len() - 1, 0)
                }
                Some((i, d)) => {
                    return Err(format!(
                        "instant {t} ms is inside the CSV's span and its nearest row is {d} ms \
                         away, past --tol {tol_ms}. The record's grid and the trace disagree; \
                         raise --tol only if that gap is smaller than the car moves in it. \
                         (nearest row: {} ms)",
                        times[i]
                    ))
                }
                None => return Err("the CSV has no rows".into()),
            };
            if dt > worst_dt {
                worst_dt = dt;
            }
            if k == 0 || k == e.times.len() - 1 {
                first_last.push((k, ri));
            }
            let a = k * ss;
            gbx::recwrite::write_transform(&mut e.raw[a..a + ss], 47, &rows[ri].1);
        }
        Ok(())
    })?;

    // THE READ-BACK. What is on disk, not what the loop believes.
    let back = gbx::record::decode_ghost(out)?;
    let mut worst_m = 0f64;
    for (k, ri) in &first_last {
        let s = back
            .samples
            .get(*k)
            .ok_or_else(|| format!("the written file has no sample {k}"))?;
        let w = &rows[*ri].1;
        let d = ((s.x as f64 - w.pos[0] as f64).powi(2)
            + (s.y as f64 - w.pos[1] as f64).powi(2)
            + (s.z as f64 - w.pos[2] as f64).powi(2))
        .sqrt();
        if d > worst_m {
            worst_m = d;
        }
    }
    if worst_m > 1e-3 {
        let _ = std::fs::remove_file(out);
        return Err(format!(
            "read-back: a written sample decodes {worst_m:.6} m from the CSV row it was written \
             from -- the file was NOT written and is deleted"
        ));
    }
    Ok(format!(
        "{} of {} samples written from {} CSV rows spanning {}..{} ms ({} held at an end, \
         worst instant gap {} ms, read-back error {:.2e} m)",
        wrote,
        nsamples,
        rows.len(),
        lo,
        hi,
        held,
        worst_dt,
        worst_m
    ))
}

fn show(a: &[String]) {
    let path = a.first().unwrap_or_else(|| die("ghost record show FILE"));
    let body = gbx::record::load_body(path).unwrap_or_else(|e| die(e));
    let (ver, blob) = gbx::record::find_entrecord_blob(&body).unwrap_or_else(|e| die(e));
    let rd = gbx::record::parse_record_data(&blob, ver).unwrap_or_else(|e| die(e));
    println!(
        "version {}  span {} .. {} ms  {} descs  {} entities  {} notices  {} bulk  {} custom",
        rd.version,
        rd.start_ms,
        rd.end_ms,
        rd.descs.len(),
        rd.ents.len(),
        rd.notices.len(),
        rd.bulk_notices.len(),
        rd.custom_modules.len()
    );
    for (i, d) in rd.descs.iter().enumerate() {
        // u04 is the descriptor's own payload. GBX.NET calls it unknown; if the
        // engine puts the recorded FIELD LIST anywhere in the file, it is here,
        // so print it rather than its length.
        println!(
            "  desc {i}: class 0x{:08X}  u01 {} u02 {} u03 {} u05 {}  u04 {} B{}",
            d.class_id,
            d.u01,
            d.u02,
            d.u03,
            d.u05,
            d.u04.len(),
            if d.u04.is_empty() {
                String::new()
            } else {
                let hex: Vec<String> = d.u04.iter().map(|b| format!("{b:02x}")).collect();
                format!(": {}", hex.join(" "))
            }
        );
    }
    let veh = pick_vehicle(&rd);
    let mut live_scene = 0usize;
    for (i, e) in rd.ents.iter().enumerate() {
        let cls = rd.descs.get(e.type_ as usize).map(|d| d.class_id).unwrap_or(0);
        let is_car = Some(i) == veh;
        let live = !e.times.is_empty();
        // NON-VEHICLE IS A CLASS, NOT "NOT THE ONE THE PICKER CHOSE". A
        // multi-car container (227654 has 29 entities, five of them cars) has
        // exactly one entity the picker returns, so an index test counts the
        // OTHER FOUR CARS as scene records and reports "live non-vehicle
        // records: 5" about a file with one. The count then reads clean in
        // exactly the case it exists to catch.
        if cls != gbx::record::CLASS_CSCENEVEHICLEVIS && live {
            live_scene += 1;
        }
        println!(
            "  ent {i}: desc {} class 0x{cls:08X}  {} samples x {} B  t {} .. {}  u01 {} u02 {} \
             u03 {} u04 {}  {} delta2 block(s)  {}{}",
            e.type_,
            e.times.len(),
            e.sample_size,
            e.times.first().copied().unwrap_or(0),
            e.times.last().copied().unwrap_or(0),
            e.u01,
            e.u02,
            e.u03,
            e.u04,
            e.deltas2.len(),
            if live { "LIVE" } else { "placeholder (0 samples)" },
            if is_car { "   <- the car this project reads" } else { "" }
        );
    }
    // THE COUNT THAT WAS NOT PRINTED WHILE TWO CRASHERS AND A CLEAN FILE READ
    // AS "STRUCTURALLY INDISTINGUISHABLE". Restoring a live non-vehicle record
    // is what repaired both ghosts that crash the game client on import
    // (measured 2026-08-23) — and `TAS_67319` has none and imports cleanly, so
    // this is a lead, not a rule. `ghost verify` V11 says the same in one line.
    println!(
        "live non-vehicle records: {}{}",
        live_scene,
        if live_scene == 0 {
            "   <- the shape of the two files that crash the client on import: see ghost verify V11"
        } else {
            ""
        }
    );
}


/// Make the scene end when the car does, WITHOUT touching the car's samples.
///
/// The repair for the 31 published ghosts whose record outlives their run. Those
/// files' telemetry is fine -- the car's samples are its own and stop where the
/// run stops. What is wrong is the frame around them: the record node's declared
/// `end_ms` is the CONTAINER DONOR's, and the donor's non-vehicle entities
/// (0x2D001000, 13 bytes a sample) are still there at their own full length.
///
/// So this drops those and shortens the span, and does not regenerate anything.
/// That matters for three reasons: it needs no engine and no map, so it can be
/// run over a whole checkout in seconds; it cannot change a trajectory, so it
/// cannot introduce the defects a regeneration can; and re-regenerating 31
/// verified files to fix a number in their header would be the more dangerous
/// operation by far.
///
/// It refuses rather than guessing when the vehicle entity is not obvious --
/// a file with several live cars in it is a different problem, and 227654's
/// 27-entity carrier is the reason to say so out loud.
pub fn shorten_scene(inp: &str, out: &str) -> Result<String, String> {
    let mut before = String::new();
    let mut after = String::new();
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity: nothing to shorten to")?;
        let last = rd.ents[vi].times.last().copied().unwrap_or(0);
        let scene = rd
            .ents
            .iter()
            .filter_map(|e| e.times.last().copied())
            .max()
            .unwrap_or(0)
            .max(rd.end_ms);
        before = format!(
            "{} entities, scene to {}, car to {}",
            rd.ents.len(),
            secs(scene as i64),
            secs(last as i64)
        );
        let dropped: Vec<String> = rd
            .ents
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != vi)
            .map(|(_, e)| {
                format!(
                    "0x{:08X} x{}@{}",
                    rd.descs.get(e.type_ as usize).map(|d| d.class_id).unwrap_or(0),
                    e.times.len(),
                    secs(e.times.last().copied().unwrap_or(0) as i64)
                )
            })
            .collect();
        let car = rd.ents.swap_remove(vi);
        rd.ents = vec![car];
        rd.bulk_notices.clear();
        rd.custom_modules.clear();
        rd.end_ms = last;
        after = format!("1 entity, scene to {} (dropped {})", secs(last as i64), dropped.join(", "));
        Ok(())
    })?;

    // The car must come through untouched. This is the whole claim of the
    // operation, so it is checked rather than asserted: same sample count, same
    // times, same bytes.
    let a = gbx::record::decode_ghost(inp)?;
    let b = gbx::record::decode_ghost(out)?;
    if a.samples.len() != b.samples.len() || a.raw != b.raw {
        return Err(format!(
            "the car's samples changed ({} -> {} samples, {} -> {} bytes) -- refusing, this \
             operation must not touch the trajectory",
            a.samples.len(),
            b.samples.len(),
            a.raw.len(),
            b.raw.len()
        ));
    }
    Ok(format!("{before} -> {after}; the car's {} samples are byte-identical", a.samples.len()))
}

/// Write a fixed value into named per-sample bytes of the vehicle entity.
///
/// The other half of the bisect, and the one that separates *"the game needs
/// THIS driver's values"* from *"the game chokes on the value we write"*. A
/// channel a regeneration fills with raw zero is not neutral when its encoding
/// is offset-binary -- the dampen bytes decode 0 as **-2, fully extended
/// wheels**, and byte 32/33 hold 128/42 at rest in every recording the game
/// made itself.
/// Named sample bytes, one row an instant. The other half of `bytediff`: the
/// diff says WHICH byte disagrees, this says WHAT it holds -- which is what
/// separates "a channel the run genuinely drives" from "a channel pinned at a
/// value the engine never writes".
/// Author sample bytes 108..111 -- the client-side countdown -- from the file's
/// own race clock.
///
/// **The one channel a regenerated car cannot get from the dedicated server.**
/// `SAMPLE-LAYOUT.md` reads the writer as `-2 - min(now - state[0x340], 3000)`:
/// the value in the state is the TIMESTAMP OF AN EVENT and `now` is the
/// archiver's caller-supplied clock, which the dedicated server does not hand
/// the gather. So `fk --carrier layout` leaves these four bytes alone and they
/// stay whatever the rebuild's neutral template held -- which is the constant
/// `-2`, i.e. **"the event happened this instant", on every single sample**.
///
/// That is what makes the car a hologram. Measured on 287431, 2026-08-29: a
/// film whose only remaining difference from a rendering donor was these four
/// bytes drew a transparent wireframe, and restoring them drew a solid car.
///
/// The law needs no borrowed bytes, because `now - t` is a DIFFERENCE and the
/// engine clock cancels out of it. `t` is constant across an entity's life --
/// an event does not move -- so `now - t` advances exactly with race time, and
/// the offset at the entity's own first instant is zero: the event IS the
/// entity's creation. Every recording the game wrote on this map says so, and
/// `--check` re-derives the field on a file that already has it and requires
/// every byte back:
///
/// * the FIRST vehicle entity carries `-1` throughout -- no event yet;
/// * each LATER one counts `-2 - min(t - t0, 3000)` from its own first instant.
pub fn countdown(inp: &str, out: &str, check: bool) -> Result<String, String> {
    const SAT: i32 = 3000;
    let mut lines: Vec<String> = Vec::new();
    let mut wrong = 0usize;
    let mut total = 0usize;
    rewrite_ghost(inp, out, |rd| {
        let veh: Vec<usize> = (0..rd.ents.len())
            .filter(|i| {
                rd.descs.get(rd.ents[*i].type_ as usize).map(|d| d.class_id).unwrap_or(0)
                    == gbx::record::CLASS_CSCENEVEHICLEVIS
                    && !rd.ents[*i].times.is_empty()
            })
            .collect();
        if veh.is_empty() {
            return Err("no live vehicle entity: there is no car to time".into());
        }
        // Time order, because "first" means first in the run, not first in the
        // entity list -- `record entorder` moves the car and would otherwise
        // silently change which segment is the spawn.
        let mut order = veh.clone();
        order.sort_by_key(|i| rd.ents[*i].times[0]);
        for (rank, i) in order.iter().enumerate() {
            let e = &mut rd.ents[*i];
            let ss = e.sample_size;
            if ss < 112 {
                return Err(format!("entity {i}'s samples are {ss} B -- no byte 108..111 in them"));
            }
            let t0 = e.times[0];
            let mut changed = 0usize;
            for (k, t) in e.times.clone().iter().enumerate() {
                let v: i32 = if rank == 0 { -1 } else { -2 - (t - t0).clamp(0, SAT) };
                let b = v.to_le_bytes();
                let at = k * ss + 108;
                total += 1;
                if e.raw[at..at + 4] != b {
                    changed += 1;
                    if check {
                        wrong += 1;
                    }
                }
                e.raw[at..at + 4].copy_from_slice(&b);
            }
            lines.push(format!(
                "  entity {i} ({} samples from {}): {}, {changed} sample(s) rewritten",
                e.times.len(),
                secs(t0 as i64),
                if rank == 0 {
                    "-1 throughout (the spawn segment: no event yet)".to_string()
                } else {
                    format!("-2 - min(t - {t0}, {SAT})")
                }
            ));
        }
        Ok(())
    })?;
    if check {
        let _ = std::fs::remove_file(out);
        if wrong == 0 {
            return Ok(format!(
                "{inp}: CONTROL PASSED -- the law reproduces all {total} of this file's own \
                 countdown samples, byte for byte.\n{}",
                lines.join("\n")
            ));
        }
        return Err(format!(
            "{inp}: CONTROL FAILED -- the law disagrees with this file on {wrong} of {total} \
             samples.\n{}",
            lines.join("\n")
        ));
    }
    Ok(format!("{out}: countdown authored on {total} sample(s)\n{}", lines.join("\n")))
}

pub fn dumpbytes(
    path: &str,
    bytes: &[usize],
    stride: usize,
    ent: Option<usize>,
) -> Result<String, String> {
    // `--ent N` reads a NAMED entity rather than the one `decode_ghost` picks.
    // A two-segment container's freefall car is never the picked one, and its
    // bytes are exactly where a per-entity law (the countdown's epoch) shows
    // itself.
    let (times, raw, ss, label) = match ent {
        Some(i) => {
            let body = gbx::record::load_body(path)?;
            let (ver, blob) = gbx::record::find_entrecord_blob(&body)?;
            let rd = gbx::record::parse_record_data(&blob, ver)?;
            let e = rd.ents.get(i).ok_or(format!("{path} has no entity {i}"))?;
            (e.times.clone(), e.raw.clone(), e.sample_size, format!("ent {i}"))
        }
        None => {
            let g = gbx::record::decode_ghost(path)?;
            (
                g.samples.iter().map(|s| s.time_ms).collect(),
                g.raw.clone(),
                g.sample_size,
                "the picked car".to_string(),
            )
        }
    };
    for b in bytes {
        if *b >= ss {
            return Err(format!("byte {b} is past the {ss}-byte sample"));
        }
    }
    let mut o = format!("{path} ({label}): {} samples x {ss} B\n     ms", times.len());
    for b in bytes {
        o.push_str(&format!(" {b:>4}"));
    }
    o.push('\n');
    for (i, t) in times.iter().enumerate() {
        if i % stride != 0 || (i + 1) * ss > raw.len() {
            continue;
        }
        o.push_str(&format!("{t:7}"));
        for b in bytes {
            o.push_str(&format!(" {:4}", raw[i * ss + b]));
        }
        o.push('\n');
    }
    Ok(o)
}

/// Per-byte disagreement between two records' driving-car samples, over the
/// instants they share.
///
/// The bisect's map. `copy_channels` can move a set of bytes from one file to
/// another, but choosing WHICH set to move was an argument until this printed
/// the answer: on a film that renders no car beside a donor that renders one,
/// the bytes that differ are the whole suspect list, and the bytes that agree
/// can be struck off without a render.
pub fn bytediff(left: &str, right: &str) -> Result<String, String> {
    let l = gbx::record::decode_ghost(left)?;
    let r = gbx::record::decode_ghost(right)?;
    let ss = l.sample_size;
    if ss != r.sample_size {
        return Err(format!("sample size {ss} vs {} -- not the same shape", r.sample_size));
    }
    let rmap: std::collections::HashMap<i32, &[u8]> = r
        .samples
        .iter()
        .enumerate()
        .map(|(i, s)| (s.time_ms, &r.raw[i * ss..(i + 1) * ss]))
        .collect();
    let mut diff = vec![0usize; ss];
    let mut shared = 0usize;
    for (i, s) in l.samples.iter().enumerate() {
        let Some(rs) = rmap.get(&s.time_ms) else { continue };
        shared += 1;
        let ls = &l.raw[i * ss..(i + 1) * ss];
        for b in 0..ss {
            if ls[b] != rs[b] {
                diff[b] += 1;
            }
        }
    }
    let mut o = format!(
        "{} shared instant(s) of {} here and {} there; {}-byte samples\n",
        shared,
        l.samples.len(),
        r.samples.len(),
        ss
    );
    if shared == 0 {
        return Ok(o);
    }
    let mut agree: Vec<usize> = Vec::new();
    for b in 0..ss {
        if diff[b] == 0 {
            agree.push(b);
        } else {
            o.push_str(&format!(
                "  byte {b:3}: {:5} of {shared} differ ({:6.2} %)\n",
                diff[b],
                100.0 * diff[b] as f64 / shared as f64
            ));
        }
    }
    o.push_str(&format!("identical on every shared instant: {agree:?}\n"));
    Ok(o)
}

pub fn set_channels(inp: &str, out: &str, pairs: &[(usize, u8)]) -> Result<String, String> {
    if pairs.is_empty() {
        return Err("no bytes named: --set N=V,N=V is the experiment".into());
    }
    let mut n = 0usize;
    let mut changed = 0usize;
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity to write into")?;
        let e = &mut rd.ents[vi];
        let ss = e.sample_size;
        for (b, _) in pairs {
            if *b >= ss {
                return Err(format!("byte {b} is past the {ss}-byte sample"));
            }
        }
        n = e.times.len();
        for i in 0..n {
            for (b, v) in pairs {
                if e.raw[i * ss + b] != *v {
                    e.raw[i * ss + b] = *v;
                    changed += 1;
                }
            }
        }
        Ok(())
    })?;
    Ok(format!("{n} sample(s), set {pairs:?}: {changed} byte(s) changed"))
}

/// Copy named per-sample bytes of the vehicle entity from another file,
/// matching samples by their own timestamps.
///
/// **A bisect instrument, in the family of `record entfields`.** `ghost regen`
/// writes the transform and the tape echo and ZEROES eleven channels it cannot
/// read out of the engine — and one of them is read by something outside this
/// project. Measured 2026-08-23 on 276877: a regenerated 9.415 whose position,
/// velocity and quaternion are bit-identical to the file the published clip was
/// shot from renders with the chase camera BURIED IN THE RAMP for the last
/// second of the run, and re-filming that older file on the same build in the
/// same session reproduces the good camera. So the game reads a channel the
/// gate cannot see, and finding WHICH one is a one-byte-at-a-time question that
/// needs one file per hypothesis.
///
/// It refuses on a length change and reports the count it actually altered, so
/// "the copy did nothing" cannot be mistaken for "the byte does not matter" --
/// the null result of a bisect step has to be distinguishable from a no-op.
pub fn copy_channels(inp: &str, out: &str, donor: &str, bytes: &[usize]) -> Result<String, String> {
    if bytes.is_empty() {
        return Err("no bytes named: --bytes N,N,... is the experiment".into());
    }
    let d = gbx::record::decode_ghost(donor)?;
    let dss = d.sample_size;
    let src: std::collections::HashMap<i32, &[u8]> = d
        .samples
        .iter()
        .enumerate()
        .map(|(i, s)| (s.time_ms, &d.raw[i * dss..(i + 1) * dss]))
        .collect();
    let mut changed = 0usize;
    let mut missing = 0usize;
    let mut same = 0usize;
    let mut n = 0usize;
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity to write into")?;
        let e = &mut rd.ents[vi];
        let ss = e.sample_size;
        if ss != dss {
            return Err(format!(
                "sample size {ss} here and {dss} in the donor -- these are not the same shape of \
                 record"
            ));
        }
        for b in bytes {
            if *b >= ss {
                return Err(format!("byte {b} is past the {ss}-byte sample"));
            }
        }
        n = e.times.len();
        for (i, t) in e.times.clone().iter().enumerate() {
            match src.get(t) {
                None => missing += 1,
                Some(s) => {
                    for b in bytes {
                        if e.raw[i * ss + b] == s[*b] {
                            same += 1;
                        } else {
                            e.raw[i * ss + b] = s[*b];
                            changed += 1;
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(format!(
        "{} sample(s), bytes {:?} from {}: {} byte(s) changed, {} already equal, {} sample(s) had \
         no counterpart in the donor and were left alone",
        n,
        bytes,
        donor,
        changed,
        same,
        missing
    ))
}

/// Lay a fresh 50 ms grid out to `span_ms`, one vehicle entity, every sample a
/// copy of one template. The library form of `ghost record rebuild`; `regen`
/// calls it on every run so the grid the engine writes into is ours.
///
/// `template_from` names another ghost to take the template sample from, for a
/// file whose own record has no vehicle entity at all.
pub fn rebuild_to(
    inp: &str,
    out: &str,
    span_ms: i64,
    template_from: Option<&str>,
    period: i64,
    hold_last: bool,
) -> Result<String, String> {
    if period <= 0 || span_ms <= 0 {
        return Err("a rebuilt record needs a positive span and period".into());
    }
    let template: Vec<u8> = {
        let src = template_from.unwrap_or(inp);
        let body = gbx::record::load_body(src)?;
        let (ver, blob) = gbx::record::find_entrecord_blob(&body)?;
        let rd = gbx::record::parse_record_data(&blob, ver)?;
        let vi = pick_vehicle(&rd).ok_or_else(|| {
            format!(
                "{src} has no vehicle entity to take a template sample from -- name one with \
                 --template-from"
            )
        })?;
        let e = &rd.ents[vi];
        e.raw[..e.sample_size].to_vec()
    };

    let n = (span_ms / period) as usize + 1;
    let mut before = String::new();
    let mut note = String::new();
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd);
        let (type_, u01, u04) = match vi {
            Some(i) => (rd.ents[i].type_, rd.ents[i].u01, rd.ents[i].u04),
            None => {
                let d = rd
                    .descs
                    .iter()
                    .position(|d| d.class_id == gbx::record::CLASS_CSCENEVEHICLEVIS)
                    .ok_or(
                        "this record declares no CSceneVehicleVis desc, so a rebuilt entity would \
                         have no class",
                    )?;
                (d as i32, 0, 0)
            }
        };
        before = format!(
            "{} entities, car {} samples spanning {} .. {}",
            rd.ents.len(),
            vi.map(|i| rd.ents[i].times.len()).unwrap_or(0),
            secs(vi.and_then(|i| rd.ents[i].times.first().copied()).unwrap_or(0) as i64),
            secs(vi.and_then(|i| rd.ents[i].times.last().copied()).unwrap_or(0) as i64),
        );
        let ss = template.len();
        // A REBUILD THAT ALREADY HAS THE RUN KEEPS IT.
        //
        // Every sample used to be a copy of the car's FIRST sample, which makes
        // the rebuilt car's recorded path ONE CONSTANT POINT. That is right for
        // the case this was written for -- a carrier whose grid does not cover
        // the run, where the samples that do exist are the DONOR's and keeping
        // them would hand the regenerator a stranger's path as an answer key.
        //
        // It is wrong for the case that turns up the moment a file has been
        // repaired once: a container whose car already covers the whole span on
        // this exact grid. There the rebuild changes no shape at all and
        // destroys the only thing in the file that can IDENTIFY the car --
        // 227654's `TAS_57482` came in with 1150 correct positions and went to
        // the engine as 1150 copies of the spawn point. Downstream, three
        // things then have no reference and none of them says so plainly: the
        // locate's chooser abstains ("a constant trajectory identifies
        // nothing"), `fk regen`'s orientation veto reports "the container's own
        // recording is NOT this run" -- which reads as a transplant and here
        // means the recording was thrown away -- and `--carrier`'s field gather
        // is left ranking copies of the car with no answer key. On this map
        // that is fatal: twenty-nine regeneration attempts refused, every one
        // of them for a reason two layers from the cause.
        //
        // So: carry a sample over when the grid instant is one the car already
        // recorded, and only stamp the neutral template on instants it does
        // not have. Gated on FULL cover -- if a single grid instant is missing,
        // the old all-constant behaviour stands, because a partly-carried grid
        // is exactly the donor-path answer key this must not create.
        let (have, cover): (std::collections::HashMap<i32, &[u8]>, bool) = match vi {
            Some(i) => {
                let e = &rd.ents[i];
                let m: std::collections::HashMap<i32, &[u8]> = e
                    .times
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (k + 1) * e.sample_size <= e.raw.len())
                    .map(|(k, t)| (*t, &e.raw[k * e.sample_size..(k + 1) * e.sample_size]))
                    .collect();
                // `hold_last` extends a record the car DOES hold, by repeating
                // its last sample past the end of the run -- the car parked
                // where it finished. It is opt-in and it is only honest on a
                // file whose record is already its own run: on a transplanted
                // container it would hold the DONOR's last state. The caller
                // says so; the default is unchanged.
                let full = e.sample_size == ss
                    && ((0..n).all(|k| m.contains_key(&((k as i64 * period) as i32)))
                        || (hold_last
                            && !m.is_empty()
                            && (0..n)
                                .take_while(|k| {
                                    (*k as i64 * period) as i32 <= *e.times.last().unwrap()
                                })
                                .all(|k| m.contains_key(&((k as i64 * period) as i32)))));
                (m, full)
            }
            None => (Default::default(), false),
        };
        let last_sample: Option<&[u8]> = match vi {
            Some(i) if hold_last => {
                let e = &rd.ents[i];
                let k = e.times.len().saturating_sub(1);
                if (k + 1) * e.sample_size <= e.raw.len() && e.sample_size == ss {
                    Some(&e.raw[k * e.sample_size..(k + 1) * e.sample_size])
                } else {
                    None
                }
            }
            _ => None,
        };
        let mut raw = Vec::with_capacity(n * ss);
        let mut times = Vec::with_capacity(n);
        let mut carried = 0usize;
        let mut held = 0usize;
        for k in 0..n {
            let t = (k as i64 * period) as i32;
            times.push(t);
            match if cover { have.get(&t) } else { None } {
                Some(s) => {
                    raw.extend_from_slice(s);
                    carried += 1;
                }
                None => match if cover { last_sample } else { None } {
                    // Past the car's own last sample, on a `--hold-last`
                    // rebuild: the car stands where it finished. Counted
                    // separately, because "held" is not "recorded".
                    Some(s) => {
                        raw.extend_from_slice(s);
                        held += 1;
                    }
                    None => raw.extend_from_slice(&template),
                },
            }
        }
        note = if carried > 0 {
            format!(
                " -- the car already covered this grid, so its own {carried} samples were KEPT \
                 (nothing was replaced by the neutral template){}",
                if held > 0 {
                    format!(", and {held} more HOLD its last sample: the car stands where it \
                             finished, which is a state it was never in")
                } else {
                    String::new()
                }
            )
        } else {
            String::new()
        };
        let carried_note = note.clone();
        // ONE CAR, AND EVERY NON-VEHICLE RECORD THE CONTAINER HAD.
        //
        // The vehicle entities go: on 227654 the carrier holds 29 of them and
        // 28 are the SAME driver across 26 restarts (contiguous, non-overlapping
        // time ranges), so a render that kept them would draw a phantom car per
        // restart. That half was always right.
        //
        // What was wrong was dropping everything else. **Restoring a LIVE
        // non-vehicle record is what repaired both ghosts that crash the game
        // client on ghost import** — measured on the render box 2026-08-23 with
        // a same-session control on every run: the container this project's
        // 173691 film is built in imports fine with its live 0x2D001000 record
        // intact (X0); the same file rebuilt without it crashes, three times
        // across two revisions; grafting that one record back makes it import
        // (X5, X7); and three other single-field repairs each crashed with the
        // graft absent (X1, X3, X6).
        //
        // **It is not known to be a rule**: TAS_67319 has no live non-vehicle
        // record either and imports cleanly, so what makes those two files
        // different is still open (`ghost verify` V11 says so in one line).
        // Keeping what the container had is the conservative choice either way
        // — the game wrote those records and nothing here has a reason to
        // remove them.
        //
        // No headless check can see this: the dedicated server re-simulates the
        // input chunk and never reads the scene, so TAS_57482 passes
        // `ghost verify` V1..V10, re-simulates to 57.482, and kills the client.
        //
        // The kept records are clipped to the run's span, so this cannot put
        // back the scene-outlives-the-run defect that `shorten` exists to
        // remove.
        let car = Ent {
            type_,
            u01,
            u02: 0,
            u03: span_ms as i32,
            u04,
            times,
            raw,
            sample_size: ss,
            deltas2: Vec::new(),
        };
        // Every entity that is not a car, clipped to the span, in its original
        // order; then the car.
        let veh: std::collections::HashSet<i32> = rd
            .descs
            .iter()
            .enumerate()
            .filter(|(_, d)| d.class_id == gbx::record::CLASS_CSCENEVEHICLEVIS)
            .map(|(i, _)| i as i32)
            .collect();
        let mut kept: Vec<Ent> = Vec::new();
        let mut live = 0usize;
        for e in rd.ents.drain(..) {
            if veh.contains(&e.type_) {
                continue;
            }
            let mut e = e;
            let keep = e.times.iter().take_while(|t| **t <= span_ms as i32).count();
            if keep < e.times.len() && e.sample_size > 0 {
                e.times.truncate(keep);
                e.raw.truncate(keep * e.sample_size);
            }
            e.u03 = e.u03.min(span_ms as i32);
            e.deltas2.retain(|(_, t, _)| *t <= span_ms as i32);
            if !e.times.is_empty() {
                live += 1;
            }
            kept.push(e);
        }
        let n_kept = kept.len();
        // THE CAR GOES FIRST. It was pushed LAST here, and in every ghost the
        // game itself wrote the car is entity 0 while every file this rebuild
        // produced had it at the end -- a four-row separation the render arm
        // isolated across four maps, where the repo's own file imports and our
        // rebuild of that same file does not (two of them raise a FrameMessage
        // and never gain a ghost block, two kill the process outright), with
        // V1..V11 including the oracle passing on all four.
        //
        // Order is not cosmetic to a consumer that resolves entities by index.
        // `ghost record entorder IN OUT --car-first | --car-last` is the same
        // change in both directions, because one direction proves nothing:
        // tonight the scene-record lead, the notice lead and the u01 lead each
        // looked decisive from one side and died from the other.
        kept.insert(0, car);
        rd.ents = kept;
        // KEEP THE NOTICE LISTS -- but NOT because they fix the import crash.
        // They do not. THE HYPOTHESIS THAT PUT THIS LINE HERE IS DEAD, killed
        // in both directions by the command written to test it (2026-08-23,
        // render box, each behind a same-session `scene ready` control):
        //
        //   forward  blev_regen + the container's 82 notices  -> STILL CRASHES
        //   mirror   TAS_67319 with its 82 notices stripped   -> STILL IMPORTS
        //
        // Neither necessary nor sufficient. The six-specimen table that made
        // notices look like the separating field was a majority-class
        // coincidence, the same shape as the `u01` lead that survived until
        // somebody ran its reverse swap. It would have been written down as a
        // mechanism if the mirror had not been run.
        //
        // The lines stay because the CONSERVATIVE choice is to carry what the
        // container had: the game wrote these, the rebuild has no reason to
        // remove them, and clearing them was never justified by a measurement
        // either. That is the whole claim -- do not let it grow back into a
        // fix.
        //
        // The crash is still open. What survives every refutation so far is the
        // car entity's own encoding: the donor's car carries 31 `delta2` blocks
        // and every rebuilt car carries NONE (on 227654's untouched file the
        // car segments carry 11, 13 and 4; ours carries 0). That is the only
        // structural difference left between a file that imports and one that
        // does not, and it is the same difference on both crashers.
        rd.start_ms = 0;
        rd.end_ms = span_ms as i32;
        note = format!(
            ", kept {} non-vehicle record(s) of which {} live{}",
            n_kept, live, carried_note
        );
        Ok(())
    })?;

    // Read it back: this is the check that THIS write produced the grid that
    // was asked for, and it costs one parse.
    let d = gbx::record::decode_ghost(out).map_err(|e| format!("reading {out} back: {e}"))?;
    if d.samples.len() != n {
        return Err(format!("wrote {n} samples and read back {}", d.samples.len()));
    }
    Ok(format!(
        "{before} -> the car regridded to {n} samples every {period} ms spanning 0.000 .. {}{note}",
        secs(span_ms)
    ))
}

/// Set the UNIDENTIFIED header fields of the record's vehicle entity.
///
/// This exists for one investigation and says so: two of this project's
/// regenerated ghosts crash the game client on import while four others from
/// the same path import cleanly, and the only field that separates them in any
/// dump we have is the car entity's `u01` — `0x02000006` on all four that
/// import, `0x0200000B` and `0x020000F1` on the two that crash. The value is
/// inherited from the container donor and nothing in this project knows what it
/// means, so the only way to find out is to move it, one field at a time, in
/// both directions, and ask the client.
///
/// It touches no sample and no time: the car's trajectory is required to come
/// back byte-identical, exactly as `shorten` requires.
/// Move the car entity to the front or the back of the record — the two
/// directions of the entity-order experiment.
///
/// In every ghost the game itself wrote, the car is **entity 0**. Every file
/// `rebuild_to` produced before 2026-08-23 had it **last**, and four of four
/// such rebuilds failed to import while the repo's own file for the same map
/// imported in the same session. That is a suggestive separation and nothing
/// more until it is run BOTH ways, which is what this is for:
///
/// * `--car-first` — a rebuilt file with the car put back at index 0.
/// * `--car-last` — a file that imports, with its car moved to the end.
///
/// If the first imports and the second crashes, order is the mechanism. If both
/// import, order was never it, and the rebuild's other differences — the
/// dropped `delta2` blocks, the scene record one sample short — are next.
///
/// Nothing but a client import can see any of this: the dedicated server
/// re-simulates the input chunk and never reads the entity list, so both
/// outputs re-simulate to the same time as their input. The car itself is
/// required to come through untouched.
/// Where the car sits among the record's entities, as (index, count).
///
/// The game client is not indifferent to this. `ghost regen` rebuilds the
/// record and puts the car at index 0; on 203072, whose container has the car
/// LAST of three, the rebuilt file then fails to import — the ghost-block
/// count never rises and the game raises a `FrameMessage`, four attempts
/// running. Its own published file, same map, same scene, imports first time.
/// Nothing headless sees this: that file passes `ghost verify` V1–V11 with
/// kappa 1.000 and the oracle exact.
pub fn car_index(path: &str) -> Result<(usize, usize), String> {
    let body = gbx::record::load_body(path)?;
    let (v, blob) = gbx::record::find_entrecord_blob(&body)?;
    let rd = gbx::record::parse_record_data(&blob, v)?;
    let vi = pick_vehicle(&rd).ok_or("no vehicle entity")?;
    Ok((vi, rd.ents.len()))
}

pub fn set_ent_order(inp: &str, out: &str, car_first: bool) -> Result<String, String> {
    let mut before = String::new();
    let mut after = String::new();
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity: nothing to reorder")?;
        if rd.ents.len() < 2 {
            return Err(format!(
                "this record has {} entity: order is not a property of it",
                rd.ents.len()
            ));
        }
        before = format!("car at index {} of {}", vi, rd.ents.len());
        let car = rd.ents.remove(vi);
        if car_first {
            rd.ents.insert(0, car);
        } else {
            rd.ents.push(car);
        }
        let vj = pick_vehicle(rd).unwrap_or(usize::MAX);
        after = format!("car at index {} of {}", vj, rd.ents.len());
        if vi == vj {
            return Err(format!(
                "the car is already {} ({}): writing this file would produce an import result \
                 about nothing",
                if car_first { "first" } else { "last" },
                before
            ));
        }
        Ok(())
    })?;
    let a = gbx::record::decode_ghost(inp)?;
    let b = gbx::record::decode_ghost(out)?;
    if a.samples.len() != b.samples.len() {
        return Err(format!(
            "the car changed: {} samples in, {} out",
            a.samples.len(),
            b.samples.len()
        ));
    }
    Ok(format!("{before} -> {after}; car unchanged ({} samples)", a.samples.len()))
}

/// Restore or remove the record's NOTICE LISTS — the two directions of one
/// experiment.
///
/// **The hypothesis this was written to test is DEAD, and this command is what
/// killed it.** The notice list looked like the single field separating every
/// importable ghost from `blev_regen`, which crashes the game client with
/// V1..V11 passing and its scene record present. Both directions, run on the
/// render box 2026-08-23, each behind a same-session `scene ready` control:
///
/// * forward — `blev_regen` **plus** the container's 82 notices: still crashes.
/// * mirror — `TAS_67319` with its 82 notices **stripped**: still imports.
///
/// So notices are neither necessary nor sufficient, and the six-specimen table
/// that made them look decisive was a majority-class coincidence — the same
/// shape as the `u01` lead that survived until its reverse swap was run. It
/// would have been written into `GHOSTS.md` as a mechanism if the mirror had
/// not been run.
///
/// The command stays, because the next candidate wants the same treatment:
///
/// * `--from DONOR` — put the container's notices back into a file that crashes.
/// * `--strip` — take them out of a file that imports.
///
/// It refuses unless exactly one direction is named, and it requires the car to
/// come through untouched: an import result means nothing if the tape moved.
///
/// Nothing headless can see any of this: the dedicated server re-simulates the
/// input chunk and never reads the record, so both outputs re-simulate to the
/// same time as their input. A client import is the only instrument.
pub fn set_notices(
    inp: &str,
    out: &str,
    donor: Option<&str>,
    strip: bool,
) -> Result<String, String> {
    let taken = match donor {
        None => None,
        Some(d) => {
            let dbody = gbx::record::load_body(d)?;
            let (dver, dblob) = gbx::record::find_entrecord_blob(&dbody)?;
            let drd = gbx::record::parse_record_data(&dblob, dver)?;
            if drd.notices.is_empty() && drd.bulk_notices.is_empty() {
                return Err(format!(
                    "{d} has no notices to give: restoring nothing would look like a passing \
                     experiment"
                ));
            }
            Some((drd.notices, drd.bulk_notices, drd.custom_modules))
        }
    };
    let mut before = String::new();
    let mut after = String::new();
    rewrite_ghost(inp, out, |rd| {
        before = format!(
            "{} notices, {} bulk, {} custom",
            rd.notices.len(),
            rd.bulk_notices.len(),
            rd.custom_modules.len()
        );
        if strip {
            rd.notices.clear();
            rd.bulk_notices.clear();
            rd.custom_modules.clear();
        } else if let Some((n, b, c)) = taken.clone() {
            rd.notices = n;
            rd.bulk_notices = b;
            rd.custom_modules = c;
        }
        after = format!(
            "{} notices, {} bulk, {} custom",
            rd.notices.len(),
            rd.bulk_notices.len(),
            rd.custom_modules.len()
        );
        Ok(())
    })?;
    // The car must be untouched: this changes the record's notices and nothing
    // about the driving, and an import result means nothing if the tape moved.
    let a = gbx::record::decode_ghost(inp)?;
    let b = gbx::record::decode_ghost(out)?;
    let av = a.samples.len();
    let bv = b.samples.len();
    if av != bv {
        return Err(format!(
            "the car changed: {av} samples in, {bv} out -- refusing to write a file whose import \
             result would not be about notices"
        ));
    }
    Ok(format!("{before} -> {after}; car unchanged ({av} samples)"))
}

pub fn set_ent_fields(
    inp: &str,
    out: &str,
    u01: Option<i32>,
    u02: Option<i32>,
    u04: Option<i32>,
) -> Result<String, String> {
    if u01.is_none() && u02.is_none() && u04.is_none() {
        return Err("nothing to set: give --u01, --u02 or --u04".into());
    }
    let mut before = String::new();
    let mut after = String::new();
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity to set fields on")?;
        let e = &mut rd.ents[vi];
        before = format!("u01 {} u02 {} u04 {}", e.u01, e.u02, e.u04);
        if let Some(v) = u01 {
            e.u01 = v;
        }
        if let Some(v) = u02 {
            e.u02 = v;
        }
        if let Some(v) = u04 {
            e.u04 = v;
        }
        after = format!("u01 {} u02 {} u04 {}", e.u01, e.u02, e.u04);
        Ok(())
    })?;
    let a = gbx::record::decode_ghost(inp)?;
    let b = gbx::record::decode_ghost(out)?;
    if a.samples.len() != b.samples.len() || a.raw != b.raw {
        return Err("the car's samples changed -- refusing, this operation must not touch the \
                    trajectory"
            .into());
    }
    Ok(format!("{before} -> {after}; the car's {} samples are byte-identical", a.samples.len()))
}

/// Put the container's NON-VEHICLE entities back, from the file the car was
/// rebuilt out of.
///
/// The inverse of what `rebuild_to` does, and it exists because of a measured
/// fork. `ghost regen` lays a fresh grid for the car and, in doing so, keeps
/// **one** entity and drops everything else the record had. Two regenerated
/// files crash the game client on import; the container one of them was built
/// in — the same version, the same 107-byte samples, the game's own bytes —
/// **imports fine with its three entities intact** (measured on the render box,
/// 2026-08-23, five variants each behind a same-session control). Three other
/// candidate causes died in that batch: the entity's `u01`, the declared
/// checkpoint count, and the container generation. What is left is what the
/// rebuild removed.
///
/// So this grafts those entities back, from `donor`, at their own times, and:
///
/// * it copies **only** entities the vehicle picker does not choose — another
///   car is not scenery, and 227654's container is one driver across 26
///   restarts rather than 26 opponents, so "everything else" would be wrong
///   there;
/// * it clips their samples to the record's span, because a scene entity that
///   outlives the run is the defect `shorten` exists to remove and this must not
///   reintroduce it;
/// * it requires the car's samples back **byte-identical**.
///
/// **These samples are the DONOR's**, and the file says so: they are 13 bytes
/// per sample of somebody else's session state, not a trajectory. Run
/// `ghost verify` afterwards and read V3 and the raw-bytes backstop before
/// publishing anything built this way.
/// Re-cut ONE car entity into the segment structure a donor container uses.
///
/// THE RECORD IS ONE ENTITY PER SEGMENT, AND ON SOME MAPS THAT IS NOT OPTIONAL.
///
/// `rebuild_to` lays the whole run down as ONE `CSceneVehicleVis` entity, which
/// is what the game writes on an ordinary map and what every published
/// regeneration in this project is. On 227654 it is what the game CLIENT
/// refuses: the single-entity file kills the process on ghost import (measured
/// on the render box, four variants, each behind a same-session control — the
/// container's own 29-entity file imports on the same map in the same session,
/// with our tape in it and our declared time on it).
///
/// The boundaries are not the driver's. That container's vehicle entities break
/// at 1.300 / 19.480 / 36.300 / 54.330 s, and the engine — running OUR tape,
/// which has no respawn in it at all — moves the car's live state between
/// objects at **19.500 and 36.300 s** (`fk regen --carrier --verbose` prints
/// the windows). Two different runs, the same boundaries: they are the map's,
/// not the run's. This map is `DesertCar / SnowCar / Bobsleigh` and it changes
/// the vehicle under you; each vehicle is its own entity, in the recording and
/// in engine memory both.
///
/// So: keep our samples exactly as they are, and hand them to the segments the
/// donor declares, with each segment's own `u01`. Sample bytes are never
/// touched — the read-back below requires every one of them back, in order.
pub fn resegment(inp: &str, out: &str, donor: &str) -> Result<String, String> {
    let dbody = gbx::record::load_body(donor)?;
    let (dver, dblob) = gbx::record::find_entrecord_blob(&dbody)?;
    let drd = gbx::record::parse_record_data(&dblob, dver)?;
    // The donor's vehicle entities, in time order: (first, last, u01, u04).
    let mut segs: Vec<(i32, i32, i32, i32)> = drd
        .ents
        .iter()
        .filter(|e| {
            drd.descs.get(e.type_ as usize).map(|d| d.class_id).unwrap_or(0)
                == gbx::record::CLASS_CSCENEVEHICLEVIS
                && !e.times.is_empty()
        })
        .map(|e| (*e.times.first().unwrap(), *e.times.last().unwrap(), e.u01, e.u04))
        .collect();
    segs.sort_by_key(|s| s.0);
    if segs.len() < 2 {
        return Err(format!(
            "{donor} has {} vehicle segment(s) -- there is nothing to copy the shape of",
            segs.len()
        ));
    }
    let mut before = String::new();
    let mut after = String::new();
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity to re-cut")?;
        let nveh = {
            let veh: Vec<usize> = (0..rd.ents.len())
                .filter(|i| {
                    rd.descs.get(rd.ents[*i].type_ as usize).map(|d| d.class_id).unwrap_or(0)
                        == gbx::record::CLASS_CSCENEVEHICLEVIS
                })
                .collect();
            veh.len()
        };
        if nveh != 1 {
            return Err(format!(
                "this file has {nveh} vehicle entities -- resegment re-cuts a rebuilt car, of \
                 which there must be exactly one"
            ));
        }
        let car = rd.ents.remove(vi);
        let ss = car.sample_size;
        if ss == 0 || car.times.is_empty() {
            return Err("the car entity has no samples".into());
        }
        before = format!(
            "1 car entity, {} samples spanning {} .. {}",
            car.times.len(),
            secs(*car.times.first().unwrap() as i64),
            secs(*car.times.last().unwrap() as i64)
        );
        let span = rd.end_ms;
        let mut cut: Vec<Ent> = Vec::new();
        let mut used = 0usize;
        let mut bounds: Vec<String> = Vec::new();
        for (i, (s, e, u01, u04)) in segs.iter().enumerate() {
            if *s > span {
                continue;
            }
            // The last segment inside the span carries the run to its end: the
            // donor's own last boundary is ITS run's, and ours finishes sooner.
            let hi = if segs.get(i + 1).map(|n| n.0 > span).unwrap_or(true) { span } else { *e };
            let mut times = Vec::new();
            let mut raw = Vec::new();
            for (k, t) in car.times.iter().enumerate() {
                if *t >= *s && *t <= hi && (k + 1) * ss <= car.raw.len() {
                    times.push(*t);
                    raw.extend_from_slice(&car.raw[k * ss..(k + 1) * ss]);
                }
            }
            if times.is_empty() {
                continue;
            }
            used += times.len();
            bounds.push(format!(
                "{}..{} x{}",
                secs(*times.first().unwrap() as i64),
                secs(*times.last().unwrap() as i64),
                times.len()
            ));
            cut.push(Ent {
                type_: car.type_,
                u01: *u01,
                u02: *times.first().unwrap(),
                u03: *times.last().unwrap(),
                u04: *u04,
                times,
                raw,
                sample_size: ss,
                deltas2: Vec::new(),
            });
        }
        if used != car.times.len() {
            return Err(format!(
                "the donor's segments cover {used} of the car's {} samples -- a re-cut that \
                 drops or duplicates a sample is not a re-cut",
                car.times.len()
            ));
        }
        // The donor's order: the scene records first, then the cars in time
        // order -- which is how the game writes them and how `rebuild_to`
        // learned to put the car first.
        let mut ents: Vec<Ent> = Vec::new();
        let ncut = cut.len();
        ents.append(&mut cut);
        ents.extend(rd.ents.drain(..));
        rd.ents = ents;
        after = format!("{ncut} car segments [{}]", bounds.join(", "));
        Ok(())
    })?;
    // READ IT BACK: every sample, in order, byte for byte. A re-cut that moves
    // a byte is a different recording wearing this one's name.
    //
    // NOT through `decode_ghost`: that takes the entity with the most samples
    // and calls it the record, which on a re-cut file is one segment (363 of
    // 1150). The check has to walk every vehicle entity, which is also the
    // shape of what this file now is.
    let all = |p: &str| -> Result<Vec<(i32, Vec<u8>)>, String> {
        let body = gbx::record::load_body(p)?;
        let (v, blob) = gbx::record::find_entrecord_blob(&body)?;
        let rd = gbx::record::parse_record_data(&blob, v)?;
        let mut out: Vec<(i32, Vec<u8>)> = Vec::new();
        for e in &rd.ents {
            if rd.descs.get(e.type_ as usize).map(|d| d.class_id).unwrap_or(0)
                != gbx::record::CLASS_CSCENEVEHICLEVIS
            {
                continue;
            }
            for (k, t) in e.times.iter().enumerate() {
                if (k + 1) * e.sample_size <= e.raw.len() {
                    out.push((*t, e.raw[k * e.sample_size..(k + 1) * e.sample_size].to_vec()));
                }
            }
        }
        out.sort_by_key(|s| s.0);
        Ok(out)
    };
    let a = all(inp)?;
    let b = all(out)?;
    if a != b {
        return Err(format!(
            "the car's samples did not come back: {} in, {} out{}",
            a.len(),
            b.len(),
            if a.len() == b.len() { " (same count, different bytes or times)" } else { "" }
        ));
    }
    Ok(format!("{before} -> {after}; all {} samples kept, byte for byte", a.len()))
}

/// `resample` — put OUR car's sample bytes into a record the client ACCEPTS.
///
/// **The experiment this exists for.** Three maps in this corpus produce a
/// regenerated ghost that the game client refuses or dies on, while the
/// container's own untouched record imports every time: 228811, 279209 and
/// 227654 (24 variants, one variable each, on that last one). Every headless
/// gate passes on the refused files — `ghost verify` V1–V11, kappa 1.000, the
/// plain oracle exact — so the defect is invisible to everything except a real
/// client. What has never been separated is:
///
///   * our SAMPLE BYTES — the 116-byte-per-instant car state we write, and
///   * our RECORD — the container structure the rebuild emits around them.
///
/// `graft-scene` tests the second by putting a donor's scene back. This tests
/// the first, from the other side: take a file that imports, and overwrite
/// **only the car's sample bytes**, instant for instant, with ours. Nothing
/// else in the file moves — not the entity list, not the deltas, not the
/// spans, not the other entities, not a single header field.
///
/// If the result imports, our bytes are innocent and the rebuilt RECORD is the
/// defect. If it dies, our bytes are the defect and the record is innocent.
/// Either answer closes the question; that is what makes it worth a
/// subcommand rather than a one-off.
///
/// **The guards, and why each one is here.** This writes one file's telemetry
/// into another file's container, which is exactly the shape of every splice
/// defect this project has published a retraction for. So it refuses unless
/// the two files are already the same run:
///
///   * the same input tape, byte for byte — otherwise the result is a record
///     of one run wearing another's inputs, which is the KAPPA.md defect;
///   * the same declared time;
///   * the same sample size;
///   * and every instant it writes has to exist in both. Coverage is reported
///     and a partial overwrite is refused, because a file that is our
///     trajectory for 90 % of its length and somebody else's for the rest is
///     worse than either.
/// What `resample` may do beyond its one-entity, whole-sample default.
#[derive(Default)]
pub struct ResampleOpts {
    /// Write EVERY vehicle entity, not just the one `pick_vehicle` chose. A
    /// server replay splits one car across many entities; the file the client
    /// accepts on 227654 is exactly such a record, so the experiment needs all
    /// of them.
    pub all_cars: bool,
    /// Measure and print the overlap; write nothing.
    pub report: bool,
    /// Acknowledge that IN and SRC are DIFFERENT RUNS. The result is a
    /// diagnostic file — a container carrying another run's car — and must
    /// never be published as a recording of either. Without this the same
    /// same-run guards apply as before.
    pub mixed_run: bool,
    /// Write only these byte offsets of each sample, leaving the rest as the
    /// target's. This is the bisect: it makes "which byte does the client
    /// reject" a bounded search instead of an argument.
    pub bytes: Option<Vec<usize>>,
    /// After the source ends, repeat its LAST sample instead of leaving the
    /// target's: the car parks where it finished rather than another driver
    /// carrying on.
    pub hold_last: bool,
    /// Milliseconds of slack for a target instant the source does not have.
    /// 227654's container re-aligns its 50 ms grid after every entity
    /// boundary, so six of its ~1155 in-span instants sit 10-20 ms off ours.
    /// With no slack those six would keep the TARGET's car — a one-frame
    /// teleport into another driver's position — which is worse than our own
    /// sample from 20 ms away. The worst offset actually used is reported.
    pub fill_tol_ms: i32,
}

/// `19,47..69,91` -- the bisect's argument.
pub fn parse_byte_ranges(s: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        match part.split_once("..") {
            Some((a, b)) => {
                let a: usize = a.parse().expect("byte range start");
                let b: usize = b.parse().expect("byte range end");
                v.extend(a..b);
            }
            None => v.push(part.parse().expect("byte offset")),
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

pub fn resample(inp: &str, out: &str, src: &str) -> Result<String, String> {
    resample_opt(inp, out, src, &ResampleOpts::default())
}

pub fn resample_opt(
    inp: &str,
    out: &str,
    src: &str,
    opt: &ResampleOpts,
) -> Result<String, String> {
    // 1. SAME RUN? Tape first: it is the cheapest and the most damning.
    //    --mixed-run says out loud that this file is a DIAGNOSTIC and not a
    //    recording of anything; every other caller still gets the guard.
    if !opt.mixed_run {
        let ti = gbx::tape::Tape::from_file(inp)?;
        let ts = gbx::tape::Tape::from_file(src)?;
        // COMPARE THE INPUTS, NOT THE ENCODING.
        //
        // This compared whole encoded payloads, and an encoder is allowed to
        // spend a different number of bits on the same drive. `ghost declare`
        // re-encodes the tape it is only relabelling and emits ONE EXTRA
        // PADDING BYTE, so on 286279 the declared container and the
        // regenerated file differed at payload byte 24 -- the bitstream length
        // -- while carrying identical inputs: same 25350 packets, same
        // accel_on 17561, brake_on 1159, steer32 0, respawn 1, and the
        // regenerated file matching the SOURCE bit for bit. The guard refused
        // a faithful pair and the map needed --mixed-run, a flag that means
        // "never publish this".
        //
        // What the guard is FOR is the KAPPA.md defect: a record of one run
        // wearing another's inputs. That is a property of the input SEQUENCE,
        // so compare the decoded steer/accel/brake/respawn per packet. Two
        // encodings of one drive now pass; two different drives still fail,
        // because their packets differ wherever the driving differs.
        let keys = |t: &gbx::tape::Tape| -> Vec<(u32, u32, u32, bool)> {
            t.archives
                .iter()
                .flat_map(|a| a.packets.iter())
                .map(|p| (p.steer, p.accel, p.brake, p.respawn()))
                .collect()
        };
        let (ki, ks) = (keys(&ti), keys(&ts));
        let (pi, ps) = (
            ti.to_payload(gbx::tape::Encoding::Verbatim),
            ts.to_payload(gbx::tape::Encoding::Verbatim),
        );
        if ki != ks {
            // SAY WHAT ACTUALLY DIFFERS. The comparison is on the whole
            // encoded payload; reporting only tick counts produced
            // "do not carry the same input tape (25350 vs 25350 ticks)" on
            // 286279 -- a message that reads like a bug in itself and sent me
            // looking in the wrong place. Same length, different bytes, and
            // the first differing byte is the thing worth printing.
            let first_diff = ki
                .iter()
                .zip(ks.iter())
                .position(|(a, b)| a != b)
                .map(|i| format!("first differing INPUT at packet {}", i))
                .unwrap_or_else(|| {
                    format!(
                        "{} and {} packets ({} and {} payload bytes)",
                        ki.len(),
                        ks.len(),
                        pi.len(),
                        ps.len()
                    )
                });
            return Err(format!(
                "{inp} and {src} do not carry the same input tape ({} vs {} ticks, {}). resample \
                 writes one file's car into another's container; with different tapes the result \
                 is a record of one run wearing another's inputs, which is the defect KAPPA.md is \
                 about. Pass --mixed-run only for a file that will never be published.",
                ti.n(),
                ts.n(),
                first_diff
            ));
        }
    }
    let di = gbx::record::decode_ghost(inp)?;
    let ds = gbx::record::decode_ghost(src)?;
    if !opt.mixed_run && di.race_time_ms != ds.race_time_ms {
        return Err(format!(
            "declared times differ: {:?} vs {:?}",
            di.race_time_ms, ds.race_time_ms
        ));
    }
    if di.sample_size != ds.sample_size {
        return Err(format!(
            "sample sizes differ: {} vs {}",
            di.sample_size, ds.sample_size
        ));
    }
    // 2. THE SOURCE'S CAR, indexed by instant.
    let ss = ds.sample_size;
    let mut by_ms: std::collections::HashMap<i32, &[u8]> = std::collections::HashMap::new();
    for (i, s) in ds.samples.iter().enumerate() {
        let a = i * ss;
        if a + ss <= ds.raw.len() {
            by_ms.insert(s.time_ms, &ds.raw[a..a + ss]);
        }
    }
    let src_lo = ds.samples.iter().map(|s| s.time_ms).min().unwrap_or(0);
    let src_hi = ds.samples.iter().map(|s| s.time_ms).max().unwrap_or(0);
    if let Some(b) = &opt.bytes {
        if b.iter().any(|o| *o >= ss) {
            return Err(format!("--bytes names an offset past the {ss}-byte sample"));
        }
    }

    let mut wrote = 0usize;
    let mut filled = 0usize;
    let mut worst_dt = 0i32;
    let mut worst_gap_m = 0f32;
    let mut held = 0usize;
    let mut inside: Vec<i32> = Vec::new(); // target instants INSIDE the source's span with no sample
    let mut outside = 0usize; // target instants past either end of the source
    let mut per_ent: Vec<String> = Vec::new();
    let mut note;

    // the source's instants, sorted, for the nearest-instant fill
    let mut src_times: Vec<i32> = by_ms.keys().copied().collect();
    src_times.sort_unstable();

    let mut edit = |rd: &mut RecordData| -> Result<(), String> {
        let cars: Vec<usize> = if opt.all_cars {
            (0..rd.ents.len())
                .filter(|i| {
                    let e = &rd.ents[*i];
                    let cls = rd
                        .descs
                        .get(e.type_ as usize)
                        .map(|d| d.class_id)
                        .unwrap_or(0);
                    cls == gbx::record::CLASS_CSCENEVEHICLEVIS && !e.times.is_empty()
                })
                .collect()
        } else {
            vec![pick_vehicle(rd).ok_or("no vehicle entity to resample")?]
        };
        if cars.is_empty() {
            return Err("no CSceneVehicleVis entity with samples to resample".into());
        }
        for ci in cars {
            let e = &mut rd.ents[ci];
            if e.sample_size != ss {
                return Err(format!(
                    "entity {ci}'s samples are {} B and the source's are {} B",
                    e.sample_size, ss
                ));
            }
            let mut hit = 0usize;
            for (k, t) in e.times.iter().enumerate() {
                // exact instant, or the nearest one within the slack
                let pick: Option<(&[u8], i32)> = match by_ms.get(t) {
                    Some(b) => Some((*b, 0)),
                    None if opt.fill_tol_ms > 0 && *t >= src_lo && *t <= src_hi => {
                        let j = src_times.partition_point(|x| x < t);
                        let cand = [j.checked_sub(1).map(|i| src_times[i]), src_times.get(j).copied()];
                        // how far the car moves between the two source samples
                        // that bracket this instant IS the error bound of the
                        // substitution, and it belongs in the report.
                        if let (Some(a), Some(b)) = (cand[0], cand[1]) {
                            let (pa, pb) = (pos_of(by_ms[&a]), pos_of(by_ms[&b]));
                            let d = ((pa[0] - pb[0]).powi(2)
                                + (pa[1] - pb[1]).powi(2)
                                + (pa[2] - pb[2]).powi(2))
                            .sqrt();
                            if d > worst_gap_m {
                                worst_gap_m = d;
                            }
                        }
                        cand.iter()
                            .flatten()
                            .map(|s| (*s, (*s - *t).abs()))
                            .filter(|(_, d)| *d <= opt.fill_tol_ms)
                            .min_by_key(|(_, d)| *d)
                            .map(|(s, d)| (*by_ms.get(&s).unwrap(), d))
                    }
                    None => None,
                };
                match pick {
                    Some((b, dt)) => {
                        let a = k * ss;
                        if a + ss <= e.raw.len() {
                            match &opt.bytes {
                                Some(sel) => {
                                    for &o in sel {
                                        e.raw[a + o] = b[o];
                                    }
                                }
                                None => e.raw[a..a + ss].copy_from_slice(b),
                            }
                            wrote += 1;
                            hit += 1;
                            if dt != 0 {
                                filled += 1;
                                worst_dt = worst_dt.max(dt);
                            }
                        }
                    }
                    None => {
                        if *t >= src_lo && *t <= src_hi {
                            inside.push(*t);
                        } else if opt.hold_last && *t > src_hi {
                            // The car PARKS where it finished. Without this the
                            // container's tail is another driver still racing,
                            // 90 s of it, and only the video cut keeps him out
                            // of the clip.
                            let a = k * ss;
                            let b = by_ms[&src_hi];
                            if a + ss <= e.raw.len() {
                                e.raw[a..a + ss].copy_from_slice(b);
                                held += 1;
                                hit += 1;
                            }
                        } else {
                            outside += 1;
                        }
                    }
                }
            }
            per_ent.push(format!("e{ci}:{hit}/{}", e.times.len()));
        }
        Ok(())
    };

    if opt.report {
        // Measure against a throwaway copy: a report writes nothing.
        let tmp = format!("{out}.resample-report.tmp");
        rewrite_ghost(inp, &tmp, &mut edit)?;
        let _ = std::fs::remove_file(&tmp);
    } else {
        rewrite_ghost(inp, out, &mut edit)?;
    }

    note = format!(
        "{wrote} sample(s) overwritten{} from {src} ({}), {filled} of them from the NEAREST \
         instant (worst {worst_dt} ms, the car moves {worst_gap_m:.3} m between the two source \
         samples that bracket it), {held} tail instant(s) holding the source's last sample, {} \
         target instant(s) outside the source's {}..{} span left as the target's [{}]",
        opt.bytes
            .as_ref()
            .map(|b| format!(" at {} byte offset(s)", b.len()))
            .unwrap_or_default(),
        if opt.all_cars { "every car entity" } else { "the longest car entity" },
        outside,
        secs(src_lo as i64),
        secs(src_hi as i64),
        per_ent.join(" ")
    );
    if !inside.is_empty() {
        if !opt.report {
            let _ = std::fs::remove_file(out);
        }
        return Err(format!(
            "{} target instant(s) INSIDE the source's own span have no sample at the same time in \
             {src} (first: {:?}). That is a hole in the middle of the overwrite, not a tail, so \
             nothing was written.",
            inside.len(),
            &inside[..inside.len().min(5)]
        ));
    }
    if wrote == 0 {
        if !opt.report {
            let _ = std::fs::remove_file(out);
        }
        return Err(format!(
            "no target instant matched any source instant; the two records do not share a sampling \
             grid. {note}"
        ));
    }
    if opt.report {
        note.push_str("  [--report: nothing written]");
        return Ok(note);
    }

    // THE READ-BACK GATE. Every claim above is about what this function
    // intended; this is what the file says. Re-read the written record and
    // require that every car instant the source covers now carries the
    // source's own bytes.
    let back = gbx::record::load_body(out)?;
    let (bver, bblob) = gbx::record::find_entrecord_blob(&back)?;
    let brd = gbx::record::parse_record_data(&bblob, bver)?;
    let mut checked = 0usize;
    let mut bad = 0usize;
    for e in &brd.ents {
        let cls = brd
            .descs
            .get(e.type_ as usize)
            .map(|d| d.class_id)
            .unwrap_or(0);
        if cls != gbx::record::CLASS_CSCENEVEHICLEVIS || e.sample_size != ss {
            continue;
        }
        for (k, t) in e.times.iter().enumerate() {
            if let Some(want) = by_ms.get(t) {
                checked += 1;
                if &e.raw[k * ss..(k + 1) * ss] != *want {
                    bad += 1;
                }
            }
        }
    }
    if bad > 0 {
        let _ = std::fs::remove_file(out);
        return Err(format!(
            "read-back: {bad} of {checked} instants the source covers do NOT carry the source's \
             bytes in the written file. Nothing kept."
        ));
    }
    note.push_str(&format!(
        "  [read-back: {checked} exact instants carry the source's own bytes]"
    ));
    Ok(note)
}

pub fn graft_scene(inp: &str, out: &str, donor: &str, car_deltas: bool) -> Result<String, String> {    let dbody = gbx::record::load_body(donor)?;
    let (dver, dblob) = gbx::record::find_entrecord_blob(&dbody)?;
    let drd = gbx::record::parse_record_data(&dblob, dver)?;
    let dveh = pick_vehicle(&drd);
    // GRAFT THE SCENE, NOT THE OTHER CARS. This filtered on "not the index the
    // vehicle picker chose", which is right only for a container holding one
    // car. 227654's holds 29 entities and FIVE cars, so the index test grafted
    // four live CSceneVehicleVis records back — phantom cars in the render,
    // the exact hazard `rebuild_to` documents. `rebuild_to` filters on the desc
    // class; so does this now, and so does the `show` counter above.
    let scene: Vec<Ent> = drd
        .ents
        .iter()
        .enumerate()
        .filter(|(i, e)| {
            let cls = drd.descs.get(e.type_ as usize).map(|d| d.class_id).unwrap_or(0);
            cls != gbx::record::CLASS_CSCENEVEHICLEVIS && Some(*i) != dveh
        })
        .map(|(_, e)| e.clone())
        .collect();
    if scene.is_empty() {
        return Err(format!("{donor} has no non-vehicle entity to graft"));
    }
    let mut before = String::new();
    let mut after = String::new();
    rewrite_ghost(inp, out, |rd| {
        let vi = pick_vehicle(rd).ok_or("no vehicle entity: nothing to graft onto")?;
        if rd.ents.len() > 1 {
            return Err(format!(
                "this file already has {} entities -- graft-scene is for a record the rebuild \
                 reduced to one",
                rd.ents.len()
            ));
        }
        let span = rd.end_ms;
        before = format!("1 entity, scene to {}", secs(span as i64));
        let mut car = rd.ents.remove(vi);
        // The car's own delta2 blocks are dropped by the rebuild too — 31 of
        // them on this donor, none on any rebuilt file. They are per-entity
        // side-channel records, not samples, so restoring them cannot move the
        // trajectory; whether the client needs them is exactly the open
        // question this flag exists to ask.
        let mut car_note = String::new();
        if car_deltas {
            if let Some(di) = dveh {
                let mut d = drd.ents[di].deltas2.clone();
                d.retain(|(_, t, _)| *t <= span);
                car_note = format!(", car deltas2 0 -> {}", d.len());
                car.deltas2 = d;
            }
        }
        let mut added = Vec::new();
        for e in &scene {
            let mut e = e.clone();
            // Clip to the span: a scene entity that outlives the run is the
            // defect `shorten` removes, and this must not put one back.
            let keep = e.times.iter().take_while(|t| **t <= span).count();
            if keep < e.times.len() && e.sample_size > 0 {
                e.times.truncate(keep);
                e.raw.truncate(keep * e.sample_size);
            }
            e.u03 = e.u03.min(span);
            e.deltas2.retain(|(_, t, _)| *t <= span);
            let cls = drd.descs.get(e.type_ as usize).map(|d| d.class_id).unwrap_or(0);
            added.push(format!(
                "0x{:08X} x{}@{}B",
                cls,
                e.times.len(),
                e.sample_size
            ));
            rd.ents.push(e);
        }
        rd.ents.push(car);
        after = format!("{} entities (grafted {}){}", rd.ents.len(), added.join(", "), car_note);
        Ok(())
    })?;
    let a = gbx::record::decode_ghost(inp)?;
    let b = gbx::record::decode_ghost(out)?;
    if a.samples.len() != b.samples.len() || a.raw != b.raw {
        return Err(format!(
            "the car's samples changed ({} -> {} samples) -- refusing, this operation must not \
             touch the trajectory",
            a.samples.len(),
            b.samples.len()
        ));
    }
    Ok(format!("{before} -> {after}; the car's {} samples are byte-identical", a.samples.len()))
}

fn rebuild(a: &[String]) {
    let inp = a.first().unwrap_or_else(|| die("ghost record rebuild IN OUT --span MS"));
    let out = a.get(1).unwrap_or_else(|| die("ghost record rebuild IN OUT --span MS"));
    let span: i64 =
        num(a, "--span").unwrap_or_else(|| die("--span MS: how far the record must reach"));
    let period: i64 = num(a, "--period").unwrap_or(50);
    let tf = flag(a, "--template-from");
    match rebuild_to(inp, out, span, tf, period, has(a, "--hold-last")) {
        // The second line is only true of a grid that WAS stamped from one
        // template sample. When the car already covered the grid its own
        // samples are kept, and telling the operator the file is not a run
        // would be false -- `rebuild_to` says which happened.
        Ok(msg) if msg.contains("were KEPT") => println!("{out}: {msg}"),
        Ok(msg) => println!(
            "{out}: {msg}\nThis file is NOT yet a run -- every sample is a copy of the template. \
             `ghost regen` is what makes it one, and it does this step itself."
        ),
        Err(e) => die(e),
    }
}

/// The sample's position, three f32 at 47 -- `gbx::sample`'s `TRANSFORM`.
fn pos_of(s: &[u8]) -> [f32; 3] {
    let f = |o: usize| f32::from_le_bytes(s[o..o + 4].try_into().unwrap());
    [f(47), f(51), f(55)]
}
