//! `ghost splice` -- delete whole intervals out of the MIDDLE of a run and
//! close the gap, so what is left is the driver's own ticks in their own order
//! with the deleted material simply absent.
//!
//! `ghost trim` owns a run's two ENDS. This owns its middle, and the two are
//! different operations: a trim keeps every surviving tick's race time, a
//! splice cannot -- closing a gap moves everything after it earlier. So a
//! splice has to retime the tape, the telemetry, the splits and the declared
//! time together, and a file where those four disagree is the failure mode this
//! command exists to make impossible.
//!
//! # THE RETRY RULE (`--rule retries`), stated so it can be re-derived
//!
//! A respawn is an editable INPUT: bit 31 of an input packet's state literal,
//! which unpacks into `word0` bit 5 (`Packet::respawn()`). On a map with
//! checkpoints it is a SOFT respawn -- the car is restored to the state it had
//! when it crossed the checkpoint it is respawning to. So the material between
//! a checkpoint crossing and the last respawn before the NEXT checkpoint is,
//! exactly, the attempts at that segment that failed, and the state on either
//! side of that material is the same checkpoint-crossing state.
//!
//! Let the recording declare crossings `s_1 < s_2 < ... < s_n` (the splits;
//! `s_n` is the finish), and let segment `k` be the ticks with
//! `s_{k-1} < t <= s_k` (segment 1 also holds every tick before the start
//! line). Let `a_k` be the LAST tick of segment `k` that carries a respawn
//! press, if the segment has one. Then:
//!
//! > **RULE R.** For every segment that contains a respawn press, delete every
//! > tick of that segment from its beginning up to and INCLUDING `a_k`.
//! > Delete nothing else. Do not shorten, reorder or edit any surviving tick.
//!
//! What survives is one attempt per segment -- the one that reached the
//! checkpoint -- and the run's new time is
//! `s_n - 10 ms x (number of deleted ticks)`. The respawn presses themselves go
//! with the material they belong to, so a spliced file carries ZERO respawn
//! ticks and the gate below asserts it.
//!
//! **This is not a lap and the command will not let a file pretend it is.** At
//! each junction the car's state jumps -- by however far the respawn moved it,
//! which the command MEASURES and prints from the file's own recording. A
//! splice is a study artefact: the driver's own driving with the retries taken
//! out. What it is not is a run somebody drove, and re-simulating one is not
//! expected to return the number it declares.
//!
//! # `--drop A..B`
//!
//! Extra intervals to delete, in the ORIGINAL file's race milliseconds, applied
//! together with (or instead of) the rule. This is how a de-looped artefact is
//! built: hand it the intervals a loop census found. The command does not
//! decide what a loop is; it deletes exactly the intervals it is given and
//! reports the state jump at each one, so the caller's rule has to be stated
//! somewhere the caller can be held to.

use crate::cli::{die, flag, has, num};
use crate::oracle::{self, MapsMode};
use crate::trim::{rewrite_result, set_all_declared};
use gbx::container::{secs, Container};
use gbx::tape::{Encoding, StateEnc, Tape};

/// A half-open interval of tick indices to delete.
#[derive(Clone, Copy, Debug)]
struct Cut {
    lo: usize,
    hi: usize, // exclusive
    why: &'static str,
}

pub fn cmd(a: &[String]) {
    let inp = a
        .first()
        .unwrap_or_else(|| die("ghost splice IN OUT [--rule retries] [--drop A..B,...]"))
        .clone();
    let out = a
        .get(1)
        .unwrap_or_else(|| die("ghost splice IN OUT [--rule retries] [--drop A..B,...]"))
        .clone();
    let rule = flag(a, "--rule").unwrap_or("").to_string();
    if !rule.is_empty() && rule != "retries" {
        die(format!("unknown --rule {:?} (this command knows `retries`)", rule));
    }
    let drops = flag(a, "--drop").unwrap_or("").to_string();
    if rule.is_empty() && drops.is_empty() {
        die("nothing to do: give --rule retries and/or --drop A..B,...");
    }
    let keep_record = has(a, "--keep-record");
    let driver_only = has(a, "--driver-only");

    let c = Container::load(&inp).unwrap_or_else(|e| die(e));
    let t = Tape::from_file(&inp).unwrap_or_else(|e| die(e));
    t.verbatim_is_identity()
        .unwrap_or_else(|e| die(format!("refusing to splice a file whose tape does not round-trip: {}", e)));
    if t.archives.len() != 1 {
        die(format!(
            "this file has {} input archives; a splice would have to choose one to cut and \
             nothing in the file says which. Refusing.",
            t.archives.len()
        ));
    }
    let so = t.archives[0].start_offset_ms as i64;
    let n = t.archives[0].packets.len();
    let tms = |i: usize| so + 10 * i as i64;
    // The first tick at or after `ms`.
    let first_at = |ms: i64| -> usize { (((ms - so) + 9) / 10).clamp(0, n as i64) as usize };

    let splits: Vec<i32> = c.splits();
    let declared_before = c.declared_times().first().map(|x| x.1 as i64).unwrap_or(0);

    // ---- the cuts -------------------------------------------------------
    let mut cuts: Vec<Cut> = Vec::new();
    let mut seg_report: Vec<(usize, i64, i64, usize, Option<i64>)> = Vec::new();
    if rule == "retries" {
        if splits.is_empty() {
            die("--rule retries needs the file's checkpoint list, and this container declares none");
        }
        let mut lo = 0usize; // first tick of the segment
        for (k, s) in splits.iter().enumerate() {
            // segment k+1 is [lo, hi) with hi the first tick past the crossing
            let hi = first_at(*s as i64 + 1).max(lo);
            let mut last_resp: Option<usize> = None;
            let mut nresp = 0usize;
            for i in lo..hi {
                if t.archives[0].packets[i].respawn() {
                    last_resp = Some(i);
                    nresp += 1;
                }
            }
            seg_report.push((
                k + 1,
                if k == 0 { 0 } else { splits[k - 1] as i64 },
                *s as i64,
                nresp,
                last_resp.map(tms),
            ));
            if let Some(r) = last_resp {
                cuts.push(Cut { lo, hi: r + 1, why: "retries" });
            }
            lo = hi;
        }
    }
    for spec in drops.split(',').filter(|s| !s.is_empty()) {
        let (x, y) = spec
            .split_once("..")
            .unwrap_or_else(|| die(format!("--drop wants A..B in milliseconds, got {:?}", spec)));
        let (x, y): (i64, i64) = (
            x.trim().parse().unwrap_or_else(|_| die(format!("bad --drop start {:?}", x))),
            y.trim().parse().unwrap_or_else(|_| die(format!("bad --drop end {:?}", y))),
        );
        if y <= x {
            die(format!("--drop {}..{}: the end must be after the start", x, y));
        }
        cuts.push(Cut { lo: first_at(x), hi: first_at(y), why: "drop" });
    }
    cuts.retain(|c| c.hi > c.lo);
    cuts.sort_by_key(|c| c.lo);
    for w in cuts.windows(2) {
        if w[1].lo < w[0].hi {
            die(format!(
                "two cuts overlap: {} .. {} and {} .. {}. A splice deletes disjoint intervals; \
                 merge them and say so.",
                secs(tms(w[0].lo)),
                secs(tms(w[0].hi)),
                secs(tms(w[1].lo)),
                secs(tms(w[1].hi))
            ));
        }
    }
    if cuts.is_empty() {
        die("the rule and the --drop list select nothing to cut");
    }

    // ---- the kept ticks, and the retiming map ---------------------------
    let mut cut_of = vec![false; n];
    for cu in &cuts {
        for i in cu.lo..cu.hi.min(n) {
            cut_of[i] = true;
        }
    }
    // `shift[i]` = milliseconds deleted STRICTLY BEFORE original tick i.
    let mut shift = vec![0i64; n + 1];
    for i in 0..n {
        shift[i + 1] = shift[i] + if cut_of[i] { 10 } else { 0 };
    }
    let total_cut_ms = shift[n];
    let retime = |ms: i64| -> i64 {
        // milliseconds deleted at or before `ms`
        let mut d = 0i64;
        for cu in &cuts {
            let (a, b) = (tms(cu.lo), tms(cu.hi));
            if ms >= b {
                d += b - a;
            } else if ms >= a {
                d += ms - a;
            }
        }
        ms - d
    };
    let in_cut = |ms: i64| -> bool { cuts.iter().any(|cu| ms >= tms(cu.lo) && ms < tms(cu.hi)) };


    // ---- the tape -------------------------------------------------------
    let mut nt = t.clone();
    {
        let ar = &mut nt.archives[0];
        let old = std::mem::take(&mut ar.packets);
        let mut kept: Vec<gbx::tape::Packet> = Vec::with_capacity(n - (total_cut_ms / 10) as usize);
        let mut prev_orig: Option<usize> = None;
        for (i, p) in old.into_iter().enumerate() {
            if cut_of[i] {
                continue;
            }
            let mut p = p;
            let contiguous = prev_orig.map(|j| j + 1 == i).unwrap_or(false);
            if !contiguous {
                // The packet before this one is no longer in the file, so a
                // "same as the previous tick" state word would now repeat a
                // DIFFERENT word. Freeze it into the literal the decoder itself
                // derives from the word it had.
                if let StateEnc::Prev | StateEnc::Prev2(_, _) = p.state {
                    p.state = StateEnc::Lit(gbx::tape::literal_for(p.word0, p.flags));
                }
                p.vsame = false;
            }
            prev_orig = Some(i);
            kept.push(p);
        }
        ar.packets = kept;
        // `start_offset_ms` does not move: the first kept tick keeps its own
        // race time unless it was itself cut, and the rule never cuts before
        // the start line in a segment with no respawn.
        if cut_of[0] {
            ar.start_offset_ms = retime(so) as i32;
        }
    }
    let new_n = nt.archives[0].packets.len();

    // ---- declared time and splits ---------------------------------------
    let declared = retime(declared_before);
    let new_splits: Vec<i64> = splits.iter().map(|s| retime(*s as i64)).collect();

    let mut body = nt.splice_into(c.body(), Encoding::Explicit).unwrap_or_else(|e| die(e));
    set_all_declared(&mut body, declared as u32);
    let body = rewrite_result(&body, |r| {
        r.race_ms = declared as i32;
        for e in r.entries.iter_mut() {
            e.0 = retime(e.0 as i64) as i32;
        }
    })
    .unwrap_or_else(|e| die(e));

    let tmp = format!("{}.splice-stage", out);
    gbx::container::write_gbx(&c.gbx, body, &tmp).unwrap_or_else(|e| die(e));

    // ---- the telemetry ---------------------------------------------------
    let mut kept_s = 0usize;
    let mut dropped_s = 0usize;
    let mut pruned = 0usize;
    let mut merged: Option<(usize, usize)> = None; // (lives merged, other vehicle entities dropped)
    let had_record = !keep_record
        && gbx::recwrite::find_rec_site(&Container::load(&tmp).unwrap().gbx.body).is_ok();
    if had_record {
        let r = gbx::recwrite::rewrite_ghost(&tmp, &out, |rd| {
            // --driver-only: collapse the driver's tiled lives into ONE entity
            // and drop every other car. A splice breaks the tiling (the lives
            // on either side of a cut are no longer 10 ms apart), so a spliced
            // multi-car record leaves the BIGGEST entity -- another player --
            // as what every stock reader and every renderer would call the run.
            // Merging is done BEFORE the time rewrite, while the chain is still
            // recoverable.
            if driver_only {
                let (chain, _, _) = player_chain(rd);
                if chain.len() > 1 || rd.ents.iter().filter(|e| e.sample_size >= 103).count() > 1 {
                    let ss = rd.ents[chain[0]].sample_size;
                    if chain.iter().all(|i| rd.ents[*i].sample_size == ss) {
                        let mut times: Vec<i32> = Vec::new();
                        let mut raw: Vec<u8> = Vec::new();
                        let mut d2: Vec<(i32, i32, Vec<u8>)> = Vec::new();
                        for &i in &chain {
                            times.extend_from_slice(&rd.ents[i].times);
                            raw.extend_from_slice(&rd.ents[i].raw);
                            d2.extend(rd.ents[i].deltas2.iter().cloned());
                        }
                        let head = chain[0];
                        let mut veh_dropped = 0usize;
                        let keepset: Vec<bool> = (0..rd.ents.len())
                            .map(|i| {
                                if i == head {
                                    true
                                } else if chain.contains(&i) {
                                    false
                                } else if rd.ents[i].sample_size >= 103 && !rd.ents[i].times.is_empty() {
                                    // another car in the same record
                                    let cid = rd
                                        .descs
                                        .get(rd.ents[i].type_.max(0) as usize)
                                        .filter(|_| rd.ents[i].type_ >= 0)
                                        .map(|d| d.class_id);
                                    if cid == Some(gbx::record::CLASS_CSCENEVEHICLEVIS) {
                                        veh_dropped += 1;
                                        false
                                    } else {
                                        true
                                    }
                                } else {
                                    true
                                }
                            })
                            .collect();
                        rd.ents[head].times = times;
                        rd.ents[head].raw = raw;
                        rd.ents[head].deltas2 = d2;
                        let mut k = 0usize;
                        rd.ents.retain(|_| {
                            k += 1;
                            keepset[k - 1]
                        });
                        merged = Some((chain.len(), veh_dropped));
                    }
                }
            }
            for e in rd.ents.iter_mut() {
                if e.times.is_empty() || e.sample_size == 0 {
                    continue;
                }
                let ss = e.sample_size;
                let keep: Vec<bool> = e.times.iter().map(|t| !in_cut(*t as i64)).collect();
                let mut nt2: Vec<i32> = Vec::new();
                let mut nraw: Vec<u8> = Vec::new();
                for (i, k) in keep.iter().enumerate() {
                    if *k {
                        nt2.push(retime(e.times[i] as i64) as i32);
                        nraw.extend_from_slice(&e.raw[i * ss..(i + 1) * ss]);
                    } else {
                        dropped_s += 1;
                    }
                }
                kept_s += nt2.len();
                e.times = nt2;
                e.raw = nraw;
                e.deltas2.retain(|(_, t, _)| !in_cut(*t as i64));
                for d in e.deltas2.iter_mut() {
                    d.1 = retime(d.1 as i64) as i32;
                }
            }
            let last = rd.ents.iter().filter_map(|e| e.times.last().copied()).max();
            let first = rd.ents.iter().filter_map(|e| e.times.first().copied()).min();
            rd.start_ms = first.unwrap_or(0);
            rd.end_ms = last.unwrap_or(declared as i32);
            // An entity left with no samples crashes the game client on import
            // (see `ghost trim`), so it is removed and the count is printed.
            let before = rd.ents.len();
            rd.ents.retain(|e| !e.times.is_empty() && e.sample_size > 0);
            pruned = before - rd.ents.len();
            Ok(())
        });
        if let Err(e) = r {
            die(format!("rewriting the telemetry record: {}", e));
        }
        let _ = std::fs::remove_file(&tmp);
    } else {
        std::fs::rename(&tmp, &out).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
    }

    // ---- the gate: re-read what was written -----------------------------
    let c2 = Container::load(&out).unwrap_or_else(|e| die(e));
    let t2 = Tape::from_file(&out).unwrap_or_else(|e| die(e));
    let mut fail: Vec<String> = Vec::new();
    if t2.n() != new_n {
        fail.push(format!("tick count {} on disk, {} written", t2.n(), new_n));
    }
    if rule == "retries" && !drops.is_empty() {
        // no extra obligation
    }
    let resp_left = t2.archives[0].packets.iter().filter(|p| p.respawn()).count();
    if rule == "retries" && resp_left != 0 {
        fail.push(format!("{} respawn ticks survive the retry rule", resp_left));
    }
    for (k, (p, q)) in nt.archives[0].packets.iter().zip(t2.archives[0].packets.iter()).enumerate() {
        if p.steer != q.steer || p.accel != q.accel || p.brake != q.brake || p.word0 != q.word0 {
            fail.push(format!("tick {} does not read back", k));
            break;
        }
    }
    let dts: Vec<u32> = c2.declared_times().into_iter().map(|x| x.1).collect();
    if dts.iter().any(|v| *v as i64 != declared) {
        fail.push(format!("declared time copies disagree: {:?}", dts));
    }
    let sp2: Vec<i32> = c2.splits();
    if sp2.iter().map(|x| *x as i64).ne(new_splits.iter().copied()) {
        fail.push(format!("splits on disk {:?} are not the retimed ones", sp2));
    }
    if sp2.windows(2).any(|w| w[1] < w[0]) {
        fail.push("the retimed splits are not increasing".into());
    }
    if had_record {
        // Every surviving sample must be a retimed original -- so the check is
        // not a bound but an identity: recompute the whole expected time vector
        // from the INPUT file and require the output's to equal it. (A bound
        // like `<= declared` would be wrong anyway: this recording legitimately
        // runs 0.345 s past its own finish.)
        let want: Option<Vec<i64>> = if driver_only {
            gbx::record::load_body(&inp)
                .ok()
                .and_then(|b| gbx::record::find_entrecord_blob(&b).ok())
                .and_then(|(v, blob)| gbx::record::parse_record_data(&blob, v).ok())
                .map(|rec| {
                    let (_, track, _) = player_chain(&rec);
                    track.iter().map(|(t, _)| *t as i64).filter(|t| !in_cut(*t)).map(retime).collect()
                })
        } else {
            gbx::record::decode_ghost(&inp).ok().map(|d| {
                d.samples.iter().map(|s| s.time_ms as i64).filter(|t| !in_cut(*t)).map(retime).collect()
            })
        };
        if let (Ok(d), Some(want)) = (gbx::record::decode_ghost(&out), want) {
            let got: Vec<i64> = d.samples.iter().map(|s| s.time_ms as i64).collect();
            if got != want {
                let k = got.iter().zip(want.iter()).position(|(a, b)| a != b);
                fail.push(format!(
                    "the vehicle entity's retimed sample times are not the ones the cuts imply \
                     ({} on disk, {} expected{})",
                    got.len(),
                    want.len(),
                    match k {
                        Some(i) => format!(", first disagreement at index {}: {} vs {}", i, secs(got[i]), secs(want[i])),
                        None => String::new(),
                    }
                ));
            }
        }
    }
    if !fail.is_empty() {
        for f in &fail {
            eprintln!("  FAIL {}", f);
        }
        die("the spliced file failed its own coherence gate; it has been left in place for inspection");
    }

    // ---- the report ------------------------------------------------------
    println!("wrote {}", out);
    println!("  rule            {}", if rule.is_empty() { "(--drop only)" } else { "retries (R): cut each segment from its start through its LAST respawn press" });
    if rule == "retries" {
        println!("  segments        seg  from       to         respawns  last press   cut");
        for (k, from, to, nresp, last) in &seg_report {
            let cut = cuts
                .iter()
                .find(|c| c.why == "retries" && tms(c.hi) > *from && tms(c.lo) <= *to)
                .map(|c| tms(c.hi) - tms(c.lo))
                .unwrap_or(0);
            println!(
                "                  {:>3}  {:>9}  {:>9}  {:>8}  {:>10}  {:>9}",
                k,
                secs(*from),
                secs(*to),
                nresp,
                last.map(secs).unwrap_or_else(|| "-".into()),
                if cut > 0 { secs(cut) } else { "-".into() }
            );
        }
    }
    println!("  cuts            {} intervals, {} ticks, {} deleted", cuts.len(), total_cut_ms / 10, secs(total_cut_ms));
    for cu in &cuts {
        println!(
            "                  {} .. {}   {}   ({})",
            secs(tms(cu.lo)),
            secs(tms(cu.hi)),
            secs(tms(cu.hi) - tms(cu.lo)),
            cu.why
        );
    }
    println!("  ticks           {} -> {}", n, new_n);
    println!("  declared        {} -> {}  in {} copies, all equal", secs(declared_before), secs(declared), dts.len());
    println!(
        "  checkpoints     {:?}",
        sp2.iter().map(|s| secs(*s as i64)).collect::<Vec<_>>()
    );
    if had_record {
        println!("  telemetry       {} samples kept, {} dropped, times remapped", kept_s, dropped_s);
        if let Some((lives, others)) = merged {
            println!(
                "  driver-only     the driver's {} tiled lives merged into ONE entity; {} other cars\n\
                \x20                 dropped from the record. A splice breaks the tiling, so without\n\
                \x20                 this the biggest entity left in the file -- what every stock\n\
                \x20                 reader calls the run -- is another player on the server.",
                lives, others
            );
        }
        if pruned > 0 {
            println!("  entities        {} emptied by the splice and REMOVED", pruned);
        }
    } else {
        println!("  telemetry       untouched (--keep-record): the recording still spans the WHOLE original run");
    }
    junction_report(&inp, &cuts, &tms);
    println!(
        "  NOTE            this file is a SPLICE, not a lap. The car's state jumps at every\n\
        \x20                 junction above, so the plain oracle re-simulating it is NOT expected\n\
        \x20                 to return {}. Publish it as what it is.",
        secs(declared)
    );

    if let Some(sd) = flag(a, "--server") {
        if !has(a, "--no-oracle") {
            let mapf = flag(a, "--map");
            let server = oracle::server_dir(Some(sd));
            let mode = match mapf {
                Some(m) => MapsMode::One(std::path::Path::new(m)),
                None => MapsMode::Empty,
            };
            match oracle::validate(&server, std::path::Path::new(&out), mode, "splice") {
                Ok(r) => println!("  oracle          {} (cps {:?})  -- reported, never expected to match", r.secs(), r.cps),
                Err(e) => println!("  oracle          not run: {}", e),
            }
        }
    }
    let _ = num(a, "--unused");
}

/// THE JUNCTION JUMP: how far the car moves, in the recording, across each
/// deleted interval. This is the number that says how much of a fiction the
/// splice is, and it is measured off the ORIGINAL file rather than asserted.
///
/// It cannot use `record::decode_ghost`, which returns the single
/// `CSceneVehicleVis` entity with the most samples. On a server-recorded
/// marathon that is the WRONG CAR -- 153527's record holds 55 entities, the
/// biggest of them another player on the server, while the driver is 46 short
/// per-life entities tiling the race (the car is destroyed and recreated at
/// every respawn, which is exactly where a junction falls).
///
/// So the driver is recovered by TILING plus PATH LENGTH, which are rules and
/// not guesses. Every vehicle entity is used as a seed for a maximal chain:
/// from that entity, repeatedly take the entity whose first sample is exactly
/// one 10 ms step after the current one's last. A car destroyed and recreated
/// at a respawn tiles with itself; a second player on the same server does not
/// line up with it. Then the chain that DROVE is the one with the greatest
/// path length -- this record holds spectator cars parked at the spawn for the
/// whole race, which tile perfectly and cover the whole span while travelling
/// zero metres, so span and sample count cannot separate them and path length
/// can. Steps over 15 m in one 10 ms sample are teleports and are not counted
/// as path. The top candidates are printed, so a wrong pick is visible.
fn player_chain(rec: &gbx::record::RecordData) -> (Vec<usize>, Vec<(i32, [f64; 3])>, Vec<(usize, usize, f64)>) {
    let mut veh: Vec<usize> = Vec::new();
    for (i, e) in rec.ents.iter().enumerate() {
        let cid = rec.descs.get(e.type_.max(0) as usize).filter(|_| e.type_ >= 0).map(|d| d.class_id);
        if cid == Some(gbx::record::CLASS_CSCENEVEHICLEVIS) && e.sample_size >= 103 && !e.times.is_empty() {
            veh.push(i);
        }
    }
    let samples_of = |i: usize| -> Vec<(i32, [f64; 3])> {
        let e = &rec.ents[i];
        let ss = e.sample_size;
        e.times
            .iter()
            .enumerate()
            .map(|(k, t)| {
                let s = gbx::record::decode_vehicle_sample(&e.raw[k * ss..(k + 1) * ss]);
                (*t, [s.x as f64, s.y as f64, s.z as f64])
            })
            .collect()
    };
    let build = |seed: usize| -> Vec<usize> {
        let mut chain = vec![seed];
        let mut used = vec![false; rec.ents.len()];
        used[seed] = true;
        loop {
            let last = *rec.ents[*chain.last().unwrap()].times.last().unwrap();
            let next = veh.iter().copied().filter(|i| !used[*i] && rec.ents[*i].times[0] == last + 10).min();
            match next {
                Some(i) => {
                    used[i] = true;
                    chain.push(i);
                }
                None => break,
            }
        }
        chain
    };
    let mut best: Option<(f64, Vec<usize>, Vec<(i32, [f64; 3])>)> = None;
    let mut table: Vec<(usize, usize, f64)> = Vec::new();
    for &seed in &veh {
        let chain = build(seed);
        let mut track: Vec<(i32, [f64; 3])> = Vec::new();
        for &i in &chain {
            track.extend(samples_of(i));
        }
        track.sort_by_key(|(t, _)| *t);
        let mut path = 0.0f64;
        for w in track.windows(2) {
            let d = ((w[0].1[0] - w[1].1[0]).powi(2) + (w[0].1[1] - w[1].1[1]).powi(2) + (w[0].1[2] - w[1].1[2]).powi(2)).sqrt();
            if d <= 15.0 {
                path += d;
            }
        }
        table.push((chain.len(), track.len(), path));
        if best.as_ref().map(|b| path > b.0).unwrap_or(true) {
            best = Some((path, chain, track));
        }
    }
    table.sort_by(|a, b| b.2.total_cmp(&a.2));
    table.truncate(4);
    match best {
        Some((_, c, t)) => (c, t, table),
        None => (vec![], vec![], table),
    }
}

fn junction_report(path: &str, cuts: &[Cut], tms: &dyn Fn(usize) -> i64) {
    let body = match gbx::record::load_body(path) {
        Ok(b) => b,
        Err(_) => return,
    };
    let (version, blob) = match gbx::record::find_entrecord_blob(&body) {
        Ok(x) => x,
        Err(_) => return,
    };
    let rec = match gbx::record::parse_record_data(&blob, version) {
        Ok(r) => r,
        Err(_) => return,
    };
    let (chain, track, table) = player_chain(&rec);
    if track.is_empty() {
        return;
    }
    println!(
        "  driver track    {} tiled entities, {} samples, {} .. {}, path {:.0} m",
        chain.len(),
        track.len(),
        secs(track[0].0 as i64),
        secs(track[track.len() - 1].0 as i64),
        table[0].2
    );
    println!("                  runners-up (lives, samples, path m): {:?}", &table[1..]);
    let at = |ms: i64, side: i64| -> Option<(i64, [f64; 3])> {
        let mut best: Option<(i64, i64, [f64; 3])> = None;
        for (t, p) in &track {
            let d = *t as i64 - ms;
            if (side < 0 && d > 0) || (side > 0 && d < 0) {
                continue;
            }
            if best.map(|b| d.abs() < b.0).unwrap_or(true) {
                best = Some((d.abs(), *t as i64, *p));
            }
        }
        best.map(|(_, t, p)| (t, p))
    };
    println!("  junctions       the driver's own state jump across each cut:");
    for cu in cuts {
        let (a, b) = (tms(cu.lo), tms(cu.hi));
        match (at(a - 10, -1), at(b, 1)) {
            (Some(p), Some(q)) => println!(
                "                  cut at {:>9}  jump {:>8.2} m   ({:.1},{:.1},{:.1}) @{} -> ({:.1},{:.1},{:.1}) @{}",
                secs(a),
                ((p.1[0] - q.1[0]).powi(2) + (p.1[1] - q.1[1]).powi(2) + (p.1[2] - q.1[2]).powi(2)).sqrt(),
                p.1[0], p.1[1], p.1[2], secs(p.0),
                q.1[0], q.1[1], q.1[2], secs(q.0)
            ),
            _ => println!(
                "                  cut at {:>9}  jump UNMEASURED (no driver sample within 200 ms of an end)",
                secs(a)
            ),
        }
    }
}

/// `ghost record chain FILE` -- who is the driver, on a record that holds more
/// than one car.
///
/// The stock reader takes the `CSceneVehicleVis` entity with the most samples.
/// That is right on a solo ghost and WRONG on a server recording, where the
/// driver's car is destroyed and recreated at every respawn and the longest
/// single entity can be somebody else. This prints the tiling chains and the
/// path each one drove, so the pick can be seen.
pub fn print_chain(path: &str) {
    let body = gbx::record::load_body(path).unwrap_or_else(|e| die(e));
    let (version, blob) = gbx::record::find_entrecord_blob(&body).unwrap_or_else(|e| die(e));
    let rec = gbx::record::parse_record_data(&blob, version).unwrap_or_else(|e| die(e));
    let (chain, track, table) = player_chain(&rec);
    if track.is_empty() {
        die("no CSceneVehicleVis entity in this record");
    }
    println!("{}", path);
    println!(
        "  driver chain    {} tiled entities, {} samples, {} .. {}, path {:.0} m",
        chain.len(),
        track.len(),
        secs(track[0].0 as i64),
        secs(track[track.len() - 1].0 as i64),
        table[0].2
    );
    println!("  runners-up      (lives, samples, path m) {:?}", &table[1..]);
    println!("  lives           #   ent    t_first     t_last     n");
    for (k, &i) in chain.iter().enumerate() {
        let e = &rec.ents[i];
        println!(
            "                  {:>3}  {:>4}  {:>9}  {:>9}  {:>6}",
            k,
            i,
            secs(*e.times.first().unwrap() as i64),
            secs(*e.times.last().unwrap() as i64),
            e.times.len()
        );
    }
}

/// `ghost record life FILE --life N [--csv OUT]` -- the RECORDED trajectory of
/// ONE life of a server recording.
///
/// `record chain` says which entities tile the driver's race; this prints what
/// the car actually did inside one of them. On a practice replay that is the
/// only honest way to see an individual attempt: the stock reader takes the
/// entity with the most samples and hands back somebody else's car, or another
/// attempt, with no warning.
pub fn life_csv(path: &str, life: usize, out: Option<&str>, shift_ms: i32) {
    let body = gbx::record::load_body(path).unwrap_or_else(|e| die(e));
    let (version, blob) = gbx::record::find_entrecord_blob(&body).unwrap_or_else(|e| die(e));
    let rec = gbx::record::parse_record_data(&blob, version).unwrap_or_else(|e| die(e));
    let (chain, _, _) = player_chain(&rec);
    if chain.is_empty() {
        die("no CSceneVehicleVis entity in this record");
    }
    if life >= chain.len() {
        die(format!("--life {} but the driver chain has {} lives", life, chain.len()));
    }
    let e = &rec.ents[chain[life]];
    let ss = e.sample_size;
    let mut s = String::from(
        "time_ms,x,y,z,vx,vy,vz,speed_kmh,yaw,pitch,roll,steer,gas,brake,ground,wetness,gear\n",
    );
    let mut path = 0.0f64;
    let mut prev: Option<[f64; 3]> = None;
    for (k, t) in e.times.iter().enumerate() {
        let sm = gbx::record::decode_vehicle_sample(&e.raw[k * ss..(k + 1) * ss]);
        if let Some(p) = prev {
            let d = ((p[0] - sm.x as f64).powi(2)
                + (p[1] - sm.y as f64).powi(2)
                + (p[2] - sm.z as f64).powi(2))
            .sqrt();
            if d <= 15.0 {
                path += d;
            }
        }
        prev = Some([sm.x as f64, sm.y as f64, sm.z as f64]);
        s.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.5},{:.5},{:.5},{:.4},{:.4},{:.4},{},{:.4},{:.3}\n",
            t + shift_ms, sm.x, sm.y, sm.z, sm.vx, sm.vy, sm.vz, sm.speed_kmh, sm.yaw, sm.pitch, sm.roll,
            sm.steer, sm.gas, sm.brake, if sm.is_ground_contact { 1 } else { 0 }, sm.wetness, sm.gear
        ));
    }
    match out {
        Some(f) => {
            std::fs::write(f, &s).unwrap_or_else(|e| die(format!("{}: {}", f, e)));
            println!(
                "wrote {}  (life {} of {}, entity {}, {} samples, {} .. {}, path {:.0} m)",
                f,
                life,
                chain.len(),
                chain[life],
                e.times.len(),
                secs(*e.times.first().unwrap() as i64),
                secs(*e.times.last().unwrap() as i64),
                path
            );
        }
        None => print!("{}", s),
    }
}
