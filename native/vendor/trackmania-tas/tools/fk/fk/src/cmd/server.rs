//! `fk server` — start a fork server, and prove it is telling the truth.
//!
//! Three verbs, and they are three different questions:
//!
//! * `probe` — *where did this server actually stop?* Cheap (about 2 s), so the
//!   load studies can run it hundreds of times.
//! * `check` — *does a resume give the same answer as a full validation?* This
//!   is THE control for the whole crate and it is the acceptance test for any
//!   new map or fork configuration.
//! * `bench` — *how much faster is it?* Against a BATCHED baseline, because
//!   nearly all of a validation's cost is the server launch.
//!
//! The old surface was `fk fs --mode auto|edge|test|bench|cal|scan` plus a
//! separate `fk fsprobe`: six modes behind one flag, three of which
//! (`test`, `cal`, `scan`) were subsets of `auto` reachable only by reading the
//! source. They are gone; `check` does what `auto` did, and its boundary sweep
//! is what `edge` and `cal` did.

use crate::oracle::validate_batch;
use crate::session::{Checkpoint, Engine, Session};
use crate::tape::Tape;
use forkoracle::forksrv::parse_result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// xorshift64. Deterministic from a seed, so a mismatch is reproducible: the
/// candidate set is part of the evidence, not an accident of the run.
pub struct Rng(u64);
impl Rng {
    pub fn new(s: u64) -> Rng {
        Rng(s ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A candidate: the reference tape with a run of up to `span` ticks perturbed
/// somewhere at or after `from`.
///
/// The four perturbation kinds are not decoration. The engine consumes steer,
/// gas and brake at DIFFERENT instants — at a 94.9% checkpoint the steer of
/// tick 2313 was still live while the gas and brake of ticks 2313–2315 had
/// already been taken — so a candidate set that only moves steer cannot see a
/// boundary error on the other two axes.
pub fn make_candidate(
    tape: &Tape,
    from: usize,
    span: usize,
    rng: &mut Rng,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut steer, mut accel, mut brake) =
        (tape.steer.clone(), tape.accel.clone(), tape.brake.clone());
    let n = tape.n();
    if from >= n {
        return (steer, accel, brake);
    }
    let start = from + rng.below(n - from);
    let end = (start + 1 + rng.below(span)).min(n);
    let kind = rng.below(4);
    for t in start..end {
        match kind {
            0 => {
                let d = (rng.below(120) as i32) - 60;
                steer[t] = ((steer[t] as i8 as i32 + d).clamp(-127, 127) as i8) as u8;
            }
            1 => steer[t] = (rng.below(255) as u8).max(1),
            2 => accel[t] = 1 - accel[t].min(1),
            _ => brake[t] = 1 - brake[t].min(1),
        }
    }
    (steer, accel, brake)
}

/// The exact first tick a resume may rewrite.
///
/// Perturb every one of the three axes at every tick in a window around the
/// page-fault probe, run each both ways, and take the last disagreement + 1.
///
/// **The clamp at the end is the whole point of this function.** The probe is
/// authoritative about what has already been consumed, so the calibration may
/// only push the boundary LATER, never earlier. Returning `probe - 6` when the
/// sweep finds nothing was wrong for a specific measured reason: single-tick
/// perturbations a few ticks before the finish often do not move the
/// interpolated millisecond at all, so the sweep sees no disagreement,
/// concludes "all safe", and hands back a boundary six ticks too early. A
/// multi-tick candidate then changes the real answer while the fork cannot, and
/// the fork returns a plausible time 1–2 ms off. Measured on map 2 at a 99.4%
/// checkpoint: **23 of 100 candidates silently wrong**, with the oracle itself
/// proven repeatable. The original 4700/4700 evidence never exercised this
/// path.
pub fn calibrate_boundary(
    srv: &mut forkoracle::forksrv::ForkServer,
    tape: &Tape,
    engine: &Engine,
    probe: usize,
) -> Result<usize, String> {
    let n = tape.n();
    let lo = probe.saturating_sub(6);
    let hi = (probe + 10).min(n - 1);
    let mut rows: Vec<(usize, PathBuf, Option<i64>)> = Vec::new();
    for t in lo..=hi {
        for kind in 0..3u8 {
            let (st, ac, br) = perturb_one(tape, t, kind);
            let path = engine.work.join(format!("cal{}_{:04}.Ghost.Gbx", kind, t));
            tape.write_candidate(&st, &ac, &br, &path)?;
            let recs = records_from(&st, &ac, &br, t);
            let out = srv.run(t, &recs);
            rows.push((t, path, parse_result(&out).0));
        }
    }
    let full = full_times(engine, &rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>(), "cal")?;
    let mut last_bad = None;
    for (t, path, forked) in &rows {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if *forked != full.get(&name).cloned().unwrap_or(None) {
            last_bad = Some(*t);
        }
    }
    Ok(match last_bad {
        Some(t) => (t + 1).max(probe),
        None => probe,
    })
}

fn perturb_one(tape: &Tape, t: usize, kind: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut s, mut a, mut b) = (tape.steer.clone(), tape.accel.clone(), tape.brake.clone());
    match kind {
        0 => s[t] = ((((tape.steer[t] as i8) as i32) + 90).clamp(-127, 127) as i8) as u8,
        1 => a[t] = 1 - a[t].min(1),
        _ => b[t] = 1 - b[t].min(1),
    }
    (s, a, b)
}

fn records_from(
    steer: &[u8],
    accel: &[u8],
    brake: &[u8],
    from: usize,
) -> Vec<forkoracle::forksrv::Rec> {
    (from..steer.len())
        .map(|t| forkoracle::forksrv::rec_of(steer[t], accel[t], brake[t]))
        .collect()
}

fn full_times(
    engine: &Engine,
    files: &[PathBuf],
    tag: &str,
) -> Result<HashMap<String, Option<i64>>, String> {
    let refs: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
    Ok(validate_batch(&engine.server, &engine.map, &refs, tag)?
        .into_iter()
        .map(|r| (r.file, r.time_ms))
        .collect())
}

// ---------------------------------------------------------------- fk server probe

/// Start a server, say where it stopped, and quit.
///
/// The number that matters is `probe`, and the reason it is worth a command of
/// its own is that **it is not the same on two servers started the same way.**
/// `lroundf` is bit-identical only on an idle box; under load the count moves in
/// whole chunks of ~62 calls (a wall-clock catch-up branch), and 62 calls is
/// about a quarter of a tick. Measured: 5 of 400 quiet starts stopped one tick
/// later than the rest, and **104 of 150 workers did when 150 servers start at
/// once**. Anything derived from where a server stopped is per-server.
pub fn probe(engine: &Engine, tape: Tape, at: Checkpoint) -> Result<(), String> {
    let t0 = Instant::now();
    let mut s = Session::start(engine, tape, at)?;
    let up = t0.elapsed().as_secs_f64();
    let p = s.probe_tick();
    let race = p.as_ref().ok().map(|p| s.tape.race_ms(*p));
    println!(
        "checkpoint lroundf #{}  input array {:#x}  probe tick {}  race {}  up {:.2}s",
        s.checkpoint_clock,
        s.srv.base,
        match &p {
            Ok(v) => v.to_string(),
            Err(e) => format!("FAILED ({})", e),
        },
        match race {
            Some(ms) => crate::secs(ms),
            None => "-".into(),
        },
        up
    );
    s.srv.quit();
    p.map(|_| ())
}

// ---------------------------------------------------------------- fk server check

pub struct CheckOpts {
    pub n: usize,
    pub seed: u64,
    pub span: usize,
}

/// THE CONTROL: run the same candidates through the fork and through a full
/// from-tick-0 validation, and require identical answers.
///
/// Five things happen here and each one can fail the run:
///
/// 1. **identity resume** — resuming with the reference's own inputs must give
///    the reference's own time. A failure here means the resume path is broken
///    before any candidate is involved.
/// 2. **page-fault probe** — hard abort if it fails. There is no fallback: a
///    resume that rewrites an already-consumed record is a silent no-op that
///    scores exactly the incumbent's score, so `delta == 0` is accepted and
///    that lineage is contaminated for free.
/// 3. **boundary calibration** — see [`calibrate_boundary`].
/// 4. **oracle repeatability** — the same candidate set validated twice. If the
///    oracle disagrees with itself, a fork/full mismatch is not evidence about
///    the fork. This is the positive control for the comparison itself.
/// 5. **exactness** — fork vs full, per candidate, on time AND on checkpoint
///    count for the DNFs.
///
/// **What passing this does NOT prove.** It measures the regime it samples:
/// perturbations of a reference tape at or after the boundary. It says nothing
/// about a tape that differs from its template early or wholesale, where the
/// fork was measured to lie on 312 of 312 reported finishes. Run
/// `--span 60 --at tick:<boundary>` style stress windows too: the production
/// window gave 0 phantoms in 289 banked tapes, which is exactly why the defect
/// survived four investigations. **A quiet run is not a clean one.**
pub fn check(engine: &Engine, tape: Tape, at: Checkpoint, o: CheckOpts) -> Result<bool, String> {
    let refp = engine.work.join("reference.Ghost.Gbx");
    let n = tape.n();
    engine.check()?;
    std::fs::create_dir_all(&engine.work).map_err(|e| e.to_string())?;
    tape.write_reference(&refp)?;
    let ref_time = full_times(engine, &[refp.clone()], "ref")?
        .values()
        .next()
        .cloned()
        .flatten();
    println!(
        "reference: {} ticks, oracle says {}, file declares {}",
        n,
        crate::secs_opt(ref_time),
        crate::secs_opt(tape.declared_ms.map(|v| v as i64))
    );

    let mut s = Session::start(engine, tape, at)?;
    println!(
        "fork server up: input array {:#x}, checkpoint at lroundf #{}",
        s.srv.base, s.checkpoint_clock
    );

    // 1. identity resume
    let recs = records_from(&s.tape.steer, &s.tape.accel, &s.tape.brake, 0);
    let t0 = Instant::now();
    let out = s.srv.run(0, &recs);
    let one = t0.elapsed().as_secs_f64();
    let (t, _) = parse_result(&out);
    let identity_ok = t == ref_time;
    println!(
        "identity resume -> {} (reference {}) {}   [{:.1} ms]",
        crate::secs_opt(t),
        crate::secs_opt(ref_time),
        if identity_ok { "EXACT" } else { "WRONG" },
        one * 1000.0
    );
    if !identity_ok {
        s.srv.quit();
        return Err("the identity resume does not reproduce the reference; nothing measured \
                    after this point would mean anything"
            .into());
    }

    // 2 + 3. boundary
    let probe = s.probe_tick().map_err(|e| {
        format!("boundary probe failed ({e}); a resume cannot be trusted without it")
    })?;
    let from = calibrate_boundary(&mut s.srv, &s.tape, engine, probe)?;
    println!(
        "boundary tick {} (probe {}) -- race {}, {:.1}% of the run",
        from,
        probe,
        crate::secs(s.tape.race_ms(from)),
        100.0 * s.tape.race_ms(from) as f64 / ref_time.unwrap_or(1) as f64
    );

    // build the candidate set
    let mut rng = Rng::new(o.seed);
    let mut cands = Vec::new();
    for i in 0..o.n {
        let (st, ac, br) = make_candidate(&s.tape, from, o.span, &mut rng);
        let p = engine.work.join(format!("c{:04}.Ghost.Gbx", i));
        s.tape.write_candidate(&st, &ac, &br, &p)?;
        cands.push((p, st, ac, br));
    }

    // HOW FAR IS THIS SET FROM THE REFERENCE? Without this line the exactness
    // number below is a number about an unnamed regime. The fork was exact on
    // 4700 of 4700 candidates that perturbed a human reference by a few ticks
    // LATE in the run, and reported 312 finishes out of 312 that were not there
    // on tapes differing early or wholesale. Nothing inside a fork can see
    // which of those it is running.
    {
        let reference =
            forkoracle::inputs::Inputs::from_arrays(&s.tape.steer, &s.tape.accel, &s.tape.brake);
        let ds: Vec<forkoracle::inputs::Distance> = cands
            .iter()
            .map(|(_, st, ac, br)| {
                forkoracle::inputs::Inputs::from_arrays(st, ac, br).distance_from(&reference)
            })
            .collect();
        let mut diffs: Vec<usize> = ds.iter().map(|d| d.diff_ticks).collect();
        diffs.sort_unstable();
        let earliest = ds.iter().filter_map(|d| d.first_diff_tick).min();
        println!(
            "candidate distance from the reference: earliest divergence {}, \
             median {} of {} ticks differ, worst {}",
            match earliest {
                Some(t) => format!("tick {} (race {})", t, crate::secs(s.tape.race_ms(t))),
                None => "none -- every candidate is the reference".into(),
            },
            diffs.get(diffs.len() / 2).copied().unwrap_or(0),
            n,
            diffs.last().copied().unwrap_or(0),
        );
    }
    let files: Vec<PathBuf> = cands.iter().map(|x| x.0.clone()).collect();

    // 4. the full run, twice
    let tg = Instant::now();
    let full = full_times(engine, &files, "full")?;
    let full_secs = tg.elapsed().as_secs_f64();
    let again = full_times(engine, &files, "again")?;
    let unstable = full
        .iter()
        .filter(|(k, v)| again.get(*k).cloned().unwrap_or(None) != **v)
        .count();
    println!(
        "oracle repeatability: {} of {} candidates differ between two full runs",
        unstable, o.n
    );

    // 5. exactness
    let tf = Instant::now();
    let (mut ok, mut bad, mut finished) = (0usize, 0usize, 0usize);
    for (p, st, ac, br) in &cands {
        let out = s.srv.run(from, &records_from(st, ac, br, from));
        let (t, _cps) = parse_result(&out);
        if t.is_some() {
            finished += 1;
        }
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let g = full.get(&name).cloned().unwrap_or(None);
        if t == g {
            ok += 1;
        } else {
            bad += 1;
            if bad <= 3 {
                println!(
                    "MISMATCH {}: fork {}  full {}",
                    name,
                    crate::secs_opt(t),
                    crate::secs_opt(g)
                );
            }
        }
    }
    let fork_secs = tf.elapsed().as_secs_f64();
    println!(
        "exactness: {}/{} identical ({} finished, {} DNF), {} MISMATCHES",
        ok,
        o.n,
        finished,
        o.n - finished,
        bad
    );
    println!(
        "full {:.3}s = {:.1} ms/cand | fork {:.3}s = {:.1} ms/cand | speedup {:.2}x",
        full_secs,
        1000.0 * full_secs / o.n as f64,
        fork_secs,
        1000.0 * fork_secs / o.n as f64,
        full_secs / fork_secs
    );
    s.srv.quit();
    Ok(bad == 0 && unstable == 0)
}

// ---------------------------------------------------------------- fk server bench

/// Throughput only: no correctness claim, and it says so.
///
/// Separate from `check` because the two get quoted separately and a speedup
/// measured on a run that did not also prove exactness is a number about
/// nothing.
pub fn bench(engine: &Engine, tape: Tape, at: Checkpoint, n: usize, seed: u64) -> Result<(), String> {
    let mut s = Session::start(engine, tape, at)?;
    let probe = s.probe_tick()?;
    let mut rng = Rng::new(seed);
    let mut cands = Vec::new();
    for i in 0..n {
        let (st, ac, br) = make_candidate(&s.tape, probe + 1, 60, &mut rng);
        let p = engine.work.join(format!("b{:04}.Ghost.Gbx", i));
        s.tape.write_candidate(&st, &ac, &br, &p)?;
        cands.push((p, st, ac, br));
    }
    let files: Vec<PathBuf> = cands.iter().map(|x| x.0.clone()).collect();
    let tg = Instant::now();
    full_times(engine, &files, "bench")?;
    let full_secs = tg.elapsed().as_secs_f64();
    let tf = Instant::now();
    for (_, st, ac, br) in &cands {
        s.srv.run(probe + 1, &records_from(st, ac, br, probe + 1));
    }
    let fork_secs = tf.elapsed().as_secs_f64();
    println!(
        "full {:.1} ms/cand | fork {:.1} ms/cand | speedup {:.2}x  \
         (throughput only -- run `fk server check` for the exactness claim)",
        1000.0 * full_secs / n as f64,
        1000.0 * fork_secs / n as f64,
        full_secs / fork_secs
    );
    s.srv.quit();
    Ok(())
}
