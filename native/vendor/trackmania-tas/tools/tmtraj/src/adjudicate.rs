//! `tmtraj adjudicate` — does a file's own record match what the engine does
//! with that file's own tape?
//!
//! This settles the `corpus dup` verdict `UNRESOLVED-INERT-OR-SPLICE`. Two
//! published files hold bit-identical positions across a stretch where their
//! input tapes disagree, and there are two explanations with opposite meanings:
//!
//!   (a) the differing inputs had NO AUTHORITY there — countdown, wedge,
//!       ballistic flight, a `SpecialNoSteering` road. Innocent.
//!   (b) one file is CARRYING THE OTHER'S RECORDING for that stretch. A defect.
//!
//! ## The test, and why it does not depend on the locate being perfect
//!
//! Re-simulate **both** tapes through the real engine (`fk trace`) and ask
//! whether the two SIMULATIONS differ over the window where the two RECORDS
//! agree.
//!
//! * simulations differ, records identical  ⇒ **at most one record is right**,
//!   so at least one file carries a trajectory that is not its own tape's. A
//!   defect, and it does not matter which copy of the car the locate picked,
//!   because both traces came from the same instrument on the same map.
//! * simulations agree too ⇒ the inputs really were inert there. Innocent, and
//!   now MEASURED rather than assumed.
//!
//! That framing is deliberate. An absolute comparison (trace vs record) is at
//! the mercy of which copy of the car `fk` located — and ~0.5 mm of
//! disagreement is exactly what a wrong copy looks like, which is the defect
//! this project spent 2026-08-22 discovering. A **differential** comparison
//! between two traces from the same instrument cancels that: a wrong copy is
//! wrong the same way in both.
//!
//! ## ONE TRACE IS NOT ENOUGH, and this cost a false accusation
//!
//! A single trace disagreeing with its own file's record is a statement about
//! **that trace**, not about the file. Measured here on 2026-08-22:
//!
//! ```text
//! 173636 TAS_22072, fork tick 400   worst 0.3002 m, median 0.2415 m   "does NOT match"
//! 173636 TAS_22072, fork tick 700   worst 0.3002 m, median 0.2520 m   "does NOT match"
//! 173636 TAS_22072, fork tick 1000  worst 0.0008 m, median 0.0005 m   MATCHES
//! ```
//!
//! Same file, same map, same binary — three fork points, and two of them found
//! an object that is not the car. **Two agreeing wrong answers**, which is the
//! reproduction-count trap this project has hit before: a majority must never
//! outrank a test that can identify the answer.
//!
//! So the acceptance rule is: **sweep fork ticks and take the BEST agreement**,
//! and only call a record foreign when *no* fork point reproduces it. A file
//! that matches at any fork tick has been shown to be its own tape's run; a
//! file that matches at none is either foreign or a map the locate cannot do,
//! and those two are separated by whether the *other* files on the same map
//! locate cleanly.

use crate::fmt::secs;
use gbx::record;

struct Trace {
    /// race ms -> (x, y, z), one row per 10 ms tick
    rows: Vec<(i64, f64, f64, f64)>,
}

fn read_trace(path: &str) -> Result<Trace, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut lines = txt.lines();
    let header = lines.next().ok_or_else(|| format!("{path}: empty"))?;
    let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    let find = |want: &str| -> Result<usize, String> {
        cols.iter()
            .position(|c| *c == want)
            .ok_or_else(|| format!("{path}: no column {want:?} in {cols:?}"))
    };
    // `fk trace` writes race time in ms plus the position triple.
    let (ct, cx, cy, cz) = (
        find("race_ms").or_else(|_| find("t_ms")).or_else(|_| find("time_ms"))?,
        find("x")?,
        find("y")?,
        find("z")?,
    );
    let mut rows = Vec::new();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split(',').collect();
        let g = |i: usize| -> Option<f64> { f.get(i)?.trim().parse().ok() };
        if let (Some(t), Some(x), Some(y), Some(z)) = (g(ct), g(cx), g(cy), g(cz)) {
            rows.push((t as i64, x, y, z));
        }
    }
    if rows.is_empty() {
        return Err(format!("{path}: no data rows"));
    }
    Ok(Trace { rows })
}

impl Trace {
    /// The traced position at a record sample's race time. Exact tick only —
    /// no interpolation, because an interpolated position is a number the
    /// engine never computed and this is an identity test.
    fn at(&self, ms: i64) -> Option<(f64, f64, f64)> {
        self.rows
            .binary_search_by_key(&ms, |r| r.0)
            .ok()
            .map(|i| (self.rows[i].1, self.rows[i].2, self.rows[i].3))
    }
}

fn dist(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
}

pub fn cmd(argv: &[String]) -> i32 {
    let pos: Vec<&String> = argv.iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() < 4 {
        eprintln!(
            "usage: tmtraj adjudicate A.Ghost.Gbx A.trace.csv B.Ghost.Gbx B.trace.csv [--to SECONDS]"
        );
        eprintln!();
        eprintln!("  Settles a corpus-dup UNRESOLVED pair. Traces come from");
        eprintln!("  `fk trace --tape FILE --map M --at tick:N --out CSV`.");
        return 2;
    }
    let to_ms = argv
        .iter()
        .position(|s| s == "--to")
        .and_then(|i| argv.get(i + 1))
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| (v * 1000.0) as i64)
        .unwrap_or(i64::MAX);

    let (fa, ta, fb, tb) = (pos[0], pos[1], pos[2], pos[3]);
    let (ra, rb) = match (record::decode_ghost(fa), record::decode_ghost(fb)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("tmtraj adjudicate: {e}");
            return 2;
        }
    };
    let (sa, sb) = match (read_trace(ta), read_trace(tb)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("tmtraj adjudicate: {e}");
            return 2;
        }
    };

    // Walk the record samples the two files share.
    let mut n = 0usize;
    let mut rec_ident = 0usize;
    let mut sim_ident = 0usize;
    let mut worst_sim = 0.0f64;
    let mut worst_a = 0.0f64;
    let mut worst_b = 0.0f64;
    let mut cmp_a = 0usize;
    let mut cmp_b = 0usize;
    let mut da_all: Vec<f64> = Vec::new();
    let mut db_all: Vec<f64> = Vec::new();
    // The instant the records stop agreeing and the simulations disagree by a
    // margin no wrong-copy offset explains.
    let mut first_split: Option<i64> = None;
    // THE DECISIVE STATISTIC, and the summary lines above are not a substitute
    // for it. Restrict to the samples where the two RECORDS are bit-identical,
    // and ask how far apart the ENGINE puts the two cars there.
    //
    //   ~0        the records agree because the inputs were inert. Innocent.
    //   metres    the records agree while the runs did not. A defect.
    //
    // Whole-file rates cannot answer this: on 210218 the records agree on
    // 93.8 % of samples and the simulations on 94.1 %, and reading those two
    // numbers side by side tells you nothing about whether they are the SAME
    // samples. They are — but that had to be measured, not inferred.
    let mut worst_sim_where_rec_same = 0.0f64;
    let mut n_rec_same_covered = 0usize;
    let mut sim_same_where_rec_same = 0usize;

    let m = ra.samples.len().min(rb.samples.len());
    for i in 0..m {
        let (pa, pb) = (&ra.samples[i], &rb.samples[i]);
        let t = pa.time_ms as i64;
        if t > to_ms {
            break;
        }
        let (qa, qb) = ((pa.x as f64, pa.y as f64, pa.z as f64), (pb.x as f64, pb.y as f64, pb.z as f64));
        n += 1;
        let records_same = qa == qb;
        if records_same {
            rec_ident += 1;
        }
        let (Some(va), Some(vb)) = (sa.at(t), sb.at(t)) else {
            continue;
        };
        let dsim = dist(va, vb);
        worst_sim = worst_sim.max(dsim);
        if dsim == 0.0 {
            sim_ident += 1;
        }
        let da = dist(va, qa);
        let db = dist(vb, qb);
        worst_a = worst_a.max(da);
        worst_b = worst_b.max(db);
        cmp_a += 1;
        cmp_b += 1;
        da_all.push(da);
        db_all.push(db);
        if records_same {
            n_rec_same_covered += 1;
            worst_sim_where_rec_same = worst_sim_where_rec_same.max(dsim);
            if dsim == 0.0 {
                sim_same_where_rec_same += 1;
            }
        }
        // A metre of simulated separation cannot be a copy-choice artefact:
        // the wrong-copy offset measured on this project is ~0.5 mm.
        if records_same && dsim > 1.0 && first_split.is_none() {
            first_split = Some(t);
        }
    }

    if cmp_a == 0 {
        println!("UNTESTED  no record sample fell on a traced tick -- coverage is zero, which is");
        println!("          not a clean result. Check the traces cover the window.");
        return 2;
    }

    println!("compared {n} record samples ({cmp_a} of them covered by both traces)");
    println!(
        "  records identical      {rec_ident} of {n}  ({:.1} %)",
        100.0 * rec_ident as f64 / n.max(1) as f64
    );
    println!(
        "  SIMULATIONS identical  {sim_ident} of {cmp_a}  worst separation {worst_sim:.4} m"
    );
    println!("  trace A vs A's own record   worst {worst_a:.4} m   median {:.4} m", median(&mut da_all));
    println!("  trace B vs B's own record   worst {worst_b:.4} m   median {:.4} m", median(&mut db_all));
    println!();
    println!("  THE DECISIVE ONE -- over the {n_rec_same_covered} samples where the two RECORDS are");
    println!("  bit-identical, the engine puts the two cars at most {worst_sim_where_rec_same:.4} m apart");
    println!(
        "  ({sim_same_where_rec_same} of those {n_rec_same_covered} are bit-identical in the simulation too, {:.1} %)",
        100.0 * sim_same_where_rec_same as f64 / n_rec_same_covered.max(1) as f64
    );
    println!();

    // PER-FILE, which is the question a reader actually has: does THIS file's
    // record match what the engine does with THIS file's tape? That is the
    // C-route test, and it is what condemns or clears a single file rather
    // than a pair.
    //
    // The bar is 0.01 m. It is 20x the ~0.5 mm wrong-car-copy offset, and it
    // is far below the metres a genuinely different run produces. A trace that
    // sits above it is EITHER a record that is not its own tape's run OR a
    // locate that landed on the wrong object -- this test cannot tell those
    // apart on its own, and says so rather than picking.
    const OWN: f64 = OWN_BAR;
    let verdict = |name: &str, worst: f64, med: f64| {
        if worst <= OWN {
            println!("  {name}: record IS its own tape's run (worst {worst:.4} m, median {med:.4} m)");
            true
        } else {
            println!("  {name}: record does NOT match the engine's run of its own tape");
            println!("     worst {worst:.4} m, median {med:.4} m -- either the record is not this");
            println!("     tape's, or this trace's locate found the wrong object. Not separable here.");
            false
        }
    };
    println!("PER FILE:");
    let ok_a = verdict(&base(fa), worst_a, median(&mut da_all.clone()));
    let ok_b = verdict(&base(fb), worst_b, median(&mut db_all.clone()));
    println!();

    // THE VERDICT.
    if let Some(t) = first_split {
        println!("DEFECT -- the two files' records are identical where the engine says the two");
        println!("  tapes are not. First such instant: race {}.", secs(t));
        println!("  The engine, run on each file's own tape, separates them by up to");
        println!("  {worst_sim:.4} m over a stretch their records agree on bit for bit.");
        println!("  So at least one of these records is not its own tape's run.");
        println!("  Which one: whichever trace does NOT match its own record above.");
        return 1;
    }
    if worst_sim_where_rec_same == 0.0 {
        println!("INNOCENT -- wherever the two records agree bit for bit, the engine agrees bit");
        println!("  for bit too, running each file's OWN tape. The identical stretch is what");
        println!("  physics does here: those differing inputs had no authority. MEASURED.");
        return 0;
    }
    println!("INCONCLUSIVE -- where the records agree the simulations differ by up to");
    println!("  {worst_sim_where_rec_same:.4} m, which is under the 1 m bar. That bar exists because a wrong");
    println!("  car-copy pick offsets a whole trace by ~0.5 mm and must not read as a");
    println!("  divergence. Confirm the locate (carscan path + spawn) to tighten it.");
    2
}

/// The median of a sample, by sorting a copy. Reported beside every worst-case
/// because a single bad sample and a systematically wrong trace look identical
/// in a maximum, and only one of them is a defect.
fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// The file's base name, for the per-file verdict lines.
fn base(p: &str) -> String {
    p.rsplit('/').next().unwrap_or(p).to_string()
}

// ===========================================================================
// The reusable core, so `adjudicate-batch` runs the same comparison as
// `adjudicate` rather than a second copy of it. Every bug this project's
// `ghost` crate exists to prevent was a second copy of a reader disagreeing
// with the first.
// ===========================================================================

#[derive(Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Where the records agree, the engine agrees too. The differing inputs
    /// had no authority there.
    Innocent,
    /// The records agree where the engine says the runs were metres apart.
    Defect,
    /// Between the two: above bit-identity, below the 1 m bar.
    Inconclusive,
    /// No usable trace for one of the files.
    /// No usable trace, or no covered sample. A statement about the TEST.
    NoTrace,
}

impl Verdict {
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Innocent => "INNOCENT-INERT-INPUTS",
            Verdict::Defect => "DEFECT-SHARED-RECORD",
            Verdict::Inconclusive => "INCONCLUSIVE",
            Verdict::NoTrace => "UNTESTED-NO-COVERAGE",
        }
    }
}

/// How well a trace reproduces a ghost's own record: (worst, median) metres,
/// **scanned over integer tick shifts and reported at the best one**.
///
/// THE SHIFT SCAN IS NOT OPTIONAL. A regenerated record can sit a whole physics
/// tick from the engine's own clock — a known, documented property of this
/// pipeline — and a one-tick offset is a PURE TIME SHIFT, so it shows up as a
/// distance that scales with speed and looks exactly like a wrong trajectory.
/// Measured here: 210218's two files read **1.54 m worst / 0.56 m median** at
/// shift 0 and **0.005 m** at the best shift. Judging them at shift 0 would
/// have called two sound files foreign.
///
/// This is the corpus's own rule, learned on the position decoys: *test for a
/// TIME SHIFT, not a distance.*
pub fn trace_vs_record(ghost: &str, trace: &str) -> Option<(f64, f64)> {
    trace_vs_record_shift(ghost, trace).map(|(w, m, _)| (w, m))
}

/// As `trace_vs_record`, and also the SHIFT at which it was achieved. The
/// caller needs that shift: comparing two traces that were each accepted at a
/// different shift, without applying it, reintroduces exactly the one-tick
/// error the scan removed. Measured: on 126859 that produced FOUR identical
/// "defects" at 2.140111 m -- one tick at 222 m/s -- on five files every one of
/// which reproduces its own record to 0.007 m.
pub fn trace_vs_record_shift(ghost: &str, trace: &str) -> Option<(f64, f64, i64)> {
    let r = record::decode_ghost(ghost).ok()?;
    let s = read_trace(trace).ok()?;
    let mut best: Option<(f64, f64, i64)> = None;
    for shift in -3i64..=3 {
        let mut d: Vec<f64> = Vec::new();
        for p in &r.samples {
            if let Some(v) = s.at(p.time_ms as i64 + shift * 10) {
                d.push(dist(v, (p.x as f64, p.y as f64, p.z as f64)));
            }
        }
        if d.is_empty() {
            continue;
        }
        let worst = d.iter().cloned().fold(0.0f64, f64::max);
        let med = median(&mut d);
        if best.as_ref().is_none_or(|(bw, _, _)| worst < *bw) {
            best = Some((worst, med, shift));
        }
    }
    best
}

/// Of every swept trace for this file, the one that reproduces the file's own
/// record best. Returns (path, worst, median).
///
/// THE ACCEPTANCE RULE, and it is not "take the majority": two of 173636
/// TAS_22072's three fork points agreed with each other on a wrong object and
/// the third found the car. A test that can identify the answer beats a count.
pub fn best_trace(dir: &str, mapid: &str, stem: &str, ghost: &str) -> Option<(String, f64, f64, i64)> {
    let rd = std::fs::read_dir(dir).ok()?;
    let prefix = format!("{mapid}__{stem}__t");
    let mut best: Option<(String, f64, f64, i64)> = None;
    for e in rd.flatten() {
        let p = e.path().to_string_lossy().to_string();
        let n = p.rsplit('/').next().unwrap_or(&p).to_string();
        if !n.starts_with(&prefix) || !n.ends_with(".csv") {
            continue;
        }
        if let Some((w, m, sh)) = trace_vs_record_shift(ghost, &p) {
            // AMONG TRACES THAT HAVE FOUND THE CAR, MORE COVERAGE WINS.
            // A trace agreeing with the file's own record to <= 1 cm has
            // found it; agreeing to 0.0005 m instead of 0.005 m is not a
            // better answer to that question, and picking on it silently
            // chose late fork points that did not reach the window the pair
            // test needs -- 16 pairs read UNTESTED for that reason alone.
            let start = first_ms(&p).unwrap_or(i64::MAX);
            let better = match &best {
                None => true,
                Some((bp, bw, _, _)) => {
                    let bstart = first_ms(bp).unwrap_or(i64::MAX);
                    match (w <= OWN_BAR, *bw <= OWN_BAR) {
                        (true, true) => start < bstart,
                        (true, false) => true,
                        (false, true) => false,
                        (false, false) => w < *bw,
                    }
                }
            };
            if better {
                best = Some((p, w, m, sh));
            }
        }
    }
    best
}

/// The pair question: over the samples where the two RECORDS are bit-identical,
/// how far apart does the engine put the two cars? Returns (verdict, worst
/// separation on those samples, how many there were).
/// Each trace is read at ITS OWN accepted shift -- see `trace_vs_record_shift`.
pub fn adjudicate_pair(
    fa: &str,
    ta: &str,
    sha: i64,
    fb: &str,
    tb: &str,
    shb: i64,
) -> Option<(Verdict, f64, usize)> {
    let ra = record::decode_ghost(fa).ok()?;
    let rb = record::decode_ghost(fb).ok()?;
    let sa = read_trace(ta).ok()?;
    let sb = read_trace(tb).ok()?;
    let mut worst = 0.0f64;
    let mut n = 0usize;
    for i in 0..ra.samples.len().min(rb.samples.len()) {
        let (pa, pb) = (&ra.samples[i], &rb.samples[i]);
        if (pa.x, pa.y, pa.z) != (pb.x, pb.y, pb.z) {
            continue;
        }
        let t = pa.time_ms as i64;
        let (Some(va), Some(vb)) = (sa.at(t + sha * 10), sb.at(t + shb * 10)) else { continue };
        worst = worst.max(dist(va, vb));
        n += 1;
    }
    if n == 0 {
        // No sample where the records agree fell inside BOTH traces. That is a
        // coverage failure of this test, not a statement about the files.
        return None;
    }
    let v = if worst > 1.0 {
        Verdict::Defect
    } else if worst == 0.0 {
        Verdict::Innocent
    } else {
        Verdict::Inconclusive
    };
    Some((v, worst, n))
}

/// A trace agreeing with its file's own record to within this is accepted as
/// having found the car. 20x the ~0.5 mm wrong-copy offset, and orders of
/// magnitude below a genuinely different run.
pub const OWN_BAR: f64 = 0.01;

/// The first race instant a trace covers, for preferring coverage among traces
/// that have all found the car.
fn first_ms(path: &str) -> Option<i64> {
    let t = std::fs::read_to_string(path).ok()?;
    let mut l = t.lines();
    l.next()?;
    l.next()?.split(',').next()?.trim().parse().ok()
}
