//! `tmtraj diff` — are these two recordings the same run?
//!
//! One command for what used to be five: `sep`, `seplag`, `nearident`,
//! `recdiff` and `intg pair`. They asked overlapping questions with different
//! alignment rules, different tolerances and different failure behaviour, and
//! the differences were where the bugs lived.
//!
//! ## The one invariant
//!
//! **An empty denominator is not a measurement.** Every mode here reports how
//! many samples it actually compared, and refuses (exit 3, `UNMEASURED`) rather
//! than returning a verdict from nothing.
//!
//! This is not a hypothetical failure. `sep` used to walk two files index by
//! index and `break` on the first time-key mismatch, printing a note to stderr
//! that every pipeline discarded. Sample times are SESSION times, so two
//! recordings made in different sessions disagree at index 0: all ten of
//! 228607's published files were compared against `AUTHOR_LAP_20258`, produced
//! ZERO rows each, and the audit read ten silences as ten CLEAN verdicts.
//! `nearident` inherited the same shape and printed `VERDICT INDEPENDENT` with
//! an overlap of 0 and a mean of `f64::MAX`.
//!
//! ## What each mode proves, and what it does not
//!
//! * **`--at-shared-instants`** (default) — per-instant separation on the
//!   instants the two files share, plus which run is further along its own
//!   path. This is the number the two-car render test needs: a second car is
//!   only demonstrably DRAWN if there is a frame where the runs are far enough
//!   apart to be two cars and close enough that both are in one chase camera.
//!   "Not drawn" and "out of frame" are indistinguishable without it.
//!
//! * **`--lag`** — the same comparison with the time labels IGNORED, scanned
//!   over every integer sample offset. Use it when the grids do not line up.
//!   A donor graft shows as a long run of exactly-zero distances at some lag.
//!
//! * **`--near`** — the same scan at a millimetre band instead of exact
//!   equality, because a copy that has been through a float re-encode is never
//!   bit-identical. **It requires `--control`** and refuses without one: on
//!   199100 the exact test returned "INDEPENDENT — no identical position at any
//!   lag" for a pair that is one run (mean 0.000476 m over 800 samples,
//!   byte-identical input tapes), and a fixed 1 mm band then cried COPY on four
//!   clips that were fine, because 1 mm is TWICE our own writer's noise floor
//!   (0.482 / 0.483 / 0.489 / 0.518 mm against the game's own recording on four
//!   maps). The verdict is a RATIO against a pair known to be two different
//!   runs on this map — dimensionless, and untunable. Human against human is
//!   the pairing that works: neither can have come out of our pipeline.
//!
//! * **`--bytes`** — per-byte agreement of the raw 116-byte samples. Two ghosts
//!   that encode different runs must not carry the same recorded motion; before
//!   this test existed, 17 of the repo's 29 multi-ghost maps did, which is why
//!   two of our own tapes once rendered as a single car.
//!
//! ## What a shared prefix does NOT prove
//!
//! Nothing. The simulation is deterministic, so two tapes with the same opening
//! inputs give identical f32 positions for as long as their inputs agree — our
//! own sibling tapes are 67 % bit-identical on one 203072 pair. The proof of a
//! splice is RE-CONVERGENCE: identical, then more than `--minsep` apart, then
//! exactly identical again. Driving cannot do that. `tmtraj gate` reads the
//! runs this command prints and applies that rule.

use crate::cli;
use crate::fmt::secs;
use gbx::record::{self, Decoded, Sample};
use std::collections::HashMap;

const USAGE: &str = "\
usage: tmtraj diff A.Ghost.Gbx B.Ghost.Gbx [mode] [flags]

modes (default: compare at the instants the two files share)
  --lag                 ignore the time labels; scan every integer sample lag
  --near                the lag scan at a millimetre band -- needs --control
  --bytes               per-byte agreement of the raw samples

flags
  --control X Y         two recordings of this map that are KNOWN to be two
                        different runs (repeatable). Required by --near.
  --mm N                the band for --near, in millimetres (default 1.0)
  --ratio N             how many times closer than the control the subject must
                        sit to be called a copy (default 10)
  --run N               minimum in-band run length for --near (default 100)
  --minsep M            separation that counts as 'apart' for re-convergence
                        (default 5.0 m)
  --rows                print the per-instant table (default: verdict only)
  --csv FILE            write the per-instant table to a file

exit 0 compared and independent, 2 a copy or a splice, 3 UNMEASURED
";

const SPAWN_USAGE: &str = "\
usage: tmtraj spawn GHOST... --ref HUMAN.Ghost.Gbx [--pos-tol M] [--ang-tol DEG]

Does this file start where every run on this map starts, FACING THE WAY THEY
ALL FACE? Free on every map: every run spawns identically, so a downloaded
human recording is the answer key.
";

fn dist(p: &Sample, q: &Sample) -> f64 {
    (((p.x - q.x) as f64).powi(2) + ((p.y - q.y) as f64).powi(2) + ((p.z - q.z) as f64).powi(2)).sqrt()
}

fn load(p: &str) -> Result<Decoded, String> {
    record::decode_ghost(p).map_err(|e| format!("{}: {}", p, e))
}

/// Cumulative path length, so "ahead" is along the track and not in a straight
/// line: a car behind the camera target is off-screen however close it is.
fn path_lengths(s: &[Sample]) -> Vec<f64> {
    let mut v = Vec::with_capacity(s.len());
    let mut acc = 0.0f64;
    for (i, p) in s.iter().enumerate() {
        if i > 0 {
            acc += dist(p, &s[i - 1]);
        }
        v.push(acc);
    }
    v
}

pub fn cmd(argv: &[String]) -> i32 {
    let a = cli::parse("tmtraj diff", argv, &["lag", "near", "bytes", "rows"]);
    let mm: f64 = a.num("mm", 1.0);
    let ratio_bar: f64 = a.num("ratio", 10.0);
    let min_run: usize = a.num("run", 100);
    let minsep: f64 = a.num("minsep", 5.0);
    let csv = a.one("csv").map(|s| s.to_string());
    let controls = a.many("control");
    let (lag, near, bytes) = (a.has("lag"), a.has("near"), a.has("bytes"));
    let rows = a.has("rows");
    let a = a.finish(USAGE);
    if a.positional.len() != 2 {
        eprint!("{}", USAGE);
        return 2;
    }
    let (pa, pb) = (&a.positional[0], &a.positional[1]);
    let (da, db) = match (load(pa), load(pb)) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("UNMEASURED: {}", e);
            return 3;
        }
    };
    eprintln!(
        "A {} samples, declared {}   B {} samples, declared {}",
        da.samples.len(),
        da.race_time_ms.map_or("-".into(), |v| secs(v as i64)),
        db.samples.len(),
        db.race_time_ms.map_or("-".into(), |v| secs(v as i64)),
    );
    if bytes {
        return bytes_mode(&da, &db);
    }
    if near {
        return near_mode(pa, pb, &da, &db, &controls, mm, ratio_bar, min_run);
    }
    if lag {
        return lag_mode(&da, &db);
    }
    shared_instants(pa, pb, &da, &db, minsep, csv.as_deref(), rows)
}

// ---------------------------------------------------------------------------
// default: the instants the two files share
// ---------------------------------------------------------------------------

fn shared_instants(
    pa: &str,
    pb: &str,
    da: &Decoded,
    db: &Decoded,
    minsep: f64,
    csv: Option<&str>,
    rows_wanted: bool,
) -> i32 {
    let la = path_lengths(&da.samples);
    let lb = path_lengths(&db.samples);
    let ib: HashMap<i32, usize> =
        db.samples.iter().enumerate().map(|(i, s)| (s.time_ms, i)).collect();

    let mut out = String::from("t_s\tdist_m\tdlen_m\tax\tay\taz\tbx\tby\tbz\n");
    let mut rows = 0usize;
    // The re-convergence state machine: identical -> apart -> identical again.
    let (mut ident, mut ever_apart, mut reconverged, mut max_sep) = (0usize, false, 0usize, 0.0f64);
    let mut was_ident = false;
    for (i, p) in da.samples.iter().enumerate() {
        let Some(&j) = ib.get(&p.time_ms) else { continue };
        let q = &db.samples[j];
        rows += 1;
        let d = dist(p, q);
        if d == 0.0 {
            ident += 1;
            if ever_apart && !was_ident {
                reconverged += 1;
            }
            was_ident = true;
        } else {
            was_ident = false;
            if d > minsep {
                ever_apart = true;
            }
            if d > max_sep {
                max_sep = d;
            }
        }
        out.push_str(&format!(
            "{}\t{:.6}\t{:+.6}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\n",
            secs(p.time_ms as i64),
            d,
            lb[j] - la[i], // + => B is further along its own path than A
            p.x,
            p.y,
            p.z,
            q.x,
            q.y,
            q.z
        ));
    }
    // The one rule, decided by `outcome` so that the unit test above tests the
    // shipped rule and not a second copy of it.
    let verdict = outcome(rows, ident, reconverged, 90.0);
    if verdict == Outcome::Unmeasured {
        eprintln!(
            "UNMEASURED: {} and {} share no sample instant, so nothing was compared.\n\
             Sample times are SESSION times; two recordings from different sessions have no\n\
             time key in common. This says nothing about either file and is NOT a clean\n\
             result. Use `tmtraj diff A B --lag`, which needs no alignment.",
            pa, pb
        );
        return verdict.exit_code();
    }
    match (csv, rows_wanted) {
        (Some(f), _) => {
            std::fs::write(f, &out).expect("write csv");
            eprintln!("wrote {}", f);
        }
        (None, true) => print!("{}", out),
        // The per-instant table is opt-in: the verdict and the coverage are
        // what a caller almost always wants, and a 455-row dump ahead of them
        // is how the coverage line got scrolled off and ignored.
        (None, false) => {}
    }
    println!(
        "compared {} shared instants: {} bit-identical ({:.1} %), worst separation {:.3} m",
        rows,
        ident,
        100.0 * ident as f64 / rows as f64,
        max_sep
    );
    match verdict {
        // Once two runs are more than minsep apart they are different physical
        // states, and no sequence of inputs returns them to EXACTLY 0.000000 m.
        Outcome::Splice => println!(
            "VERDICT SPLICE: {} re-convergence(s) to exactly 0 m after separating past {:.1} m",
            reconverged, minsep
        ),
        Outcome::IsTheReference => println!(
            "VERDICT IS-THE-REFERENCE: {} of {} shared samples bit-identical -- this file's \
             telemetry is that recording",
            ident, rows
        ),
        Outcome::NoSpliceFound => println!(
            "VERDICT no splice found against this reference \
             (a shared prefix is determinism, not evidence; only re-convergence is)"
        ),
        Outcome::Unmeasured => unreachable!("handled above"),
    }
    verdict.exit_code()
}

// ---------------------------------------------------------------------------
// --lag: alignment-free
// ---------------------------------------------------------------------------

struct LagScan {
    lag: i64,
    overlap: usize,
    zeros: usize,
    longest: usize,
    mean: f64,
}

fn scan_lags(sa: &[Sample], sb: &[Sample], tol: f64, span: i64) -> Option<LagScan> {
    let shorter = sa.len().min(sb.len()) as i64;
    let mut best: Option<LagScan> = None;
    for lag in -span..=span {
        let (mut n, mut sum, mut run, mut longest, mut zeros) = (0usize, 0.0f64, 0usize, 0usize, 0usize);
        for (i, p) in sa.iter().enumerate() {
            let j = i as i64 + lag;
            if j < 0 || j as usize >= sb.len() {
                continue;
            }
            let d = dist(p, &sb[j as usize]);
            if !d.is_finite() {
                continue;
            }
            n += 1;
            sum += d;
            if d <= tol {
                zeros += 1;
                run += 1;
                longest = longest.max(run);
            } else {
                run = 0;
            }
        }
        if n == 0 || n < (shorter / 4).max(1) as usize {
            continue;
        }
        let cand = LagScan { lag, overlap: n, zeros, longest, mean: sum / n as f64 };
        let better = match &best {
            None => true,
            Some(b) => cand.longest > b.longest || (cand.longest == b.longest && cand.mean < b.mean),
        };
        if better {
            best = Some(cand);
        }
    }
    best
}

fn lag_mode(da: &Decoded, db: &Decoded) -> i32 {
    let (sa, sb) = (&da.samples, &db.samples);
    if sa.is_empty() || sb.is_empty() {
        eprintln!("UNMEASURED: one of the files has no vehicle samples");
        return 3;
    }
    let span = (sa.len().max(sb.len()) as i64) - (sa.len().min(sb.len()) as i64) / 4;
    let Some(b) = scan_lags(sa, sb, 0.0, span) else {
        eprintln!("UNMEASURED: no lag in +-{} gives a usable overlap", span);
        return 3;
    };
    println!(
        "best lag {} samples  overlap {}  exact zeros {}  longest identical run {}  mean {:.6} m",
        b.lag, b.overlap, b.zeros, b.longest, b.mean
    );
    if b.longest >= 10 {
        println!(
            "VERDICT DONOR-GRAFT: {} consecutive identical positions at lag {}",
            b.longest, b.lag
        );
        return 2;
    }
    if b.zeros > 0 {
        println!(
            "VERDICT incidental: {} scattered identical positions, longest run {}",
            b.zeros, b.longest
        );
        return 0;
    }
    println!("VERDICT independent: no identical position at any lag");
    0
}

// ---------------------------------------------------------------------------
// --near: a copy after a float re-encode, judged against a control
// ---------------------------------------------------------------------------

fn near_mode(
    pa: &str,
    pb: &str,
    da: &Decoded,
    db: &Decoded,
    controls: &[String],
    mm: f64,
    ratio_bar: f64,
    min_run: usize,
) -> i32 {
    if controls.len() < 2 || controls.len() % 2 != 0 {
        eprintln!(
            "REFUSED: --near needs --control X Y (in pairs).\n\
             \n\
             It used to answer without one and the answer was wrong four times. Half a\n\
             millimetre means \"copy\" only if a pair known to be two DIFFERENT runs measures\n\
             much further apart ON THIS MAP: our own writer sits ~0.5 mm from the game's own\n\
             recording of the same run, which is inside any 1 mm band, so the band alone\n\
             flags every honest regeneration.\n\
             \n\
             Pass two recordings of this map that cannot be the same run. Human against human\n\
             is the pairing that works -- neither came out of our pipeline, so whatever they\n\
             measure is what \"independent\" looks like here."
        );
        return 3;
    }
    let tol = mm / 1000.0;
    let Some(subj) = scan_lags(&da.samples, &db.samples, tol, 750) else {
        eprintln!(
            "UNMEASURED: {} and {} share no comparable samples at any lag in +-750.\n\
             Nothing was compared -- this is NOT a clean result.",
            pa, pb
        );
        return 3;
    };
    println!(
        "band {:.3} mm, min run {}, ratio bar {}x",
        mm, min_run, ratio_bar
    );
    println!(
        "subject   lag {}  overlap {}  longest in band {}  mean {:.6} m",
        subj.lag, subj.overlap, subj.longest, subj.mean
    );
    let mut ctl_mean = f64::MAX;
    let mut ctl_run = usize::MAX;
    let mut n_ctl = 0;
    for pair in controls.chunks(2) {
        let (Ok(x), Ok(y)) = (load(&pair[0]), load(&pair[1])) else {
            eprintln!("control {} / {}: cannot decode -- skipped", pair[0], pair[1]);
            continue;
        };
        let Some(c) = scan_lags(&x.samples, &y.samples, tol, 750) else {
            eprintln!("control {} / {}: no comparable samples -- skipped", pair[0], pair[1]);
            continue;
        };
        println!(
            "control   lag {}  overlap {}  longest in band {}  mean {:.6} m   [{} / {}]",
            c.lag,
            c.overlap,
            c.longest,
            c.mean,
            pair[0].rsplit('/').next().unwrap_or(&pair[0]),
            pair[1].rsplit('/').next().unwrap_or(&pair[1]),
        );
        ctl_mean = ctl_mean.min(c.mean);
        ctl_run = ctl_run.min(c.longest);
        n_ctl += 1;
    }
    if n_ctl == 0 {
        eprintln!("REFUSED: every --control pair failed to measure. A verdict with no control is the thing that cost four clips.");
        return 3;
    }
    let ratio = if subj.mean > 0.0 { ctl_mean / subj.mean } else { f64::INFINITY };
    println!("closest control mean {:.6} m -> subject is {:.1}x closer", ctl_mean, ratio);
    if subj.longest >= min_run && ratio >= ratio_bar && subj.longest > ctl_run {
        println!(
            "VERDICT COPY: {} consecutive samples inside {:.3} mm, {:.1}x closer than a pair known to be two runs",
            subj.longest, mm, ratio
        );
        return 2;
    }
    println!("VERDICT independent against this control");
    0
}

// ---------------------------------------------------------------------------
// --bytes: do the two records carry the same motion, byte for byte
// ---------------------------------------------------------------------------

fn bytes_mode(da: &Decoded, db: &Decoded) -> i32 {
    let ra: Vec<&[u8]> = da.raw_samples().collect();
    let rb: Vec<&[u8]> = db.raw_samples().collect();
    let ss = da.sample_size.min(db.sample_size);
    let n = ra.len().min(rb.len());
    if n == 0 || ss == 0 {
        eprintln!("UNMEASURED: no raw samples to compare");
        return 3;
    }
    let mut same = vec![0usize; ss];
    let mut vary = vec![std::collections::BTreeSet::new(); ss];
    for i in 0..n {
        for b in 0..ss {
            if ra[i][b] == rb[i][b] {
                same[b] += 1;
            }
            vary[b].insert(ra[i][b]);
        }
    }
    println!("{:>5} {:>10} {:>10}", "byte", "identical", "distinct_A");
    for b in 0..ss {
        if same[b] != n || vary[b].len() > 1 {
            println!("{:>5} {:>9}/{} {:>10}", b, same[b], n, vary[b].len());
        }
    }
    let carrying: usize = vary.iter().filter(|v| v.len() > 1).count();
    let ident: usize = same.iter().sum();
    let total = n * ss;
    println!(
        "{} of {} bytes carry information in A; {} of {} sample bytes identical ({:.2} %)",
        carrying,
        ss,
        ident,
        total,
        100.0 * ident as f64 / total as f64
    );
    if ident == total {
        // Two ghosts that encode different runs must not carry the same
        // recorded motion. 17 of the repo's 29 multi-ghost maps once did.
        println!("VERDICT IDENTICAL-TELEMETRY: these two files carry ONE recorded run");
        return 2;
    }
    println!("VERDICT DIFFERENT-RUNS");
    0
}

// ---------------------------------------------------------------------------
// spawn
// ---------------------------------------------------------------------------

pub fn cmd_spawn(argv: &[String]) -> i32 {
    let a = cli::parse("tmtraj spawn", argv, &[]);
    let refp = a.one("ref").map(|s| s.to_string());
    let pos_tol: f64 = a.num("pos-tol", 2.0);
    let ang_tol: f64 = a.num("ang-tol", 0.99);
    let a = a.finish(SPAWN_USAGE);
    let (Some(refp), false) = (refp, a.positional.is_empty()) else {
        eprint!("{}", SPAWN_USAGE);
        return 2;
    };
    let r = match load(&refp) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("UNMEASURED: reference {}", e);
            return 3;
        }
    };
    let Some(r0) = r.samples.first() else {
        eprintln!("UNMEASURED: reference has no samples");
        return 3;
    };
    let mut worst = 0;
    for p in &a.positional {
        let d = match load(p) {
            Ok(d) => d,
            Err(e) => {
                println!("{:<44} DECODE-FAIL  {}", short(p), e);
                worst = worst.max(3);
                continue;
            }
        };
        let Some(s0) = d.samples.first() else {
            println!("{:<44} NO-SAMPLES", short(p));
            worst = worst.max(3);
            continue;
        };
        let dp = dist(s0, r0);
        let dot = quat_agreement([s0.qx as f64, s0.qy as f64, s0.qz as f64, s0.qw as f64], [r0.qx as f64, r0.qy as f64, r0.qz as f64, r0.qw as f64]);
        let verdict = spawn_verdict(dp, dot, pos_tol, ang_tol);
        if verdict != "SPAWN-OK" {
            worst = worst.max(2);
        }
        println!("{:<44} dpos {:>7.3} m  |dot| {:>6.4}  {}", short(p), dp, dot, verdict);
    }
    worst
}

fn short(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

/// What a comparison is allowed to conclude, given how much of it happened.
///
/// A free function with no I/O so the ONE rule this module exists to enforce
/// can be tested without a fixture: **zero compared samples is UNMEASURED, and
/// UNMEASURED is not clean.** Ten of 228607's published files were recorded
/// CLEAN on exactly zero compared rows each.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Outcome {
    /// nothing was compared; this says nothing about either file
    Unmeasured,
    /// identical, then apart, then exactly identical again
    Splice,
    /// the file simply IS the reference recording
    IsTheReference,
    /// no splice found AGAINST THIS REFERENCE
    NoSpliceFound,
}

impl Outcome {
    pub fn exit_code(self) -> i32 {
        match self {
            Outcome::Unmeasured => 3,
            Outcome::Splice | Outcome::IsTheReference => 2,
            Outcome::NoSpliceFound => 0,
        }
    }
}

pub fn outcome(rows: usize, identical: usize, reconverged: usize, ident_pct_bar: f64) -> Outcome {
    if rows == 0 {
        return Outcome::Unmeasured;
    }
    if reconverged > 0 {
        return Outcome::Splice;
    }
    if 100.0 * identical as f64 / rows as f64 >= ident_pct_bar {
        return Outcome::IsTheReference;
    }
    Outcome::NoSpliceFound
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_denominator_is_never_a_clean_result() {
        // the shape that produced ten CLEAN verdicts on zero evidence
        assert_eq!(outcome(0, 0, 0, 90.0), Outcome::Unmeasured);
        assert_eq!(outcome(0, 0, 0, 90.0).exit_code(), 3);
        // and it must not be reachable by any other argument combination
        assert_eq!(outcome(0, 999, 999, 90.0), Outcome::Unmeasured);
    }

    #[test]
    fn only_re_convergence_is_a_splice() {
        // a shared PREFIX is determinism: our own sibling tapes are 67 %
        // bit-identical on one 203072 pair, and that is not evidence.
        assert_eq!(outcome(1000, 670, 0, 90.0), Outcome::NoSpliceFound);
        // wholesale identity is the file simply BEING the reference
        assert_eq!(outcome(1000, 1000, 0, 90.0), Outcome::IsTheReference);
        // identical -> apart -> exactly identical again cannot be driven
        assert_eq!(outcome(1000, 500, 1, 90.0), Outcome::Splice);
        // and a splice outranks the identity reading
        assert_eq!(outcome(1000, 1000, 1, 90.0), Outcome::Splice);
    }
}

/// How closely two orientations agree, AS ROTATIONS.
///
/// `q` and `-q` are the SAME rotation. Five 199100 files read
/// `(-0.7071, 0, 0.7071, 0)` against the humans' `(0.7071, 0, -0.7071, 0)` and
/// are perfectly correct; a naive equality test condemns them, our own
/// regenerated 47.483 included. A check that cries wolf gets switched off, and
/// then it is not a check.
pub fn quat_agreement(a: [f64; 4], b: [f64; 4]) -> f64 {
    (0..4).map(|k| a[k] * b[k]).sum::<f64>().abs()
}

pub fn spawn_verdict(dpos: f64, dot: f64, pos_tol: f64, ang_tol: f64) -> &'static str {
    if dpos > pos_tol {
        // > 2 m from where every run on this map starts means a different map.
        // 276874's first roof candidate began on one, with a clean container
        // and a clean identity.
        "WRONG-MAP-OR-SPAWN"
    } else if dot < ang_tol {
        // 197047's filmed tape carried the IDENTITY quaternion where all 26
        // human recordings read (3.39e-05, -0.7071, 0, 0.7071). Its positions
        // matched 1917 of 1917 samples and the car was sideways for the whole
        // 100-second clip.
        "FACING-WRONG"
    } else {
        "SPAWN-OK"
    }
}

#[cfg(test)]
mod spawn_tests {
    use super::*;

    #[test]
    fn q_and_minus_q_are_the_same_rotation() {
        let q = [0.7071, 0.0, -0.7071, 0.0];
        let neg = [-0.7071, 0.0, 0.7071, 0.0];
        assert!(quat_agreement(q, neg) > 0.999, "q and -q must agree");
        assert_eq!(spawn_verdict(0.0, quat_agreement(q, neg), 2.0, 0.99), "SPAWN-OK");
        // the identity quaternion against a real spawn attitude is 90 degrees
        // out and must be refused
        let ident = [0.0, 0.0, 0.0, 1.0];
        assert_eq!(spawn_verdict(0.0, quat_agreement(q, ident), 2.0, 0.99), "FACING-WRONG");
        // and a different map is a different map however it is facing
        assert_eq!(spawn_verdict(9.0, 1.0, 2.0, 0.99), "WRONG-MAP-OR-SPAWN");
    }
}

// ---------------------------------------------------------------------------
// tmtraj inputs — a run's inputs, read out of the telemetry alone
// ---------------------------------------------------------------------------

const INPUTS_USAGE: &str = "\
usage: tmtraj inputs GHOST [--csv FILE] [--events]

The steer / gas / brake the car was being given, recovered EXACTLY from bytes
14, 15 and 18 of every 50 ms telemetry sample. No input chunk, no engine.

  --events   only the samples where an input changed
  --csv F    write the table to F

Why this is exact and why that matters: byte 14 is
`floor((steer_i8 + 127) * 255 / 254)`, measured against the corpus, and the map
is injective — so the tape value is recoverable, not approximable. This crate
used to decode byte 14 as the float ((v/255) - 0.5) * 2 and every comparison
against a tape then needed a +-1 slop, which is precisely wide enough to hide a
`round`-instead-of-`floor` encoder error. Finding that error took one
verification statistic from a Cohen's kappa of 0.467 to 1.000.

An UNREACHABLE byte is reported, never rounded to the nearest legal value: a
byte 14 no steer value can produce was not written by the game recording a car
being steered, and that is a finding.

CAVEAT, and do not let it be forgotten: this is what the RECORD says the car
was given. It is a 50 ms resampling of a 10 ms channel, so it is not an input
count -- six README-stated counts were once checked against telemetry and all
six were wrong (14->17, 23->16, 19->16, 15->59, 2->523, 3->1753). For what the
driver pressed, read the input chunk: `ghost tape extract`.
";

pub fn cmd_inputs(argv: &[String]) -> i32 {
    let a = cli::parse("tmtraj inputs", argv, &["events"]);
    let csv = a.one("csv").map(|s| s.to_string());
    let events_only = a.has("events");
    let a = a.finish(INPUTS_USAGE);
    let Some(path) = a.positional.first() else {
        eprint!("{}", INPUTS_USAGE);
        return 2;
    };
    let d = match load(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("UNMEASURED: {}", e);
            return 3;
        }
    };
    let mut out = String::from("t_s\tsteer_i8\tgas\tbrake\n");
    let (mut prev, mut n, mut changes) = (None, 0usize, 0usize);
    let (mut off_steer, mut off_pedal) = (0usize, 0usize);
    for (i, s) in d.samples.iter().enumerate() {
        let raw = match d.raw_sample(i) {
            Some(r) if r.len() > 18 => r,
            _ => continue,
        };
        let steer = record::steer_i8_from_byte(raw[14]);
        let gas = record::pedal_from_byte(raw[15]);
        let brake = record::pedal_from_byte(raw[18]);
        if steer.is_none() {
            off_steer += 1;
        }
        if gas.is_none() || brake.is_none() {
            off_pedal += 1;
        }
        let cur = (steer, gas, brake);
        let changed = prev.map_or(true, |p| p != cur);
        if changed {
            changes += 1;
        }
        prev = Some(cur);
        n += 1;
        if events_only && !changed {
            continue;
        }
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            secs(s.time_ms as i64),
            steer.map_or_else(|| format!("OFFGRID(0x{:02x})", raw[14]), |v| v.to_string()),
            gas.map_or_else(|| format!("ANALOGUE(0x{:02x})", raw[15]), |v| u8::from(v).to_string()),
            brake.map_or_else(|| format!("ANALOGUE(0x{:02x})", raw[18]), |v| u8::from(v).to_string()),
        ));
    }
    match csv {
        Some(f) => {
            std::fs::write(&f, &out).expect("write");
            eprintln!("wrote {}", f);
        }
        None => print!("{}", out),
    }
    println!(
        "{} samples, {} input changes on the 50 ms grid; {} steer byte(s) off the tape's i8 grid, \
         {} pedal byte(s) that are neither 0 nor 255",
        n, changes, off_steer, off_pedal
    );
    // NOT a refusal. Both counts are measurements, and see the module notes:
    // an off-grid steer byte is what a `round` produces where the corpus model
    // uses `floor`, and an off-digital pedal byte means the pedal was analogue.
    // Both occur in the game's own downloaded recordings.
    0
}
