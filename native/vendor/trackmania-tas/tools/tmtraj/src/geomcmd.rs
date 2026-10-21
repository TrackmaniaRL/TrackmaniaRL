//! `tmtraj geom` — what the SHAPE of a run says, as against what its clock says.
//!
//! WHY (arm `ksi2`, 134672, 2026-08-22)
//! ------------------------------------
//! On a map where every attempt to drive the known route faster has been
//! measured shut, the remaining question is whether the route is the right
//! length. A lap time is `distance / mean speed`, and this project has spent
//! all of its effort on the denominator. These three commands read the
//! numerator.
//!
//! * `path` — how far a run actually travels, in total and per sector, and the
//!   mean speed that implies. Two runs on the same map with different path
//!   lengths are not two qualities of driving, they are two routes.
//! * `selfcut` — every place a run's own line comes back close to itself after
//!   a long interval. Each such pair is a candidate shortcut, and the time it
//!   would save is the interval. **This is the test a cell-coverage census
//!   cannot do**: a shortcut does not need an undriven cell, it needs to SKIP
//!   driven ones, so "every cell has been driven" is not evidence against one.
//! * `near` — the closest approach between two runs, sample by sample, so a
//!   claim that two laps take "the same line" is a number rather than a
//!   picture.
//!
//! All three read the recorded trajectory, so they inherit the standing
//! caveat: **a synthesised tape carries its template's telemetry.** Point them
//! at a downloaded human recording or a regenerated file, never at a raw
//! search output. `tmtraj diff` is how you find out which one you have.

use crate::cli;
use gbx::record::{self, Decoded};

const USAGE: &str = "\
tmtraj geom — the shape of a run, not its clock.

  geom path    FILE...              arclength, per sector, and the implied mean speed
        [--race MS]                 clamp to a race window ending at MS
  geom selfcut FILE                 where a line comes back near itself: candidate cuts
        [--mingap S]                a pair must be this far apart in TIME   [3.0]
        [--maxdist M]               ...and this close in SPACE             [30.0]
        [--top N]                   print the N best clusters              [40]
  geom near    A B                  closest approach of B to each sample of A
        [--stride N]                report every Nth sample                [20]
        [--segments]                measure to B's POLYLINE, not to its nearest
                                    SAMPLE -- the only form that can compare two
                                    recordings sampled on different 50 ms phases
                                    (sample-to-sample has a floor of half a
                                    sample-step: 2.2 m at 320 km/h)
  geom pace    A B...               WHERE B is ahead of A, at matched DISTANCE
        [--bin M]                   arclength grain                        [50.0]
  geom at      FILE... --arc M,M    the whole state of each run at one place
  geom track   GHOST --route TRACE.csv   engine-vs-recorded drift, and where it starts
        [--thresh M] [--every S]
  geom envelope FILE... --ref R      what the ROUTE is worth: the field's own speed envelope
        [--bin M] [--target S] [--quantile Q] [--self-control] [--per-bin]

Times print as seconds. Distances are metres.
";

struct Line {
    name: String,
    t: Vec<f64>,
    p: Vec<[f64; 3]>,
    cps: Vec<i32>,
    race_ms: Option<i32>,
}

fn load(path: &str, race_cap: Option<i32>) -> Result<Line, String> {
    let d: Decoded = record::decode_ghost(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut t = Vec::new();
    let mut p = Vec::new();
    for s in &d.samples {
        if let Some(c) = race_cap {
            if s.time_ms < 0 || s.time_ms > c {
                continue;
            }
        }
        t.push(s.time_ms as f64 / 1000.0);
        p.push([s.x as f64, s.y as f64, s.z as f64]);
    }
    if p.len() < 2 {
        return Err(format!("{}: {} usable samples", path, p.len()));
    }
    Ok(Line {
        name: path.rsplit('/').next().unwrap_or(path).to_string(),
        t,
        p,
        cps: d.checkpoints_ms.clone(),
        race_ms: d.race_time_ms,
    })
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Cumulative arclength along the sample polyline.
///
/// This is a LOWER bound on the distance travelled: the samples are 50 ms
/// apart and the car turns between them. At 40 m/s a sample step is 2 m and
/// the chord/arc error over a 30 m radius corner is under 0.1 %, so it is a
/// tight one — but it is a bound, and two runs are only comparable here
/// because they are both measured this way.
fn arclen(p: &[[f64; 3]]) -> Vec<f64> {
    let mut s = vec![0.0];
    for i in 1..p.len() {
        s.push(s[i - 1] + dist(&p[i - 1], &p[i]));
    }
    s
}

fn cmd_path(args: &[String]) -> i32 {
    let a = cli::parse("tmtraj geom path", args, &[]);
    let cap: i32 = a.num("race", -1);
    let a = a.finish(USAGE);
    let cap = if cap < 0 { None } else { Some(cap) };
    if a.positional.is_empty() {
        print!("{}", USAGE);
        return 2;
    }
    println!(
        "{:<38} {:>7} {:>9} {:>8} {:>8}  {}",
        "file", "time", "path_m", "mean_ms", "mean_kmh", "per-sector path_m (time)"
    );
    for f in &a.positional {
        let l = match load(f, cap) {
            Ok(l) => l,
            Err(e) => {
                println!("{:<38} {}", f.rsplit('/').next().unwrap_or(f), e);
                continue;
            }
        };
        let s = arclen(&l.p);
        let total = *s.last().unwrap();
        let secs = match l.race_ms {
            Some(m) => m as f64 / 1000.0,
            None => l.t[l.t.len() - 1] - l.t[0],
        };
        // Split the arclength at the checkpoint times.
        let mut marks: Vec<f64> = l.cps.iter().map(|c| *c as f64 / 1000.0).collect();
        marks.push(secs);
        let mut per = String::new();
        let mut prev_s = 0.0;
        let mut prev_t = 0.0;
        for m in &marks {
            // arclength at the sample nearest this time
            let i = l
                .t
                .iter()
                .enumerate()
                .min_by(|x, y| {
                    (x.1 - m).abs().partial_cmp(&(y.1 - m).abs()).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            per.push_str(&format!("{:.0}({:.3}) ", s[i] - prev_s, m - prev_t));
            prev_s = s[i];
            prev_t = *m;
        }
        println!(
            "{:<38} {:>7.3} {:>9.1} {:>8.2} {:>8.1}  {}",
            l.name,
            secs,
            total,
            total / secs,
            total / secs * 3.6,
            per.trim_end()
        );
    }
    0
}

/// A cluster of (i, j) pairs that are one geometric event.
struct Cut {
    ti: f64,
    tj: f64,
    d: f64,
    dy: f64,
    dhoriz: f64,
    cps_between: usize,
}

fn cmd_selfcut(args: &[String]) -> i32 {
    let a = cli::parse("tmtraj geom selfcut", args, &[]);
    let mingap: f64 = a.num("mingap", 3.0);
    let maxdist: f64 = a.num("maxdist", 30.0);
    let top: usize = a.num("top", 40);
    let a = a.finish(USAGE);
    let Some(f) = a.positional.first() else {
        print!("{}", USAGE);
        return 2;
    };
    let l = match load(f, None) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    let n = l.p.len();
    // Best (smallest distance) partner for each i, among j with t_j - t_i >= mingap.
    let mut best: Vec<Option<(usize, f64)>> = vec![None; n];
    for i in 0..n {
        let mut b: Option<(usize, f64)> = None;
        for j in (i + 1)..n {
            if l.t[j] - l.t[i] < mingap {
                continue;
            }
            let d = dist(&l.p[i], &l.p[j]);
            if d > maxdist {
                continue;
            }
            if b.map(|(_, bd)| d < bd).unwrap_or(true) {
                b = Some((j, d));
            }
        }
        best[i] = b;
    }
    // Cluster: consecutive i with a partner, split when the partner index jumps.
    let mut cuts: Vec<Cut> = Vec::new();
    let mut i = 0;
    while i < n {
        if best[i].is_none() {
            i += 1;
            continue;
        }
        let mut k = i;
        let mut arg = i;
        while k + 1 < n {
            let Some((jk, _)) = best[k] else { break };
            let Some((jk1, _)) = best[k + 1] else { break };
            if (jk1 as i64 - jk as i64).abs() > 20 {
                break;
            }
            k += 1;
            if best[k].unwrap().1 < best[arg].unwrap().1 {
                arg = k;
            }
        }
        let (j, d) = best[arg].unwrap();
        let cps_between = l
            .cps
            .iter()
            .filter(|c| {
                let s = **c as f64 / 1000.0;
                s > l.t[arg] && s < l.t[j]
            })
            .count();
        let dy = l.p[j][1] - l.p[arg][1];
        let dh = ((l.p[j][0] - l.p[arg][0]).powi(2) + (l.p[j][2] - l.p[arg][2]).powi(2)).sqrt();
        cuts.push(Cut { ti: l.t[arg], tj: l.t[j], d, dy, dhoriz: dh, cps_between });
        i = k + 1;
    }
    cuts.sort_by(|a, b| (b.tj - b.ti).partial_cmp(&(a.tj - a.ti)).unwrap());
    println!(
        "{}: {} samples, {:.1} m of line, {} checkpoints",
        l.name,
        n,
        arclen(&l.p).last().unwrap(),
        l.cps.len()
    );
    println!("pairs at least {:.1} s apart and within {:.1} m:", mingap, maxdist);
    println!(
        "{:>8} {:>8} {:>8} {:>7} {:>8} {:>8}  {}",
        "from", "to", "saves", "gap_m", "dy_m", "horiz_m", "note"
    );
    for c in cuts.iter().take(top) {
        let note = if c.cps_between > 0 {
            format!("SKIPS {} checkpoint(s) -- void unless detoured", c.cps_between)
        } else {
            String::new()
        };
        println!(
            "{:>8.3} {:>8.3} {:>8.3} {:>7.2} {:>+8.2} {:>8.2}  {}",
            c.ti,
            c.tj,
            c.tj - c.ti,
            c.d,
            c.dy,
            c.dhoriz,
            note
        );
    }
    if cuts.is_empty() {
        println!("(none)");
    }
    0
}

fn cmd_near(args: &[String]) -> i32 {
    let a = cli::parse("tmtraj geom near", args, &["segments"]);
    let stride: usize = a.num("stride", 20);
    let segments = a.has("segments");
    let a = a.finish(USAGE);
    if a.positional.len() < 2 {
        print!("{}", USAGE);
        return 2;
    }
    let (x, y) = (&a.positional[0], &a.positional[1]);
    let (la, lb) = match (load(x, None), load(y, None)) {
        (Ok(p), Ok(q)) => (p, q),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("{}", e);
            return 2;
        }
    };
    println!("A = {}   B = {}", la.name, lb.name);
    if segments {
        println!("(B as a POLYLINE: distance to the nearest point ON it, not to its nearest SAMPLE)");
    }
    println!("{:>8} {:>10} {:>10} {:>10}", "t_A", "nearest_m", "t_B", "dt");
    let mut worst = (0.0f64, 0.0f64);
    for i in (0..la.p.len()).step_by(stride.max(1)) {
        let b = if segments {
            nearest_on_polyline(&la.p[i], &lb.p, &lb.t)
        } else {
            let mut b = (f64::MAX, 0.0f64);
            for j in 0..lb.p.len() {
                let d = dist(&la.p[i], &lb.p[j]);
                if d < b.0 {
                    b = (d, lb.t[j]);
                }
            }
            b
        };
        if b.0 > worst.0 {
            worst = (b.0, la.t[i]);
        }
        println!("{:>8.3} {:>10.4} {:>10.3} {:>+10.3}", la.t[i], b.0, b.1, b.1 - la.t[i]);
    }
    println!("worst separation {:.4} m at t_A {:.3}", worst.0, worst.1);
    0
}

/// Distance from a point to the nearest point ON a polyline, with the time
/// interpolated at that point.
///
/// WHY THIS EXISTS, and why sample-to-sample was not enough. Two recordings of
/// the SAME path can be sampled on different 50 ms phases -- a regeneration
/// lands its grid where the engine put it, and the published file's grid came
/// from wherever the game started counting. Compared sample to sample, the
/// nearest B sample to an A point is then up to half a sample-step away, which
/// at 320 km/h is **2.2 m of pure sampling phase and no divergence at all**.
/// On 228811 that read as a mean 1.78 m "shift" on a path that had not moved.
///
/// A point-to-SEGMENT distance has no such floor: it is zero wherever the two
/// polylines coincide, whatever phase either was sampled on. Its own floor is
/// the chord error of the sampling (a straight segment across a curve), which
/// is a second-order term, not a first-order one.
fn nearest_on_polyline(p: &[f64; 3], q: &[[f64; 3]], t: &[f64]) -> (f64, f64) {
    let mut best = (f64::MAX, 0.0f64);
    if q.is_empty() {
        return (f64::MAX, 0.0);
    }
    if q.len() == 1 {
        return (dist(p, &q[0]), t[0]);
    }
    for j in 0..q.len() - 1 {
        let (a, b) = (&q[j], &q[j + 1]);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
        let den = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        // A zero-length segment (a stationary car) degenerates to its endpoint
        // rather than dividing by zero.
        let f = if den > 0.0 {
            ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / den).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let c = [a[0] + f * ab[0], a[1] + f * ab[1], a[2] + f * ab[2]];
        let d = dist(p, &c);
        if d < best.0 {
            best = (d, t[j] + f * (t[j + 1] - t[j]));
        }
    }
    best
}

/// Interpolate a run's race time and speed at a given arclength.
fn at_arc(t: &[f64], s: &[f64], v: &[f64], target: f64) -> Option<(f64, f64)> {
    if target < 0.0 || target > *s.last()? {
        return None;
    }
    let i = match s.binary_search_by(|x| x.partial_cmp(&target).unwrap()) {
        Ok(i) => i.max(1),
        Err(i) => i.max(1).min(s.len() - 1),
    };
    let (s0, s1) = (s[i - 1], s[i]);
    let f = if s1 > s0 { (target - s0) / (s1 - s0) } else { 0.0 };
    Some((t[i - 1] + f * (t[i] - t[i - 1]), v[i - 1] + f * (v[i] - v[i - 1])))
}

/// `geom pace` — WHERE one run is ahead of another, measured at matched
/// DISTANCE rather than at matched time.
///
/// Checkpoint splits give five numbers on a 67-second lap. The same
/// information at a 50 m grain is fifty, and it is the difference between
/// "sector 3 is slow" and "the 2.6 s is one corner at 1450 m".
///
/// Matching on arclength is what makes two runs comparable at all: at matched
/// TIME two runs are in different places and the comparison is meaningless.
fn cmd_pace(args: &[String]) -> i32 {
    let a = cli::parse("tmtraj geom pace", args, &[]);
    let bin: f64 = a.num("bin", 50.0);
    let a = a.finish(USAGE);
    if a.positional.len() < 2 {
        print!("{}", USAGE);
        return 2;
    }
    struct R {
        name: String,
        t: Vec<f64>,
        s: Vec<f64>,
        v: Vec<f64>,
    }
    let mut runs: Vec<R> = Vec::new();
    for f in &a.positional {
        let d = match record::decode_ghost(f) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{}: {}", f, e);
                return 2;
            }
        };
        let mut t = Vec::new();
        let mut p = Vec::new();
        let mut v = Vec::new();
        for sm in &d.samples {
            if sm.time_ms < 0 {
                continue;
            }
            t.push(sm.time_ms as f64 / 1000.0);
            p.push([sm.x as f64, sm.y as f64, sm.z as f64]);
            v.push(sm.speed_kmh as f64);
        }
        let s = arclen(&p);
        runs.push(R { name: f.rsplit('/').next().unwrap_or(f).to_string(), t, s, v });
    }
    let names: Vec<&str> = runs.iter().map(|r| r.name.as_str()).collect();
    print!("{:>8}", "arc_m");
    for n in &names {
        let short: String = n.chars().take(12).collect();
        print!(" {:>13}", short);
    }
    println!("     (t, then dt vs first; speed km/h in brackets)");
    let end = runs[0].s.last().copied().unwrap_or(0.0);
    let mut x = 0.0;
    // cumulative per-run gain, so the last row is the whole-lap decomposition
    while x <= end + 1e-9 {
        let base = at_arc(&runs[0].t, &runs[0].s, &runs[0].v, x);
        print!("{:>8.0}", x);
        for (k, r) in runs.iter().enumerate() {
            match at_arc(&r.t, &r.s, &r.v, x) {
                None => print!(" {:>13}", "-"),
                Some((tt, vv)) => {
                    if k == 0 {
                        print!(" {:>7.3}[{:>3.0}]", tt, vv);
                    } else {
                        let d = tt - base.map(|b| b.0).unwrap_or(tt);
                        print!(" {:>+7.3}[{:>3.0}]", d, vv);
                    }
                }
            }
        }
        println!();
        x += bin;
    }
    0
}

/// `geom at` — the full state of every run at one place on the ribbon.
///
/// The state, not the clock: a handover between two searched sectors is only
/// as good as the position, velocity AND attitude it hands over, and on this
/// family of maps lateral speed has been the discriminator that arrival time
/// could not see.
fn cmd_at(args: &[String]) -> i32 {
    let a0 = cli::parse("tmtraj geom at", args, &[]);
    let arcs_src = a0.one("arc").unwrap_or("").to_string();
    let a = a0.finish(USAGE);
    let arcs: Vec<f64> = arcs_src
        .as_str()
        .split(',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    if arcs.is_empty() || a.positional.is_empty() {
        print!("{}", USAGE);
        return 2;
    }
    println!(
        "{:>8} {:<26} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "arc_m", "file", "t", "kmh", "side_ms", "x", "y", "z", "yaw", "grnd"
    );
    for target in &arcs {
        for f in &a.positional {
            let Ok(d) = record::decode_ghost(f) else { continue };
            let sm: Vec<_> = d.samples.iter().filter(|s| s.time_ms >= 0).collect();
            if sm.len() < 2 {
                continue;
            }
            let p: Vec<[f64; 3]> = sm.iter().map(|s| [s.x as f64, s.y as f64, s.z as f64]).collect();
            let s = arclen(&p);
            let i = match s.binary_search_by(|x| x.partial_cmp(target).unwrap()) {
                Ok(i) => i,
                Err(i) => i.min(s.len() - 1),
            };
            let q = sm[i];
            println!(
                "{:>8.0} {:<26} {:>7.3} {:>8.1} {:>8.2} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>7}",
                target,
                f.rsplit('/').next().unwrap_or(f).chars().take(26).collect::<String>(),
                q.time_ms as f64 / 1000.0,
                q.speed_kmh,
                q.side_speed,
                q.x,
                q.y,
                q.z,
                q.yaw.to_degrees(),
                if q.is_ground_contact { "yes" } else { "AIR" }
            );
        }
        println!();
    }
    0
}
/// `geom envelope` — see [`crate::envelope`] for the reasoning; this is the
/// command surface.
fn cmd_envelope(args: &[String]) -> i32 {
    use crate::envelope as ev;
    let a = cli::parse("tmtraj geom envelope", args, &["self-control", "per-bin"]);
    let bin: f64 = a.num("bin", 10.0);
    let target: f64 = a.num("target", 0.0);
    let quant: f64 = a.num("quantile", 0.98);
    let refpath = a.one("ref").map(|s| s.to_string());
    // The fastest speed any car on this map has been seen at, m/s: the cap on
    // how far a projection may advance in one 50 ms sample.
    let selfctl = a.has("self-control");
    let perbin = a.has("per-bin");
    let a = a.finish(USAGE);
    if a.positional.is_empty() || refpath.is_none() {
        eprintln!("geom envelope needs --ref REFERENCE_GHOST and at least one run");
        return 2;
    }
    let refpath = refpath.unwrap();
    let Some(rd) = ev::load(&refpath) else {
        eprintln!("cannot read reference {}", refpath);
        return 2;
    };
    let r = ev::Ref::from(&rd);
    let total = r.total();

    println!("reference {} : {:.1} m", refpath.rsplit('/').next().unwrap(), total);

    let mut runs: Vec<ev::Run> = Vec::new();
    for f in &a.positional {
        let Some(d) = ev::load(f) else {
            eprintln!("skip {}", f);
            continue;
        };
        let nm = f.rsplit('/').next().unwrap_or(f).to_string();
        let run = ev::project_run(&nm, &d, &r);
        // A run whose projection never reaches the end of the reference did
        // not drive this route; say so rather than folding it in.
        let reach = run.s.last().copied().unwrap_or(0.0);
        if reach < total * 0.98 {
            println!("  EXCLUDED {:<28} projection reaches only {:.0} m of {:.0}", nm, reach, total);
            continue;
        }
        runs.push(run);
    }
    if runs.is_empty() {
        eprintln!("no runs project onto this reference");
        return 2;
    }

    const VSTEP: f64 = 5.0;
    const NV: usize = 24;

    if selfctl {
        // THE CONTROL. On one run's own data the whole pipeline must return
        // that run's own lap time; anything else is an instrument fault, and
        // the field bound below would be measuring it.
        println!();
        println!("{:<30} {:>9} {:>10} {:>10} {:>9}", "self-control", "real", "raw", "feasible", "err_raw");
        for run in &runs {
            let one = std::slice::from_ref(run);
            let b = ev::envelope(one, total, bin);
            let (ac, de) = ev::accel_limits(one, VSTEP, NV, quant);
            let mut vm = b.vmax();
            ev::interpolate_holes(&mut vm);
            let f = ev::feasible(&vm, bin, &ac, &de, VSTEP);
            let real = run.t.last().copied().unwrap_or(0.0) - run.t[0];
            let (raw, _) = ev::integrate(&vm, bin, total);
            let (fea, _) = ev::integrate(&f, bin, total);
            let cov = b.tmin.iter().filter(|x| x.is_some()).count();
            println!(
                "{:<30} {:>9.3} {:>10.3} {:>10.3} {:>+9.3}   {} of {} bins, {:.2} m off the reference",
                run.name, real, raw, fea, raw - real, cov, b.n, run.median_miss
            );
        }
    }

    let b = ev::envelope(&runs, total, bin);
    let (ac, de) = ev::accel_limits(&runs, VSTEP, NV, quant);
    let mut vm = b.vmax();
    // A bin nobody crossed has no measured speed. The forward-backward pass
    // reads a zero there as "the car is stationary" and drags the whole lap to
    // 690 s. Interpolate across the hole, and the WARNING below says how many
    // there were, so the reader can price the assumption.
    ev::interpolate_holes(&mut vm);
    let f = ev::feasible(&vm, bin, &ac, &de, VSTEP);
    let (raw, up1) = ev::integrate(&vm, bin, total);
    let (fea, _) = ev::integrate(&f, bin, total);
    if up1 > 0 {
        println!("WARNING: {} of {} bins were crossed by no run and are UNPRICED", up1, b.n);
    }

    if perbin {
        println!();
        println!("{:>8} {:>8} {:>10} {:>10}  {}", "arc_m", "raw_ms", "feas_ms", "feas_kmh", "fastest run there");
        for i in 0..b.n {
            let w = b.who[i];
            println!(
                "{:>8.0} {:>8.2} {:>10.2} {:>10.1}  {}",
                i as f64 * bin,
                vm[i],
                f[i],
                f[i] * 3.6,
                if w == usize::MAX { "-" } else { &runs[w].name }
            );
        }
    }

    println!();
    println!("acceleration limits at the {:.0}th percentile of this field, m/s^2:", quant * 100.0);
    for k in 0..NV {
        if ac[k] > 0.0 || de[k] > 0.0 {
            println!(
                "  {:>3.0}-{:>3.0} m/s   accel {:>6.2}   decel {:>6.2}",
                k as f64 * VSTEP,
                (k + 1) as f64 * VSTEP,
                ac[k],
                de[k]
            );
        }
    }
    let mut owners: Vec<(usize, usize)> = Vec::new();
    for k in 0..runs.len() {
        let c = b.who.iter().filter(|w| **w == k).count();
        if c > 0 {
            owners.push((c, k));
        }
    }
    owners.sort_by(|a, b| b.0.cmp(&a.0));
    println!();
    println!("{} runs, {} bins of {:.0} m over {:.1} m of route", runs.len(), b.n, bin, total);
    for (c, k) in owners {
        println!("  fastest in {:>4} bins   {}", c, runs[k].name);
    }
    println!("RAW ENVELOPE      = {:.3}", raw);
    println!("FEASIBLE ENVELOPE = {:.3}   (forward-backward under this field's own accel limits)", fea);
    if target > 0.0 {
        println!(
            "target {:.3}: raw {:+.3}, feasible {:+.3}",
            target,
            raw - target,
            fea - target
        );
    }
    0
}

/// `geom track` — how far the ENGINE's own run of a tape drifts from the
/// trajectory that tape's file records.
///
/// The use this was built for: a recording made on an older game build
/// replays exactly for a while and then loses the car. Whether that is a
/// physics difference or a one-off rounding seed is decided by the SHAPE of
/// the divergence — a physics difference shows a drift that is present from
/// the first metre and grows steadily; a seed shows a flat zero and then a
/// sudden departure at the first place on the map that amplifies.
///
/// `--route` is a `fk trace` CSV (the engine's own per-tick state);
/// the positional argument is the ghost whose recorded telemetry is the
/// reference. Both are matched on race time, so a trace that starts late is
/// fine.
fn cmd_track(args: &[String]) -> i32 {
    let a = cli::parse("tmtraj geom track", args, &[]);
    let route = a.one("route").map(|s| s.to_string());
    let thresh: f64 = a.num("thresh", 1.0);
    let every: f64 = a.num("every", 1.0);
    let fit_window: f64 = a.num("fitwindow", 2.0);
    let a = a.finish(USAGE);
    let (Some(route), Some(f)) = (route, a.positional.first()) else {
        eprintln!("geom track needs GHOST --route TRACE.csv");
        return 2;
    };
    let Ok(d) = record::decode_ghost(f) else {
        eprintln!("cannot read {}", f);
        return 2;
    };
    let Ok(txt) = std::fs::read_to_string(&route) else {
        eprintln!("cannot read {}", route);
        return 2;
    };
    // trace CSV: time_ms,x,y,z,...
    let mut tr: Vec<(f64, [f64; 3])> = Vec::new();
    for (i, line) in txt.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 4 {
            continue;
        }
        let (Ok(t), Ok(x), Ok(y), Ok(z)) =
            (c[0].parse::<f64>(), c[1].parse::<f64>(), c[2].parse::<f64>(), c[3].parse::<f64>())
        else {
            continue;
        };
        tr.push((t / 1000.0, [x, y, z]));
    }
    if tr.is_empty() {
        eprintln!("{}: no rows", route);
        return 2;
    }
    println!("{} recorded samples vs {} engine ticks", d.samples.len(), tr.len());
    // THE LAG SCAN. A recording and the engine's own trace of its tape are on
    // different clocks by a whole number of 10 ms ticks, and at 150 km/h one
    // tick is 0.42 m. Comparing at lag 0 reports that as "drift" — on a file
    // known to replay to the millisecond, at every point of a 68 s lap.
    // A magnitude cannot see which side of a tick a file is on: scan the lag,
    // pick the one that minimises the median, and report it.
    // Fit the lag on the FIRST FEW SECONDS the trace covers, never on the
    // whole run: on a diverging tape the median over the lap is hundreds of
    // metres and the scan then picks a lag at random. That is how a run whose
    // divergence starts at zero came out reading 1.3 m at its own fork tick.
    let t0 = tr[0].0;
    let fitwin: f64 = fit_window;
    let mut best_lag = 0i64;
    let mut best_med = f64::MAX;
    for lag in -4i64..=4 {
        let mut v = Vec::new();
        for s in d.samples.iter().filter(|s| s.time_ms >= 0 && (s.time_ms as f64 / 1000.0) <= t0 + fitwin) {
            let t = s.time_ms as f64 / 1000.0 + lag as f64 * 0.01;
            if let Some((tt, p)) =
                tr.iter().min_by(|a, b| (a.0 - t).abs().partial_cmp(&(b.0 - t).abs()).unwrap())
            {
                if (tt - t).abs() <= 0.006 {
                    v.push(dist(&[s.x as f64, s.y as f64, s.z as f64], p));
                }
            }
        }
        if v.len() < 10 {
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        if med < best_med {
            best_med = med;
            best_lag = lag;
        }
    }
    println!(
        "best whole-tick lag {:+} ({} ms), median residual there {:.4} m",
        best_lag,
        best_lag * 10,
        best_med
    );
    println!("{:>9} {:>12} {:>10} {:>10}", "race", "drift_m", "lateral_m", "engine_kmh");
    let mut first_over: Option<f64> = None;
    let mut next_print = 0.0;
    let samples: Vec<_> = d.samples.iter().filter(|s| s.time_ms >= 0).collect();
    for (si, s) in samples.iter().enumerate() {
        let t = s.time_ms as f64 / 1000.0 + best_lag as f64 * 0.01;
        // nearest engine tick in time
        let Some((tt, p)) = tr
            .iter()
            .min_by(|a, b| (a.0 - t).abs().partial_cmp(&(b.0 - t).abs()).unwrap())
            .copied()
        else {
            break;
        };
        if (tt - t).abs() > 0.006 {
            continue;
        }
        let dr = dist(&[s.x as f64, s.y as f64, s.z as f64], &p);
        // SIGNED LATERAL OFFSET, left/right of where the recording was going.
        //
        // The magnitude says how far apart the two runs are; the sign says
        // whether the engine turned LESS than the recording (the car runs wide)
        // or MORE. That distinction decides whether a repair is even
        // expressible: a tape already at full lock has no "more" available, so
        // a divergence that needs more lock cannot be corrected by steering at
        // all.
        let lat = {
            let j = (si + 1).min(samples.len() - 1);
            let (hx, hz) = ((samples[j].x - s.x) as f64, (samples[j].z - s.z) as f64);
            let n = (hx * hx + hz * hz).sqrt().max(1e-9);
            let (ux, uz) = (hx / n, hz / n);
            // left-normal of the heading in (x, z)
            let (nx, nz) = (-uz, ux);
            (p[0] - s.x as f64) * nx + (p[2] - s.z as f64) * nz
        };
        if dr > thresh && first_over.is_none() {
            first_over = Some(t);
        }
        if t >= next_print {
            println!(
                "{:>9.3} {:>12.4} {:>+10.4} {:>10.1}",
                t - best_lag as f64 * 0.01,
                dr,
                lat,
                s.speed_kmh
            );
            next_print += every;
        }
    }
    match first_over {
        Some(t) => println!("\nFIRST DIVERGENCE past {:.2} m at race {:.3}", thresh, t),
        None => println!("\nnever diverges past {:.2} m", thresh),
    }
    0
}

pub fn cmd(args: &[String]) -> i32 {
    match args.first().map(|s| s.as_str()) {
        Some("path") => cmd_path(&args[1..]),
        Some("selfcut") => cmd_selfcut(&args[1..]),
        Some("near") => cmd_near(&args[1..]),
        Some("pace") => cmd_pace(&args[1..]),
        Some("at") => cmd_at(&args[1..]),
        Some("envelope") => cmd_envelope(&args[1..]),
        Some("track") => cmd_track(&args[1..]),
        _ => {
            print!("{}", USAGE);
            2
        }
    }
}
