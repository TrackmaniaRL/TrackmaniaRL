//! `ghost phase` -- IS THIS RECORD ON THE GAME'S OWN PHYSICS TICK?
//!
//! A regenerated record can be right about every position and still be a
//! single 10 ms tick away from the run it claims to be. Solo it is invisible:
//! the trajectory is the same curve, sampled a moment later. In a
//! frame-synchronous comparison it is fatal -- one of this project's two-car
//! clips came down for exactly this, a pure time shift of 0.336 m along the
//! track and 0.004 m across it, which is the encoder's own floor.
//!
//! WHY A MAGNITUDE ALONE CANNOT ANSWER IT. `ghost trajdiff` compares two files
//! at whole-SAMPLE shifts, and a sample is 50 ms -- five physics ticks. A
//! one-tick offset therefore shows up as a small constant separation at shift
//! 0 that no shift removes, and a small constant separation is exactly what a
//! correct file looks like too. The number is the same shape either way.
//!
//! THE ONLY THING THAT SETTLES IT IS A DOWNLOADED CONTROL. A recording the
//! game made itself is on the game's phase by definition. Regenerate THAT from
//! its own inputs and the residual is the instrument's zero on this map -- and
//! it is not 0.000, it is whatever this map and this binary happen to give.
//! Measured here across thirteen maps it is 0.0005 m on most and 0.36 m on
//! two, and 0.36 m is a whole tick. Judging a subject against an assumed zero
//! of 0.000 convicts every honest file on those two maps.
//!
//! AND ONE RUN IS NOT THE ZERO. The regenerator's tick alignment is a property
//! of the RUN, not of the file: the same command on the same binary and the
//! same inputs lands a tick out roughly one time in seven. So the zero is the
//! MODE over repeated runs, the run varied rather than the probe tick, and a
//! command that takes `--runs 1` would be a worse instrument than none.
//!
//! It writes nothing. It measures, prints the distribution it measured, and
//! leaves the decision to a person.

use crate::regen;

/// One map's answer.
pub struct Phase {
    /// Every run's residual against the control, in metres.
    pub control_runs: Vec<f64>,
    /// The along-track / across-track decomposition of the modal run.
    pub control_split: Option<(f64, f64, f64)>,
    /// The mode: runs clustered to 1 mm, the biggest cluster's median.
    pub control_zero: f64,
    /// How far the car moves between two RECORDED SAMPLES, from the control's
    /// own trajectory -- the scale that turns a residual into ticks.
    pub sample_m: f64,
    pub subjects: Vec<Subject>,
}

pub struct Subject {
    pub file: String,
    pub runs: Vec<f64>,
    pub delta: f64,
    pub split: Option<(f64, f64, f64)>,
}

fn mode_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    // cluster to the millimetre; the biggest cluster wins, ties to the smaller
    // value, and the answer is that cluster's median rather than its mean so
    // one wild run cannot move it.
    let mut buckets: std::collections::BTreeMap<i64, Vec<f64>> = Default::default();
    for x in v {
        buckets.entry((x * 1000.0).round() as i64).or_default().push(*x);
    }
    let best = buckets.iter().max_by_key(|(k, c)| (c.len(), -**k)).unwrap();
    let mut c = best.1.clone();
    c.sort_by(|a, b| a.total_cmp(b));
    c[c.len() / 2]
}

/// Mean separation between two files' recorded trajectories at zero shift,
/// decomposed into ALONG-TRACK and ACROSS-TRACK components.
///
/// WHY THE DECOMPOSITION IS THE WHOLE MEASUREMENT. A magnitude cannot tell a
/// time shift from a physics divergence: both are "the car is somewhere else".
/// But a pure time shift is displacement ALONG the direction of travel and
/// nothing across it, and the two runs are then the same curve at two
/// different instants. That is exactly what took 270053's two-car clip down --
/// 0.336 m along the track, 0.004 m across it, and the cross-track number is
/// the position encoder's own floor.
///
/// The along-track component is signed, so it also answers WHICH WAY: divide
/// it by the speed and it is a time in seconds. A clean instrument returns a
/// few tens of microseconds; a tick-offset one returns +/-0.010 s.
struct Sep {
    mean: f64,
    along_m: f64,
    across_m: f64,
    /// the along-track offset as a time, in seconds: along / speed
    dt_s: f64,
    n: usize,
}

fn sep_at_zero(a: &str, b: &str) -> Option<Sep> {
    let da = gbx::record::decode_ghost(a).ok()?;
    let db = gbx::record::decode_ghost(b).ok()?;
    let pos = |s: &gbx::record::Sample| [s.x as f64, s.y as f64, s.z as f64];
    let m: std::collections::HashMap<i64, [f64; 3]> =
        db.samples.iter().map(|s| (s.time_ms as i64, pos(s))).collect();
    let (mut sum, mut along, mut across, mut dt, mut n) = (0.0, 0.0, 0.0, 0.0, 0usize);
    for (i, s) in da.samples.iter().enumerate() {
        let Some(p) = m.get(&(s.time_ms as i64)) else { continue };
        let q = pos(s);
        let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let dist = (0..3).map(|k| d[k] * d[k]).sum::<f64>().sqrt();
        sum += dist;
        n += 1;
        // The direction of travel, from the file's OWN neighbouring samples --
        // not from a velocity field, which is one of the channels under
        // suspicion.
        let j = if i + 1 < da.samples.len() { i + 1 } else { i.saturating_sub(1) };
        if j == i {
            continue;
        }
        let (u, v) = (pos(&da.samples[i.min(j)]), pos(&da.samples[i.max(j)]));
        let step = [v[0] - u[0], v[1] - u[1], v[2] - u[2]];
        let sl = (0..3).map(|k| step[k] * step[k]).sum::<f64>().sqrt();
        if sl < 1e-9 {
            continue;
        }
        let unit = [step[0] / sl, step[1] / sl, step[2] / sl];
        let al = (0..3).map(|k| d[k] * unit[k]).sum::<f64>();
        along += al;
        across += (dist * dist - al * al).max(0.0).sqrt();
        // sl metres per 50 ms sample -> speed; the along-track offset in time
        let speed = sl / 0.05;
        if speed > 0.5 {
            dt += al / speed;
        }
    }
    if n == 0 {
        return None;
    }
    let f = n as f64;
    Some(Sep { mean: sum / f, along_m: along / f, across_m: across / f, dt_s: dt / f, n })
}

/// Median distance between consecutive recorded samples -- the scale on which
/// a tick is 1/5 of a step.
fn sample_step(a: &str) -> Option<f64> {
    let d = gbx::record::decode_ghost(a).ok()?;
    let mut v: Vec<f64> = Vec::new();
    for w in d.samples.windows(2) {
        let a = [w[0].x as f64, w[0].y as f64, w[0].z as f64];
        let b = [w[1].x as f64, w[1].y as f64, w[1].z as f64];
        v.push((0..3).map(|k| (b[k] - a[k]).powi(2)).sum::<f64>().sqrt());
    }
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

fn regen_once(src: &str, map: &str, tag: &str) -> Option<String> {
    let out = std::env::temp_dir()
        .join(format!("ghost-phase-{}-{}.Ghost.Gbx", std::process::id(), tag))
        .to_string_lossy()
        .to_string();
    let _ = std::fs::remove_file(&out);
    let r = regen::run_regen(src, map, &out, &["--trim-outside".to_string()]);
    if r.ok && std::path::Path::new(&out).exists() {
        Some(out)
    } else {
        eprintln!("  (a run produced no file: {})", r.log.lines().last().unwrap_or(""));
        None
    }
}

/// Regenerate `src` `runs` times and return every run's residual against it,
/// plus the along/across decomposition of the run nearest the mode.
fn residuals(src: &str, map: &str, runs: usize, tag: &str) -> (Vec<f64>, Option<(f64, f64, f64)>) {
    let mut out = Vec::new();
    let mut splits: Vec<(f64, Sep)> = Vec::new();
    for i in 0..runs {
        if let Some(o) = regen_once(src, map, &format!("{}{}", tag, i)) {
            if let Some(s) = sep_at_zero(src, &o) {
                out.push(s.mean);
                splits.push((s.mean, s));
            }
            let _ = std::fs::remove_file(&o);
        }
    }
    if out.is_empty() {
        return (out, None);
    }
    let m = mode_of(&out);
    let split = splits
        .iter()
        .min_by(|a, b| (a.0 - m).abs().total_cmp(&(b.0 - m).abs()))
        .map(|(_, s)| (s.along_m, s.across_m, s.dt_s));
    (out, split)
}

pub fn measure(control: &str, map: &str, runs: usize, subjects: &[String]) -> Result<Phase, String> {
    let sample_m = sample_step(control).ok_or("the control has no readable trajectory")?;
    let (control_runs, control_split) = residuals(control, map, runs, "c");
    if control_runs.is_empty() {
        return Err("no run of the control produced a comparable file".into());
    }
    let control_zero = mode_of(&control_runs);
    let mut subs = Vec::new();
    for s in subjects {
        let (rr, split) = residuals(s, map, runs, "s");
        let delta = mode_of(&rr);
        subs.push(Subject { file: s.clone(), runs: rr, delta, split });
    }
    Ok(Phase { control_runs, control_split, control_zero, sample_m, subjects: subs })
}

fn fmt_runs(v: &[f64]) -> String {
    v.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(" ")
}

pub fn cmd(a: &[String]) {
    let map = crate::cli::need(a, "--map");
    let control = crate::cli::need(a, "--control");
    let runs: usize = crate::cli::flag(a, "--runs").and_then(|v| v.parse().ok()).unwrap_or(5);
    if runs < 3 {
        crate::cli::die(
            "--runs must be at least 3: the regenerator lands a tick out about one run in \
             seven, so a single run is not a measurement of anything.",
        );
    }
    let mut subjects: Vec<String> = Vec::new();
    let mut i = 0;
    while i < a.len() {
        if a[i].starts_with("--") {
            i += 2;
        } else {
            subjects.push(a[i].clone());
            i += 1;
        }
    }
    let p = match measure(&control, &map, runs, &subjects) {
        Ok(p) => p,
        Err(e) => crate::cli::die(e),
    };
    // A tick is a fifth of a recorded sample: the record is 50 ms, the physics
    // is 10 ms.
    let tick = p.sample_m / 5.0;
    println!("control      {}", control);
    println!("  {} runs     {}", p.control_runs.len(), fmt_runs(&p.control_runs));
    println!(
        "  ZERO       {:.4} m  ({:.2} ticks; one recorded sample is {:.3} m on this map)",
        p.control_zero,
        p.control_zero / tick,
        p.sample_m
    );
    if let Some((al, ac, dt)) = p.control_split {
        // IS IT A TIME SHIFT, OR IS IT A DIFFERENT RUN? A pure time shift is
        // displacement along the direction of travel and nothing across it.
        println!(
            "  split      {:+.4} m along the track, {:.4} m across it  =  {:+.2} ms of time",
            al, ac, dt * 1000.0
        );
        if ac < 0.05 * al.abs() && dt.abs() > 0.004 {
            println!(
                "             a PURE TIME SHIFT: the same curve, sampled {:.1} ms {}. The cross-track\n\
                 \x20            component is the position encoder's own floor, so there is no physics here\n\
                 \x20            to disagree about -- only which instant each sample is labelled with.",
                dt.abs() * 1000.0,
                if dt > 0.0 { "late" } else { "early" }
            );
        }
    }
    if p.control_zero > 0.5 * tick {
        println!(
            "  the instrument is NOT on the game's phase here: regenerating a recording the game\n\
             \x20 made itself moves it {:.4} m. Every subject below is judged against THAT, not\n\
             \x20 against zero -- an assumed zero of 0.000 convicts every honest file on this map.",
            p.control_zero
        );
    }
    for s in &p.subjects {
        println!("subject      {}", s.file);
        println!("  {} runs     {}", s.runs.len(), fmt_runs(&s.runs));
        println!("  delta      {:.4} m  ({:.2} ticks)", s.delta, s.delta / tick);
        if let Some((al, ac, dt)) = s.split {
            println!(
                "  split      {:+.4} m along, {:.4} m across  =  {:+.2} ms",
                al, ac, dt * 1000.0
            );
        }
        // WHICH WAY ROUND. `delta` is how far regenerating the subject MOVES
        // it, so it is small when the subject is already where this instrument
        // puts things and large when it is not:
        //
        //   delta ~ 0            the subject sits on the INSTRUMENT's tick
        //   delta ~ control_zero the subject sits on the GAME's tick
        //
        // and those two are the same statement only when the instrument's zero
        // is itself ~0. The first version of this function had the labels the
        // other way round -- it read a delta of 0.000 as "same phase as the
        // game" -- which is the mislabelled-operand failure exactly: the
        // arithmetic was right, the words on it were wrong, and a wrong label
        // gets corroborated rather than checked.
        let on_instrument = s.delta < 0.4 * tick;
        let on_game = (s.delta - p.control_zero).abs() < 0.4 * tick;
        let verdict = match (on_instrument, on_game) {
            // the instrument is on the game's tick: both statements coincide
            (true, true) => "on the game's tick, and so is this instrument -- nothing to choose",
            (true, false) => {
                "ON THIS INSTRUMENT'S TICK, WHICH IS NOT THE GAME'S. Some regenerator wrote this \
                 record, and it is one physics tick from the run the file claims. Regenerating it \
                 again reproduces the error; it cannot be repaired by this tool on this map."
            }
            (false, true) => {
                "on the GAME's tick -- correct as it stands. Regenerating it would MOVE IT OFF, \
                 because this instrument is a tick away on this map. Repair its input-echo bytes \
                 with `ghost tape sync-record`, which needs no engine, and leave the transform \
                 alone."
            }
            (false, false) => {
                "NEITHER this instrument's tick nor the game's -- read the runs above before \
                 concluding anything"
            }
        };
        println!("  verdict    {}", verdict);
    }
}
