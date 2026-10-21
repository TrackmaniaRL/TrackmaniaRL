//! `fk watch` -- arm the early-abort watchdog and measure whether it is worth
//! having.
//!
//! The whole point of aborting a doomed candidate early is throughput, and the
//! whole risk is that the abort is wrong. So the harness measures both at once,
//! on the same candidate set, in one pass:
//!
//! * **exactness** -- every candidate is run twice through the same fork
//!   server, once with predicates armed and once without, and once more as a
//!   full from-tick-0 validation. A candidate that does not trip must give
//!   the identical answer all three ways. Arming a watchdog may not perturb
//!   the physics.
//! * **false positives** -- for every candidate that DID trip, the unarmed run
//!   says what it would have done. A trip on a candidate that would have
//!   finished is a good line permanently hidden, and the rate is reported
//!   whether it is zero or not.
//! * **speedup** -- the wall time of the two passes over the identical
//!   candidate list, so the number includes the sampler's own overhead on the
//!   survivors rather than pretending it is free.
//! * **score safety** -- an aborted candidate loses its checkpoint count, so
//!   the search has to score it from `progress` instead. The harness checks the
//!   claim that makes that safe: the checkpoint count implied by progress is
//!   never HIGHER than the checkpoint count the candidate really reached.
//!
//! Plus the standing rule: an identity control in every batch. The reference
//! tape is run through both paths and must return the reference's own
//! millisecond, and must not trip anything.

use forkoracle::blind::{bounds_from, locate_blind};
use forkoracle::forksrv::{parse_result, rec_of, write_key, ForkServer, Rec};
use forkoracle::pred_core::Summary;
use forkoracle::pred::{outcome, parse_spec, Outcome, RefLineData, Watch};
use forkoracle::layout::{segments, Layout, Row, R_CLOCK, R_POS, R_QUAT, R_VEL, REC_LEN};
use std::path::{Path, PathBuf};
use std::time::Instant;
use crate::tape::Tape as Factory;
use forkoracle::inputs::{mutate, Inputs as State, OpSet, Rng as MRng};

pub struct Cfg {
    pub template: String,
    pub map: String,
    pub server: String,
    pub work: String,
    pub shim: String,
    pub refcsv: String,
    pub out: String,
    /// The trajectory `fk watch replay` reads. An INPUT, despite the old name.
    pub traj_in: String,
    pub tick: i64,
    pub ckpt: u64,
    pub specs: Vec<String>,
    pub segs: Vec<String>,
    pub n: usize,
    pub seed: u64,
    pub nops: i64,
    pub lo: usize,
    pub hi: usize,
    pub window: usize,
    pub ops: String,
    pub corridor: f32,
    pub ahead: i32,
    pub back: i32,
    pub every: u64,
    pub finishmargin: f32,
    pub fast: u32,
    pub reftime: i64,
    /// The state objective: a box and a key, for trying one against a measured
    /// trajectory before spending a search on it.
    pub gate: String,
    pub gate_key: String,
    /// The event clause: a thing that happens, and what to score after it.
    pub fire: String,
    pub fire_at: f32,
    pub fire_need: u32,
    pub fire_where: String,
    pub after_key: String,
    pub after_ticks: u32,
    pub after_from_end: bool,
}

fn parse(args: &[String]) -> Cfg {
    let mut c = Cfg {
        // No hardcoded template or map. The old defaults pointed at one
        // agent's scratch paths, so a missing flag ran a whole measurement
        // against somebody else's incumbent instead of failing.
        template: String::new(),
        map: String::new(),
        server: std::env::var("TM_SERVER").unwrap_or_else(|_| "/tmp/tmoracle/server".into()),
        work: crate::session::Engine::default_work().to_string_lossy().into(),
        shim: std::env::var("FK_SHIM")
            .ok()
            .or_else(|| crate::session::default_shim().map(|p| p.to_string_lossy().into()))
            .unwrap_or_default(),
        refcsv: String::new(),
        out: String::new(),
        traj_in: String::new(),
        tick: 60,
        ckpt: 0,
        specs: Vec::new(),
        segs: Vec::new(),
        n: 200,
        seed: 1,
        nops: 1,
        lo: 0,
        hi: usize::MAX,
        window: 0,
        // `local` is what `mix` was called before the operator set was named.
        ops: "local".into(),
        corridor: 40.0,
        ahead: 24,
        back: 8,
        every: 1,
        finishmargin: 10.0,
        fast: 1,
        reftime: 0,
        gate: String::new(),
        gate_key: String::new(),
        fire: String::new(),
        fire_at: 0.0,
        fire_need: 1,
        fire_where: String::new(),
        after_key: String::new(),
        after_ticks: 0,
        after_from_end: false,
    };
    let mut i = 0;
    while i < args.len() {
        let next = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i)
                .unwrap_or_else(|| panic!("flag {} needs a value", args[*i - 1]))
                .clone()
        };
        match args[i].as_str() {
            "--template" => c.template = next(&mut i),
            "--map" => c.map = next(&mut i),
            "--server" => c.server = next(&mut i),
            "--work" => c.work = next(&mut i),
            "--shim" => c.shim = next(&mut i),
            "--reference" => c.refcsv = next(&mut i),
            "--out" => c.out = next(&mut i),
            "--trajectory" => c.traj_in = next(&mut i),
            "--tick" => c.tick = next(&mut i).parse().unwrap(),
            "--ckpt" => c.ckpt = next(&mut i).parse().unwrap(),
            "--pred" => c.specs.push(next(&mut i)),
            "--seg" => c.segs.push(next(&mut i)),
            "--n" => c.n = next(&mut i).parse().unwrap(),
            "--seed" => c.seed = next(&mut i).parse().unwrap(),
            "--nops" => c.nops = next(&mut i).parse().unwrap(),
            "--lo" => c.lo = next(&mut i).parse().unwrap(),
            "--hi" => c.hi = next(&mut i).parse().unwrap(),
            "--window" => c.window = next(&mut i).parse().unwrap(),
            "--ops" => c.ops = next(&mut i),
            "--corridor" => c.corridor = next(&mut i).parse().unwrap(),
            "--ahead" => c.ahead = next(&mut i).parse().unwrap(),
            "--back" => c.back = next(&mut i).parse().unwrap(),
            "--every" => c.every = next(&mut i).parse().unwrap(),
            "--finishmargin" => c.finishmargin = next(&mut i).parse().unwrap(),
            "--fast" => c.fast = next(&mut i).parse().unwrap(),
            "--reference-ms" => c.reftime = next(&mut i).parse().unwrap(),
            "--gate" => c.gate = next(&mut i),
            "--gate-key" => c.gate_key = next(&mut i),
            "--fire" => c.fire = next(&mut i),
            "--fire-at" => c.fire_at = next(&mut i).parse().unwrap(),
            "--fire-need" => c.fire_need = next(&mut i).parse().unwrap(),
            "--fire-where" => c.fire_where = next(&mut i),
            "--after-ticks" => c.after_ticks = next(&mut i).parse().unwrap(),
            "--after-from" => c.after_from_end = next(&mut i) == "end",
            "--after-key" => c.after_key = next(&mut i),
            x => crate::die(format!(
                "fk watch: unknown flag {:?}. A flag this command does not use is \
                 a measurement you did not ask for, so it is an error rather than \
                 something to ignore.",
                x
            )),
        }
        i += 1;
    }
    c
}

/// Read the first four columns (`time_ms,x,y,z`) of a trajectory CSV and
/// resample onto tape ticks. `fk btraj` writes one row per 10 ms tick, so this
/// is a re-index rather than a resample; ticks before the first row (the
/// standing start) clamp to it.
pub fn ref_from_csv(path: &str, start_offset_ms: i32, nticks: usize) -> Result<RefLineData, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut pts: Vec<Option<[f64; 3]>> = vec![None; nticks];
    let mut nrow = 0;
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.trim().split(',').collect();
        if f.len() < 4 {
            continue;
        }
        let (ms, x, y, z) = match (
            f[0].parse::<i64>(),
            f[1].parse::<f64>(),
            f[2].parse::<f64>(),
            f[3].parse::<f64>(),
        ) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            _ => continue,
        };
        let t = (ms - start_offset_ms as i64) / 10;
        if t >= 0 && (t as usize) < nticks {
            pts[t as usize] = Some([x, y, z]);
            nrow += 1;
        }
    }
    if nrow < 10 {
        return Err(format!("{}: only {} usable rows", path, nrow));
    }
    // fill the holes: clamp at the ends, linearly interpolate inside
    let first = pts.iter().position(|p| p.is_some()).unwrap();
    let last = pts.iter().rposition(|p| p.is_some()).unwrap();
    for i in 0..first {
        pts[i] = pts[first];
    }
    for i in last + 1..nticks {
        pts[i] = pts[last];
    }
    let mut i = first;
    while i <= last {
        if pts[i].is_some() {
            i += 1;
            continue;
        }
        let a = i - 1;
        let mut b = i;
        while pts[b].is_none() {
            b += 1;
        }
        let (pa, pb) = (pts[a].unwrap(), pts[b].unwrap());
        for k in a + 1..b {
            let u = (k - a) as f64 / (b - a) as f64;
            pts[k] = Some([
                pa[0] + u * (pb[0] - pa[0]),
                pa[1] + u * (pb[1] - pa[1]),
                pa[2] + u * (pb[2] - pa[2]),
            ]);
        }
        i = b + 1;
    }
    let flat: Vec<[f64; 3]> = pts.into_iter().map(|p| p.unwrap()).collect();
    Ok(RefLineData::from_points(&flat))
}

fn write_cand(f: &Factory, steer: &[u8], accel: &[u8], brake: &[u8], path: &Path) {
    f.write_candidate(steer, accel, brake, path)
        .unwrap_or_else(|e| crate::die(e));
}

fn tail_recs(steer: &[u8], accel: &[u8], brake: &[u8], from: usize) -> Vec<Rec> {
    (from..steer.len())
        .map(|t| rec_of(steer[t], accel[t], brake[t]))
        .collect()
}

/// The checkpoint-count lower bound implied by progress along the line.
///
/// `cp_s[k]` is the arclength at which the reference crossed checkpoint k+1.
pub fn cps_lower(progress: f32, cp_s: &[f32]) -> u32 {
    cp_s.iter().filter(|&&s| progress >= s).count() as u32
}

struct Setup {
    clock0: i64,
    segs: Vec<(u64, u32)>,
    f: Factory,
    srv: ForkServer,
    gt: crate::oracle::Batch,
    layout: Layout,
    watch: Watch,
    boundary: usize,
    ref_time: Option<i64>,
    cp_s: Vec<f32>,
    work: PathBuf,
}

/// Everything that has to happen before a single candidate can be judged:
/// start the server, calibrate the resume boundary, locate the car's state,
/// build the reference line, arm the predicates. Every step is a hard abort.
fn setup(c: &Cfg) -> Setup {
    let work = PathBuf::from(&c.work);
    let _ = std::fs::create_dir_all(&work);
    let f = Factory::load(&c.template).unwrap_or_else(|e| crate::die(e));
    let n = f.n();
    let ckpt = if c.ckpt > 0 {
        c.ckpt
    } else {
        crate::session::clock_for_tick(c.tick, f.start_offset_ms)
    };
    let refp = work.join("ref.Ghost.Gbx");
    write_cand(&f, &f.steer, &f.accel, &f.brake, &refp);
    let key = work.join("key.bin");
    write_key(&key, &f.steer);

    let gt = crate::oracle::Batch::new(Path::new(&c.server), Path::new(&c.map), "watch-gt");
    let ref_time = gt.times(&[refp.clone()]).first().and_then(|r| r.time_ms);
    println!(
        "tape {}: {} ticks, start_offset {} ms, validated {:?} ms",
        c.template, n, f.start_offset_ms, ref_time
    );

    // checkpoint arclengths, from the segment maps: a segment map moves the
    // finish to checkpoint k, so validating the reference on it says exactly
    // when the reference crossed that checkpoint.
    let mut cp_times: Vec<i64> = Vec::new();
    for m in &c.segs {
        let w = crate::oracle::Batch::new(
            Path::new(&c.server),
            Path::new(m),
            &format!("watch-seg{}", cp_times.len()),
        );
        match w.times(&[refp.clone()]).first().and_then(|r| r.time_ms) {
            Some(t) => cp_times.push(t),
            None => panic!("reference does not finish segment map {}", m),
        }
    }

    let mut srv = ForkServer::start(
        &work.join("srv"),
        Path::new(&c.server),
        Path::new(&c.map),
        &refp,
        &key,
        Path::new(&c.shim),
        ckpt,
    )
    .unwrap_or_else(|e| panic!("fork server failed: {}", e));
    println!(
        "fork server up: input array {:#x}, checkpoint at lroundf #{}",
        srv.base, srv.clock
    );
    let probe = srv
        .probe_tick()
        .unwrap_or_else(|e| panic!("ABORT: boundary probe failed ({})", e));
    let engine = crate::session::Engine {
        server: PathBuf::from(&c.server),
        map: PathBuf::from(&c.map),
        shim: PathBuf::from(&c.shim),
        work: work.clone(),
        work_is_temporary: false,
    };
    let boundary = crate::cmd::server::calibrate_boundary(&mut srv, &f, &engine, probe)
        .unwrap_or_else(|e| crate::abort(e));
    println!(
        "boundary tick {} (probe {}) = race {} ms",
        boundary,
        probe,
        boundary as i64 * 10 + f.start_offset_ms as i64
    );

    // the reference line, and the bounds the blind locate needs
    let refline = if c.refcsv.is_empty() {
        RefLineData::default()
    } else {
        ref_from_csv(&c.refcsv, f.start_offset_ms, n).unwrap_or_else(|e| panic!("{}", e))
    };
    let bounds = if refline.n > 0 {
        let rows: Vec<Row> = (0..refline.n)
            .map(|i| Row {
                time_ms: 0,
                x: refline.xyz[3 * i] as f64,
                y: refline.xyz[3 * i + 1] as f64,
                z: refline.xyz[3 * i + 2] as f64,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
                qx: 0.0,
                qy: 0.0,
                qz: 0.0,
                qw: 0.0,
                wetness: 0.0,
            })
            .collect();
        bounds_from(&rows, 200.0)
    } else {
        (-64000.0, 64000.0, -1000.0, 4000.0, -64000.0, 64000.0)
    };
    let lrecs = tail_recs(&f.steer, &f.accel, &f.brake, probe);
    let layout = locate_blind(&mut srv, probe, &lrecs, f.start_offset_ms, c.every.max(1), bounds, true)
        .unwrap_or_else(|e| panic!("ABORT: {}", e));
    println!(
        "state located: position {:#x}, clock {:#x} (bias {:+} ms), self-consistency {:.3} m/s",
        layout.pos, layout.clock, layout.clock_bias, layout.rms
    );

    let cp_s: Vec<f32> = cp_times
        .iter()
        .map(|&t| {
            let tick = ((t - f.start_offset_ms as i64) / 10).max(0) as usize;
            refline.s_at_tick(tick)
        })
        .collect();
    if !cp_s.is_empty() {
        println!(
            "checkpoint times {:?} ms -> arclengths {:?} m (line is {:.0} m long)",
            cp_times,
            cp_s.iter().map(|s| s.round()).collect::<Vec<_>>(),
            refline.s_at_tick(usize::MAX)
        );
    }

    let mut watch = Watch::new();
    watch.corridor = c.corridor;
    watch.ahead = c.ahead;
    watch.back = c.back;
    watch.finish_s = finish_s(&refline, ref_time, f.start_offset_ms, c.finishmargin);
    watch.fast = c.fast;
    watch.refline = refline;
    for s in &c.specs {
        watch
            .preds
            .push(parse_spec(s).unwrap_or_else(|e| panic!("{}", e)));
    }
    print!("{}", watch.describe());
    let clock0 = layout.clock_bias + f.start_offset_ms as i64;
    let segs = segments(&layout);
    let ack = srv.arm(&watch.arm_payload(
        clock0,
        R_CLOCK as u32,
        R_QUAT as u32,
        R_POS as u32,
        R_VEL as u32,
        REC_LEN as u32,
        &segs,
    ));
    println!("arm: {}", ack.trim());
    if !ack.starts_with("ARMED") {
        panic!("the shim refused to arm: {}", ack);
    }
    Setup {
        clock0,
        segs,
        f,
        srv,
        gt,
        layout,
        watch,
        boundary,
        ref_time,
        cp_s,
        work,
    }
}

/// One candidate, generated the way the search generates them.
fn make(f: &Factory, rng: &mut MRng, lo: usize, hi: usize, nops: i64, ops: OpSet) -> State {
    let mut s = State::from_arrays(&f.steer, &f.accel, &f.brake);
    let k = if nops < 0 {
        rng.range(1, -nops) as usize
    } else {
        nops as usize
    };
    for _ in 0..k {
        mutate(&mut s, rng, lo, hi, ops);
    }
    s
}

/// Three verbs, because they are three different controls and each was
/// reachable before only through `--mode`, which is a flag that changes what
/// the command IS.
///
///   measure  armed vs unarmed vs a full validation, on one candidate set
///   replay   the same evaluator, no server, against a trajectory CSV --
///            the cross-check that the in-child judge and the out-of-process
///            one are the same judge
///   paths    do the two in-child sampling paths return identical summaries?
///            The cheap one reads four bytes per `lroundf` call instead of
///            gathering the whole record, on the argument that the state
///            cannot change in between. That is an argument; this measures it.
pub fn run(args: &[String]) -> Result<(), String> {
    let c = parse(&args[1.min(args.len())..]);
    if c.template.is_empty() || c.map.is_empty() {
        return Err("fk watch needs --template FILE and --map FILE".into());
    }
    if c.shim.is_empty() {
        return Err("no --shim: pass one or set FK_SHIM".into());
    }
    match args.first().map(|s| s.as_str()).unwrap_or("") {
        "measure" => audit(&c),
        "replay" => offline(&c),
        "paths" => equiv(&c),
        m => return Err(format!("fk watch <measure|replay|paths>, not {:?}", m)),
    }
    Ok(())
}

/// Evaluate the armed predicates against a trajectory CSV, with no server at
/// all. This is the cross-check on the in-child evaluator: the same core, the
/// same rows, so the two must agree on tick and predicate.
fn offline(c: &Cfg) {
    let f = Factory::load(&c.template).unwrap_or_else(|e| crate::die(e));
    let n = f.n();
    let refline = if c.refcsv.is_empty() {
        RefLineData::default()
    } else {
        ref_from_csv(&c.refcsv, f.start_offset_ms, n).unwrap()
    };
    let mut watch = Watch::new();
    watch.corridor = c.corridor;
    watch.ahead = c.ahead;
    watch.back = c.back;
    watch.refline = refline;
    for s in &c.specs {
        watch.preds.push(parse_spec(s).unwrap());
    }
    if !c.gate.is_empty() {
        watch.gate = forkoracle::pred::parse_gate(&c.gate, &c.gate_key)
            .unwrap_or_else(|e| crate::die(e));
    }
    if !c.fire.is_empty() {
        watch.fire =
            forkoracle::pred::parse_fire(
                &c.fire, c.fire_at, c.fire_need, &c.fire_where, &c.after_key, c.after_ticks,
                c.after_from_end,
            )
                .unwrap_or_else(|e| crate::die(e));
    }
    // `--trajectory`, not `--out`: this verb READS a trajectory. The flag was
    // called `--out` because the file it evaluates is usually one another
    // command wrote, which is a fact about a workflow and not about this
    // command's arguments.
    let path = if c.traj_in.is_empty() {
        crate::die("fk watch replay needs --trajectory CSV (the trajectory to evaluate)")
    } else {
        c.traj_in.clone()
    };
    watch.finish_s = finish_s(&watch.refline.clone(), if c.reftime > 0 { Some(c.reftime) } else { None }, f.start_offset_ms, c.finishmargin);
    let sum = eval_csv(&watch, &path, f.start_offset_ms, n);
    // THE SAMPLE RATE IS PART OF EVERY ANSWER HERE, and it is not the fork's.
    //
    // This verb reads a trajectory CSV. Recorded telemetry is on a 50 ms grid;
    // the fork child evaluates every 10 ms tick. So a `--fire-need 3` calibrated
    // here is a 150 ms run, and the SAME FLAG in a search is a 30 ms run -- and
    // `dspeed` and `domega` are per-sample differences, so their VALUES are not
    // comparable between the two at all, only the fired/not-fired verdict is.
    // Reported by 284238 after confirming the load detector across both.
    if watch.fire.armed {
        let step = sample_step_ms(&path);
        println!(
            "  NOTE: this trajectory samples every {} ms; the fork evaluates every 10 ms. \
             --fire-need {} is {} ms here and {} ms in a search, and per-sample terms \
             (dspeed, domega) are not comparable between the two -- only fired/not-fired is.",
            step,
            watch.fire.need,
            step * watch.fire.need as i64,
            10 * watch.fire.need as i64
        );
    }
    println!(
        "{}: {} ticks, trip {} at tick {} value {:.3}, progress {:.1} m, travelled {:.1} m, off_max {:.2} m",
        path,
        sum.nticks,
        watch.name_of(sum.trip_pred),
        sum.trip_tick,
        sum.trip_value,
        sum.progress,
        sum.travelled,
        sum.off_max
    );
    // THE STATE OBJECTIVE, evaluated offline against this trajectory: what a
    // gate search would have scored this tape, without a server.
    if watch.fire.armed {
        if sum.fire_tick >= 0 {
            println!(
                "  fire: at tick {} ({:+.2}) at ({:.2}, {:.2}, {:.2}); {} run(s), {}{}",
                sum.fire_tick,
                sum.fire_value,
                sum.fire_pos[0], sum.fire_pos[1], sum.fire_pos[2],
                sum.fire_runs,
                if sum.fire_end_tick >= 0 {
                    format!("ended at tick {} ({} ticks)", sum.fire_end_tick, sum.fire_end_tick - sum.fire_tick + 1)
                } else {
                    "still holding when the run ended".to_string()
                },
                if sum.after_tick >= 0 {
                    format!("; after {:+.4} at tick {}", sum.after_key, sum.after_tick)
                } else {
                    String::new()
                }
            );
        } else {
            println!("  fire: the event never fired");
        }
    }
    if watch.gate.armed {
        if sum.gate_tick >= 0 {
            println!(
                "  gate: key {:+.4} at tick {} -- pos ({:.2}, {:.2}, {:.2}) vel ({:.2}, {:.2}, {:.2}) \
                 quat ({:.4}, {:.4}, {:.4}, {:.4})",
                sum.gate_key,
                sum.gate_tick,
                sum.gate_pos[0], sum.gate_pos[1], sum.gate_pos[2],
                sum.gate_vel[0], sum.gate_vel[1], sum.gate_vel[2],
                sum.gate_quat[0], sum.gate_quat[1], sum.gate_quat[2], sum.gate_quat[3]
            );
        } else if sum.gate_miss.is_finite() {
            println!("  gate: never entered; closest approach {:.3} m", sum.gate_miss);
        } else {
            println!("  gate: never entered, and never came within measuring distance");
        }
    }
}

/// Run the shared evaluator over a trajectory CSV.
pub fn eval_csv(watch: &Watch, path: &str, start_offset_ms: i32, nticks: usize) -> Summary {
    let text = std::fs::read_to_string(path).unwrap();
    let mut ev = forkoracle::pred_core::Eval::ZERO;
    ev.reset();
    ev.np = watch.preds.len();
    for (i, p) in watch.preds.iter().enumerate() {
        ev.preds[i] = p.pred;
    }
    ev.finish_s = watch.finish_s;
    // The state objective too, so a key can be tried against a measured
    // trajectory with no server at all -- the same `Eval`, the same program.
    ev.gate = watch.gate;
    ev.fire = watch.fire;
    ev.rl = forkoracle::pred_core::RefLine {
        n: watch.refline.n,
        xyz: watch.refline.xyz.as_ptr(),
        s: watch.refline.s.as_ptr(),
        corridor: watch.corridor,
        ahead: watch.ahead,
        back: watch.back,
    };
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.trim().split(',').collect();
        if f.len() < 9 {
            continue;
        }
        let g = |i: usize| f[i].parse::<f32>().unwrap_or(0.0);
        let ms: i64 = match f[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let tick = ((ms - start_offset_ms as i64) / 10) as i32;
        if tick < 0 || tick as usize >= nticks {
            continue; // past the end of the tape: not part of this run
        }
        // `time_ms,x,y,z,speed_kmh,speed_ms,vx,vy,vz,yaw,pitch,roll,qx,qy,qz,qw,...`
        // -- the format `fk btraj` and `tmtraj decode --csv` both write. The
        // quaternion is only present in the wider form; without it the gate's
        // body-frame terms read as an unrotated car, which is why a key that
        // uses them wants the wide CSV.
        let quat = if f.len() >= 16 {
            [g(15), g(12), g(13), g(14)]
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };
        if ev.feed(tick, [g(1), g(2), g(3)], [g(6), g(7), g(8)], quat) >= 0 {
            break;
        }
    }
    ev.sum
}

struct Row1 {
    obs: Outcome,
    idx: usize,
    plain_time: Option<i64>,
    plain_cps: Option<u32>,
    w: Outcome,
    full_time: Option<i64>,
    full_cps: Option<u32>,
}

fn audit(c: &Cfg) {
    let mut s = setup(c);
    let n = s.f.n();
    let from = s.boundary;
    let lo = c.lo.max(from);
    let hi = c.hi.min(n);
    println!(
        "mutating ticks [{},{}) of {}; {} candidates, seed {}, ops {}",
        lo, hi, n, c.n, c.seed, c.ops
    );

    // ---- identity control: the reference tape itself, both ways
    let idrecs = tail_recs(&s.f.steer, &s.f.accel, &s.f.brake, from);
    let (idt, _) = parse_result(&s.srv.run(from, &idrecs));
    let (j, b) = s.srv.run_watched(from, &idrecs);
    let ido = outcome(&j, &b);
    println!(
        "IDENTITY CONTROL  unarmed {:?}  armed {:?}  reference {:?}  {}  | trip {} | progress {:.1} m of {:.1} m, {} ticks",
        idt,
        ido.time,
        s.ref_time,
        if idt == s.ref_time && ido.time == s.ref_time && ido.tripped().is_none() {
            "PASS"
        } else {
            "FAIL"
        },
        s.watch.name_of(ido.sum.map(|x| x.trip_pred).unwrap_or(-1)),
        ido.progress(),
        s.watch.refline.s_at_tick(usize::MAX),
        ido.sum.map(|x| x.nticks).unwrap_or(0),
    );

    // ---- the candidate set
    let mut rng = MRng::new(c.seed);
    let mut cands: Vec<(State, PathBuf)> = Vec::with_capacity(c.n);
    let cdir = s.work.join("cand");
    let _ = std::fs::create_dir_all(&cdir);
    for i in 0..c.n {
        let (l, h) = if c.window > 0 {
            // the search mutates inside a sliding window, not the whole tape
            let nwin = ((hi.saturating_sub(lo)).saturating_sub(c.window) / c.window.max(1)).max(1);
            let k = i % nwin;
            (lo + k * c.window, (lo + (k + 1) * c.window).min(hi))
        } else {
            (lo, hi)
        };
        let st = make(&s.f, &mut rng, l, h, c.nops, opset(&c.ops));
        let p = cdir.join(format!("c{:04}.Ghost.Gbx", i));
        write_cand(&s.f, &st.steer_u8(), &st.gas_u8(), &st.brake_u8(), &p);
        cands.push((st, p));
    }

    // ---- HOW FAR IS THIS CANDIDATE SET FROM THE REFERENCE?
    //
    // Every exactness number the fork server has ever produced is a number
    // about a REGIME, not about the fork: it was exact on 4700 of 4700
    // candidates that perturbed a human reference by a few ticks late in the
    // run, and it reported 312 finishes out of 312 that were not there on tapes
    // that differ from their template early or wholesale. Nothing inside a fork
    // can see which regime it is in.
    //
    // So a harness that reports "0 false positives" without saying how far its
    // candidates were from the reference has reported a number that cannot be
    // applied to anything. `Distance` (first differing tick, how many differ,
    // largest steering move) is what makes the number transferable.
    let reference = State::from_arrays(&s.f.steer, &s.f.accel, &s.f.brake);
    let dists: Vec<forkoracle::inputs::Distance> =
        cands.iter().map(|(st, _)| st.distance_from(&reference)).collect();
    {
        let firsts: Vec<usize> = dists.iter().filter_map(|d| d.first_diff_tick).collect();
        let mut diffs: Vec<usize> = dists.iter().map(|d| d.diff_ticks).collect();
        diffs.sort_unstable();
        println!(
            "DISTANCE FROM THE REFERENCE  earliest divergence tick {} (race {}), \
             median {} of {} ticks differ, worst {}; {} candidates identical to the reference",
            firsts.iter().min().map(|t| t.to_string()).unwrap_or("-".into()),
            firsts
                .iter()
                .min()
                .map(|t| crate::secs(*t as i64 * 10 + s.f.start_offset_ms as i64))
                .unwrap_or("-".into()),
            diffs.get(diffs.len() / 2).copied().unwrap_or(0),
            n,
            diffs.last().copied().unwrap_or(0),
            dists.iter().filter(|d| d.first_diff_tick.is_none()).count()
        );
        println!(
            "  Every number below is about THAT regime. The fork is exact for late \
             perturbations of a human seed and lied on 312 of 312 outside it."
        );
    }

    // ---- pass 1: no watchdog at all -- the honest baseline
    let t0 = Instant::now();
    let mut plain: Vec<(Option<i64>, Option<u32>)> = Vec::with_capacity(c.n);
    for (st, _) in &cands {
        let recs = tail_recs(&st.steer_u8(), &st.gas_u8(), &st.brake_u8(), from);
        plain.push(parse_result(&s.srv.run(from, &recs)));
    }
    let t_plain = t0.elapsed().as_secs_f64();

    // ---- pass 2: the watchdog OBSERVING but with nothing armed.
    //
    // This is the arm the search's own head-to-head uses as its control: same
    // per-tick reading, same progress measure, no aborts. It isolates what the
    // observation costs from what the abort saves, and it gives every
    // candidate the progress it would have reached if it had never been cut
    // short -- which is what makes the scoring invariant measurable rather
    // than merely argued.
    let armed_preds = std::mem::take(&mut s.watch.preds);
    rearm(&mut s);
    let t1 = Instant::now();
    let mut obs: Vec<Outcome> = Vec::with_capacity(c.n);
    for (st, _) in &cands {
        let recs = tail_recs(&st.steer_u8(), &st.gas_u8(), &st.brake_u8(), from);
        let (j, b) = s.srv.run_watched(from, &recs);
        obs.push(outcome(&j, &b));
    }
    let t_obs = t1.elapsed().as_secs_f64();

    // ---- pass 3: predicates armed
    s.watch.preds = armed_preds;
    rearm(&mut s);
    let t1 = Instant::now();
    let mut watched: Vec<Outcome> = Vec::with_capacity(c.n);
    for (st, _) in &cands {
        let recs = tail_recs(&st.steer_u8(), &st.gas_u8(), &st.brake_u8(), from);
        let (j, b) = s.srv.run_watched(from, &recs);
        watched.push(outcome(&j, &b));
    }
    let t_watch = t1.elapsed().as_secs_f64();

    // ---- ground truth: a full validation of every candidate from tick 0
    let files: Vec<PathBuf> = cands.iter().map(|x| x.1.clone()).collect();
    let t2 = Instant::now();
    let gtr = s.gt.times(&files);
    let t_full = t2.elapsed().as_secs_f64();
    let mut gtmap = std::collections::HashMap::new();
    for r in &gtr {
        gtmap.insert(r.file.clone(), (r.time_ms, r.cps));
    }

    let rows: Vec<Row1> = (0..c.n)
        .map(|i| {
            let name = cands[i]
                .1
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            let g = gtmap.get(&name).cloned().unwrap_or((None, None));
            Row1 {
                idx: i,
                plain_time: plain[i].0,
                plain_cps: plain[i].1,
                w: watched[i].clone(),
                obs: obs[i].clone(),
                full_time: g.0,
                full_cps: g.1,
            }
        })
        .collect();

    // ---- 1. exactness
    let mut n_trip = 0;
    let mut same_armed = 0;
    let mut diff_armed = 0;
    let mut fork_vs_full_bad = 0;
    let mut obs_vs_plain_bad = 0;
    let mut nosum = 0;
    for r in &rows {
        if r.w.sum.is_none() {
            nosum += 1;
        }
        // the observation-only pass must be invisible: same answer as no
        // watchdog at all, for every candidate
        if r.obs.time != r.plain_time
            || (r.obs.time.is_none() && r.obs.cps != r.plain_cps)
        {
            obs_vs_plain_bad += 1;
            if obs_vs_plain_bad <= 5 {
                println!(
                    "  OBSERVING PERTURBS c{:04}: watched {:?}/cp{:?}  unwatched {:?}/cp{:?}",
                    r.idx, r.obs.time, r.obs.cps, r.plain_time, r.plain_cps
                );
            }
        }
        if r.w.tripped().is_some() {
            n_trip += 1;
            continue;
        }
        let same = r.w.time == r.plain_time
            && (r.w.time.is_some() || r.w.cps == r.plain_cps);
        if same {
            same_armed += 1;
        } else {
            diff_armed += 1;
            if diff_armed <= 5 {
                println!(
                    "  ARMED DIFFERS c{:04}: armed {:?}/cp{:?}  unarmed {:?}/cp{:?}",
                    r.idx, r.w.time, r.w.cps, r.plain_time, r.plain_cps
                );
            }
        }
        if r.plain_time != r.full_time
            || (r.plain_time.is_none() && r.plain_cps != r.full_cps)
        {
            fork_vs_full_bad += 1;
            if fork_vs_full_bad <= 5 {
                println!(
                    "  FORK vs FULL c{:04}: fork {:?}/cp{:?}  full {:?}/cp{:?}",
                    r.idx, r.plain_time, r.plain_cps, r.full_time, r.full_cps
                );
            }
        }
    }
    println!(
        "\nEXACTNESS  {} of {} did not trip: {} identical armed vs unarmed, {} DIFFER; \
         {} disagree with the full validation; {} perturbed by watching alone; {} missing summaries",
        c.n - n_trip,
        c.n,
        same_armed,
        diff_armed,
        fork_vs_full_bad,
        obs_vs_plain_bad,
        nosum
    );

    // ---- 2. false positives
    let mut fin_unarmed = 0;
    let mut fp_finish: Vec<(usize, i64, &str, i32)> = Vec::new();
    let mut trip_by: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    for r in &rows {
        if r.plain_time.is_some() {
            fin_unarmed += 1;
        }
        if let Some((pi, tick, _v)) = r.w.tripped() {
            let nm = s.watch.name_of(pi).to_string();
            let e = trip_by.entry(nm.clone()).or_insert((0, 0));
            e.0 += 1;
            if r.plain_time.is_some() {
                e.1 += 1;
                fp_finish.push((r.idx, r.plain_time.unwrap(), s.watch.name_of(pi), tick));
            }
        }
    }
    println!(
        "TRIPS      {} of {} candidates aborted ({:.1}%); {} of {} would have finished",
        n_trip,
        c.n,
        100.0 * n_trip as f64 / c.n as f64,
        fp_finish.len(),
        fin_unarmed
    );
    for (nm, (cnt, fps)) in &trip_by {
        println!(
            "  {:<12} fired {:4}   of which would have finished: {}",
            nm, cnt, fps
        );
    }
    println!(
        "FALSE POSITIVES  {} / {} candidates ({:.2}% of all, {:.2}% of finishers)",
        fp_finish.len(),
        c.n,
        100.0 * fp_finish.len() as f64 / c.n as f64,
        if fin_unarmed > 0 {
            100.0 * fp_finish.len() as f64 / fin_unarmed as f64
        } else {
            0.0
        }
    );
    if !fp_finish.is_empty() {
        let best = s.ref_time.unwrap_or(0);
        let mut better = 0;
        for (i, t, nm, tick) in fp_finish.iter().take(20) {
            println!(
                "  c{:04} would finish {} ms ({:+} vs incumbent), killed by {} at tick {}",
                i,
                t,
                t - best,
                nm,
                tick
            );
            if *t < best {
                better += 1;
            }
        }
        println!(
            "  of the {} false positives, {} were FASTER than the incumbent",
            fp_finish.len(),
            fp_finish.iter().filter(|(_, t, _, _)| *t < best).count()
        );
        let _ = better;
    }

    // ---- 3. scoring safety
    //
    // The search scores a DNF by `progress` -- how far along the incumbent's
    // line it got -- because the validator's own DNF signal on this map is
    // nearly binary (see below) and, more importantly, because progress is
    // computed IDENTICALLY for an aborted and a completed run. That makes the
    // invariant checkable rather than argued: aborting truncates the run, so
    // an aborted candidate's progress must be <= the progress the very same
    // candidate reaches when nothing is armed. If that holds, arming
    // predicates can only ever LOWER a candidate's score, so a dead candidate
    // can never displace a live one.
    let mut viol = 0;
    let mut lost: Vec<f32> = Vec::new();
    for r in &rows {
        let (pa, pf) = (r.w.progress(), r.obs.progress());
        if pa > pf + 1e-3 {
            viol += 1;
            if viol <= 5 {
                println!(
                    "  PROGRESS VIOLATION c{:04}: aborted {:.2} m > unaborted {:.2} m",
                    r.idx, pa, pf
                );
            }
        }
        if r.w.tripped().is_some() {
            lost.push(pf - pa);
        }
    }
    lost.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "SCORE SAFETY  progress(aborted) <= progress(same candidate, nothing armed): {} of {}, {} VIOLATIONS",
        c.n - viol,
        c.n,
        viol
    );
    if !lost.is_empty() {
        let mean: f32 = lost.iter().sum::<f32>() / lost.len() as f32;
        println!(
            "  progress given up by aborting: mean {:.1} m, median {:.1} m, worst {:.1} m (line is {:.0} m)",
            mean,
            lost[lost.len() / 2],
            lost[lost.len() - 1],
            s.watch.refline.s_at_tick(usize::MAX)
        );
    }
    // The same check against the validator's checkpoint count, for the
    // candidates where it says anything at all. On map 2 a DNF returns either
    // "reached some checkpoints (2)" or the information-free "wrong simu",
    // which is exactly why the score does not lean on it.
    if !s.cp_s.is_empty() {
        let mut checked = 0;
        let mut cviol = 0;
        for r in &rows {
            let truth = match (r.obs.time, r.obs.cps) {
                (Some(_), _) => 3,
                (None, Some(k)) if k > 0 => k,
                _ => continue, // "wrong simu": no checkpoint information at all
            };
            checked += 1;
            if cps_lower(r.w.progress(), &s.cp_s) > truth {
                cviol += 1;
            }
        }
        println!(
            "  cross-check: cps_lower(progress) <= validator checkpoints on the {} candidates it reports for: {} violations",
            checked, cviol
        );
    }

    // ---- 4. throughput
    let ticks_total = (n - from) as f64;
    let mut sum_ticks = 0.0;
    let mut n_ab = 0.0;
    for r in &rows {
        if let Some((_, tick, _)) = r.w.tripped() {
            sum_ticks += (tick as f64 - from as f64).max(0.0);
            n_ab += 1.0;
        }
    }
    println!(
        "\nTHROUGHPUT  no watchdog {:.2} ms/cand | watching, nothing armed {:.2} ms/cand | armed {:.2} ms/cand | full validation {:.2} ms/cand",
        1000.0 * t_plain / c.n as f64,
        1000.0 * t_obs / c.n as f64,
        1000.0 * t_watch / c.n as f64,
        1000.0 * t_full / c.n as f64
    );
    println!(
        "  speedup from aborting: {:.3}x vs the observing control, {:.3}x vs no watchdog at all, {:.3}x vs full validation",
        t_obs / t_watch,
        t_plain / t_watch,
        t_full / t_watch
    );
    if n_ab > 0.0 {
        println!(
            "  aborted candidates stopped after {:.0} of {:.0} tail ticks on average ({:.0}% of the tail)",
            sum_ticks / n_ab,
            ticks_total,
            100.0 * sum_ticks / n_ab / ticks_total
        );
    }
    let fin_armed = rows.iter().filter(|r| r.w.time.is_some()).count();
    println!(
        "  finishers: {} unarmed, {} armed; DNFs {} / {}",
        fin_unarmed,
        fin_armed,
        c.n - fin_unarmed,
        c.n
    );

    // ---- optional per-candidate dump
    if !c.out.is_empty() {
        let mut o = String::from(
            "idx,plain_time,plain_cps,armed_time,armed_cps,trip,trip_tick,trip_value,progress,travelled,last_tick,full_time,full_cps\n",
        );
        for r in &rows {
            let (tp, tt, tv) = match r.w.tripped() {
                Some((p, t, v)) => (s.watch.name_of(p).to_string(), t, v),
                None => ("-".to_string(), -1, 0.0),
            };
            o.push_str(&format!(
                "{},{},{},{},{},{},{},{:.3},{:.2},{:.2},{},{},{}\n",
                r.idx,
                r.plain_time.map(|v| v.to_string()).unwrap_or_default(),
                r.plain_cps.map(|v| v.to_string()).unwrap_or_default(),
                r.w.time.map(|v| v.to_string()).unwrap_or_default(),
                r.w.cps.map(|v| v.to_string()).unwrap_or_default(),
                tp,
                tt,
                tv,
                r.w.progress(),
                r.w.travelled(),
                r.w.last_tick(),
                r.full_time.map(|v| v.to_string()).unwrap_or_default(),
                r.full_cps.map(|v| v.to_string()).unwrap_or_default(),
            ));
        }
        std::fs::write(&c.out, o).unwrap();
        println!("wrote {}", c.out);
    }

    let _ = s.layout;
    s.srv.quit();
}

/// Where on the reference line the FINISH is, in metres of arclength, minus a
/// margin. Past that point the candidate has banked its time and the watchdog
/// must not fire: the engine keeps simulating for a few hundred milliseconds
/// after the finish, and the reference line's own tail runs past it too.
pub fn finish_s(rl: &RefLineData, ref_time: Option<i64>, start_offset_ms: i32, margin: f32) -> f32 {
    match ref_time {
        Some(t) if rl.n > 0 => {
            let tick = ((t - start_offset_ms as i64) / 10).max(0) as usize;
            (rl.s_at_tick(tick) - margin).max(1.0)
        }
        _ => 0.0,
    }
}

/// Are the two in-child sampling paths the same judge?
///
/// The cheap path reads four bytes per `lroundf` call instead of gathering the
/// whole record, on the argument that the state cannot change between the last
/// call of one tick and the first call of the next. That is an argument, not a
/// measurement -- so measure it: run the identical candidates both ways and
/// compare every field of the summary, bit for bit.
fn equiv(c: &Cfg) {
    let mut s = setup(c);
    let from = s.boundary;
    let n = s.f.n();
    let lo = c.lo.max(from);
    let hi = c.hi.min(n);
    let mut rng = MRng::new(c.seed);
    let cands: Vec<State> = (0..c.n)
        .map(|_| make(&s.f, &mut rng, lo, hi, c.nops, opset(&c.ops)))
        .collect();
    let mut runs: Vec<Vec<Outcome>> = Vec::new();
    for fast in [1u32, 0u32] {
        s.watch.fast = fast;
        let ack = s.srv.arm(&s.watch.arm_payload(
            s.clock0,
            R_CLOCK as u32,
            R_QUAT as u32,
            R_POS as u32,
            R_VEL as u32,
            REC_LEN as u32,
            &s.segs,
        ));
        assert!(ack.starts_with("ARMED"), "re-arm failed: {}", ack);
        let t = Instant::now();
        let mut out = Vec::with_capacity(c.n);
        for st in &cands {
            let recs = tail_recs(&st.steer_u8(), &st.gas_u8(), &st.brake_u8(), from);
            let (j, b) = s.srv.run_watched(from, &recs);
            out.push(outcome(&j, &b));
        }
        println!(
            "fast={} : {:.2} ms/cand",
            fast,
            1000.0 * t.elapsed().as_secs_f64() / c.n as f64
        );
        runs.push(out);
    }
    let mut same = 0;
    let mut tail_only = 0;
    let mut diff = 0;
    for i in 0..c.n {
        let (a, b) = (&runs[0][i], &runs[1][i]);
        let (sa, sb) = match (a.sum, b.sum) {
            (Some(x), Some(y)) => (x, y),
            _ => {
                diff += 1;
                continue;
            }
        };
        // WHAT COUNTS AS THE SAME JUDGE. The question this control asks is
        // whether the two in-child sampling paths reach the same VERDICT, and a
        // verdict is: the finish time, the checkpoint count, whether it
        // tripped, which predicate tripped, at which tick, on what value, and
        // the progress the candidate is scored by. Nothing else feeds a
        // decision.
        //
        // `off_max` and `travelled` are DIAGNOSTICS and they are not in this
        // list, because the fast path evaluates one more tick than the full one
        // on a run that reaches the end (the full path only judges a tick once
        // the clock has moved past it) and both grow by construction with that
        // tick. Including them made this control report a diagnostic difference
        // as a verdict difference: it read "2 of 8 REALLY DIFFER" on eight
        // candidates whose every verdict was identical. A control that cries
        // wolf gets ignored, which costs more than the one it was guarding.
        let judged = a.time == b.time
            && a.cps == b.cps
            && sa.trip_pred == sb.trip_pred
            && sa.trip_tick == sb.trip_tick
            && sa.trip_value.to_bits() == sb.trip_value.to_bits()
            && sa.progress.to_bits() == sb.progress.to_bits();
        let exact = judged
            && sa.nticks == sb.nticks
            && sa.last_tick == sb.last_tick
            && sa.off_max.to_bits() == sb.off_max.to_bits()
            && sa.travelled.to_bits() == sb.travelled.to_bits();
        let tail = judged && !exact && sa.nticks == sb.nticks + 1;
        if exact {
            same += 1;
        } else if tail {
            tail_only += 1;
        } else {
            diff += 1;
            if diff <= 8 {
                println!(
                    "  DIFFER c{:04}: fast {:?} trip {}@{} prog {:.3} ticks {} off_max {:.3} | \
                     full {:?} trip {}@{} prog {:.3} ticks {} off_max {:.3}",
                    i, a.time, sa.trip_pred, sa.trip_tick, sa.progress, sa.nticks, sa.off_max,
                    b.time, sb.trip_pred, sb.trip_tick, sb.progress, sb.nticks, sb.off_max
                );
            }
        }
    }
    println!(
        "EQUIVALENCE  {} of {} identical in every field; {} identical in every verdict but one extra final tick; {} REALLY DIFFER",
        same, c.n, tail_only, diff
    );
    s.srv.quit();
}

/// Re-send the arm frame after changing the watch (e.g. disarming every
/// predicate for the observation-only control pass).
fn rearm(s: &mut Setup) {
    let ack = s.srv.arm(&s.watch.arm_payload(
        s.clock0,
        R_CLOCK as u32,
        R_QUAT as u32,
        R_POS as u32,
        R_VEL as u32,
        REC_LEN as u32,
        &s.segs,
    ));
    assert!(ack.starts_with("ARMED"), "re-arm failed: {}", ack);
}

/// The search names its operator set; `fk watch` has to name the same one, or
/// the false-positive rate it measures is a number about a different
/// distribution of candidates.
fn opset(name: &str) -> OpSet {
    name.parse().unwrap_or_else(|e| {
        eprintln!("fk: {}", e);
        std::process::exit(2)
    })
}

/// The median gap between a trajectory CSV's sample times, in ms.
///
/// Read rather than assumed: a recorded ghost is on a 50 ms grid and `fk btraj`
/// writes one row per 10 ms tick, and the same `--fire-need` means a different
/// duration in each.
fn sample_step_ms(path: &str) -> i64 {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return 0,
    };
    let mut ts: Vec<i64> = Vec::new();
    for line in text.lines().skip(1) {
        if let Some(f) = line.split(',').next() {
            if let Ok(v) = f.trim().parse::<i64>() {
                ts.push(v);
            }
        }
    }
    if ts.len() < 3 {
        return 0;
    }
    let mut d: Vec<i64> = ts.windows(2).map(|w| w[1] - w[0]).filter(|&x| x > 0).collect();
    if d.is_empty() {
        return 0;
    }
    d.sort_unstable();
    d[d.len() / 2]
}
