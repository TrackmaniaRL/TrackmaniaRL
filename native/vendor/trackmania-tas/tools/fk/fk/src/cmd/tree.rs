//! `fk tree` — THE SAVESTATE-TREE RIG. Q1 and rung 0.5, in one binary.
//!
//! Three verbs, three different questions, and none of them answers another's:
//!
//! * **`fk tree cost`** — *what does a branch cost?* The Q1 number, with every
//!   baseline re-measured on this box in the same batch, plus depth and memory.
//! * **`fk tree scale`** — *how many branch-evals per second does this box do?*
//!   Throughput only, and it says so.
//! * **`fk tree exact`** — *is a forward-only fork answer true?* Rung 0.5:
//!   positive control, negative control, and a DEPTH sweep, because exactness
//!   at depth 1 is not exactness at depth 50.
//!
//! # Why the arms are what they are
//!
//! The published cost model is `10.3 ms fixed + 26.9 µs per remaining simulated
//! tick` (`tm2020-forkserver.md`, measured on another box). The fixed part is
//! not one thing: it is the fork, the copy-on-write faults as the child touches
//! the physics working set, and **the validator's finish-and-print path**. A
//! branch child pays the first two and none of the third, which is the whole
//! reason a tree could be cheap. So:
//!
//! | arm | isolates |
//! |---|---|
//! | `N` null fork | the fork alone — **the floor**; nothing that forks is cheaper |
//! | `R` fork-to-finish | the published model, re-fitted here |
//! | `B` branch, swept over `k` | fixed + per-tick cost of a branch |
//! | `B` at `k = the whole prefix` | **the fallback is the same arm** |
//!
//! That last row is the design decision worth naming. Re-simulating a prefix
//! from the root — the "no savestate tree" fallback — *is* a branch whose `k` is
//! the prefix, so one sweep gives the tree cost and the no-tree cost from one
//! instrument against one baseline. Measuring the fallback with a different
//! mechanism would mean comparing two numbers from two rigs, and a paired
//! difference transfers only if the baselines match.
//!
//! # What refuses to run
//!
//! A timing batch taken while this box is busy. `lroundf` is bit-identical only
//! on an idle box; under contention it moves in whole chunks of ~62 calls, so a
//! fixed checkpoint count lands at a different simulation point (104 of 150
//! workers stopped a tick later when 150 servers started at once). A timing
//! number taken under load is a number about the load.

use crate::oracle::validate_batch;
use crate::session::{Checkpoint, Engine, Session};
use crate::tape::Tape;
use branch::{ForkAnswer, Forest, Handle, TraceCfg, ROOT};
use forkoracle::forksrv::{parse_result, rec_of, ForkServer, Rec};
use forkoracle::layout::Layout;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------- small helpers

/// xorshift64, so a candidate set is part of the evidence rather than an
/// accident of the run.
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
        (self.next() % n.max(1) as u64) as usize
    }
}

fn load1() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(f64::NAN)
}

/// Refuse a timing batch on a busy box. See the module header.
fn require_idle(limit: f64, allow: bool) -> Result<f64, String> {
    let l = load1();
    if !allow && l > limit {
        return Err(format!(
            "load average is {:.2} (limit {:.2}) -- refusing to publish a TIMING batch taken \
             under load. `lroundf` moves in whole ~62-call chunks under contention, so a fixed \
             checkpoint lands at a different simulation point and the number would be about the \
             load. Pass --allow-load to measure anyway, and say so in the report.",
            l, limit
        ));
    }
    Ok(l)
}

/// Median rather than mean: one scheduling stall should not set the headline.
fn med(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p).round() as usize]
}

/// Private (non-shared) dirty pages of a live node, in MB.
///
/// The question depth asks of memory is not "how big is the process" — every
/// node maps the same ~150 MB engine — it is **how much of it has stopped being
/// shared**. `Private_Dirty` is that number, and it is the one that decides
/// whether a 500-node beam fits.
fn private_dirty_mb(pid: i32) -> Option<f64> {
    let s = std::fs::read_to_string(format!("/proc/{}/smaps_rollup", pid)).ok()?;
    for l in s.lines() {
        if let Some(v) = l.strip_prefix("Private_Dirty:") {
            let kb: f64 = v.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

fn recs_from(steer: &[u8], accel: &[u8], brake: &[u8], from: usize) -> Vec<Rec> {
    (from..steer.len()).map(|t| rec_of(steer[t], accel[t], brake[t])).collect()
}

/// A macro action: hold one (steer, gas, brake) triple for `k` ticks.
///
/// The steer ladder is deliberately coarse — this is the alphabet the explorer
/// will use, so the rig exercises the thing that will actually be run rather
/// than a synthetic single-tick perturbation.
fn macro_action(rng: &mut Rng) -> (u8, u8, u8) {
    const LADDER: [i8; 9] = [-127, -95, -64, -32, 0, 32, 64, 95, 127];
    let s = LADDER[rng.below(LADDER.len())] as u8;
    let gas = if rng.below(4) == 0 { 0 } else { 1 };
    let brake = if rng.below(8) == 0 { 1 } else { 0 };
    (s, gas, brake)
}

/// Take the server out of a session, leaving the tape behind.
///
/// `Session` holds the two together because the one question no measurement of
/// a simulated trajectory should be trusted without is *is the simulator
/// running the tape I asked about* — so this is only ever done AFTER
/// `assert_running_our_tape`.
fn split(s: Session) -> (ForkServer, Tape, u64) {
    let Session { srv, tape, checkpoint_clock } = s;
    (srv, tape, checkpoint_clock)
}

/// Least squares on `(x, y)`, returning `(intercept, slope)`.
fn fit(p: &[(f64, f64)]) -> (f64, f64) {
    let n = p.len() as f64;
    let sx: f64 = p.iter().map(|q| q.0).sum();
    let sy: f64 = p.iter().map(|q| q.1).sum();
    let sxx: f64 = p.iter().map(|q| q.0 * q.0).sum();
    let sxy: f64 = p.iter().map(|q| q.0 * q.1).sum();
    let d = n * sxx - sx * sx;
    if d.abs() < 1e-12 {
        return (sy / n, 0.0);
    }
    let slope = (n * sxy - sx * sy) / d;
    ((sy - slope * sx) / n, slope)
}

/// Locate the car for the state trace, or say plainly that there is none.
///
/// A missing layout is **UNMEASURED**, not "no trace needed": the cost arms
/// still run and their trace column reads UNMEASURED rather than 0.
fn locate_layout(srv: &mut ForkServer, probe: usize, recs: &[Rec], off: i32) -> Option<Layout> {
    let wide = (-1.0e6, 1.0e6, -1.0e6, 1.0e6, -1.0e6, 1.0e6);
    match crate::locate::locate_v2(srv, probe, recs, off, wide, 40_000, 24, false) {
        Ok(l) => {
            println!(
                "state readout located: pos {:#x}, clock {:#x} (bias {}), self-consistency \
                 {:.3} m/s",
                l.pos, l.clock, l.clock_bias, l.rms
            );
            Some(l)
        }
        Err(e) => {
            eprintln!(
                "fk tree: the car could not be located ({}). The timing arms still run; every \
                 state-trace column reads UNMEASURED. This is a harness limit, not an absence: \
                 the engine computes the state and it is in memory.",
                e
            );
            None
        }
    }
}

// ---------------------------------------------------------------- fk tree cost

pub struct CostOpts {
    pub reps: usize,
    pub ks: Vec<u64>,
    pub depth: usize,
    pub seed: u64,
    pub load_limit: f64,
    pub allow_load: bool,
    pub trace: bool,
}

/// Q1: what does a branch cost, against every baseline, on this box.
pub fn cost(engine: &Engine, tape: Tape, at: Checkpoint, o: CostOpts) -> Result<(), String> {
    let load = require_idle(o.load_limit, o.allow_load)?;
    let n_ticks = tape.n();
    let mut s = Session::start(engine, tape, at)?;
    println!(
        "# fk tree cost -- load1 {:.2}, {} cores, tape {} ticks, checkpoint lroundf #{}",
        load,
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0),
        n_ticks,
        s.checkpoint_clock
    );

    // THE IDENTITY CONTROL, first and unconditional: is this server running the
    // tape we asked about? Two runs sharing a directory produce a real,
    // self-consistent trajectory of a car that drove somewhere else, and
    // nothing internal can see it.
    s.assert_running_our_tape()?;
    println!("identity control: the engine's decoded input array IS our tape, tick for tick");

    let probe = s.probe_tick().map_err(|e| {
        format!("boundary probe failed ({e}); a resume cannot be trusted without it, and a \
                 fallback here is how a plausible number 2-3 ms off gets banked")
    })?;
    let remaining = n_ticks.saturating_sub(probe);
    println!(
        "root probe: tick {} (race {}), {} ticks remain",
        probe,
        crate::secs(s.tape.race_ms(probe)),
        remaining
    );

    let (mut srv, tape, _ck) = split(s);
    let reference = recs_from(&tape.steer, &tape.accel, &tape.brake, 0);

    let layout = if o.trace {
        locate_layout(&mut srv, probe, &reference, tape.start_offset_ms)
    } else {
        None
    };

    // ---- A0: the floor.
    let mut a0 = Vec::new();
    for _ in 0..o.reps {
        let t = Instant::now();
        let r = srv.null_fork();
        a0.push(t.elapsed().as_secs_f64() * 1000.0);
        if !r.contains("NULLFORK") {
            return Err(format!("the null-fork arm did not answer: {:?}", r.trim()));
        }
    }
    println!(
        "\nA0  null fork                    {:8.3} ms   (p10 {:.3}, p90 {:.3}, n={})   THE FLOOR",
        med(&a0),
        pct(&a0, 0.1),
        pct(&a0, 0.9),
        o.reps
    );

    // ---- A1: fork to finish. The published model, re-measured here.
    let tail = recs_from(&tape.steer, &tape.accel, &tape.brake, probe + 1);
    let mut a1 = Vec::new();
    for _ in 0..o.reps {
        let t = Instant::now();
        let out = srv.run(probe + 1, &tail);
        a1.push(t.elapsed().as_secs_f64() * 1000.0);
        let r = parse_result(&out);
        if r.0.is_none() && r.1.is_none() {
            return Err("the fork-to-finish arm produced no verdict at all -- the child is not \
                        reaching the validator, and every later arm would be timing nothing"
                .into());
        }
    }
    let a1m = med(&a1);
    println!(
        "A1  fork -> finish ({:5} ticks)  {:8.3} ms   (p10 {:.3}, p90 {:.3})",
        remaining,
        a1m,
        pct(&a1, 0.1),
        pct(&a1, 0.9)
    );

    // ---- A2: THE BRANCH, swept over k. The big-k end IS the no-tree fallback.
    let cfg = layout.as_ref().map(|l| TraceCfg {
        layout: l.clone(),
        dir: engine.work.join("traces"),
        stride: 1,
        max: 200_000,
    });
    if let Some(c) = &cfg {
        std::fs::create_dir_all(&c.dir).map_err(|e| e.to_string())?;
    }
    let mut f = Forest::new(srv, &engine.work, reference, cfg)?;
    f.probe_root()?;
    let root_b = f.probed_boundary(ROOT).unwrap_or(probe);

    println!(
        "\n      k    branch(ms)   probe(ms)    total(ms)   ticks consumed   trace rows"
    );
    let mut rows: Vec<(f64, f64)> = Vec::new();
    for &k in &o.ks {
        let reps = if k > 500 { o.reps.min(3).max(1) } else { o.reps };
        let (mut bt, mut pt) = (Vec::new(), Vec::new());
        let (mut trace_rows, mut consumed) = (0usize, 0usize);
        let mut trace_err: Option<String> = None;
        for _ in 0..reps {
            let t = Instant::now();
            let (trace, h) = match f.advance(ROOT, &[], 0, k) {
                Ok(v) => v,
                Err(e) if e.contains("not tick-continuous") => {
                    // The trace refused itself. That is the trace's verdict, not
                    // the timing's: record it and keep timing.
                    trace_err = Some(e);
                    continue;
                }
                Err(e) => return Err(e),
            };
            let total = t.elapsed().as_secs_f64() * 1000.0;
            trace_rows = trace.len();
            consumed = f.probed_boundary(h).map(|b| b.saturating_sub(root_b)).unwrap_or(0);
            // The split between branching and probing is MEASURED, not
            // apportioned: one more probe on the live node costs exactly one
            // probe.
            let tp = Instant::now();
            let _ = f.floor(h, None)?;
            let ptime = tp.elapsed().as_secs_f64() * 1000.0;
            f.release(h);
            bt.push(total - ptime);
            pt.push(ptime);
        }
        if let Some(e) = trace_err {
            eprintln!("k={}: {}", k, e);
        }
        if bt.is_empty() {
            println!("  {:5}    every branch at this k refused itself", k);
            continue;
        }
        let (b, p) = (med(&bt), med(&pt));
        println!(
            "  {:5}   {:9.3}   {:9.3}    {:9.3}   {:14}   {:10}",
            k,
            b,
            p,
            b + p,
            consumed,
            if layout.is_some() { trace_rows.to_string() } else { "UNMEASURED".into() }
        );
        rows.push((k as f64, b));
    }

    if rows.len() >= 2 {
        let (a, slope) = fit(&rows);
        println!(
            "\nfit: branch = {:.3} ms + {:.2} us/tick\n     published model for a FULL child: \
             10.300 ms + 26.90 us/tick (another box, so the comparison is of shape, not of \
             value)",
            a,
            slope * 1000.0
        );
        let modelled_full = a + slope * remaining as f64;
        println!(
            "     the finish-and-print path a branch does NOT pay:\n       A1 ({:.3} ms) - a \
             branch that simulates the same {} ticks ({:.3} ms) = {:.3} ms",
            a1m, remaining, modelled_full, a1m - modelled_full
        );
        println!(
            "     no-tree fallback, from the SAME instrument: a branch of {} ticks = {:.3} ms",
            remaining, modelled_full
        );
        let macro10 = a + slope * 10.0;
        println!(
            "     one k=10 macro = {:.3} ms  ->  {:.1} min per 3.4 M-eval forward pass on {} cores",
            macro10,
            3.4e6 * macro10 / 1000.0
                / 60.0
                / std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1) as f64,
            std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0)
        );
    }

    // ---- A4: depth. Per-generation cost and PRIVATE dirty memory.
    if o.depth > 0 {
        println!("\n  gen   branch(ms)   private dirty(MB)   next floor tick");
        let mut rng = Rng::new(o.seed);
        let mut h = ROOT;
        let mut chain: Vec<Handle> = Vec::new();
        let mut from = f.floor(ROOT, None)?;
        let k = 10u64;
        let mut gt = Vec::new();
        for g in 1..=o.depth {
            let (st, gas, br) = macro_action(&mut rng);
            let inputs: Vec<Rec> = (0..k).map(|_| rec_of(st, gas, br)).collect();
            if from + inputs.len() + 2 >= n_ticks {
                println!("  (stopped at generation {}: the tape ran out of ticks)", g);
                break;
            }
            let t = Instant::now();
            let (_, nh) = f.advance(h, &inputs, from, k)?;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            gt.push((g as f64, ms));
            let mb = f.node_pid(nh).and_then(private_dirty_mb).unwrap_or(f64::NAN);
            from = f.floor(nh, None)?;
            if g <= 5 || g % 5 == 0 || g == o.depth {
                println!("  {:3}   {:9.3}   {:17.1}   {:15}", g, ms, mb, from);
            }
            chain.push(nh);
            h = nh;
        }
        let live: Vec<f64> =
            chain.iter().filter_map(|x| f.node_pid(*x)).filter_map(private_dirty_mb).collect();
        let total: f64 = live.iter().sum();
        println!(
            "  {} live nodes hold {:.1} MB of private dirty pages ({:.2} MB each) -- a 500-node \
             beam would be {:.1} GB",
            chain.len(),
            total,
            total / chain.len().max(1) as f64,
            500.0 * total / chain.len().max(1) as f64 / 1024.0
        );
        if gt.len() >= 4 {
            let (a, slope) = fit(&gt);
            let first = med(&gt[..gt.len() / 4].iter().map(|x| x.1).collect::<Vec<_>>());
            let last = med(&gt[gt.len() * 3 / 4..].iter().map(|x| x.1).collect::<Vec<_>>());
            println!(
                "  per-generation cost: {:.3} ms + {:.4} ms/generation; first quarter {:.3} ms, \
                 last quarter {:.3} ms ({:.2}x)",
                a,
                slope,
                first,
                last,
                last / first
            );
        }
        for x in chain {
            f.release(x);
        }
    }
    println!("\n(load1 {:.2} at the start, {:.2} at the end)", load, load1());
    Ok(())
}

// ---------------------------------------------------------------- fk tree exact

pub struct ExactOpts {
    pub n: usize,
    pub depths: Vec<usize>,
    pub k: u64,
    pub seed: u64,
}

struct Cand {
    file: PathBuf,
    fork: ForkAnswer,
    depth: usize,
}

/// RUNG 0.5. Is a forward-only fork answer exact — at depth 1, and at depth 50?
///
/// Three things run here and all three must pass:
///
/// 1. **positive** — `n` candidates per depth, each built by descending a chain
///    of branches that only ever APPEND. Fork answer vs the plain oracle on the
///    byte-identical written tape, on time AND on DNF checkpoint count.
/// 2. **negative** — a deliberate sub-boundary write must reproduce the KNOWN
///    WRONG ANSWER, not merely disagree. The sharp form: the fork's answer for
///    a candidate `C` that differs below the boundary must EQUAL the plain
///    oracle's answer for the hybrid `H` (reference below the boundary, `C`
///    above) and DIFFER from the plain oracle's answer for `C` itself.
///    Disagreement alone is satisfied by any noisy instrument; equality with
///    `H` identifies the mechanism.
/// 3. **depth** — the positive control, swept. A one-tick boundary error per
///    generation is a fifty-tick error at depth 50 and every one of them is
///    individually invisible. Exactness at depth 1 does not establish a tree.
pub fn exact(engine: &Engine, tape: Tape, at: Checkpoint, o: ExactOpts) -> Result<bool, String> {
    let n_ticks = tape.n();
    let mut s = Session::start(engine, tape, at)?;
    s.assert_running_our_tape()?;
    let probe = s.probe_tick().map_err(|e| format!("boundary probe failed: {e}"))?;
    println!(
        "# fk tree exact -- root probe tick {} (race {}), tape {} ticks",
        probe,
        crate::secs(s.tape.race_ms(probe)),
        n_ticks
    );

    let (srv, tape, _) = split(s);

    // The reference's own time from the PLAIN oracle. Nothing below is scored
    // against it; a reference that does not validate means the batch is
    // measuring something other than what it thinks.
    let refp = engine.work.join("reference.Ghost.Gbx");
    tape.write_reference(&refp)?;
    let refres = validate_batch(&engine.server, &engine.map, &[refp.as_path()], "ref")?;
    println!(
        "reference validates at {}",
        crate::secs_opt(refres.first().and_then(|r| r.time_ms))
    );

    let (steer0, accel0, brake0) = (tape.steer.clone(), tape.accel.clone(), tape.brake.clone());
    let reference = recs_from(&steer0, &accel0, &brake0, 0);
    let mut f = Forest::new(srv, &engine.work, reference, None)?;
    f.probe_root()?;

    let mut rng = Rng::new(o.seed);
    let mut cands: Vec<Cand> = Vec::new();

    // ---------------- 1 + 3: the positive control, at every depth
    for &d in &o.depths {
        for i in 0..o.n {
            let (mut st, mut ac, mut br) = (steer0.clone(), accel0.clone(), brake0.clone());
            let mut h = ROOT;
            let mut from = f.floor(ROOT, None)?;
            let mut chain: Vec<Handle> = Vec::new();
            let mut ran_out = false;
            for _g in 0..d {
                let (ss, gg, bb) = macro_action(&mut rng);
                let k = o.k as usize;
                if from + k + 2 >= n_ticks {
                    ran_out = true;
                    break;
                }
                for t in from..from + k {
                    st[t] = ss;
                    ac[t] = gg;
                    br[t] = bb;
                }
                let inputs: Vec<Rec> =
                    (from..from + k).map(|t| rec_of(st[t], ac[t], br[t])).collect();
                let (_, nh) = f.advance(h, &inputs, from, o.k)?;
                chain.push(nh);
                h = nh;
                from = f.floor(h, None)?;
            }
            if ran_out {
                for x in chain {
                    f.release(x);
                }
                continue;
            }
            // The tail is the reference's own inputs from the leaf's floor on,
            // so the written file differs from the reference in exactly the
            // macro spans and nowhere else.
            let tail = recs_from(&st, &ac, &br, from);
            let fork = f.finish(h, &tail, from)?;
            let file = engine.work.join(format!("d{:02}_{:04}.Ghost.Gbx", d, i));
            tape.write_candidate(&st, &ac, &br, &file)?;
            cands.push(Cand { file, fork, depth: d });
            for x in chain {
                f.release(x);
            }
        }
    }
    if cands.is_empty() {
        return Err("no candidate could be built -- the tape is too short for these depths".into());
    }

    // The plain oracle, on the files as written. THIS is the result; every fork
    // answer above is a measurement.
    let files: Vec<&Path> = cands.iter().map(|c| c.file.as_path()).collect();
    let plain = validate_batch(&engine.server, &engine.map, &files, "exact")?;
    if plain.len() != cands.len() {
        return Err(format!(
            "the plain oracle returned {} rows for {} files -- an ABSENT ROW is not a failure \
             row, and folding one in would score a file that was never simulated",
            plain.len(),
            cands.len()
        ));
    }
    let by: HashMap<String, (Option<i64>, Option<u32>)> =
        plain.into_iter().map(|r| (r.file, (r.time_ms, r.cps))).collect();

    let mut per_depth: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut shown = 0;
    for c in &cands {
        let name = c.file.file_name().unwrap().to_string_lossy().into_owned();
        let (pt, pc) = match by.get(&name) {
            Some(v) => *v,
            None => {
                return Err(format!(
                    "{} is missing from the oracle's table -- absent is not DNF",
                    name
                ))
            }
        };
        // BOTH axes. A comparison on time alone scores two different DNFs equal.
        let agree = c.fork.time_ms == pt && (c.fork.time_ms.is_some() || c.fork.dnf_cps == pc);
        let e = per_depth.entry(c.depth).or_insert((0, 0));
        e.1 += 1;
        if agree {
            e.0 += 1;
        } else if shown < 6 {
            shown += 1;
            println!(
                "MISMATCH {} (depth {}): fork {} cps {:?} | plain {} cps {:?} | first diff \
                 tick {:?}, {} ticks differ from the reference",
                name,
                c.depth,
                crate::secs_opt(c.fork.time_ms),
                c.fork.dnf_cps,
                crate::secs_opt(pt),
                pc,
                c.fork.first_diff_tick,
                c.fork.ticks_differing
            );
        }
    }
    println!("\n  depth   agree / total   (fork answer vs the plain oracle on the written tape)");
    let mut ds: Vec<usize> = per_depth.keys().copied().collect();
    ds.sort_unstable();
    let mut all_ok = true;
    for d in ds {
        let (a, t) = per_depth[&d];
        println!("  {:5}   {:5} / {:5}{}", d, a, t, if a == t { "" } else { "   <-- FAILS" });
        if a != t {
            all_ok = false;
        }
    }

    // ---------------- 2: the negative control
    println!("\nnegative control: a deliberate write BELOW the root's probed boundary");
    let (neg_ok, said) = negative_control(&mut f, &tape, engine, probe, &mut rng)?;
    println!("{}", said);

    if !neg_ok {
        println!(
            "\nRUNG 0.5: UNMEASURED. The positive half passed {} of {} but the negative control \
             did not, and a passing reading from an instrument whose control failed is worth \
             exactly as much as the control.",
            per_depth.values().map(|x| x.0).sum::<usize>(),
            cands.len()
        );
    }
    Ok(all_ok && neg_ok)
}

/// The negative half. Returns `(passed, what it said)`.
///
/// It must reproduce the KNOWN WRONG ANSWER. If it does not — if a sub-boundary
/// write turns out to be honoured after all — that is MORE interesting than if
/// it does, because the recorded defect would then not be what the note says it
/// is, and the forward-only rule this whole component rests on would be
/// resting on a mechanism nobody has reproduced.
fn negative_control(
    f: &mut Forest,
    tape: &Tape,
    engine: &Engine,
    probe: usize,
    rng: &mut Rng,
) -> Result<(bool, String), String> {
    let n = tape.n();
    let below = probe.saturating_sub(3);
    if below < 2 || probe + 40 >= n {
        return Ok((
            false,
            "UNMEASURED: this checkpoint leaves no room for a sub-boundary write".into(),
        ));
    }
    // C differs from the reference BOTH below and above the boundary, so it is
    // the shape of candidate a search could really produce.
    let (mut st, mut ac, mut br) = (tape.steer.clone(), tape.accel.clone(), tape.brake.clone());
    let (ss, gg, bb) = macro_action(rng);
    for t in below..=probe {
        st[t] = ss;
        ac[t] = gg;
        br[t] = bb;
    }
    let (s2, g2, b2) = macro_action(rng);
    for t in probe + 1..(probe + 40).min(n) {
        st[t] = s2;
        ac[t] = g2;
        br[t] = b2;
    }
    // H is the hybrid the engine would really have run: the reference below the
    // boundary, C above it. This is the "known wrong answer".
    let (mut hs, mut ha, mut hb) = (st.clone(), ac.clone(), br.clone());
    for t in below..=probe {
        hs[t] = tape.steer[t];
        ha[t] = tape.accel[t];
        hb[t] = tape.brake[t];
    }

    let cfile = engine.work.join("neg_C.Ghost.Gbx");
    let hfile = engine.work.join("neg_H.Ghost.Gbx");
    tape.write_candidate(&st, &ac, &br, &cfile)?;
    tape.write_candidate(&hs, &ha, &hb, &hfile)?;

    let recs = recs_from(&st, &ac, &br, below);
    let raw = f.root_mut().run(below, &recs);
    let (ft, fc) = parse_result(&raw);

    let res = validate_batch(
        &engine.server,
        &engine.map,
        &[cfile.as_path(), hfile.as_path()],
        "neg",
    )?;
    let get = |name: &str| -> Option<(Option<i64>, Option<u32>)> {
        res.iter().find(|r| r.file == name).map(|r| (r.time_ms, r.cps))
    };
    let (ct, cc) = get("neg_C.Ghost.Gbx").ok_or("the oracle returned no row for neg_C")?;
    let (ht, hc) = get("neg_H.Ghost.Gbx").ok_or("the oracle returned no row for neg_H")?;

    let eq = |a: (Option<i64>, Option<u32>), b: (Option<i64>, Option<u32>)| {
        a.0 == b.0 && (a.0.is_some() || a.1 == b.1)
    };
    let fork_is_hybrid = eq((ft, fc), (ht, hc));
    let c_differs = !eq((ct, cc), (ht, hc));

    let said = if !c_differs {
        format!(
            "UNMEASURED -- the sub-boundary write does not change the run at all \
             (plain C {} cps {:?} == plain H {} cps {:?}), so this candidate cannot tell a \
             honoured write from a dropped one. It is a control that cannot fail, which is \
             decoration. Retry with a tick that matters.",
            crate::secs_opt(ct),
            cc,
            crate::secs_opt(ht),
            hc
        )
    } else if fork_is_hybrid {
        format!(
            "PASS -- the fork reproduced the KNOWN WRONG ANSWER.\n  \
             fork(C), written from tick {} : {} cps {:?}\n  \
             plain(H), the hybrid          : {} cps {:?}   <- EQUAL, so ticks {}..{} were \
             silently dropped\n  \
             plain(C), the file we wrote   : {} cps {:?}   <- DIFFERENT, so those ticks mattered",
            below,
            crate::secs_opt(ft),
            fc,
            crate::secs_opt(ht),
            hc,
            below,
            probe,
            crate::secs_opt(ct),
            cc
        )
    } else {
        format!(
            "DID NOT REPRODUCE -- and this is MORE interesting than a pass.\n  \
             fork(C)  = {} cps {:?}\n  plain(H) = {} cps {:?}\n  plain(C) = {} cps {:?}\n  \
             The fork's answer is neither the hybrid nor the file. The recorded defect is not \
             what the note says it is, and the forward-only rule rests on a mechanism that has \
             not been reproduced here. CHASE THIS before anything is built on the positive half.",
            crate::secs_opt(ft),
            fc,
            crate::secs_opt(ht),
            hc,
            crate::secs_opt(ct),
            cc
        )
    };
    Ok((c_differs && fork_is_hybrid, said))
}

// ---------------------------------------------------------------- fk tree scale

pub struct ScaleOpts {
    pub servers: usize,
    pub secs: u64,
    pub k: u64,
    pub seed: u64,
    pub load_limit: f64,
    pub allow_load: bool,
}

/// Throughput only, and it says so: branch-evals per second with `servers`
/// independent fork servers side by side.
///
/// Each server gets its OWN work directory. Worker directories named by index
/// are how two concurrent searches came to validate each other's candidates and
/// credit the time to the local one; the directory lock refuses sharing rather
/// than racing.
pub fn scale(engine: &Engine, tape: Tape, at: Checkpoint, o: ScaleOpts) -> Result<(), String> {
    let load = require_idle(o.load_limit, o.allow_load)?;
    // ONE measured checkpoint for everybody: `Fraction` costs a full validation
    // every time it is resolved, and resolving it per server would put a
    // hundred server launches inside a throughput measurement.
    let clock = at.to_clock(engine, &tape)?;
    println!(
        "# fk tree scale -- {} servers, {} s, k={}, checkpoint lroundf #{}, load1 {:.2}, {} cores",
        o.servers,
        o.secs,
        o.k,
        clock,
        load,
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0)
    );
    let done = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let up = AtomicU64::new(0);
    let t_launch = Instant::now();
    let deadline = std::time::Duration::from_secs(o.secs);
    std::thread::scope(|sc| {
        for w in 0..o.servers {
            let (base, tape) = (engine.clone(), tape.clone());
            let (done, failed, up) = (&done, &failed, &up);
            sc.spawn(move || {
                let mut e = base;
                e.work = e.work.join(format!("w{:03}", w));
                e.work_is_temporary = true;
                if let Err(err) = scale_worker(&e, tape, clock, o.k, deadline, done, up) {
                    eprintln!("worker {}: {}", w, err);
                    failed.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    let n = done.load(Ordering::Relaxed);
    let wall = t_launch.elapsed().as_secs_f64();
    println!(
        "\n{} branch-evals, {} servers up, {} failed to start.\n{:.0} branch-evals/s measured over \
         the {} s window (wall including launches: {:.1} s)\n(throughput only -- the exactness \
         claim is `fk tree exact`)",
        n,
        up.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed),
        n as f64 / o.secs as f64,
        o.secs,
        wall
    );
    Ok(())
}

fn scale_worker(
    e: &Engine,
    tape: Tape,
    clock: u64,
    k: u64,
    window: std::time::Duration,
    done: &AtomicU64,
    up: &AtomicU64,
) -> Result<(), String> {
    let mut s = Session::start(e, tape, Checkpoint::Clock(clock))?;
    s.assert_running_our_tape()?;
    s.probe_tick()?;
    let (srv, tape, _) = split(s);
    let reference = recs_from(&tape.steer, &tape.accel, &tape.brake, 0);
    let mut f = Forest::new(srv, &e.work, reference, None)?;
    f.probe_root()?;
    up.fetch_add(1, Ordering::Relaxed);
    let t0 = Instant::now();
    while t0.elapsed() < window {
        let (_, h) = f.advance(ROOT, &[], 0, k)?;
        f.release(h);
        done.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}
