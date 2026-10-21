//! `fk resync` — put a recording's own tape back on its own recorded line.
//!
//! WHY (arm `ksi2`, 134672, 2026-08-22)
//! ------------------------------------
//! A recording made on an older game build can fail to replay on the current
//! one. The project's standing reading of that is "the run is lost": ten of
//! 134672's fifteen records DNF, including the 63.546 world record, and every
//! bound on that map has been computed from the five that survive.
//!
//! Measured, that reading is too pessimistic in a specific and useful way.
//! `fk trace` plus a lag-scanned comparison against the file's own telemetry
//! says the world record's tape is **bit-faithful for the first four seconds**
//! — 0.0003 m, the same floor a current-build recording holds for its whole
//! lap — and then a 5.6 cm error appears at race 4.06 and amplifies with an
//! e-folding time of about 1.5 s. All six of the old ghosts that could be
//! measured depart in the same 0.25 s window, at the onset of the lap's first
//! big slide.
//!
//! A divergence that is BORN at one instant is a seed, and a seed can be
//! cancelled: there may exist a small input correction, applied near it, that
//! puts the car back on the line the recording says it drove. That is what
//! this command searches for. If it works the run is recovered; if it does
//! not, and the control below says the instrument is sound, then the old
//! build's physics really are not ours and every old time on the map — the
//! author time included — belongs to a different game.
//!
//! Either way it is an answer, which is why the command reports the control
//! and the null in the same breath.
//!
//! ## The control, which is not optional
//!
//! `--control` runs the identical procedure on a tape that is known to track
//! its own recording, after deliberately breaking it by one steering unit at
//! one tick. The repair must recover it. A failure to repair the real subject
//! means nothing unless the same machinery, same budget, same seed, recovers a
//! break we made ourselves.
//!
//! ## What it is scored on
//!
//! The **sync horizon**: the race time at which the engine's own run of the
//! candidate tape first leaves the reference line by more than `--tol`. Not
//! finish time, not mean error — a mean is dominated by whatever happens after
//! the run is already lost, and finish time is unavailable for a tape that
//! never finishes. The horizon is monotone in exactly the thing being bought.

use crate::locate::{locate_v2, trajectory};
use crate::session::{Checkpoint, Engine, Session};
use crate::tape::Tape;
use crate::traj;
use forkoracle::forksrv::Rec;
use forkoracle::layout::Row;

pub struct Opts {
    pub reference: String,
    pub tol: f64,
    pub evals: usize,
    pub window: usize,
    pub seed: u64,
    pub out: Option<String>,
    pub control_break_tick: Option<usize>,
    pub control_break_delta: i32,
    pub maxdelta: i32,
    pub maxspan: usize,
    pub onset: f64,
    pub minstep: i64,
    /// how many ticks after the fork the locate control is judged over
    pub ctlticks: usize,
    /// explicit sweep window, overriding the one derived from the onset
    pub lo: Option<usize>,
    pub hi: Option<usize>,
    /// also sweep the pedal channels
    pub pedals: bool,
}

/// xorshift, so a run is reproducible from its seed and nothing depends on the
/// host's rng.
/// Sentinel "deltas" that mean a PEDAL edit rather than a steering one.
/// Outside any real steering range, so they can share the sweep list.
pub const PEDAL_BRAKE: i32 = 1000;
pub const PEDAL_LIFT: i32 = 1001;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Whole-tick lag between the engine's clock and the recording's, fitted on the
/// first `fit_ms` of the trace.
///
/// Not optional and not cosmetic: at 150 km/h one 10 ms tick is 0.42 m, and a
/// file that replays to the millisecond reads as 0.42 m of "drift" at every
/// point of its lap if this is skipped. Fitted early, where every candidate
/// still agrees, because a median over a diverging run is meaningless.
fn fit_lag(rows: &[Row], r: &traj::Reference, fit_ms: i64) -> i64 {
    let t0 = rows.first().map(|x| x.time_ms).unwrap_or(0);
    let mut best = (0i64, f64::MAX);
    for lag in -4i64..=4 {
        let mut v = Vec::new();
        for row in rows.iter().filter(|x| x.time_ms <= t0 + fit_ms) {
            if let Some(p) = r.pos_at((row.time_ms + lag * 10) as f64) {
                let d = ((row.x - p.0).powi(2) + (row.y - p.1).powi(2) + (row.z - p.2).powi(2))
                    .sqrt();
                v.push(d);
            }
        }
        if v.len() < 10 {
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = v[v.len() / 2];
        if m < best.1 {
            best = (lag, m);
        }
    }
    best.0
}

/// The race time at which the run first leaves the line by more than `tol`,
/// and the first tick at which it leaves it by more than `onset`.
///
/// Two numbers, because they answer two different questions. The HORIZON says
/// how much of the run is still good and is what a repair is scored on; the
/// ONSET says where the error was born and is where a repair has to be made.
/// Editing near the horizon does not work: on the control below, a one-unit
/// break at tick 420 costs sixty seconds of run, and 300 candidate edits in
/// the 80 ticks before the horizon bought 0.100 s of it back.
fn horizon(rows: &[Row], r: &traj::Reference, lag: i64, tol: f64, onset: f64) -> (i64, f64, i64) {
    let mut resid = Vec::new();
    let mut last = rows.first().map(|x| x.time_ms).unwrap_or(0);
    let mut born = -1i64;
    let mut run_above = 0usize;
    let mut cand = -1i64;
    for (k, row) in rows.iter().enumerate() {
        let Some(p) = r.pos_at((row.time_ms + lag * 10) as f64) else {
            break;
        };
        let d = ((row.x - p.0).powi(2) + (row.y - p.1).powi(2) + (row.z - p.2).powi(2)).sqrt();
        // The onset needs PERSISTENCE and a dead zone at the fork. The resume
        // has a transient of a few ticks, so a bare "first tick over the
        // threshold" reports the fork tick itself for every candidate — which
        // is exactly where a repair must NOT be aimed. Ten consecutive ticks
        // over the threshold, and never in the first twenty, is a departure.
        if born < 0 && k >= 20 {
            if d > onset {
                if run_above == 0 {
                    cand = row.time_ms;
                }
                run_above += 1;
                if run_above >= 10 {
                    born = cand;
                }
            } else {
                run_above = 0;
            }
        }
        if d > tol {
            break;
        }
        resid.push(d);
        last = row.time_ms;
    }
    resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if resid.is_empty() { f64::NAN } else { resid[resid.len() / 2] };
    (last, med, born)
}

pub fn run(engine: &Engine, tape: Tape, at: Checkpoint, o: Opts) -> Result<(), String> {
    let reference = traj::Reference::load(&o.reference)?;
    let bounds = reference.bounds(400.0);
    let ntape = tape.n();
    // THE LOCATE IS NOT DETERMINISTIC and it chooses between objects. On this
    // engine it can settle on a slot whose "car" has a mean speed of 1.7 m/s
    // while the real one is doing 39, and every internal test still passes.
    // The reference is the answer key: the baseline tape is this recording's
    // own tape, so the engine's run of it MUST sit on the recording. If it
    // does not, the locate is wrong — throw the server away and try again
    // rather than searching against a slot that is not the car.
    let mut attempt = 0;
    let (mut s, probe, layout, base_recs) = loop {
        attempt += 1;
        let mut s = Session::start(engine, tape.clone(), at)?;
        let probe = s.probe_tick()?;
        let base_recs = s.tape.tail_records(probe);
        let layout = locate_v2(
            &mut s.srv,
            probe,
            &base_recs,
            s.tape.start_offset_ms,
            bounds,
            2000,
            4000,
            true,
        )?;
        // The locate control must be measured on a window where the tape is
        // known to still be on the line. Judged over 238 ticks it reads
        // 0.6123 m on the very subject of this command -- because that window
        // contains the divergence being investigated -- and six good locates
        // in a row get thrown away.
        let rows = trajectory(&mut s.srv, probe, &base_recs, &layout, o.ctlticks as u32);
        // The control has to be measured AT THE FITTED LAG. The engine's clock
        // and the recording's differ by a whole number of ticks, and at
        // 145 km/h one tick is 0.40 m: judged at lag 0, a perfect locate on a
        // recording that replays to the millisecond reads 0.3964 m and gets
        // thrown away, six times in a row.
        let lag0 = fit_lag(&rows, &reference, 500);
        let mut resid: Vec<f64> = rows
            .iter()
            .filter_map(|row| {
                reference.pos_at((row.time_ms + lag0 * 10) as f64).map(|p| {
                    ((row.x - p.0).powi(2) + (row.y - p.1).powi(2) + (row.z - p.2).powi(2)).sqrt()
                })
            })
            .collect();
        resid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if resid.is_empty() { f64::MAX } else { resid[resid.len() / 2] };
        let ok = med < 0.05;
        println!(
            "attempt {}: probe tick {} (race {}), locate control: median {:.4} m over {} ticks at lag {:+}",
            attempt,
            probe,
            crate::secs(s.tape.race_ms(probe)),
            med,
            resid.len(),
            lag0
        );
        if ok {
            break (s, probe, layout, base_recs);
        }
        s.srv.quit();
        if attempt >= 6 {
            return Err(
                "six locates in a row failed the reference control; the car was never found"
                    .into(),
            );
        }
    };
    println!("tape {} ticks, reference {}", ntape, o.reference);
    let tape_race0 = s.tape.race_ms(probe);


    // How many ticks of trajectory to gather per candidate. A full 60 s tail
    // is 6600 ticks and 0.45 s; while the horizon is at 8 s, 6000 of those
    // ticks describe a car that is already lost. The gather follows the
    // horizon and the search runs an order of magnitude faster for it.
    let gather = |h_ms: i64| -> u32 {
        let ht = ((h_ms - tape_race0) / 10).max(0) as usize;
        ((ht + 800).min(ntape - probe + 200)) as u32
    };

    // The baseline, and the lag it fixes.
    let rows0 = trajectory(&mut s.srv, probe, &base_recs, &layout, (ntape - probe + 200) as u32);
    if rows0.len() < 20 {
        return Err(format!("baseline trace returned {} rows", rows0.len()));
    }
    let lag = fit_lag(&rows0, &reference, 1500);
    let (h0, m0, born0) = horizon(&rows0, &reference, lag, o.tol, o.onset);
    println!(
        "lag {:+} ticks; BASELINE sync horizon {}, error born {} (median residual before the \
         horizon {:.4} m over {} traced ticks)",
        lag,
        crate::secs(h0),
        crate::secs(born0.max(0)),
        m0,
        rows0.len()
    );
    if m0 > o.tol {
        return Err(format!(
            "the baseline is already off the reference by {:.3} m before anything was changed — \
             this tape and this reference are not the same run, and no repair of it would mean \
             anything",
            m0
        ));
    }

    // Optionally BREAK a good tape, which is how the null gets a control.
    let mut cur: Vec<Rec> = base_recs.clone();
    let mut start_h = h0;
    let mut born = born0;
    if let Some(bt) = o.control_break_tick {
        if bt < probe || bt >= ntape {
            return Err(format!("--control-break tick {} is outside [{}, {})", bt, probe, ntape));
        }
        let i = bt - probe;
        let before = cur[i].steer;
        cur[i].steer = (cur[i].steer + o.control_break_delta as f32 / 127.0).clamp(-1.0, 1.0);
        let rows = trajectory(&mut s.srv, probe, &cur, &layout, (ntape - probe + 200) as u32);
        let (h, _, b) = horizon(&rows, &reference, lag, o.tol, o.onset);
        println!(
            "CONTROL: broke tick {} steer {:+.4} -> {:+.4}; horizon {} -> {}, error born {}",
            bt,
            before,
            cur[i].steer,
            crate::secs(h0),
            crate::secs(h),
            crate::secs(b.max(0))
        );
        if h >= h0 {
            return Err(
                "the deliberate break did not move the sync horizon, so this control cannot \
                 fail and proves nothing. Choose a tick where the tape is sensitive."
                    .into(),
            );
        }
        start_h = h;
        born = b;
    }

    // THE SEARCH. Enumerate single-run steering offsets around the tick at
    // which the error is BORN, widest deltas first, and keep anything that
    // pushes the horizon out. Enumeration rather than sampling because the
    // space is small and the answer, if it exists, is one specific correction:
    // a random walk over it wastes the budget re-testing the same neighbourhood
    // (300 random edits in the 80 ticks before the horizon bought 0.100 s).
    let mut best_h = start_h;
    let mut best_born = born;
    let mut accepted = 0usize;
    let t0 = std::time::Instant::now();
    let mut evals = 0usize;
    let mut rng = Rng(o.seed | 1);
    'restart: loop {
        let centre = if best_born > 0 {
            ((best_born - tape_race0) / 10).max(0) as usize + probe
        } else {
            probe + 1
        };
        let lo = o.lo.unwrap_or_else(|| centre.saturating_sub(o.window)).max(probe);
        let hi = o.hi.unwrap_or(centre + o.window / 4).min(ntape - 1).max(lo + 1);
        println!(
            "sweeping ticks [{}, {}] (error born {}), spans 1..{}, |delta| <= {}",
            lo,
            hi,
            crate::secs(best_born.max(0)),
            o.maxspan,
            o.maxdelta
        );
        let mut order: Vec<(usize, usize, i32)> = Vec::new();
        for t in lo..=hi {
            for span in 1..=o.maxspan {
                for d in 1..=o.maxdelta {
                    order.push((t, span, d));
                    order.push((t, span, -d));
                }
                if o.pedals {
                    // THE PEDALS ARE PART OF THE INPUT SPACE, and on this
                    // subject they are the only part that is not already
                    // saturated: the tape being repaired holds FULL LOCK for
                    // every tick of the window, so one of the two steering
                    // directions is a clamped no-op by construction. A sweep
                    // of steering alone would report "no repair exists" when
                    // what it had measured is "no repair exists in the half of
                    // the input space this sweep could reach".
                    order.push((t, span, PEDAL_BRAKE));
                    order.push((t, span, PEDAL_LIFT));
                }
            }
        }
        // shuffle so an interrupted sweep is still an unbiased sample of it
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut round_best: (i64, i64, Option<(usize, usize, i32)>) = (best_h, best_born, None);
        for (t, span, dd) in order {
            if evals >= o.evals {
                break 'restart;
            }
            evals += 1;
            let i = t - probe;
            let end = (i + span).min(cur.len());
            let saved: Vec<Rec> = cur[i..end].to_vec();
            apply(&mut cur, i, end, dd);
            let rows = trajectory(&mut s.srv, probe, &cur, &layout, gather(best_h));
            let (h, _, b) = horizon(&rows, &reference, lag, o.tol, o.onset);
            cur[i..end].copy_from_slice(&saved);
            if h > round_best.0 {
                round_best = (h, b, Some((t, span, dd)));
            }
        }
        // BEST OF THE SWEEP, not the first improvement.
        //
        // The first version accepted the first candidate that helped and
        // restarted. On the control -- a tape broken by one steering unit at
        // one tick, whose exact inverse is inside the enumeration and restores
        // the whole 68 s run -- that greedy walked off through eight partial
        // repairs and finished at 12.640, because each acceptance changed the
        // tape the exact inverse was the inverse OF. Sweeping the whole
        // neighbourhood and taking its best finds the dominating correction
        // when one exists.
        match round_best.2 {
            Some((t, span, dd)) if round_best.0 > best_h + o.minstep => {
                let i = t - probe;
                let end = (i + span).min(cur.len());
                apply(&mut cur, i, end, dd);
                best_h = round_best.0;
                best_born = round_best.1;
                accepted += 1;
                println!(
                    "  round {:>2}: horizon {}  born {}  (tick {} span {} d {:+}), {} evals so far",
                    accepted,
                    crate::secs(best_h),
                    crate::secs(best_born.max(0)),
                    t,
                    span,
                    dd,
                    evals
                );
                continue 'restart;
            }
            _ => {}
        }
        println!("  sweep exhausted with no improvement");
        break;
    }
    println!(
        "DONE horizon {} -> {} ({} accepted of {} evals, {:.1}s)",
        crate::secs(start_h),
        crate::secs(best_h),
        accepted,
        evals,
        t0.elapsed().as_secs_f64()
    );

    if let Some(p) = &o.out {
        // Write the repaired tape back as a ghost, so the plain oracle — the
        // only thing that produces a RESULT — can be asked about it.
        //
        // ALL THREE CHANNELS. The first version wrote back only `steer`, and
        // the two moves that actually helped on 134672 were brake taps: the
        // file it produced re-measured at the UNREPAIRED horizon, and a run
        // chained from it started over. A writer that silently drops part of
        // the state it just searched is a result that cannot be reproduced.
        let mut steer: Vec<u8> = s.tape.steer.clone();
        let mut accel: Vec<u8> = s.tape.accel.clone();
        let mut brake: Vec<u8> = s.tape.brake.clone();
        for (i, r) in cur.iter().enumerate() {
            let t = probe + i;
            if t < ntape {
                steer[t] = (r.steer * 127.0).round().clamp(-127.0, 127.0) as i8 as u8;
                accel[t] = u8::from(r.gas > 0.5);
                brake[t] = u8::from(r.brake > 0.5);
            }
        }
        s.tape.write_candidate(&steer, &accel, &brake, std::path::Path::new(p))?;
        println!("wrote {}", p);
    }
    s.srv.quit();
    Ok(())
}

/// Apply one sweep move to `cur[i..end]`: a steering offset, or a pedal edit.
fn apply(cur: &mut [Rec], i: usize, end: usize, dd: i32) {
    match dd {
        PEDAL_BRAKE => {
            for r in cur[i..end].iter_mut() {
                r.brake = 1.0;
            }
        }
        PEDAL_LIFT => {
            for r in cur[i..end].iter_mut() {
                r.gas = 0.0;
            }
        }
        d => {
            let f = d as f32 / 127.0;
            for r in cur[i..end].iter_mut() {
                r.steer = (r.steer + f).clamp(-1.0, 1.0);
            }
        }
    }
}
