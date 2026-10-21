//! `fk trace` — one fork, one per-tick trajectory.
//!
//! The fork server runs a candidate 4–6× faster than a from-scratch validation,
//! but on its own it only ever reported a finish time. This turns the same fork
//! into an observation: while the child simulates the tail it streams the car's
//! own state out tick by tick, and the driver writes it in the same 29 columns
//! `tmtraj decode --csv` produces, so every existing analysis tool works on it
//! unchanged.
//!
//! # This is a RESUMED run, and that is not the same run
//!
//! `fk trace` forks at a checkpoint. Measured on this engine: the same tape
//! resumed at two different checkpoints agrees on **0 of 522 ticks** and
//! diverges by metres, and against a human ghost's own recorded path a resume
//! at race -0.030 is 5.578 m out where a resume at race +0.140 is 0.0055 m.
//!
//! That is fine for a search — candidates are compared with each other at one
//! fixed checkpoint — and it is wrong for regenerating telemetry, which is why
//! `fk regen` does not use this path. If you want the clean run, that is
//! `fk regen`'s recorder: fork only to LOCATE, then let the parent simulate the
//! whole tape.
//!
//! # Nothing here is hardcoded
//!
//! The server's heap layout is bimodal run to run — two identical runs can
//! differ by 87 MB — so every address is re-derived at each server start, by
//! value. Five consecutive runs gave five different addresses and
//! byte-identical CSVs.

use crate::locate::{locate_v2, trajectory};
use crate::session::{Checkpoint, Engine, Session};
use crate::tape::Tape;
use crate::traj;
use std::path::Path;

pub struct TraceOpts {
    /// A reference CSV. **A control, not an input.** The locator does not use
    /// it to find the car; the located trajectory is compared against it
    /// afterwards and the deviation printed. Its bounding box is used to reject
    /// addresses holding float triples that are nowhere near this map, which is
    /// a filter, not a fit.
    pub reference: Option<String>,
    pub out: Option<String>,
    /// Keep every `nth` row. The default writes all of them.
    pub nth: usize,
}

pub fn run(engine: &Engine, tape: Tape, at: Checkpoint, o: TraceOpts) -> Result<(), String> {
    let reference = match &o.reference {
        Some(p) => Some(traj::Reference::load(p)?),
        None => None,
    };
    // Bounds are a REJECTION filter for the locator. Without a reference the
    // box is the world, which is correct: the alternative is to invent a
    // plausible region and quietly exclude the truth.
    let bounds = match &reference {
        Some(r) => r.bounds(400.0),
        None => (-64000.0, 64000.0, -1000.0, 4000.0, -64000.0, 64000.0),
    };

    let mut s = Session::start(engine, tape, at)?;
    let probe = s.probe_tick()?;
    println!(
        "checkpoint lroundf #{} -> probe tick {} (race {}), tape {} ticks",
        s.checkpoint_clock,
        probe,
        crate::secs(s.tape.race_ms(probe)),
        s.tape.n()
    );

    let recs = s.tape.tail_records(probe);
    let t0 = std::time::Instant::now();
    // THE LOCATE IS JUDGED WHERE THE CAR IS, SO WHERE IT IS JUDGED MATTERS.
    //
    // `locate_v2`'s discriminator is d(pos)/dt against the stored velocity,
    // qualified over the ticks after the probe against a threshold of 2 % of
    // the mean speed in that window. A probe where the car is slow gets the
    // TIGHTEST threshold and the LARGEST real residual -- so the same tape, the
    // same map and the same engine locate or refuse purely on where the
    // checkpoint fell. Measured on 276874 (a 12.759 lap), one command each:
    //
    //     frac 0.30   REFUSED   1.00 m/s residual at mean speed 17.2
    //     frac 0.50   ok        0.247 m/s          at mean speed 60.3
    //     frac 0.70   ok        0.191 m/s          at mean speed 66.7
    //     frac 0.85   REFUSED   1.12 m/s           at mean speed  1.7
    //
    // The car is real at all four; only the middle two are anywhere the test
    // can see it. `fk regen` has walked a ladder of checkpoints for exactly
    // this reason since the day it was written ("a trial map is barely moving
    // at tick 200, a short map has no tick 200 at all"), and `fk trace` took
    // the first checkpoint and reported its refusal as if it were a property of
    // the file. That is how a working tape reads as an unlocatable one.
    //
    // So trace walks the same ladder. The default probe is tried first, then
    // fractions through the middle of the run where the car is fastest. Every
    // attempt still faces the unchanged acceptance test, so this widens where
    // we look and not what we will believe.
    let layout = {
        let mut found = None;
        let mut first_err = None;
        let mut tried: Vec<String> = Vec::new();
        for (label, cp) in std::iter::once(("the given checkpoint".to_string(), at))
            .chain(
                [0.5f64, 0.6, 0.4, 0.7, 0.3]
                    .iter()
                    .map(|f| (format!("frac:{f}"), Checkpoint::Fraction(*f))),
            )
        {
            if !tried.is_empty() {
                // A fresh session: the fork server is at a checkpoint and
                // moving the probe means re-forking, not rewinding.
                s = Session::start(engine, s.tape.clone(), cp)?;
            }
            let probe = s.probe_tick()?;
            let recs = s.tape.tail_records(probe);
            match locate_v2(
                &mut s.srv,
                probe,
                &recs,
                s.tape.start_offset_ms,
                bounds,
                2000,
                4000,
                true,
            ) {
                Ok(l) => {
                    if !tried.is_empty() {
                        println!(
                            "locate: the given checkpoint refused; {} located it",
                            label
                        );
                    }
                    found = Some((l, probe, recs));
                    break;
                }
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e.clone());
                    }
                    tried.push(format!("{label}: {e}"));
                }
            }
        }
        match found {
            Some(f) => f,
            None => {
                return Err(format!(
                    "the car could not be located at any of {} probe positions -- \
                     this is a property of the file, not of where we looked:\n  {}",
                    tried.len(),
                    tried.join("\n  ")
                ))
            }
        }
    };
    let (layout, probe, recs) = layout;
    println!("locate: {:.1}s", t0.elapsed().as_secs_f64());

    let t1 = std::time::Instant::now();
    let rows = trajectory(
        &mut s.srv,
        probe,
        &recs,
        &layout,
        (s.tape.n() - probe + 200) as u32,
    );
    println!(
        "trajectory: {} ticks in {:.1}s ({} .. {})",
        rows.len(),
        t1.elapsed().as_secs_f64(),
        crate::secs(rows.first().map(|r| r.time_ms).unwrap_or(0)),
        crate::secs(rows.last().map(|r| r.time_ms).unwrap_or(0))
    );

    // THREE INDEPENDENT TESTS over every row, two of which have nothing to do
    // with the signature the locator searched on. Agreement between independent
    // tests is what makes a reference-free measurement trustworthy.
    let selfcheck = forkoracle::layout::check_rows(&rows);
    match &selfcheck {
        Ok(k) => println!("self-check ok: {}", k),
        Err(e) => println!("SELF-CHECK FAILED: {}", e),
    }

    let mut control_ok = true;
    if let Some(r) = &reference {
        match traj::compare(&rows, r) {
            Some(a) => {
                println!("control vs {}: {}", o.reference.as_ref().unwrap(), a);
                // 5 cm is the locator's own qualification tolerance; a median
                // above it means the located slot is not the car even though
                // every internal test passed.
                control_ok = a.median < 0.05;
            }
            None => {
                println!("control: the reference covers none of the measured window");
                control_ok = false;
            }
        }
    }

    if let Some(p) = &o.out {
        let thinned: Vec<_> = if o.nth > 1 {
            rows.iter().step_by(o.nth).cloned().collect()
        } else {
            rows.clone()
        };
        std::fs::write(Path::new(p), traj::to_csv(&thinned, &s.tape))
            .map_err(|e| format!("{}: {}", p, e))?;
        println!("wrote {} ({} rows)", p, thinned.len());
    }
    s.srv.quit();

    if selfcheck.is_err() || !control_ok {
        return Err("the trajectory did not pass its own checks; do not use it".into());
    }
    Ok(())
}
