//! `fk liveness` — is this anchor the copy of the car that has the fields?
//!
//! The engine keeps several copies of the vehicle state. They hold the same
//! position and pass every structural test a locator can apply — unit
//! quaternion, velocity equal to the position's derivative — but only one of
//! them has the surrounding fields alive. The others are bare position copies
//! with dead memory around them, and a regeneration anchored on one of those
//! writes zeroed wheel rotations and gear into a file that passes the whole
//! acceptance gate, because none of those bytes affects the simulation.
//!
//! So this asks the question that needs no answer key: at the four wheel-record
//! slots, do the rotation floats MOVE? Four live against four dead, nothing in
//! between. It is a property of the copy, not of the run, and it costs one
//! fork.
//!
//! Offsets are given relative to THIS driver's anchor — `Layout::pos`, the
//! address the locator returns. The carrier-bytes table published by the
//! byte-naming arm is relative to the live-wheeled copy's position triple,
//! which on this fixture sits 408 bytes above; its `car + 88 + 44k` is this
//! anchor's `car + 496 + 44k`.

use crate::locate::{gather_ticks, locate_v2};
use crate::session::{Checkpoint, Engine, Session};
use crate::tape::Tape;
use crate::traj;

/// Wheel records, relative to `Layout::pos`.
///
/// **496 IS A FIXTURE CONSTANT, NOT A PROPERTY OF THE ENGINE, AND THIS IS THE
/// BUG IN THIS FILE.** The locator's anchor and the copy of the car that holds
/// the fields are two different objects; 496 is `WHEEL0 + 408` because on the
/// one fixture this was measured on, the field copy sat 408 bytes above the
/// anchor. That distance is NOT fixed: on untitled 01 (2026-08-23) it is 124
/// bytes, and `fk regen` no longer assumes any value for it — it resolves the
/// engine's own pointer (`fk ptr`) or searches. Run against a fixture with a
/// different shadow, this command reads four unrelated floats and reports 0 of
/// 4 live for a car whose wheels are turning: the exact false negative that
/// reads like "the window does not reach the car".
///
/// Left in place, named, rather than silently corrected: the constant is
/// honest about the one fixture it was measured on, and the fix is to take the
/// pointer rather than to guess a better number. `--shadow` overrides it.
pub const WHEEL0: i64 = 496;
/// The shadow this file's default offsets assume, so the arithmetic above can
/// be undone by a caller who knows better.
pub const ASSUMED_SHADOW: i64 = 408;
pub const WHEEL_STRIDE: i64 = crate::vislayout::WHEEL_STRIDE;
/// Rotation within a wheel record.
pub const WHEEL_ROT: i64 = crate::vislayout::WHEEL_ROT;

pub struct LivenessOpts {
    pub reference: Option<String>,
    /// Extra offsets to report, relative to this anchor.
    pub also: Vec<i64>,
}

pub fn run(engine: &Engine, tape: Tape, at: Checkpoint, o: LivenessOpts) -> Result<(), String> {
    let bounds = match &o.reference {
        Some(p) => traj::Reference::load(p)?.bounds(400.0),
        None => (-64000.0, 64000.0, -1000.0, 4000.0, -64000.0, 64000.0),
    };
    let mut s = Session::start(engine, tape, at)?;
    let probe = s.probe_tick()?;
    let recs = s.tape.tail_records(probe);
    let layout = locate_v2(
        &mut s.srv,
        probe,
        &recs,
        s.tape.start_offset_ms,
        bounds,
        2000,
        4000,
        true,
    )?;

    let mut want: Vec<(String, i64)> = (0..4)
        .map(|k| (format!("wheel{k}_rot"), WHEEL0 + WHEEL_STRIDE * k + WHEEL_ROT))
        .collect();
    for a in &o.also {
        want.push((format!("car{a:+}"), *a));
    }
    let lo = want.iter().map(|w| w.1).min().unwrap() - 4;
    let hi = want.iter().map(|w| w.1).max().unwrap() + 8;
    let segs = vec![
        (layout.clock, 4u32),
        (layout.pos.wrapping_add(lo as u64), (hi - lo) as u32),
    ];
    let rows = gather_ticks(&mut s.srv, probe, &recs, &segs, 600, 4000, (0, 4));
    if rows.len() < 50 {
        return Err(format!("only {} ticks gathered", rows.len()));
    }
    println!("\nanchor {:#x} (Layout::pos), {} ticks", layout.pos, rows.len());
    println!("what\toffset\tdistinct\tmin\tmax\tverdict");
    let mut live = 0;
    for (name, off) in &want {
        let i = (off - lo) as usize + 4;
        let vals: Vec<f64> = rows
            .iter()
            .map(|t| f32::from_le_bytes(t.rec[i..i + 4].try_into().unwrap()) as f64)
            .collect();
        let mut d: Vec<u64> = vals.iter().map(|v| v.to_bits()).collect();
        d.sort_unstable();
        d.dedup();
        let (mn, mx) = vals.iter().fold((f64::MAX, f64::MIN), |a, v| (a.0.min(*v), a.1.max(*v)));
        // Live means MOVING, not merely non-zero: a constant is as dead as a
        // zero for this purpose, and a dead slot in this engine is exactly one
        // repeated value.
        let alive = d.len() > 8;
        if name.starts_with("wheel") && alive {
            live += 1;
        }
        println!(
            "{name}\tcar{off:+}\t{}\t{:.4}\t{:.4}\t{}",
            d.len(),
            mn,
            mx,
            if alive { "LIVE" } else { "dead" }
        );
    }
    println!();
    match live {
        4 => println!(
            "VERDICT: all four wheel rotations are live -- this anchor IS the copy that \
             carries the fields, so an offset measured from it is an intra-struct offset."
        ),
        0 => println!(
            "VERDICT: no wheel rotation moves -- this anchor is a BARE POSITION COPY. \
             Anything read around it is dead memory, and a regeneration anchored here \
             writes zeros into a file that will still pass every gate."
        ),
        n => println!(
            "VERDICT: {n} of 4 wheel rotations move. That is neither shape this test \
             expects, and it means the wheel-record stride or base is wrong here rather \
             than that the copy is half alive."
        ),
    }
    Ok(())
}
