//! `fk` — the driver for the TM2020 dedicated server used as a physics oracle.
//!
//! # What this crate is for, and where its edge is
//!
//! There are exactly three things `fk` can do that nothing else in this project
//! can, and all three require a **live engine**:
//!
//! 1. **Stop the simulation mid-run and resume it with different inputs.**
//!    `lroundf` is called ~25.5 times per simulated millisecond and nothing else
//!    in the simulation is non-deterministic, so an `LD_PRELOAD` interposer can
//!    count calls and halt at an exact point. `fork()` from that halted state is
//!    a complete simulator that costs ~11 ms instead of a from-scratch run.
//!    See [`server`].
//! 2. **Read the validator-controlled car's own state out of the running
//!    engine.** Identity comes from a one-shot hook on the validator's
//!    simulation-binding callback and a fixed ownership chain to its sole
//!    participant and `CGameVehiclePhy`; no state-shaped records are ranked.
//!    See [`validator`].
//! 3. **Watch a run tick by tick inside the fork child and abort it early.**
//!    See [`predicate`].
//!
//! Everything about a **file** — the input chunk, the telemetry record, the
//! declared time, the carried map, identity — belongs to the `ghost` crate and
//! this crate calls it. That boundary is not a preference. Two implementations
//! of the input-chunk codec is how this project got silent corruption before,
//! and `fk` used to be the second one.
//!
//! # The three claims this crate makes, and the controls behind them
//!
//! * *A fork resume gives the same answer as a from-scratch validation.*
//!   Control: `fk server check`, which runs both on the same candidates.
//!   **Its regime is narrow** — see the warning below.
//! * *The trajectory read out of memory is the car's.* Control: three
//!   independent tests over every row ([`layout::check_rows`]) plus, when a
//!   reference recording exists, a known-answer comparison against it.
//! * *The tape being simulated is the tape that was asked for.* Control:
//!   [`layout::verify_tape`] reads the engine's own decoded input array back
//!   and compares it tick for tick. This is not paranoia: two runs sharing a
//!   work directory swapped replays in production 17% of the time, and the
//!   result is a real, self-consistent trajectory of a car that drove somewhere
//!   else. Nothing internal can see it.
//!
//! # THE REGIME LIMIT — read this before believing a fork-reported time
//!
//! The fork server was measured exact on **4700 of 4700** candidates, but every
//! one of those perturbed a human reference tape by a few ticks at a checkpoint
//! 48–99% of the way through the run. Outside that regime it lies: on
//! cold-start work, **0 of 312 fork-reported finishes survived a full
//! `/validatepath` of the byte-identical bitstream**. One tape gave DNF from
//! boundary 170 and a finish from boundary 305 with the same inputs, and the
//! file was a DNF.
//!
//! So: late-window perturbation of a human seed is inside the validated regime.
//! A re-derived prefix, a structural splice, a cold start or a segment
//! traversal is **not**, and its headline must be confirmed with the plain
//! oracle on the written file. A banked incumbent is not a result until the
//! plain oracle re-simulates the tape on disk.

pub mod carrier;
pub mod cmd;
/// The CLOCK-FIRST locator: find the engine's race clock, then key every
/// sample on it. The reference-free locator the SEARCH uses is
/// `forkoracle::blind`; the difference is that this one is correct when the car
/// does not move (a respawn) or the engine writes the state twice in a tick.
pub mod locate;
pub mod oracle;
pub mod ptr;
pub mod record;
pub mod session;
pub mod tape;
pub mod traj;
pub mod validator;
pub mod vislayout;

/// Print a fatal error and exit 2. Reserved for a caller's mistake (a missing
/// flag, an unreadable file). A measurement that fails its own control exits 3,
/// so the two are distinguishable from a script.
pub fn die(m: impl AsRef<str>) -> ! {
    eprintln!("fk: {}", m.as_ref());
    std::process::exit(2)
}

/// Print why a measurement cannot be trusted and exit 3.
///
/// Every locate, every resume and every trajectory has a way of being wrong
/// that looks exactly like being right. When one of those checks fails the only
/// safe thing is to produce no number at all: a fallback here is how a plausible
/// answer 2–3 ms off gets banked.
pub fn abort(m: impl AsRef<str>) -> ! {
    eprintln!("fk: ABORT: {}", m.as_ref());
    std::process::exit(3)
}

/// Milliseconds as **seconds with a decimal** — `22.730`, never `22730`.
/// Every time this project prints, in prose, in tables and in output, is in
/// this form.
pub fn secs(ms: i64) -> String {
    let neg = ms < 0;
    let a = ms.abs();
    format!("{}{}.{:03}", if neg { "-" } else { "" }, a / 1000, a % 1000)
}

/// `Some(ms)` as seconds, `None` as `DNF`.
pub fn secs_opt(ms: Option<i64>) -> String {
    match ms {
        Some(v) => secs(v),
        None => "DNF".into(),
    }
}
