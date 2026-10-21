//! Reference-free location of the car's state.
//!
//! WHY THIS EXISTS
//! ---------------
//! `locate` (in `traj.rs`) finds the vehicle struct by scanning for addresses
//! whose values track a KNOWN trajectory -- the reference ghost's own recorded
//! telemetry. That is exact and self-validating, and it is the right tool when
//! the tape being run is the reference.
//!
//! It cannot work for a tape the search has evolved. An improved candidate has
//! no recorded telemetry: it is a patched ghost that still carries its seed's
//! samples, which is precisely the blindness this whole effort removed. Match
//! against the seed's path and the candidate's true position is metres away, so
//! either nothing qualifies (tolerance tight) or the wrong slot does (loose).
//! Measured: an incumbent 80 ms faster than its seed already deviates 6.9 m
//! RMS, and no tolerance separates the real slot from three decoys.
//!
//! So the compensator could be fitted from measured data once, at the seed, but
//! could not RE-MEASURE the incumbent as the search moved -- the one thing that
//! would kill its remaining staleness.
//!
//! THE SIGNATURE
//! -------------
//! The structure is known from the reference-matched work, and none of it
//! depends on which tape is running:
//!
//!   * a `u32` race clock that advances by EXACTLY 10 on every tick;
//!   * the vehicle struct within a few KB of it, laid out
//!     `qw qx qy qz | x y z | vx vy vz` as f32;
//!   * position components that are finite, inside the map's bounding box, and
//!     move smoothly (a bounded step per 10 ms tick);
//!   * `d(pos)/dt` agreeing with the velocity triple stored 12 bytes later.
//!
//! That last check is the one that makes this trustworthy without a reference:
//! it is an INTERNAL consistency test between two independent parts of the
//! struct. A stale copy, a render mirror or an unrelated float triple will not
//! have a velocity slot 12 bytes on that differentiates it.
//!
//! The clock is found first because its test -- "+10 every single tick, no
//! exceptions, over hundreds of ticks" -- is essentially unforgeable, and it
//! anchors the search for everything else.

use crate::forksrv::{ForkServer, Rec};
use crate::layout::{Layout, Row};


/// A candidate vehicle-state address and how well it holds together.
#[derive(Clone, Debug)]
pub struct SelfHit {
    pub pos: u64,
    /// mean |d(pos)/dt - v| in m/s, the internal consistency score
    pub vel_err: f64,
    /// mean speed over the window, m/s -- a stationary slot proves nothing
    pub mean_speed: f64,
    pub samples: usize,
}

fn getf32(b: &[u8], o: usize) -> f64 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64
}

/// Find the race clock with no reference at all: stream a wide window and keep
/// every u32 slot that advances by exactly 10 on every tick.
/* `find_clock_blind` was here: the v1 blind clock finder. Superseded by
   `fk::locate::find_clock2`, which scans mapped windows nearest the input
   array first and confirms a slot over hundreds of ticks instead of tens.
   Deleted rather than kept: two clock finders is two answers. */

/// Given a clock, find the vehicle struct near it by internal consistency:
/// position must be finite, in-bounds, smooth, and its derivative must match
/// the velocity triple 12 bytes later.
// Not `pub`: `locate_blind` is the entry point. The `start_offset_ms` this used
// to take was never read -- it labels samples, and nothing here labels one.
fn qualify_blind_window(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    win_lo: u64,
    win_len: u32,
    stride: u64,
    ticks: u32,
    bounds: (f64, f64, f64, f64, f64, f64),
) -> Vec<SelfHit> {
    let lo = win_lo;
    let len = win_len;

    let (_j, blob) = srv.run_sampled(probe, recs, lo, len, stride, ticks, (0, len));
    let recsz = 8 + len as usize;
    let m = blob.len() / recsz;
    if m < 20 {
        if std::env::var("FKDBG").is_ok() {
            eprintln!("  window {:#x}: only {} of 150 samples", lo, m);
        }
        return Vec::new();
    }
    if m < 150 && std::env::var("FKDBG").is_ok() {
        eprintln!("DBG qualify {:#x}: SHORT stream, {} of 150 samples", lo, m);
    }
    let dt = 0.01 * stride as f64;
    let (xlo, xhi, ylo, yhi, zlo, zhi) = bounds;
    let mut out = Vec::new();
    // the velocity triple sits 12 bytes after the position triple
    for o in (0..len as usize).step_by(4) {
        if o + 24 > len as usize {
            break;
        }
        let at = |i: usize, k: usize| getf32(&blob, i * recsz + 8 + o + k);
        let ok_row = |i: usize| -> bool {
            let (x, y, z) = (at(i, 0), at(i, 4), at(i, 8));
            x.is_finite()
                && y.is_finite()
                && z.is_finite()
                && x >= xlo
                && x <= xhi
                && y >= ylo
                && y <= yhi
                && z >= zlo
                && z <= zhi
        };
        if !(0..m).all(ok_row) {
            continue;
        }
        // smoothness and the velocity cross-check
        let mut verr = 0.0;
        let mut speed = 0.0;
        let mut bad = false;
        for i in 0..m - 1 {
            let (dx, dy, dz) = (
                at(i + 1, 0) - at(i, 0),
                at(i + 1, 4) - at(i, 4),
                at(i + 1, 8) - at(i, 8),
            );
            let step = (dx * dx + dy * dy + dz * dz).sqrt();
            // 200 m/s is far beyond anything the car does; a jump beyond it
            // means this is not a smoothly integrated position
            if step / dt > 200.0 {
                bad = true;
                break;
            }
            speed += step / dt;
            let (vx, vy, vz) = (at(i, 12), at(i, 16), at(i, 20));
            if !(vx.is_finite() && vy.is_finite() && vz.is_finite()) {
                bad = true;
                break;
            }
            verr += ((dx / dt - vx).powi(2) + (dy / dt - vy).powi(2) + (dz / dt - vz).powi(2))
                .sqrt();
        }
        if bad {
            continue;
        }
        let n = (m - 1) as f64;
        out.push(SelfHit {
            pos: lo + o as u64,
            vel_err: verr / n,
            mean_speed: speed / n,
            samples: m,
        });
    }
    // a slot that never moves is trivially self-consistent and tells us nothing
    out.retain(|h| h.mean_speed > 1.0);
    out.sort_by(|a, b| a.vel_err.partial_cmp(&b.vel_err).unwrap());
    out
}

/// The full reference-free locate.
///
/// Returns the same `Layout` the reference-matched path produces, so every
/// downstream consumer is unchanged. `rms`/`max_dev` carry the internal
/// velocity-consistency error instead of a deviation from a known path --
/// there is no known path here, which is the entire point.
pub fn locate_blind(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    start_offset_ms: i32,
    stride: u64,
    bounds: (f64, f64, f64, f64, f64, f64),
    verbose: bool,
) -> Result<Layout, String> {
    // 1. the clock, by its unforgeable +10 signature. Sweep outward from the
    //    heap in windows; the clock and the vehicle state are near each other
    //    (measured -7916, -11268, -14780 on different runs) but the sign and
    //    distance are not fixed, so search both ways.
    // The vehicle state and the clock both live near the input array, which
    // the server reports at handshake (`srv.base`). Measured on this build:
    // position is ~0x6f1be0 below base and the clock another 7916 below that,
    // but neither distance is stable across runs, so sweep a window around it.
    //
    // Window size is bounded by the pipe: each tick streams the whole window,
    // so 64 KB x 120 ticks is ~8 MB per probe fork. Sweeping +-16 MB in 64 KB
    // slices would be 256 forks, so the sweep starts where the state actually
    // was and widens only if it has to.
    // MEASURED: the vehicle struct sits a little under 590 KB below the input
    // array, whose address the server reports at handshake (`srv.base`) --
    // pos-base came out -603936, -603296, -603296 on three runs of the same
    // map. The absolute address moves with ASLR and the bimodal heap, but the
    // DISTANCE is stable to within a few hundred bytes, so a small window
    // around it is all that has to be searched. That keeps each probe fork's
    // pipe traffic to a few MB instead of streaming the whole heap.
    //
    // Only mapped, writable regions are scanned: the sampler reads the window
    // every tick inside the child, and an unmapped page would kill it.
    let base = srv.base;
    let regions = crate::procmem::maps(srv.pid());
    let mapped = |a: u64, l: u32| -> bool {
        regions
            .iter()
            .any(|r| r.perms.contains('w') && a >= r.start && a + l as u64 <= r.end)
    };
    let slice: u32 = 64 * 1024;
    let centre = base.saturating_sub(603_616);
    let mut wins: Vec<u64> = Vec::new();
    for k in 0..24u64 {
        for cand in [
            centre + k * slice as u64,
            centre.saturating_sub((k + 1) * slice as u64),
        ] {
            let a = cand & !0xFFF;
            if mapped(a, slice) {
                wins.push(a);
            }
        }
    }
    if verbose {
        println!(
            "blind locate: base {:#x}, searching {} mapped 64 KB windows around {:#x}",
            base,
            wins.len(),
            centre
        );
    }

    // TWO PHASES, because the sampler's per-fork budget is spent on window
    // WIDTH x tick COUNT. A 64 KB window only streams a handful of ticks
    // before the budget runs out -- enough to shortlist, not enough to judge.
    //
    // Phase 1: wide and shallow. A few ticks of each 64 KB window is enough to
    // reject everything that is not a finite, in-bounds, moving float triple.
    // Phase 2: narrow and deep. Re-sample each survivor as a 24-byte window
    // over many ticks, which affords the real test -- does d(pos)/dt match the
    // velocity triple stored 12 bytes on.
    let dbg = std::env::var("FKDBG").is_ok();
    let t_p1 = std::time::Instant::now();
    let mut shortlist: Vec<u64> = Vec::new();
    let mut nwin = 0usize;
    for w in &wins {
        let cands = shortlist_window(srv, probe, recs, *w, slice, stride, bounds);
        nwin += 1;
        if dbg {
            eprintln!("DBG win {:#x}: {} candidates", w, cands.len());
        }
        shortlist.extend(cands);
        if shortlist.len() > 400 {
            break;
        }
    }
    if verbose {
        println!("blind locate: {} shortlisted float triples", shortlist.len());
    }
    if dbg {
        eprintln!(
            "DBG phase1: {} windows of {} scanned, {} shortlisted, {:.1}s",
            nwin,
            wins.len(),
            shortlist.len(),
            t_p1.elapsed().as_secs_f64()
        );
    }
    let t_p2 = std::time::Instant::now();
    let full = std::env::var("FKBLIND_FULL").is_ok();
    let cap = if full { usize::MAX } else { 400 };
    let mut hits: Vec<SelfHit> = Vec::new();
    let mut examined = 0usize;
    let mut winner_idx: i64 = -1;
    for (ix, a) in shortlist.iter().take(cap).enumerate() {
        let mut got = qualify_blind_window(srv, probe, recs, *a, 24, stride, 150, bounds);
        for h in got.iter_mut() {
            h.pos = *a;
        }
        if dbg && !got.is_empty() {
            eprintln!(
                "DBG   [{}] {:#x} verr {:.3} speed {:.1} samples {}",
                ix, a, got[0].vel_err, got[0].mean_speed, got[0].samples
            );
        }
        hits.append(&mut got);
        examined = ix + 1;
        if !full && hits.iter().any(|h| h.vel_err < 1.5) {
            winner_idx = ix as i64;
            break;
        }
    }
    if dbg {
        eprintln!(
            "DBG phase2: examined {} of {} (cap 400), early-break at {}, {:.1}s",
            examined,
            shortlist.len(),
            winner_idx,
            t_p2.elapsed().as_secs_f64()
        );
        let mut sorted = hits.clone();
        sorted.sort_by(|a, b| a.vel_err.partial_cmp(&b.vel_err).unwrap());
        for h in sorted.iter().take(20) {
            eprintln!(
                "DBG hit {:#x} (base{:+}) verr {:.4} speed {:.2} samples {}",
                h.pos,
                h.pos as i64 - base as i64,
                h.vel_err,
                h.mean_speed,
                h.samples
            );
        }
    }
    hits.sort_by(|a, b| a.vel_err.partial_cmp(&b.vel_err).unwrap());
    if verbose {
        for h in hits.iter().take(3) {
            println!(
                "  candidate {:#x}  vel_err {:.3} m/s  mean_speed {:.1} m/s  ({} samples)",
                h.pos, h.vel_err, h.mean_speed, h.samples
            );
        }
    }
    let hit = hits
        .into_iter()
        .next()
        .ok_or("no self-consistent vehicle state found: state not located")?;
    if hit.vel_err > 5.0 {
        return Err(format!(
            "best candidate is not self-consistent enough (vel_err {:.2} m/s): refusing to guess",
            hit.vel_err
        ));
    }

    // 2. the clock, by its unforgeable +10-every-tick signature, near the state
    let (clock, bias) = crate::layout::find_clock(srv, probe, recs, start_offset_ms, hit.pos, 16384, 256, stride)?;
    Ok(Layout {
        pos: hit.pos,
        clock,
        clock_bias: bias,
        rms: hit.vel_err,
        max_dev: hit.vel_err,
    })
}

/// Map bounding box with a generous margin, from a reference trajectory.
pub fn bounds_from(rows: &[Row], margin: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut b = (f64::MAX, f64::MIN, f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for r in rows {
        b.0 = b.0.min(r.x);
        b.1 = b.1.max(r.x);
        b.2 = b.2.min(r.y);
        b.3 = b.3.max(r.y);
        b.4 = b.4.min(r.z);
        b.5 = b.5.max(r.z);
    }
    (
        b.0 - margin,
        b.1 + margin,
        b.2 - margin,
        b.3 + margin,
        b.4 - margin,
        b.5 + margin,
    )
}

/// Phase 1: from a few ticks of a wide window, return every 4-byte-aligned
/// offset whose float triple is finite, inside the map, and actually moving.
fn shortlist_window(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    lo: u64,
    len: u32,
    stride: u64,
    bounds: (f64, f64, f64, f64, f64, f64),
) -> Vec<u64> {

    let (_j, blob) = srv.run_sampled(probe, recs, lo, len, stride, 6, (0, len));
    let recsz = 8 + len as usize;
    let m = blob.len() / recsz;
    if std::env::var("FKDBG").is_ok() && m < 6 {
        eprintln!("DBG shortlist window {:#x}: only {} of 6 samples arrived", lo, m);
    }
    if m < 3 {
        return Vec::new();
    }
    let (xlo, xhi, ylo, yhi, zlo, zhi) = bounds;
    let at = |i: usize, o: usize| getf32(&blob, i * recsz + 8 + o);
    let mut out = Vec::new();
    for o in (0..len as usize).step_by(4) {
        if o + 24 > len as usize {
            break;
        }
        let inb = |i: usize| {
            let (x, y, z) = (at(i, o), at(i, o + 4), at(i, o + 8));
            x.is_finite()
                && y.is_finite()
                && z.is_finite()
                && x >= xlo && x <= xhi && y >= ylo && y <= yhi && z >= zlo && z <= zhi
        };
        if !(0..m).all(inb) {
            continue;
        }
        // it has to move: a constant triple is some other vector entirely
        let moved = (1..m).any(|i| {
            (at(i, o) - at(i - 1, o)).abs()
                + (at(i, o + 4) - at(i - 1, o + 4)).abs()
                + (at(i, o + 8) - at(i - 1, o + 8)).abs()
                > 1e-4
        });
        if moved {
            out.push(lo + o as u64);
        }
    }
    out
}
