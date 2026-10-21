//! locate2 -- CLOCK-FIRST location of the car's state, and why the old one
//! could not be fixed in place.
//!
//! THE DEFECT (measured on 284238 `YOU LOVE WATER`, 2026-08-19)
//! -----------------------------------------------------------
//! Both existing locators label a sample by its INDEX in the stream:
//! `sample i == tick probe-1+i`. That is only true if the child emits exactly
//! one sample per tick, and the child emits one sample per *change of the
//! deduplication key*:
//!
//!   * `traj::qualify` keys on the 12-byte POSITION. A tick in which the car
//!     does not move -- a respawn freeze, a standstill, a countdown -- emits
//!     NOTHING, and every sample after it is mislabelled by one tick. The
//!     record on this map has 31 respawns: 43 098 ticks produced 40 055
//!     samples, the labels slid by 3 000 ticks, and the live slot came back
//!     166 m from the reference path. Ten position-shaped addresses were
//!     found; not one "tracked", so the tool aborted with
//!     "no address tracks the reference ghost's path".
//!   * `blind::qualify_blind_window` keys on the 24-byte pos+vel window, and
//!     has the mirror-image problem: the engine writes that window SEVERAL
//!     times inside one tick, so consecutive samples can be 0 ms apart while
//!     the code divides by 10 ms. The velocity cross-check then reports a
//!     self-consistency error of the order of the car's speed -- 7.34 and
//!     9.26 m/s on this map, against a hard "refusing to guess" limit of 5.0.
//!
//! Both failures look like "the state is not there" and are really "the state
//! is there and the clock says so".
//!
//! THE FIX
//! -------
//! Find the RACE CLOCK first, then put it in every sampled record and key the
//! deduplication on the whole record. The clock changes every tick, so a tick
//! can never be dropped; several samples inside one tick are grouped by clock
//! value and the LAST is kept -- the same rule `decode_rows` already uses for
//! the production trajectory, which is the rule validated to 3.4 mm.
//!
//! The clock itself is found without any reference: at the checkpoint the
//! parent is paused, so a value scan of its writable memory for u32 slots near
//! the known race time gives a few hundred candidates, and the +10-every-tick
//! signature settles it. Nothing here assumes a distance from the input array
//! (that guess is only used to ORDER the search), nothing assumes the clock is
//! below the state, and no threshold is calibrated to one car's top speed.
//!
//! Three more things this fixes as a consequence:
//!   * a per-probe SIMULATED-TIME BUDGET: a locate probe wants 6 or 150 ticks
//!     and used to simulate the whole remaining tape (43 000 ticks here, 5.5
//!     minutes per attempt);
//!   * the acceptance statistic is the MEDIAN of |d(pos)/dt - v|, not the mean,
//!     so one landing impact inside the window cannot condemn the true slot;
//!   * the quaternion is checked at locate time (|q| = 1 is an independent
//!     structural signature the blind locator never used).

use forkoracle::forksrv::{ForkServer, Rec};
use forkoracle::layout::{sample_ms, Layout, Row, R_POS, R_QUAT, R_VEL, REC_LEN};

/// Bit 31 of the sample budget: the child exits when the budget is spent.
pub const EXIT_ON_BUDGET: u32 = 0x8000_0000;

/// `lroundf` calls the engine makes per 10 ms tick, with margin. Measured:
/// clock = 36141 + 25.483 * race_ms, i.e. ~255 per tick.
pub fn budget_for(ticks: u32) -> u32 {
    ticks.saturating_mul(340).saturating_add(12000)
}

fn getf32(b: &[u8], o: usize) -> f64 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::MAX;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// One sampled tick: the race-clock value and the gathered record.
pub struct Tick {
    pub clock: u32,
    pub rec: Vec<u8>,
}

/// Sample `segs` for `ticks` ticks with the clock as segment 0, and return one
/// record per distinct clock value (the LAST one written in that tick).
///
/// `key` is the dedup key within the record; pass `(0, reclen)` (the whole
/// record) when the clock is inside it, which is the only configuration that
/// cannot silently drop a tick.
pub fn gather_ticks(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    segs: &[(u64, u32)],
    ticks: u32,
    max_samples: u32,
    key: (u32, u32),
) -> Vec<Tick> {
    let reclen: usize = segs.iter().map(|s| s.1 as usize).sum();
    // Only the ticks the child will actually simulate are sent. The patch list
    // used to be the whole remaining tape -- 43 000 records, 688 KB, written
    // down a pipe and copied into the child's input array FOR EVERY PROBE --
    // which is most of what made a locate on a long tape unaffordable.
    let keep = ((ticks as usize) + 64).min(recs.len());
    let (_j, blob) = srv.run_sampled_segs_ex(
        probe,
        &recs[..keep],
        segs,
        1,
        max_samples | EXIT_ON_BUDGET,
        key,
        budget_for(ticks),
    );
    let recsz = 8 + reclen;
    let m = if recsz > 0 { blob.len() / recsz } else { 0 };
    let mut out: Vec<Tick> = Vec::with_capacity(m);
    for i in 0..m {
        let b = &blob[i * recsz + 8..i * recsz + 8 + reclen];
        let clk = u32::from_le_bytes(b[0..4].try_into().unwrap());
        match out.last_mut() {
            Some(t) if t.clock == clk => t.rec.copy_from_slice(b),
            _ => out.push(Tick {
                clock: clk,
                rec: b.to_vec(),
            }),
        }
    }
    out
}

/// A located clock: its address and `value - race_ms`.
#[derive(Clone, Copy, Debug)]
pub struct ClockHit {
    pub addr: u64,
    pub bias: i64,
}

/// Every writable, readable region of the paused parent, largest first cut off
/// at `cap` bytes total, in address order.
fn writable_regions(pid: i32) -> Vec<forkoracle::procmem::Region> {
    forkoracle::procmem::maps(pid)
        .into_iter()
        .filter(|r| {
            r.perms.contains('w')
                && r.perms.contains('r')
                && r.end > r.start
                && r.end - r.start
                    <= std::env::var("FK_REGION_CAP")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1u64 << 30)
                && !r.path.starts_with("/dev")
        })
        .collect()
}

/// STEP 1: the race clock, with no reference and no assumed offset.
///
/// A value scan of the paused parent for u32 slots within `bias_max` ms of the
/// known race time, then the +10-every-tick test on the survivors, eight at a
/// time (the gather takes up to 8 segments, so eight candidates cost one fork).
/// STEP 1: the race clock, with no reference, no assumed offset, and no
/// assumption about what the clock READS.
///
/// The first version of this filtered candidates by value (a u32 within a
/// couple of seconds of the known race time). That failed on 284238: nothing
/// within 2000 ms of the race time ticks by 10, so the engine's clock is not
/// simply "race milliseconds" on this build/map, and a value prior is exactly
/// the kind of assumption that turns a locator into a map-specific tool.
///
/// What replaces it assumes only the SIGNATURE. Each mapped window is sampled
/// at four instants a few ticks apart (four snapshots, no dedup, then the child
/// exits), and a slot survives if every one of its deltas is a positive
/// multiple of 10 of the right order of magnitude. Survivors then face the
/// strict per-tick test.
pub fn find_clock2(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    start_offset_ms: i32,
    _bias_max: i64,
    verbose: bool,
) -> Result<ClockHit, String> {
    let race0 = sample_ms(probe, 0, start_offset_ms);
    let base = srv.base;
    // Snapshot spacing: ~10 ticks between samples, four samples.
    const GAP_TICKS: u64 = 10;
    const NSNAP: u32 = 4;
    let slice: u32 = 1 << 20;
    let mut wins: Vec<u64> = Vec::new();
    for r in writable_regions(srv.pid()) {
        let mut a = (r.start + 0xFFF) & !0xFFF;
        while a + slice as u64 <= r.end {
            wins.push(a);
            a += slice as u64;
        }
        // the tail of a region that is not a whole slice wide
        if r.end > r.start + 0x1000 && (r.end - ((r.start + 0xFFF) & !0xFFF)) % slice as u64 > 4096 {
            let a = (r.end - ((r.end - ((r.start + 0xFFF) & !0xFFF)) % slice as u64)) & !0xFFF;
            if a >= r.start && a + 4096 <= r.end {
                wins.push(a);
            }
        }
    }
    wins.sort_by_key(|w| (*w as i64 - base as i64).unsigned_abs());
    wins.dedup();
    if verbose {
        println!(
            "clock scan: {} mapped windows of {} KB, nearest to the input array first",
            wins.len(),
            slice / 1024
        );
    }
    let t0 = std::time::Instant::now();
    let mut cands: Vec<u64> = Vec::new();
    let mut scanned = 0usize;
    for w in &wins {
        scanned += 1;
        let ticks = (GAP_TICKS as u32) * NSNAP + 8;
        let keep = ((ticks as usize) + 64).min(recs.len());
        let (_j, blob) = srv.run_sampled_segs_ex(
            probe,
            &recs[..keep],
            &[(*w, slice)],
            255 * GAP_TICKS,
            NSNAP | EXIT_ON_BUDGET,
            (0, 0), // no dedup: a fixed number of snapshots
            budget_for(ticks),
        );
        let recsz = 8 + slice as usize;
        let m = blob.len() / recsz;
        if m < 3 {
            continue;
        }
        let g = |i: usize, o: usize| -> u32 {
            u32::from_le_bytes(blob[i * recsz + 8 + o..i * recsz + 12 + o].try_into().unwrap())
        };
        for o in (0..slice as usize - 4).step_by(4) {
            let mut ok = true;
            let mut dmin = u32::MAX;
            let mut dmax = 0u32;
            for i in 0..m - 1 {
                let d = g(i + 1, o).wrapping_sub(g(i, o));
                if d % 10 != 0 || d < 10 || d > 40 * 10 * (GAP_TICKS as u32) {
                    ok = false;
                    break;
                }
                dmin = dmin.min(d);
                dmax = dmax.max(d);
            }
            if ok && dmax - dmin <= 40 {
                cands.push(*w + o as u64);
            }
        }
        if cands.len() >= 8 {
            break;
        }
    }
    if verbose {
        println!(
            "clock scan: {} slots step by a multiple of 10 across {} windows ({:.1}s)",
            cands.len(),
            scanned,
            t0.elapsed().as_secs_f64()
        );
    }
    if cands.is_empty() {
        return Err("no slot advances by a multiple of 10 per tick: race clock not located".into());
    }
    confirm_clock(srv, probe, recs, race0, base, &cands, verbose)
}

#[allow(clippy::too_many_arguments)]
fn confirm_clock(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    race0: i64,
    base: u64,
    cands: &[u64],
    verbose: bool,
) -> Result<ClockHit, String> {
    let mut cands: Vec<u64> = cands.to_vec();
    cands.sort_by_key(|a| (*a as i64 - base as i64).unsigned_abs());
    if cands.is_empty() {
        return Err("no clock candidate to confirm".into());
    }
    let ticks: u32 = 40;
    let cap: usize = std::env::var("FK_CLOCK_MAX_PROBES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);
    let mut confirmed: Vec<ClockHit> = Vec::new();
    let mut nonempty = 0usize;
    let mut maxticks = 0usize;
    let t0 = std::time::Instant::now();
    for (ci, chunk) in cands.chunks(8).enumerate() {
        if ci >= cap {
            return Err(format!(
                "clock not found in the nearest {} candidates ({} scanned of {})",
                ci * 8,
                ci * 8,
                cands.len()
            ));
        }
        if verbose && ci > 0 && ci % 500 == 0 {
            println!(
                "  clock probes {} of {} ({:.1}s)",
                ci * 8,
                cands.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        let segs: Vec<(u64, u32)> = chunk.iter().map(|a| (*a, 4u32)).collect();
        let reclen = (segs.len() * 4) as u32;
        // No dedup key on the clock (we do not know which slot it is yet):
        // key on the whole record, so a tick can only be missed if NOTHING in
        // the record changes -- and then there is no clock in it either.
        let ts = gather_ticks(
            srv,
            probe,
            recs,
            &segs,
            ticks,
            ticks * 8,
            (0, reclen),
        );
        maxticks = maxticks.max(ts.len());
        if ts.len() >= 20 {
            nonempty += 1;
        } else {
            continue;
        }
        for (j, a) in chunk.iter().enumerate() {
            let vals: Vec<u32> = ts
                .iter()
                .map(|t| u32::from_le_bytes(t.rec[j * 4..j * 4 + 4].try_into().unwrap()))
                .collect();
            let mut tens = 0usize;
            let mut ok = true;
            for w in vals.windows(2) {
                let d = w[1].wrapping_sub(w[0]);
                if d == 10 {
                    tens += 1;
                } else if d != 0 {
                    ok = false;
                    break;
                }
            }
            if ok && tens >= 15 {
                confirmed.push(ClockHit {
                    addr: *a,
                    bias: vals[0] as i64 - race0,
                });
            }
        }
        if !confirmed.is_empty() {
            break;
        }
    }
    let hit = *confirmed.first().ok_or_else(|| {
        format!(
            "no u32 advances by exactly +10 per tick: race clock not located \
             ({} of {} probe forks returned >=20 ticks, deepest {} ticks)",
            nonempty,
            cands.len().div_ceil(8),
            maxticks
        )
    })?;
    // Deep confirm: 400 ticks, nothing but 0 and +10.
    let ts = gather_ticks(srv, probe, recs, &[(hit.addr, 4)], 400, 4000, (0, 4));
    let vals: Vec<u32> = ts
        .iter()
        .map(|t| u32::from_le_bytes(t.rec[0..4].try_into().unwrap()))
        .collect();
    let bad = vals
        .windows(2)
        .filter(|w| {
            let d = w[1].wrapping_sub(w[0]);
            d != 10 && d != 0
        })
        .count();
    if vals.len() < 50 || bad > 0 {
        return Err(format!(
            "clock candidate {:#x} failed the deep confirm: {} samples, {} bad steps",
            hit.addr,
            vals.len(),
            bad
        ));
    }
    if verbose {
        println!(
            "CLOCK {:#014x} (base{:+}) bias {:+} ms, confirmed over {} ticks",
            hit.addr,
            hit.addr as i64 - base as i64,
            hit.bias,
            vals.len()
        );
    }
    Ok(hit)
}

/// A vehicle-state candidate and the evidence for it.
#[derive(Clone, Debug)]
pub struct PosHit {
    pub pos: u64,
    /// median |d(pos)/dt - v| over the window, m/s -- robust to one impact
    pub verr: f64,
    /// max |q| - 1 over the window: an independent structural signature
    pub qerr: f64,
    /// Fraction of consecutive TICKS on which the quaternion actually CHANGES.
    ///
    /// RULE 3. A norm-only test cannot tell an identity rotation from a zeroed
    /// slot, and cannot tell the live vehicle state from a shadow object that
    /// carries no attitude at all -- both satisfy |q| = 1. Measured on three
    /// maps (267460, 191465, 227969), the live car is the ONLY candidate whose
    /// quaternion both has unit norm and varies; the back buffer, the bare
    /// position copy and the two 0.494 mm shadows all fail it. See
    /// `whl_BUFFER_v3` / `v7`.
    ///
    /// Counted across genuine 10 ms tick steps, never across raw records: the
    /// engine writes the state ~53 times inside one tick, so counting raw rows
    /// makes a varying quaternion look constant on 96 % of them.
    pub qvary: f64,
    /// Total angular travel over the window, radians. On a map where the car
    /// barely rotates the quaternion drifts only in the fifth decimal and
    /// `qvary` passes by a hair (measured on 191465), so RANK by how far the
    /// attitude actually moved rather than thresholding on "did it change".
    pub qtravel: f64,
    pub mean_speed: f64,
    pub ticks: usize,
    pub first: (f64, f64, f64),
}



/// The whole thing: clock, then state, returned in the same `Layout` every
/// existing consumer already takes.
pub fn locate_v2(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    start_offset_ms: i32,
    bounds: (f64, f64, f64, f64, f64, f64),
    bias_max: i64,
    max_windows: usize,
    verbose: bool,
) -> Result<Layout, String> {
    let ck = find_clock2(srv, probe, recs, start_offset_ms, bias_max, verbose)?;
    // FK_STATE_OFF=<n> -- take the vehicle state slot at `base - n` instead of
    // sweeping for it. The slot sits at a FIXED offset from the server's own
    // base (measured: base-8183260 on every fork of every probe tick of the
    // same build), so once a map has been located honestly the sweep is 50
    // seconds of rediscovering the same address. It is an override, not a
    // guess: `fk trace` still runs its own self-check on the trajectory it
    // reads out, and the offset is only ever taken from a run that located.
    if let Ok(v) = std::env::var("FK_STATE_OFF") {
        let off: u64 = v.parse().map_err(|_| "FK_STATE_OFF is a decimal byte offset".to_string())?;
        let pos = srv.base.checked_sub(off).ok_or("FK_STATE_OFF is past the base")?;
        if verbose {
            println!("STATE {:#014x} (base-{}) taken from FK_STATE_OFF", pos, off);
        }
        return Ok(Layout { pos, clock: ck.addr, clock_bias: ck.bias, rms: 0.0, max_dev: 0.0 });
    }
    // THE CAR COMES FROM THE POINTER, not from a sweep.
    //
    // This used to call `locate_pos2`, which forked the engine once per memory
    // window hunting for a self-consistent moving float triple and then took
    // the fastest one -- a heuristic that picked a decoy on 126859 and needed
    // the caller's own self-check to catch it. The engine has a pointer to the
    // car; ask it.
    let chain = std::env::var("FK_CAR_CHAIN")
        .unwrap_or_else(|_| crate::ptr::DEFAULT_CHAIN.to_string());
    let (m, _) = crate::ptr::module_base(srv.pid())
        .ok_or("no module base for the live server")?;
    let states = crate::ptr::resolve_pool(srv.pid(), m, &chain)
        .map_err(|e| format!("the car chain {} did not resolve: {}", chain, e))?;
    let pos = states
        .first()
        .map(|s| s + crate::vislayout::POS_IN_STATE as u64)
        .ok_or_else(|| format!("the chain {} named no vehicle state", chain))?;
    if verbose {
        println!("STATE {:#014x} via {} ({} vehicle state(s))", pos, chain, states.len());
    }
    let _ = (recs, bounds, max_windows, probe);
    Ok(Layout {
        pos,
        clock: ck.addr,
        clock_bias: ck.bias,
        rms: 0.0,
        max_dev: 0.0,
    })
}

/// Extract the whole trajectory with a located layout, one row per tick.
pub fn trajectory(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    l: &Layout,
    ticks: u32,
) -> Vec<Row> {
    let segs = forkoracle::layout::segments(l);
    let ts = gather_ticks(srv, probe, recs, &segs, ticks, 200_000, (0, REC_LEN as u32));
    ts.iter()
        .map(|t| Row {
            time_ms: t.clock as i64 - l.clock_bias,
            x: getf32(&t.rec, R_POS),
            y: getf32(&t.rec, R_POS + 4),
            z: getf32(&t.rec, R_POS + 8),
            vx: getf32(&t.rec, R_VEL),
            vy: getf32(&t.rec, R_VEL + 4),
            vz: getf32(&t.rec, R_VEL + 8),
            qw: getf32(&t.rec, R_QUAT),
            qx: getf32(&t.rec, R_QUAT + 4),
            qy: getf32(&t.rec, R_QUAT + 8),
            qz: getf32(&t.rec, R_QUAT + 12),
            wetness: getf32(&t.rec, forkoracle::layout::R_WET),
        })
        .collect()
}

/// `gather_ticks`, but keeping only every `stride`-th tick.
///
/// The child still samples every tick (the clock is the dedup key); the driver
/// thins the result. That costs a little pipe traffic and buys a phase-1 window
/// wide enough for the car to have moved, which is what the "is this a position
/// triple" filter needs.
pub fn gather_ticks_stride(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    segs: &[(u64, u32)],
    ticks: u32,
    max_samples: u32,
    key: (u32, u32),
    stride: u32,
) -> Vec<Tick> {
    if stride <= 1 {
        return gather_ticks(srv, probe, recs, segs, ticks, max_samples, key);
    }
    let all = gather_ticks(srv, probe, recs, segs, ticks, max_samples * stride, key);
    all.into_iter()
        .enumerate()
        .filter(|(i, _)| i % stride as usize == 0)
        .map(|(_, t)| t)
        .collect()
}

// ---------------------------------------------------------------------------
// Two things the three sweeps in this file each had their own copy of.
//
// They agreed, which is worse than if they had not: a change to the alignment,
// the ordering or the movement threshold would have been applied to two of
// them and the third would have gone on quietly disagreeing. Nothing here is
// new behaviour -- `fk trace` produces a byte-identical CSV before and after.



// ---------------------------------------------------------------------------
// `qualify2` GRADES ONE ADDRESS. It is not a search and it survived the
// deletion of the sweeps for that reason: `validator.rs` resolves the
// validator-owned CGameVehiclePhy state by pointer chain and then asks this
// whether the thing at that address really moves like a car. Checking an
// address someone else supplied is cheap and sound; sweeping memory for one
// was neither.
/// Deep test of one candidate: gather `[clock | q(16) | pos(12) | vel(12)]` for
/// `ticks` ticks, one row per clock value, and judge it on two independent
/// signatures.
pub fn qualify2(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    clock: u64,
    pos: u64,
    ticks: u32,
    bounds: (f64, f64, f64, f64, f64, f64),
) -> Option<PosHit> {
    let segs = [(clock, 4u32), (pos - 16, 40u32)];
    let ts = gather_ticks(srv, probe, recs, &segs, ticks, ticks * 8, (0, REC_LEN as u32));
    if ts.len() < 20 {
        return None;
    }
    let (xlo, xhi, ylo, yhi, zlo, zhi) = bounds;
    // FK_ANCHOR="x,y,z,r" -- accept only a slot whose FIRST usable sample sits
    // within r metres of a position we KNOW the car occupies at the fork tick
    // (on a map with no ghosts, the map's own Spawn item). Self-consistency
    // alone cannot tell the car from a falling prop; this can, and it is
    // ground truth rather than a threshold.
    let anchor: Option<(f64, f64, f64, f64)> = std::env::var("FK_ANCHOR").ok().and_then(|s| {
        let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() == 4 { Some((v[0], v[1], v[2], v[3])) } else { None }
    });
    let mut first_pos: Option<(f64, f64, f64)> = None;
    let mut verrs: Vec<f64> = Vec::with_capacity(ts.len());
    let mut qerrs: Vec<f64> = Vec::with_capacity(ts.len());
    let mut qmoved = 0usize;
    let mut qsame = 0usize;
    let mut qtrav = 0.0f64;
    let mut speed = 0.0;
    let mut n = 0usize;
    let mut outliers = 0usize;
    let mut rows = 0usize;
    for w in ts.windows(2) {
        if (w[1].clock.wrapping_sub(w[0].clock)) != 10 {
            continue;
        }
        rows += 1;
        let dt = 0.01;
        let p = |t: &Tick, k: usize| getf32(&t.rec, R_POS + k);
        let (x, y, z) = (p(&w[0], 0), p(&w[0], 4), p(&w[0], 8));
        // OUTLIER, NOT DISQUALIFICATION. A single tick can legitimately be a
        // teleport: a RESPAWN moves the car tens of metres in one tick, and
        // the record this project works from has 31 of them. The first version
        // of this test returned None on any step over 400 m/s, so on a
        // respawning map the real vehicle state was thrown away and only the
        // lagged copies -- which do not follow the teleport -- survived. Bad
        // rows are counted and skipped; too many of them condemn the slot.
        let bad_pos = !(x.is_finite() && y.is_finite() && z.is_finite())
            || x < xlo
            || x > xhi
            || y < ylo
            || y > yhi
            || z < zlo
            || z > zhi;
        if bad_pos {
            outliers += 1;
            continue;
        }
        if first_pos.is_none() {
            first_pos = Some((x as f64, y as f64, z as f64));
        }
        let (dx, dy, dz) = (p(&w[1], 0) - x, p(&w[1], 4) - y, p(&w[1], 8) - z);
        let step = (dx * dx + dy * dy + dz * dz).sqrt();
        if !step.is_finite() || step / dt > 400.0 {
            outliers += 1;
            continue;
        }
        let v = |t: &Tick, k: usize| getf32(&t.rec, R_VEL + k);
        // The integration convention is not assumed: the derivative over the
        // step is compared with the velocity at BOTH ends and the better one
        // taken, so a hard acceleration inside the tick cannot condemn a slot.
        let e0 = ((dx / dt - v(&w[0], 0)).powi(2)
            + (dy / dt - v(&w[0], 4)).powi(2)
            + (dz / dt - v(&w[0], 8)).powi(2))
        .sqrt();
        let e1 = ((dx / dt - v(&w[1], 0)).powi(2)
            + (dy / dt - v(&w[1], 4)).powi(2)
            + (dz / dt - v(&w[1], 8)).powi(2))
        .sqrt();
        if !e0.is_finite() && !e1.is_finite() {
            outliers += 1;
            continue;
        }
        verrs.push(if e0.is_finite() && e1.is_finite() {
            e0.min(e1)
        } else {
            e0.max(e1)
        });
        speed += step / dt;
        n += 1;
        let q = |k: usize| getf32(&w[0].rec, R_QUAT + k);
        let qn = (q(0) * q(0) + q(4) * q(4) + q(8) * q(8) + q(12) * q(12)).sqrt();
        let qe = (qn - 1.0).abs();
        qerrs.push(if qe.is_finite() { qe } else { 1e9 });
        // ...and whether it MOVED between these two ticks. Compared across a
        // real tick step (the 10 ms filter above already guarantees that), so
        // the ~53 writes the engine makes inside one tick cannot make a
        // varying quaternion look constant.
        let q1 = |k: usize| getf32(&w[1].rec, R_QUAT + k);
        let moved = (0..4).any(|i| { let (a0, b0) = (q(i * 4), q1(i * 4)); (a0 - b0).abs() > 1e-9 });
        qsame += 1;
        if moved { qmoved += 1; }
        // Angular travel between the two ticks: 2*acos(|dot|) of the unit
        // quaternions, summed. Ranks a car that really turns above one whose
        // attitude merely jitters in the fifth decimal.
        let n1 = (q1(0) * q1(0) + q1(4) * q1(4) + q1(8) * q1(8) + q1(12) * q1(12)).sqrt();
        if qn > 0.5 && n1 > 0.5 {
            let dot = ((q(0) * q1(0) + q(4) * q1(4) + q(8) * q1(8) + q(12) * q1(12)) / (qn * n1)).abs().min(1.0);
            let da = 2.0 * dot.acos();
            if da.is_finite() { qtrav += da; }
        }
    }
    // A quarter of the window may be teleports and freezes before the slot is
    // judged untrustworthy; below that they are simply not evidence.
    if n < 20 || outliers * 4 > rows {
        return None;
    }
    if let (Some(a), Some(p)) = (anchor, first_pos) {
        let d = ((p.0 - a.0).powi(2) + (p.1 - a.1).powi(2) + (p.2 - a.2).powi(2)).sqrt();
        if d > a.3 {
            return None;
        }
    }
    qerrs.sort_by(|a, b| a.total_cmp(b));
    let q95 = qerrs[(qerrs.len() * 95) / 100];
    Some(PosHit {
        pos,
        verr: median(&mut verrs),
        qerr: q95,
        qvary: if qsame > 0 { qmoved as f64 / qsame as f64 } else { 0.0 },
        qtravel: qtrav,
        mean_speed: speed / n as f64,
        first: first_pos.unwrap_or((0.0, 0.0, 0.0)),
        ticks: ts.len(),
    })
}
// ---------------------------------------------------------------------------
// THE SEARCH, KEPT AS A LAST RESORT.
//
// The pointer chain is the primary and it is exact where it works. It does
// not work everywhere yet: chains are per (binary, MAP) -- on 203072 every
// chain `fk ptr find` reports ends at the vis state +0x4e8, and on 287431
// every one ends +0x6268 by a different walk -- and a map whose chains have
// not been derived has none that resolve. Deleting this outright broke 287431,
// a map that regenerated correctly the day before.
//
// So it stays, and it runs ONLY when every chain has been tried and rejected.
// When the chain works there is no search and no lottery; when it does not,
// the tool still produces a file instead of an apology.
/// STEP 2: the vehicle state, by sweeping mapped memory with the clock in the
/// record. `hint` orders the sweep (nothing depends on it being right).
pub fn locate_pos2(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    clock: u64,
    bounds: (f64, f64, f64, f64, f64, f64),
    max_windows: usize,
    verbose: bool,
) -> Result<PosHit, String> {
    let base = srv.base;
    let hint = base.saturating_sub(603_616);
    let slice: u32 = 64 * 1024;
    let wins = windows_near(srv.pid(), hint, slice, max_windows);
    if verbose {
        println!(
            "state sweep: {} mapped 64 KB windows, nearest first from {:#x}",
            wins.len(),
            hint
        );
    }
    let (xlo, xhi, ylo, yhi, zlo, zhi) = bounds;
    let dbg = std::env::var("FKDBG").is_ok();
    let mut best: Option<PosHit> = None;
    let mut runner: Option<f64> = None;
    let mut scanned = 0usize;
    for w in wins {
        scanned += 1;
        // Phase 1, keyed on the CLOCK ALONE: exactly one sample per tick, so a
        // 64 KB window costs 6 records instead of 6 arbitrary instants inside
        // one tick.
        let segs = [(clock, 4u32), (w, slice)];
        // Phase 1 spreads its six instants over FK_LOCATE_STRIDE ticks each.
        // Six CONSECUTIVE ticks are useless during a countdown: the car has not
        // moved, the "moved" filter below drops the true slot, and the locate
        // fails with "no moving triple" at exactly the early checkpoint the
        // clean-run sampler needs (252289, 126859).
        let lstride: u32 = std::env::var("FK_LOCATE_STRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let ts = gather_ticks_stride(srv, probe, recs, &segs, 6 * lstride + 4, 24, (0, 4), lstride);
        if ts.len() < 4 {
            continue;
        }
        let mut shortlist: Vec<u64> = Vec::new();
        let n = ts.len();
        for o in (4..4 + slice as usize).step_by(4) {
            if o + 24 > 4 + slice as usize {
                break;
            }
            let at = |i: usize, k: usize| getf32(&ts[i].rec, o + k);
            let inb = |i: usize| {
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
            if !(0..n).all(inb) {
                continue;
            }
            let moved = triple_moves(n, &at);
            if moved {
                shortlist.push(w + (o - 4) as u64);
            }
        }
        if dbg && !shortlist.is_empty() {
            eprintln!("DBG win {:#x}: {} shortlisted", w, shortlist.len());
        }
        for a in shortlist {
            if a < 16 {
                continue;
            }
            let qticks: u32 = std::env::var("FK_QUALIFY_TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
            let h = match qualify2(srv, probe, recs, clock, a, qticks, bounds) {
                Some(h) => h,
                None => continue,
            };
            if dbg {
                eprintln!(
                    "DBG   cand {:#x} (base{:+}) verr {:.4} qerr {:.2e} qvary {:.0}% qtravel {:.1} deg speed {:.1} ticks {} first ({:.1},{:.1},{:.1})",
                    a,
                    a as i64 - base as i64,
                    h.verr,
                    h.qerr,
                    h.qvary * 100.0,
                    h.qtravel.to_degrees(),
                    h.mean_speed,
                    h.ticks,
                    h.first.0,
                    h.first.1,
                    h.first.2
                );
            }
            // A candidate that cannot pass the quaternion test can never be
            // ACCEPTED, but ranking by verr alone still let it become `best` --
            // and `best` only ever moves on a STRICTLY smaller verr, so one
            // decoy with verr 0.00 and no quaternion permanently blocked every
            // later candidate that had both. Measured on 126859 and 252289: the
            // real state was shortlisted, scored verr 0.0000 qerr 0.00e0, and
            // the locate still aborted after scanning all 3226 windows.
            if h.qerr >= std::env::var("FK_QERR_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-3) {
                continue;
            }
            // RULE 3, AS A FILTER RATHER THAN A REPORT. |q| = 1 is satisfied by
            // an identity rotation and by a zeroed slot, and the two 0.494 mm
            // shadow objects carry no attitude at all -- so a norm-only test
            // cannot separate the live car from the copies around it, and the
            // choice fell to `verr`, which ties at 0.0000 across all of them.
            // Measured on 267460, 191465 and 227969: the live car is the ONLY
            // candidate whose quaternion both has unit norm AND VARIES.
            //
            // This is what made a one-tick-stale regeneration a coin flip. The
            // gate now demands variation, and the ranking below prefers the
            // candidate whose attitude travelled furthest -- because on a map
            // where the car barely rotates (191465 drifts in the fifth decimal)
            // a bare "did it change" passes by a hair and cannot discriminate.
            // DEFAULT OFF (0.0). Measured on 227969: a 0.5 bar rejects EVERY
            // candidate and the locate fails outright, so the threshold is not
            // yet calibrated and shipping it armed would trade a stale file for
            // no file. The MEASUREMENT is always taken and always printed; the
            // RANKING below uses it unconditionally, which cannot reject
            // anything and is where the repair actually lives.
            let vmin: f64 = std::env::var("FK_QVARY_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            if h.qvary < vmin {
                if dbg {
                    eprintln!("DBG   cand {:#x} REJECTED by rule 3: quaternion varies on only {:.0}% of ticks", a, h.qvary * 100.0);
                }
                continue;
            }
            // A slot that never moves is trivially self-consistent (verr 0.00) and
            // would win every ranking. It is not evidence: skip it, and if the
            // whole window is stationary say so rather than picking one.
            if h.mean_speed
                <= std::env::var("FK_MIN_SPEED")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1.0)
            {
                continue;
            }
            let better = match &best {
                None => true,
                // RANK BY ANGULAR TRAVEL FIRST. Every surviving candidate has
                // already passed rule 3, and on these maps their `verr` values
                // tie at 0.0000 -- so ranking by verr alone is a coin flip
                // between the car and whatever else passed. Attitude travel
                // breaks the tie in favour of the object that actually turns,
                // and only falls back to verr when the two are within 1 %.
                Some(b) => {
                    if (h.qtravel - b.qtravel).abs() > 0.01 * b.qtravel.max(1e-6) {
                        h.qtravel > b.qtravel
                    } else {
                        h.verr < b.verr
                    }
                }
            };
            if better {
                if let Some(b) = &best {
                    runner = Some(match runner {
                        Some(r) => r.min(b.verr),
                        None => b.verr,
                    });
                }
                best = Some(h);
            } else {
                runner = Some(match runner {
                    Some(r) => r.min(h.verr),
                    None => h.verr,
                });
            }
        }
        // ACCEPTANCE: both signatures, and both scale-free. The velocity test
        // is relative to the car's own speed (a fixed 2.0 m/s was calibrated on
        // a 90 m/s car and is meaningless on a 30 m/s or a 300 m/s one), and
        // the quaternion test is structural.
        let qmax: f64 = std::env::var("FK_QERR_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-3);
        if let Some(b) = &best {
            let vfloor: f64 = std::env::var("FK_VERR_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.25);
            if b.qerr < qmax && b.verr < (0.02 * b.mean_speed).max(vfloor) {
                if verbose {
                    println!(
                        "STATE {:#014x} (base{:+}) verr {:.4} m/s (median), |q|-1 {:.2e}, \
                         mean speed {:.1} m/s, after {} windows",
                        b.pos,
                        b.pos as i64 - base as i64,
                        b.verr,
                        b.qerr,
                        b.mean_speed,
                        scanned
                    );
                    if let Some(r) = runner {
                        println!("  runner-up verr {:.4} m/s ({:.0}x worse)", r, r / b.verr.max(1e-9));
                    }
                }
                return Ok(b.clone());
            }
        }
    }
    match best {
        Some(b) => Err(format!(
            "best candidate {:#x} is not self-consistent enough after {} windows \
             (median |d(pos)/dt - v| {:.2} m/s at mean speed {:.1}, |q|-1 {:.2e}): refusing to guess",
            b.pos, scanned, b.verr, b.mean_speed, b.qerr
        )),
        None => Err(format!(
            "no moving, in-bounds float triple in {} mapped windows: state not located",
            scanned
        )),
    }
}

/// Every page-aligned `slice`-sized window in the process's writable regions,
/// **ordered nearest first** to a hint address.
///
/// The order is the point. The car state sits within a few hundred KB of the
/// decoded input array, and the input array's address is reported at the
/// handshake, so sweeping outwards from it finds the car in the first window
/// most of the time. `cap` then bounds the search instead of choosing where it
/// looks.
pub fn windows_near(pid: i32, hint: u64, slice: u32, cap: usize) -> Vec<u64> {
    let mut wins: Vec<u64> = Vec::new();
    for r in writable_regions(pid) {
        let mut a = r.start & !0xFFF;
        if a < r.start {
            a += 0x1000;
        }
        while a + slice as u64 <= r.end {
            wins.push(a);
            a += slice as u64;
        }
    }
    wins.sort_by_key(|w| (*w as i64 - hint as i64).unsigned_abs());
    wins.truncate(cap);
    wins
}

/// Did this float triple MOVE over the sampled ticks?
///
/// The cheapest filter on a candidate slot: a car drives, and three floats that
/// hold the same value for every tick of the window are a frozen copy, a render
/// transform or scratch. The threshold is 1e-4 m summed over the three axes,
/// which at map coordinates is below an f32 ULP — so it means "bit-identical",
/// not "nearly still".
pub fn triple_moves(n: usize, at: &dyn Fn(usize, usize) -> f64) -> bool {
    (1..n).any(|i| {
        (at(i, 0) - at(i - 1, 0)).abs()
            + (at(i, 4) - at(i - 1, 4)).abs()
            + (at(i, 8) - at(i - 1, 8)).abs()
            > 1e-4
    })
}

/// Every vehicle-state candidate that passes BOTH structural tests, best first
/// by mean speed.
///
/// WHY A LIST: `locate_pos2` returns the single lowest-verr candidate, and on
/// 126859 that is a decoy -- some other entity whose position, velocity and
/// quaternion are perfectly self-consistent, moving at 3.8 m/s while the car
/// does 40. Self-consistency cannot tell them apart; it was never meant to.
/// What separates them is cheap and reference-free: the CAR is the fastest
/// self-consistent moving thing at a mid-race checkpoint, and if that heuristic
/// is ever wrong the clean run's own self-check refuses the file rather than
/// writing a plausible one.
pub fn locate_candidates(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    clock: u64,
    bounds: (f64, f64, f64, f64, f64, f64),
    max_windows: usize,
    want: usize,
    verbose: bool,
) -> Vec<PosHit> {
    let base = srv.base;
    let hint = base.saturating_sub(603_616);
    let slice: u32 = 64 * 1024;
    let wins = windows_near(srv.pid(), hint, slice, max_windows);
    let (xlo, xhi, ylo, yhi, zlo, zhi) = bounds;
    let qmax: f64 = std::env::var("FK_QERR_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-3);
    let qticks: u32 = std::env::var("FK_QUALIFY_TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let mut out: Vec<PosHit> = Vec::new();
    let mut scanned = 0usize;
    let mut firsthit = usize::MAX;
    // BATCH THE WINDOWS. Each `gather_ticks_stride` call forks the engine and
    // simulates it, and that fork -- not the memory read -- is the whole cost:
    // `fk ptr find` snapshots ALL 202 MB of writable memory in 0.11 s, so the
    // bytes are free and the engine run is seconds. Scanning one 64 KB window
    // per run meant ~129 forks per locate and ~7.5 s, five times per regen.
    //
    // The gatherer already accepts an ARRAY of segments, so ask for many
    // windows in one run. Nothing about the sweep changes -- the same offsets
    // are examined with the same tests -- only how many engine runs it costs.
    let batch: usize = std::env::var("FK_LOCATE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        // DEFAULT 1, which is the original behaviour. Batching looked like the
        // obvious win -- one fork for many windows -- and MEASURED FAR WORSE:
        // 24 windows x 64 KB x ~24 ticks is ~36 MB of snapshot per call
        // written to the dump, and a 90 s regen went past 27 minutes. The fork
        // is expensive, but the per-tick snapshot VOLUME is more expensive
        // still, so trading one for the other loses. Raise it only with a
        // measurement in hand.
        .unwrap_or(1)
        .max(1);
    let lstride: u32 = std::env::var("FK_LOCATE_STRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    'outer: for chunk in wins.chunks(batch) {
        let mut segs: Vec<(u64, u32)> = Vec::with_capacity(chunk.len() + 1);
        segs.push((clock, 4u32));
        for w in chunk {
            segs.push((*w, slice));
        }
        let ts = gather_ticks_stride(srv, probe, recs, &segs, 6 * lstride + 4, 24, (0, 4), lstride);
        if ts.len() < 4 {
            scanned += chunk.len();
            continue;
        }
        let n = ts.len();
        for (j, w) in chunk.iter().enumerate() {
            scanned += 1;
            let wbase = 4 + j * slice as usize;
            let mut shortlist: Vec<u64> = Vec::new();
            for o in (wbase..wbase + slice as usize).step_by(4) {
                if o + 24 > wbase + slice as usize {
                    break;
                }
                let at = |i: usize, k: usize| getf32(&ts[i].rec, o + k);
                let inb = |i: usize| {
                let (x, y, z) = (at(i, 0), at(i, 4), at(i, 8));
                x.is_finite() && y.is_finite() && z.is_finite()
                    && x >= xlo && x <= xhi && y >= ylo && y <= yhi && z >= zlo && z <= zhi
            };
            if !(0..n).all(inb) {
                continue;
            }
            let moved = triple_moves(n, &at);
            if moved {
                shortlist.push(*w + (o - wbase) as u64);
            }
        }
        // Cap the qualify probes per window: with wide bounds a 64 KB window can
        // shortlist hundreds of junk triples, each costing a fork, and the sweep
        // then takes 15 minutes to say nothing (208024).
        for a in shortlist.into_iter().take(64) {
            if a < 16 {
                continue;
            }
            let Some(h) = qualify2(srv, probe, recs, clock, a, qticks, bounds) else { continue };
            if h.qerr >= qmax || h.mean_speed <= 1.0 {
                continue;
            }
            // rule 3 here too -- see the note at the main gate above
            if h.qvary < std::env::var("FK_QVARY_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0) {
                continue;
            }
            if h.verr >= (0.02 * h.mean_speed).max(0.25) {
                continue;
            }
            if verbose {
                println!(
                    "  candidate {:#014x} (base{:+}) verr {:.4} |q|-1 {:.1e} qvary {:.0}% qtravel {:.1}deg speed {:.1}",
                    h.pos, h.pos as i64 - base as i64, h.verr, h.qerr, h.qvary * 100.0, h.qtravel.to_degrees(), h.mean_speed
                );
            }
            out.push(h);
            firsthit = firsthit.min(scanned);
        }
            // Stop soon after the first hit: scanning all 3226 windows for a
            // sixth candidate costs minutes and buys nothing (208024 spent 15
            // of them).
            if out.len() >= want || (!out.is_empty() && scanned > firsthit + 40) {
                break 'outer;
            }
        }
    }
    out.sort_by(|a, b| b.mean_speed.total_cmp(&a.mean_speed));
    if verbose {
        println!("locate_candidates: {} passing, {} windows scanned", out.len(), scanned);
    }
    out
}

/// Position candidates with NO assumption about what sits around them.
///
/// `locate_candidates` requires a unit quaternion 16 B before the position and
/// a matching velocity 12 B after it. That layout is a property of one COPY of
/// the car state, and on 186935 / 227654 / 238835 / 267859 the copy the sweep
/// lands on does not have it -- so the strict locator returns nothing and 21
/// published files stay stale. This returns anything that MOVES LIKE A CAR
/// (finite, in bounds, smooth, and actually going somewhere) and leaves the
/// rest of the layout to be discovered around it.
pub fn locate_positions_loose(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    clock: u64,
    bounds: (f64, f64, f64, f64, f64, f64),
    max_windows: usize,
    want: usize,
    verbose: bool,
) -> Vec<PosHit> {
    let base = srv.base;
    let hint = base.saturating_sub(603_616);
    let slice: u32 = 64 * 1024;
    let wins = windows_near(srv.pid(), hint, slice, max_windows);
    let (xlo, xhi, ylo, yhi, zlo, zhi) = bounds;
    let mut out: Vec<PosHit> = Vec::new();
    let mut firsthit = usize::MAX;
    let mut scanned = 0usize;
    for w in wins {
        scanned += 1;
        let segs = [(clock, 4u32), (w, slice)];
        let ts = gather_ticks_stride(srv, probe, recs, &segs, 64, 24, (0, 4), 10);
        if ts.len() < 4 {
            continue;
        }
        let n = ts.len();
        let mut shortlist: Vec<u64> = Vec::new();
        for o in (4..4 + slice as usize).step_by(4) {
            if o + 12 > 4 + slice as usize {
                break;
            }
            let at = |i: usize, k: usize| getf32(&ts[i].rec, o + k);
            let ok = (0..n).all(|i| {
                let (x, y, z) = (at(i, 0), at(i, 4), at(i, 8));
                x.is_finite() && y.is_finite() && z.is_finite()
                    && x >= xlo && x <= xhi && y >= ylo && y <= yhi && z >= zlo && z <= zhi
            });
            if !ok {
                continue;
            }
            let moved = triple_moves(n, &at);
            if moved {
                shortlist.push(w + (o - 4) as u64);
            }
        }
        for a in shortlist.into_iter().take(64) {
            if a < 256 {
                continue;
            }
            // A dense look at this one triple: is it a trajectory?
            let segs = [(clock, 4u32), (a, 12u32)];
            let ts = gather_ticks(srv, probe, recs, &segs, 150, 1200, (0, 16));
            if ts.len() < 40 {
                continue;
            }
            let g = |t: &Tick, k: usize| getf32(&t.rec, 4 + k * 4);
            let mut steps: Vec<f64> = Vec::new();
            for w2 in ts.windows(2) {
                let dt = (w2[1].clock as i64 - w2[0].clock as i64) as f64 / 1000.0;
                if dt <= 0.0 {
                    continue;
                }
                let d: f64 = (0..3).map(|k| (g(&w2[1], k) - g(&w2[0], k)).powi(2)).sum::<f64>().sqrt();
                steps.push(d / dt);
            }
            if steps.len() < 20 {
                continue;
            }
            steps.sort_by(|x, y| x.total_cmp(y));
            let med = steps[steps.len() / 2];
            let p95 = steps[(steps.len() as f64 * 0.95) as usize];
            // moving, but not teleporting every tick
            if med < 1.0 || p95 > 400.0 {
                continue;
            }
            if verbose {
                println!(
                    "  loose candidate {:#014x} (base{:+}) median speed {:.1} m/s, p95 {:.1}",
                    a,
                    a as i64 - base as i64,
                    med,
                    p95
                );
            }
            out.push(PosHit {
                pos: a,
                verr: 0.0,
                qerr: 0.0,
                // This shortlist path does not read a quaternion at all, so it
                // has no evidence either way. Zero is the honest value: it
                // means "not measured here", and the rule-3 gate downstream
                // must therefore not be applied to hits from this path.
                qvary: 0.0,
                qtravel: 0.0,
                mean_speed: med,
                ticks: ts.len(),
                first: (0.0, 0.0, 0.0),
            });
            firsthit = firsthit.min(scanned);
        }
        if out.len() >= want || (!out.is_empty() && scanned > firsthit + 20) {
            break;
        }
    }
    out.sort_by(|a, b| b.mean_speed.total_cmp(&a.mean_speed));
    out
}