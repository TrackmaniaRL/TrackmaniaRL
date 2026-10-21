//! Where the car's state lives in one server process, and how to find it
//! without any recorded telemetry.
//!
//! Extracted from `fk` into the shared driver crate so the SEARCH can locate
//! the vehicle struct in its own fork servers -- the search cannot depend on
//! `fk` (that dependency runs the other way), and a second copy of this code
//! would be a second thing to get wrong.
//!
//! Everything here is parameterised by the input tape as `&[Rec]` rather than
//! by a ghost `Factory`, which is what keeps this crate free of tmsearch.

use crate::forksrv::{ForkServer, Rec};

/// Where the car's state lives in one particular server process.
#[derive(Clone, Debug)]
pub struct Layout {
    /// f32 x,y,z. The anchor everything else is expressed against.
    pub pos: u64,
    /// u32 race clock, ticking by exactly 10 ms.
    pub clock: u64,
    /// `clock_value - race_ms`, constant for the whole run.
    pub clock_bias: i64,
    /// Deviation of the located position from the ghost's own path, metres.
    pub rms: f64,
    pub max_dev: f64,
}

/// Offsets within the gathered record, once the two segments are concatenated.
pub const R_CLOCK: usize = 0;
pub const R_QUAT: usize = 4; // qw qx qy qz
pub const R_POS: usize = 20; // x y z
pub const R_VEL: usize = 32; // vx vy vz
pub const R_WET: usize = 44; // f32 tyre wetness, 0..1
pub const REC_LEN: usize = 48;

/// Where the wetness f32 sits relative to the position anchor. MEASURED, not
/// assumed: `fk probe` searched a 2 KB window around the car against the
/// game's own recording and found car+180 reproducing it on 95.4-96.0 % of
/// steady ticks with a correlation of 0.9997, against a next-best offset at
/// 31-47 %. Confirmed at two probe ticks and on two different ghosts, and the
/// negative control -- the same tape against ANOTHER run's answer key -- finds
/// nothing (44 %).
///
/// Wetness matters because it is a POSITIONAL INTEGRAL: a function of where
/// the car has been rather than of its state, so it separates two runs that
/// speed cannot tell apart (measured: >10 points apart at 49 % of the shared
/// instants of two human replays of one map).
pub const WET_OFF: i64 = 180;

/// The segments the production sampler gathers: the clock, the car block, and
/// the wetness word (see `WET_OFF`).
pub fn segments(l: &Layout) -> Vec<(u64, u32)> {
    vec![(l.clock, 4), (l.pos - 16, 40), (l.pos.wrapping_add(WET_OFF as u64), 4)]
}

fn getf32(b: &[u8], o: usize) -> f64 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64
}

/// One extracted tick.
#[derive(Clone, Debug)]
pub struct Row {
    pub time_ms: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub qx: f64,
    pub qy: f64,
    pub qz: f64,
    pub qw: f64,
    /// Tyre wetness, 0..1.
    pub wetness: f64,
}

/// Find the engine's race clock near an already-qualified position address.
///
/// One extra fork: stream a wide window keyed on the position, then look for
/// the 4-byte slot that advances by exactly 10 on every one of those ticks.
/// Demanding *every* step rules out anything that merely correlates.
pub fn find_clock(
    srv: &mut ForkServer,
    probe: usize,
    recs: &[Rec],
    start_offset_ms: i32,
    pos: u64,
    back: u64,
    ahead: u64,
    stride: u64,
) -> Result<(u64, i64), String> {
    let lo = pos - back;
    let len = (back + ahead) as u32;

    // 400 ticks is plenty to pin a strict +10-every-tick slot, and keeps the
    // discovery fork's pipe traffic to a few MB.
    let (_j, blob) = srv.run_sampled(probe, recs, lo, len, stride, 400, (back as u32, 12));
    let recsz = 8 + len as usize;
    let m = blob.len() / recsz;
    if m < 50 {
        return Err(format!("clock discovery: only {} samples", m));
    }
    let g = |i: usize, o: usize| -> u32 {
        u32::from_le_bytes(blob[i * recsz + 8 + o..i * recsz + 12 + o].try_into().unwrap())
    };
    let mut found: Vec<(u64, i64)> = Vec::new();
    for o in (0..len as usize - 4).step_by(4) {
        if (0..m - 1).all(|i| g(i + 1, o).wrapping_sub(g(i, o)) == 10) {
            let race0 = sample_ms(probe, 0, start_offset_ms);
            found.push((lo + o as u64, g(0, o) as i64 - race0));
        }
    }
    if std::env::var("FKDBG").is_ok() {
        eprintln!(
            "DBG find_clock: {} samples (asked 400), {} slots step by exactly +10: {:?}",
            m,
            found.len(),
            found
                .iter()
                .map(|(a, b)| format!("{:#x} bias{:+}", a, b))
                .collect::<Vec<_>>()
        );
    }
    match found.first() {
        Some(&(a, b)) => Ok((a, b)),
        None => Err("no u32 advances by exactly 10 every tick near the vehicle state".into()),
    }
}
/// Decode a gathered sample blob into one row per tick.
///
/// The record is keyed on its whole content, so the engine may emit several
/// samples inside one tick; the last one carries that tick's finished state.
/// The clock makes the stream self-timing: a missing or duplicated tick shows
/// up as a gap rather than silently shifting everything after it.
pub fn decode_rows(blob: &[u8], l: &Layout, label_shift: i64) -> (Vec<Row>, Vec<String>) {
    let recsz = 8 + REC_LEN;
    let m = blob.len() / recsz;
    let mut rows: Vec<Row> = Vec::new();
    let mut warn = Vec::new();
    for i in 0..m {
        let b = &blob[i * recsz + 8..i * recsz + 8 + REC_LEN];
        let clk = u32::from_le_bytes(b[R_CLOCK..R_CLOCK + 4].try_into().unwrap()) as i64;
        let t = clk - l.clock_bias + label_shift;
        let row = Row {
            time_ms: t,
            x: getf32(b, R_POS),
            y: getf32(b, R_POS + 4),
            z: getf32(b, R_POS + 8),
            vx: getf32(b, R_VEL),
            vy: getf32(b, R_VEL + 4),
            vz: getf32(b, R_VEL + 8),
            qw: getf32(b, R_QUAT),
            qx: getf32(b, R_QUAT + 4),
            qy: getf32(b, R_QUAT + 8),
            qz: getf32(b, R_QUAT + 12),
            wetness: getf32(b, R_WET),
        };
        match rows.last_mut() {
            Some(last) if last.time_ms == t => *last = row,
            _ => rows.push(row),
        }
    }
    for w in rows.windows(2) {
        if w[1].time_ms - w[0].time_ms != 10 {
            warn.push(format!(
                "clock gap: {} -> {} ms",
                w[0].time_ms, w[1].time_ms
            ));
        }
    }
    (rows, warn)
}

/// Race time of sample `i` of a stream started at boundary tick `probe`.
///
/// Sample 0 is the state at the end of tick `probe - 1`: the resume rewrites
/// tick `probe` onwards, so the first state the child reports is the one the
/// prefix left behind.
pub fn sample_ms(probe: usize, i: usize, start_offset_ms: i32) -> i64 {
    (probe as i64 - 1 + i as i64) * 10 + start_offset_ms as i64
}

/// One tick of input for every tape tick from `from` to the end.
pub fn tail_recs(steer: &[u8], accel: &[u8], brake: &[u8], from: usize) -> Vec<Rec> {
    (from..steer.len())
        .map(|t| crate::forksrv::rec_of(steer[t], accel[t], brake[t]))
        .collect()
}

// --------------------------------------------------------------- self-checks
//
// Two questions no measurement of a simulated trajectory should be trusted
// without, both answered from data the run already produced:
//
//   1. Is the simulator running the tape I asked about?
//   2. Is the thing I read out of it the car?
//
// Question 1 sounds impossible to get wrong and was, in production, wrong 17%
// of the time: two `fk btraj` processes sharing a work directory swap replays,
// so one of them measures the OTHER tape's prefix with its own tail patched in.
// The result is a genuine, self-consistent trajectory of a car that drove
// somewhere else -- no internal consistency test can see it, because nothing
// about it is inconsistent. Only comparing against the tape itself can.

/// THE IDENTITY CONTROL: the decoded input array in the server's memory must
/// be, tick for tick, the tape we mean to measure.
///
/// `base` is the array the shim located and reported at handshake; the layout
/// is one 32-byte record per tick, `+0` steer, `+4` gas, `+8` brake as f32.
/// Reading it back through /proc/<pid>/mem costs one 70 KB read and settles the
/// question completely.
pub fn verify_tape(
    pid: i32,
    base: u64,
    steer: &[u8],
    accel: &[u8],
    brake: &[u8],
) -> Result<(), String> {
    let n = steer.len();
    let buf = crate::procmem::read_at(pid, base, n * crate::forksrv::STRIDE)
        .ok_or_else(|| format!("tape check: cannot read {} bytes at {:#x} of pid {}", n * 32, base, pid))?;
    let mut bad = 0usize;
    let mut first = String::new();
    for t in 0..n {
        let o = t * crate::forksrv::STRIDE;
        let g = |k: usize| f32::from_le_bytes(buf[o + k..o + k + 4].try_into().unwrap());
        let want = crate::forksrv::rec_of(steer[t], accel[t], brake[t]);
        if g(0) != want.steer || g(4) != want.gas || g(8) != want.brake {
            if bad == 0 {
                first = format!(
                    "tick {}: server has ({}, {}, {}), tape says ({}, {}, {})",
                    t, g(0), g(4), g(8), want.steer, want.gas, want.brake
                );
            }
            bad += 1;
        }
    }
    if bad > 0 {
        return Err(format!(
            "TAPE MISMATCH: {} of {} ticks differ -- the simulator is not running the tape \
             that was asked for (first difference: {}). This is what a shared work directory \
             does; give every run its own --work.",
            bad, n, first
        ));
    }
    Ok(())
}

/// What a whole-run self-check found. All of it is measured over every row of
/// the extracted trajectory, not the 150-sample window the locator used.
#[derive(Debug, Clone)]
pub struct RowCheck {
    pub rows: usize,
    pub quat_err: f64,
    pub vel_err: f64,
    pub gaps: usize,
    pub mean_speed: f64,
}

impl std::fmt::Display for RowCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} rows, |q|-1 p99.5 {:.2e}, |d(pos)/dt - v| median {:.3} m/s, {} clock gaps, \
             mean speed {:.1} m/s",
            self.rows, self.quat_err, self.vel_err, self.gaps, self.mean_speed
        )
    }
}

/// QUESTION 2, over the whole run instead of a 150-tick window.
///
/// Three independent things must hold if the rows are the vehicle state:
/// the quaternion is a UNIT quaternion (a structural property of the struct,
/// nothing to do with the velocity test that selected the slot), the position
/// derivative matches the velocity triple, and the clock advances by exactly
/// one tick per row. Two of the three are independent of the signature the
/// locator searched on, which is the point: agreement between independent
/// tests is what makes a reference-free measurement trustworthy.
pub fn check_rows(rows: &[Row]) -> Result<RowCheck, String> {
    if rows.len() < 50 {
        return Err(format!("only {} rows extracted", rows.len()));
    }
    let mut qs: Vec<f64> = Vec::with_capacity(rows.len());
    for r in rows {
        let n = (r.qw * r.qw + r.qx * r.qx + r.qy * r.qy + r.qz * r.qz).sqrt();
        qs.push((n - 1.0).abs());
    }
    qs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // The 99.5th percentile, not the max: one row of a respawn transition is
    // not evidence about the layout, and a record with 31 respawns has 31 of
    // them.
    let qmax: f64 = qs[((qs.len() as f64 - 1.0) * 0.995) as usize];
    let mut verrs: Vec<f64> = Vec::with_capacity(rows.len());
    let mut speed = 0.0;
    let mut n = 0usize;
    let mut gaps = 0usize;
    for w in rows.windows(2) {
        let dt = (w[1].time_ms - w[0].time_ms) as f64 / 1000.0;
        if (w[1].time_ms - w[0].time_ms) != 10 {
            gaps += 1;
            continue;
        }
        let (dx, dy, dz) = (w[1].x - w[0].x, w[1].y - w[0].y, w[1].z - w[0].z);
        verrs.push(
            ((dx / dt - w[0].vx).powi(2) + (dy / dt - w[0].vy).powi(2) + (dz / dt - w[0].vz).powi(2))
                .sqrt(),
        );
        speed += (dx * dx + dy * dy + dz * dz).sqrt() / dt;
        n += 1;
    }
    // MEDIAN, not mean. A respawn moves the car tens of metres in one tick and
    // the mean of |d(pos)/dt - v| over a 31-respawn record is 16 m/s while the
    // typical row is 0.1 -- the mean condemns a perfect measurement.
    verrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let vmed = if verrs.is_empty() {
        f64::MAX
    } else {
        verrs[verrs.len() / 2]
    };
    let c = RowCheck {
        rows: rows.len(),
        quat_err: qmax,
        vel_err: vmed,
        gaps,
        mean_speed: if n > 0 { speed / n as f64 } else { 0.0 },
    };
    // Thresholds, all with two orders of magnitude of headroom against
    // measured good runs (|q|-1 ~ 1e-7, vel_err ~ 0.1 m/s, 0 gaps). The
    // velocity bound is RELATIVE to the car's own speed: a fixed 2.0 m/s was
    // calibrated on a 90 m/s car and means nothing on a 30 or a 300 m/s one.
    if c.quat_err > 1e-3 {
        return Err(format!("not a unit quaternion (p99.5 |q|-1 = {:.3e}): {}", c.quat_err, c));
    }
    if c.vel_err > (0.02 * c.mean_speed).max(0.5) {
        return Err(format!("position derivative disagrees with the velocity triple: {}", c));
    }
    if c.gaps * 200 > c.rows {
        return Err(format!("clock is not advancing one tick per row: {}", c));
    }
    if c.mean_speed < 1.0 {
        return Err(format!("the car never moves: {}", c));
    }
    Ok(c)
}
