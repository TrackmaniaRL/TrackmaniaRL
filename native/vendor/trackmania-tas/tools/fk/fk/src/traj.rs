//! The 29-column trajectory CSV: reading one, and writing one.
//!
//! This is the format `tmtraj decode --csv` produces from a ghost's own
//! recorded telemetry, so a trajectory `fk` measures out of engine memory and a
//! trajectory decoded from a file are the same table and every analysis tool
//! works on both. That is the whole reason for matching it column for column.
//!
//! **Twelve of the columns are empty** in anything `fk` writes: `gear`,
//! `rpm_raw`, `side_speed`, `is_turbo`, `is_ground_contact`, `turbo_time` and
//! the four wheel-dampen columns. The production state readout is the 40 bytes
//! of transform plus the race clock, and those quantities are not in it.
//!
//! They are not *unavailable*. The engine computes every one of them and they
//! are in its memory; a previous pass measured encodings for several of them
//! (`fk fit`, deleted — see FK.md §"What was measured and then deleted"). What
//! is true is that nothing reads them today. "I have not found where X lives
//! yet" is a task; "X is not available" would be a harness limit reported as a
//! physics limit, and this project has been wrong that way before.

use forkoracle::layout::Row;
use std::collections::HashMap;

pub const COLS: &[&str] = &[
    "time_ms", "x", "y", "z", "speed_kmh", "speed_ms", "vx", "vy", "vz", "yaw", "pitch", "roll",
    "qx", "qy", "qz", "qw", "gear", "rpm_raw", "steer", "gas", "brake", "side_speed", "is_turbo",
    "is_ground_contact", "turbo_time", "fl_dampen", "fr_dampen", "rr_dampen", "rl_dampen",
    // Read out of the engine as of 2026-08-22, and named by matching the
    // game's own recording rather than by inspection: see layout::WET_OFF.
    "wetness",
];

/// One decoded row of somebody else's trajectory — a reference, not a
/// measurement.
#[derive(Clone, Debug, Default)]
pub struct Sample {
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
}

/// A reference trajectory read from a CSV: a ghost's own recorded telemetry,
/// used as the KNOWN ANSWER a located trajectory is checked against.
pub struct Reference {
    pub s: Vec<Sample>,
}

impl Reference {
    pub fn load(path: &str) -> Result<Reference, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        let mut lines = text.lines();
        let hdr: Vec<&str> = lines.next().ok_or("empty csv")?.split(',').collect();
        let idx: HashMap<&str, usize> =
            hdr.iter().enumerate().map(|(i, h)| (h.trim(), i)).collect();
        let f = |row: &[&str], name: &str| -> f64 {
            idx.get(name)
                .and_then(|&i| row.get(i))
                .and_then(|v| v.trim().parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        let mut s = Vec::new();
        for l in lines {
            if l.trim().is_empty() {
                continue;
            }
            let r: Vec<&str> = l.split(',').collect();
            s.push(Sample {
                time_ms: f(&r, "time_ms") as i64,
                x: f(&r, "x"),
                y: f(&r, "y"),
                z: f(&r, "z"),
                vx: f(&r, "vx"),
                vy: f(&r, "vy"),
                vz: f(&r, "vz"),
                qx: f(&r, "qx"),
                qy: f(&r, "qy"),
                qz: f(&r, "qz"),
                qw: f(&r, "qw"),
            });
        }
        if s.is_empty() {
            return Err(format!("{}: no samples", path));
        }
        Ok(Reference { s })
    }

    /// Position at an arbitrary race time, linearly interpolated.
    ///
    /// Interpolation is not a nicety: telemetry is on a 50 ms grid and ticks
    /// are 10 ms, so at 100 km/h the live position is up to 0.7 m from the
    /// nearest recorded sample. Comparing against samples instead of against
    /// the interpolated path finds only STALE copies of the state.
    pub fn pos_at(&self, ms: f64) -> Option<(f64, f64, f64)> {
        let n = self.s.len();
        if n < 2 {
            return None;
        }
        let first = self.s[0].time_ms as f64;
        if ms < first - 1.0 || ms > self.s[n - 1].time_ms as f64 + 1.0 {
            return None;
        }
        let period = (self.s[1].time_ms - self.s[0].time_ms) as f64;
        let k = (((ms - first) / period).floor().max(0.0) as usize).min(n - 2);
        let (a, b) = (&self.s[k], &self.s[k + 1]);
        let span = (b.time_ms - a.time_ms) as f64;
        let u = if span == 0.0 { 0.0 } else { (ms - a.time_ms as f64) / span };
        Some((a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u))
    }

    /// The bounding box of the reference path, grown by `pad` metres.
    ///
    /// Given to the locator so it can reject an address holding a plausible
    /// float triple that is nowhere near this map.
    pub fn bounds(&self, pad: f64) -> (f64, f64, f64, f64, f64, f64) {
        let mut b = (f64::MAX, f64::MIN, f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for s in &self.s {
            b.0 = b.0.min(s.x);
            b.1 = b.1.max(s.x);
            b.2 = b.2.min(s.y);
            b.3 = b.3.max(s.y);
            b.4 = b.4.min(s.z);
            b.5 = b.5.max(s.z);
        }
        (b.0 - pad, b.1 + pad, b.2 - pad, b.3 + pad, b.4 - pad, b.5 + pad)
    }
}

/// How a measured trajectory compares with a reference one.
#[derive(Debug, Clone)]
pub struct Agreement {
    pub compared: usize,
    pub median: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    pub max_at_ms: i64,
    pub within_5cm_pct: f64,
}

impl std::fmt::Display for Agreement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} compared, median {:.4} m, p90 {:.4}, p99 {:.4}, max {:.4} m at {}, \
             {:.2}% of ticks within 5 cm",
            self.compared,
            self.median,
            self.p90,
            self.p99,
            self.max,
            crate::secs(self.max_at_ms),
            self.within_5cm_pct
        )
    }
}

/// Compare a measured trajectory with a reference.
///
/// Reported as QUANTILES, not as an rms. On a record with respawns the rms is
/// meaningless: a respawn teleport is a legitimate 40 m difference between the
/// engine state and the ghost's telemetry for about a second, and 31 of them
/// drag an otherwise centimetre-exact match to 8 m. The median and the
/// within-5-cm fraction say what actually happened; the max says where to look.
pub fn compare(rows: &[Row], reference: &Reference) -> Option<Agreement> {
    let mut ds: Vec<(f64, i64)> = Vec::with_capacity(rows.len());
    for r in rows {
        if let Some(p) = reference.pos_at(r.time_ms as f64) {
            let d = ((p.0 - r.x).powi(2) + (p.1 - r.y).powi(2) + (p.2 - r.z).powi(2)).sqrt();
            ds.push((d, r.time_ms));
        }
    }
    if ds.is_empty() {
        return None;
    }
    let (max, max_at_ms) = ds.iter().cloned().fold((0.0f64, 0i64), |a, b| if b.0 > a.0 { b } else { a });
    let within = ds.iter().filter(|d| d.0 < 0.05).count();
    ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let q = |f: f64| ds[((ds.len() as f64 - 1.0) * f) as usize].0;
    Some(Agreement {
        compared: ds.len(),
        median: q(0.5),
        p90: q(0.9),
        p99: q(0.99),
        max,
        max_at_ms,
        within_5cm_pct: 100.0 * within as f64 / ds.len() as f64,
    })
}

/// Write measured rows in the 29-column format, filling the three input columns
/// from the tape that produced them.
pub fn to_csv(rows: &[Row], tape: &crate::tape::Tape) -> String {
    use tmtraj::json::fmt_g6;
    let tick_of = |ms: i64| -> Option<usize> {
        let t = (ms - tape.start_offset_ms as i64) / 10;
        if t >= 0 && (t as usize) < tape.n() && (ms - tape.start_offset_ms as i64) % 10 == 0 {
            Some(t as usize)
        } else {
            None
        }
    };
    let mut out = String::with_capacity(rows.len() * 200);
    out.push_str(&COLS.join(","));
    out.push_str("\r\n");
    for r in rows {
        let (yaw, pitch, roll) = gbx::record::quat_to_ypr([r.qx, r.qy, r.qz, r.qw]);
        let speed_ms = (r.vx * r.vx + r.vy * r.vy + r.vz * r.vz).sqrt();
        let t = tick_of(r.time_ms);
        let inp = |g: &dyn Fn(usize) -> f64| -> String {
            match t {
                Some(t) => fmt_g6(g(t)),
                None => String::new(),
            }
        };
        let mut first = true;
        for c in COLS {
            if !first {
                out.push(',');
            }
            first = false;
            out.push_str(&match *c {
                "time_ms" => r.time_ms.to_string(),
                "x" => fmt_g6(r.x),
                "y" => fmt_g6(r.y),
                "z" => fmt_g6(r.z),
                "speed_ms" => fmt_g6(speed_ms),
                "speed_kmh" => fmt_g6(speed_ms * 3.6),
                "vx" => fmt_g6(r.vx),
                "vy" => fmt_g6(r.vy),
                "vz" => fmt_g6(r.vz),
                "yaw" => fmt_g6(yaw),
                "pitch" => fmt_g6(pitch),
                "roll" => fmt_g6(roll),
                "qx" => fmt_g6(r.qx),
                "qy" => fmt_g6(r.qy),
                "qz" => fmt_g6(r.qz),
                "qw" => fmt_g6(r.qw),
                "wetness" => fmt_g6(r.wetness),
                "steer" => inp(&|t| ((tape.steer[t] as i8) as f64) / 127.0),
                "gas" => inp(&|t| if tape.accel[t] != 0 { 1.0 } else { 0.0 }),
                "brake" => inp(&|t| if tape.brake[t] != 0 { 1.0 } else { 0.0 }),
                // the twelve the readout does not carry; see the module doc
                _ => String::new(),
            });
        }
        out.push_str("\r\n");
    }
    out
}

/// One named column of a reference CSV, held on its own so a probe can ask for
/// a channel by name without every consumer of `Reference` growing a field.
///
/// A channel is a STEP function of race time, not an interpolated one: the
/// values it exists to match — wetness, gear, a contact bit — are quantised in
/// the record and interpolating them would invent values the game never had.
pub struct Channel {
    pts: Vec<(i64, f64)>,
}

impl Channel {
    /// The value in force at `ms`: the last sample at or before it, within one
    /// sampling period. `None` outside the record.
    pub fn at(&self, ms: i64) -> Option<f64> {
        if self.pts.is_empty() {
            return None;
        }
        let period = if self.pts.len() > 1 { self.pts[1].0 - self.pts[0].0 } else { 50 };
        let i = match self.pts.binary_search_by_key(&ms, |p| p.0) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        if ms - self.pts[i].0 > period {
            return None;
        }
        Some(self.pts[i].1)
    }
}

impl Reference {
    /// A column of the CSV this reference was loaded from, by name.
    /// Re-reads the file: a probe asks for one channel, once.
    pub fn channel_from(path: &str, name: &str) -> Option<Channel> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut lines = text.lines();
        let hdr: Vec<&str> = lines.next()?.split(',').collect();
        let ct = hdr.iter().position(|h| h.trim() == "time_ms")?;
        let cv = hdr.iter().position(|h| h.trim() == name)?;
        let mut pts = Vec::new();
        for l in lines {
            if l.trim().is_empty() {
                continue;
            }
            let r: Vec<&str> = l.split(',').collect();
            let (Some(Ok(t)), Some(v)) = (r.get(ct).map(|v| v.trim().parse::<i64>()), r.get(cv))
            else {
                continue;
            };
            // Booleans print as True/False in this CSV; take them as 1 and 0.
            let val = match v.trim() {
                "True" => 1.0,
                "False" => 0.0,
                s => match s.parse::<f64>() {
                    Ok(x) => x,
                    Err(_) => continue,
                },
            };
            pts.push((t, val));
        }
        (!pts.is_empty()).then_some(Channel { pts })
    }
}
