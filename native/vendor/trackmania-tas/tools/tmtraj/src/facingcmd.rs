//! `tmtraj facing` -- does the stored orientation point the car where it is
//! actually going?
//!
//! WHY (arm `r165`, 2026-08-21)
//! ---------------------------
//! `fk regen`'s layout probe reports the quaternion's memory KIND, and on
//! 165922 that kind flipped between runs of the same command: three of nine
//! files came out `(x,y,z,w)` where six were `(w,x,y,z)`. Positions are
//! identical either way, so the oracle, the tape md5, the contamination pair
//! and every C-check pass -- and the car faces the wrong way for the whole
//! render. Nothing in the gate could see it.
//!
//! The fleet has since settled the convention: the live car's quaternion is
//! **(w,x,y,z) in memory, and forward is body +z**. That makes the defect
//! directly testable on the WRITTEN FILE, with no engine and no reference:
//! rotate the body's forward axis by each sample's stored orientation and
//! compare it with the direction the car is travelling. A car drives roughly
//! where it points; a car whose quaternion components have been permuted does
//! not, and the disagreement is tens of degrees.
//!
//! What it does NOT claim. A real car drifts, flies and lands sideways, so a
//! per-sample angle of 20-30 deg is normal and the MEDIAN over a lap is the
//! statistic. The bar is set by a human recording of the same map, which is
//! the only thing that says what "normal" is here -- pass `--ref` and the
//! command prints both, which is the positive control the verdict needs.
//!
//! The record stores the quaternion as (x,y,z,w).

use gbx::record;
/// (ms, vx,vy,vz, qx,qy,qz,qw) straight off the decoded samples.
///
/// This used to decode the ghost, write `entrec::csv_string` to a temp file
/// under /tmp, read it back and parse it by HARDCODED COLUMN INDEX (0,6,7,8,
/// 12,13,14,15) — three times in this file. Reordering `CSV_COLUMNS` would
/// have silently changed what `facing` measured, with no compile error.
fn rows_of(d: &record::Decoded) -> Vec<[f64; 8]> {
    d.samples
        .iter()
        .map(|s| [s.time_ms as f64, s.vx as f64, s.vy as f64, s.vz as f64, s.qx as f64, s.qy as f64, s.qz as f64, s.qw as f64])
        .collect()
}


/// Rotate v by the unit quaternion (x,y,z,w).
fn rot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Median angle in degrees between the rotated body axis and the velocity,
/// over samples above `minspeed`, for each of the four axis choices and both
/// component orders. Returns (median for +z under the stated convention, the
/// full table).
fn measure(rows: &[[f64; 8]], minspeed: f64) -> (f64, Vec<(String, f64)>) {
    let orders: [(&str, [usize; 4]); 2] = [
        // as stored: the record's (x,y,z,w)
        ("stored (x,y,z,w)", [0, 1, 2, 3]),
        // what a wrong-KIND file looks like: memory (x,y,z,w) written as if it
        // were (w,x,y,z), i.e. the components rotated by one
        ("rotated by one", [1, 2, 3, 0]),
    ];
    let axes: [(&str, [f64; 3]); 3] = [
        ("+z", [0.0, 0.0, 1.0]),
        ("+x", [1.0, 0.0, 0.0]),
        ("+y", [0.0, 1.0, 0.0]),
    ];
    let mut table = Vec::new();
    let mut primary = f64::NAN;
    for (oname, ord) in orders {
        for (aname, ax) in axes {
            let mut angs = Vec::new();
            for r in rows {
                let v = [r[1], r[2], r[3]];
                let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if !(n > minspeed) {
                    continue;
                }
                let q = [r[4 + ord[0]], r[4 + ord[1]], r[4 + ord[2]], r[4 + ord[3]]];
                let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                if !(qn > 0.5) {
                    continue;
                }
                let q = [q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn];
                let f = rot(q, ax);
                let dot = (f[0] * v[0] + f[1] * v[1] + f[2] * v[2]) / n;
                angs.push(dot.clamp(-1.0, 1.0).acos().to_degrees());
            }
            let m = median(angs);
            if oname.starts_with("stored") && aname == "+z" {
                primary = m;
            }
            table.push((format!("{} body {}", oname, aname), m));
        }
    }
    (primary, table)
}

pub fn cmd(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "--route") {
        return cmd_route(args);
    }
    let flag = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let files: Vec<String> =
        args.iter().filter(|a| a.ends_with(".Ghost.Gbx")).cloned().collect();
    if files.is_empty() {
        eprintln!(
            "usage: tmtraj facing GHOST... [--ref HUMAN.Ghost.Gbx] [--minspeed M] [--bar DEG] [--table]"
        );
        return 2;
    }
    let minspeed: f64 = flag("--minspeed").and_then(|v| v.parse().ok()).unwrap_or(5.0);
    let table = args.iter().any(|a| a == "--table");
    let refp = flag("--ref");

    // The bar is a human recording of the same map, not a constant: a flight
    // map's car spends real time pointing where it is not going.
    let (refmed, refname) = match &refp {
        Some(p) => {
            let Ok(d) = record::decode_ghost(p) else {
                eprintln!("cannot decode reference {}", p);
                return 3;
            };
            let rows = rows_of(&d);
            let (m, t) = measure(&rows, minspeed);
            println!(
                "reference {}: median {:.2} deg between body +z and the velocity ({} samples over {} m/s)",
                p,
                m,
                rows.len(),
                minspeed
            );
            if table {
                for (n, v) in &t {
                    println!("    {:<26} {:.2} deg", n, v);
                }
            }
            (m, p.clone())
        }
        None => (f64::NAN, String::new()),
    };
    let bar: f64 = flag("--bar")
        .and_then(|v| v.parse().ok())
        .unwrap_or(if refmed.is_finite() { refmed * 2.0 + 10.0 } else { 45.0 });

    let mut worst = 0i32;
    for f in &files {
        let Ok(d) = record::decode_ghost(f) else {
            println!("{:<40} DECODE-FAIL", f);
            worst = worst.max(3);
            continue;
        };
        let rows = rows_of(&d);
        let (m, t) = measure(&rows, minspeed);
        let verdict = if !m.is_finite() {
            worst = worst.max(3);
            "NO-DATA"
        } else if m <= bar {
            "FACING-OK"
        } else {
            worst = worst.max(2);
            "FACING-WRONG"
        };
        println!(
            "{:<40} {:>7.2} deg  bar {:.2}  {}",
            f.rsplit('/').next().unwrap_or(f),
            m,
            bar,
            verdict
        );
        if table {
            for (n, v) in &t {
                println!("    {:<26} {:.2} deg", n, v);
            }
        }
    }
    if !refname.is_empty() {
        println!(
            "bar = 2x the reference's own median + 10 deg. A permuted quaternion reads tens of degrees out; \
             the reference is what says how much of that is just the car drifting."
        );
    }
    worst
}

/// `tmtraj facing --route ROUTE.CSV GHOST` -- the EXACT form of the same
/// question, and the one that needs no bar.
///
/// `fk btraj2` re-simulates the ghost's own tape in the live engine and writes
/// the car's state per tick, orientation included. So the file's stored
/// quaternion can be compared with the engine's own, instant by instant: a
/// correct file agrees to the encoder's rounding, and a file whose components
/// were permuted by a wrong layout KIND is tens of degrees out. There is no
/// threshold to argue about — the reference is the engine.
///
/// The rotation angle between two unit quaternions is `2 acos |a.b|` (the
/// absolute value because q and -q are the same rotation).
pub fn cmd_route(args: &[String]) -> i32 {
    let flag = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let route = flag("--route").expect("--route ROUTE.csv");
    let files: Vec<String> =
        args.iter().filter(|a| a.ends_with(".Ghost.Gbx")).cloned().collect();
    let shift: i64 = flag("--shift-ms").and_then(|v| v.parse().ok()).unwrap_or(0);
    // route: ms -> (q, pos)
    let txt = std::fs::read_to_string(&route).unwrap_or_default();
    let mut rq: std::collections::HashMap<i64, ([f64; 4], [f64; 3])> = Default::default();
    for (i, l) in txt.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let f: Vec<&str> = l.split(',').collect();
        if f.len() < 16 {
            continue;
        }
        let g = |k: usize| f[k].parse::<f64>().unwrap_or(f64::NAN);
        let (t, q, p) = (
            f[0].parse::<i64>().unwrap_or(i64::MIN),
            [g(12), g(13), g(14), g(15)],
            [g(1), g(2), g(3)],
        );
        if t != i64::MIN && q.iter().all(|v| v.is_finite()) {
            rq.insert(t, (q, p));
        }
    }
    let mut worst = 0i32;
    for f in &files {
        let Ok(d) = record::decode_ghost(f) else {
            println!("{:<40} DECODE-FAIL", f);
            worst = worst.max(3);
            continue;
        };
        let mut angs: Vec<f64> = Vec::new();
        let mut angs_perm: Vec<f64> = Vec::new();
        let mut dpos: Vec<f64> = Vec::new();
        for s in &d.samples {
            let pos = [s.x as f64, s.y as f64, s.z as f64];
            let t = s.time_ms as i64 + shift;
            let Some((qr, pr)) = rq.get(&t) else { continue };
            let qs = [s.qx as f64, s.qy as f64, s.qz as f64, s.qw as f64];
            let perm = [qs[1], qs[2], qs[3], qs[0]];
            let ang = |a: [f64; 4], b: &[f64; 4]| -> f64 {
                let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
                let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3]).sqrt();
                if !(na > 0.5) || !(nb > 0.5) {
                    return f64::NAN;
                }
                let dot: f64 =
                    (0..4).map(|k| a[k] / na * b[k] / nb).sum::<f64>().abs().clamp(0.0, 1.0);
                2.0 * dot.acos().to_degrees()
            };
            let a = ang(qs, qr);
            if a.is_finite() {
                angs.push(a);
            }
            let ap = ang(perm, qr);
            if ap.is_finite() {
                angs_perm.push(ap);
            }
            let mut s = 0.0;
            for k in 0..3 {
                let q = pos[k] - pr[k];
                s += q * q;
            }
            dpos.push(s.sqrt());
        }
        if angs.len() < 5 {
            println!("{:<40} n/a  only {} shared instants", f, angs.len());
            worst = worst.max(3);
            continue;
        }
        let m = median(angs.clone());
        let mp = median(angs_perm.clone());
        let dp = median(dpos.clone());
        let verdict = if m < 5.0 {
            "ORIENTATION-OK"
        } else if mp < 5.0 {
            worst = worst.max(2);
            "ORIENTATION-PERMUTED"
        } else {
            worst = worst.max(2);
            "ORIENTATION-FOREIGN"
        };
        println!(
            "{:<40} {} shared, median {:>7.3} deg vs the engine (permuted reading {:>7.3}), position {:.4} m  {}",
            f.rsplit('/').next().unwrap_or(f),
            angs.len(),
            m,
            mp,
            dp,
            verdict
        );
    }
    worst
}
