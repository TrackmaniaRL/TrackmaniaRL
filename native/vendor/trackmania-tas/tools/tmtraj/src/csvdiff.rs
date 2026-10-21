//! Two trajectory CSVs, compared on the instants they share.
//!
//! Both `tmtraj export` (a ghost's own recorded telemetry) and `fk trace` (the
//! engine's per-tick state, read out of the running server) write a CSV whose
//! first column is race milliseconds. Comparing them is the control that says
//! the engine readout found the CAR: a ghost the game recorded itself knows
//! where it was, and a located state slot either reproduces that or does not.
//!
//! Compare on TIME, never row by row: the two are sampled on different grids
//! (50 ms against 10 ms) and a row-index comparison of a trajectory silently
//! measures the offset instead of the disagreement.

use std::collections::BTreeMap;

pub struct Row {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub kmh: f64,
}

fn load(path: &str) -> Result<BTreeMap<i64, Row>, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut lines = txt.lines();
    let hdr: Vec<&str> = lines.next().ok_or("empty file")?.split(',').collect();
    let col = |name: &str| -> Result<usize, String> {
        hdr.iter().position(|h| *h == name).ok_or(format!("{path}: no column {name}"))
    };
    let (ct, cx, cy, cz, cs) =
        (col("time_ms")?, col("x")?, col("y")?, col("z")?, col("speed_kmh")?);
    let mut m = BTreeMap::new();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split(',').collect();
        let g = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NAN);
        let t: i64 = match f.get(ct).and_then(|v| v.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        m.insert(t, Row { x: g(cx), y: g(cy), z: g(cz), kmh: g(cs) });
    }
    Ok(m)
}

/// Print the per-instant disagreement between two trajectory CSVs.
pub fn cmd(argv: &[String]) -> i32 {
    let mut files: Vec<&String> = Vec::new();
    let mut tol: i64 = 5;
    let mut from: i64 = i64::MIN;
    let mut to: i64 = i64::MAX;
    let mut per: i64 = 0;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--tol-ms" => {
                i += 1;
                tol = argv.get(i).and_then(|v| v.parse().ok()).unwrap_or(5);
            }
            "--from" => {
                i += 1;
                from = argv.get(i).and_then(|v| v.parse().ok()).unwrap_or(i64::MIN);
            }
            "--to" => {
                i += 1;
                to = argv.get(i).and_then(|v| v.parse().ok()).unwrap_or(i64::MAX);
            }
            // A single median over a whole run hides WHERE two runs part company,
            // and on a run that starts in free fall it is dominated by the plunge:
            // at 277 m/s one 10 ms sampling offset is 2.8 m of "deviation" that is
            // not a difference in the line at all. This prints the profile.
            "--per" => {
                i += 1;
                per = argv.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            s if s.starts_with("--") => {
                eprintln!("tmtraj csvdiff: unknown flag {s}");
                return 2;
            }
            _ => files.push(&argv[i]),
        }
        i += 1;
    }
    if files.len() != 2 {
        eprintln!("usage: tmtraj csvdiff A.csv B.csv [--tol-ms N] [--from MS] [--to MS] [--per MS]");
        return 2;
    }
    let (a, b) = match (load(files[0]), load(files[1])) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("tmtraj csvdiff: {e}");
            return 1;
        }
    };
    let mut d: Vec<f64> = Vec::new();
    let mut ds: Vec<f64> = Vec::new();
    let mut prof: Vec<(i64, f64)> = Vec::new();
    let mut span = (i64::MAX, i64::MIN);
    for (t, ra) in &a {
        if *t < from || *t > to {
            continue;
        }
        let mut best: Option<(i64, &Row)> = None;
        for (u, rb) in b.range(t - tol..=t + tol) {
            let dt = (u - t).abs();
            if best.map_or(true, |(bd, _)| dt < bd) {
                best = Some((dt, rb));
            }
        }
        let Some((_, rb)) = best else { continue };
        span = (span.0.min(*t), span.1.max(*t));
        let dist = ((ra.x - rb.x).powi(2) + (ra.y - rb.y).powi(2) + (ra.z - rb.z).powi(2)).sqrt();
        d.push(dist);
        prof.push((*t, dist));
        if ra.kmh.is_finite() && rb.kmh.is_finite() {
            ds.push((ra.kmh - rb.kmh).abs());
        }
    }
    if d.is_empty() {
        println!("no shared instants within {tol} ms");
        return 1;
    }
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ds.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p = |v: &Vec<f64>, q: usize| v[(v.len() - 1) * q / 100];
    println!(
        "{} shared instants, race {:.3} .. {:.3}",
        d.len(),
        span.0 as f64 / 1000.0,
        span.1 as f64 / 1000.0
    );
    println!(
        "position  median {:.3} m   p95 {:.3} m   max {:.3} m",
        p(&d, 50),
        p(&d, 95),
        d[d.len() - 1]
    );
    if !ds.is_empty() {
        println!(
            "speed     median {:.2} km/h  p95 {:.2} km/h  max {:.2} km/h",
            p(&ds, 50),
            p(&ds, 95),
            ds[ds.len() - 1]
        );
    }
    if per > 0 {
        println!("  window        n   median_m      max_m");
        let mut k = 0usize;
        while k < prof.len() {
            let w0 = prof[k].0 - prof[k].0.rem_euclid(per);
            let mut v: Vec<f64> = Vec::new();
            while k < prof.len() && prof[k].0 < w0 + per {
                v.push(prof[k].1);
                k += 1;
            }
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            println!(
                "  {:>7.3}  {:>4}  {:>9.3}  {:>9.3}",
                w0 as f64 / 1000.0,
                v.len(),
                v[(v.len() - 1) / 2],
                v[v.len() - 1]
            );
        }
    }
    0
}
