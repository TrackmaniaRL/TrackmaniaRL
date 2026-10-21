//! `tmtraj airborne` — find the ballistic episodes in a trajectory CSV, with no
//! reference line and no contact flag.
//!
//! ## Why this exists, and what it replaces
//!
//! On 285885 the natural way to ask "did this candidate leave the surface" is a
//! gate placed some metres above the reference line. It does not work, and the
//! way it fails is instructive enough to be worth a tool:
//!
//! * 12 rungs were placed **8 m above the fast route's own trajectory** across
//!   the finish face. The reference tape fires none of them (a car on the
//!   surface has its tested point below the trigger's floor) and rungs placed
//!   0.4 m above the same line fire for it at exactly the right times. That
//!   looks like a controlled detector.
//! * 26 candidates out of 920 fired the 8 m rungs. **Every one of them traced
//!   as firmly on the ground**: they simply drove a line further up the ramp,
//!   where the surface itself is 8 m higher than the reference line was.
//!
//! A gate above a sloping surface is a HEIGHT detector wherever the candidate's
//! line differs from the line the gate was placed on — the same family as the
//! "tilt detector at `plane(x,z) − Δ`" that cost an earlier arm six rounds. The
//! control that passes (the reference fires nothing) does not exclude it,
//! because the reference is the one line the rung was fitted to.
//!
//! ## What this measures instead
//!
//! Free fall, from the trajectory alone: a window in which the second
//! difference of `y` matches the map's own gravity. Nothing about a reference
//! line, nothing about the derived contact bit (which on a synthesised tape is
//! the carrier's anyway).
//!
//! ```text
//! tmtraj airborne TRACE.csv [--g -24.308] [--tol 3.0] [--min 0.10]
//!                           [--in x0,z0:x1,z1] [--quiet]
//! ```
//!
//! **`--g` is per map and must be measured on the map in question** — −24.308
//! on 285885, −25.20 on 145875, −22.3 on 276874. A three-point difference
//! quantises badly at 1 mm / 10 ms, so the acceleration is taken from a
//! least-squares quadratic over a sliding window.

pub struct Episode {
    pub t0: f64,
    pub t1: f64,
    pub x0: f64,
    pub y0: f64,
    pub z0: f64,
    pub x1: f64,
    pub y1: f64,
    pub z1: f64,
    pub ymax: f64,
    pub kmh: f64,
}

/// Rows are `(t_s, x, y, z, kmh)`.
pub fn episodes(rows: &[(f64, f64, f64, f64, f64)], g: f64, tol: f64, min_s: f64) -> Vec<Episode> {
    // Per-sample acceleration from a 5-point least-squares quadratic. A
    // 3-point second difference on 1 mm position data at 10 ms quantises to a
    // ~10 m/s² comb, which would make every classification noise.
    const W: usize = 2;
    let n = rows.len();
    let mut acc = vec![f64::NAN; n];
    for i in W..n.saturating_sub(W) {
        let t0 = rows[i].0;
        let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut b0, mut b1, mut b2) = (0.0, 0.0, 0.0);
        for k in (i - W)..=(i + W) {
            let dt = rows[k].0 - t0;
            let y = rows[k].2;
            let (p1, p2, p3, p4) = (dt, dt * dt, dt * dt * dt, dt * dt * dt * dt);
            s0 += 1.0;
            s1 += p1;
            s2 += p2;
            s3 += p3;
            s4 += p4;
            b0 += y;
            b1 += y * p1;
            b2 += y * p2;
        }
        // Solve the 3x3 normal equations for the quadratic coefficient c in
        // y = a + b t + c t^2; the acceleration is 2c.
        let m = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]];
        let rhs = [b0, b1, b2];
        if let Some(sol) = solve3(m, rhs) {
            acc[i] = 2.0 * sol[2];
        }
    }
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        if acc[i].is_nan() || (acc[i] - g).abs() > tol {
            i += 1;
            continue;
        }
        let s = i;
        while i < n && !acc[i].is_nan() && (acc[i] - g).abs() <= tol {
            i += 1;
        }
        let e = i - 1;
        if rows[e].0 - rows[s].0 + 1e-9 < min_s {
            continue;
        }
        let ymax = rows[s..=e].iter().fold(f64::NEG_INFINITY, |m, r| m.max(r.2));
        let kmh = rows[s..=e].iter().fold(0.0f64, |m, r| m.max(r.4));
        out.push(Episode {
            t0: rows[s].0,
            t1: rows[e].0,
            x0: rows[s].1,
            y0: rows[s].2,
            z0: rows[s].3,
            x1: rows[e].1,
            y1: rows[e].2,
            z1: rows[e].3,
            ymax,
            kmh,
        });
    }
    out
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for c in 0..3 {
        let mut p = c;
        for r in c + 1..3 {
            if a[r][c].abs() > a[p][c].abs() {
                p = r;
            }
        }
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..3 {
            let f = a[r][c] / a[c][c];
            for k in c..3 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; 3];
    for r in (0..3).rev() {
        let mut s = b[r];
        for k in r + 1..3 {
            s -= a[r][k] * x[k];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}

fn read_csv(path: &str) -> Result<Vec<(f64, f64, f64, f64, f64)>, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut out = Vec::new();
    for (i, l) in txt.lines().enumerate() {
        if i == 0 || l.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = l.trim().split(',').collect();
        if f.len() < 5 {
            continue;
        }
        let g = |k: usize| f[k].trim().parse::<f64>().unwrap_or(f64::NAN);
        out.push((g(0) / 1000.0, g(1), g(2), g(3), g(4)));
    }
    if out.len() < 8 {
        return Err(format!("{}: only {} rows", path, out.len()));
    }
    Ok(out)
}

pub fn cmd(argv: &[String]) -> i32 {
    let f = |n: &str| -> Option<String> {
        argv.iter().position(|a| a == n).and_then(|i| argv.get(i + 1)).cloned()
    };
    let g: f64 = f("--g").and_then(|s| s.parse().ok()).unwrap_or(-24.308);
    let tol: f64 = f("--tol").and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let min_s: f64 = f("--min").and_then(|s| s.parse().ok()).unwrap_or(0.10);
    let quiet = argv.iter().any(|a| a == "--quiet");
    let bx: Option<Vec<f64>> = f("--in").map(|s| {
        s.split([',', ':'])
            .map(|v| v.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect::<Vec<_>>()
    });
    let files: Vec<&String> = argv.iter().filter(|a| !a.starts_with("--")).collect();
    let files: Vec<&String> = files
        .into_iter()
        .filter(|a| a.ends_with(".csv"))
        .collect();
    if files.is_empty() {
        eprintln!(
            "tmtraj airborne TRACE.csv... [--g -24.308] [--tol 3.0] [--min 0.10] \
             [--in x0,z0:x1,z1] [--quiet]\n\n  --g is PER MAP and must be measured on the map in \
             question. Free fall is -24.308 on 285885, -25.20 on 145875, -22.3 on 276874."
        );
        return 2;
    }
    println!("g {:.3} m/s^2, tol +/-{:.1}, minimum episode {:.3} s", g, tol, min_s);
    let mut any = 0usize;
    for p in files {
        let rows = match read_csv(p) {
            Ok(r) => r,
            Err(e) => {
                println!("{}\tREAD FAILED\t{}", p, e);
                continue;
            }
        };
        let eps = episodes(&rows, g, tol, min_s);
        let eps: Vec<&Episode> = eps
            .iter()
            .filter(|e| match &bx {
                None => true,
                Some(b) if b.len() == 4 => {
                    let (x0, z0, x1, z1) =
                        (b[0].min(b[2]), b[1].min(b[3]), b[0].max(b[2]), b[1].max(b[3]));
                    e.x0 >= x0 && e.x0 <= x1 && e.z0 >= z0 && e.z0 <= z1
                }
                _ => true,
            })
            .collect();
        any += eps.len();
        if eps.is_empty() {
            if !quiet {
                println!("{}\t0 episodes", p);
            }
            continue;
        }
        println!("{}\t{} episodes", p, eps.len());
        for e in eps {
            println!(
                "  {:.3} .. {:.3}  ({:.2} s)  ({:.1},{:.1},{:.1}) -> ({:.1},{:.1},{:.1})  \
                 apex {:.1}  up to {:.0} km/h",
                e.t0, e.t1, e.t1 - e.t0, e.x0, e.y0, e.z0, e.x1, e.y1, e.z1, e.ymax, e.kmh
            );
        }
    }
    println!("{} episodes in total", any);
    0
}
