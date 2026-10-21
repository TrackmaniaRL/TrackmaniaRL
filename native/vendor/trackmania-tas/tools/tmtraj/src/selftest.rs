//! Port of `entrec.py --selftest`: validation of the decoder against
//! independent ground truth (map geometry, the ghost's own split-time chunk,
//! and a hand measurement of six 4 m gates).
//!
//! `tests/selftest.rs` runs exactly this and asserts every check passes;
//! `tmtraj selftest` prints the same report the Python printed.

use gbx::record;
use record::{decode_ghost, quat_rotate, Decoded, Sample};

/// Where to look for the reference ghosts.
pub const GHOST_DIRS: &[&str] = &[
    // Checked in at tools/testdata/decoder-goldens/ghosts. This list used to be
    // four /tmp paths; with none of them present `tmtraj selftest` printed
    // "SELFTEST: ALL PASS (0 checks, 0 failed)" and exited 0 — a vacuous pass.
    concat!(env!("CARGO_MANIFEST_DIR"), "/../testdata/decoder-goldens/ghosts"),
];

/// Known "Summer 2026 - 01" geometry (32 m cells, centre = 32*cell + 16).
pub const MAP_GEOM: &[(&str, (f64, f64))] = &[
    ("START", (49.0 * 32.0 + 16.0, 24.0 * 32.0 + 16.0)), // (1584, 784)
    ("CP1", (38.0 * 32.0 + 16.0, 30.0 * 32.0 + 16.0)),   // (1232, 976)
    ("CP2", (1154.0, 1328.0)),                           // gate item, exact
    ("CP3", (42.0 * 32.0 + 16.0, 34.0 * 32.0 + 16.0)),   // (1360, 1104)
    ("FINISH", (42.0 * 32.0 + 16.0, 21.0 * 32.0 + 16.0)), // (1360, 688)
];

fn geom(k: &str) -> (f64, f64) {
    MAP_GEOM.iter().find(|(n, _)| *n == k).unwrap().1
}

/// Independently measured (finish-gate bisection) WR times at six points 4 m
/// apart.
pub const USER_GATE_TIMES_MS: [f64; 6] = [614.0, 946.0, 1188.0, 1388.0, 1563.0, 1720.0];

/// Both selftest cases, with the file names they may appear under. The WR
/// ghost was `01_19538.Ghost.Gbx` in the Python's tree; the copies preserved
/// here (`wr_original_19538`, `p00001_19538`) are byte-identical to it.
pub fn cases() -> Vec<(&'static str, Vec<&'static str>, Vec<i32>)> {
    vec![
        (
            "01_19538.Ghost.Gbx",
            vec![
                "01_19538.Ghost.Gbx",
                "wr_original_19538.Ghost.Gbx",
                "p00001_19538.Ghost.Gbx",
            ],
            vec![7617, 13308, 16316, 19538],
        ),
        (
            "slow_p10000_19812.Ghost.Gbx",
            vec!["slow_p10000_19812.Ghost.Gbx"],
            vec![7630, 13406, 16572, 19812],
        ),
    ]
}

pub fn find_ghost(names: &[&str]) -> Option<String> {
    for d in GHOST_DIRS {
        for n in names {
            let p = format!("{}/{}", d, n);
            if std::path::Path::new(&p).is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn lerp(xs: &[f64], ys: &[f64], xq: f64) -> f64 {
    if xq <= xs[0] {
        return ys[0];
    }
    if xq >= *xs.last().unwrap() {
        return *ys.last().unwrap();
    }
    let (mut lo, mut hi) = (0usize, xs.len() - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= xq {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let f = (xq - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + f * (ys[hi] - ys[lo])
}

fn times(s: &[Sample]) -> Vec<f64> {
    s.iter().map(|x| x.time_ms as f64).collect()
}

fn interp_path(s: &[Sample], tq: f64) -> (f64, f64, f64) {
    let ts = times(s);
    let g = |f: fn(&Sample) -> f64| lerp(&ts, &s.iter().map(f).collect::<Vec<_>>(), tq);
    (g(|p| p.x as f64), g(|p| p.y as f64), g(|p| p.z as f64))
}

/// `_dense`: uniform 1 ms (or `step`) resample of the position track.
fn dense(s: &[Sample], step: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ts = times(s);
    let (t0, t1) = (ts[0], *ts.last().unwrap());
    let n = ((t1 - t0) / step) as usize + 1;
    let xs: Vec<f64> = s.iter().map(|p| p.x as f64).collect();
    let ys: Vec<f64> = s.iter().map(|p| p.y as f64).collect();
    let zs: Vec<f64> = s.iter().map(|p| p.z as f64).collect();
    let t: Vec<f64> = (0..n).map(|i| t0 + i as f64 * step).collect();
    let x = t.iter().map(|&q| lerp(&ts, &xs, q)).collect();
    let y = t.iter().map(|&q| lerp(&ts, &ys, q)).collect();
    let z = t.iter().map(|&q| lerp(&ts, &zs, q)).collect();
    (t, x, y, z)
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

pub struct SelfTest {
    pub ok: bool,
    pub report: String,
    pub checks: usize,
    pub failures: Vec<String>,
    pub skipped: Vec<String>,
}

pub fn selftest(verbose: bool) -> SelfTest {
    let mut ok = true;
    let mut lines: Vec<String> = Vec::new();
    let mut checks = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    macro_rules! say {
        ($($a:tt)*) => {{
            let s = format!($($a)*);
            if verbose { println!("{}", s); }
            lines.push(s);
        }};
    }

    for (label, names, declared) in cases() {
        let Some(path) = find_ghost(&names) else {
            say!("SKIP  {} (not found)", label);
            skipped.push(label.to_string());
            continue;
        };
        let dec: Decoded = match decode_ghost(&path) {
            Ok(d) => d,
            Err(e) => {
                ok = false;
                failures.push(format!("{}: decode failed: {}", label, e));
                say!("FAIL  {}: {}", label, e);
                continue;
            }
        };
        let s = &dec.samples;
        say!("");
        say!("=== {} ===", label);
        say!(
            "  chunk version {}, {} samples @ {} ms, sample size {}, record start={} end={}",
            dec.version,
            s.len(),
            dec.sample_period_ms.unwrap(),
            dec.sample_size,
            dec.start_ms,
            dec.end_ms
        );

        let mut check = |good: bool, msg: String| {
            checks += 1;
            if !good {
                ok = false;
                failures.push(msg.clone());
            }
            let line = format!("  [{}] {}", if good { "PASS" } else { "FAIL" }, msg);
            if verbose {
                println!("{}", line);
            }
            lines.push(line);
        };

        // T0: structural -- the blob must be consumed exactly
        let good = dec.bytes_consumed == dec.bytes_total;
        check(
            good,
            format!(
                "T0 blob fully consumed: {} / {} bytes",
                dec.bytes_consumed, dec.bytes_total
            ),
        );

        // T1: declared checkpoints in the ghost chunk match the ones we were given
        let cps = &dec.checkpoints_ms;
        let good = cps.len() >= declared.len() && cps[..declared.len()] == declared[..];
        check(
            good,
            format!(
                "T1 ghost chunk 0x0309202B checkpoints {:?} == declared {:?}",
                cps, declared
            ),
        );

        // T2: start position == start block centre
        let s0 = &s[0];
        let (sx, sz) = geom("START");
        let d0 = (s0.x as f64 - sx).hypot(s0.z as f64 - sz);
        check(
            d0 < 1.0,
            format!(
                "T2 t=0 position ({:.3}, {:.3}, {:.3}) vs start block centre ({:.0}, -, {:.0}): {:.3} m",
                s0.x, s0.y, s0.z, sx, sz, d0
            ),
        );

        // T3: the car is at each checkpoint's map location at its declared time
        let (tt, xx, _yy, zz) = dense(s, 1.0);
        for (i, key) in ["CP1", "CP2", "CP3"].iter().enumerate() {
            let (gx, gz) = geom(key);
            let tq = declared[i] as f64;
            let (px, _py, pz) = interp_path(s, tq);
            let dist = (px - gx).hypot(pz - gz);
            let mut best = 1e9;
            let mut bt = 0.0;
            for (j, &t) in tt.iter().enumerate() {
                if (t - tq).abs() > 1500.0 {
                    continue;
                }
                let dd = (xx[j] - gx).hypot(zz[j] - gz);
                if dd < best {
                    best = dd;
                    bt = t;
                }
            }
            let tol = if *key == "CP2" { 12.0 } else { 6.0 };
            check(
                best < tol,
                format!(
                    "T3 {}: at t={} decoded ({:.1}, {:.1}); nominal ({:.1}, {:.1}); dist {:.2} m; \
                     closest approach {:.2} m at t={:.0} ({:+.0} ms)",
                    key, declared[i], px, pz, gx, gz, dist, best, bt, bt - tq
                ),
            );
        }

        // T3b: FINISH -- extrapolate the last sample to the declared finish time
        let (gx, gz) = geom("FINISH");
        let last = s.last().unwrap();
        let dt = (declared[3] - last.time_ms) as f64 / 1000.0;
        let (ex, ez) = (last.x as f64 + last.vx as f64 * dt, last.z as f64 + last.vz as f64 * dt);
        let inside = (ex - gx).abs() <= 16.0 && (ez - gz).abs() <= 16.0;
        check(
            inside,
            format!(
                "T3 FINISH: last sample t={} at ({:.1}, {:.1}); extrapolated {:+.0} ms at \
                 {:.0} km/h -> ({:.1}, {:.1}); finish block cell x[{:.0},{:.0}] z[{:.0},{:.0}]",
                last.time_ms,
                last.x,
                last.z,
                dt * 1000.0,
                last.speed_kmh,
                ex,
                ez,
                gx - 16.0,
                gx + 16.0,
                gz - 16.0,
                gz + 16.0
            ),
        );

        // T4: continuity -- no teleports
        let mut mx: f64 = 0.0;
        for w in s.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let dd = (((b.x - a.x) as f64).powi(2) + ((b.y - a.y) as f64).powi(2) + ((b.z - a.z) as f64).powi(2)).sqrt();
            let dt = (b.time_ms - a.time_ms) as f64 / 1000.0;
            mx = mx.max(if dt != 0.0 { dd / dt } else { 0.0 });
        }
        check(
            mx < 200.0,
            format!("T4 continuity: max inter-sample implied speed {:.1} km/h", mx * 3.6),
        );

        // T5: speed sanity
        let sp: Vec<f64> = s.iter().map(|p| p.speed_kmh as f64).collect();
        let peak = sp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let peak_i = sp.iter().position(|&v| v == peak).unwrap();
        check(
            sp[0] < 5.0 && (300.0..=600.0).contains(&peak),
            format!(
                "T5 speed: v(0)={:.2} km/h, peak {:.1} km/h at t={}",
                sp[0], peak, s[peak_i].time_ms
            ),
        );

        // T6: speed field == |finite difference of position|
        let mut err = Vec::new();
        for i in 1..s.len() - 1 {
            let dt = (s[i + 1].time_ms - s[i - 1].time_ms) as f64 / 1000.0;
            let fd = [
                (s[i + 1].x - s[i - 1].x) as f64 / dt,
                (s[i + 1].y - s[i - 1].y) as f64 / dt,
                (s[i + 1].z - s[i - 1].z) as f64 / dt,
            ];
            let fdm = (fd[0] * fd[0] + fd[1] * fd[1] + fd[2] * fd[2]).sqrt();
            err.push((fdm - s[i].speed_ms as f64).abs());
        }
        let mut err_sorted = err.clone();
        let med = median(&mut err_sorted);
        let p95 = err_sorted[(err_sorted.len() as f64 * 0.95) as usize];
        check(
            med < 1.0,
            format!(
                "T6 |d(pos)/dt| vs decoded speed: median error {:.3} m/s, p95 {:.3} m/s",
                med, p95
            ),
        );

        // T7: velocity vector direction == d(pos)/dt direction
        let mut dots = Vec::new();
        for i in 1..s.len() - 1 {
            let dt = (s[i + 1].time_ms - s[i - 1].time_ms) as f64 / 1000.0;
            let fd = [
                (s[i + 1].x - s[i - 1].x) as f64 / dt,
                (s[i + 1].y - s[i - 1].y) as f64 / dt,
                (s[i + 1].z - s[i - 1].z) as f64 / dt,
            ];
            let n1 = (fd[0] * fd[0] + fd[1] * fd[1] + fd[2] * fd[2]).sqrt();
            let v = [s[i].vx as f64, s[i].vy as f64, s[i].vz as f64];
            let n2 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if n1 > 5.0 && n2 > 5.0 {
                dots.push((fd[0] * v[0] + fd[1] * v[1] + fd[2] * v[2]) / (n1 * n2));
            }
        }
        let dmin = dots.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut dsorted = dots.clone();
        let dmed = median(&mut dsorted);
        check(
            dmed > 0.999,
            format!(
                "T7 velocity direction vs path tangent: median cos {:.5}, min {:.4}",
                dmed, dmin
            ),
        );

        // T8: quaternion is unit and its local +Z is the car's forward axis
        let mut qn = Vec::new();
        let mut fw = Vec::new();
        for i in 1..s.len() - 1 {
            let q = [s[i].qx as f64, s[i].qy as f64, s[i].qz as f64, s[i].qw as f64];
            qn.push((1.0 - (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()).abs());
            let f = quat_rotate(q, [0.0, 0.0, 1.0]);
            let v = [s[i].vx as f64, s[i].vy as f64, s[i].vz as f64];
            let n2 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if n2 > 20.0 {
                fw.push((f[0] * v[0] + f[1] * v[1] + f[2] * v[2]) / n2);
            }
        }
        let qmax = qn.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut fws = fw.clone();
        let fwmed = median(&mut fws);
        check(
            qmax < 1e-3 && fwmed > 0.99,
            format!(
                "T8 quaternion: max |1-|q|| = {:.2e}; median cos(local +Z, velocity) = {:.4}",
                qmax, fwmed
            ),
        );

        // T9 (WR only): independent finish-gate measurement
        if label.starts_with("01_") {
            let (tt, xx, yy, zz) = dense(s, 1.0);
            let mut arc = vec![0.0f64];
            for i in 1..tt.len() {
                let d = ((xx[i] - xx[i - 1]).powi(2)
                    + (yy[i] - yy[i - 1]).powi(2)
                    + (zz[i] - zz[i - 1]).powi(2))
                .sqrt();
                arc.push(arc[i - 1] + d);
            }
            let at: Vec<f64> = USER_GATE_TIMES_MS.iter().map(|&t| lerp(&tt, &arc, t)).collect();
            let gaps: Vec<f64> = at.windows(2).map(|w| w[1] - w[0]).collect();
            let good = gaps.iter().all(|g| (g - 4.0).abs() < 0.25);
            check(
                good,
                format!(
                    "T9 independent 4 m gate timings {:?} -> arc-length gaps {:?} m (expect 4.00)",
                    USER_GATE_TIMES_MS.map(|v| v as i64),
                    gaps.iter().map(|g| (g * 1000.0).round() / 1000.0).collect::<Vec<_>>()
                ),
            );
        }

        // T10: gear quantisation -- raw gear must be 1 + 4*k
        let mut raws: Vec<u8> = s.iter().map(|p| p.gear_raw).collect();
        raws.sort_unstable();
        raws.dedup();
        let bad: Vec<u8> = raws.iter().cloned().filter(|r| (r - 1) % 4 != 0).collect();
        check(
            bad.is_empty(),
            format!(
                "T10 gear byte quantised as 1+4*gear: raw values {:?}{}",
                raws,
                if bad.is_empty() {
                    String::new()
                } else {
                    format!(" BAD:{:?}", bad)
                }
            ),
        );
    }

    say!("");
    say!(
        "SELFTEST: {} ({} checks, {} failed)",
        if ok { "ALL PASS" } else { "FAILURES PRESENT" },
        checks,
        failures.len()
    );
    SelfTest {
        ok,
        report: lines.join("\n"),
        checks,
        failures,
        skipped,
    }
}
