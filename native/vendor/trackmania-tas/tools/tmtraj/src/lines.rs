//! `lines` -- arc-length-parameterised racing-line comparison and clustering.
//!
//! ONE implementation merged from the two overlapping Python originals:
//!
//!   * `tmtas/trajectories/cluster_lines.py` -- densify to 5 ms, resample by
//!     arc length, measure each run's SIGNED LATERAL offset from a reference
//!     line by closest-point projection, distance = RMS difference of the two
//!     lateral profiles. This is [`Metric::Projection`] and is the default; the
//!     report in the repo uses it for all its absolute numbers.
//!   * `tmtas/code/lines.py` -- resample the raw samples by arc length and
//!     compare station k of A with station k of B directly ([`Metric::Station`]),
//!     or elastically with a Sakoe-Chiba-banded DTW ([`Metric::Dtw`]). Station
//!     k is at the same FRACTION of each run's own total arc length, so runs of
//!     different total length are compared at points displaced along the track;
//!     that is a real flaw (see the report) but it is kept here because the
//!     published cluster counts come from it.
//!
//! Everything downstream -- complete-linkage clustering with a metres-valued
//! threshold, the per-run tables, the ASCII plots, the seed-per-line output --
//! is shared between the three metrics.

use crate::json::{self, J};
use std::collections::BTreeMap;

pub const MAP_GEOM: &[(&str, (f64, f64))] = &[
    ("CP1", (1232.0, 976.0)),
    ("CP2", (1154.0, 1328.0)),
    ("CP3", (1360.0, 1104.0)),
    ("FINISH", (1360.0, 688.0)),
    ("START", (1584.0, 784.0)),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// cluster_lines.py: RMS difference of lateral offsets from the reference.
    Projection,
    /// lines.py: RMS point-to-point separation of fraction-matched stations.
    Station,
    /// lines.py --dtw: banded dynamic time warping over the same stations.
    Dtw,
}

impl Metric {
    pub fn parse(s: &str) -> Option<Metric> {
        match s {
            "projection" | "proj" => Some(Metric::Projection),
            "station" | "lines" => Some(Metric::Station),
            "dtw" => Some(Metric::Dtw),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Metric::Projection => "projection",
            Metric::Station => "station",
            Metric::Dtw => "dtw",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Run {
    pub name: String,
    pub time_ms: i64,
    pub checkpoints_ms: Vec<i64>,
    pub t: Vec<f64>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub v: Vec<f64>,
}

/// A resampled station: `(s, x, y, z, speed)`.
pub type Station = (f64, f64, f64, f64, f64);

/// Load a directory of per-run JSON files, in sorted-filename order
/// (`sorted(glob.glob(...))` in both Python originals).
pub fn load_dir(dir: &str) -> Result<Vec<Run>, String> {
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {}", dir, e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().to_string())
        .filter(|p| p.ends_with(".json"))
        .collect();
    files.sort();
    let mut runs = Vec::new();
    for f in files {
        let txt = std::fs::read_to_string(&f).map_err(|e| format!("{}: {}", f, e))?;
        let j = json::parse(&txt).map_err(|e| format!("{}: {}", f, e))?;
        let s = j.get("samples").ok_or("no samples")?.arr();
        let name = match j.get("name") {
            Some(J::Str(s)) => s.clone(),
            _ => f.rsplit('/').next().unwrap().to_string(),
        };
        runs.push(Run {
            name,
            time_ms: j.get("time_ms").map_or(0, |v| v.int()),
            checkpoints_ms: j
                .get("checkpoints_ms")
                .map_or_else(Vec::new, |v| v.arr().iter().map(|e| e.int()).collect()),
            t: s.iter().map(|e| e.get("t").unwrap().num()).collect(),
            x: s.iter().map(|e| e.get("x").unwrap().num()).collect(),
            y: s.iter().map(|e| e.get("y").unwrap().num()).collect(),
            z: s.iter().map(|e| e.get("z").unwrap().num()).collect(),
            v: s
                .iter()
                .map(|e| e.get("speed").map_or(0.0, |v| v.num()))
                .collect(),
        });
    }
    Ok(runs)
}

// ---------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------

/// `cluster_lines.densify`: linear resample onto a uniform `step_ms` time grid.
pub fn densify(r: &Run, step_ms: f64) -> (Vec<f64>, [Vec<f64>; 4]) {
    let t = &r.t;
    let cols: [&Vec<f64>; 4] = [&r.x, &r.y, &r.z, &r.v];
    let mut out_t = Vec::new();
    let mut out: [Vec<f64>; 4] = Default::default();
    let mut tt = t[0];
    let mut i = 0usize;
    let last = *t.last().unwrap();
    while tt <= last {
        while i + 2 < t.len() && t[i + 1] < tt {
            i += 1;
        }
        let f = if t[i + 1] != t[i] {
            (tt - t[i]) / (t[i + 1] - t[i])
        } else {
            0.0
        };
        for k in 0..4 {
            out[k].push(cols[k][i] + f * (cols[k][i + 1] - cols[k][i]));
        }
        out_t.push(tt);
        tt += step_ms;
    }
    (out_t, out)
}

/// Cumulative HORIZONTAL distance along a path (y is height in TM).
pub fn arclen(x: &[f64], z: &[f64]) -> Vec<f64> {
    let mut s = Vec::with_capacity(x.len());
    s.push(0.0);
    for i in 1..x.len() {
        let d = (x[i] - x[i - 1]).hypot(z[i] - z[i - 1]);
        s.push(s[i - 1] + d);
    }
    s
}

/// `cluster_lines.resample_by_arc`: densify to 5 ms, then take `n_stations`
/// points equally spaced by arc length. Returns the stations and the total
/// path length.
pub fn resample_by_arc(r: &Run, n_stations: usize) -> (Vec<Station>, f64) {
    let (_, d) = densify(r, 5.0);
    let (x, y, z, v) = (&d[0], &d[1], &d[2], &d[3]);
    let s = arclen(x, z);
    let total = *s.last().unwrap();
    let mut out = Vec::with_capacity(n_stations);
    let mut j = 0usize;
    for k in 0..n_stations {
        let sq = total * k as f64 / (n_stations - 1) as f64;
        while j + 2 < s.len() && s[j + 1] < sq {
            j += 1;
        }
        let den = {
            let d = s[j + 1] - s[j];
            if d == 0.0 {
                1e-9
            } else {
                d
            }
        };
        let f = (sq - s[j]) / den;
        out.push((
            sq,
            x[j] + f * (x[j + 1] - x[j]),
            y[j] + f * (y[j + 1] - y[j]),
            z[j] + f * (z[j + 1] - z[j]),
            v[j] + f * (v[j + 1] - v[j]),
        ));
    }
    (out, total)
}

/// `lines.resample`: the same idea applied to the RAW samples (no densify),
/// used by [`Metric::Station`] and [`Metric::Dtw`].
pub fn resample_raw(r: &Run, n: usize) -> (Vec<[f64; 3]>, Vec<f64>, f64) {
    let s = arclen(&r.x, &r.z);
    let total = *s.last().unwrap();
    assert!(total > 0.0, "degenerate path");
    let mut out_p = Vec::with_capacity(n);
    let mut out_e = Vec::with_capacity(n);
    let mut j = 0usize;
    for k in 0..n {
        let target = total * k as f64 / (n - 1) as f64;
        while j + 1 < s.len() && s[j + 1] < target {
            j += 1;
        }
        if j + 1 >= s.len() {
            let l = r.x.len() - 1;
            out_p.push([r.x[l], r.y[l], r.z[l]]);
            out_e.push(*r.v.last().unwrap());
            continue;
        }
        let span = s[j + 1] - s[j];
        let f = if span <= 0.0 {
            0.0
        } else {
            (target - s[j]) / span
        };
        out_p.push([
            r.x[j] + f * (r.x[j + 1] - r.x[j]),
            r.y[j] + f * (r.y[j + 1] - r.y[j]),
            r.z[j] + f * (r.z[j + 1] - r.z[j]),
        ]);
        out_e.push(r.v[j] + f * (r.v[j + 1] - r.v[j]));
    }
    (out_p, out_e, total)
}

/// `cluster_lines.lateral_profile`: signed lateral offset (metres, + = left of
/// the reference heading) of `run`'s path at each reference station. For each
/// station take the reference tangent, find the closest point of the run's
/// densified path, and project the separation onto the reference normal.
pub fn lateral_profile(ref_stations: &[Station], run: &Run) -> Vec<f64> {
    let (_, d) = densify(run, 5.0);
    let (x, z) = (&d[0], &d[2]);
    let n = ref_stations.len();
    let mut prof = Vec::with_capacity(n);
    let mut hint = 0usize;
    for (i, &(_, rx, _ry, rz, _rv)) in ref_stations.iter().enumerate() {
        // reference tangent
        let j0 = i.saturating_sub(1);
        let j1 = (i + 1).min(n - 1);
        let mut tx = ref_stations[j1].1 - ref_stations[j0].1;
        let mut tz = ref_stations[j1].3 - ref_stations[j0].3;
        let tn = {
            let h = tx.hypot(tz);
            if h == 0.0 {
                1e-9
            } else {
                h
            }
        };
        tx /= tn;
        tz /= tn;
        let (nx, nz) = (-tz, tx); // left normal in the x-z plane
        // closest point on the run path (search forward from the last hint)
        let mut best = 1e18f64;
        let mut bi = hint;
        let lo = hint.saturating_sub(400);
        for k in lo..x.len() {
            let dd = (x[k] - rx).powi(2) + (z[k] - rz).powi(2);
            if dd < best {
                best = dd;
                bi = k;
            } else if dd > best + 10000.0 {
                break;
            }
        }
        hint = bi;
        let (dx, dz) = (x[bi] - rx, z[bi] - rz);
        prof.push(dx * nx + dz * nz);
    }
    prof
}

pub fn rms(a: &[f64]) -> f64 {
    (a.iter().map(|v| v * v).sum::<f64>() / a.len() as f64).sqrt()
}

/// `lines.rms_separation`: RMS point-to-point horizontal separation of two
/// equally-stationed paths.
pub fn rms_separation(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let n = a.len().min(b.len());
    let mut acc = 0.0;
    for k in 0..n {
        acc += (a[k][0] - b[k][0]).hypot(a[k][2] - b[k][2]).powi(2);
    }
    (acc / n as f64).sqrt()
}

/// `lines.dtw_separation`: elastic alternative, Sakoe-Chiba band keeps it
/// O(n*band).
pub fn dtw_separation(a: &[[f64; 3]], b: &[[f64; 3]], band: usize) -> f64 {
    let (n, m) = (a.len(), b.len());
    let inf = f64::INFINITY;
    let mut prev = vec![inf; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        let mut cur = vec![inf; m + 1];
        let lo = 1.max(i.saturating_sub(band));
        let hi = m.min(i + band);
        for j in lo..=hi {
            let d = (a[i - 1][0] - b[j - 1][0]).hypot(a[i - 1][2] - b[j - 1][2]);
            cur[j] = d + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        prev = cur;
    }
    prev[m] / n.max(m) as f64
}


// ---------------------------------------------------------------------------
// clustering
// ---------------------------------------------------------------------------

/// Agglomerative, complete linkage, threshold in METRES: two groups join only
/// if EVERY pair across the merged group stays within `eps`. Merges the
/// globally closest admissible pair first, exactly as both Python versions do.
pub fn cluster(d: &[Vec<f64>], eps: f64) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = (0..d.len()).map(|i| vec![i]).collect();
    loop {
        let mut best: Option<f64> = None;
        let mut pair = (0usize, 0usize);
        for a in 0..clusters.len() {
            for b in (a + 1)..clusters.len() {
                let mut worst = f64::NEG_INFINITY;
                for &i in &clusters[a] {
                    for &j in &clusters[b] {
                        if d[i][j] > worst {
                            worst = d[i][j];
                        }
                    }
                }
                if best.is_none() || worst < best.unwrap() {
                    best = Some(worst);
                    pair = (a, b);
                }
            }
        }
        match best {
            None => break,
            Some(bv) if bv > eps => break,
            _ => {}
        }
        let (a, b) = pair;
        let moved = clusters.remove(b);
        clusters[a].extend(moved);
    }
    clusters
}

pub fn spread(d: &[Vec<f64>], c: &[usize]) -> f64 {
    let mut m: f64 = 0.0;
    for &i in c {
        for &j in c {
            m = m.max(d[i][j]);
        }
    }
    m
}

// ---------------------------------------------------------------------------
// the analysis, all metrics
// ---------------------------------------------------------------------------

pub struct Analysis {
    pub metric: Metric,
    pub stations: usize,
    pub names: Vec<String>,
    pub runs: Vec<Run>,
    pub ref_idx: usize,
    pub ref_stations: Vec<Station>,
    pub ref_total: f64,
    /// projection metric only: per-run lateral profile against the reference
    pub profiles: BTreeMap<String, Vec<f64>>,
    /// per-run arc-length stations (projection metric) for the plots
    pub stations_by_run: BTreeMap<String, Vec<Station>>,
    pub totals: BTreeMap<String, f64>,
    pub d: Vec<Vec<f64>>,
}

/// Order the runs the way the corresponding Python did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sort {
    /// cluster_lines.py: fastest first (stable, so ties keep filename order).
    Time,
    /// lines.py: filename order, untouched.
    Name,
}

pub fn analyse(
    mut runs: Vec<Run>,
    metric: Metric,
    stations: usize,
    ref_name: Option<&str>,
    sort: Sort,
) -> Analysis {
    if sort == Sort::Time {
        runs.sort_by(|a, b| a.time_ms.cmp(&b.time_ms));
    }
    let names: Vec<String> = runs.iter().map(|r| r.name.clone()).collect();
    let ref_idx = match ref_name {
        Some(n) => names.iter().position(|x| x == n).unwrap_or_else(|| {
            panic!("reference run {:?} not found", n);
        }),
        None => 0,
    };
    let (ref_stations, ref_total) = resample_by_arc(&runs[ref_idx], stations);

    let mut stations_by_run = BTreeMap::new();
    let mut totals = BTreeMap::new();
    let mut profiles = BTreeMap::new();
    let mut raw_st: Vec<Vec<[f64; 3]>> = Vec::new();
    for r in &runs {
        let (st, tot) = resample_by_arc(r, stations);
        stations_by_run.insert(r.name.clone(), st);
        totals.insert(r.name.clone(), tot);
        // the lateral profile is the report's descriptor for every metric; only
        // Projection also uses it as the distance.
        profiles.insert(r.name.clone(), lateral_profile(&ref_stations, r));
        if metric != Metric::Projection {
            raw_st.push(resample_raw(r, stations).0);
        }
    }

    let n = runs.len();
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v = match metric {
                Metric::Projection => {
                    let a = &profiles[&names[i]];
                    let b = &profiles[&names[j]];
                    let diff: Vec<f64> = a.iter().zip(b).map(|(p, q)| p - q).collect();
                    rms(&diff)
                }
                Metric::Station => rms_separation(&raw_st[i], &raw_st[j]),
                Metric::Dtw => dtw_separation(&raw_st[i], &raw_st[j], 12),
            };
            d[i][j] = v;
            d[j][i] = v;
        }
    }

    Analysis {
        metric,
        stations,
        names,
        runs,
        ref_idx,
        ref_stations,
        ref_total,
        profiles,
        stations_by_run,
        totals,
        d,
    }
}

impl Analysis {
    pub fn ref_name(&self) -> &str {
        &self.names[self.ref_idx]
    }

    /// Station index nearest each checkpoint on the reference line.
    pub fn cp_stations(&self) -> Vec<(&'static str, usize)> {
        let mut out = Vec::new();
        for (k, (gx, gz)) in MAP_GEOM {
            if *k == "START" {
                continue;
            }
            let mut best = f64::INFINITY;
            let mut bi = 0;
            for (i, p) in self.ref_stations.iter().enumerate() {
                let d = (p.1 - gx).powi(2) + (p.3 - gz).powi(2);
                if d < best {
                    best = d;
                    bi = i;
                }
            }
            out.push((*k, bi));
        }
        out
    }

    pub fn pair_distances(&self) -> Vec<f64> {
        let n = self.names.len();
        let mut all: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                all.push(self.d[i][j]);
            }
        }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all
    }
}

// ---------------------------------------------------------------------------
// display
// ---------------------------------------------------------------------------

/// `cluster_lines.ascii_xz`
pub fn ascii_xz(
    stations_by_run: &BTreeMap<String, Vec<Station>>,
    labels: &[(String, char)],
    marks: &[(&str, (f64, f64))],
    w: usize,
    h: usize,
) -> String {
    let xs: Vec<f64> = stations_by_run.values().flatten().map(|p| p.1).collect();
    let zs: Vec<f64> = stations_by_run.values().flatten().map(|p| p.3).collect();
    let x0 = xs.iter().cloned().fold(f64::INFINITY, f64::min) - 10.0;
    let x1 = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 10.0;
    let z0 = zs.iter().cloned().fold(f64::INFINITY, f64::min) - 10.0;
    let z1 = zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 10.0;
    let mut grid = vec![vec![' '; w]; h];
    let cell = |x: f64, z: f64| -> (usize, usize) {
        let cx = ((x - x0) / (x1 - x0) * (w - 1) as f64) as i64;
        let cy = ((z1 - z) / (z1 - z0) * (h - 1) as f64) as i64;
        (
            cy.clamp(0, h as i64 - 1) as usize,
            cx.clamp(0, w as i64 - 1) as usize,
        )
    };
    for (name, ch) in labels {
        for p in &stations_by_run[name] {
            let (r, c) = cell(p.1, p.3);
            if grid[r][c] == ' ' {
                grid[r][c] = *ch;
            }
        }
    }
    for (nm, (mx, mz)) in marks {
        let (r, c) = cell(*mx, *mz);
        grid[r][c] = if *nm == "FINISH" {
            'F'
        } else {
            nm.chars().next().unwrap()
        };
    }
    let mut out = vec![format!(
        "+{}+ x:[{:.0}..{:.0}] z:[{:.0}..{:.0}]  (up = +z, left = +x)",
        "-".repeat(w),
        x1,
        x0,
        z0,
        z1
    )];
    for row in grid {
        out.push(format!("|{}|", row.into_iter().collect::<String>()));
    }
    out.push(format!("+{}+", "-".repeat(w)));
    out.join("\n")
}

/// `cluster_lines.ascii_series`
pub fn ascii_series(
    series: &[(char, Vec<(f64, f64)>)],
    w: usize,
    h: usize,
    ylab: &str,
    xlab: &str,
) -> String {
    let xs: Vec<f64> = series.iter().flat_map(|(_, p)| p.iter().map(|q| q.0)).collect();
    let ys: Vec<f64> = series.iter().flat_map(|(_, p)| p.iter().map(|q| q.1)).collect();
    let x0 = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x1 = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y0 = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let y1 = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut grid = vec![vec![' '; w]; h];
    for (ch, pts) in series {
        for (x, y) in pts {
            let c = ((x - x0) / if x1 - x0 != 0.0 { x1 - x0 } else { 1.0 } * (w - 1) as f64) as i64;
            let r = ((y1 - y) / if y1 - y0 != 0.0 { y1 - y0 } else { 1.0 } * (h - 1) as f64) as i64;
            if r >= 0 && (r as usize) < h && c >= 0 && (c as usize) < w {
                let (r, c) = (r as usize, c as usize);
                if grid[r][c] == ' ' {
                    grid[r][c] = *ch;
                }
            }
        }
    }
    let mut out = vec![format!("  {:8.1} +{}+", y1, "-".repeat(w))];
    for row in grid {
        out.push(format!(
            "  {:8} |{}|",
            "",
            row.into_iter().collect::<String>()
        ));
    }
    out.push(format!("  {:8.1} +{}+  {}", y0, "-".repeat(w), ylab));
    out.push(format!(
        "  {:8}  {:<width$}{}",
        "",
        format!(" {:.0}", x0),
        format!("{:.0}  {}", x1, xlab),
        width = w - 10
    ));
    out.join("\n")
}

/// `lines.demo`: two clearly distinct synthetic lines plus noise, to validate
/// the algorithm without any ghost data.
pub fn demo() -> Vec<Run> {
    let mut runs = Vec::new();
    for i in 0..6 {
        let inside = i < 3;
        let (mut x, mut y, mut z, mut v, mut t) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for k in 0..400 {
            let u = k as f64 / 399.0;
            let ang = std::f64::consts::PI * u;
            let r = (if inside { 40.0 } else { 52.0 }) + (i % 3) as f64 * 0.4;
            x.push(r * ang.sin());
            y.push(0.0);
            z.push(-r * ang.cos());
            v.push((if inside { 150.0 } else { 175.0 }) + 40.0 * ang.sin());
            t.push(k as f64 * 10.0);
        }
        runs.push(Run {
            name: format!("{}{}", if inside { "inside" } else { "outside" }, i),
            time_ms: 20000 + i * 7,
            checkpoints_ms: Vec::new(),
            t,
            x,
            y,
            z,
            v,
        });
    }
    runs
}

/// The `--out` JSON, same shape as `cluster_lines.py --out`.
pub fn clusters_json(a: &Analysis, eps_list: &[f64], cl: &[(f64, Vec<Vec<usize>>)]) -> String {
    use json::py_repr;
    let mut o = String::new();
    o.push_str(&format!(
        "{{\"result\": {{\"reference\": \"{}\", \"n_runs\": {}, \"clusters\": {{",
        a.ref_name(),
        a.names.len()
    ));
    for (i, (eps, groups)) in cl.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push_str(&format!("\"{:.1}\": [", eps));
        for (gi, g) in groups.iter().enumerate() {
            if gi > 0 {
                o.push_str(", ");
            }
            let mut mem: Vec<usize> = g.clone();
            mem.sort_by_key(|&i| a.runs[i].time_ms);
            o.push_str("{\"members\": [");
            for (mi, m) in mem.iter().enumerate() {
                if mi > 0 {
                    o.push_str(", ");
                }
                o.push_str(&format!("\"{}\"", a.names[*m]));
            }
            o.push_str(&format!(
                "], \"spread_m\": {}}}",
                py_repr(spread(&a.d, g))
            ));
        }
        o.push(']');
    }
    let _ = eps_list;
    o.push_str("}}, \"distance_matrix\": [");
    for (i, row) in a.d.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push('[');
        for (j, v) in row.iter().enumerate() {
            if j > 0 {
                o.push_str(", ");
            }
            o.push_str(&py_repr(*v));
        }
        o.push(']');
    }
    o.push_str("], \"names\": [");
    for (i, n) in a.names.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push_str(&format!("\"{}\"", n));
    }
    o.push_str("], \"lateral_profiles\": {");
    for (i, n) in a.names.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push_str(&format!("\"{}\": [", n));
        if let Some(p) = a.profiles.get(n) {
            for (k, v) in p.iter().enumerate() {
                if k > 0 {
                    o.push_str(", ");
                }
                o.push_str(&py_repr(*v));
            }
        }
        o.push(']');
    }
    o.push_str("}, \"ref_stations\": [");
    for (i, p) in a.ref_stations.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push_str(&format!(
            "[{}, {}, {}, {}]",
            py_repr(p.0),
            py_repr(p.1),
            py_repr(p.2),
            py_repr(p.3)
        ));
    }
    o.push_str("]}");
    o
}
