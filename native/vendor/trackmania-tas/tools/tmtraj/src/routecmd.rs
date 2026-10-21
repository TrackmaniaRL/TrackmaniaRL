//! `tmtraj route` — geometric queries over an exported trajectory CSV.
//!
//! Written for the 173691 landing arm, where every question of the shape "does
//! this run ever come near that lip, and in what state" was being answered with
//! `awk` one-liners that nobody could re-run or check. A query that decides
//! whether an artefact is publishable belongs in the tool.
//!
//! The CSV is whatever `tmtraj export --csv` writes: a header line naming the
//! columns, then one row per sample. Columns are looked up BY NAME, never by
//! index, because two producers in this project disagree on column order and a
//! positional reader silently measures the wrong field. CRLF is stripped: the
//! banked CSVs have it and a naive parse turns the last column into a string.
//! A `.Ghost.Gbx` / `.Replay.Gbx` is decoded into the same table, so the same
//! query runs on a banked file without an export step in between.
//!
//! # Two queries that are not row filters
//!
//! `--where z<692` selects SAMPLES. Two questions on this project are about the
//! points BETWEEN them, and answering either by picking a nearby row is how two
//! arms got a wrong number:
//!
//! * **`--cross`** — where the run crossed a plane, linearly interpolated
//!   between the two samples that straddle it, and **every** crossing rather
//!   than the first. The grid is 50 ms: at 45 m/s the nearest sample is up to
//!   1.1 m from the plane, and a run with a pit loop crosses the same plane
//!   three times (267460's does).
//! * **`--margin`** — over a FAMILY of runs, the worst margin at a plane
//!   against per-axis thresholds, sorted. That is the frontier of a reachable
//!   set: "how close did the best of these come, and in which coordinate did it
//!   run out". It is the same `min` that `tmsearch --key corner:` scores, so a
//!   hand-built family and a search are comparable on one number; on 267460 the
//!   two agreed to 0.1 m.

use crate::cli;

const USAGE: &str = "\
usage: tmtraj route CSV|GHOST... [query...]

  --summary                     rows, span, bbox, path length, first and last
                                -- of the SELECTION, so --where narrows it
  --pace                        how the selection's TIME divides by speed band
                                (<5, 5-20, 20-40, 40-80, 80-120, >=120 km/h),
                                skipping any gap over 1 s so a splice junction
                                is not counted as a duration
  --segments A,B,C              the per-segment ledger in ONE invocation: split
                                the selection at these millisecond boundaries
                                and print seconds, metres, mean km/h and the
                                seconds under 5 and under 20 km/h for each
  --near X,Y,Z [--top N]        the N samples closest to a point (default 5)
  --where 'COL OP VALUE'        filter; repeatable, ANDed. OP is < <= > >= == !=
  --first N / --last N          print the first / last N rows of the selection
  --every N                     print every Nth row of the selection
  --cols a,b,c                  which columns to print (default t,x,y,z,km/h,vy)
                                plus a derived `s`: cumulative path length in
                                metres from the first row, which is the same
                                quantity the fork search reports progress in

  --cross AXIS=VALUE            EVERY crossing of a plane, interpolated between
                                the two samples that straddle it, with its
                                direction. Repeatable.
  --margin AXIS=V[,AXIS=V]      over ALL the files given, at the --cross plane:
                                the worst margin against these thresholds, and
                                which axis ran out. One line per file, sorted.

Times print as seconds. A selection that is empty says so and exits 1, because
'no rows' and 'no output' are different answers.
";

/// A usage/read error: say what is wrong and stop. Never a default.
fn die(msg: &str) -> ! {
    eprintln!("tmtraj route: {}", msg);
    std::process::exit(2)
}

pub(crate) struct Table {
    pub(crate) names: Vec<String>,
    pub(crate) rows: Vec<Vec<f64>>,
}

impl Table {
    pub(crate) fn col(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
    /// Append `s`: cumulative path length in metres from the table's FIRST row.
    ///
    /// It is a derived column rather than a separate query because that is
    /// what makes it composable — `--where 's>500'`, `--cols time_ms,s,x,y,z`
    /// and `--first 1` then answer "where is the 500 m mark" with the tools
    /// already here. The search's reference-line arclength is this same
    /// quantity measured from the same first sample, so a `DNF 507 m of 560`
    /// can be read straight off the reference trace.
    fn add_arclength(&mut self) {
        let (Some(cx), Some(cy), Some(cz)) = (self.col("x"), self.col("y"), self.col("z")) else {
            return;
        };
        if self.col("s").is_some() {
            return;
        }
        let mut acc = 0.0f64;
        let mut prev: Option<(f64, f64, f64)> = None;
        // A CSV row may carry more fields than the header names; `s` has to
        // land at the index the name will have, so the row is cut to the
        // header first.
        let base = self.names.len();
        for row in self.rows.iter_mut() {
            let p = (row[cx], row[cy], row[cz]);
            if let Some(q) = prev {
                let d = ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2) + (p.2 - q.2).powi(2)).sqrt();
                if d.is_finite() {
                    acc += d;
                }
            }
            prev = Some(p);
            row.truncate(base);
            row.push(acc);
        }
        self.names.push("s".to_string());
    }
    /// Column index or a fatal error naming what the file does have.
    pub(crate) fn need(&self, name: &str) -> usize {
        self.col(name).unwrap_or_else(|| {
            die(&format!(
                "no column {:?} in this CSV; it has: {}",
                name,
                self.names.join(",")
            ))
        })
    }
}

pub(crate) fn load(path: &str) -> Table {
    // A ghost or replay decodes into the same table. `tmtraj export --csv`
    // writes these columns from this decoder, so the two are the same data and
    // the query must not care which one it was handed.
    let low = path.to_ascii_lowercase();
    if low.ends_with(".gbx") {
        let d = gbx::record::decode_ghost(path)
            .unwrap_or_else(|e| die(&format!("cannot decode {}: {}", path, e)));
        if d.samples.is_empty() {
            die(&format!("{}: no vehicle samples", path));
        }
        let names: Vec<String> = ["time_ms", "x", "y", "z", "speed_kmh", "vx", "vy", "vz", "yaw",
            "pitch", "roll", "gear", "is_ground_contact", "is_turbo", "steer", "gas", "brake"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rows = d
            .samples
            .iter()
            .map(|s| {
                vec![
                    s.time_ms as f64, s.x as f64, s.y as f64, s.z as f64, s.speed_kmh as f64, s.vx as f64, s.vy as f64, s.vz as f64, s.yaw as f64,
                    s.pitch as f64, s.roll as f64, s.gear as f64,
                    if s.is_ground_contact { 1.0 } else { 0.0 },
                    if s.is_turbo { 1.0 } else { 0.0 },
                    s.steer as f64, s.gas as f64, s.brake as f64,
                ]
            })
            .collect();
        let mut t = Table { names, rows };
        t.add_arclength();
        return t;
    }
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_else(|| die("empty CSV"));
    let names: Vec<String> =
        header.trim_end_matches('\r').split(',').map(|s| s.trim().to_string()).collect();
    let mut rows = Vec::new();
    for line in lines {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let row: Vec<f64> = line
            .split(',')
            .map(|f| f.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        if row.len() < names.len() {
            continue;
        }
        rows.push(row);
    }
    let mut t = Table { names, rows };
    t.add_arclength();
    t
}

struct Pred {
    col: usize,
    op: String,
    val: f64,
    text: String,
}

fn parse_pred(t: &Table, s: &str) -> Pred {
    // "y>130", "y > 130", "speed_kmh >= 90"
    let ops = ["<=", ">=", "==", "!=", "<", ">"];
    for op in ops {
        if let Some(i) = s.find(op) {
            let lhs = s[..i].trim().to_string();
            let rhs = s[i + op.len()..].trim();
            let val: f64 = rhs
                .parse()
                .unwrap_or_else(|_| die(&format!("--where {:?}: {:?} is not a number", s, rhs)));
            return Pred { col: t.need(&lhs), op: op.to_string(), val, text: s.to_string() };
        }
    }
    die(&format!("--where {:?}: no comparison operator (< <= > >= == !=)", s))
}

fn holds(p: &Pred, row: &[f64]) -> bool {
    let v = row[p.col];
    if v.is_nan() {
        return false;
    }
    match p.op.as_str() {
        "<" => v < p.val,
        "<=" => v <= p.val,
        ">" => v > p.val,
        ">=" => v >= p.val,
        "==" => v == p.val,
        "!=" => v != p.val,
        _ => unreachable!(),
    }
}

fn fmt_row(t: &Table, row: &[f64], cols: &[usize], tcol: usize) -> String {
    let mut out = Vec::new();
    for &c in cols {
        let v = row[c];
        if c == tcol {
            out.push(format!("{:>10}", crate::fmt::secs(v as i64)));
        } else if v.is_nan() {
            out.push(format!("{:>10}", "-"));
        } else {
            out.push(format!("{:>10.3}", v));
        }
    }
    let _ = t;
    out.join(" ")
}

/// A plane to cross: which column, and at what value.
pub(crate) struct Plane {
    pub(crate) axis: String,
    pub(crate) col: usize,
    pub(crate) at: f64,
}

pub(crate) fn parse_plane(t: &Table, s: &str) -> Plane {
    let Some((ax, v)) = s.split_once('=') else {
        die(&format!("--cross {:?}: wants AXIS=VALUE, e.g. z=692", s))
    };
    let ax = ax.trim();
    let at: f64 = v
        .trim()
        .parse()
        .unwrap_or_else(|_| die(&format!("--cross {:?}: {:?} is not a number", s, v)));
    Plane { axis: ax.to_string(), col: t.need(ax), at }
}

/// Every crossing of `p`, interpolated, as (fraction-interpolated row, going up).
///
/// A sample sitting exactly ON the plane is one crossing at that sample, not
/// two — otherwise a run that touches the plane and comes back reports twice.
pub(crate) fn crossings(t: &Table, p: &Plane) -> Vec<(Vec<f64>, bool)> {
    let mut out = Vec::new();
    for w in t.rows.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let (u, v) = (a[p.col] - p.at, b[p.col] - p.at);
        if u.is_nan() || v.is_nan() {
            continue;
        }
        if u == 0.0 {
            out.push((a.clone(), v > 0.0));
            continue;
        }
        if (u < 0.0) != (v < 0.0) && v != 0.0 {
            let f = u / (u - v);
            let row: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + (y - x) * f).collect();
            out.push((row, v > u));
        }
    }
    out
}

/// The worst margin of `row` against `thresholds`, and which axis it was.
fn worst_margin(t: &Table, row: &[f64], thresholds: &[(String, usize, f64)]) -> (f64, String) {
    let _ = t;
    let mut worst = f64::INFINITY;
    let mut which = String::new();
    for (name, col, v) in thresholds {
        let d = row[*col] - v;
        if d < worst {
            worst = d;
            which = name.clone();
        }
    }
    (worst, which)
}

pub fn cmd(argv: &[String]) -> i32 {
    let a = cli::parse("tmtraj route", argv, &["summary", "pace"]);
    if a.positional.is_empty() {
        eprint!("{}", USAGE);
        return 2;
    }
    let near = a.one("near").map(|s| s.to_string());
    let top: usize = a.num("top", 5);
    let wheres: Vec<String> = a.many("where").iter().map(|s| s.to_string()).collect();
    let firstn = a.one("first").map(|s| s.parse::<usize>().unwrap_or(5));
    let lastn = a.one("last").map(|s| s.parse::<usize>().unwrap_or(5));
    let every = a.one("every").map(|s| s.parse::<usize>().unwrap_or(1));
    let want_cols = a.one("cols").map(|s| s.to_string());
    let summary = a.has("summary");
    let pace = a.has("pace");
    let segments = a.one("segments").map(|s| s.to_string());
    let cross_specs: Vec<String> = a.repeated("cross");
    let margin_spec = a.one("margin").map(|s| s.to_string());
    let a = a.finish(USAGE);

    if let Some(spec) = &margin_spec {
        return cmd_margin(&a.positional, &cross_specs, spec);
    }

    // Every positional, not just the first. The usage has always said
    // `CSV|GHOST...` and only `--margin` honoured it: every other query read
    // `positional[0]` and silently ignored the rest, so a shell glob over a
    // family of candidates reported ONE file's answer under a command line
    // that named forty. That is the failure mode this crate exists to avoid --
    // an answer that looks like the question you asked.
    let mut rc = 0;
    let mut empty = 0usize;
    let many = a.positional.len() > 1;
    for (i, p) in a.positional.iter().enumerate() {
        if many && i > 0 {
            println!();
        }
        match run_one(
            p, &near, top, &wheres, firstn, lastn, every, &want_cols, summary, pace, &segments, &cross_specs,
        ) {
            0 => {}
            1 => empty += 1,
            n => rc = n,
        }
    }
    // An empty selection is exit 1 -- and over a family it stays exit 1 only
    // when EVERY file was empty, because "none of them" and "some of them" are
    // different answers.
    if rc == 0 && empty == a.positional.len() {
        rc = 1;
    }
    rc
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    path: &str,
    near: &Option<String>,
    top: usize,
    wheres: &[String],
    firstn: Option<usize>,
    lastn: Option<usize>,
    every: Option<usize>,
    want_cols: &Option<String>,
    summary: bool,
    pace: bool,
    segments: &Option<String>,
    cross_specs: &[String],
) -> i32 {
    let path = path.to_string();
    let path = path.as_str();
    let t = load(path);
    let (cx, cy, cz) = (t.need("x"), t.need("y"), t.need("z"));
    let ct = t.col("time_ms").unwrap_or(0);

    let cols: Vec<usize> = match &want_cols {
        Some(s) => s.split(',').map(|n| t.need(n.trim())).collect(),
        None => {
            let mut v = vec![ct, cx, cy, cz];
            for n in ["speed_kmh", "vy", "vz"] {
                if let Some(i) = t.col(n) {
                    v.push(i);
                }
            }
            v
        }
    };
    let head: Vec<&str> = cols.iter().map(|&c| t.names[c].as_str()).collect();

    println!("{}  ({} rows, {} columns)", path, t.rows.len(), t.names.len());

    // The row set every aggregate below is computed over. `--summary` used to
    // ignore `--where` entirely and report the WHOLE file's path length under a
    // command line that named a time window -- an answer that looks like the
    // question you asked, which is the failure this crate exists to avoid.
    let preds: Vec<Pred> = wheres.iter().map(|s| parse_pred(&t, s)).collect();
    let sel_all: Vec<usize> = t
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| preds.iter().all(|p| holds(p, r)))
        .map(|(i, _)| i)
        .collect();

    if summary || (near.is_none() && wheres.is_empty() && firstn.is_none() && lastn.is_none()) {
        let mut len = 0.0;
        let (mut xmin, mut xmax) = (f64::MAX, f64::MIN);
        let (mut ymin, mut ymax) = (f64::MAX, f64::MIN);
        let (mut zmin, mut zmax) = (f64::MAX, f64::MIN);
        for (k, &i) in sel_all.iter().enumerate() {
            let r = &t.rows[i];
            if k > 0 {
                let p = &t.rows[sel_all[k - 1]];
                len += ((r[cx] - p[cx]).powi(2) + (r[cy] - p[cy]).powi(2) + (r[cz] - p[cz]).powi(2))
                    .sqrt();
            }
            xmin = xmin.min(r[cx]);
            xmax = xmax.max(r[cx]);
            ymin = ymin.min(r[cy]);
            ymax = ymax.max(r[cy]);
            zmin = zmin.min(r[cz]);
            zmax = zmax.max(r[cz]);
        }
        if !wheres.is_empty() {
            println!("  filter {} ({} of {} rows)", preds.iter().map(|p| p.text.clone()).collect::<Vec<_>>().join(" AND "), sel_all.len(), t.rows.len());
        }
        if let (Some(&f0), Some(&l0)) = (sel_all.first(), sel_all.last()) {
            let (f, l) = (&t.rows[f0], &t.rows[l0]);
            println!(
                "  span   {} .. {}",
                crate::fmt::secs(f[ct] as i64),
                crate::fmt::secs(l[ct] as i64)
            );
            println!("  first  ({:.2}, {:.2}, {:.2})", f[cx], f[cy], f[cz]);
            println!("  last   ({:.2}, {:.2}, {:.2})", l[cx], l[cy], l[cz]);
        }
        println!("  bbox   x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}", xmin, xmax, ymin, ymax, zmin, zmax);
        println!("  path   {:.1} m", len);
    }

    // `--pace`: how the selection's TIME is spent by speed band. On a map whose
    // whole difficulty is the slow parts, "165.868 s of the route is under
    // 20 km/h" is the load-bearing number and there was no tool that produced
    // it -- every arm that quoted it wrote a one-off.
    if pace {
        if let Some(cs) = t.col("speed_kmh") {
            let bands = [0.0f64, 5.0, 20.0, 40.0, 80.0, 120.0, f64::INFINITY];
            let names = ["<5", "5-20", "20-40", "40-80", "80-120", ">=120"];
            let mut secs_in = [0.0f64; 6];
            let mut total = 0.0f64;
            for (k, &i) in sel_all.iter().enumerate() {
                if k == 0 {
                    continue;
                }
                let dt = (t.rows[i][ct] - t.rows[sel_all[k - 1]][ct]) / 1000.0;
                if dt <= 0.0 || dt > 1.0 {
                    continue; // a splice junction is not a duration
                }
                let v = t.rows[i][cs];
                for b in 0..6 {
                    if v >= bands[b] && v < bands[b + 1] {
                        secs_in[b] += dt;
                        break;
                    }
                }
                total += dt;
            }
            println!("  pace   over {:.3} s of the selection", total);
            for b in 0..6 {
                println!(
                    "         {:>7} km/h  {:>9.3} s  {:>5.2} %",
                    names[b],
                    secs_in[b],
                    if total > 0.0 { 100.0 * secs_in[b] / total } else { 0.0 }
                );
            }
        }
    }

    // `--segments A,B,C`: the per-segment ledger, in one invocation. Every arm
    // that wanted "seconds, metres and pace between consecutive checkpoints"
    // has so far run the tool once per segment and reassembled the answer in a
    // shell pipeline, which is how `--summary` ignoring `--where` went unnoticed
    // for so long.
    if let Some(spec) = segments.as_deref() {
        let cs = t.col("speed_kmh");
        let mut bounds: Vec<f64> = vec![f64::MIN];
        for p in spec.split(',').filter(|s| !s.trim().is_empty()) {
            bounds.push(p.trim().parse::<f64>().unwrap_or_else(|_| die(&format!("--segments: {:?} is not a millisecond value", p))));
        }
        println!(
            "\n{:>3} {:>10} {:>10} {:>9} {:>9} {:>8} {:>8} {:>8}",
            "seg", "from", "to", "secs", "path_m", "km/h", "s<5", "s<20"
        );
        for w in bounds.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            let sel: Vec<usize> = sel_all.iter().copied().filter(|&i| t.rows[i][ct] > lo && t.rows[i][ct] <= hi).collect();
            let (mut len, mut u5, mut u20, mut tot) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (k, &i) in sel.iter().enumerate() {
                if k == 0 {
                    continue;
                }
                let p = &t.rows[sel[k - 1]];
                let r = &t.rows[i];
                len += ((r[cx] - p[cx]).powi(2) + (r[cy] - p[cy]).powi(2) + (r[cz] - p[cz]).powi(2)).sqrt();
                let dt = (r[ct] - p[ct]) / 1000.0;
                if dt <= 0.0 || dt > 1.0 {
                    continue;
                }
                tot += dt;
                if let Some(c) = cs {
                    if r[c] < 5.0 {
                        u5 += dt;
                    }
                    if r[c] < 20.0 {
                        u20 += dt;
                    }
                }
            }
            let span = if sel.len() > 1 { (t.rows[*sel.last().unwrap()][ct] - t.rows[sel[0]][ct]) / 1000.0 } else { 0.0 };
            println!(
                "{:>3} {:>10.3} {:>10.3} {:>9.3} {:>9.1} {:>8.1} {:>8.3} {:>8.3}",
                bounds.iter().position(|v| *v == lo).unwrap_or(0),
                if lo == f64::MIN { 0.0 } else { lo / 1000.0 },
                hi / 1000.0,
                span,
                len,
                if tot > 0.0 { 3.6 * len / tot } else { 0.0 },
                u5,
                u20
            );
        }
    }

    // --near: rank by distance to a point.
    if let Some(p) = &near {
        let nums: Vec<f64> = p.split(',').map(|s| s.trim().parse().unwrap_or(f64::NAN)).collect();
        if nums.len() != 3 || nums.iter().any(|v| v.is_nan()) {
            die("--near wants X,Y,Z");
        }
        let mut idx: Vec<(f64, usize)> = t
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let d = ((r[cx] - nums[0]).powi(2) + (r[cy] - nums[1]).powi(2) + (r[cz] - nums[2]).powi(2))
                    .sqrt();
                (d, i)
            })
            .collect();
        idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        println!("\nnearest to ({:.2}, {:.2}, {:.2}):", nums[0], nums[1], nums[2]);
        println!("{:>10} {}", "dist_m", head.iter().map(|h| format!("{:>10}", h)).collect::<Vec<_>>().join(" "));
        for (d, i) in idx.iter().take(top) {
            println!("{:>10.3} {}", d, fmt_row(&t, &t.rows[*i], &cols, ct));
        }
    }

    // --cross: every crossing of a plane, interpolated. Not a row filter --
    // see the module docs for why a nearby row is the wrong answer.
    for spec in cross_specs {
        let p = parse_plane(&t, spec);
        let cs = crossings(&t, &p);
        println!("\ncrossings of {}={}: {}", p.axis, p.at, cs.len());
        if !cs.is_empty() {
            println!(
                "{} {:>5}",
                head.iter().map(|h| format!("{:>10}", h)).collect::<Vec<_>>().join(" "),
                "dir"
            );
        }
        for (row, up) in &cs {
            println!("{} {:>5}", fmt_row(&t, row, &cols, ct), if *up { "+" } else { "-" });
        }
    }

    // --where: filter, then print.
    if !wheres.is_empty() || firstn.is_some() || lastn.is_some() {
        let preds: Vec<Pred> = wheres.iter().map(|s| parse_pred(&t, s)).collect();
        let sel: Vec<usize> = t
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| preds.iter().all(|p| holds(p, r)))
            .map(|(i, _)| i)
            .collect();
        let what = if preds.is_empty() {
            "all rows".to_string()
        } else {
            preds.iter().map(|p| p.text.clone()).collect::<Vec<_>>().join(" AND ")
        };
        println!("\nselection [{}]: {} of {} rows", what, sel.len(), t.rows.len());
        if sel.is_empty() {
            return 1;
        }
        println!("{}", head.iter().map(|h| format!("{:>10}", h)).collect::<Vec<_>>().join(" "));
        let mut shown: Vec<usize> = Vec::new();
        if let Some(n) = firstn {
            shown.extend(sel.iter().take(n).copied());
        }
        if let Some(n) = lastn {
            shown.extend(sel.iter().rev().take(n).rev().copied());
        }
        if let Some(n) = every {
            let n = n.max(1);
            shown.extend(sel.iter().step_by(n).copied());
        }
        if shown.is_empty() {
            shown.push(sel[0]);
            if sel.len() > 1 {
                shown.push(sel[sel.len() - 1]);
            }
        }
        shown.sort_unstable();
        shown.dedup();
        for i in shown {
            println!("{}", fmt_row(&t, &t.rows[i], &cols, ct));
        }
    }
    0
}

/// `--margin`: the frontier of a FAMILY at one plane.
///
/// One line per file, sorted best-first, printing where each run crossed and
/// the worst of its per-axis margins. A run that never reaches the plane sorts
/// last and says so — it is not a near miss and must not read as one.
fn cmd_margin(paths: &[String], cross: &[String], spec: &str) -> i32 {
    if cross.len() != 1 {
        die("--margin needs exactly one --cross PLANE to measure at");
    }
    println!(
        "{:>34}  {:>9} {:>8} {:>9}  {:>8}  {:>9} {:>5}",
        "file", "x", "y", "z", "km/h", "margin", "on"
    );
    let mut lines: Vec<(f64, String)> = Vec::new();
    for path in paths {
        let t = load(path);
        let p = parse_plane(&t, &cross[0]);
        let thresholds: Vec<(String, usize, f64)> = spec
            .split(',')
            .map(|part| {
                let Some((ax, v)) = part.split_once('=') else {
                    die(&format!("--margin {:?}: wants AXIS=VALUE", part))
                };
                let ax = ax.trim().to_string();
                let col = t.need(&ax);
                let val: f64 = v
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| die(&format!("--margin {:?}: not a number", part)));
                (ax, col, val)
            })
            .collect();
        let (cx, cy, cz) = (t.need("x"), t.need("y"), t.need("z"));
        let ckmh = t.col("speed_kmh");
        let name = path
            .rsplit('/')
            .next()
            .unwrap_or(path)
            .trim_end_matches(".csv")
            .trim_end_matches(".Ghost.Gbx")
            .trim_end_matches(".Replay.Gbx")
            .to_string();
        match crossings(&t, &p).first() {
            Some((row, _)) => {
                let (m, w) = worst_margin(&t, row, &thresholds);
                lines.push((
                    m,
                    format!(
                        "{:>34}  {:9.2} {:8.2} {:9.2}  {:8.2}  {:9.2} {:>5}",
                        name,
                        row[cx],
                        row[cy],
                        row[cz],
                        ckmh.map(|c| row[c]).unwrap_or(f64::NAN),
                        m,
                        w
                    ),
                ));
            }
            None => lines.push((
                f64::NEG_INFINITY,
                format!("{:>34}  never crosses {}={}", name, p.axis, p.at),
            )),
        }
    }
    lines.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, l) in lines {
        println!("{}", l);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(xs: &[(f64, f64)]) -> Table {
        Table {
            names: ["time_ms", "x", "y", "z"].iter().map(|s| s.to_string()).collect(),
            rows: xs
                .iter()
                .enumerate()
                .map(|(i, (x, y))| vec![(i as f64) * 50.0, *x, *y, 0.0])
                .collect(),
        }
    }

    /// The trap `--cross` exists for: at 50 ms and 45 m/s the nearest SAMPLE is
    /// up to 1.1 m from the plane, and the interpolated crossing is exact on a
    /// straight segment. A `--where` filter cannot answer this question.
    #[test]
    fn a_crossing_is_interpolated_not_snapped_to_a_sample() {
        let t = table(&[(790.0, 0.0), (795.0, 0.0)]);
        let p = Plane { axis: "x".into(), col: 1, at: 794.0 };
        let c = crossings(&t, &p);
        assert_eq!(c.len(), 1);
        assert!((c[0].0[1] - 794.0).abs() < 1e-9);
        assert!((c[0].0[0] - 40.0).abs() < 1e-9, "the TIME is interpolated too");
        assert!(c[0].1, "west to east is +");
    }

    /// Three crossings, three reports. 267460's pit loops cross the same plane
    /// on the way out and again on the way back, and a first-crossing reader
    /// silently measures the wrong lap.
    #[test]
    fn every_crossing_is_reported_with_its_direction() {
        let t = table(&[(790.0, 0.0), (800.0, 0.0), (780.0, 0.0), (810.0, 0.0)]);
        let p = Plane { axis: "x".into(), col: 1, at: 794.0 };
        let c = crossings(&t, &p);
        assert_eq!(c.len(), 3);
        assert_eq!(c.iter().map(|(_, u)| *u).collect::<Vec<_>>(), vec![true, false, true]);
    }

    #[test]
    fn a_sample_exactly_on_the_plane_is_one_crossing_not_two() {
        let t = table(&[(790.0, 0.0), (794.0, 0.0), (800.0, 0.0)]);
        let p = Plane { axis: "x".into(), col: 1, at: 794.0 };
        assert_eq!(crossings(&t, &p).len(), 1);
    }

    /// A minimum of margins cannot be paid in the wrong currency. On 267460 the
    /// car must arrive both east of x=922 and above y=110; a key that trades
    /// one for the other lets a run buy easting with height and fly into the
    /// wall lower down, which is what a distance-to-a-point key did there.
    #[test]
    fn the_margin_is_the_worst_axis_not_a_trade() {
        let t = table(&[(0.0, 0.0)]);
        let th = vec![("x".to_string(), 1, 922.0), ("y".to_string(), 2, 110.0)];
        let short_east = worst_margin(&t, &[0.0, 913.0, 114.0, 0.0], &th);
        let too_low = worst_margin(&t, &[0.0, 980.0, 64.0, 0.0], &th);
        assert!((short_east.0 - -9.0).abs() < 1e-9);
        assert_eq!(short_east.1, "x");
        assert!((too_low.0 - -46.0).abs() < 1e-9);
        assert_eq!(too_low.1, "y");
        assert!(too_low.0 < short_east.0, "58 m of easting must not pay for 46 m of height");
    }
}
