//! Population statistics over a decoded field of runs -- the analysis the
//! Python's `reports/analysis.txt` and `analysis2.txt` contain (those numbers
//! were produced by ad-hoc code that was not preserved; this reconstructs them
//! from the same inputs, and `tests/golden_stats.rs` checks every published
//! figure).

use crate::lines::{rms, Analysis};

pub struct Stats {
    pub pairs: Vec<f64>,
    pub mean: f64,
    pub sd: f64,
    pub largest_gap: (f64, f64, f64),
    /// per-run mean distance to all other runs
    pub centrality: Vec<(String, f64)>,
    pub cent_mean: f64,
    pub cent_sd: f64,
    /// (station index, sd of the lateral offset across the field)
    pub lateral_sd: Vec<f64>,
    pub most_separated: (String, String, f64),
}

pub fn stats(an: &Analysis) -> Stats {
    let n = an.names.len();
    let pairs = an.pair_distances();
    let mean = pairs.iter().sum::<f64>() / pairs.len() as f64;
    let sd = (pairs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / pairs.len() as f64).sqrt();
    let mut gap = (0.0, 0.0, 0.0);
    for w in pairs.windows(2) {
        if w[1] - w[0] > gap.0 {
            gap = (w[1] - w[0], w[0], w[1]);
        }
    }

    let centrality: Vec<(String, f64)> = (0..n)
        .map(|i| {
            let s: f64 = (0..n).filter(|&j| j != i).map(|j| an.d[i][j]).sum();
            (an.names[i].clone(), s / (n - 1) as f64)
        })
        .collect();
    let cvals: Vec<f64> = centrality.iter().map(|(_, v)| *v).collect();
    let cent_mean = cvals.iter().sum::<f64>() / n as f64;
    let cent_sd = (cvals.iter().map(|v| (v - cent_mean).powi(2)).sum::<f64>() / n as f64).sqrt();

    let mut lateral_sd = Vec::with_capacity(an.stations);
    for k in 0..an.stations {
        let col: Vec<f64> = an.names.iter().map(|nm| an.profiles[nm][k]).collect();
        let m = col.iter().sum::<f64>() / col.len() as f64;
        lateral_sd
            .push((col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / col.len() as f64).sqrt());
    }

    let mut ms = (String::new(), String::new(), 0.0f64);
    for i in 0..n {
        for j in (i + 1)..n {
            if an.d[i][j] > ms.2 {
                ms = (an.names[i].clone(), an.names[j].clone(), an.d[i][j]);
            }
        }
    }

    Stats {
        pairs,
        mean,
        sd,
        largest_gap: gap,
        centrality,
        cent_mean,
        cent_sd,
        lateral_sd,
        most_separated: ms,
    }
}

/// Sector times: `[S1..S4]` per run from its checkpoint splits.
pub fn sectors(cps: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(cps.len());
    let mut prev = 0;
    for &c in cps {
        out.push(c - prev);
        prev = c;
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // statistics.median: the mean of the two middle values for an even count
    if v.len() % 2 == 0 {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    } else {
        v[v.len() / 2]
    }
}

pub fn print_stats(an: &Analysis) {
    let s = stats(an);
    let n = an.names.len();

    println!("PAIRWISE RMS LATERAL SEPARATION -- histogram ({} pairs)", s.pairs.len());
    let hi = s.pairs.last().unwrap();
    let nbins = (hi / 0.5).ceil() as usize;
    let mut counts = vec![0usize; nbins];
    for p in &s.pairs {
        let b = ((p / 0.5) as usize).min(nbins - 1);
        counts[b] += 1;
    }
    let cmax = *counts.iter().max().unwrap() as f64;
    for (i, c) in counts.iter().enumerate() {
        println!(
            "  {:4.1}-{:4.1} m |{:<60}{:4}",
            i as f64 * 0.5,
            (i + 1) as f64 * 0.5,
            "#".repeat((*c as f64 / cmax * 60.0) as usize),
            c
        );
    }
    println!(
        "  mean {:.2}  sd {:.2}  min {:.2}  max {:.2}",
        s.mean,
        s.sd,
        s.pairs[0],
        s.pairs.last().unwrap()
    );
    println!(
        "  largest gap in the sorted pairwise distances: {:.3} m (between {:.2} and {:.2})",
        s.largest_gap.0, s.largest_gap.1, s.largest_gap.2
    );

    println!();
    println!("IS THE REFERENCE RUN GEOMETRICALLY SPECIAL?");
    println!(
        "  mean distance to all other runs: population mean {:.2} m, sd {:.2} m",
        s.cent_mean, s.cent_sd
    );
    let mut order = s.centrality.clone();
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let refname = an.ref_name().to_string();
    let refval = s.centrality[an.ref_idx].1;
    let rank = order.iter().position(|(nm, _)| *nm == refname).unwrap() + 1;
    println!(
        "  reference {}: {:.2} m  -> z = {:+.2}  (rank {} of {}, 1 = most central)",
        refname,
        refval,
        (refval - s.cent_mean) / s.cent_sd,
        rank,
        n
    );
    let fmt = |v: &[(String, f64)]| {
        v.iter()
            .map(|(nm, d)| format!("{} {:.2}", nm, d))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("  most central runs: {}", fmt(&order[..5.min(order.len())]));
    println!(
        "  most outlying runs: {}",
        fmt(&order[order.len().saturating_sub(5)..])
    );
    let mut nb: Vec<(usize, f64)> = (0..n)
        .filter(|&j| j != an.ref_idx)
        .map(|j| (j, an.d[an.ref_idx][j]))
        .collect();
    nb.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!(
        "  nearest neighbour of the reference: {} at {:.2} m",
        an.names[nb[0].0], nb[0].1
    );

    println!();
    println!(
        "WHERE ALONG THE LAP DOES THE FIELD DIVERGE? (sd of lateral offset across the {} runs)",
        n
    );
    let smax = s.lateral_sd.iter().cloned().fold(0.0f64, f64::max);
    println!(
        "  sd (m) vs distance along lap, 0 .. {:.0} m; max sd {:.2} m",
        an.ref_total, smax
    );
    for (k, i) in an.cp_stations() {
        println!(
            "  sd at {}: {:.2} m (station {} of {})",
            k, s.lateral_sd[i], i, an.stations
        );
    }
    println!(
        "  sd overall: min {:.2}  median {:.2}  max {:.2} m",
        s.lateral_sd.iter().cloned().fold(f64::INFINITY, f64::min),
        median(s.lateral_sd.clone()),
        smax
    );

    println!();
    println!(
        "MOST GEOMETRICALLY SEPARATED PAIR: {} <-> {} at RMS {:.2} m",
        s.most_separated.0, s.most_separated.1, s.most_separated.2
    );
    let a = &an.profiles[&s.most_separated.0];
    let b = &an.profiles[&s.most_separated.1];
    let diff: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let peak = diff
        .iter()
        .cloned()
        .fold(0.0f64, |acc, v| if v.abs() > acc.abs() { v } else { acc });
    let cp3 = an.cp_stations().iter().find(|(k, _)| *k == "CP3").unwrap().1;
    println!(
        "  their lateral difference: max |{:.2}| m, and {:.2} m RMS over the part of the lap \
         BEFORE CP3",
        peak,
        rms(&diff[..cp3])
    );
    println!(
        "  CP3 is station {} of {} ({:.0} m of {:.0})",
        cp3,
        an.stations,
        an.ref_stations[cp3].0,
        an.ref_total
    );
    // whole-field restriction to start..CP3
    let mut before: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d: Vec<f64> = an.profiles[&an.names[i]][..cp3]
                .iter()
                .zip(&an.profiles[&an.names[j]][..cp3])
                .map(|(x, y)| x - y)
                .collect();
            before.push(rms(&d));
        }
    }
    println!(
        "  RMS lateral separation using only start..CP3: mean {:.2}  max {:.2} m \
         (vs {:.2} / {:.2} for the whole lap)",
        before.iter().sum::<f64>() / before.len() as f64,
        before.iter().cloned().fold(0.0f64, f64::max),
        s.mean,
        s.pairs.last().unwrap()
    );

    // ---- sector times -------------------------------------------------
    let wr = &an.runs[an.ref_idx];
    if wr.checkpoints_ms.len() >= 2 {
        let wsec = sectors(&wr.checkpoints_ms);
        println!();
        println!("SECTOR TIMES (ms): where the field loses time relative to {}", wr.name);
        print!("  reference sectors:");
        for (i, v) in wsec.iter().enumerate() {
            print!("  S{} {}", i + 1, v);
        }
        println!("  (total {})", wr.time_ms);
        let mut per_sector: Vec<Vec<f64>> = vec![Vec::new(); wsec.len()];
        for r in &an.runs {
            if r.checkpoints_ms.len() != wsec.len() {
                continue;
            }
            for (i, v) in sectors(&r.checkpoints_ms).iter().enumerate() {
                per_sector[i].push((v - wsec[i]) as f64);
            }
        }
        let labels = ["S1 start->CP1", "S2 CP1->CP2", "S3 CP2->CP3", "S4 CP3->finish"];
        for (i, v) in per_sector.iter().enumerate() {
            let worst = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "  {:<16} field vs reference: median {:+.0} ms, worst {:+.0} ms",
                labels.get(i).copied().unwrap_or("S?"),
                median(v.clone()),
                worst
            );
        }
    }

    // ---- speed profile ------------------------------------------------
    println!();
    println!("SPEED: reference vs the field, at 12 stations round the lap (km/h)");
    let idx: Vec<usize> = (0..12).map(|k| k * (an.stations - 1) / 11).collect();
    print!("  {:<12}", "s (m)");
    for &i in &idx {
        print!(" {:>6.0}", an.ref_stations[i].0);
    }
    println!();
    print!("  {:<12}", "reference");
    for &i in &idx {
        print!(" {:>6.1}", an.stations_by_run[an.ref_name()][i].4);
    }
    println!();
    let med: Vec<f64> = idx
        .iter()
        .map(|&i| {
            median(
                an.names
                    .iter()
                    .filter(|nm| *nm != an.ref_name())
                    .map(|nm| an.stations_by_run[nm][i].4)
                    .collect(),
            )
        })
        .collect();
    print!("  {:<12}", "field med");
    for v in &med {
        print!(" {:>6.1}", v);
    }
    println!();
    print!("  {:<12}", "ref - med");
    for (k, &i) in idx.iter().enumerate() {
        print!(" {:>+6.1}", an.stations_by_run[an.ref_name()][i].4 - med[k]);
    }
    println!();
}
