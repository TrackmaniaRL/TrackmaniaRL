//! `tmtraj adjudicate-batch DIR` — settle a whole set of `corpus dup`
//! UNRESOLVED pairs from a directory of swept traces.
//!
//! The traces are produced outside this tool (`fk trace` is a different
//! binary and needs the dedicated server), one per file per fork tick:
//!
//! ```text
//! <DIR>/<mapid>__<file-stem>__t<tick>.csv
//! ```
//!
//! For each file it uses **the trace that reproduces that file's own record
//! best**, which is the acceptance rule the 173636 false accusation forced —
//! see the module docs on `adjudicate`. Then, per pair, it asks the decisive
//! question: over the samples where the two RECORDS are bit-identical, how far
//! apart does the engine put the two cars?

use crate::adjudicate::{adjudicate_pair, best_trace, Verdict};
use std::collections::BTreeMap;

pub fn cmd(argv: &[String]) -> i32 {
    let pos: Vec<&String> = argv.iter().filter(|a| !a.starts_with("--")).collect();
    if pos.len() < 2 {
        eprintln!("usage: tmtraj adjudicate-batch TRACE_DIR PAIRS.tsv [--repo DIR]");
        eprintln!();
        eprintln!("  PAIRS.tsv: <mapdir>\\t<fileA>\\t<fileB> per line, as `corpus dup` prints.");
        eprintln!("  TRACE_DIR: <mapid>__<stem>__t<tick>.csv, from `fk trace`.");
        return 2;
    }
    let dir = pos[0].clone();
    let pairs_file = pos[1].clone();
    let repo = argv
        .iter()
        .position(|s| s == "--repo")
        .and_then(|i| argv.get(i + 1))
        .cloned()
        .unwrap_or_else(|| ".".to_string());

    let txt = match std::fs::read_to_string(&pairs_file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tmtraj adjudicate-batch: {pairs_file}: {e}");
            return 2;
        }
    };

    // Per-file verdicts are cached: a file appears in many pairs and its own
    // record check does not depend on the partner.
    let mut own: BTreeMap<String, (f64, f64, String, i64)> = BTreeMap::new();
    let mut rows: Vec<(String, String, String, Verdict, f64, usize)> = Vec::new();

    for line in txt.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 3 || f[0].is_empty() {
            continue;
        }
        let (mapdir, a, b) = (f[0].to_string(), f[1].to_string(), f[2].to_string());
        let mapid = mapdir.split('-').next().unwrap_or(&mapdir).to_string();
        let pa = format!("{repo}/{mapdir}/replays/{a}");
        let pb = format!("{repo}/{mapdir}/replays/{b}");

        let mut pick = |ghost: &str, stem: &str| -> Option<(String, f64, f64, i64)> {
            let key = format!("{mapid}/{stem}");
            if let Some((w, m, t, sh)) = own.get(&key) {
                return Some((t.clone(), *w, *m, *sh));
            }
            let got = best_trace(&dir, &mapid, stem, ghost)?;
            own.insert(key, (got.1, got.2, got.0.clone(), got.3));
            Some(got)
        };

        let sa = a.trim_end_matches(".Ghost.Gbx");
        let sb = b.trim_end_matches(".Ghost.Gbx");
        let (Some((ta, wa, ma, sha)), Some((tb, wb, mb, shb))) = (pick(&pa, sa), pick(&pb, sb)) else {
            rows.push((mapdir, a, b, Verdict::NoTrace, 0.0, 0));
            continue;
        };
        let _ = (ma, mb);
        match adjudicate_pair(&pa, &ta, sha, &pb, &tb, shb) {
            Some((v, sep, n)) => rows.push((mapdir, a, b, v, sep, n)),
            None => rows.push((mapdir, a, b, Verdict::NoTrace, 0.0, 0)),
        }
        let _ = (wa, wb);
    }

    println!("map\tfile_a\tfile_b\tverdict\tworst_sep_where_records_agree_m\tn_samples");
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (m, a, b, v, sep, n) in &rows {
        println!("{m}\t{a}\t{b}\t{}\t{sep:.6}\t{n}", v.word());
        *counts.entry(v.word()).or_default() += 1;
    }
    println!();
    println!("PER-FILE, does the file's record match the engine's run of its own tape:");
    let mut bad = 0usize;
    for (k, (w, md, t, sh)) in &own {
        let tag = if *w <= 0.01 { "OWN-RUN" } else { bad += 1; "MISMATCH" };
        println!("  {tag:9} {k}   worst {w:.4} m  median {md:.4} m   (best of the sweep: {}, shift {sh})", t.rsplit('/').next().unwrap_or(t));
    }
    println!();
    for (k, v) in &counts {
        println!("{v:4}  {k}");
    }
    if bad > 0 {
        println!();
        println!("{bad} file(s) whose record no fork point reproduced. That is EITHER a foreign");
        println!("record OR a map this locate cannot do -- separated by whether the other files");
        println!("on the same map came out OWN-RUN.");
    }
    0
}
