//! `tmtraj samplescan` -- the two questions you can ask a rebuilt record's
//! sample bytes without a game.
//!
//! The client dies importing some of our regenerated ghosts while the
//! container's own record imports every time, and every headless gate passes on
//! the file it dies on. Before spending a game launch on a bisect, two things
//! are answerable offline:
//!
//! 1. **Is any f32 in any sample non-finite?** A NaN or an infinity is the
//!    classic way a renderer dies where a headless oracle does not care -- the
//!    oracle reads the input TAPE and re-simulates; the client renders the
//!    RECORD. The scan reads every 4-byte window, aligned or not, because the
//!    116-byte sample is not a struct of aligned floats and a bad value does
//!    not care where we think the field boundary is.
//!
//! 2. **Does any byte take a value the donor's own record never takes?** The
//!    donor here is a record the client is known to import. A byte where ours
//!    goes somewhere the game's own writer never goes is a lead -- especially
//!    anything that could be an INDEX, because an index out of range
//!    dereferences something that is not there.
//!
//! Neither question is a proof: a value the donor never took can still be
//! perfectly legal, and a clean scan does not mean the file imports. They are
//! cheap, and they cost no game.

use gbx::record::{find_entrecord_blob, load_body, parse_record_data, Ent};

/// Every vehicle entity in the file, in file order. A server replay splits one
/// car across many entities; a scan that took only the longest would miss most
/// of the donor's own values.
fn vehicle_ents(path: &str) -> Result<(usize, Vec<Ent>), String> {
    let body = load_body(path)?;
    let (ver, blob) = find_entrecord_blob(&body)?;
    let rd = parse_record_data(&blob, ver)?;
    let ents: Vec<Ent> = rd
        .ents
        .iter()
        .filter(|e| e.sample_size >= 100 && !e.times.is_empty())
        .cloned()
        .collect();
    if ents.is_empty() {
        return Err(format!("{path}: no CSceneVehicleVis entity with samples"));
    }
    let ss = ents[0].sample_size;
    for e in &ents {
        if e.sample_size != ss {
            return Err(format!(
                "{path}: mixed sample sizes {} and {}",
                ss, e.sample_size
            ));
        }
    }
    Ok((ss, ents))
}

struct Scan {
    ss: usize,
    n: usize,
    /// value -> count, per byte offset
    vals: Vec<[u32; 256]>,
    /// per 4-byte window offset: (nan count, inf count, first sample index)
    bad: Vec<(usize, usize, Option<usize>)>,
}

fn scan(ss: usize, ents: &[Ent]) -> Scan {
    let mut s = Scan {
        ss,
        n: 0,
        vals: vec![[0u32; 256]; ss],
        bad: vec![(0, 0, None); ss.saturating_sub(3)],
    };
    let mut idx = 0usize;
    for e in ents {
        for k in 0..e.times.len() {
            let smp = &e.raw[k * ss..(k + 1) * ss];
            for (o, b) in smp.iter().enumerate() {
                s.vals[o][*b as usize] += 1;
            }
            for o in 0..ss - 3 {
                let f = f32::from_le_bytes(smp[o..o + 4].try_into().unwrap());
                if f.is_nan() {
                    s.bad[o].0 += 1;
                    s.bad[o].2.get_or_insert(idx);
                } else if f.is_infinite() {
                    s.bad[o].1 += 1;
                    s.bad[o].2.get_or_insert(idx);
                }
            }
            idx += 1;
            s.n += 1;
        }
    }
    s
}

impl Scan {
    fn distinct(&self, o: usize) -> Vec<u8> {
        (0..256)
            .filter(|v| self.vals[o][*v] > 0)
            .map(|v| v as u8)
            .collect()
    }
}

pub fn cmd(args: &[String]) -> i32 {
    let mut files: Vec<String> = Vec::new();
    let mut against: Option<String> = None;
    let mut only: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--against" => {
                against = Some(args[i + 1].clone());
                i += 2;
            }
            // `--bytes 19,20` reduces the report to a value census of those
            // byte positions, which is the form a corpus-wide control takes:
            // the same question asked of every file at once.
            "--bytes" => {
                only = args[i + 1]
                    .split(',')
                    .map(|s| s.trim().parse().expect("byte offset"))
                    .collect();
                i += 2;
            }
            a => {
                files.push(a.to_string());
                i += 1;
            }
        }
    }
    if files.is_empty() {
        println!("usage: tmtraj samplescan FILE... [--against DONOR]");
        return 2;
    }

    let donor = match &against {
        Some(p) => match vehicle_ents(p) {
            Ok((ss, e)) => {
                let sc = scan(ss, &e);
                println!(
                    "donor {p}\n  {} vehicle entities, {} samples x {} B",
                    e.len(),
                    sc.n,
                    ss
                );
                Some(sc)
            }
            Err(e) => {
                println!("{e}");
                return 2;
            }
        },
        None => None,
    };

    let mut rc = 0;
    for f in &files {
        let (ss, ents) = match vehicle_ents(f) {
            Ok(v) => v,
            Err(e) => {
                println!("{e}");
                rc = 2;
                continue;
            }
        };
        let s = scan(ss, &ents);
        if !only.is_empty() {
            let mut line = format!("{:<58} {:>5} smp", short(f), s.n);
            for &o in &only {
                let d = s.distinct(o);
                let desc = if d.len() == 1 {
                    format!("={}", d[0])
                } else {
                    format!("{}..{} ({} distinct)", d[0], d[d.len() - 1], d.len())
                };
                line.push_str(&format!("  b{o}:{desc}"));
            }
            println!("{line}");
            continue;
        }
        println!(
            "\n{f}\n  {} vehicle entities, {} samples x {} B",
            ents.len(),
            s.n,
            ss
        );

        // 1. non-finite f32, every window
        let mut any = false;
        for o in 0..s.bad.len() {
            let (nan, inf, first) = s.bad[o];
            if nan + inf == 0 {
                continue;
            }
            // A window that is non-finite in the DONOR too is not a float
            // there; say so rather than raising it.
            let donor_too = donor
                .as_ref()
                .map(|d| d.bad.get(o).map_or(false, |b| b.0 + b.1 > 0))
                .unwrap_or(false);
            any = true;
            println!(
                "  f32@{o:<3} NaN {nan} inf {inf}  first at sample {}{}",
                first.unwrap_or(0),
                if donor_too { "   (donor too: not a float here)" } else { "   <-- OURS ONLY" }
            );
            if !donor_too {
                rc = rc.max(1);
            }
        }
        if !any {
            println!("  no non-finite f32 at any of the {} windows", s.bad.len());
        }

        // 2. per-byte value sets against the donor
        if let Some(d) = &donor {
            let mut novel = 0;
            for o in 0..ss.min(d.ss) {
                let mine = s.distinct(o);
                let theirs: Vec<u8> = d.distinct(o);
                let outside: Vec<u8> = mine
                    .iter()
                    .copied()
                    .filter(|v| d.vals[o][*v as usize] == 0)
                    .collect();
                if outside.is_empty() {
                    continue;
                }
                novel += 1;
                let n_out: u32 = outside.iter().map(|v| s.vals[o][*v as usize]).sum();
                let show: Vec<String> = outside.iter().take(12).map(|v| v.to_string()).collect();
                println!(
                    "  byte {o:<3} {} value(s) the donor never writes: [{}{}]  in {n_out}/{} samples  (donor's set: {} distinct, {}..{})",
                    outside.len(),
                    show.join(","),
                    if outside.len() > 12 { ",..." } else { "" },
                    s.n,
                    theirs.len(),
                    theirs.first().copied().unwrap_or(0),
                    theirs.last().copied().unwrap_or(0),
                );
            }
            if novel == 0 {
                println!("  every byte stays inside the donor's own value set");
            }
        }
    }
    rc
}

/// `<map dir>/<file>` -- enough to tell two files apart in a corpus sweep
/// without a column of identical prefixes.
fn short(p: &str) -> String {
    let parts: Vec<&str> = p.trim_end_matches(".Ghost.Gbx").split('/').collect();
    let n = parts.len();
    if n >= 3 {
        format!("{}/{}", parts[n - 3], parts[n - 1])
    } else {
        parts[n - 1].to_string()
    }
}
