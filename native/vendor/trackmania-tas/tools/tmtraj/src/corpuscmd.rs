//! `tmtraj corpus` — the same question asked of every published file at once.
//!
//! This replaces seven shell scripts. They were the right *questions*; being
//! shell was what made them fragile, because every one of them piped a tool's
//! stdout through awk and discarded its stderr — and every tool in this
//! project signals "I could not measure that" on stderr. Four separate bugs in
//! this project have the same shape: **an instrument fell silent and the
//! pipeline read the silence as a clean result.**
//!
//! | was | is | why |
//! |---|---|---|
//! | `ghost-splice-audit.sh` | `corpus splice` | ported |
//! | `sep-truncation-scan.sh` | *(gone)* | it hunted the silent-comparison bug in our own instrument. The instrument no longer has it: `tmtraj diff` and this scan report coverage per pair and refuse a verdict on zero rows. A scan for a bug you have made impossible is dead work. |
//! | `record-stops-short-scan.sh` | `corpus span` | ported |
//! | `jump-recheck-speedometer.sh` | *(gone)* | it re-graded C3's distance rule against the car's own speedometer, one work queue at a time. The speedometer rule is now C3 itself (`tmtraj check`), so there is nothing to re-grade. |
//! | `skincheck.sh` | `corpus qc` (and `ghost identity`) | ported |
//! | `ship-clip.sh`, `splitscreen.sh` | `clip ship`, `clip split` | ported to Rust |
//!
//! ## The corpus layout this walks
//!
//! `<root>/<mapid>-<slug>/replays/*.Ghost.Gbx`, which is how this repo stores
//! its published files. A name matching `wr|human|rank|author` (case
//! insensitively) is a REFERENCE — a recording made by somebody else — and
//! everything else is ours.
//!
//! ## Who is a trustworthy reference
//!
//! Only a human recording. Same-pipeline siblings re-converge too (203072's
//! KEYBOARD and our own TAS re-converge after 9 m apart), so a sibling
//! comparison is excluded rather than reported as evidence. **A map with no
//! human recording is UNTESTED, and untested is not clean.**

use crate::cli;
use crate::fmt::{delta, secs};
use gbx::record::{self, Decoded, Sample};
use std::collections::{BTreeMap, BTreeSet, HashMap};

const USAGE: &str = "\
usage: tmtraj corpus <scan> --root DIR [flags]

  splice   is a published file's telemetry another driver's?
           --minsep M (5.0)  --ident-pct N (90)  --extra MAPID=FILE (repeatable)
  span     does the record cover the run? a record that stops before the line
           cannot show the finish, and one that runs past it shows a stranger
           --tol MS (60)
  qc       pre-render QC, the declared-time census, and the car skin
  claims   does a map's page agree with the files in its own directory?
           a name that disagrees with its own header, a linked file that is
           not there, a file the page never names, a headline no file backs
  shipped  does the ROOT README's `this TAS` column name a run that is
           published? held-but-not-shipped is a different status from beaten
  bytes    which of the 116 sample bytes ever vary, across the whole corpus
  dup      two published files of one map carrying the SAME recorded motion
  audit    the splice test with the references named in a refs.tsv
           --refs FILE

Every scan prints one row per file and a summary. Coverage is always stated:
a verdict from zero compared samples is reported as UNTESTED, never as clean.
";

// ---------------------------------------------------------------------------

pub fn cmd(argv: &[String]) -> i32 {
    // `--help` BEFORE the subcommand check: this command reads the filesystem
    // to decide what to do, so asking it for help used to walk the corpus and
    // fail with "no <mapid>-<slug>/replays/*.Ghost.Gbx" instead of printing
    // usage. Help must never depend on the state of the working directory.
    if argv.iter().any(|s| s == "--help" || s == "-h") {
        print!("{}", USAGE);
        return 0;
    }
    let Some(sub) = argv.first().map(|s| s.to_string()) else {
        eprint!("{}", USAGE);
        return 2;
    };
    let a = cli::parse("tmtraj corpus", &argv[1..], &[]);
    let root = a.one("root").unwrap_or(".").to_string();
    let minsep: f64 = a.num("minsep", 5.0);
    let ident_pct: f64 = a.num("ident-pct", 90.0);
    let tol: i64 = a.num("tol", 60);
    let extra = a.many("extra");
    let refs_file = a.one("refs").map(|s| s.to_string());
    let a = a.finish(USAGE);
    let _ = &a;

    let maps = walk(&root);
    if maps.is_empty() {
        eprintln!(
            "tmtraj corpus: no <mapid>-<slug>/replays/*.Ghost.Gbx under {}",
            root
        );
        return 2;
    }
    match sub.as_str() {
        "splice" => splice(&maps, minsep, ident_pct, &extra),
        "span" => span(&maps, tol),
        "qc" => qc(&maps),
        // Does a map's page agree with the files in its own directory? The
        // failure mode this catches is a headline no file supports -- see
        // claimscmd.rs.
        "claims" => crate::claimscmd::claims(&maps, &root),
        // One level up from `claims`: does the ROOT README's `this TAS` column
        // name a run that is actually in that map's replays/? "Held but not
        // shipped" is a different status from "beaten" -- see shippedcmd.rs.
        "shipped" => crate::shippedcmd::shipped(&maps, &root),
        "bytes" => bytes(&maps),
        // Cross-file: within each map, two published ghosts that carry the SAME
        // recorded motion although their tapes diverged long before. A class no
        // per-file gate can see, because each file is individually coherent --
        // and 17 of the repo's 29 multi-ghost maps once had it, which is why
        // two of our own tapes rendered as a single car.
        "dup" => {
            crate::intgcmd::cmd_dup(&["--corpus".to_string(), root.clone()]);
            0
        }
        // The same splice test as `corpus splice`, but with the references
        // named in a refs.tsv rather than inferred from filenames. Use it when
        // the human recordings for a map are not in the repo.
        "audit" => {
            let mut v = vec!["--corpus".to_string(), root.clone()];
            if let Some(f) = refs_file {
                v.push("--refs".into());
                v.push(f);
            }
            crate::intgcmd::cmd_audit(&v);
            0
        }
        other => {
            eprintln!("tmtraj corpus: unknown scan {:?}\n", other);
            eprint!("{}", USAGE);
            2
        }
    }
}

// ---------------------------------------------------------------------------
// the corpus
// ---------------------------------------------------------------------------

pub struct MapDir {
    pub id: String,
    pub ours: Vec<String>,
    pub refs: Vec<String>,
}

/// A name declaring itself human-derived is SUPPOSED to match a human
/// recording; high identity there is the label being truthful.
fn is_derived_as_labelled(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["authorcut", "humancut", "authormin", "author_lap", "author_at"]
        .iter()
        .any(|k| n.contains(k))
}

fn is_reference(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["wr", "human", "rank", "author"].iter().any(|k| n.contains(k))
}

fn walk(root: &str) -> Vec<MapDir> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else { return out };
    let mut dirs: Vec<std::path::PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    dirs.sort();
    for d in dirs {
        let replays = d.join("replays");
        if !replays.is_dir() {
            continue;
        }
        let name = d.file_name().unwrap_or_default().to_string_lossy().to_string();
        let id = name.split('-').next().unwrap_or(&name).to_string();
        let Ok(files) = std::fs::read_dir(&replays) else { continue };
        let mut all: Vec<String> = files
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_string_lossy().to_string())
            .filter(|p| p.ends_with(".Ghost.Gbx") || p.ends_with(".Replay.Gbx"))
            .collect();
        all.sort();
        let (refs, ours): (Vec<String>, Vec<String>) =
            all.into_iter().partition(|p| is_reference(base(p)));
        out.push(MapDir { id, ours, refs });
    }
    out
}

pub(crate) fn base(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

fn dist(p: &Sample, q: &Sample) -> f64 {
    (((p.x - q.x) as f64).powi(2) + ((p.y - q.y) as f64).powi(2) + ((p.z - q.z) as f64).powi(2)).sqrt()
}

// ---------------------------------------------------------------------------
// splice
// ---------------------------------------------------------------------------

struct Pairing {
    rows: usize,
    min_samples: usize,
    identical: usize,
    max_sep: f64,
    reconverged: usize,
}

/// Compare on the instants the two files SHARE, and report how many that was.
///
/// The predecessor walked the two arrays index by index and stopped at the
/// first mismatched time key, printing to stderr. Sample times are SESSION
/// times, so two recordings from different sessions disagree at index 0: all
/// ten of 228607's files produced ZERO compared rows against `AUTHOR_LAP_20258`
/// and the audit recorded ten CLEAN verdicts. `rows` is on every line here for
/// that reason, and a `rows == 0` pairing can never reach a verdict.
fn compare(a: &Decoded, b: &Decoded, minsep: f64) -> Pairing {
    let ib: HashMap<i32, usize> =
        b.samples.iter().enumerate().map(|(i, s)| (s.time_ms, i)).collect();
    let mut p = Pairing {
        rows: 0,
        min_samples: a.samples.len().min(b.samples.len()),
        identical: 0,
        max_sep: 0.0,
        reconverged: 0,
    };
    let (mut ever_apart, mut was_ident) = (false, false);
    for s in &a.samples {
        let Some(&j) = ib.get(&s.time_ms) else { continue };
        p.rows += 1;
        let d = dist(s, &b.samples[j]);
        if d == 0.0 {
            p.identical += 1;
            if ever_apart && !was_ident {
                p.reconverged += 1;
            }
            was_ident = true;
        } else {
            was_ident = false;
            if d > minsep {
                ever_apart = true;
            }
            p.max_sep = p.max_sep.max(d);
        }
    }
    p
}

fn splice(maps: &[MapDir], minsep: f64, ident_pct: f64, extra: &[String]) -> i32 {
    // --extra MAPID=/path/to/human.Ghost.Gbx, for references we hold but the
    // repo does not ship.
    let mut extra_by_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for e in extra {
        if let Some((k, v)) = e.split_once('=') {
            extra_by_map.entry(k.to_string()).or_default().push(v.to_string());
        }
    }
    println!(
        "map\tfile\treference\trows\tmin_samples\tident\tident_pct\tmax_sep_m\treconverged\tverdict"
    );
    let mut refused = 0usize;
    let mut untested = 0usize;
    let mut clean = 0usize;
    for m in maps {
        let mut refs = m.refs.clone();
        refs.extend(extra_by_map.get(&m.id).cloned().unwrap_or_default());
        for f in &m.ours {
            let Ok(a) = record::decode_ghost(f) else {
                println!("{}\t{}\t-\t-\t-\t-\t-\t-\t-\tDECODE-FAIL", m.id, base(f));
                untested += 1;
                continue;
            };
            if refs.is_empty() {
                // UNTESTED. It does not mean clean.
                println!("{}\t{}\t-\t0\t-\t-\t-\t-\t-\tNO-HUMAN-REFERENCE", m.id, base(f));
                untested += 1;
                continue;
            }
            // keep the most incriminating reference: re-convergence first,
            // then wholesale identity.
            let mut best: Option<(&String, Pairing)> = None;
            for r in &refs {
                let Ok(b) = record::decode_ghost(r) else { continue };
                let p = compare(&a, &b, minsep);
                let better = match &best {
                    None => true,
                    Some((_, q)) => {
                        p.reconverged > q.reconverged
                            || (p.reconverged == q.reconverged
                                && pct(p.identical, p.rows) > pct(q.identical, q.rows))
                    }
                };
                if better {
                    best = Some((r, p));
                }
            }
            let Some((r, p)) = best else {
                println!("{}\t{}\t-\t0\t-\t-\t-\t-\t-\tREFERENCES-UNREADABLE", m.id, base(f));
                untested += 1;
                continue;
            };
            let ip = pct(p.identical, p.rows);
            let verdict = if p.rows == 0 {
                untested += 1;
                "UNTESTED-NO-SHARED-INSTANT"
            } else if is_derived_as_labelled(base(f)) && ip >= ident_pct {
                clean += 1;
                "DERIVED-AS-LABELLED"
            } else if p.reconverged > 0 {
                refused += 1;
                "CONTAMINATED-SPLICE"
            } else if ip >= ident_pct {
                refused += 1;
                "CONTAMINATED-IS-THE-HUMAN-RECORD"
            } else {
                clean += 1;
                "CLEAN"
            };
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.0}%\t{:.2}\t{}\t{}",
                m.id,
                base(f),
                base(r),
                p.rows,
                p.min_samples,
                p.identical,
                ip,
                p.max_sep,
                p.reconverged,
                verdict
            );
        }
    }
    println!(
        "\n{} refused, {} clean against the references held, {} UNTESTED (which is not clean)",
        refused, clean, untested
    );
    println!(
        "A shared PREFIX proves nothing -- the simulation is deterministic and our own sibling\n\
         tapes are 67 % bit-identical on one 203072 pair. Only RE-CONVERGENCE (identical, then\n\
         more than {:.1} m apart, then exactly identical again) can only be a splice.",
        minsep
    );
    i32::from(refused > 0) * 2
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}

// ---------------------------------------------------------------------------
// span
// ---------------------------------------------------------------------------

/// Does the record cover the run?
///
/// Two opposite defects, and only one of them was ever looked for. The tail
/// past the line is the familiar shape: a transplanted ghost inherits its
/// CARRIER's trajectory, and what survives past our finish is a stranger's car
/// snapping backwards through the air the instant it crosses. Every downloaded
/// human recording measured stops AT its own finish — 0 samples after, 5 of 5.
///
/// The opposite is that the record stops SHORT. 126859's published files end
/// 0.095 s before their declared race time, so the crossing is simply not in
/// the record and no clip can show it. Nothing else looks for that.
/// AND A THIRD CASE, which cost an afternoon before anything looked for it:
/// the samples stop where they should and the RECORD NODE's own declared span
/// does not. 286279's `BEST_218812` reads `span 0.000 .. 441.000` for a 218.812
/// run -- the span of Bald_tm's 441.002 recording, the container it was
/// regenerated in -- with the car's last sample correctly at 217.95. Every
/// check above passes it.
///
/// The bar for this one is PROPORTIONATE, not flat: a record that runs a
/// fraction of a second past the finish is the game's own tail (58 files here,
/// none of them a defect), while a carrier's span is 87 % to 10 500 % long.
/// There is nothing in between.
///
/// It is not cosmetic. The MediaTracker renders a clip as long as its longest
/// block, so filming that file produced 441 s of video for a 218 s run, and
/// when the camera's target entity ended at 218 s the camera drifted to the top
/// of the map and stayed there for the second half. Both were reported as
/// rendering faults. The same inheritance keeps the carrier's non-vehicle
/// entities (0x2D001000, 13 bytes a sample) at their own full length, which is
/// what holds the scene open.
fn span(maps: &[MapDir], tol: i64) -> i32 {
    println!("map\tfile\trace\tlast_sample\tdelta\tspan_end\toutliving\tnote");
    let (mut short, mut long, mut ok) = (0usize, 0usize, 0usize);
    let mut outlive = 0usize;
    for m in maps {
        for f in m.ours.iter().chain(m.refs.iter()) {
            let Ok(d) = record::decode_ghost(f) else {
                println!("{}\t{}\t-\t-\t-\tDECODE-FAIL", m.id, base(f));
                continue;
            };
            let (Some(race), Some(last)) = (d.race_time_ms, d.samples.last().map(|s| s.time_ms))
            else {
                println!("{}\t{}\t-\t-\t-\tNO-DECLARED-TIME", m.id, base(f));
                continue;
            };
            let dt = last as i64 - race as i64;
            // Does anything in this file outlive the car? The record's own
            // declared end, and every entity's own last sample.
            let over: Vec<String> = d
                .ents
                .iter()
                .filter(|e| e.t_last.unwrap_or(0) as i64 > last as i64 + tol)
                .map(|e| format!("0x{:08X}@{}", e.class_id.unwrap_or(0), secs(e.t_last.unwrap_or(0) as i64)))
                .collect();
            // HOW FAR PAST THE CAR, AS A FRACTION OF THE RUN -- not a flat bar.
            //
            // A record legitimately runs a little past the finish: the game
            // keeps sampling for a moment and 58 of this corpus's files sit
            // within 2 s, none of which is a defect. A carrier's span is a
            // different population entirely: 87 % to 10 500 % long. There is
            // nothing in the gap, which is what makes it safe to separate them,
            // and it is the project's own GROWTH-NOT-MAGNITUDE rule -- a flat
            // bar in seconds would refuse a 5-second run's honest tail and pass
            // a 13-minute run carrying somebody else's.
            let scene_end = (d.end_ms as i64).max(
                d.ents.iter().filter_map(|e| e.t_last).max().unwrap_or(0) as i64,
            );
            let past = scene_end - last as i64;
            let frac = past as f64 / (last as f64).max(1.0);
            let outliving = if past <= 2000 || frac < 0.10 {
                "-".to_string()
            } else {
                outlive += 1;
                let what = if over.is_empty() { "the record's span".to_string() } else { over.join(",") };
                format!("{} (+{:.0} %)", what, frac * 100.0)
            };
            let note = if dt < -tol {
                short += 1;
                "RECORD STOPS SHORT -- the finish is not in it"
            } else if dt > tol {
                long += 1;
                "tail past the finish -- a transplant carries the carrier's car"
            } else if !outliving.starts_with('-') {
                "THE SCENE OUTLIVES THE CAR -- the render runs long and the camera loses its target"
            } else {
                ok += 1;
                continue;
            };
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                m.id,
                base(f),
                secs(race as i64),
                secs(last as i64),
                delta(dt),
                secs(d.end_ms as i64),
                outliving,
                note
            );
        }
    }
    println!(
        "\n{} stop short of the line, {} run past it, {} within {} ms.\n\
         {} have something that OUTLIVES THE CAR -- the record's own declared span, or an \
         entity the container donor left in. Those film long and lose the camera; \
         `ghost record rebuild IN OUT --span MS` is the repair.",
        short, long, ok, tol, outlive
    );
    i32::from(short + long + outlive > 0) * 2
}

// ---------------------------------------------------------------------------
// qc
// ---------------------------------------------------------------------------

/// The only car skin this project may ship.
const OUR_SKIN: &str = "Skins\\Models\\CarSport\\TAS.zip";

/// Pre-render QC plus the two identity facts a positional check cannot see.
///
/// A tape that validates as a time is not a tape that draws a car: validation
/// reads the input chunk, the video reads the telemetry. And a file can be
/// clean on login, on account id and on declared time, and still put a
/// stranger's paint on the car — 276874's two WATCH tapes read login `TAS`,
/// carried no account id, imported as `Ghost:TAS`, and carried
/// `Skins\Models\CarSport\frckitbot (1)(1)_756eeda4-....zip` and its Nadeo
/// storage URL. Every other identity field is metadata; this one is on screen.
fn qc(maps: &[MapDir]) -> i32 {
    println!("map\tfile\tverdict\trace\tnsamp\tpath_m\tvmax_kmh\tstart_xyz\tskin");
    let mut bad = 0usize;
    for m in maps {
        for f in m.ours.iter().chain(m.refs.iter()) {
            let skin = skin_of(f);
            let skin_bad = skin.iter().any(|s| s != OUR_SKIN);
            let skin_txt = if skin.is_empty() {
                "-".to_string()
            } else {
                skin.join(" ")
            };
            match record::decode_ghost(f) {
                Err(e) => {
                    println!("{}\t{}\tDECODE-FAIL\t-\t-\t-\t-\t-\t{}\t{}", m.id, base(f), skin_txt, e);
                    bad += 1;
                }
                Ok(d) => {
                    let s = &d.samples;
                    let nan = s.iter().any(|p| {
                        !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() || !p.speed_kmh.is_finite()
                    });
                    let mut len = 0.0f64;
                    let mut vmax = 0.0f64;
                    for i in 1..s.len() {
                        let d3 = dist(&s[i], &s[i - 1]);
                        if d3.is_finite() {
                            len += d3;
                        }
                        if s[i].speed_kmh.is_finite() {
                            vmax = vmax.max(s[i].speed_kmh as f64);
                        }
                    }
                    let f0 = s.first();
                    // A run that begins at the world origin begins where no
                    // start block is.
                    let origin =
                        f0.map_or(false, |p| p.x.abs() < 2.0 && p.y.abs() < 2.0 && p.z.abs() < 2.0);
                    let verdict = if nan {
                        "NAN"
                    } else if s.len() < 10 {
                        "SHORT"
                    } else if len < 0.01 {
                        "STATIC"
                    } else if origin {
                        "ORIGIN"
                    } else if len < 5.0 {
                        "CREEP"
                    } else if skin_bad {
                        "FOREIGN-SKIN"
                    } else {
                        "OK"
                    };
                    if verdict != "OK" {
                        bad += 1;
                    }
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t({:.1},{:.1},{:.1})\t{}",
                        m.id,
                        base(f),
                        verdict,
                        d.race_time_ms.map_or("-".into(), |v| secs(v as i64)),
                        s.len(),
                        len,
                        vmax,
                        f0.map_or(0.0, |p| p.x),
                        f0.map_or(0.0, |p| p.y),
                        f0.map_or(0.0, |p| p.z),
                        skin_txt
                    );
                }
            }
        }
    }
    println!("\n{} files need attention", bad);
    i32::from(bad > 0) * 2
}

/// Skin paths and storage-object URLs, read out of the raw bytes: unlike the
/// nickname, the skin path is stored as plain text in the container.
fn skin_of(path: &str) -> Vec<String> {
    let Ok(data) = std::fs::read(path) else { return Vec::new() };
    let mut out = BTreeSet::new();
    let mut cur = Vec::new();
    for &b in &data {
        if (0x20..0x7f).contains(&b) {
            cur.push(b);
        } else {
            if cur.len() >= 6 {
                let s = String::from_utf8_lossy(&cur).to_string();
                let l = s.to_ascii_lowercase();
                if l.contains("skins\\models") || l.contains("skins/models") || l.contains("storageobjects") {
                    out.insert(s);
                }
            }
            cur.clear();
        }
    }
    out.into_iter().collect()
}

// ---------------------------------------------------------------------------
// bytes — what we can and cannot see about a run
// ---------------------------------------------------------------------------

/// A census of which of the 116 bytes of a `CSceneVehicleVis` sample ever
/// change, over the whole corpus.
///
/// WHY. `tmtraj fields` names 48 quantities read out of **56** of the 116
/// bytes; the decoder never touches bytes 103-115 at all, and `ghost regen`
/// reports that **91 of the 116 are still the carrier's** after it writes the
/// 22 transform bytes and the three input-echo bytes from the tape. "Unnamed"
/// has been a shrug for as long as this decoder has existed.
///
/// This turns it into an enumerated set: for every byte offset, how many
/// distinct values it takes, how often it changes between consecutive samples,
/// and whether it is CONSTANT across the corpus (nothing to learn), CONSTANT
/// PER FILE (an identity or a mode, not a per-tick quantity) or LIVE (a signal
/// nobody has named). A byte that is live in 158 files and named by nobody is
/// a concrete thing to go and find in engine memory, which is the difference
/// between "the harness cannot see it" and "I have not found where it lives".
fn bytes(maps: &[MapDir]) -> i32 {
    const N: usize = 116;
    let mut distinct: Vec<BTreeSet<u8>> = vec![BTreeSet::new(); N];
    let mut changes = vec![0usize; N];
    let mut steps = 0usize;
    let mut const_per_file = vec![0usize; N];
    let mut files = 0usize;
    for m in maps {
        for f in m.ours.iter().chain(m.refs.iter()) {
            let Ok(d) = record::decode_ghost(f) else { continue };
            if d.sample_size < N {
                continue;
            }
            files += 1;
            let mut prev: Option<&[u8]> = None;
            let mut here: Vec<BTreeSet<u8>> = vec![BTreeSet::new(); N];
            for s in d.raw_samples() {
                for b in 0..N {
                    distinct[b].insert(s[b]);
                    here[b].insert(s[b]);
                }
                if let Some(p) = prev {
                    steps += 1;
                    for b in 0..N {
                        if p[b] != s[b] {
                            changes[b] += 1;
                        }
                    }
                }
                prev = Some(s);
            }
            for b in 0..N {
                if here[b].len() <= 1 {
                    const_per_file[b] += 1;
                }
            }
        }
    }
    if files == 0 {
        eprintln!("tmtraj corpus bytes: nothing decoded");
        return 3;
    }
    let named = named_bytes();
    println!(
        "{} files, {} sample steps\n",
        files, steps
    );
    println!("{:>4} {:>9} {:>9} {:>12} {:>10}  {}", "byte", "distinct", "change%", "const_files", "class", "named as");
    for b in 0..N {
        let ch = 100.0 * changes[b] as f64 / steps.max(1) as f64;
        let class = if distinct[b].len() <= 1 {
            "CONSTANT"
        } else if const_per_file[b] == files {
            "PER-FILE"
        } else {
            "LIVE"
        };
        println!(
            "{:>4} {:>9} {:>8.2}% {:>9}/{} {:>10}  {}",
            b,
            distinct[b].len(),
            ch,
            const_per_file[b],
            files,
            class,
            named.get(&b).copied().unwrap_or("-")
        );
    }
    carry_test(maps, &named);
    let unnamed_live: Vec<usize> = (0..N)
        .filter(|b| !named.contains_key(b) && distinct[*b].len() > 1 && const_per_file[*b] < files)
        .collect();
    println!(
        "\n{} of {} bytes are LIVE and unnamed: {:?}",
        unnamed_live.len(),
        N,
        unnamed_live
    );
    println!(
        "These are not a harness limit. The engine computes every one of them and they are in\n\
         its memory; nothing here has read them yet. That is a task, not a conclusion -- and\n\
         this list is where to start, because a byte that is constant across the corpus has\n\
         nothing to find and a byte that is constant per file is an identity, not a signal."
    );
    0
}

/// Which byte each documented field is read from, taken from the field table
/// itself rather than a second hand-written list — so a new field cannot be
/// named in one place and missing from the other.
fn named_bytes() -> BTreeMap<usize, &'static str> {
    let mut m = BTreeMap::new();
    for f in record::FIELD_CONFIDENCE {
        for b in parse_byte_refs(f.note) {
            m.entry(b).or_insert(f.name);
        }
    }
    // The transform block is documented as a unit ("read_transform") rather
    // than byte by byte: 47..68 inclusive is position, orientation, speed and
    // the velocity direction.
    for b in 47..=68 {
        m.entry(b).or_insert("transform (pos/quat/speed/vel-dir)");
    }
    m
}

/// Pull the byte offsets out of a field note: `byte 14: ...`, `bytes 6,7: ...`,
/// `u16 at byte 2: ...`, `byte91 - 1`. Only digits that FOLLOW the word `byte`
/// count, and only the comma-separated run immediately after it — the note for
/// `gear` reads "byte only ever takes values 1+4k (5,9,13,17,21 => 1..5)", and a
/// looser rule claimed bytes 5, 9, 13 and 17 for it.
fn parse_byte_refs(note: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let lower = note.to_ascii_lowercase();
    let b = lower.as_bytes();
    let mut i = 0usize;
    while let Some(k) = lower[i..].find("byte") {
        let mut j = i + k + 4;
        if b.get(j) == Some(&b's') {
            j += 1;
        }
        while b.get(j) == Some(&b' ') {
            j += 1;
        }
        // a comma-separated run of numbers, and nothing else
        loop {
            let s = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j == s {
                break;
            }
            if let Ok(v) = lower[s..j].parse::<usize>() {
                if v < 116 {
                    out.push(v);
                }
            }
            if b.get(j) == Some(&b',') {
                j += 1;
            } else {
                break;
            }
        }
        i = i + k + 4;
    }
    out
}

/// Are we reading only HALF of a quantity?
///
/// `rpm_raw` is documented as "byte 5, 0..255, monotone with engine load; the
/// absolute RPM scale factor is NOT known". Byte 4 sits beside it, changes on
/// 84.6 % of steps, and nothing names it. If bytes 4 and 5 are one
/// little-endian u16, then the "unknown scale factor" is not unknown at all —
/// we are reading the high byte of a 16-bit number and calling the quantisation
/// a mystery.
///
/// THE TEST, and it needs no engine: a real little-endian u16 carries. When the
/// low byte wraps (|delta| > 128 with the sign of a wrap), the high byte must
/// step by exactly +1 or -1 in the matching direction. For two unrelated bytes
/// that co-occurrence is chance.
///
/// POSITIVE CONTROLS IN THE SAME TABLE, so the reading is not taken on trust:
/// bytes (6,7), (8,9), (10,11), (12,13) are the four documented wheel
/// (rotation, rotation-count) pairs and must score high; the byte pairs inside
/// the f32 position field at 47..58 are structured and must too. A pair that
/// scores like those is the same kind of thing. A pair that scores like two
/// CONSTANT bytes is not.
fn carry_test(maps: &[MapDir], named: &BTreeMap<usize, &'static str>) {
    const N: usize = 116;
    let mut wraps = vec![0usize; N];
    let mut carried = vec![0usize; N];
    for m in maps {
        for f in m.ours.iter().chain(m.refs.iter()) {
            let Ok(d) = record::decode_ghost(f) else { continue };
            if d.sample_size < N {
                continue;
            }
            let mut prev: Option<Vec<u8>> = None;
            for s in d.raw_samples() {
                if let Some(p) = &prev {
                    for b in 0..N - 1 {
                        let dlo = s[b] as i32 - p[b] as i32;
                        if dlo.abs() <= 128 {
                            continue;
                        }
                        wraps[b] += 1;
                        // low wrapped upward past 255 -> high must be +1
                        let want = if dlo < 0 { 1i32 } else { -1i32 };
                        if s[b + 1] as i32 - p[b + 1] as i32 == want {
                            carried[b] += 1;
                        }
                    }
                }
                prev = Some(s.to_vec());
            }
        }
    }
    println!("\nIS A BYTE THE LOW HALF OF A u16? (low byte wraps -> high byte must step +-1)");
    println!("{:>7} {:>8} {:>9}  {:>9}  {}", "pair", "wraps", "carried", "carry%", "named as");
    let mut rows: Vec<(usize, usize, usize, f64)> = (0..N - 1)
        .filter(|b| wraps[*b] >= 20)
        .map(|b| (b, wraps[b], carried[b], 100.0 * carried[b] as f64 / wraps[b] as f64))
        .collect();
    rows.sort_by(|a, b| b.3.total_cmp(&a.3));
    for (b, w, c, p) in &rows {
        let lo = named.get(b).copied().unwrap_or("-");
        let hi = named.get(&(b + 1)).copied().unwrap_or("-");
        println!(
            "{:>3},{:<3} {:>8} {:>9}  {:>8.1}%  {} / {}",
            b,
            b + 1,
            w,
            c,
            p,
            lo,
            hi
        );
    }
    // Read the controls out of this table before reading any candidate off it.
    let get = |b: usize| rows.iter().find(|r| r.0 == b).map(|r| (r.1, r.3));
    println!("\ncontrols in this table:");
    for (b, what) in [
        (2usize, "side_speed, DOCUMENTED as a u16 at byte 2 -- byte 3 is its high half"),
        (49, "inside the f32 x at 47..50 -- a float carries by construction"),
        (53, "inside the f32 y at 51..54"),
        (4, "MY OWN HYPOTHESIS: that rpm_raw at byte 5 is the high half of a u16 rpm"),
    ] {
        match get(b) {
            Some((n, p)) => {
                println!("  {:>3},{:<3} {:>6.1}% over {} wraps   {}", b, b + 1, p, n, what)
            }
            None => println!("  {:>3},{:<3}  too few wraps to measure   {}", b, b + 1, what),
        }
    }
    println!(
        "  The last row is why the controls are printed: bytes 4 and 5 carry on ~10 % of\n  \
         wraps, so they are NOT one u16 and the 'unknown rpm scale factor' is not a missing\n  \
         low byte. A plausible mechanism is not a measurement."
    );
    let candidates: Vec<(usize, usize, f64)> = rows
        .iter()
        .filter(|(b, w, _, p)| {
            *p >= 85.0 && *w >= 200 && (!named.contains_key(b) || !named.contains_key(&(b + 1)))
        })
        .map(|(b, w, _, p)| (*b, *w, *p))
        .collect();
    if candidates.is_empty() {
        println!("\nNo unnamed pair carries like a u16 at this corpus size.");
    } else {
        println!("\nCARRIES LIKE A u16 AND IS NOT FULLY NAMED:");
        for (b, w, p) in &candidates {
            println!(
                "  bytes {},{}   {:.1} % of {} wraps carry -- one 16-bit quantity of which we\n  \
                 decode at most one byte. Go and find it in engine memory.",
                b,
                b + 1,
                p,
                w
            );
        }
    }
}

/// `is_reference`, for the `claims` scan: a file whose name marks it as
/// somebody else's recording is not one of ours and must not set the
/// directory's "our best" figure.
pub(crate) fn is_reference_pub(name: &str) -> bool {
    is_reference(name)
}
