//! `tmtraj` — read-only analysis of a Trackmania 2020 run.
//!
//! ## The shape of the tool
//!
//! Every command answers one of six questions, and the top level says which:
//!
//! | group | the question |
//! |---|---|
//! | `show` / `export` / `fields` | what does this file SAY the car did |
//! | `diff` / `spawn` | are these two recordings the same run |
//! | `check` / `gate` / `manifest` | is this a coherent run of a car, and is it ours |
//! | `motion` / `wheels` / `facing` | what does the TRAJECTORY say, as against what a flag claims |
//! | `corpus` | the same questions asked of every published file at once |
//! | `lines` | racing-line clustering across a population of runs |
//!
//! ## What is deliberately NOT here
//!
//! **`tmtraj` never writes a ghost.** Everything that mutates a file lives in
//! `ghost`, which owns the container, the tape and the identity, and runs the
//! oracle: `tail fix` is `ghost trim --auto`, `setdecl` is `ghost declare`,
//! `anon` / `hdr setlogin` / `body setlogin` are `ghost identity set`,
//! `recspan` is `ghost trim`, `rec roundtrip` is `ghost selftest`'s codec
//! identity check, `rectime` is `ghost record shift`. That boundary is the
//! point: one tool reads and one tool writes, so a read can never be the thing
//! that corrupted the file.
//!
//! The file format itself is in `gbx` — one implementation, shared by `gbx`,
//! `tmtraj`, `ghost` and `tmsite`.

use gbx::record::{self, Decoded};
use crate::cli;
use crate::fmt::secs;
use crate::lines::{Analysis, Metric, Sort};
use crate::serial;
use crate::{lines, selftest, stats};

const USAGE: &str = "\
tmtraj — read-only analysis of a TM2020 run. Times print as seconds (36.049).

WHAT DOES THE FILE SAY
  show    FILE...                    span, checkpoints, entities, first samples
  export  FILE   [--csv F] [--json F] [--full-json F] [--head N]
  csvdiff A.csv B.csv [--tol-ms N]  two trajectory CSVs, on the instants they share
  export  --dir D... [--out-csv D] [--out-json D] [--jobs N]
  fields                             every decoded field, with its confidence tier
  bytes   FILE... --bytes N,N        the RAW value of named sample bytes, per sample

ARE THESE THE SAME RUN
  diff    A B  [--lag] [--bytes] [--near --control C1 --control C2] [--csv F]
  spawn   FILE... --ref R            same start position AND attitude
  inputs  FILE [--events] [--csv F]  the steer/gas/brake the record carries, exactly

IS IT A COHERENT RUN, AND IS IT OURS
  check   FILE... [--race S] [--g G]        C1-C13; exit 0 clean / 1 warn / 2 REFUSED
  gate    FILE... --race S --refs F --mapid ID [...]   the publish gate
  manifest new|verify|show ...

WHAT DOES THE TRAJECTORY SAY (not the flag)
  motion  FILE [--race S] [--g G]    ballistic / supported / unknown, and the flag beside it
  samplescan FILE... [--against DONOR] [--bytes N,N]
                                     non-finite f32s in every 4-byte window, and
                                     byte values the donor never writes
  provenance FILE --carrier C        which of the 116 bytes are ours and which
                                     are still the container donor's
  impacts FILE... [--bar KMH] [--race S] [--against OTHER]
                                     one-sample speed losses: what the car hit,
                                     and whether a second engine reading agrees
  wheels  FILE [--race S]            wheel radius, and whether the wheel bytes are alive
  facing  FILE... [--ref R] [--route CSV] [--shift-ms N]
  route   CSV|GHOST... [--summary] [--near X,Y,Z --top N] [--where 'y>130'] [--first N]
          [--cross z=692] -- EVERY crossing of a plane, interpolated, not a row
          [--margin x=922,y=110] -- over all the files: the frontier at that plane
  splits  REF OTHER... [--axis x] --from A --to B --step S [--stations LIST] [--csv F]
          two runs differenced AS A FUNCTION OF DISTANCE: crossing time at each
          station, the lead there, and what each segment alone was worth

EVERY PUBLISHED FILE AT ONCE
  corpus  splice --root R [--refs F]   telemetry that is another driver's
  corpus  span   --root R              a record that stops short of the line, or runs past it
  corpus  qc     --root R              pre-render QC, the declared-time census, the car skin
  corpus  bytes  --root R              which of the 116 sample bytes ever vary
  corpus  dup    --root R              two files of one map with the same recorded motion
  corpus  audit  --root R --refs F     the splice test against a named reference list

A POPULATION OF RUNS
  geom    path|selfcut|near ...        the SHAPE of a run: length, cuts, line-vs-line
  lines   report|matrix|stats|demo --dir D [--stations N] [--eps E,...]
          [--metric projection|station|dtw] [--ref NAME] [--sort time|name]

  selftest [--strict]

Race times are printed as seconds with a decimal. A tick index is a count and
stays an integer.
";

/// The command line, as a library entry point.
///
/// It lives in the lib and not in `main.rs` for one reason: everything a
/// `[[bin]]` reaches has to be `pub`, and `pub` switches off the dead-code
/// warning. That is how a crate accretes sixty commands and nobody notices
/// forty of them stopped being called. With the dispatcher inside the lib,
/// every module below can be `pub(crate)` and `cargo build` reports anything
/// nothing reaches.
pub fn run() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        print!("{}", USAGE);
        std::process::exit(2);
    }
    let rest = &argv[1..];
    // `<subcommand> --help` for EVERY subcommand. Most route through
    // `cli::finish`, which prints that command's OWN usage; but the ones that
    // read a positional argument first fail in their own way before help is
    // considered -- "unknown flag --help", "unknown manifest subcommand",
    // even "--help: No such file or directory". Six of twenty behaved so.
    //
    // Dispatch normally FIRST, so a command with specific usage still prints
    // it, and only fall back to the top-level text when the command answered
    // with a usage error (exit 2) to a lone `--help`. That way the guard
    // rescues the six without shadowing the fourteen.
    let lone_help = rest.len() == 1 && (rest[0] == "--help" || rest[0] == "-h");
    let code = match argv[0].as_str() {
        "show" => cmd_show(rest),
        "export" => cmd_export(rest),
        "csvdiff" => crate::csvdiff::cmd(rest),
        "fields" => {
            record::print_field_confidence();
            0
        }
        "bytes" => cmd_bytes(rest),
        "diff" => crate::diffcmd::cmd(rest),
        "spawn" => crate::diffcmd::cmd_spawn(rest),
        "inputs" => crate::diffcmd::cmd_inputs(rest),
        "check" => {
            crate::checkcmd::cmd(rest);
            0
        }
        "gate" => {
            crate::intgcmd::cmd(rest);
            0
        }
        "manifest" => {
            crate::manifest::cmd(rest);
            0
        }
        "impacts" => crate::impactcmd::cmd(rest),
        "provenance" => crate::provcmd::cmd(rest),
        "samplescan" => crate::scancmd::cmd(rest),
        "airborne" => crate::aircmd::cmd(rest),
        "motion" => crate::whlcmd::cmd_motion(rest),
        "wheels" => crate::whlcmd::cmd_wheels(rest),
        "facing" => crate::facingcmd::cmd(rest),
        "route" => crate::routecmd::cmd(rest),
        "splits" => crate::splitscmd::cmd(rest),
        "blockdiff" => crate::blockdiffcmd::cmd(rest),
        "corpus" => crate::corpuscmd::cmd(rest),
        // Settle a corpus-dup UNRESOLVED pair against the engine.
        "adjudicate" => crate::adjudicate::cmd(rest),
        "adjudicate-batch" => crate::adjudicate_batch::cmd(rest),
        // How many ticks differ between two tapes in a race-time window.
        // UNTRUNCATED, which is the point: `ghost tape diff` prints at most 80
        // rows and then stops, so its output reads as "the differences end at
        // tick 79". That cost this audit a wrong reading -- 203330's pair
        // looked like it had zero differences after the countdown and actually
        // has 1041, 227 of them inside the stretch where the two files'
        // positions are bit-identical.
        "tapediff" => {
            if rest.len() < 2 {
                eprintln!("usage: tmtraj tapediff A.Ghost.Gbx B.Ghost.Gbx [--from SECONDS] [--to SECONDS]");
                2
            } else {
            let secs_arg = |flag: &str, dflt: f64| -> f64 {
                rest.iter()
                    .position(|s| s == flag)
                    .and_then(|i| rest.get(i + 1))
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(dflt)
            };
            let from = (secs_arg("--from", 0.0) * 1000.0) as i64;
            let to = (secs_arg("--to", 1.0e9) * 1000.0) as i64;
            match crate::intgcmd::tape_diffs_in_window(&rest[0], &rest[1], from, to) {
                Err(e) => {
                    eprintln!("tmtraj tapediff: {e}");
                    2
                }
                Ok(n) => {
                    println!(
                        "{} ticks differ between race {:.3} and {:.3}",
                        n,
                        from as f64 / 1000.0,
                        if to > 1_000_000 { -1.0 } else { to as f64 / 1000.0 }
                    );
                    0
                }
            }
            }
        }
        "lines" => cmd_lines(rest),
        "geom" => crate::geomcmd::cmd(rest),
        "selftest" => {
            let r = selftest::selftest(true);
            if !r.skipped.is_empty() {
                // A skip is not a pass. Say what was not run, on stderr, and
                // let --strict make it a failure: this crate once printed
                // "SELFTEST: ALL PASS (0 checks, 0 failed)" and exited 0 on a
                // box with none of its fixtures.
                eprintln!("SKIPPED (not run, not passed): {}", r.skipped.join(", "));
                if rest.iter().any(|a| a == "--strict") {
                    eprintln!("--strict: a skip is a failure");
                    std::process::exit(1);
                }
            }
            i32::from(!r.ok)
        }
        "-h" | "--help" | "help" => {
            print!("{}", USAGE);
            0
        }
        other => {
            eprintln!("tmtraj: unknown command {:?}\n", other);
            print!("{}", USAGE);
            2
        }
    };
    // The fallback described above: a lone `--help` that the subcommand
    // answered with a usage error gets the top-level text and exit 0. A
    // command with its own help has already printed it and returned 0, so it
    // never reaches here.
    if lone_help && code == 2 {
        print!("{}", USAGE);
        std::process::exit(0);
    }
    std::process::exit(code);
}

// ---------------------------------------------------------------------------
// show / export
// ---------------------------------------------------------------------------

const SHOW_USAGE: &str = "usage: tmtraj show GHOST... [--head N]\n";

fn cmd_bytes(argv: &[String]) -> i32 {
    const U: &str = "tmtraj bytes FILE... --bytes N,N,...  [--head N]\n\
                     \n\
                     The RAW value of named sample bytes, per sample, for every file named --\n\
                     the reader half of a per-byte bisect. `ghost record channels` moves a byte\n\
                     from one file into another; this says what the byte held and whether it\n\
                     varies at all. A channel a regeneration writes as ONE CONSTANT for a whole\n\
                     run is the shape that found 276877's camera defect.\n";
    let a = cli::parse("tmtraj bytes", argv, &[]);
    let head: usize = a.num("head", 0);
    let want: Vec<usize> = a.many("bytes").iter().filter_map(|s| s.parse().ok()).collect();
    let a = a.finish(U);
    if a.positional.is_empty() || want.is_empty() {
        eprint!("{U}");
        return 2;
    }
    for path in &a.positional {
        let dec = match record::decode_ghost(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("FAIL {path}: {e}");
                return 1;
            }
        };
        let ss = dec.sample_size;
        println!("{path}  {} samples x {ss} B", dec.samples.len());
        print!("{:>9}", "time");
        for b in &want {
            print!("{:>7}", format!("b{b}"));
        }
        println!();
        let n = dec.samples.len();
        let show = if head == 0 { n } else { head.min(n) };
        for (i, s) in dec.samples.iter().enumerate().take(show) {
            print!("{:>9}", secs(s.time_ms as i64));
            for b in &want {
                match dec.raw.get(i * ss + *b) {
                    Some(v) => print!("{v:>7}"),
                    None => print!("{:>7}", "-"),
                }
            }
            println!();
        }
        for b in &want {
            let mut vals: Vec<u8> =
                (0..n).filter_map(|i| dec.raw.get(i * ss + *b).copied()).collect();
            vals.sort_unstable();
            vals.dedup();
            println!(
                "  byte {b}: {} distinct value(s){}",
                vals.len(),
                if vals.len() == 1 { format!(" -- CONSTANT {}", vals[0]) } else { String::new() }
            );
        }
    }
    0
}

fn cmd_show(argv: &[String]) -> i32 {
    let a = cli::parse("tmtraj show", argv, &[]);
    let head: usize = a.num("head", 10);
    let a = a.finish(SHOW_USAGE);
    if a.positional.is_empty() {
        eprint!("{}", SHOW_USAGE);
        return 2;
    }
    let mut worst = 0;
    for path in &a.positional {
        let dec = match record::decode_ghost(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("FAIL {}: {}", path, e);
                worst = worst.max(1);
                continue;
            }
        };
        print_header(path, &dec);
        println!(
            "{:>9} {:>10} {:>8} {:>10} {:>9} {:>6} {:>5}",
            "t", "x", "y", "z", "km/h", "gear", "rpm"
        );
        for s in dec.samples.iter().take(head) {
            println!(
                "{:>9} {:>10.3} {:>8.3} {:>10.3} {:>9.2} {:>6.1} {:>5}",
                secs(s.time_ms as i64),
                s.x,
                s.y,
                s.z,
                s.speed_kmh,
                s.gear,
                s.rpm_raw
            );
        }
    }
    worst
}

fn print_header(path: &str, dec: &Decoded) {
    // The sample count and the declared checkpoints are the two-second tell for
    // a synthesised tape carrying its TEMPLATE's telemetry: a poisoned file had
    // 281 samples and declared 14.018 where the clean regeneration of the same
    // run had 280 and 13.984. Both are on the first line for that reason.
    println!("{}", path);
    println!(
        "  version {}  samples {}  period {} ms  sample_size {}  span {} .. {}",
        dec.version,
        dec.samples.len(),
        dec.sample_period_ms.map_or("None".into(), |v| v.to_string()),
        dec.sample_size,
        secs(dec.start_ms as i64),
        secs(dec.end_ms as i64),
    );
    println!(
        "  declared {}   checkpoints {}",
        dec.race_time_ms.map_or("-".to_string(), |v| secs(v as i64)),
        if dec.checkpoints_ms.is_empty() {
            "-".to_string()
        } else {
            dec.checkpoints_ms.iter().map(|c| secs(*c as i64)).collect::<Vec<_>>().join(" ")
        }
    );
    let ents: Vec<String> = dec
        .ents
        .iter()
        .map(|e| format!("0x{:08X}x{}@{}B", e.class_id.unwrap_or(0), e.n_samples, e.sample_size))
        .collect();
    println!("  entities {}", ents.join(" "));
    if dec.bytes_consumed != dec.bytes_total {
        // The structural control on the whole decode: the grammar must consume
        // the blob to its exact last byte. Anything else means a field width in
        // the grammar is wrong, and every number above is suspect.
        println!(
            "  !! record blob NOT consumed exactly: {} of {} bytes",
            dec.bytes_consumed, dec.bytes_total
        );
    }
}

const EXPORT_USAGE: &str = "\
usage: tmtraj export GHOST [--csv F] [--json F] [--full-json F]
       tmtraj export --dir D... [--out-csv D] [--out-json D] [--jobs N]
";

fn cmd_export(argv: &[String]) -> i32 {
    let a = cli::parse("tmtraj export", argv, &[]);
    let dirs = a.many("dir");
    if !dirs.is_empty() {
        return export_tree(&a, &dirs);
    }
    // Read the output flags BEFORE finish(): finish() rejects any flag that
    // has not been asked for yet, so asking afterwards made every --csv /
    // --json / --full-json an "unknown flag" and this command unusable.
    let outs: Vec<(Option<String>, fn(&Decoded) -> String)> = vec![
        (a.one("csv").map(|s| s.to_string()), serial::csv_string as fn(&Decoded) -> String),
        (a.one("json").map(|s| s.to_string()), serial::path_json_string),
        (a.one("full-json").map(|s| s.to_string()), serial::full_json_string),
    ];    let a = a.finish(EXPORT_USAGE);
    let Some(path) = a.positional.first() else {
        eprint!("{}", EXPORT_USAGE);
        return 2;
    };
    let dec = match record::decode_ghost(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL {}: {}", path, e);
            return 1;
        }
    };
    let mut wrote = false;
    for (f, text) in &outs {
        if let Some(f) = f {
            std::fs::write(f, text(&dec)).expect("write");            println!("wrote {}", f);
            wrote = true;
        }
    }
    if !wrote {
        eprint!("{}", EXPORT_USAGE);
        return 2;
    }
    0
}
fn export_tree(a: &cli::Args, dirs: &[String]) -> i32 {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let mut ghosts: Vec<String> = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d) else {
            eprintln!("cannot read dir {}", d);
            continue;
        };
        let mut here: Vec<String> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_string_lossy().to_string())
            .filter(|p| p.ends_with(".Ghost.Gbx"))
            .collect();
        here.sort();
        ghosts.extend(here);
    }
    let out_json = a.one("out-json").map(|s| s.to_string());
    let out_csv = a.one("out-csv").map(|s| s.to_string());
    for d in [&out_json, &out_csv].into_iter().flatten() {
        std::fs::create_dir_all(d).expect("mkdir");
    }
    let jobs: usize = a.num("jobs", std::thread::available_parallelism().map_or(8, |v| v.get()));

    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<(String, String, usize, String, usize, bool, Vec<i32>)>> =
        Mutex::new(Vec::new());
    let fails: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..jobs.max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= ghosts.len() {
                    break;
                }
                let p = &ghosts[i];
                if std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) == 0 {
                    println!("SKIP empty {}", p);
                    continue;
                }
                let nm = record::name_for(p);
                match record::decode_ghost(p) {
                    Err(e) => fails.lock().unwrap().push(format!("FAIL {}: {}", nm, e)),
                    Ok(dec) => {
                        if let Some(d) = &out_json {
                            std::fs::write(
                                format!("{}/{}.json", d, nm),
                                serial::path_json_string(&dec),
                            )
                            .expect("write json");
                        }
                        if let Some(d) = &out_csv {
                            std::fs::write(format!("{}/{}.csv", d, nm), serial::csv_string(&dec))
                                .expect("write csv");
                        }
                        rows.lock().unwrap().push((
                            nm,
                            dec.race_time_ms.map_or("-".into(), |v| secs(v as i64)),
                            dec.samples.len(),
                            dec.sample_period_ms
                                .map_or("None".into(), |v| v.to_string()),
                            dec.sample_size,
                            dec.bytes_consumed == dec.bytes_total,
                            dec.checkpoints_ms.clone(),
                        ));
                    }
                }
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for f in fails.into_inner().unwrap() {
        println!("{}", f);
    }
    println!(
        "{:<22} {:>7} {:>6} {:>5} {:>5} {:>5} {}",
        "run", "time", "nsamp", "per", "ssz", "exact", "checkpoints (s)"
    );
    for r in &rows {
        println!(
            "{:<22} {:>7} {:>6} {:>5} {:>5} {:>5} {}",
            r.0,
            r.1,
            r.2,
            r.3,
            r.4,
            if r.5 { "True" } else { "False" },
            r.6.iter().map(|c| secs(*c as i64)).collect::<Vec<_>>().join(" ")
        );
    }
    println!(
        "decoded {} ghosts{}",
        rows.len(),
        match &out_json {
            Some(d) => format!(" -> {}", d),
            None => String::new(),
        }
    );
    0
}


// ---------------------------------------------------------------------------
// lines — racing-line analysis over a population of runs
// ---------------------------------------------------------------------------

const LINES_USAGE: &str = "\
usage: tmtraj lines report|matrix|stats|demo --dir D
         [--stations N] [--eps E,...] [--ref NAME]
         [--metric projection|station|dtw] [--sort time|name]
         [--out FILE] [--no-plots]

  report   full: per-run lateral summary, distance distribution, clusters,
           a seed per line at each eps, ASCII plots
  matrix   the pairwise distance matrix and its distribution, nothing else
  stats    population analysis: separation histogram, centrality of --ref,
           lateral spread along the lap, most separated pair, sector times
  demo     the two synthetic lines, as a smoke test (~0.8 m within a line,
           ~11 m between) -- needs no data
";

fn cmd_lines(argv: &[String]) -> i32 {
    let Some(sub) = argv.first().map(|s| s.to_string()) else {
        eprint!("{}", LINES_USAGE);
        return 2;
    };
    let a = cli::parse("tmtraj lines", &argv[1..], &["no-plots"]);
    let stations: usize = a.num("stations", 400);
    let metric = a.enumerated(
        "metric",
        &[("projection", Metric::Projection), ("station", Metric::Station), ("dtw", Metric::Dtw)],
        Metric::Projection,
    );
    let sort = a.enumerated("sort", &[("time", Sort::Time), ("name", Sort::Name)], Sort::Time);
    let eps: Vec<f64> = {
        let v: Vec<f64> = a.many("eps").iter().filter_map(|s| s.parse().ok()).collect();
        if v.is_empty() { vec![1.0, 2.0, 5.0] } else { v }
    };
    let dir = a.one("dir").unwrap_or("/tmp/entrec/paths").to_string();
    let refname = a.one("ref").map(|s| s.to_string());
    let out = a.one("out").map(|s| s.to_string());
    let plots = !a.has("no-plots");
    let a = a.finish(LINES_USAGE);
    let _ = &a;

    if sub == "demo" {
        let an = lines::analyse(lines::demo(), metric, stations, None, Sort::Name);
        report(&an, &eps, true, false, out.as_deref());
        return 0;
    }
    let runs = match lines::load_dir(&dir) {
        Ok(r) if !r.is_empty() => r,
        Ok(_) => {
            eprintln!("tmtraj lines: no trajectories in {}", dir);
            return 2;
        }
        Err(e) => {
            eprintln!("tmtraj lines: {}", e);
            return 2;
        }
    };
    match sub.as_str() {
        "stats" => {
            let an = lines::analyse(runs, metric, stations, refname.as_deref(), sort);
            stats::print_stats(&an);
            0
        }
        "report" | "matrix" => {
            let an = lines::analyse(runs, metric, stations, refname.as_deref(), sort);
            report(&an, &eps, sub == "report", plots, out.as_deref());
            0
        }
        other => {
            eprintln!("tmtraj lines: unknown subcommand {:?}\n", other);
            eprint!("{}", LINES_USAGE);
            2
        }
    }
}
fn report(an: &Analysis, eps_list: &[f64], full: bool, plots: bool, out: Option<&str>) {
    let n = an.names.len();
    println!(
        "{} runs, {} .. {} ms; reference = {}   [metric: {}]",
        n,
        an.runs.iter().map(|r| r.time_ms).min().unwrap(),
        an.runs.iter().map(|r| r.time_ms).max().unwrap(),
        an.ref_name(),
        an.metric.name()
    );
    println!(
        "reference path length {:.1} m over {} stations ({:.2} m spacing)",
        an.ref_total,
        an.stations,
        an.ref_total / (an.stations - 1) as f64
    );

    let cps = an.cp_stations();
    let cp = |k: &str| cps.iter().find(|(n, _)| *n == k).unwrap().1;

    if full {
        println!();
        println!(
            "PER-RUN SUMMARY (lateral offset vs {}, + = left of the reference heading)",
            an.ref_name()
        );
        println!(
            "{:<18} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "run", "time", "pathlen", "rms_lat", "max_lat", "CP1", "CP2", "CP3", "vmax"
        );
        for r in &an.runs {
            let p = &an.profiles[&r.name];
            let max_lat = p
                .iter()
                .cloned()
                .fold(0.0f64, |acc, v| if v.abs() > acc.abs() { v } else { acc });
            println!(
                "{:<18} {:>7} {:>8.1} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.1}",
                r.name,
                r.time_ms,
                an.totals[&r.name],
                lines::rms(p),
                max_lat,
                p[cp("CP1")],
                p[cp("CP2")],
                p[cp("CP3")],
                r.v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
        }
    }

    println!();
    println!("PAIRWISE SEPARATION (m) -- distribution");
    let allp = an.pair_distances();
    if !allp.is_empty() {
        println!(
            "  min {:.2}  p25 {:.2}  median {:.2}  p75 {:.2}  max {:.2}  ({} pairs)",
            allp[0],
            allp[allp.len() / 4],
            allp[allp.len() / 2],
            allp[3 * allp.len() / 4],
            allp[allp.len() - 1],
            allp.len()
        );
        let ri = an.ref_idx;
        let mut vs: Vec<f64> = (0..n).filter(|&j| j != ri).map(|j| an.d[ri][j]).collect();
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  vs {} specifically: min {:.2}  median {:.2}  max {:.2}",
            an.ref_name(),
            vs[0],
            vs[(n - 1) / 2],
            vs[vs.len() - 1]
        );
        let mean = allp.iter().sum::<f64>() / allp.len() as f64;
        let sd = (allp.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / allp.len() as f64).sqrt();
        let gap = allp
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0f64, f64::max);
        println!(
            "  mean {:.2}  sd {:.2}  largest gap anywhere in the sorted list {:.3} m",
            mean, sd, gap
        );
    }

    let mut cl_out: Vec<(f64, Vec<Vec<usize>>)> = Vec::new();
    for &eps in eps_list {
        let mut cl = lines::cluster(&an.d, eps);
        cl.sort_by_key(|c| c.iter().map(|&i| an.runs[i].time_ms).min().unwrap());
        println!();
        println!(
            "=== eps = {:.1} m : {} distinct line{} ===",
            eps,
            cl.len(),
            if cl.len() == 1 { "" } else { "s" }
        );
        for (ci, c) in cl.iter().enumerate() {
            let mut mem: Vec<usize> = c.clone();
            mem.sort_by_key(|&i| an.runs[i].time_ms);
            let names: Vec<&str> = mem.iter().map(|&i| an.names[i].as_str()).collect();
            println!(
                "  line {} (n={:2}, internal spread {:.2} m, seed {} @ {} ms): {}",
                ci + 1,
                c.len(),
                lines::spread(&an.d, c),
                an.names[mem[0]],
                an.runs[mem[0]].time_ms,
                names.join(", ")
            );
        }
        if cl.len() > 1 {
            let mut seps = Vec::new();
            for ci in 0..cl.len() {
                for cj in (ci + 1)..cl.len() {
                    let mut m = f64::INFINITY;
                    for &i in &cl[ci] {
                        for &j in &cl[cj] {
                            m = m.min(an.d[i][j]);
                        }
                    }
                    seps.push(m);
                }
            }
            println!(
                "  min separation between distinct lines: {:.2} m",
                seps.iter().cloned().fold(f64::INFINITY, f64::min)
            );
        }
        cl_out.push((eps, cl));
    }

    if plots && full {
        println!();
        println!("OVERHEAD VIEW (x-z), all runs; C=CP, F=finish, S=start");
        let labels: Vec<(String, char)> = an
            .names
            .iter()
            .map(|nm| {
                (
                    nm.clone(),
                    if nm == an.ref_name() { 'W' } else { '.' },
                )
            })
            .collect();
        println!(
            "{}",
            lines::ascii_xz(&an.stations_by_run, &labels, lines::MAP_GEOM, 96, 40)
        );

        println!();
        println!(
            "SPEED vs DISTANCE ALONG LAP (km/h vs m):  W = {}, . = others",
            an.ref_name()
        );
        let mut series: Vec<(char, Vec<(f64, f64)>)> = an
            .names
            .iter()
            .filter(|nm| *nm != an.ref_name())
            .map(|nm| {
                (
                    '.',
                    an.stations_by_run[nm].iter().map(|p| (p.0, p.4)).collect(),
                )
            })
            .collect();
        series.push((
            'W',
            an.stations_by_run[an.ref_name()]
                .iter()
                .map(|p| (p.0, p.4))
                .collect(),
        ));
        println!("{}", lines::ascii_series(&series, 100, 18, "km/h", "m along lap"));

        println!();
        println!("LATERAL OFFSET vs DISTANCE ALONG LAP (m):  0 = the reference line");
        let series: Vec<(char, Vec<(f64, f64)>)> = an
            .names
            .iter()
            .filter(|nm| *nm != an.ref_name())
            .map(|nm| {
                (
                    '.',
                    (0..an.stations)
                        .map(|i| (an.ref_stations[i].0, an.profiles[nm][i]))
                        .collect(),
                )
            })
            .collect();
        println!(
            "{}",
            lines::ascii_series(&series, 100, 18, "m (+ = left of ref)", "m along lap")
        );
    }

    if let Some(f) = out {
        std::fs::write(f, lines::clusters_json(an, eps_list, &cl_out)).expect("write out");
        println!("\nwrote {}", f);
    }
}
