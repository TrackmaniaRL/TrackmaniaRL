//! `tmtraj impacts` -- the collision census: where a run loses speed in one
//! sample, and by how much.
//!
//! This exists because "the car keeps banging off the edges — did we leave a
//! debug collision in?" is a question about a video, and a video cannot answer
//! it. Three facts turn it into a measurement:
//!
//! 1. **A MediaTracker render runs no physics.** It plays a recorded
//!    trajectory back. So whatever the clip shows is in the telemetry, and the
//!    question is whether the TELEMETRY's impacts are the engine's.
//! 2. **An impact has a signature the record carries**: the car's own recorded
//!    speed drops by more than a bar in a single 50 ms sample. That is the
//!    definition 134672's own analysis used to order the whole leaderboard —
//!    rank 2 and the previous best TAS have zero, the back half of the field
//!    six to eight — so it is not invented here.
//! 3. **Two independent regenerations of one tape are two readings of the same
//!    engine.** Each one locates the car in a fresh process and samples its
//!    state; they share no address, no run and no day. If both report the same
//!    impacts at the same milliseconds, the impacts are what the engine did.
//!
//! `--against` does exactly that comparison and prints the matched pairs, so a
//! disagreement names the millisecond rather than the file.
//!
//! Speeds print in km/h because that is the unit the car's own dashboard and
//! this project's collision bar are in.

use gbx::record::decode_ghost;

pub struct Impact {
    pub t_ms: i32,
    pub before_kmh: f64,
    pub after_kmh: f64,
    pub pos: [f64; 3],
}

impl Impact {
    pub fn loss(&self) -> f64 {
        self.before_kmh - self.after_kmh
    }
}

/// Every one-sample speed loss above `bar` km/h, inside `[0, race_ms]`.
///
/// The loss is measured on the car's OWN recorded speed, not on a position
/// difference: a position difference over one 50 ms sample also moves when the
/// car turns, and a hard corner is not a collision.
pub fn census(path: &str, bar: f64, race_ms: Option<i64>) -> Result<Vec<Impact>, String> {
    let d = decode_ghost(path)?;
    let mut out = Vec::new();
    for w in d.samples.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if let Some(r) = race_ms {
            if b.time_ms as i64 > r {
                break;
            }
        }
        if !a.speed_kmh.is_finite() || !b.speed_kmh.is_finite() {
            continue;
        }
        if a.speed_kmh as f64 - b.speed_kmh as f64 > bar {
            out.push(Impact {
                t_ms: b.time_ms,
                before_kmh: a.speed_kmh as f64,
                after_kmh: b.speed_kmh as f64,
                pos: [b.x as f64, b.y as f64, b.z as f64],
            });
        }
    }
    Ok(out)
}

const USAGE: &str = "\
usage: tmtraj impacts GHOST... [--bar KMH] [--race S] [--against OTHER.Ghost.Gbx]

Every one-sample loss of speed above --bar (default 15 km/h, the bar this
project's 134672 analysis used to order the whole leaderboard). --against
compares two independent regenerations of one tape: they share no located
address and no run, so impacts appearing in both at the same millisecond are
the engine's, not the recording's.
";

pub fn cmd(argv: &[String]) -> i32 {
    let a = crate::cli::parse("tmtraj impacts", argv, &[]);
    let bar: f64 = a.num("bar", 15.0);
    let race_s: f64 = a.num("race", f64::MAX);
    let against = a.one("against").map(|s| s.to_string());
    let a = a.finish(USAGE);
    let files: Vec<String> = a.positional.clone();
    if files.is_empty() {
        eprint!("{USAGE}");
        return 2;
    }
    let race: Option<i64> =
        if race_s == f64::MAX { None } else { Some((race_s * 1000.0).round() as i64) };

    let mut sets: Vec<(String, Vec<Impact>)> = Vec::new();
    for f in &files {
        match census(f, bar, race) {
            Ok(v) => sets.push((f.clone(), v)),
            Err(e) => {
                eprintln!("{f}: {e}");
                return 2;
            }
        }
    }
    for (f, v) in &sets {
        println!("{f}");
        println!("  {} impact(s) losing more than {:.0} km/h in one sample", v.len(), bar);
        for i in v {
            println!(
                "    {:>8.3}  {:7.1} -> {:6.1} km/h  (-{:5.1})  at ({:.1}, {:.1}, {:.1})",
                i.t_ms as f64 / 1000.0,
                i.before_kmh,
                i.after_kmh,
                i.loss(),
                i.pos[0],
                i.pos[1],
                i.pos[2]
            );
        }
    }

    let Some(other) = against else { return 0 };
    let b = match census(&other, bar, race) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{other}: {e}");
            return 2;
        }
    };
    println!("\n=== against {other} ({} impacts)", b.len());
    let mut worst = 0i32;
    let mut unmatched = 0usize;
    for (f, v) in &sets {
        println!("{f}:");
        for i in v {
            // Two readings of one engine run land on the same 50 ms sample or
            // on its neighbour; anything further apart is a different event.
            match b.iter().min_by_key(|j| (j.t_ms - i.t_ms).abs()) {
                Some(j) if (j.t_ms - i.t_ms).abs() <= 50 => {
                    worst = worst.max((j.t_ms - i.t_ms).abs());
                    println!(
                        "  {:>8.3}  -{:5.1} km/h   matched at {:>8.3} (-{:5.1})  dt {:+} ms, dloss {:+.1}",
                        i.t_ms as f64 / 1000.0,
                        i.loss(),
                        j.t_ms as f64 / 1000.0,
                        j.loss(),
                        j.t_ms - i.t_ms,
                        j.loss() - i.loss()
                    );
                }
                _ => {
                    unmatched += 1;
                    println!(
                        "  {:>8.3}  -{:5.1} km/h   NO MATCH in the other reading",
                        i.t_ms as f64 / 1000.0,
                        i.loss()
                    );
                }
            }
        }
    }
    let n: usize = sets.iter().map(|s| s.1.len()).sum();
    if unmatched == 0 && n == b.len() {
        println!(
            "\nVERDICT: the two readings agree on all {n} impacts, worst timing difference {worst} ms. \
             Two independent locates of the car in two separate engine processes produced the same \
             collisions at the same instants -- these are the engine's, not the recording's."
        );
        0
    } else {
        println!(
            "\nVERDICT: {unmatched} impact(s) appear in one reading and not the other ({n} vs {}). \
             The two readings of this tape DISAGREE about what the car hit.",
            b.len()
        );
        1
    }
}
