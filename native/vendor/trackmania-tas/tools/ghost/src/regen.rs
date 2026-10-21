//! `ghost regen` -- from INPUTS + MAP, generate the car state and write it into
//! the ghost, so the recorded trajectory matches the tape.
//!
//! This is the operation that keeps biting the project, and it bites in a
//! specific way: the state reader locates the car in a live engine's memory,
//! and WHICH COPY it lands on varies between runs. A wrong pick produces a
//! file that passes coverage, passes a manifest, passes a two-run agreement
//! test, and carries a frozen memory slot instead of a car.
//!
//! So this wrapper does not just call the regenerator. It runs it, then puts
//! the result through a gate that can IDENTIFY the answer rather than count
//! votes:
//!
//!   G1  the written file's own telemetry agrees with the tape it carries
//!       (the two input channels in one file, no reference needed);
//!   G2  the first sample is at the map's spawn, not somewhere else;
//!   G3  the path length is a plausible driven distance, not 0 m and not 1e28;
//!   G4  the plain oracle re-simulates THE WRITTEN FILE to the declared time;
//!   G5  no sample is NaN and no position is constant for the whole run.
//!
//! and `ghost regen-control` runs the strongest control the write path has: a
//! ghost that already carries its own true telemetry, regenerated from its own
//! inputs, must reproduce that telemetry. If the fixed point does not hold on a
//! known-good file, nothing the regenerator says about an unknown one counts.

use gbx::container::secs;
use crate::verify;
use crate::cli::{die, flag, has, num};
use std::path::Path;
use std::process::Command;

fn fk_binary() -> String {
    if let Ok(v) = std::env::var("FK_BIN") {
        return v;
    }
    if let Ok(me) = std::env::current_exe() {
        if let Some(d) = me.parent() {
            let p = d.join("fk");
            if p.exists() {
                return p.to_string_lossy().into();
            }
            // `tools/fk` IS IN THIS REPOSITORY, and it is the first place to
            // look. It is excluded from the workspace -- it pins -O3 + LTO for
            // the engine paths -- so its artefact is never beside this binary,
            // and the resolver that only knew "beside me, or on PATH" reported
            // `cannot run fk: No such file or directory` on a checkout that
            // had just built it. The shim lookup below then had no fk
            // directory to anchor on either, so fixing that one alone left the
            // pair still broken.
            for rel in [
                "../../fk/target/release/fk",       // tools/<crate>/target/release
                "../../../fk/target/release/fk",    // tools/target/release (the workspace)
                "../../../tools/fk/target/release/fk",
            ] {
                let p = d.join(rel);
                if p.exists() {
                    return p.to_string_lossy().into();
                }
            }
        }
    }
    // `fk` is part of the internal fork-server toolchain, not of this crate, so
    // it may also be somewhere else on PATH. Resolve it there and remember the
    // directory: its shim lives beside it.
    if let Ok(paths) = std::env::var("PATH") {
        for d in paths.split(':') {
            let p = std::path::Path::new(d).join("fk");
            if p.exists() {
                return p.to_string_lossy().into();
            }
        }
    }
    "fk".into()
}

/// Where the LD_PRELOAD shim is.
///
/// **Both names are tried.** The crate was renamed `forkshim` and its artefact
/// with it (`libforkshim.so`); `fk`'s own lookup accepts either, and this one
/// did not, so from a clean clone `ghost regen` handed the fork server a path
/// that does not exist. The server then panics inside `forksrv` with a bare
/// `NotFound` — once per attempt, twenty-four times — and the output reads
/// exactly like twenty-four bad locates. That is the failure this crate's own
/// README warns about, reproduced by this crate.
///
///
/// This was fixed twice on the same day, independently. The other fix deleted
/// this lookup outright on the principle that `fk` should own resolving what
/// `fk` loads -- which is right about the duplication and wrong about the
/// failure mode: forwarding nothing means a missing shim is discovered by the
/// fork server, twenty-four times, in the words of twenty-four failed locates.
/// Refusing here, by name, is worth the second copy.
///
/// Returns `None` rather than a guess: a made-up default is what turns a
/// wiring error into a physics story. The caller refuses and names the knob.
fn shim() -> Option<String> {
    if let Ok(v) = std::env::var("FK_SHIM") {
        return Some(v);
    }
    const NAMES: [&str; 2] = ["libforkshim.so", "libfkshim.so"];
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    // beside the fk binary that will actually run -- NOT beside this one. They
    // are different crates in different trees.
    let fk = fk_binary();
    if let Some(d) = std::path::Path::new(&fk).parent() {
        dirs.push(d.to_path_buf());
        // `cargo build` leaves the shim in the SEARCH workspace's target dir,
        // never in fk's own: the shim and the driver `#[path]`-include one
        // `pred_core.rs`, so a second copy would be a second judge.
        dirs.push(d.join("../../../search/target/release"));
    }
    if let Ok(me) = std::env::current_exe() {
        if let Some(d) = me.parent() {
            dirs.push(d.to_path_buf());
            dirs.push(d.join("../search/target/release"));
            dirs.push(d.join("../../search/target/release"));
        }
    }
    dirs.push(std::path::PathBuf::from("/tmp/fk/rs/target/release"));
    for d in dirs {
        for n in NAMES {
            let p = d.join(n);
            if p.exists() {
                return Some(p.to_string_lossy().into());
            }
        }
    }
    None
}


pub struct RegenOut {
    pub ok: bool,
    pub log: String,
}

/// Run the regenerator once. `dump` MUST be absolute: a relative path fails
/// inside the forked server with a bare `go ERR open`.
pub fn run_regen(template: &str, map: &str, out: &str, extra: &[String]) -> RegenOut {
    let shim = match shim() {
        Some(s) => s,
        None => {
            // Name the knob rather than launching. A shim that is not there
            // makes 24 launches that never happened report in the words of 24
            // failed locates.
            return RegenOut {
                ok: false,
                log: "the LD_PRELOAD shim (libforkshim.so / libfkshim.so) was not found beside \
                      fk, in tools/search/target/release, or at $FK_SHIM. Build it with \
                      `cd tools/search && cargo build --release -p forkshim`, or set FK_SHIM. \
                      NOT LAUNCHING: a missing shim reads exactly like a failed locate."
                    .into(),
            };
        }
    };
    let dump = std::env::temp_dir()
        .join(format!("ghost-regen-{}-{}.bin", std::process::id(), rand_tag()))
        .to_string_lossy()
        .to_string();
    let mut args: Vec<String> = vec![
        "regen".into(),
        "--template".into(),
        abspath(template),
        "--map".into(),
        abspath(map),
        "--out".into(),
        abspath(out),
        "--dump".into(),
        dump,
        "--shim".into(),
        shim,
        "--server".into(),
        crate::oracle::server_dir(None).to_string_lossy().to_string(),
    ];
    args.extend_from_slice(extra);
    let o = Command::new(fk_binary()).args(&args).output();
    match o {
        Err(e) => RegenOut { ok: false, log: format!("cannot run {}: {}", fk_binary(), e) },
        Ok(o) => {
            let log = format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            RegenOut { ok: o.status.success() && Path::new(out).exists(), log }
        }
    }
}

fn rand_tag() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!("{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos())
}

fn abspath(p: &str) -> String {
    std::fs::canonicalize(p)
        .map(|x| x.to_string_lossy().to_string())
        .unwrap_or_else(|_| {
            let cwd = std::env::current_dir().unwrap();
            cwd.join(p).to_string_lossy().to_string()
        })
}

/// Path length of the written file's own recorded trajectory, and its first
/// sample. `carscan`'s test in one place: a car's path is the map's ribbon, a
/// wrong memory copy's is 0.0 m or 1e28 m.
pub fn path_stats(file: &str) -> Option<(f64, [f64; 3], usize, bool)> {
    let d = gbx::record::decode_ghost(file).ok()?;
    if d.samples.is_empty() {
        return None;
    }
    let mut len = 0.0;
    let mut finite = true;
    let mut moved = false;
    for w in d.samples.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let dx = (b.x - a.x) as f64;
        let dy = (b.y - a.y) as f64;
        let dz = (b.z - a.z) as f64;
        let s = (dx * dx + dy * dy + dz * dz).sqrt();
        if !s.is_finite() {
            finite = false;
            continue;
        }
        if s > 1e-6 {
            moved = true;
        }
        len += s;
    }
    let f = &d.samples[0];
    Some((len, [f.x as f64, f.y as f64, f.z as f64], d.samples.len(), finite && moved))
}

pub fn cmd(a: &[String]) {
    let inp = a.first().unwrap_or_else(|| die("ghost regen IN OUT --map MAP"));
    let out = a.get(1).unwrap_or_else(|| die("ghost regen IN OUT --map MAP"));
    let map = flag(a, "--map").unwrap_or_else(|| die("--map MAP.Map.Gbx"));
    let extra: Vec<String> = {
        let mut v = Vec::new();
        // Value flags, forwarded verbatim. `--fieldmap` is NOT among them any
        // more: `fk` deleted it along with the field-fitting machinery, and a
        // wrapper that forwards a flag the callee does not know is a silent
        // no-op wearing the shape of a feature.
        for k in ["--anchorticks", "--segs", "--biastick", "--inputshift", "--anchor", "--race", "--quat-kind", "--pair-shift-ms"] {
            if let Some(x) = flag(a, k) {
                v.push(k.to_string());
                v.push(x.to_string());
            }
        }
        for k in ["--verbose", "--inherit-outside", "--allow-partial",
                  "--keep-transform", "--noanchor"] {
            if has(a, k) {
                v.push(k.to_string());
            }
        }
        // ALWAYS ON, no flag, no way to turn them off.
        //
        // `--neutralise` zeros the 49 per-run bytes the transform encoder does
        // not write; `--inputs` rewrites the record's steer/gas/brake echo from
        // the tape; `--trim-outside` drops samples with no engine instant
        // behind them. All three were opt-in, all three were forgotten, and
        // each omission shipped: the tyres throwing a stranger's dirt is the
        // first, a record describing the donor's driving is the second.
        //
        // Making them options was the mistake. There is no case for a file that
        // is publishable-except-for-one-of-these -- that is just a defect with
        // a flag in front of it.
        v.push("--neutralise".into());
        v.push("--inputs".into());
        v.push("--trim-outside".into());
        // AND THE CARRIER, for the same reason. Without it the transform writes
        // 22 of the 116 sample bytes and the other 94 are ZEROED -- which on
        // 287431 (measured 2026-08-29) draws the car as a TRANSPARENT
        // WIREFRAME. Three clips shipped like that, every one gating at V6
        // kappa 1.000, because kappa compares the tape to the record and
        // cannot see whether a car is drawn.
        //
        // `layout` is the game's own writer transcribed from the archiver at
        // 0x9cfed0 (see `vislayout`): no coefficients, no hand-fitted table to
        // drift, and it covers the packed bit-fields no per-byte affine fit
        // can represent. There was a `--carrier` flag to select it. It is gone
        // -- a file with unauthored visual channels is not publishable, so
        // there is nothing for the flag to have meant.
        v.push("--carrier".into());
        v.push("layout".into());
        v
    };
    // A FLAG THAT DOES NOTHING IS WORSE THAN NO FLAG. All five above are pushed
    // unconditionally, so passing one by hand changed nothing and the run said
    // nothing either. On 287431 `--neutralise` was tested on and off, the two
    // outputs were byte-identical (md5 b9a8bc3a...), and the byte hypothesis --
    // which was the answer -- was eliminated on the strength of it. The flags
    // are gone from the usage; a caller who still passes one is told plainly
    // rather than silently humoured.
    for k in ["--neutralise", "--inputs", "--trim-outside", "--carrier"] {
        if has(a, k) {
            println!(
                "NOTE: {k} is UNCONDITIONAL in this pipeline -- passing it changes nothing, and \
                 there is no run without it. Do not read a difference between two runs as this \
                 flag's doing."
            );
        }
    }
    // ONE ATTEMPT. There is nothing to retry.
    //
    // These defaulted to 24 and 12: a dozen parallel engine runs per
    // regeneration, because the car was found by sweeping memory for
    // something that "moves like a car" and that sweep landed on a decoy about
    // seven times in eight. The car is now resolved from a pointer chain out
    // of static data, which is right in every process, so a second attempt
    // does exactly what the first did. Measured after the change: 5 of 5 runs
    // succeeded and all five were byte-identical.
    //
    // The knobs remain for `--census`, which deliberately runs a batch to ask
    // whether the candidates AGREE. They are not a reliability mechanism.
    // ONE ATTEMPT, THEN ESCALATE IF IT FAILED.
    //
    // The two paths inside `fk regen` have opposite needs and only the tool
    // knows which one ran. A map whose pointer chain is known is exact: one
    // attempt, ~20 s, and five runs in a row come out byte-identical. A map
    // whose chain is not known falls back to the memory search, which is a
    // lottery -- a `base±N` anchor measured in one process is right in the
    // next only when the allocation repeated -- and needs a dozen tickets.
    //
    // Defaulting to 12 taxes every good map; defaulting to 1 broke 287431.
    // So: start with 1, and if nothing passed the gate, run the batch. The
    // fast path stays fast and the fallback keeps the reliability it needs.
    let user_set_tries = num(a, "--tries").is_some() || num(a, "--jobs").is_some();
    let mut tries: i64 = num(a, "--tries").unwrap_or(1);
    let mut jobs: usize = num(a, "--jobs").unwrap_or(1).max(1) as usize;
    let mut escalated = false;
    let force = has(a, "--force");
    // The container this file is built in, kept before `inp` becomes the
    // rebuilt grid: the finishing pass measures per-byte provenance against it,
    // and against the grid the answer would be meaningless.
    let template_for_provenance = inp.clone();

    // ---- STEP 0: THE GRID THIS RUN WILL BE WRITTEN INTO IS OURS -------------
    //
    // The regenerator writes engine state into the record the template already
    // has, so a template carrying the donor's span produces a file carrying the
    // donor's span -- 33 of 159 published ghosts do, and the symptom is a clip
    // twice as long as the run with the camera stranded at the top of the map
    // after our car's entity ends. This used to be `ghost record rebuild`, a
    // separate command run afterwards by whoever knew to.
    //
    // It is also what makes 227654 regenerable at all: that carrier is ONE CAR
    // SPLIT INTO 27 ENTITIES at its respawns, so every reader in this project
    // sees "365 samples spanning 1.310 .. 19.480" for a 57.482 run and the
    // regenerator hard-errors on the 38 seconds it cannot fill.
    //
    // The span is the file's own simulated time, asked of the plain oracle --
    // never the filename and never the header, which on a synthesised tape is
    // the seed's.
    let inp = {
        // SAY WHICH OF THE THREE THINGS HAPPENED.
        //
        // "no oracle available" covered all of: no server on this box, the
        // server failing to run, and the server running fine and returning DNF.
        // Those need different responses and the message named the first, so a
        // DNF on a 32-minute tape read as a missing install and sent me to
        // check TM_SERVER, which was correct all along. An absent signal read
        // as a specific negative -- the project's characteristic bug, in my own
        // code, on the same day I fixed two of them elsewhere.
        let t = crate::oracle::server_dir(flag(a, "--server"));
        let simulated = if !t.join("TrackmaniaServer").exists() {
            println!(
                "== step 0: no dedicated server at {} -- the record's span is left as it is. \
                 Pass --server DIR or set TM_SERVER to have it rebuilt.",
                t.display()
            );
            None
        } else {
            match crate::oracle::validate(
                &t,
                std::path::Path::new(inp),
                crate::oracle::MapsMode::One(std::path::Path::new(&map)),
                "regen-span",
            ) {
                Err(e) => {
                    println!("== step 0: the oracle could not run ({e}) -- the span is left as it is");
                    None
                }
                Ok(r) => match r.time_ms {
                    None => {
                        println!(
                            "== step 0: the oracle returns DNF for this file, so there is no \
                             finish to shorten the record to -- the span is left as it is. That \
                             is expected on a deliberately partial tape and is a REAL FINDING on \
                             one that should finish."
                        );
                        None
                    }
                    Some(ms) => Some(ms),
                },
            }
        };
        match simulated {
            None => inp.clone(),
            Some(ms) => {
                // .Ghost.Gbx, not `{out}.grid`: the dedicated server IGNORES a
                // file with any other extension and returns a bare DNF that
                // cannot be told from a genuine one -- the oracle wrapper says
                // so in as many words, and it caught this staging name on the
                // first run.
                let staged = format!("{}.grid.Ghost.Gbx", out.trim_end_matches(".Ghost.Gbx"));
                match crate::record::rebuild_to(inp, &staged, ms, None, 50, false) {
                    Ok(msg) => println!("== step 0: the record grid is ours\n   {msg}"),
                    Err(e) => die(format!("could not rebuild the record grid: {e}")),
                }
                // AND THE DECLARED TIME, HERE, BEFORE THE ENGINE RUNS.
                //
                // It has to be before: G4 re-simulates the WRITTEN file and
                // compares against what that file declares, so a grid still
                // carrying the carrier's 147.031 makes every one of 24 correct
                // regenerations fail a check about a number none of them wrote.
                // Declaring it afterwards would mean the gate ran against a
                // claim we already knew was wrong.
                let dstage = format!("{}.decl.Ghost.Gbx", out.trim_end_matches(".Ghost.Gbx"));
                // `--declared-known` skips the oracle here, and only a caller
                // that already knows the time may pass it.
                //
                // `--from-oracle` exists so a declared time is never TYPED by
                // a human or copied out of a search log -- it asks the engine
                // what the file does. That is right for a person at a prompt
                // and redundant for `ghost film`, which read the span out of
                // the input tape and just rebuilt this grid to `ms`. The
                // oracle is a whole engine boot (~3 s of a 12.9 s
                // `ghost regen`) to re-derive a number we were handed.
                //
                // G4 still re-simulates the WRITTEN file at the end and
                // compares it against what that file declares, so a wrong
                // value here cannot ship: it fails the gate instead.
                let mut d: Vec<String> = if has(a, "--declared-known") {
                    println!("   declaring {} (given by the caller; G4 still checks it)", secs(ms));
                    vec![staged.clone(), dstage.clone(), "--time".into(), ms.to_string()]
                } else {
                    vec![staged.clone(), dstage.clone(), "--from-oracle".into(),
                         "--map".into(), map.to_string()]
                };
                if let Some(sv) = flag(a, "--server") {
                    d.push("--server".into());
                    d.push(sv.to_string());
                }
                crate::declare::cmd(&d);
                let _ = std::fs::remove_file(&staged);
                dstage
            }
        }
    };
    let inp = &inp;

    // THE LOCATE IS A CHOOSER AND IT IS NONDETERMINISTIC. Measured on the
    // fixture map: 8 identical runs, one found the car and seven found a
    // neighbouring object -- and the gate below refused all seven. A base rate
    // near 1 in 8 is why this runs a DOZEN attempts at once instead of
    // retrying serially and giving up. So run several
    // attempts AT ONCE, each with a different anchor ladder, and take the first
    // the gate accepts. Diversifying the anchor is not cosmetic: which object
    // the locate finds follows from where it starts looking.
    // AND DIVERSIFYING THE ANCHOR MAKES IT WORSE, WHICH IS NOT WHAT I EXPECTED.
    // Eight runs on the default ladder: 1 found the car. Twenty-four runs
    // spread over seven hand-picked anchor ladders: 0 did. So every attempt
    // uses the default ladder unless the caller asks for one, and the only
    // thing that buys reliability here is running more of them at once.
    let ladders: Vec<Option<&str>> = vec![None];
    let mut accepted: Option<String> = None;
    let mut passed: Vec<String> = Vec::new();
    // `--census`: gate EVERY candidate in the batch instead of stopping at the
    // first one home, and say how many distinct trajectories they were.
    //
    // IT DOES NOT CHOOSE, AND THE FIRST VERSION OF IT DID. I wrote this to
    // take the modal trajectory, on the strength of downloaded controls where
    // the good phase was the majority -- 270051 4 of 5 runs on the game's tick,
    // 279218 4 of 5. Then I measured two more maps: 285268 is 1 of 5 and
    // 279209 is 1 of 5, so on those the mode is the WRONG tick and a chooser
    // built on it would have shipped the error with a majority behind it. The
    // batch can tell you THAT it disagreed; only a downloaded control of the
    // same map can tell you which side is the game's.
    let census = has(a, "--census");
    let mut lastlog = String::new();
    let mut round = 0usize;
    // THE IN-PROCESS LOCATE GOES FIRST, ALONE.
    //
    // There are two locates in the state reader. The default one forks, hunts
    // the child for an object that moves like a car, and hands an address back
    // to the parent; the other locates in the clean process itself and needs no
    // cross-process assumption at all. The forking one is the brittle one, and
    // it was running first and usually winning with a decoy.
    //
    // Measured on the fixture map: six runs with the in-process locate produced
    // BIT-IDENTICAL trajectories in 13.7 s each; six runs on the default path
    // took ~90 s and disagreed. It is not universal -- it cannot see a car that
    // is barely moving at the handover, and on a short tape it finds nothing --
    // so the search is still there behind it. But it is the right thing to try
    // first, and when it works there is no search at all.
    // ...EXCEPT WHEN THE CARRIER FIELDS ARE WANTED.
    //
    // This path passes `--noanchor`, and the field gather is driven off the
    // anchors: with none measured it does not run at all. So `--carrier` was
    // silently dropped on the path that usually wins, and the acceptance gate
    // then refused the file for having dead wheel rotations -- correctly, and
    // for a reason two layers away from the cause. The log's tell is that
    // "carrier" appears zero times in it.
    //
    // A flag that is quietly ignored is worse than one that fails, so when
    // fields are asked for this path is skipped rather than run without them.
    // ALWAYS true: the carrier is unconditional now (the flag was removed), so
    // the in-process path must always be skipped. Reading the user flag here
    // after deleting it silently disabled the field gather -- regen then ran
    // the fast path, wrote 22 of 116 bytes, and the reactor stayed frozen at
    // its template value while the gate passed. Measured on 203072.
    let want_fields = true;
    if !has(a, "--no-inprocess") && !want_fields {
        let cand = format!("{}.ip", out);
        let raw = format!("{}.raw", cand);
        let mut ex = extra.clone();
        ex.push("--noanchor".into());
        let r = run_regen(inp, map, &raw, &ex);
        lastlog = r.log.clone();
        if r.ok {
            match write_input_channels(&raw, &cand) {
                Ok((w, sk)) => println!("   [in-process] input channels rewritten from the tape on {} samples ({} outside)", w, sk),
                Err(_) => {
                    let _ = std::fs::rename(&raw, &cand);
                }
            }
            let _ = std::fs::remove_file(&raw);
            match gate(&cand, map, a, inp) {
                Ok(msg) => {
                    println!("   [in-process] accepted -- no cross-process search was run");
                    println!("{}", msg);
                    accepted = Some(cand);
                }
                Err(msg) => {
                    println!("{}", msg);
                    println!("   [in-process] refused; falling back to the anchor search");
                    let _ = std::fs::remove_file(&cand);
                }
            }
        } else {
            let _ = std::fs::remove_file(&raw);
            println!("   [in-process] no file (this tape is probably not moving at the handover); falling back to the anchor search");
        }
    }
    while (round as i64) < tries && accepted.is_none() {
        let batch: Vec<usize> = (round..(round + jobs).min(tries as usize)).collect();
        if batch.is_empty() {
            break;
        }
        println!("== regeneration attempts {:?} of {} (in parallel)", batch, tries);
        let results: Vec<(usize, String, RegenOut)> = std::thread::scope(|sc| {
            let hs: Vec<_> = batch
                .iter()
                .map(|kk| {
                    let k = *kk;
                    let inp = inp.to_string();
                    let mapx = map.to_string();
                    let cand = format!("{}.try{}", out, k);
                    let mut ex = extra.clone();
                    if !ex.iter().any(|x| x == "--anchorticks") {
                        if let Some(Some(l)) = ladders.get(k % ladders.len()) {
                            ex.push("--anchorticks".into());
                            ex.push((*l).to_string());
                        }
                    }
                    sc.spawn(move || {
                        let raw = format!("{}.raw", cand);
                        let r = run_regen(&inp, &mapx, &raw, &ex);
                        (k, cand, r)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (k, cand, r) in results {
            let raw = format!("{}.raw", cand);
            lastlog = r.log.clone();
            if let Some(l) = r.log.lines().find(|l| l.starts_with("record layout:")) {
                println!("   [{}] {}", k, l);
            }
            if !r.ok {
                println!("   [{}] the regenerator did not produce a file", k);
                for l in r.log.lines().rev().take(2) {
                    println!("   [{}] | {}", k, l);
                }
                let _ = std::fs::remove_file(&raw);
                continue;
            }
            // Widen the write: the recorded steer / gas / brake channels come
            // from the TAPE, not from the engine, and leaving them as the
            // carrier's is how a regenerated file describes somebody else's
            // driving.
            match write_input_channels(&raw, &cand) {
                Ok((w, sk)) => println!(
                    "   [{}] input channels rewritten from the tape on {} samples ({} outside the tape)",
                    k, w, sk
                ),
                Err(e) => {
                    println!("   [{}] could not rewrite the input channels: {}", k, e);
                    let _ = std::fs::rename(&raw, &cand);
                }
            }
            let _ = std::fs::remove_file(&raw);
            if accepted.is_some() && !census {
                let _ = std::fs::remove_file(&cand);
                continue;
            }
            match gate(&cand, map, a, inp) {
                Ok(msg) => {
                    println!("   [{}] accepted", k);
                    println!("{}", msg);
                    if accepted.is_none() {
                        accepted = Some(cand.clone());
                    }
                    passed.push(cand);
                }
                Err(msg) => {
                    println!("{}", msg);
                    println!("   [{}] REFUSED -- the locate found something that is not the car.", k);
                    let _ = std::fs::remove_file(&cand);
                }
            }
        }
        round += jobs;
        // Nothing passed on the single default attempt: the fallback search
        // ran, and it needs its tickets. Escalate ONCE, then let the loop
        // continue with the wider batch.
        if accepted.is_none() && !user_set_tries && !escalated && (round as i64) >= tries {
            escalated = true;
            tries = 12;
            jobs = 12;
            println!("== the single attempt did not pass; escalating to {} parallel attempts", jobs);
        }
        // THE BATCH ALREADY HOLDS THE ANSWER AND THE OLD CODE THREW IT AWAY.
        //
        // Twelve attempts run in parallel and the first one that passed the
        // gate was kept. But the gate cannot see the one thing these attempts
        // disagree about: WHICH PHYSICS TICK the samples are labelled with.
        // The regenerator's clock bias is measured per run, and on some maps it
        // lands a tick out one run in five -- so "the first that passed" is a
        // lottery over a 10 ms time shift, which is invisible in one ghost and
        // fatal in a frame-synchronous two-car comparison. Measured on the
        // downloaded human recordings the game made itself: 270051 4 of 5 runs
        // on the game's tick, 279218 4 of 5, 285268 3 of 5.
        //
        // The candidates are already there, already gate-passed, and free. So
        // cluster them and take the MODE, which is the rule this project
        // already learned for every other jittering instrument.
        if census && passed.len() >= 3 {
            match trajectory_census(&passed) {
                Some(text) => println!("== {}", text),
                None => println!("== could not compare the attempts' trajectories"),
            }
            for c in &passed {
                if Some(c) != accepted.as_ref() {
                    let _ = std::fs::remove_file(c);
                }
            }
            break;
        }
    }
    match accepted {
        None => {
            eprintln!("--- last regenerator log ---");
            eprintln!("{}", lastlog);
            // A verdict, not a usage error: exit 1 (see cli::refuse).
            crate::cli::refuse(format!(
                "no regeneration passed the gate in {} attempts. NOTHING was written to {}: a file \
                 that fails this gate is a frozen memory slot, not a car.",
                tries, out
            ));
        }
        Some(c) => {
            std::fs::rename(&c, out).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
            finish(out, &template_for_provenance, &map, a, force);
        }
    }
}

/// Everything between "the engine state is in the record" and "this file is
/// publishable" -- run on every regeneration, not as a follow-up somebody has
/// to remember.
///
/// The list used to live in people's heads and every item on it shipped at
/// least once: a stranger's dirt-tyre bytes, a 441 s span on a 218 s run, a
/// header declaring the donor's time, a foreign login in a field nothing read.
/// Each was found days later by someone noticing something odd in a video.
///
/// Every step here refuses on its own terms, and this refuses on theirs.
fn finish(out: &str, carrier: &str, map: &str, a: &[String], force: bool) {
    println!("\n== the finishing pass");

    // The declared time is written in step 0, BEFORE the engine runs -- G4
    // re-simulates the written file and checks it against what the file
    // claims, so a claim fixed afterwards would mean the gate ran against a
    // number we already knew was wrong.

    // 1. THE IDENTITY. Nine fields, two of them found the day this was written
    //    -- the ranked badge and the zone -- by raw-stringing published files.
    //    `--anonymise` is position-based, so a field nobody has named yet is
    //    still found by the same scan.
    let tmp = format!("{out}.fin");
    let mut i: Vec<String> = vec!["set".into(), out.to_string(), tmp.clone(), "--anonymise".into(),
                                  "--allow-noop".into(),
                                  "--map".into(), map.to_string()];
    if let Some(s) = flag(a, "--server") {
        i.push("--server".into());
        i.push(s.to_string());
    }
    crate::ident::cmd(&i);
    let _ = std::fs::rename(&tmp, out);

    // 1b. THE ENTITY ORDER, RESTORED TO THE CONTAINER'S.
    //
    // The rebuild puts the car at index 0. The game client is not indifferent
    // to that: 203072's container has the car LAST of three entities, and the
    // regenerated file with the car first would not import at all -- the
    // ghost-block count stayed 0 through four attempts and the game raised a
    // FrameMessage each time, while its own published file imported first try
    // in the same scene, on the same map, seconds apart. That file passed
    // `ghost verify` V1-V11 with kappa 1.000 and the oracle exact, so nothing
    // headless could see it; the only symptom is a render that never starts.
    //
    // The rule is NOT "car first" (`entorder`'s help says every ghost the game
    // wrote has the car at 0, and on this container that is simply false). The
    // rule is: give the file back the order its container had.
    match (crate::record::car_index(carrier), crate::record::car_index(out)) {
        (Ok((ci, cn)), Ok((oi, on))) if cn > 1 && on > 1 && ci != oi => {
            let want_first = ci == 0;
            if want_first || ci == cn - 1 {
                let tmp2 = format!("{out}.ord");
                match crate::record::set_ent_order(out, &tmp2, want_first) {
                    Ok(m) => {
                        let _ = std::fs::rename(&tmp2, out);
                        println!("   entity order restored to the container's: {m}");
                    }
                    Err(e) => println!("   entity order could not be restored: {e}"),
                }
            } else {
                println!(
                    "   NOTE: the container has the car at index {ci} of {cn} and this file has \
                     it at {oi} of {on}; only first and last can be restored, so the order is \
                     left as rebuilt. If the client refuses the import, that is where to look."
                );
            }
        }
        _ => {}
    }

    // 2. THE ACCEPTANCE TEST: what, if anything, is still the donor's.
    println!("\n== what this file is made of");
    let carrier_written_early = has(a, "--carrier");
    let mut refused: Vec<String> = Vec::new();
    match crate::finish::inherited_bytes_split(out, carrier, carrier_written_early) {
        Ok(v) => {
            if !v.not_ours.is_empty() {
                refused.push(format!(
                    "{} sample byte(s) THIS PIPELINE DOES NOT WRITE are bit-identical to the donor \
                     on EVERY sample: {:?}. Those describe THEIR run -- every tyre effect and \
                     contact spark in a render fires at the instant they had it, not ours.",
                    v.not_ours.len(),
                    v.not_ours
                ));
            }
            // THE GATHER NOT RUNNING AT ALL is the failure that must not slip
            // through here (c01a842: ~99 channels reached the file as the
            // donor's). It does not look like three bytes matching; it looks
            // like the whole written set matching at once. So the ratio is the
            // gate, and it is printed either way.
            if carrier_written_early && v.ours_varying > 0 {
                let f = v.reproduced.len() as f64 / v.ours_varying as f64;
                if f >= 0.90 {
                    refused.push(format!(
                        "{} of the {} channels this pass WRITES that vary in the donor are \
                         bit-identical to it ({:.0} %). At that fraction the gather did not run: \
                         the file is the donor's telemetry wearing our transform.",
                        v.reproduced.len(), v.ours_varying, 100.0 * f
                    ));
                } else if v.reproduced.is_empty() {
                    println!("   no sample byte is bit-identical to the container donor throughout.");
                } else {
                    println!(
                        "   reproduced exactly (written by this pass, and equal to the donor on \
                         every sample -- {} of {} varying channels, {:.0} %): {:?}",
                        v.reproduced.len(), v.ours_varying, 100.0 * f, v.reproduced
                    );
                    println!(
                        "   That is reproducibility, not inheritance: the donor here is this map's \
                         own previous generation of the SAME tape, and these are low-entropy \
                         channels two runs of one tape land on identically."
                    );
                }
            } else if v.not_ours.is_empty() {
                println!("   no sample byte is bit-identical to the container donor throughout.");
            }
        }
        Err(e) => println!("   per-byte provenance could not be measured: {e}"),
    }
    match crate::finish::outlives_the_car(out) {
        Ok(None) => println!("   nothing in the file outlives the car."),
        Ok(Some(w)) => refused.push(format!("something outlives the car: {w}")),
        Err(e) => println!("   the span could not be measured: {e}"),
    }

    // 3. IS WHAT WE DID WRITE ACTUALLY ALIVE?
    //
    // A bare position copy once wrote zeroed wheels into a file that passed the
    // whole verify gate. Zero is legal for a channel on any one sample; what no
    // driven run does is hold one at a single value for 3 km. The list is
    // whatever the pipeline currently claims to write, so a channel promoted
    // out of `unwritten_channels` comes under this on the same day.
    // Two lists, because refusing is only right where a constant is impossible.
    // A car that drives 3 km always turns its wheels; a car on a short run may
    // never change gear or touch the turbo, and refusing those would refuse
    // honest work -- which is how C3 and C8 had to be superseded.
    let carrier_written = has(a, "--carrier");
    let claim: Vec<(usize, &str)> = crate::finish::must_be_live()
        .iter()
        .copied()
        .filter(|(o, _)| carrier_written || *o >= 14)
        .collect();
    match crate::finish::dead_channels(out, &crate::finish::may_rest()) {
        Ok(v) if v.is_empty() || !carrier_written => {}
        Ok(v) => {
            for d in &v {
                println!("   at rest (written, but a constant value is legitimate): {d}");
            }
        }
        Err(_) => {}
    }
    // A CHANNEL THE SOURCE ALSO HOLDS CONSTANT IS NOT DEAD.
    //
    // `dead_channels` refuses a claimed byte that never varies, which is the
    // right catch: it is how a gather that silently did not run gets caught
    // (c01a842, ~99 channels arriving as the donor's). It excuses a constant
    // only when the TAPE says the driver held that input all run -- and
    // `echoes` maps just steer/accel/brake, so an ENGINE-derived byte can
    // never be excused that way.
    //
    // 287431 is the case: byte 59 (orientation angle) holds 0x00 on all 416
    // samples of our output AND on the source file we are regenerating, and
    // on 203072's film, which passes every gate. The engine is not withholding
    // it from the pointer; it is zero. Refusing there is refusing a faithful
    // reproduction.
    //
    // So a byte that is constant in the SOURCE too is not evidence of a dead
    // gather. Every other constant still is -- the c01a842 failure has ~99
    // channels constant in our output and VARYING in the donor, so it is
    // untouched by this.
    // `carrier` here is `template_for_provenance`, i.e. the ORIGINAL input
    // captured before step 0 rebuilds the grid -- the file we are regenerating,
    // which is exactly the right comparison.
    let src_const = crate::finish::constant_bytes(carrier).unwrap_or_default();
    match crate::finish::dead_channels(out, &claim) {
        Ok(v) if v.is_empty() => println!("   every channel this pipeline writes varies over the run."),
        Ok(v) => {
            for d in &v {
                let excused = d
                    .split_whitespace()
                    .nth(1)
                    .and_then(|n| n.parse::<usize>().ok())
                    .map(|b| src_const.contains(&b))
                    .unwrap_or(false);
                if excused {
                    println!(
                        "   constant in our output AND in the source, so not a dead gather: {d}"
                    );
                } else {
                    refused.push(format!("a channel we claim to write is dead: {d}"));
                }
            }
        }
        Err(e) => println!("   channel liveness could not be measured: {e}"),
    }

    // 4. WHAT WE DID NOT WRITE, BY NUMBER. A harness limit, said as one.
    //
    // The set depends on whether the carrier ran: without it only the transform
    // is ours, and saying "11 channels" there would be a report of a run that
    // did not happen. It used to say exactly that.
    let un = if carrier_written {
        crate::finish::unwritten_channels()
    } else {
        crate::finish::unwritten_channels_without_carrier()
    };
    println!(
        "   UNWRITTEN, left as the carrier's ({} channels): {}",
        un.len(),
        un.iter().map(|(o, n)| format!("{o} {n}")).collect::<Vec<_>>().join(", ")
    );
    if carrier_written {
        println!(
            "   These are not a harness limit in the old sense: 19, 20, 34 and the four dirt\n\
             \x20  slots read identically ZERO in the dedicated server, so writing them would\n\
             \x20  put a confident zero where a real value belongs. Sourcing them means running\n\
             \x20  the CLIENT. 108-111 is the countdown, which needs the race clock, not the car."
        );
    }
    if !carrier_written {
        println!(
            "   Those quantities ARE in engine memory and `--carrier layout` reads them. This\n\
             \x20  run did not ask for it, so they are the donor container's."
        );
    }

    if !refused.is_empty() {
        for r in &refused {
            eprintln!("REFUSED: {r}");
        }
        if !force {
            let _ = std::fs::remove_file(out);
            // NAME THE CHECKS THAT FAILED, in the message that kills the run.
            //
            // These refusals were counted here and printed nowhere near the
            // G1-G4 lines, so a run could show four passing gates and still
            // delete its output with only "1 check(s) failed" to explain it.
            // On 287431 the real reason -- "byte 59 (orientation angle) holds
            // 0x00 on all 416 samples" -- was reachable only by grepping the
            // log for words you had to already know. A gate that can delete
            // the output has to say which gate it was.
            // EXIT 1, NOT 2 (see cli::refuse): the pipeline ran correctly and
            // the answer is "not publishable". A caller must be able to tell
            // that from "you called me wrong", which is exit 2.
            crate::cli::refuse(format!(
                "{} check(s) failed and {out} has been DELETED. Pass --force to keep the file \
                 anyway; it is not publishable.\n{}",
                refused.len(),
                refused
                    .iter()
                    .map(|r| format!("  FAILED: {}", r))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        eprintln!("--force: keeping {out} despite {} failed check(s).", refused.len());
    }
    println!("\n{out} is finished.");
}

/// The acceptance gate for one regenerated candidate.
fn gate(cand: &str, map: &str, a: &[String], template: &str) -> Result<String, String> {
    let mut s = String::new();
    let (len, _first, n, ok) = path_stats(cand).ok_or("   G3 the written file has no samples")?;
    if !ok {
        return Err(format!(
            "   G5 the trajectory is not finite or never moves ({} samples, {:.1} m)",
            n, len
        ));
    }
    if !(1.0..=1.0e6).contains(&len) {
        return Err(format!(
            "   G3 path length {:.1} m over {} samples -- not a driven path",
            len, n
        ));
    }
    s.push_str(&format!("   G3 path length {:.1} m over {} samples\n", len, n));
    // G2: THE RUN MUST START WHERE THE RUN STARTS. The template is a recording
    // of the same map from the same spawn, so its own first sample is the
    // answer key -- free, in the file, and no reference elsewhere needed. This
    // is what separates the car from the other moving objects the engine keeps:
    // measured on the fixture map, one candidate in six traces a perfectly
    // plausible 1.6 km path that is nowhere near the track.
    //
    // AND THE ANSWER KEY IS ONLY AN ANSWER KEY IF THE TEMPLATE STARTS AT RACE
    // ZERO. Compare two samples taken at DIFFERENT race times and the check
    // measures how far the car drove in between: 227654's carrier begins at
    // race 1.310 s doing 66 km/h, so a candidate that is correctly at the spawn
    // reads 11.2 m out and is refused, while a candidate that is 11.2 m down
    // the road passes. That is the check inverted, on the exact class of file
    // this pipeline exists to repair. So compare at a COMMON INSTANT: the
    // candidate's own sample nearest the template's first, and refuse to judge
    // at all when there is no overlap rather than judging against the wrong
    // moment.
    // The reference defaults to the template, which is right when the template
    // is a real recording of this map. It is NOT right when the template is a
    // grid `ghost record rebuild` laid down -- every sample there is a copy of
    // one, so the record does not move and identifies nothing. Name a
    // downloaded recording with `--spawn-ref` for those.
    let spawn_ref = flag(a, "--spawn-ref").unwrap_or(template);
    let ref_moves = gbx::record::decode_ghost(spawn_ref)
        .ok()
        .map(|d| {
            d.samples
                .windows(2)
                .map(|w| {
                    (((w[1].x - w[0].x) as f64).powi(2)
                        + ((w[1].y - w[0].y) as f64).powi(2)
                        + ((w[1].z - w[0].z) as f64).powi(2))
                    .sqrt()
                })
                .sum::<f64>()
                > 5.0
        })
        .unwrap_or(false);
    match gbx::record::decode_ghost(spawn_ref)
        .ok()
        .filter(|_| ref_moves)
        .and_then(|d| d.samples.first().cloned())
    {
        None => s.push_str(
            "   G2 UNMEASURED: the reference record has no motion in it (a rebuilt grid is a \
             constant), so it identifies nothing. Name a downloaded recording of this map with \
             --spawn-ref. The start position has NOT been checked.\n",
        ),
        Some(t0) => {
            let here = gbx::record::decode_ghost(cand)
                .ok()
                .and_then(|d| {
                    d.samples
                        .iter()
                        .min_by_key(|s| (s.time_ms - t0.time_ms).abs())
                        .map(|s| (s.time_ms, [s.x, s.y, s.z]))
                });
            let tol: f64 = flag(a, "--spawn-tol").and_then(|v| v.parse().ok()).unwrap_or(1.0);
            match here {
                Some((t, p)) if (t - t0.time_ms).abs() <= 60 => {
                    let d = (((p[0] - t0.x) as f64).powi(2)
                        + ((p[1] - t0.y) as f64).powi(2)
                        + ((p[2] - t0.z) as f64).powi(2))
                    .sqrt();
                    if d > tol {
                        return Err(format!(
                            "   G2 at {} ms the run is at ({:.2}, {:.2}, {:.2}), {:.1} m from where \
                             the template is at the same instant ({:.2}, {:.2}, {:.2}) -- this is \
                             not the car",
                            t, p[0], p[1], p[2], d, t0.x, t0.y, t0.z
                        ));
                    }
                    s.push_str(&format!(
                        "   G2 at {} ms, {:.3} m from the template at the same instant\n",
                        t, d
                    ));
                }
                _ => s.push_str(&format!(
                    "   G2 UNMEASURED: the template's first sample is at {} ms and this record has \
                     none within 60 ms of it, so there is no common instant to compare. The start \
                     position has NOT been checked.\n",
                    t0.time_ms
                )),
            }
        }
    }
    match verify::tape_record_agreement(cand) {
        None => return Err("   G1 the written file has no telemetry to compare with its tape".into()),
        Some((kappa, rate, lag, k)) => {
            let _ = rate;
            if kappa < 0.60 {
                return Err(format!(
                    "   G1 tape/record agreement kappa {:.3} over {} samples (best lag {} ms) -- the record \
                     written is not this tape's run",
                    kappa, k, lag
                ));
            }
            s.push_str(&format!(
                "   G1 tape/record agreement kappa {:.3} over {} samples (best lag {} ms)\n",
                kappa, k, lag
            ));
        }
    }
    // G4: the plain oracle on THE WRITTEN FILE
    if !has(a, "--no-oracle") {
        let server = crate::oracle::server_dir(flag(a, "--server"));
        if server.join("TrackmaniaServer").exists() {
            match crate::oracle::validate(&server, Path::new(cand), crate::oracle::MapsMode::One(Path::new(map)), "regen") {
                Ok(res) => {
                    let decl = gbx::container::Container::load(cand)
                        .ok()
                        .and_then(|c| c.declared_times().first().map(|x| x.1 as i64));
                    match (res.time_ms, decl) {
                        (Some(t), Some(d)) if t == d => {
                            s.push_str(&format!("   G4 oracle on the written file: {}\n", secs(t)))
                        }
                        (Some(t), Some(d)) if has(a, "--expect-dnf") => s.push_str(&format!(
                            "   G4 oracle: {} against a declared {} -- ACCEPTED, --expect-dnf was \
                             given (the declared time is the carrier's, not this run's)\n",
                            secs(t),
                            secs(d)
                        )),
                        (Some(t), Some(d)) => {
                            return Err(format!(
                                "   G4 oracle re-simulated the written file to {} but it declares {}",
                                secs(t),
                                secs(d)
                            ))
                        }
                        (Some(t), None) => s.push_str(&format!("   G4 oracle: {}\n", secs(t))),
                        // A DNF IS A LEGITIMATE THING TO REGENERATE, AND THE
                        // GATE COULD NOT EXPRESS IT.
                        //
                        // G4 exists so a regenerated file cannot claim a lap it
                        // does not drive. But a tape that never finishes cannot
                        // satisfy it by construction -- and those are exactly
                        // the tapes worth looking at: a failed human attempt, a
                        // search candidate that reaches a state our operators
                        // cannot, a reconstruction from video. Before this flag
                        // the only way to film one was to harvest the
                        // `OUT.tryN.raw` that a REFUSED attempt leaves on disk,
                        // which is a valid ghost the tool deleted its own
                        // pointer to. That is a trap, not a workflow.
                        //
                        // --expect-dnf says "I know this does not finish";
                        // every other gate still runs, and the DNF is REPORTED
                        // rather than waived silently.
                        (None, _) if has(a, "--expect-dnf") => s.push_str(&format!(
                            "   G4 oracle: DNF (cps {:?}) -- ACCEPTED, --expect-dnf was given\n",
                            res.cps
                        )),
                        (None, _) => {
                            return Err(format!(
                                "   G4 oracle: DNF on the written file (cps {:?}). If this tape is \
                                 MEANT not to finish, pass --expect-dnf; the other gates still run.",
                                res.cps
                            ))
                        }
                    }
                }
                Err(e) => s.push_str(&format!("   G4 oracle not run: {}\n", e)),
            }
        } else {
            s.push_str("   G4 oracle not run: no dedicated server\n");
        }
    }
    s.push_str("   GATE PASSED");
    Ok(s)
}

/// The fixed-point control: regenerate a file that already carries its own true
/// telemetry, and require the regeneration to reproduce it.
pub fn control(a: &[String]) {
    let inp = a.first().unwrap_or_else(|| die("ghost regen-control FILE --map MAP"));
    let map = flag(a, "--map").unwrap_or_else(|| die("--map MAP.Map.Gbx"));
    let d0 = gbx::record::decode_ghost(inp)
        .unwrap_or_else(|e| die(format!("{} has no telemetry to control against: {}", inp, e)));
    let out = format!("{}.regen-control", inp);
    let r = run_regen(inp, map, &out, &[]);
    if !r.ok {
        eprintln!("{}", r.log);
        die("the regenerator did not produce a file");
    }
    let d1 = gbx::record::decode_ghost(&out).unwrap_or_else(|e| die(e));
    let n = d0.samples.len().min(d1.samples.len());
    let mut sum = 0.0;
    let mut worst = 0.0f64;
    let mut exact = 0usize;
    for i in 0..n {
        let (a2, b) = (&d0.samples[i], &d1.samples[i]);
        let dd = (((a2.x - b.x) as f64).powi(2)
            + ((a2.y - b.y) as f64).powi(2)
            + ((a2.z - b.z) as f64).powi(2))
        .sqrt();
        sum += dd;
        worst = worst.max(dd);
        if dd == 0.0 {
            exact += 1;
        }
    }
    let mean = if n > 0 { sum / n as f64 } else { f64::NAN };
    println!("fixed-point control on {}", inp);
    println!("  samples        {} original, {} regenerated", d0.samples.len(), d1.samples.len());
    println!("  mean position  {:.6} m", mean);
    println!("  worst          {:.6} m", worst);
    println!("  bit-identical  {} of {}", exact, n);
    // a one-tick offset is a PURE TIME SHIFT: check for it explicitly rather
    // than letting a small mean hide it
    let shifted = |k: i64| -> f64 {
        let mut s = 0.0;
        let mut c = 0usize;
        for i in 0..n {
            let j = i as i64 + k;
            if j < 0 || j >= d1.samples.len() as i64 {
                continue;
            }
            let (a2, b) = (&d0.samples[i], &d1.samples[j as usize]);
            s += (((a2.x - b.x) as f64).powi(2)
                + ((a2.y - b.y) as f64).powi(2)
                + ((a2.z - b.z) as f64).powi(2))
            .sqrt();
            c += 1;
        }
        if c == 0 {
            f64::INFINITY
        } else {
            s / c as f64
        }
    };
    let m1 = shifted(1);
    let mm1 = shifted(-1);
    println!("  one sample early/late: {:.6} m / {:.6} m", mm1, m1);
    if m1 < mean * 0.5 || mm1 < mean * 0.5 {
        println!(
            "  VERDICT: FAIL -- the regeneration is a WHOLE SAMPLE out of phase. \
             A solo clip cannot look wrong from this, but every frame-synchronous comparison is."
        );
        std::process::exit(2);
    }
    if mean.is_finite() && mean < 0.05 {
        println!("  VERDICT: PASS -- the write path is a fixed point on a file that already knew its own answer");
    } else {
        println!("  VERDICT: FAIL -- regenerating a known-good file did not reproduce it");
        std::process::exit(2);
    }
}

pub const CLASS_CSCENEVEHICLEVIS: u32 = 0x0A018000;

/// The vehicle entity's raw sample block: (sample_size, bytes).
pub fn raw_vehicle_samples(path: &str) -> Result<(usize, Vec<u8>), String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let g = gbx::container::Gbx::parse(&data);
    let (version, blob) = gbx::record::find_entrecord_blob(&g.body)?;
    let rd = gbx::record::parse_record_data(&blob, version)?;
    let mut best: Option<&gbx::record::Ent> = None;
    for e in &rd.ents {
        let cid = rd.descs.get(e.type_.max(0) as usize).filter(|_| e.type_ >= 0).map(|d| d.class_id);
        if cid == Some(CLASS_CSCENEVEHICLEVIS) && best.map_or(true, |b| e.times.len() > b.times.len()) {
            best = Some(e);
        }
    }
    let e = best.ok_or("no CSceneVehicleVis entity")?;
    Ok((e.sample_size, e.raw.clone()))
}

/// The recorded steer byte for a tape steer value.
///
/// MEASURED, not assumed: over the project's whole ghost corpus this
/// reproduces byte 14 of every vehicle sample exactly. The mapping is
/// `floor((s + 127) * 255 / 254)` -- note the 254, and note the FLOOR: a round
/// misses s = 60 and s = 0, which is how a "close enough" encoder ends up one
/// grid step out on half the file.
pub fn steer_byte(s: i8) -> u8 {
    (((s as i32 + 127) as i64 * 255) / 254) as u8
}

/// The recorded gas / brake byte for a digital pedal.
pub fn pedal_byte(on: u32) -> u8 {
    if on != 0 {
        255
    } else {
        0
    }
}

/// Run the real engine on the file's OWN tape and compare the trajectory it
/// produces with the trajectory the file records.
///
/// Returns (mean position error, worst, samples compared, one-sample-shift).
/// The shift flag is separate on purpose: a one-tick offset is a PURE TIME
/// SHIFT, so it hides inside a small mean and only shows when the comparison is
/// re-scored against the neighbouring samples.
pub fn engine_trajectory_agreement(path: &str, map: &str) -> Result<(f64, f64, usize, bool), String> {
    let d0 = gbx::record::decode_ghost(path).map_err(|e| e.to_string())?;
    if d0.samples.is_empty() {
        return Err("this file has no telemetry to compare".into());
    }
    // The locate finds the car about one run in eight, so ask for a dozen at
    // once and take the first that identifies it. Voting between runs is not
    // the test -- two wrong picks have agreed with each other to the metre.
    let out = regen_best(path, map, 12, 24)
        .ok_or("the engine readout did not identify the car in 24 attempts")?;
    let d1 = gbx::record::decode_ghost(&out).map_err(|e| e.to_string())?;
    let n = d0.samples.len().min(d1.samples.len());
    let dist = |a: &gbx::record::Sample, b: &gbx::record::Sample| {
        // f64 for the accumulation: the samples are f32 (the precision the
        // record carries), but summing thousands of distances wants the wider
        // accumulator.
        (((a.x - b.x) as f64).powi(2) + ((a.y - b.y) as f64).powi(2) + ((a.z - b.z) as f64).powi(2))
            .sqrt()
    };
    let mut sum = 0.0;
    let mut worst = 0.0f64;
    for i in 0..n {
        let dd = dist(&d0.samples[i], &d1.samples[i]);
        sum += dd;
        worst = worst.max(dd);
    }
    let mean = sum / n.max(1) as f64;
    let shifted = |k: i64| -> f64 {
        let (mut s, mut c) = (0.0, 0usize);
        for i in 0..n {
            let j = i as i64 + k;
            if j < 0 || j >= d1.samples.len() as i64 {
                continue;
            }
            s += dist(&d0.samples[i], &d1.samples[j as usize]);
            c += 1;
        }
        if c == 0 { f64::INFINITY } else { s / c as f64 }
    };
    let shift = mean > 0.02 && (shifted(1) < mean * 0.5 || shifted(-1) < mean * 0.5);
    let _ = std::fs::remove_file(&out);
    Ok((mean, worst, n, shift))
}

/// Write the RECORDED INPUT CHANNELS from the tape.
///
/// `fk regen` rewrites the 22 transform bytes of each sample -- position,
/// orientation, speed, velocity direction -- and leaves the other 94 as the
/// carrier's. Three of those 94 are the recorded steer, gas and brake, and
/// they do not need the engine at all: the TAPE is where they come from. A
/// regenerated file that keeps the carrier's input channels is a file whose
/// analysis tools, overlays and contamination checks all describe somebody
/// else's driving, so this closes them.
///
/// It is not a claim about the other 91 bytes. `ghost regen` names those.
pub fn write_input_channels(inp: &str, out: &str) -> Result<(usize, usize), String> {
    let t = gbx::tape::Tape::from_file(inp)?;
    let a0 = &t.archives[0];
    let so = a0.start_offset_ms as i64;
    let mut written = 0usize;
    let mut skipped = 0usize;
    gbx::recwrite::rewrite_ghost(inp, out, |rd| {
        let mut vehicle: Option<usize> = None;
        for (i, e) in rd.ents.iter().enumerate() {
            let cid = rd.descs.get(e.type_.max(0) as usize).filter(|_| e.type_ >= 0).map(|d| d.class_id);
            if cid == Some(CLASS_CSCENEVEHICLEVIS)
                && vehicle.map_or(true, |v| e.times.len() > rd.ents[v].times.len())
            {
                vehicle = Some(i);
            }
        }
        let Some(vi) = vehicle else { return Err("no CSceneVehicleVis entity".into()) };
        let e = &mut rd.ents[vi];
        let ss = e.sample_size;
        if ss < 20 {
            return Err(format!("vehicle sample size {} is too small for the input bytes", ss));
        }
        for i in 0..e.times.len() {
            let idx = (e.times[i] as i64 - so) / 10;
            if idx < 0 || idx >= a0.packets.len() as i64 {
                skipped += 1;
                continue;
            }
            let p = &a0.packets[idx as usize];
            let d = &mut e.raw[i * ss..(i + 1) * ss];
            d[14] = steer_byte(p.steer_i8());
            d[15] = pedal_byte(p.accel);
            d[18] = pedal_byte(p.brake);
            written += 1;
        }
        Ok(())
    })?;
    Ok((written, skipped))
}

/// Run several regenerations at once and return the first whose result
/// identifies the car: a plausible path length, finite and moving, and a file
/// the dedicated server will still validate. Used by the engine-trajectory
/// check, which needs a trustworthy engine run of a tape and does not care what
/// the file it came from declares.
pub fn regen_best(inp: &str, map: &str, jobs: usize, tries: usize) -> Option<String> {
    let mut round = 0usize;
    while round < tries {
        let batch: Vec<usize> = (round..(round + jobs).min(tries)).collect();
        if batch.is_empty() {
            break;
        }
        let results: Vec<(String, RegenOut)> = std::thread::scope(|sc| {
            let hs: Vec<_> = batch
                .iter()
                .map(|k| {
                    let inp = inp.to_string();
                    let map = map.to_string();
                    let cand = std::env::temp_dir()
                        .join(format!("ghost-v9-{}-{}.Ghost.Gbx", std::process::id(), k))
                        .to_string_lossy()
                        .to_string();
                    sc.spawn(move || {
                        let r = run_regen(&inp, &map, &cand, &[]);
                        (cand, r)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut keep: Option<String> = None;
        for (cand, r) in results {
            let good = r.ok && gate(&cand, map, &[], inp).is_ok();
            if good && keep.is_none() {
                keep = Some(cand);
                continue;
            }
            let _ = std::fs::remove_file(&cand);
        }
        if keep.is_some() {
            return keep;
        }
        round += jobs;
    }
    None
}

/// How many DISTINCT trajectories a batch of gate-passing candidates was, and
/// how far apart they are.
///
/// IT REPORTS, IT DOES NOT CHOOSE. The regenerator measures its clock bias per
/// run and lands a physics tick out some of the time, so a batch of twelve can
/// hold two answers 10 ms apart -- and every one of them passes the gate,
/// because no check in it can see which tick a sample is labelled with. The
/// obvious move is to take the mode, and it is wrong: measured against
/// downloaded recordings the game made itself, the good phase is 4 of 5 runs
/// on 270051 and 279218 but 1 of 5 on 285268 and 279209, so on half the
/// affected maps the majority IS the error. Only a downloaded control of the
/// same map settles which side is the game's -- see `ghost phase`.
///
/// TWO CANDIDATES ARE THE SAME TRAJECTORY IF THEY ARE IN THE SAME PLACE, and
/// that bar is 1 cm. It is not a tolerance on physics: two runs of the same
/// tape through the same engine agree to 0.000031 m when they share a tick and
/// sit 0.13-0.59 m apart when they do not, which is a factor of four thousand.
/// Anything between those is a third thing and shows up as its own group.
pub fn trajectory_census(files: &[String]) -> Option<String> {
    let traj: Vec<Option<Vec<[f64; 3]>>> = files
        .iter()
        .map(|f| {
            gbx::record::decode_ghost(f).ok().map(|d| {
                d.samples.iter().map(|s| [s.x as f64, s.y as f64, s.z as f64]).collect()
            })
        })
        .collect();
    let sep = |a: &Vec<[f64; 3]>, b: &Vec<[f64; 3]>| -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return f64::INFINITY;
        }
        (0..n)
            .map(|i| (0..3).map(|k| (a[i][k] - b[i][k]).powi(2)).sum::<f64>().sqrt())
            .sum::<f64>()
            / n as f64
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, t) in traj.iter().enumerate() {
        let Some(t) = t else { continue };
        match groups.iter_mut().find(|g| {
            traj[g[0]].as_ref().map_or(false, |h| sep(t, h) <= 0.01)
        }) {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    if groups.is_empty() {
        return None;
    }
    groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    let census = groups
        .iter()
        .map(|g| {
            let d = traj[groups[0][0]]
                .as_ref()
                .zip(traj[g[0]].as_ref())
                .map(|(a, b)| sep(a, b))
                .unwrap_or(f64::NAN);
            format!("{} at {:+.4} m", g.len(), d)
        })
        .collect::<Vec<_>>()
        .join(", ");
    let census = if groups.len() == 1 {
        format!(
            "trajectory census: all {} gate-passing attempts produced ONE trajectory",
            files.len()
        )
    } else {
        format!(
            "trajectory census: {} gate-passing attempts produced {} DIFFERENT trajectories ({}), {} apart. \
             The gate cannot see which tick a sample is labelled with, so it passed all of them. \
             Settle it with `ghost phase --control <a human download of this map>`; do NOT take the mode, \
             which is the wrong tick on half the maps where this happens.",
            files.len(),
            groups.len(),
            census,
            groups
                .get(1)
                .and_then(|g| traj[groups[0][0]].as_ref().zip(traj[g[0]].as_ref()).map(|(a, b)| format!("{:.4} m", sep(a, b))))
                .unwrap_or_default()
        )
    };
    Some(census)
}
