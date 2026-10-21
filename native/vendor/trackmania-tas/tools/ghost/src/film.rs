//! `ghost film` -- ONE COMMAND from a search tape to a file the CLIENT WILL LOAD.
//!
//! WHY THIS EXISTS. Every step below was already available, and getting a tape
//! onto the render box still took half an hour of fiddling, four failed
//! renders and two published clips that showed the WRONG CAR. The knowledge
//! was spread over `record rebuild`, `regen`, `record ents`, `record
//! graft-scene`, `record entorder` and `trim`, and each one has a trap that is
//! invisible until a human watches the video:
//!
//!   1. `regen` SILENTLY KEEPS THE DONOR'S TELEMETRY whenever the donor record
//!      covers the new grid -- which, when the lap is shorter than its carrier,
//!      is always. You get V5 PASS and V6 kappa 0.457: it looks like success
//!      and it renders somebody else's driving. The strip must come first.
//!   2. ...but `record rebuild` DROPS the zero-sample `0x032CB000` placeholder
//!      that carries the checkpoint blocks, and the client HARD-CRASHES on the
//!      entity layout that leaves behind. So the scene has to be grafted back.
//!   3. `--car-last` is not an absolute index. Last-of-3 is index 2; a client
//!      that wants 3-of-4 refuses it. The reference file decides, not a flag.
//!   4. A record that RUNS PAST ITS DECLARED TIME is soft-rejected: `0 -> 0`
//!      ghost blocks, dialog `FrameMessage`, no crash, no error, nothing in
//!      the log that says why. This one cost the most.
//!
//! None of those four is checked by `ghost verify`. A tape can pass all eleven
//! gates at kappa 1.000 and still be unloadable, which is why the only honest
//! test is a REFERENCE FILE THE CLIENT IS KNOWN TO ACCEPT -- this command
//! takes one and matches the container to it.
//!
//! Usage:
//!   ghost film IN OUT --map MAP --ref ACCEPTED [--span MS] [--expect-dnf]
//!                     [--spawn-ref GHOST] [--keep-scratch]
//!
//! `--ref` is the load-bearing argument: a ghost of this map that the client
//! has actually imported. Its entity list is the specification.

use crate::{die, flag, has};
use std::path::Path;
use std::process::Command;

fn run(bin: &str, args: &[String], what: &str) -> String {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| die(format!("film: could not run `{}` for {}: {}", bin, what, e)));
    let s = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if !out.status.success() {
        eprintln!("{}", s);
        die(format!("film: {} FAILED -- nothing was written", what));
    }
    s
}

pub fn cmd(a: &[String]) {
    let inp = a.first().cloned().unwrap_or_else(|| {
        die("ghost film IN OUT --map MAP --ref ACCEPTED [--span MS] [--expect-dnf]")
    });
    let out = a
        .get(1)
        .cloned()
        .unwrap_or_else(|| die("film: give an output path"));
    let map = flag(a, "--map").map(String::from).unwrap_or_else(|| die("film: --map is required"));
    // THE REFERENCE IS NOT OPTIONAL AND IT IS NOT A NICETY. Four separate
    // rejections tonight were all "the container differs from one that works",
    // and none of them is derivable from the tape alone.
    let refr = flag(a, "--ref").map(String::from).unwrap_or_else(|| {
        die("film: --ref is required -- name a ghost of this map THE CLIENT HAS \
             IMPORTED. Its entity list is the specification; there is no way to \
             infer it from the tape, and every check `verify` runs passes on files \
             the client refuses.")
    });

    let me = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "ghost".into());
    // Scratch names: strip the output's own extension before appending, so we
    // get `out.film.rg.Ghost.Gbx` and not `out.Ghost.Gbx.film.rg.Ghost.Gbx`.
    // A DOUBLED extension is not cosmetic here -- the server's own name
    // handling keys off the suffix, and the first version of this command
    // produced scratch files it silently ignored. Keep exactly one.
    let stem = out
        .strip_suffix(".Ghost.Gbx")
        .or_else(|| out.strip_suffix(".Replay.Gbx"))
        .unwrap_or(&out)
        .to_string();
    let scratch = format!("{}.film", stem);
    let s = |n: &str| format!("{}.{}.Ghost.Gbx", scratch, n);
    let owned = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();

    // ---- 0. What does the tape actually run to, and what does it claim? ----
    // The declared time comes from the CARRIER and is routinely a lap this run
    // never drives. Step 4 reconciles them; we read it here so the log says so.
    let chain = run(&me, &owned(&["record", "chain", &inp]), "reading the input");
    println!("== film: input\n{}", chain.lines().take(2).collect::<Vec<_>>().join("\n"));

    let span = flag(a, "--span").map(String::from).unwrap_or_else(|| {
        // Default: the file's DECLARED time, not the last sample's timestamp.
        //
        // The record's own end is the last SAMPLE, which lands on the tick
        // before the finish -- 10.600 for a 10.640 run at a 20 ms period. A
        // grid rebuilt to 10600 is 40 ms short of the lap, so the regenerated
        // car stops before the line and the last frames of the film show the
        // finish being crossed by nothing. Measured on 203072.
        //
        // `declared_ms` reads the same number `verify`'s V2 census gates on,
        // which is the run's own time.
        //
        // BUT THE DECLARED TIME CAN BE WRONG, and when it is, everything
        // downstream inherits a grid that is too short: `regen` FILLS the grid
        // it is given, it never extends one. A search-created ghost that
        // declares 21.450 for a 22.750 run produces 430 samples where the
        // engine gathers 459, and the render simply stops before the finish
        // with nothing in the output saying why. Warn, with the flag that
        // fixes it, rather than let the operator discover it in the video.
        let d = declared_ms(&me, &inp);
        if let (Ok(dms), Some(last)) = (d.parse::<i64>(), last_sample_ms(&me, &inp)) {
            // The last sample sits one tick before the finish, so a healthy
            // file has `last` just under `declared`. `last` running PAST the
            // declared time means the declaration is short.
            if last > dms {
                eprintln!(
                    "ghost film: WARNING -- this file declares {} ms but its record runs to \
                     {} ms. The grid is built from the DECLARED time, so everything after \
                     this step will be {} ms short and the render will stop before the \
                     finish. If the declared time is wrong, pass --span {} (the full width \
                     of the run, from its first instant to its last).",
                    dms,
                    last,
                    last - dms,
                    last + 150
                );
            }
        }
        d
    });

    // ---- 1. Strip the donor's grid coverage (trap 1) ------------------------
    println!("== film: 1/6 stripping the donor grid at 20 ms (span {} ms)", span);
    run(
        &me,
        &owned(&["record", "rebuild", &inp, &s("rb"), "--span", &span, "--period", "20"]),
        "record rebuild",
    );

    // ---- 2. Regenerate (trap 1's other half) --------------------------------
    println!("== film: 2/6 regenerating from engine memory");
    let mut rg = owned(&[
        "regen", &s("rb"), &s("rg"), "--map", &map, // carrier and neutralise are unconditional in regen now
        // film built the grid in step 1 from the input's own declared span, so
        // `regen` need not boot an engine to re-derive it. Saves ~3 s of a
        // ~13 s regen; G4 still re-simulates the written file and checks it.
        "--declared-known",
    ]);
    // `regen` is one attempt now (it resolves the car from a pointer chain),
    // so there is no fan-out to pre-empt. This used to run a single try first
    // and fall back to the twelve-way batch.
    if let Some(sr) = flag(a, "--spawn-ref") {
        rg.push("--spawn-ref".into());
        rg.push(sr.to_string());
    }
    if has(a, "--expect-dnf") {
        rg.push("--expect-dnf".into());
    }
    // FORWARD --server. film takes one and uses it for its own oracle calls,
    // but never passed it to the regen it shells out to -- so the sub-regen
    // fell back to a default server path while every standalone control I ran
    // used the explicit one. That makes film's regen a DIFFERENT BINARY's
    // memory layout, which is exactly the kind of difference that turns a good
    // chain into "step 3 is a null pointer".
    if let Some(sv) = flag(a, "--server") {
        rg.push("--server".into());
        rg.push(sv.to_string());
    }
    run(&me, &rg, "regen");

    // ---- 3. BUILD IN THE REFERENCE'S OWN CONTAINER (traps 2, 3, and the
    //         one that crashes the client) --------------------------------
    //
    // This used to graft the reference's SCENE into our rebuilt file. That
    // survives every offline gate and CRASHES THE CLIENT on 287431:
    //
    //     staged 1 ghost(s) into _shoot
    //     read: Connection reset by peer (os error 104)
    //
    // A rebuilt container is not a container the client will load, however
    // carefully its entity list is matched afterwards -- the entity list is
    // not the only thing a container is. The only shape known to import is a
    // GAME-WRITTEN container with our samples inside it, so build there:
    // trim the reference to our tick count, inject our tape, then overwrite
    // its car with ours. Nothing of the donor's run survives that (step 5
    // strips the identity and splits, step 6 proves the car is ours), and
    // everything of the donor's CONTAINER does, which is the point.
    println!("== film: 3/6 building inside {}", refr);
    // RESEGMENT ONLY FOR THE FREEFALL CASE, which is the only one it models.
    //
    // 287431 needs it: that map spawns the car 646 m up and the game records
    // the 2.13 s fall as a SEPARATE vehicle entity, so its reference has
    // exactly 2 car segments and a single-segment file does not match it.
    //
    // Anything else must be left alone. A reference with 1 segment makes
    // `resegment --like` refuse outright ("there is nothing to copy the shape
    // of"), and a human run with RESPAWNS has one segment per life -- 203072's
    // reference has 23, covering 2714 samples against our 213, and copying
    // that shape is not merely useless but impossible. Both maps had worked a
    // commit earlier and were broken by resegmenting unconditionally.
    let ref_segs = car_segments(&me, &refr);
    let our_segs = car_segments(&me, &s("rg"));
    let seg_src = if ref_segs == 2 && our_segs == 1 {
        println!("   reference splits its car in two (a freefall spawn) -- matching that");
        run(
            &me,
            &owned(&["record", "resegment", &s("rg"), &s("seg"), "--like", &refr]),
            "record resegment --like <reference>",
        );
        s("seg")
    } else {
        if ref_segs != our_segs {
            println!(
                "   reference has {} car segment(s), ours has {} -- leaving the shape alone",
                ref_segs, our_segs
            );
        }
        s("rg")
    };
    let tape = format!("{}.tape.gtape", scratch);
    run(&me, &owned(&["tape", "extract", &s("rg"), "--out", &tape]), "tape extract");
    let want_ticks = std::fs::read_to_string(&tape)
        .map(|t| t.lines().filter(|l| l.starts_with("t=")).count())
        .unwrap_or(0);
    if want_ticks == 0 {
        die("film: the extracted tape has no ticks".to_string());
    }

    // Size the reference container to hold exactly our tick count.
    //
    // START FROM THE REFERENCE'S OWN LENGTH, which is always a valid trim, and
    // let `tape inject`'s error do the arithmetic: it reports both tick counts
    // when they disagree, which is an exact measurement of the correction. Two
    // earlier attempts computed the target directly -- a fixed `-151` tuned on
    // one map, then a formula over the reference's span -- and BOTH produced
    // negative windows on other maps ("the window 0.000 .. -1211.440 leaves no
    // ticks"). A measured correction cannot do that: it only ever moves toward
    // a length the container actually has.
    let mut ms = declared_ms(&me, &refr).parse::<i64>().unwrap_or(0);
    if ms <= 0 {
        die("film: could not read the reference's declared time".to_string());
    }
    let mut injected = false;
    for attempt in 0..8 {
        run(&me, &owned(&["trim", &refr, &s("ct"), "--to", &ms.to_string()]), "trim the reference");
        let out2 = Command::new(&me)
            .args(owned(&[
                "tape", "inject", &s("ct"), &s("inj"), "--tape", &tape,
                "--allow-telemetry-mismatch",
            ]))
            .output();
        let (ok, log) = match out2 {
            Ok(o) => (
                o.status.success(),
                format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)),
            ),
            Err(e) => (false, e.to_string()),
        };
        if ok {
            injected = true;
            break;
        }
        // "archive 0: tape has 2768 ticks, <path> has 2769." Parse the two
        // counts by their POSITION RELATIVE TO THE WORD "ticks", never by
        // scraping every number out of the line: the message contains the
        // file path, and these paths are named after the map id, so a plain
        // number scrape read "126859" as a tick count and drove the next trim
        // to -1214.918 s.
        let toks: Vec<&str> = log.split_whitespace().collect();
        let want_n = toks
            .iter()
            .position(|t| t.starts_with("ticks"))
            .and_then(|i| i.checked_sub(1))
            .and_then(|i| toks[i].parse::<i64>().ok());
        let have_n = toks
            .iter()
            .rposition(|t| *t == "has")
            .and_then(|i| toks.get(i + 1))
            .and_then(|t| t.trim_end_matches('.').parse::<i64>().ok());
        match (want_n, have_n) {
            (Some(want), Some(have)) if want != have => {
                ms += (want - have) * 10;
                println!("   container is {} ticks, tape is {} -- retrying at {} ms", have, want, ms);
                if ms <= 0 {
                    eprintln!("{}", log);
                    die("film: sizing the container went negative -- the reference is \
                         shorter than the tape".to_string());
                }
            }
            _ => {
                eprintln!("{}", log);
                die("film: tape inject failed and the error named no tick counts".to_string());
            }
        }
        if attempt == 7 {
            die("film: could not size the reference container to the tape".to_string());
        }
    }
    if !injected {
        die("film: tape inject never succeeded".to_string());
    }
    let decl = declared_ms(&me, &inp);
    run(
        &me,
        &owned(&["declare", &s("inj"), &s("dc"), "--time", &decl]),
        "declare",
    );
    // MAKE THE CONTAINER'S CAR MATCH OUR RUN, not the other way round.
    //
    // After the inject the container still carries the DONOR's car
    // segmentation, and a human's run is segmented by his respawns: ayti's
    // 203072 reference has 23 segments where our lap has 2. Resampling our
    // samples into 23 entities leaves holes -- "25 target instant(s) INSIDE
    // the source's own span have no sample at the same time" -- and nothing is
    // written.
    //
    // It is our run being filmed, so the car shape is ours. Reshape the
    // container to it. This changes the car entities of a GAME-WRITTEN
    // container, which is not the same as rebuilding one: the chunks, the
    // scene and everything the client validates on import stay exactly as the
    // game wrote them.
    // `--all-cars` ONLY WHEN OUR RUN HAS MORE THAN ONE SEGMENT.
    //
    // Plain `resample` writes the longest car entity, which is exactly right
    // when our lap is one continuous segment: the container may carry 22 of
    // them (a human's respawns) and we have no samples for the other 21.
    // `--all-cars` then tries to fill all of them and fails with holes.
    // 287431 is the case that needs it: our lap really is two segments there,
    // because the map's spawn is a 2.13 s freefall the game records
    // separately.
    //
    // Reshaping the container instead does not work: `resegment --like`
    // refuses a single-segment donor -- "there is nothing to copy the shape
    // of" -- because it re-cuts a car, it cannot merge one.
    let src_segs = car_segments(&me, &seg_src);
    let mut rs = owned(&["record", "resample", &s("dc"), &s("rs"), "--from", &seg_src]);
    if src_segs > 1 {
        rs.push("--all-cars".into());
    }
    // FORWARD --mixed-run. The resample guard's own error tells the caller to
    // "pass --mixed-run", and film swallowed it -- so the advice was
    // unfollowable through the pipeline that prints it. 286279 needs it today
    // for a reason that is NOT what the guard is about: `declare` re-encodes
    // the tape and its encoder emits one extra padding byte, so the declared
    // container and the regenerated file differ at payload byte 24 (the
    // bitstream length) while carrying identical inputs -- same 25350 packets,
    // same accel/brake/steer/respawn counts, and the regenerated file matches
    // the SOURCE exactly. Forwarding the flag does not fix that; it makes the
    // documented escape hatch reachable while it is fixed properly.
    if has(a, "--mixed-run") {
        rs.push("--mixed-run".into());
    }
    run(&me, &rs, "record resample");
    let staged = s("rs");
    // ---- 4. The record must not outlive its declared time (trap 4) ----------
    // This is the one that produces `0 -> 0` and `FrameMessage`, with no error
    // anywhere. Reconcile by trimming the record to what the file declares.
    println!("== film: 4/6 reconciling the record with the declared time");
    let trimmed = s("tr");
    run(
        &me,
        &owned(&["trim", &staged, &trimmed, "--to", &decl, "--declare", &decl]),
        "trim to the declared time",
    );

    // ---- 5. NOTHING OF THE DONOR'S MAY SHIP (the traps nobody had numbered) --
    //
    // A container we inherited carries its owner, not just their car. On the
    // YEET donor `ghost inspect` lists: two skin archive paths, a
    // storageObjects locator URL, a ranked badge ("Prestige=Yes&Level=6&..."),
    // the display name "Bonobo.e" and the trigram "DIH". Every clip built on
    // that container published a stranger's identity next to our run, and no
    // gate said a word.
    //
    // The splits are the same defect one layer down. The result chunk carries
    // the DONOR's intermediate checkpoint times; `declare` only ever rewrote
    // the final one, so a file would assert it matched a human's first six
    // checkpoints to the millisecond and then gained three seconds in the last
    // segment. `--cps N` writes 0.000 for the intermediates, which honestly
    // reads as "this file does not know its splits" -- far better than a
    // number that is someone else's.
    println!("== film: 5/6 stripping the donor's identity and splits");
    // Pass --map and --server: without them `identity set` can only PAD the
    // account id to 22 x's, because it cannot prove a shrink is safe. With
    // them it removes the field outright and runs its own oracle no-op
    // control ("20.756 before and after"), so the file loses the donor's
    // account without changing the run.
    let ided = s("id");
    let mut id_args = owned(&[
        "identity", "set", &trimmed, &ided, "--name", "TAS", "--trigram", "TAS", "--skin",
        "default",
        // --anonymise is what clears the REST of the donor: the locator URL,
        // the zone and the club tag. Without it `identity set` rewrites the
        // name, trigram and skin and leaves
        // "https://core.trackmania.nadeo.live/storageObjects/<uuid>",
        // "World|Europe|United Kingdom" and "$F0FITZY" -- the container
        // owner's country and club, sitting on our run. The leak check below
        // is what caught that.
        "--anonymise",
        "--map", &map,
    ]);
    if let Some(sv) = flag(a, "--server") {
        id_args.push("--server".into());
        id_args.push(sv.to_string());
    }
    // `identity set` EXITS NONZERO when there is nothing to change, and an
    // already-anonymous reference (one of our own published files, say) gives
    // exactly that. It is a success for our purposes, so run it permissively
    // and let the leak check below decide -- that check reads the OUTPUT, so
    // it cannot be fooled by which branch ran here.
    let id_ok = Command::new(&me)
        .args(&id_args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let pre_declare = if id_ok && Path::new(&ided).exists() {
        ided.clone()
    } else {
        println!("   identity: nothing to change (already clean)");
        trimmed.clone()
    };
    let cps = cp_count(&me, &refr);
    run(
        &me,
        &owned(&["declare", &pre_declare, &out, "--time", &decl, "--cps", &cps]),
        "declare (blank the donor's splits)",
    );
    let leaked = run(&me, &owned(&["identity", "show", &out]), "identity show");
    // Check the VALUES, not the row labels -- an earlier version of this
    // grepped for "zone" and matched the column header on every file.
    //
    // The zone is deliberately NOT fatal. `--anonymise` leaves it because it
    // is the landmark the trigram and club tag are located by, and `verify`
    // rates it V3 Warn for the same reason; clearing it can break those
    // scanners for a field that names a country and nothing else. Report it
    // and move on.
    let val = |line: &str| line.split('"').nth(1).unwrap_or("").to_string();
    for l in leaked.lines() {
        let v = val(l);
        let fatal = v.contains("storageObjects")
            || v.contains("nadeo.live")
            || v.contains("Prestige=")
            || (l.contains("account id") && !v.is_empty() && !v.starts_with("xxxx"))
            || (l.contains("club tag") && !v.is_empty())
            || (l.contains("display name") && !v.is_empty() && v != "TAS")
            || (l.contains("trigram") && !v.is_empty() && v != "TAS");
        if fatal {
            die(format!(
                "film: {} still carries the donor's identity -- do not publish it.\n{}",
                out, leaked
            ));
        }
    }
    for l in leaked.lines().filter(|l| l.contains("zone")) {
        println!("   note: {} (a country label; --anonymise keeps it as the trigram landmark)", l.trim());
    }

    // ---- 6. Prove it, and say what is NOT proven ---------------------------
    println!("== film: 6/6 verifying");
    // V7 IS THE 2.4 SECONDS HERE, and it is worth them -- but it was being
    // paid for silently. `verify` boots an oracle to re-simulate the FINAL
    // file and check it against its own declared time; everything else in
    // `verify` is instant (measured: 2.41 s with the oracle, 0.00 s with
    // --no-oracle). film was running it and then printing only V5 and V6, so
    // the one check that proves the shipped file still plays was invisible in
    // the output.
    //
    // It is not a duplicate of regen's G4: that ran on regen's output, BEFORE
    // the container was rebuilt, the record resampled and the identity
    // stripped. This is the only check that the thing actually being shipped
    // is still a valid run. So keep it, and SAY it.
    let v = run(&me, &owned(&["verify", &out, "--map", &map]), "verify");
    for l in v.lines().filter(|l| l.contains("V5") || l.contains("V6") || l.contains("V7")) {
        println!("{}", l);
    }
    if v.lines().any(|l| l.contains("V7") && l.contains("FAIL")) {
        die(format!(
            "film: the oracle does not re-simulate {} to its declared time -- the file              does not play as it claims. Do not render it.",
            out
        ));
    }
    carrier_check(&me, &out, Some(&s("rg")));
    let kappa_ok = v.lines().any(|l| l.contains("V6") && l.contains("kappa 1.000"));
    if !kappa_ok {
        die(format!(
            "film: V6 is not kappa 1.000 -- the recording in {} is NOT this tape's run. \
             Do not render it.",
            out
        ));
    }

    if !has(a, "--keep-scratch") {
        for n in ["rb","rg","seg","ct","inj","dc","cs","rs","tr","id","tape"] {
            let _ = std::fs::remove_file(s(n));
        }
    }

    println!(
        "\n== film: {} is ready.\n\
         ==   container matched to {}\n\
         ==   V6 kappa 1.000, record reconciled with its declared time\n\
         == The import is a CLIENT question and no offline check settles it: \
         if it is refused, run `ghost record show` on this file and on the \
         reference and diff the entity list.",
        out, refr
    );
}

/// Which entity index holds the car, read from `record show`'s OWN marker.
///
/// This used to count lines after "lives" in `record chain` and take the
/// second whitespace token, which on a rebuilt grid picks up a word and
/// panics downstream in `record ents` with `ParseIntError` -- a failure two
/// commands away from its cause. `record show` labels the car explicitly
/// ("<- the car this project reads"); parse the label, not a position.
fn car_index(me: &str, f: &str) -> String {
    let c = run(me, &["record".into(), "show".into(), f.into()], "record show");
    c.lines()
        .find(|l| l.contains("the car this project reads"))
        .and_then(|l| {
            l.split_whitespace()
                .nth(1)
                .map(|x| x.trim_end_matches(':').to_string())
        })
        .filter(|x| x.parse::<usize>().is_ok())
        .unwrap_or_else(|| {
            die(format!(
                "film: could not find the car entity in {} -- `record show` printed no \
                 line marked as the car. Run it by hand and look at the entity list.",
                f
            ))
        })
}

/// REFUSE a file whose visual channels never move.
///
/// The bug this exists for: `regen` writing 22 of the 116 sample bytes and
/// zeroing the other 94. The car then reports no speed, no rpm, gear 0 and no
/// ground contact, and the CLIENT DRAWS IT AS A TRANSPARENT WIREFRAME -- or,
/// with the reactor members dead, drives the whole run with the booster
/// unlit. Three clips shipped invisible and one shipped with no reactor, and
/// every one of them passed V2, V5 and V6 at kappa 1.000: kappa compares the
/// tape to the record and has no opinion about whether a car is rendered.
///
/// Bytes 89, 90, 91 and 76 carry the five packed reactor members; a real run
/// moves them constantly. If they hold one value for the whole lap, the
/// carrier did not run, and the only honest thing to do is say so BEFORE a
/// human watches the video.
fn carrier_is_live(me: &str, f: &str) {
    carrier_check(me, f, None)
}

/// Is the shipped car the one the engine authored?
///
/// Two different questions, and only one of them has a right answer offline:
///
///  * "did the ENGINE run?" -- if all four reactor bytes hold one value for
///    the whole lap, the field gather did not happen and the render will show
///    a car with no booster (and, with the other 94 carrier bytes dead, may
///    draw it as a transparent wireframe). That is always fatal.
///  * "did the PIPELINE keep what the engine produced?" -- compare the output
///    against our own regen. Anything the regen drove and the output does not
///    was lost by a later step. Also always fatal, and exact.
///
/// What is NOT fatal is a channel the REFERENCE drives and we do not. The
/// reference is a different human's run: on 287431 ITZYNO1FAN sweeps b76 and
/// our line leaves it at 0 for the whole lap -- and so does our engine-
/// authored regen, so 0 is the truth about our run, not a defect. Comparing
/// against another run can only ever be a hint, and treating it as a gate
/// blocks correct files. It stays as a warning because it is what caught the
/// under-authored reactor on 203072.
fn carrier_check(me: &str, f: &str, control: Option<&str>) {
    let read = |p: &str| -> Vec<Vec<u32>> {
        let d = run(
            me,
            &[
                "record".into(),
                "dumpbytes".into(),
                p.into(),
                "--bytes".into(),
                "76,89,90,91".into(),
            ],
            "record dumpbytes",
        );
        d.lines()
            .filter_map(|l| {
                let v: Vec<u32> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                (v.len() == 5).then_some(v[1..].to_vec())
            })
            .collect()
    };
    let moves = |rows: &[Vec<u32>], i: usize| -> bool {
        rows.len() > 1 && rows.iter().any(|r| r[i] != rows[0][i])
    };

    let rows = read(f);
    if rows.len() < 8 {
        println!("   carrier: only {} sample rows read -- not checked", rows.len());
        return;
    }
    let names = ["b76", "b89", "b90", "b91"];

    // 1. THE ENGINE MUST HAVE RUN.
    if !(0..4).any(|i| moves(&rows, i)) {
        die(format!(
            "film: THE CARRIER DID NOT RUN. Bytes 76, 89, 90 and 91 hold one value for \
             all {} samples of {}, so the reactor and the wheels are frozen: the render \
             will show a car that never fires its booster, and with the rest of the 94 \
             carrier bytes dead the client may draw it as a TRANSPARENT WIREFRAME. That \
             is what `regen` produces when the field gather is skipped -- check that its \
             log mentions the carrier and that `want_fields` is true in regen.rs. Every \
             other gate passes on this file, which is the whole problem.",
            rows.len(),
            f
        ));
    }

    // 2. THE PIPELINE MUST NOT HAVE LOST ANY OF IT.
    if let Some(c) = control {
        let cr = read(c);
        if cr.len() >= 8 {
            let lost: Vec<&str> = (0..4)
                .filter(|i| moves(&cr, *i) && !moves(&rows, *i))
                .map(|i| names[i])
                .collect();
            if !lost.is_empty() {
                die(format!(
                    "film: THE PIPELINE LOST CARRIER DATA. {} move in the engine-authored \
                     regen {} and are frozen in {}, so a step after the regen dropped \
                     them. This is exact -- both files describe the SAME run -- so it is \
                     always a bug here, never a property of the route.",
                    lost.join(", "),
                    c,
                    f
                ));
            }
            println!(
                "PASS carrier: every channel the engine authored survived to the output \
                 ({} samples)",
                rows.len()
            );
            return;
        }
    }
    println!(
        "PASS carrier: {} of 4 reactor channels move across {} samples",
        (0..4).filter(|i| moves(&rows, *i)).count(),
        rows.len()
    );
}


/// The same check with a POSITIVE CONTROL: a byte the reference moves and we
/// do not is under-authored, and no absolute threshold can tell you that.
///
/// Without the control this check can only catch a TOTAL freeze. The reactor
/// bug on 203072 froze three of the four bytes and left one twitching, which
/// reads as "live" on its own and as an obvious defect beside a real
/// recording: the reference swept b76 to 16 and b89 to 73 over the same lap
/// where ours held 0 and 1.
fn cp_count(me: &str, f: &str) -> String {
    let c = run(me, &["inspect".into(), f.into()], "inspect");
    c.lines()
        .find(|l| l.contains("checkpoints, the last is the finish"))
        .and_then(|l| {
            l.split('(')
                .nth(1)
                .and_then(|r| r.split_whitespace().next())
                .map(|x| x.to_string())
        })
        .filter(|x| x.parse::<usize>().is_ok())
        .unwrap_or_else(|| "1".to_string())
}

fn ent_count(me: &str, f: &str) -> usize {
    let c = run(me, &["record".into(), "show".into(), f.into()], "record show");
    c.split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .find_map(|w| (w[1] == "entities").then(|| w[0].parse::<usize>().ok()).flatten())
        .unwrap_or(0)
}

/// The last instant the record actually carries, in ms.
///
/// The counterpart to `declared_ms`, and the reason it exists: those two
/// disagreeing is the failure this file's `--span` default cannot see. A
/// healthy file's last sample lands one tick BEFORE its declared finish; a
/// file whose record runs PAST what it declares has a wrong declaration, and
/// building the grid from the declaration then truncates the run.
///
/// `None` when the record cannot be read at all -- the caller warns only on a
/// number it actually has, rather than guessing from an absence.
fn last_sample_ms(me: &str, f: &str) -> Option<i64> {
    let c = run(me, &["manifest".into(), f.into()], "manifest");
    // "last_ms": 22750  -- the largest across entities is the record's end.
    let mut best: Option<i64> = None;
    for part in c.split("\"last_ms\":").skip(1) {
        let v: String = part
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        if let Ok(n) = v.parse::<i64>() {
            best = Some(best.map_or(n, |b: i64| b.max(n)));
        }
    }
    best
}

fn declared_ms(me: &str, f: &str) -> String {
    let c = run(me, &["inspect".into(), f.into()], "inspect");
    c.lines()
        .find(|l| l.trim_start().starts_with("declared"))
        .and_then(|l| l.split_whitespace().nth(1).map(|x| x.to_string()))
        .and_then(|secs| secs.parse::<f64>().ok().map(|v| format!("{}", (v * 1000.0).round() as i64)))
        .unwrap_or_else(|| die(format!("film: could not read the declared time of {}", f)))
}

/// How many vehicle segments a file's car is split into.
///
/// `record show` lists one line per entity; the vehicle class is 0x0A018000.
/// A map whose spawn is a long fall records the fall as its own segment, so
/// the count is 2 there and 1 almost everywhere else. It decides whether
/// `resegment --like` is needed at all -- running it when the reference has a
/// single segment is refused outright, which broke two maps that had worked.
fn car_segments(me: &str, f: &str) -> usize {
    let c = run(me, &["record".into(), "show".into(), f.into()], "record show");
    // ENTITY lines only. `record show` prints a `desc` line for the class as
    // well as an `ent` line per entity, so counting every mention of
    // 0x0A018000 reported 2 for a single-segment file -- which sent a file
    // with one car into `resegment`, and it refused: "has 1 vehicle segment(s)
    // -- there is nothing to copy the shape of".
    c.lines()
        .filter(|l| l.trim_start().starts_with("ent ") && l.contains("0x0A018000"))
        .count()
}

/// How many ticks a file's input tape carries.
///
/// Used to size a reference container against our tape. `tape stats` prints
/// "<n> ticks"; anything unparseable returns 0 and the caller falls back.
fn tape_ticks(me: &str, f: &str) -> usize {
    let c = run(me, &["tape".into(), "stats".into(), f.into()], "tape stats");
    c.split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .find_map(|w| (w[1].starts_with("ticks")).then(|| w[0].parse::<usize>().ok()).flatten())
        .unwrap_or(0)
}
