//! `ghost` -- the TM2020 ghost/replay API.
//!
//! One binary, one job: every operation this project performs on a `.Ghost.Gbx`
//! or `.Replay.Gbx`, each with a control that proves it did what it says.
//!
//! Rust only. There is no interpreter anywhere in this pipeline and no shell
//! script carries any logic.

use ghost::cli::{die, flag, has, need, num};
use gbx::container::{secs, set_embedded_map, Container};
use ghost::regen::raw_vehicle_samples;
use gbx::tape::{Encoding, Tape};
use gbx::{container, tape};
use ghost::{census, declare, engine, hdr, ident, map_uid_of, phase, record, regen, roundtrip, selftest, splice, synth, trim, verify};

const HELP: &str = r#"ghost -- the TM2020 ghost / replay API

  Every command that WRITES a file runs a control first and refuses rather than
  writing something that would be silently wrong. Times print as seconds.

INSPECT
  ghost inspect FILE [--ticks]
        Everything the file says about itself: container kind, the map it will
        actually run on, declared time and every copy of it, splits, identity,
        the input tape, and the telemetry record.
  ghost chunks FILE
        The skippable-chunk table, for forensics.
  ghost manifest FILE
  ghost manifest diff LEFT RIGHT
        Deterministic JSON: GBX header/chunks, packet-mode histogram and payload
        shapes, full record grammar, and the first controlled-car sample.

INPUTS  (operation 1 and 2)
  ghost tape extract FILE --out TAPE.gtape
        Full fidelity: every tick, every field the packet carries -- state word,
        respawn bit, mouse axes, steer / accel / brake, trigger fields, mode and
        flags. Round-trips byte-for-byte.
  ghost tape inject IN OUT --tape TAPE.gtape [--verbatim] [--allow-telemetry-mismatch]
        Write a tape back into a ghost. Default writes every vehicle field
        EXPLICITLY so no tick inherits the previous one's inputs; --verbatim
        reproduces the file's original coding exactly.
  ghost tape expand IN OUT [--state]
        Rewrite every "same as the previous tick" packet in its explicit form.
        Semantically a no-op -- and the oracle control says so -- but afterwards
        every tick is writable.
  ghost tape diff A.gtape B.gtape
        Per-tick differences between two tapes.
  ghost tape poke IN --out T.gtape --tick-range A..B --set steer=..,accel=..,brake=..
        Override the vehicle inputs over a half-open tick range and leave every
        other tick identical: the one-variable probe -- a brake pulse, a lifted
        throttle, a unit of steer -- without hand-editing `t=` lines, where an
        off-by-one is invisible and lands in a measurement.
  ghost tape stats TAPE.gtape
        Tick count, input events, packet modes, respawns.
  ghost tape csv FILE|TAPE.gtape [--out F.csv] [--from MS] [--to MS]
        The tape as `race_ms,steer,accel,brake`, one row a tick -- the format
        every map page publishes under `inputs/`. Those files had no writer in
        the repo, so a page's inputs could not be regenerated from the ghost
        they came from, or checked against it.
  ghost tape bits FILE... [--events]
        Which bits of the state literal actually vary across a corpus; --events
        also prints each explicit literal's tick and decoded state, never inputs.

CAR STATE  (operation 3)
  ghost regen IN OUT --map MAP [--anchorticks a,b,c] [--noanchor]
        Run the real engine on this file's own inputs, capture per-sample car
        state and write it into the ghost, so the recorded trajectory MATCHES
        the tape. Refuses unless the acceptance gate passes.
        Every sample byte the engine can source is AUTHORED, from the game's
        own writer transcribed at 0x9cfed0: the transform, and the 94 other
        bytes (speed, rpm, the four wheels, gear, dampers, ground contact, the
        reactor). 11 channels the dedicated server cannot source stay the
        carrier's and the run names them; `ghost record countdown` authors four
        more of those from the file's own race clock.
        There is no flag for any of this. `--neutralise`, `--inputs`,
        `--trim-outside` and `--carrier` are UNCONDITIONAL and are not options:
        passing them changes nothing and there is no run without them. That
        cost a day on 287431 -- `--neutralise` was tested on and off, the two
        outputs were byte-identical (md5 b9a8bc3a...), and the byte hypothesis
        was eliminated on the strength of a flag that was already on. A regen
        without the carrier draws the car as a TRANSPARENT WIREFRAME in the
        game client while gating at V1-V11 and kappa 1.000; three clips shipped
        that way before anyone looked at a frame.
        --spawn-ref FILE names the recording G2 checks the start against, for a
        template whose own record is a rebuilt grid (a constant identifies
        nothing, and G2 says UNMEASURED rather than passing).
  ghost roundtrip GHOST --map MAP [--bar MM] [--keep]
        The end-to-end control: regenerate a recording the game itself wrote,
        from its own inputs, and require its own trajectory back. The answer key
        is in the file and nothing about it can be tuned. Floor 0.48-0.52 mm
        (client vs dedicated server); default bar 5 mm.
  ghost regen-control FILE --map MAP
        The fixed-point control: regenerate a ghost that already carries its own
        true telemetry and require the result to reproduce it.

MAP  (operation 4)
  ghost map show FILE
        Which map this file will actually run on -- and whether --map is real
        for it at all.
  ghost map extract FILE --out MAP.Map.Gbx
  ghost map set IN OUT --map MAP.Map.Gbx
        Replace the CARRIED map. This is the only thing that moves a recording
        onto another map: rewriting the uid does not.
  ghost unwrap IN.Replay.Gbx OUT.Ghost.Gbx
        The CGameCtnGhost a replay carries, as a standalone ghost: the node is
        sliced out byte-for-byte and its telemetry node re-indexed (2 -> 1);
        the control requires that four-byte difference and nothing else, and
        the same tape, result and declared times back. Built for the client's
        MediaTrackerCache/MTAuthorGhost<map name>.Ghost.gbx files, which are
        replays (map + the run the player finished in the editor) despite
        the extension.
  ghost lcp FILE.LaunchedCP.gbx [--json OUT] [--csv OUT] [--samples]
        The editor's launched-checkpoints cache (ProgramData\Trackmania\
        LaunchedCheckpointsCache\<map name>.LaunchedCP.gbx), written at every
        checkpoint crossing in test mode: per checkpoint reached, the car's
        full state at the crossing and the last ~1.5 s of approach samples
        (inputs included) -- the route skeleton of a run that never finished.

TRIM  (operation 5)
  ghost trim IN OUT [--from MS] [--to MS] [--declare MS]
        Set the run's WINDOW, in race-time milliseconds. Cut the head and/or
        the tail, keeping the file coherent: inputs, telemetry samples, the
        record span, the splits and every copy of the declared time. A --to
        PAST the end of the tape LENGTHENS it instead, appending ticks that
        hold the last recorded input -- room for the car to keep driving after
        the recording stopped. One command owns a run's length in both
        directions.

RECORD
  ghost record chain FILE
        Which entity is the DRIVER, on a record that holds more than one car:
        every tiling chain and the path each one drove. The stock reader takes
        the `CSceneVehicleVis` entity with the MOST SAMPLES, which is right on a
        solo ghost and wrong on a server recording -- there the driver's car is
        destroyed and recreated at every respawn, so the driver is 46 short
        entities tiling the race and the longest single entity is another
        player. Path length is what separates them: a spectator car parked at
        the spawn tiles perfectly for the whole race and travels zero metres.

SPLICE  (operation 5b -- the MIDDLE of a run, where trim owns the two ends)
  ghost splice IN OUT [--rule retries] [--drop A..B,...] [--keep-record]
                      [--map MAP] [--server DIR]
        Delete whole intervals out of the middle of a run and CLOSE THE GAP:
        the tape, the telemetry, the splits and every copy of the declared time
        are retimed together. `--rule retries` derives the intervals from the
        file itself -- for every checkpoint segment that contains a respawn
        press, delete that segment from its start through its LAST respawn
        press, i.e. every failed attempt, keeping the one that reached the
        checkpoint. Nothing else is cut and no surviving tick is edited, so the
        new time is exactly `finish - 10 ms x deleted ticks` and the file
        carries ZERO respawn ticks (the gate asserts it).
        --drop deletes extra intervals given in the ORIGINAL file's race ms;
        that is how a de-looped artefact is built, from a loop census the
        caller has to state.
        A SPLICE IS NOT A LAP. The car's state jumps at every junction; the
        command measures each jump off the file's own recording and prints it,
        and the oracle is not expected to return the declared time.

DECLARE
  ghost declare IN OUT (--time MS | --from-oracle --map M) [--splits MS,MS,...] [--cps N]
        Set the time the file DECLARES, in every copy of it, and in the
        ghost-result chunk. --cps N also sets the NUMBER of checkpoint entries,
        which a container borrowed from another map gets wrong: it declares the
        DONOR map's checkpoint list. MEASURED, so that nobody re-derives it from
        the tool: the dedicated server does NOT gate on this count -- a file
        declaring 1, 2, 3, 5 or 9 checkpoints on a 4-checkpoint map still
        validates, and the mismatch shows up only as ValidatedResult and
        DeclaredResult disagreeing. What the count breaks is THIS toolchain:
        `tmmaps` refuses to build a segment map against a ghost whose split
        count is not the map's, because every rung would be verified against
        the wrong checkpoint.

IDENTITY  (operation 6)
  ghost identity show FILE
  ghost identity set IN OUT [--name N] [--trigram XXX] [--skin PATH|default]
                            [--login L] [--zone Z] [--clubtag T] [--anonymise]
                            [--pad-ids] [--allow-noop]
        Car skin, display name and 3-letter trigram. --anonymise also drops the
        account id and the storage locator URL, which are the two foreign
        identifiers a strip-list usually misses -- AND the driver block in the
        replay header and in body chunk 0x03093018, which are the two a
        body-only anonymiser misses. --pad-ids keeps the old behaviour of
        padding an unresizable id to `xxxx...` instead of removing it.
        --allow-noop writes the output even when nothing in the body needed
        changing, for a caller that wants the STATE (this file is anonymous)
        rather than the EDIT. By hand that case is almost always the wrong
        file, so without the flag it still refuses.
  ghost header show FILE
        The replay HEADER: its chunk table, the driver fields, the map's own
        attribution (which stays), and every copy of the race time the header
        holds -- the copies the body census cannot see. A plain .Ghost.Gbx has
        no replay header and says so.

VERIFY
  ghost verify FILE [--map MAP] [--expect-ms MS] [--server DIR] [-o json]
        The acceptance gate: codec identity, tape/telemetry agreement, declared
        time census, container identity, and the plain oracle re-simulating THE
        WRITTEN FILE. `-o json` emits the same report for a caller; the verdict
        and the exit code are identical either way.
  ghost selftest [--server DIR] [--data DIR]
        The whole test suite, from one command.

PRODUCE
  ghost film IN OUT --map MAP [--mixed-run] [--server DIR]
        The full pipeline: regenerate, resegment and finish a ghost the game
        will load. Emits the two-car-entity shape a freefall spawn needs.
  ghost synth IN OUT [--zero-hashes]
        Rebuild a container from PARSED VALUES rather than copying bytes:
        every field re-emitted from what it means. Byte-identical on the whole
        reference corpus (45 of 45).

READ
  ghost census FILE          every copy of the declared time, body and header
  ghost phase FILE           the run's phase structure
  ghost dump FILE            the raw body, for eyes
  ghost engine FILE --map M  the engine's own trajectory for this file

DEBUG (`ghost debug --help`)
        Forensic probes. They answer one question each, carry no compatibility
        promise, and are listed separately so they cannot be mistaken for the
        tool's surface.
"#;

/// The probes, and what each one was built to answer.
///
/// These are kept because the questions recur, and documented because an
/// undocumented probe is indistinguishable from dead code -- six of these were
/// invisible in `ghost --help` until the release pass went looking.
const DEBUG_HELP: &str = r#"ghost debug -- forensic probes

  No compatibility promise. Each of these was built to answer one question,
  and each is kept because that question recurs.

  ghost debug split-car IN OUT --at MS
        Split the car entity in two at a tick boundary. THE REPAIR for a
        freefall recording the client refuses: 287431's run in one car entity
        crashes the game on import, and the same samples in two load fine.
        `ghost film` already emits the right shape; this fixes a raw recording.

  ghost debug swap-samples IN OUT --donor D
        Keep IN's container and entity list, take D's car samples. Separates
        "the bytes are wrong" from "the wrapper is wrong" in one run -- it is
        what proved 287431's crash lives in the container, not the telemetry.

  ghost debug car-first IN OUT
        Move the car entity to index 0. Tests whether entity ORDER matters
        (it does not -- measured, and the theory died).

  ghost debug set-uid IN OUT --uid U
        Rewrite the map uid the ghost declares (27 chars, same length): pairs with
        `tmmaps setuid` for an A/B on a re-uided copy of a map.

  ghost debug keep-ents IN OUT --keep I,J,K
        Keep only the named entity indices (as `ghost manifest` lists them) and
        drop the rest: the repair for a regenerated ghost whose multi-client
        carrier left other players' 0x2D001000 / 0x032CB000 streams in the
        record (tiny 11: 7 entities; the client imports 0 -> 0, FrameMessage).

  ghost debug set-u01 IN OUT --value N
        Rewrite the car entity's u01 word.

  ghost debug strip-events IN OUT --type N
        Drop every deltas2 record of one type from the middle entity.

  ghost debug codeccheck TAPE
        Does a verbatim re-encode of every tick reproduce the file's own
        bitstream?

  ghost debug trajdiff A B
        Sample-by-sample trajectory difference between two files.
"#;

fn main() {
    { gbx::container::lzo_init(); };
    // --version / -V. Compile-time only: CARGO_PKG_* come from the crate's
    // Cargo.toml (which inherits the one workspace version), and TAS_BUILD is
    // the git hash the release build sets. option_env! means an ordinary
    // `cargo build` still works and simply reports "dev". No dependency.
    if std::env::args().any(|x| x == "--version" || x == "-V") {
        println!(
            "{} {} ({})",
            option_env!("CARGO_BIN_NAME").unwrap_or(env!("CARGO_PKG_NAME")),
            env!("CARGO_PKG_VERSION"),
            option_env!("TAS_BUILD").unwrap_or("dev")
        );
        std::process::exit(0);
    }
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() || a[0] == "-h" || a[0] == "--help" || a[0] == "help" {
        println!("{}", HELP);
        std::process::exit(if a.is_empty() { 2 } else { 0 });
    }

    // TWO TIERS: the product verbs, and the forensic probes under `ghost debug`.
    //
    // `ghost` grew 30 top-level verbs, and 13 of them appear NOWHERE in the
    // help -- including `film` and `synth`, which are real capabilities, and
    // six probes built to answer one question each during an investigation.
    // A flat namespace where a third of the entries are undiscoverable is not
    // a namespace, and a probe sitting beside `verify` reads like a promise.
    //
    // The probes keep working under their old names for one release and say
    // where they moved. Nothing is removed here; what changes is what the help
    // presents as the tool's surface.
    const PROBES: &[&str] = &[
        "codeccheck",
        "swap-samples",
        "car-first",
        "keep-ents",
        "set-uid",
        "split-car",
        "set-u01",
        "strip-events",
        "trajdiff",
    ];
    let (sub, rest, via_debug) = if a[0] == "debug" {
        if a.len() < 2 {
            println!("{}", DEBUG_HELP);
            std::process::exit(2);
        }
        if a[1] == "-h" || a[1] == "--help" {
            println!("{}", DEBUG_HELP);
            std::process::exit(0);
        }
        (a[1].as_str(), &a[2..], true)
    } else {
        (a[0].as_str(), &a[1..], false)
    };
    if !via_debug && PROBES.contains(&sub) {
        eprintln!(
            "ghost: `{sub}` is a forensic probe and has moved to `ghost debug {sub}`; \
             the old spelling still works for this release. `ghost debug --help` lists them."
        );
    }
    if via_debug && !PROBES.contains(&sub) {
        die(format!(
            "`{sub}` is not a debug probe -- run it as `ghost {sub}`. \
             `ghost debug --help` lists the probes."
        ));
    }
    match sub {
        "inspect" => cmd_inspect(rest),
        "codeccheck" => {
            let t = Tape::from_file(&rest[0]).unwrap_or_else(|e| die(e));
            match t.verbatim_is_identity() {
                Ok(()) => println!("codec identity OK ({} ticks)", t.n()),
                Err(e) => {
                    println!("codec identity FAILED: {}", e);
                    if let Some(k) = t.first_divergent_packet(0) {
                        println!("first divergent packet: {}", k);
                        for i in k.saturating_sub(2)..(k + 3).min(t.archives[0].packets.len()) {
                            println!("  {}", {
                                let p = &t.archives[0].packets[i];
                                format!("t={} {:?}", i, p)
                            });
                        }
                    }
                    std::process::exit(1);
                }
            }
        }
        "trajdiff" => {
            // Compare two files' recorded trajectories, at every shift from
            // -3 to +3 samples. A one-sample offset is a PURE TIME SHIFT and
            // hides inside a small mean, so the shift is always reported.
            let a0 = gbx::record::decode_ghost(&rest[0]).unwrap_or_else(|e| die(e));
            let b0 = gbx::record::decode_ghost(&rest[1]).unwrap_or_else(|e| die(e));
            let n = a0.samples.len().min(b0.samples.len());
            println!("{} vs {}  ({} / {} samples)", rest[0], rest[1], a0.samples.len(), b0.samples.len());
            for k in -3i64..=3 {
                let (mut s, mut c, mut worst) = (0.0f64, 0usize, 0.0f64);
                for i in 0..n {
                    let j = i as i64 + k;
                    if j < 0 || j >= b0.samples.len() as i64 { continue }
                    let (p, q) = (&a0.samples[i], &b0.samples[j as usize]);
                    let d = (((p.x - q.x) as f64).powi(2) + ((p.y - q.y) as f64).powi(2) + ((p.z - q.z) as f64).powi(2)).sqrt();
                    s += d; worst = worst.max(d); c += 1;
                }
                if c > 0 {
                    println!("  shift {:+}: mean {:.6} m  worst {:.6} m  over {} samples", k, s / c as f64, worst, c);
                }
            }
        }
        "engine" => engine::cmd(rest),
        "manifest" => cmd_manifest(rest),
        "chunks" => cmd_chunks(rest),
        "samples" => cmd_samples(rest),
        "dump" => {
            let c = Container::load(&rest[0]).unwrap_or_else(|e| die(e));
            let at = num(rest, "--at").unwrap_or(0) as usize;
            let len = num(rest, "--len").unwrap_or(256) as usize;
            let b = &c.body()[at..(at + len).min(c.body().len())];
            for (i, row) in b.chunks(16).enumerate() {
                let hex: Vec<String> = row.iter().map(|x| format!("{:02x}", x)).collect();
                let asc: String = row
                    .iter()
                    .map(|x| if (0x20..0x7f).contains(x) { *x as char } else { '.' })
                    .collect();
                println!("{:>8}  {:<48}  {}", at + i * 16, hex.join(" "), asc);
            }
        }
        "tape" => cmd_tape(rest),
        "map" => cmd_map(rest),
        "trim" => trim::cmd(rest),
        "static" => ghost::statik::cmd(rest),
        "splice" => splice::cmd(rest),
        "synth" => synth::cmd(rest),
        "unwrap" => ghost::unwrap::cmd(rest),
        "lcp" => ghost::lcp::cmd(rest),
        // `ghost strip-events IN OUT --type N` -- drop every deltas2 record of
        // one type from the middle entity. Built to answer ONE question: why
        // does 287431's ghost kill the client on any map (17a29c8)? It carries
        // 7 type-80 records where every other ghost in the corpus carries 0 or
        // 1, and 294446 (which has 1) imports fine.
        // `ghost set-u01 IN OUT --value N` -- rewrite the car entity's u01.
        // 287431's crashing file has 0x02000009 on its single car entity; every
        // ghost that imports has 0x02000006 on the FIRST car entity. One word,
        // so it is worth testing on its own before the entity split.
        // `ghost split-car IN OUT --at MS` -- split the single car entity into
        // two at a tick boundary, the shape `ghost film` produces.
        //
        // This is the LAST difference between 287431's crashing original
        // ([415,0,416], one car) and its film, which imports ([415,0,44,374],
        // two cars split at the 2.13 s freefall handover). Everything else has
        // been tested and cleared (297fd64, f763c27).
        // `ghost car-first IN OUT` -- move the car entity to the head of the
        // entity list. 294446 imports with ONE car entity and the car FIRST;
        // 287431 crashes with one car entity and the car LAST. The split that
        // "fixed" 287431 also put a car earlier in the list, so order is the
        // confound and this isolates it.
        // `ghost swap-samples IN OUT --donor D` -- keep IN's container and entity
        // structure, but fill its car entity with DONOR's car samples
        // (truncated/padded to IN's sample count). Decides whether 287431's
        // crash lives in the SAMPLE BYTES or in the container: 294446's samples
        // are known to load, so if 287431 wearing them still crashes, the
        // sample data is innocent.
        "set-uid" => {
            // The map uid a ghost declares, rewritten in place: every 27-character
            // length-prefixed uid literal in the body becomes --uid (same length,
            // so nothing else moves). For A/Bs against a re-uided map copy
            // (`tmmaps setuid`): the client refuses a ghost whose uid is not the
            // loaded map's.
            let inp = rest.first().unwrap_or_else(|| die("ghost debug set-uid IN OUT --uid U (27 chars)"));
            let outp = rest.get(1).unwrap_or_else(|| die("ghost debug set-uid IN OUT --uid U (27 chars)"));
            let uid = flag(rest, "--uid").unwrap_or_else(|| die("--uid U (27 chars)"));
            if uid.len() != 27 || !uid.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                die(format!("--uid `{}`: a map uid is 27 ASCII chars of [A-Za-z0-9_-]", uid));
            }
            let data = std::fs::read(inp).unwrap_or_else(|e| die(format!("{}: {}", inp, e)));
            let g = gbx::Gbx::parse(&data);
            let mut body = g.body.clone();
            let mut n = 0usize;
            let mut was = Vec::new();
            let mut i = 0usize;
            while i + 31 <= body.len() {
                let len = u32::from_le_bytes(body[i..i + 4].try_into().unwrap());
                if len == 27 {
                    if let Ok(s) = std::str::from_utf8(&body[i + 4..i + 31]) {
                        if s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                            was.push(s.to_string());
                            body[i + 4..i + 31].copy_from_slice(uid.as_bytes());
                            n += 1;
                            i += 31;
                            continue;
                        }
                    }
                }
                i += 1;
            }
            if n == 0 {
                die("no uid literal in the body");
            }
            let mut file = g.header_bytes_u();
            file.extend_from_slice(&body);
            std::fs::write(outp, &file).unwrap_or_else(|e| die(format!("{}: {}", outp, e)));
            println!("{} uid literal(s) {:?} -> {} in {}", n, was, uid, outp);
        }
        "swap-samples" => {
            // IN OUT, with every other input a named flag -- the convention
            // `trim`, `splice`, `declare`, `regen`, `split-car` and the rest
            // all follow. This command shipped as `IN DONOR OUT`, three
            // positionals with the donor in the middle, which is the sort of
            // signature you have to look up every time.
            let inp = rest
                .first()
                .unwrap_or_else(|| die("ghost swap-samples IN OUT --donor D.Ghost.Gbx"));
            let outp = rest
                .get(1)
                .unwrap_or_else(|| die("ghost swap-samples IN OUT --donor D.Ghost.Gbx"));
            let donor = flag(rest, "--donor")
                .unwrap_or_else(|| die("ghost swap-samples IN OUT --donor D.Ghost.Gbx"));
            // Read the donor's car samples by running the same rewrite over it
            // and capturing them; `decode_ghost` only reports summaries.
            let mut draw: Vec<u8> = Vec::new();
            let tmp = format!("{}.donor-read", outp);
            gbx::recwrite::rewrite_ghost(donor, &tmp, |rd| {
                let e = rd
                    .ents
                    .iter()
                    .find(|e| e.sample_size == 116)
                    .ok_or("donor has no car entity")?;
                draw = e.raw.clone();
                Ok(())
            })
            .unwrap_or_else(|e| die(format!("swap-samples: reading donor: {}", e)));
            let _ = std::fs::remove_file(&tmp);
            let mut note = String::new();
            match gbx::recwrite::rewrite_ghost(inp, outp, |rd| {
                let e = rd
                    .ents
                    .iter_mut()
                    .find(|e| e.sample_size == 116)
                    .ok_or("no car entity")?;
                let n = e.times.len();
                let ss = e.sample_size;
                let mut raw = Vec::with_capacity(n * ss);
                for i in 0..n {
                    let src = (i % (draw.len() / ss)) * ss;
                    raw.extend_from_slice(&draw[src..src + ss]);
                }
                note = format!("{} samples replaced from {}", n, donor);
                e.raw = raw;
                Ok(())
            }) {
                Ok(_) => println!("{} -> {}", note, outp),
                Err(e) => die(format!("swap-samples: {}", e)),
            }
        }
        "car-first" => {
            let inp = rest.first().unwrap_or_else(|| die("ghost car-first IN OUT"));
            let outp = rest.get(1).unwrap_or_else(|| die("ghost car-first IN OUT"));
            match gbx::recwrite::rewrite_ghost(inp, outp, |rd| {
                let idx = rd
                    .ents
                    .iter()
                    .position(|e| e.sample_size == 116)
                    .ok_or("no car entity")?;
                let car = rd.ents.remove(idx);
                rd.ents.insert(0, car);
                Ok(())
            }) {
                Ok(_) => println!("moved the car entity to index 0 -> {}", outp),
                Err(e) => die(format!("car-first: {}", e)),
            }
        }
        "keep-ents" => {
            // A regenerated ghost whose CARRIER was a multi-client server
            // recording keeps every other player's 0x2D001000 / 0x032CB000
            // streams (tiny 11, 2026-09-09: 7 entities, three players' ids), and
            // the client answers the import with 0 -> 0 ghost blocks and a
            // FrameMessage -- the film.rs soft refusal. Keep the named entity
            // indices (the car and ITS two streams), drop the rest; the car's
            // samples are untouched.
            let inp = rest.first().unwrap_or_else(|| die("ghost debug keep-ents IN OUT --keep I,J,K"));
            let outp = rest.get(1).unwrap_or_else(|| die("ghost debug keep-ents IN OUT --keep I,J,K"));
            let keep: Vec<usize> = flag(rest, "--keep")
                .unwrap_or_else(|| die("--keep I,J,K (entity indices as `ghost manifest` lists them)"))
                .split(',')
                .map(|s| s.trim().parse::<usize>().unwrap_or_else(|_| die(format!("--keep: `{}` is not an index", s))))
                .collect();
            let mut note = String::new();
            match gbx::recwrite::rewrite_ghost(inp, outp, |rd| {
                let n = rd.ents.len();
                for k in &keep {
                    if *k >= n {
                        return Err(format!("--keep {}: the record has {} entities", k, n));
                    }
                }
                let kept: Vec<_> = rd
                    .ents
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| keep.contains(i))
                    .map(|(_, e)| e.clone())
                    .collect();
                note = format!(
                    "{} of {} entities kept ({}); dropped {}",
                    kept.len(),
                    n,
                    kept.iter().map(|e| format!("type {} u01 {} samples {}", e.type_, e.u01, e.times.len())).collect::<Vec<_>>().join("; "),
                    rd.ents.iter().enumerate().filter(|(i, _)| !keep.contains(i)).map(|(_, e)| format!("type {} u01 {}", e.type_, e.u01)).collect::<Vec<_>>().join("; ")
                );
                rd.ents = kept;
                Ok(())
            }) {
                Ok(_) => println!("{} -> {}", note, outp),
                Err(e) => die(format!("keep-ents: {}", e)),
            }
        }
        "split-car" => {
            let inp = rest.first().unwrap_or_else(|| die("ghost split-car IN OUT --at MS"));
            let outp = rest.get(1).unwrap_or_else(|| die("ghost split-car IN OUT --at MS"));
            let at: i32 = flag(rest, "--at")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| die("--at MS"));
            let mut note = String::new();
            match gbx::recwrite::rewrite_ghost(inp, outp, |rd| {
                let idx = rd
                    .ents
                    .iter()
                    .position(|e| e.sample_size == 116 && e.times.len() > 1)
                    .ok_or("no car entity with 116-byte samples")?;
                let e = &rd.ents[idx];
                let cut = e.times.iter().position(|t| *t >= at).ok_or("no tick at --at")?;
                if cut == 0 || cut >= e.times.len() {
                    return Err("--at is outside the run".into());
                }
                let ss = e.sample_size;
                let mut a = e.clone();
                let mut b = e.clone();
                a.times = e.times[..cut].to_vec();
                a.raw = e.raw[..cut * ss].to_vec();
                b.times = e.times[cut..].to_vec();
                b.raw = e.raw[cut * ss..].to_vec();
                note = format!("{} + {} samples at {} ms", a.times.len(), b.times.len(), at);
                rd.ents[idx] = a;
                rd.ents.insert(idx + 1, b);
                Ok(())
            }) {
                Ok(_) => println!("split the car entity: {} -> {}", note, outp),
                Err(e) => die(format!("split-car: {}", e)),
            }
        }
        "set-u01" => {
            let inp = rest.first().unwrap_or_else(|| die("ghost set-u01 IN OUT --value N"));
            let outp = rest.get(1).unwrap_or_else(|| die("ghost set-u01 IN OUT --value N"));
            let want: i32 = flag(rest, "--value")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| die("--value N"));
            let mut hit = 0usize;
            match gbx::recwrite::rewrite_ghost(inp, outp, |rd| {
                for e in rd.ents.iter_mut() {
                    if e.u01 != 0 {
                        e.u01 = want;
                        hit += 1;
                    }
                }
                Ok(())
            }) {
                Ok(_) => println!("set u01 = {} on {} entity(ies) -> {}", want, hit, outp),
                Err(e) => die(format!("set-u01: {}", e)),
            }
        }
        "strip-events" => {
            let inp = rest.first().unwrap_or_else(|| die("ghost strip-events IN OUT --type N"));
            let outp = rest.get(1).unwrap_or_else(|| die("ghost strip-events IN OUT --type N"));
            let want: i32 = flag(rest, "--type")
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| die("--type N"));
            let mut dropped = 0usize;
            let r = gbx::recwrite::rewrite_ghost(inp, outp, |rd| {
                for e in rd.ents.iter_mut() {
                    let before = e.deltas2.len();
                    e.deltas2.retain(|(t, _, _)| *t != want);
                    dropped += before - e.deltas2.len();
                }
                Ok(())
            });
            match r {
                Ok(_) => println!("dropped {} deltas2 record(s) of type {} -> {}", dropped, want, outp),
                Err(e) => die(format!("strip-events: {}", e)),
            }
        }
        "declare" => declare::cmd(rest),
        "identity" => ident::cmd(rest),
        "header" => {
            let what = rest.first().map(|s| s.as_str()).unwrap_or("show");
            match what {
                "show" => hdr::cmd_show(rest.get(1).map(|s| s.as_str()).unwrap_or_else(|| die("ghost header show FILE"))),
                o => die(format!("unknown `ghost header` operation {:?}", o)),
            }
        }
        "census" => census::cmd(rest),
        "phase" => phase::cmd(rest),
        "record" => record::cmd(rest),
        "film" => film::cmd(rest),
        "regen" => regen::cmd(rest),
        "regen-control" => regen::control(rest),
        "roundtrip" => roundtrip::cmd(rest),
        "verify" => verify::cmd(rest),
        "selftest" => selftest::cmd(rest),
        o => die(format!("unknown command {:?} (try `ghost --help`)", o)),
    }
}

// ---------------------------------------------------------------------------

fn cmd_manifest(a: &[String]) {
    match a.first().map(String::as_str) {
        Some("validation") => {
            let path = a.get(1).unwrap_or_else(|| die("ghost manifest validation FILE"));
            let c = Container::load(path).unwrap_or_else(|e| die(e));
            println!("{}", gbx::manifest::validation_manifest(c.body()).unwrap_or_else(|| "null".into()));
        }
        Some("diff") => {
            let left = a.get(1).unwrap_or_else(|| die("ghost manifest diff LEFT RIGHT"));
            let right = a.get(2).unwrap_or_else(|| die("ghost manifest diff LEFT RIGHT"));
            println!("{}", gbx::manifest::diff_files(left, right).unwrap_or_else(|e| die(e)));
        }
        Some(path) => println!("{}", gbx::manifest::manifest_file(path).unwrap_or_else(|e| die(e))),
        None => die("ghost manifest FILE | ghost manifest diff LEFT RIGHT"),
    }
}

fn cmd_chunks(a: &[String]) {
    let c = Container::load(&a[0]).unwrap_or_else(|e| die(e));
    println!("body {} B, {} skippable chunks", c.body().len(), c.chunks().len());
    for (cid, off, poff, sz) in c.chunks() {
        println!("  0x{:08X} at {:>9} payload {:>9} size {:>9}", cid, off, poff, sz);
    }
    // `--find-u32 N` -- every little-endian u32 in the body equal to N, with the
    // chunk it falls in. 227654's client threshold is an entity COUNT the file
    // must reach, and a count the record no longer matches has to be written
    // down somewhere; this is how you look for it without a hex editor.
    if let Some(i) = a.iter().position(|x| x == "--find-u32") {
        let want: u32 = a[i + 1].parse().unwrap_or_else(|_| die("--find-u32 N"));
        let b = c.body();
        let chunks = c.chunks();
        let mut n = 0;
        println!("body offsets holding the u32 {want}:");
        for o in 0..b.len().saturating_sub(3) {
            if u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) != want {
                continue;
            }
            let owner = chunks
                .iter()
                .filter(|(_, coff, _, sz)| o >= *coff && o < coff + sz + 12)
                .map(|(cid, _, poff, _)| format!("0x{cid:08X} +{}", o as i64 - *poff as i64))
                .next()
                .unwrap_or_else(|| "(outside every skippable chunk)".into());
            println!("  {o:>9}  {owner}");
            n += 1;
        }
        println!("{n} occurrence(s)");
    }
}

fn cmd_inspect(a: &[String]) {
    let path = a.first().unwrap_or_else(|| die("ghost inspect FILE"));
    let c = Container::load(path).unwrap_or_else(|e| die(e));
    println!("file          {} ({} B on disk)", path, std::fs::metadata(path).map(|m| m.len()).unwrap_or(0));
    println!("body          {} B, {} skippable chunks", c.body().len(), c.chunks().len());

    match c.embedded_map() {
        Some((o, n)) => {
            println!("EMBEDDED MAP  yes -- {} B at body offset {}", n, o);
            println!("              --map is DECORATION for this file: the server simulates the");
            println!("              copy inside it. `ghost map set` is the only way to change it.");
        }
        None => println!("EMBEDDED MAP  none -- --map is real for this file"),
    }
    let uids = c.uids();
    if !uids.is_empty() {
        let mut d: Vec<String> = uids.iter().map(|(_, s)| s.clone()).collect();
        d.sort();
        d.dedup();
        println!("map uids      {:?} ({} literal copies)", d, uids.len());
    }

    let dt = c.declared_times();
    if dt.is_empty() {
        println!("declared      none");
    } else {
        let mut vals: Vec<u32> = dt.iter().map(|(_, v)| *v).collect();
        vals.sort();
        vals.dedup();
        let shown: Vec<String> = vals.iter().map(|v| secs(*v as i64)).collect();
        println!(
            "declared      {} in {} copies{}",
            shown.join(" / "),
            dt.len(),
            if vals.len() > 1 { "   <-- DISAGREE, this file declares two different times" } else { "" }
        );
    }
    // The DECODED checkpoint list, not the chunk's raw words. Printing the raw
    // array through the seconds formatter is what made this line read
    // `splits 0.001 19.538 0.000 0.000 0.003 0.004 7.617 ...` -- a version
    // number as 0.001 and a per-entry tag as 0.002.
    if let Some(r) = c.result() {
        let s: Vec<String> = r.checkpoints().iter().map(|v| secs(*v as i64)).collect();
        println!(
            "splits        {}   ({} checkpoints, the last is the finish)",
            s.join(" "),
            r.entries.len()
        );
        println!(
            "              the result chunk declares race {}; its word 4 is {} (UNIDENTIFIED --\n\
             \x20             it was called nb_respawns and is not one: see GhostResult)",
            secs(r.race_ms as i64),
            r.word4_unidentified
        );
        if let Some((_, d)) = c.declared_times().first() {
            if *d as i32 != r.race_ms {
                println!(
                    "              <-- DISAGREE: the header declares {} and the result chunk {}",
                    secs(*d as i64),
                    secs(r.race_ms as i64)
                );
            }
        }
    }

    match Tape::from_file(path) {
        Err(e) => println!("input tape    NONE ({})", e),
        Ok(t) => {
            let a0 = &t.archives[0];
            let n = a0.packets.len();
            let same = a0.packets.iter().filter(|p| p.vsame).count();
            let resp = a0.packets.iter().filter(|p| p.respawn()).count();
            let mut modes: Vec<u32> = a0.packets.iter().map(|p| p.mode).collect();
            modes.sort();
            modes.dedup();
            println!(
                "input tape    {} archives, archive 0: {} ticks, start_offset {} ms, format v{}",
                t.archives.len(),
                n,
                a0.start_offset_ms,
                a0.format_version
            );
            println!(
                "              modes {:?}, {} same-as-previous packets, {} respawn ticks",
                modes, same, resp
            );
            match t.verbatim_is_identity() {
                Ok(()) => println!("              codec identity: OK (verbatim re-encode reproduces the file)"),
                Err(e) => println!("              codec identity: FAILED -- {}", e),
            }
        }
    }

    match gbx::record::decode_ghost(path) {
        Err(e) => println!("telemetry     NONE ({})", e),
        Ok(d) => {
            println!(
                "telemetry     {} samples, {} entities, record span {} .. {}",
                d.samples.len(),
                d.ents.len(),
                secs(d.start_ms as i64),
                secs(d.end_ms as i64)
            );
            if let (Some(f), Some(l)) = (d.samples.first(), d.samples.last()) {
                println!(
                    "              first {} at ({:.2}, {:.2}, {:.2})   last {} at ({:.2}, {:.2}, {:.2})",
                    secs(f.time_ms as i64),
                    f.x,
                    f.y,
                    f.z,
                    secs(l.time_ms as i64),
                    l.x,
                    l.y,
                    l.z
                );
            }
        }
    }
    ident::print(&c);
    if has(a, "--ticks") {
        if let Ok(t) = Tape::from_file(path) {
            print!("{}", t.to_text(path));
        }
    }
}

mod film;
mod sweep;

fn cmd_tape(a: &[String]) {
    let what = a.first().map(|s| s.as_str()).unwrap_or_else(|| die("ghost tape <extract|inject|script|expand|graft|poke|set|diff|stats|csv|bits>"));
    let rest = &a[1..];
    match what {
        "poke" => {
            // Override the vehicle inputs over a tick range, leaving every
            // other tick byte-identical. A brake pulse, a lifted throttle, a
            // held steer -- the one-variable probes this project builds by the
            // dozen to price a manoeuvre. Until now each one meant hand-editing
            // `t=` lines, which is how an off-by-one gets into a measurement
            // nobody can reproduce.
            let inp = rest.first().unwrap_or_else(|| {
                die("ghost tape poke IN[.gtape|.Ghost.Gbx] --out T.gtape --tick-range A..B --set steer=..,accel=..,brake=..")
            });
            let out = need(rest, "--out");
            // `--tick-range A..B`. This was `--ticks A..B`, which collided with
            // a `--ticks N` that means a COUNT in thirteen other files
            // (tmauto, tmmaps, uwlab) -- same name, same workspace, two
            // incompatible grammars, so anyone who learns one carries it to
            // the other and is silently wrong about what they asked for.
            //
            // The old spelling still works for one release and says so. It is
            // accepted only where it is unambiguous: `--ticks` here has always
            // wanted `A..B`.
            let range = match flag(rest, "--tick-range") {
                Some(v) => v.to_string(),
                None => match flag(rest, "--ticks") {
                    Some(v) => {
                        eprintln!(
                            "ghost: --ticks is deprecated here and will be removed; use \
                             --tick-range {} (it takes a RANGE, while --ticks means a \
                             COUNT everywhere else in the toolchain)",
                            v
                        );
                        v.to_string()
                    }
                    None => die("ghost tape poke wants --tick-range A..B"),
                },
            };
            let range = range.as_str();
            let (lo, hi) = range
                .split_once("..")
                .and_then(|(l, h)| Some((l.trim().parse::<usize>().ok()?, h.trim().parse::<usize>().ok()?)))
                .unwrap_or_else(|| die("--tick-range wants A..B, the half-open tick range to overwrite"));
            if hi <= lo {
                die(format!("--tick-range {}..{} is empty; B is exclusive and must exceed A", lo, hi));
            }
            // The fields to set, parsed once so a typo fails before any file is
            // written rather than silently poking nothing.
            let mut sets: Vec<(String, i64)> = Vec::new();
            for kv in need(rest, "--set").split(',') {
                let (k, v) = kv.split_once('=').unwrap_or_else(|| die(format!("--set {}: wants field=value", kv)));
                let k = k.trim();
                if !matches!(k, "steer" | "accel" | "brake") {
                    die(format!("--set {}: only steer, accel and brake are vehicle inputs", k));
                }
                let v: i64 = v.trim().parse().unwrap_or_else(|_| die(format!("--set {}: {} is not a number", k, v)));
                sets.push((k.to_string(), v));
            }
            let raw = std::fs::read(inp).unwrap_or_else(|e| die(format!("{}: {}", inp, e)));
            let t = if raw.first() == Some(&b'#') {
                Tape::from_text(&String::from_utf8_lossy(&raw)).unwrap_or_else(|e| die(e))
            } else {
                Tape::from_file(inp).unwrap_or_else(|e| die(e))
            };
            let txt = t.to_text(inp);
            let mut poked = 0usize;
            let mut lines: Vec<String> = Vec::new();
            for l in txt.lines() {
                let Some(after) = l.strip_prefix("t=") else {
                    lines.push(l.to_string());
                    continue;
                };
                let idx: usize = after
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| die(format!("cannot read a tick index from {:?}", l)));
                if idx < lo || idx >= hi {
                    lines.push(l.to_string());
                    continue;
                }
                let mut fields: Vec<String> = l.split_whitespace().map(|s| s.to_string()).collect();
                for (k, v) in &sets {
                    let pre = format!("{}=", k);
                    let mut found = false;
                    for f in fields.iter_mut() {
                        if f.starts_with(&pre) {
                            *f = format!("{}{}", pre, v);
                            found = true;
                        }
                    }
                    if !found {
                        die(format!("tick {} has no `{}` field to poke", idx, k));
                    }
                }
                lines.push(fields.join(" "));
                poked += 1;
            }
            if poked != hi - lo {
                die(format!(
                    "--ticks {}..{} names {} ticks but the tape has {} of them in that range",
                    lo,
                    hi,
                    hi - lo,
                    poked
                ));
            }
            let text = lines.join("\n") + "\n";
            // read it back rather than trusting the string we just built
            let re = Tape::from_text(&text).unwrap_or_else(|e| die(format!("the poked tape does not parse: {}", e)));
            if re.n() != t.n() {
                die(format!("read-back control FAILED: {} ticks in, {} out", t.n(), re.n()));
            }
            std::fs::write(out, &text).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
            let off: i64 = t.archives.first().map(|a| a.start_offset_ms as i64).unwrap_or(0);
            let sets_s: Vec<String> = sets.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
            println!(
                "wrote {}  {} ticks, {} poked ({} .. {} = race {} .. {}) with {}",
                out,
                re.n(),
                poked,
                lo,
                hi - 1,
                secs(lo as i64 * 10 + off),
                secs((hi - 1) as i64 * 10 + off),
                sets_s.join(" ")
            );
            println!("  read-back control OK: the poked tape parses to {} ticks", re.n());
        }
        "script" => {
            let base = rest.first().unwrap_or_else(|| die("ghost tape script BASE.gtape --events F --out T.gtape"));
            let ev = need(rest, "--events");
            let out = need(rest, "--out");
            let btxt = std::fs::read_to_string(base).unwrap_or_else(|e| die(format!("{}: {}", base, e)));
            let etxt = std::fs::read_to_string(ev).unwrap_or_else(|e| die(format!("{}: {}", ev, e)));
            let events = ghost::script::parse_events(&etxt).unwrap_or_else(|e| die(e));
            let keep = has(rest, "--keep-before");
            let txt = ghost::script::apply_from(&btxt, &events, keep).unwrap_or_else(|e| die(e));
            let txt = match rest.iter().position(|x| x == "--signature-at") {
                Some(i) => {
                    let ms: i64 = rest[i + 1].parse().unwrap_or_else(|_| die("--signature-at wants a race time in ms"));
                    ghost::script::signature(&txt, ms).unwrap_or_else(|e| die(e))
                }
                None => txt,
            };
            // control: the text we are about to write must parse as a tape,
            // and re-emit as the same text. A script that produced something
            // the codec cannot express is a failure here, not at inject time.
            let t = Tape::from_text(&txt).unwrap_or_else(|e| die(format!("the scripted tape does not parse: {}", e)));
            std::fs::write(&out, &txt).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
            println!("wrote {} ({} events, {} ticks)", out, events.len(), t.n());
        }
        "graft" => {
            // HEAD's first `--at` ticks, then TAIL's ticks from `--from`, as one
            // tape. This project grafts tapes constantly -- a route from one run,
            // a technique from another -- and until now every arm did it by hand
            // in the text format, renumbering `t=` with an editor. A renumbering
            // that slips by one is invisible and produces a run that simply goes
            // somewhere else.
            //
            // The output keeps HEAD's archive framing (its format, its
            // `start_offset_ms` and its tail bytes), because the container it
            // will be injected into is HEAD's. TAIL contributes vehicle inputs
            // only.
            let head = need(rest, "--head");
            let tail = need(rest, "--tail");
            let out = need(rest, "--out");
            let at: usize = num(rest, "--at").unwrap_or_else(|| die("--at TICK: where HEAD stops")) as usize;
            let from: usize = num(rest, "--from").unwrap_or(0) as usize;
            // HEAD and TAIL may each be a `.Ghost.Gbx` or an already-extracted
            // `.gtape`. Guessing by extension is how a tool ends up asserting
            // "not a GBX file" at the user; the first byte says which it is.
            let load = |p: &str| -> Tape {
                let raw = std::fs::read(p).unwrap_or_else(|e| die(format!("{}: {}", p, e)));
                if raw.first() == Some(&b'#') {
                    Tape::from_text(&String::from_utf8_lossy(&raw)).unwrap_or_else(|e| die(e))
                } else {
                    Tape::from_file(p).unwrap_or_else(|e| die(e))
                }
            };
            let h = load(head);
            let t = load(tail);
            let ht = h.to_text(head);
            let tt = t.to_text(tail);
            let mut lines: Vec<String> = Vec::new();
            let mut n_head = 0usize;
            for l in ht.lines() {
                if let Some(rest2) = l.strip_prefix("t=") {
                    let idx: usize = rest2
                        .split_whitespace()
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| die(format!("cannot read a tick index from {:?}", l)));
                    if idx >= at {
                        break;
                    }
                    lines.push(l.to_string());
                    n_head += 1;
                } else {
                    lines.push(l.to_string());
                }
            }
            if n_head != at {
                die(format!(
                    "--at {} but HEAD only has {} ticks before it",
                    at, n_head
                ));
            }
            let mut n_tail = 0usize;
            for l in tt.lines() {
                let Some(rest2) = l.strip_prefix("t=") else { continue };
                let mut it = rest2.splitn(2, ' ');
                let idx: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
                let body = it.next().unwrap_or("");
                if idx < from {
                    continue;
                }
                lines.push(format!("t={} {}", at + n_tail, body));
                n_tail += 1;
            }
            if n_tail == 0 {
                die(format!("--from {} leaves TAIL with no ticks", from));
            }
            // The `@archive` line's own packet count must match, or the reader
            // trusts a number the body contradicts.
            let total = at + n_tail;
            for l in lines.iter_mut() {
                if l.starts_with("@archive ") {
                    *l = l
                        .split_whitespace()
                        .map(|f| {
                            if let Some(v) = f.strip_prefix("packets=") {
                                let _ = v;
                                format!("packets={}", total)
                            } else {
                                f.to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                }
            }
            let text = lines.join("\n") + "\n";
            // read it back rather than trusting the string we just built
            let re = Tape::from_text(&text).unwrap_or_else(|e| die(format!("the graft does not parse: {}", e)));
            if re.n() != total {
                die(format!("read-back control FAILED: {} ticks written, {} read", total, re.n()));
            }
            std::fs::write(out, &text).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
            let off: i64 = h.archives.first().map(|a| a.start_offset_ms as i64).unwrap_or(0);
            println!(
                "wrote {}  {} ticks = {} from {} (race {} .. {}) + {} from {} (its tick {} on)",
                out,
                total,
                at,
                head,
                secs(off),
                secs(at as i64 * 10 + off),
                n_tail,
                tail,
                from
            );
            println!("  read-back control OK: the graft parses to {} ticks", re.n());
            println!(
                "  the handover is at tick {} = race {} of {}",
                at,
                secs(at as i64 * 10 + off),
                head
            );
        }
        "set" => {
            // Overwrite the vehicle inputs over a tick RANGE. The sweep
            // primitive: "hold this steer for these 200 ticks and see where
            // the car ends up" is the shape of almost every experiment on this
            // project, and doing it by editing `.gtape` text is how a window
            // ends up one tick out.
            //
            // Ranges are TICKS, not race ms, because a tape's own indexing is
            // ticks and `start_offset_ms` is a per-file trap. The command
            // prints the race window it corresponds to so the two are never
            // confused.
            let src = rest.first().unwrap_or_else(|| die("ghost tape set IN --out T.gtape --from A --to B [--steer S] [--accel 0|1] [--brake 0|1]"));
            let out = need(rest, "--out");
            let from: usize = num(rest, "--from").unwrap_or(0) as usize;
            let to: usize = num(rest, "--to").unwrap_or_else(|| die("--to TICK (exclusive)")) as usize;
            let steer = num(rest, "--steer");
            let accel = num(rest, "--accel");
            let brake = num(rest, "--brake");
            if steer.is_none() && accel.is_none() && brake.is_none() {
                die("nothing to set: give at least one of --steer / --accel / --brake");
            }
            if let Some(s) = steer {
                if !(-127..=127).contains(&s) {
                    die(format!("--steer is an i8 over 127; {} is out of range", s));
                }
            }
            let raw = std::fs::read(src).unwrap_or_else(|e| die(format!("{}: {}", src, e)));
            let t = if raw.first() == Some(&b'#') {
                Tape::from_text(&String::from_utf8_lossy(&raw)).unwrap_or_else(|e| die(e))
            } else {
                Tape::from_file(src).unwrap_or_else(|e| die(e))
            };
            if to > t.n() || from >= to {
                die(format!("--from {} --to {} does not fit a {}-tick tape", from, to, t.n()));
            }
            let mut n = 0usize;
            let text: String = t
                .to_text(src)
                .lines()
                .map(|l| {
                    let Some(rest2) = l.strip_prefix("t=") else { return l.to_string() };
                    let idx: usize =
                        rest2.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
                    if idx < from || idx >= to {
                        return l.to_string();
                    }
                    n += 1;
                    // Rewrite the named fields in place; `vsame=1` is forced to
                    // 0 because a changed value cannot be coded as "same as the
                    // previous tick", and the writer expands it anyway.
                    l.split_whitespace()
                        .map(|f| {
                            if f.starts_with("steer=") {
                                steer.map_or(f.to_string(), |s| format!("steer={}", s))
                            } else if f.starts_with("accel=") {
                                accel.map_or(f.to_string(), |s| format!("accel={}", s))
                            } else if f.starts_with("brake=") {
                                brake.map_or(f.to_string(), |s| format!("brake={}", s))
                            } else if f.starts_with("vsame=") {
                                "vsame=0".to_string()
                            } else {
                                f.to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            if n != to - from {
                die(format!("expected to rewrite {} ticks, rewrote {}", to - from, n));
            }
            let re = Tape::from_text(&text).unwrap_or_else(|e| die(format!("the edit does not parse: {}", e)));
            if re.n() != t.n() {
                die(format!("read-back control FAILED: {} ticks in, {} out", t.n(), re.n()));
            }
            if let Some(s) = steer {
                let got = re.steer_i8s();
                if (from..to).any(|i| got[i] as i64 != s) {
                    die("read-back control FAILED: a tick in the window does not carry the steer asked for");
                }
            }
            std::fs::write(out, &text).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
            let off: i64 = t.archives.first().map(|a| a.start_offset_ms as i64).unwrap_or(0);
            println!(
                "wrote {}  {} ticks rewritten, t={}..{} = race {} .. {}  (read-back control OK)",
                out,
                n,
                from,
                to,
                secs(from as i64 * 10 + off),
                secs(to as i64 * 10 + off)
            );
        }
        "extract" => {
            let src = rest.first().unwrap_or_else(|| die("ghost tape extract FILE --out T"));
            let out = need(rest, "--out");
            let t = Tape::from_file(src).unwrap_or_else(|e| die(e));
            t.verbatim_is_identity().unwrap_or_else(|e| {
                die(format!(
                    "REFUSING to extract: this file's input chunk does not survive a verbatim \
                     re-encode ({}). Extracting it would produce a tape that cannot be injected \
                     back losslessly.",
                    e
                ))
            });
            std::fs::write(out, t.to_text(src)).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
            println!(
                "wrote {}  ({} archives, {} ticks, codec identity OK)",
                out,
                t.archives.len(),
                t.n()
            );
        }
        "inject" => {
            let inp = rest.first().unwrap_or_else(|| die("ghost tape inject IN OUT --tape T"));
            let out = rest.get(1).unwrap_or_else(|| die("ghost tape inject IN OUT --tape T"));
            let tp = need(rest, "--tape");
            let txt = std::fs::read_to_string(tp).unwrap_or_else(|e| die(format!("{}: {}", tp, e)));
            let newt = Tape::from_text(&txt).unwrap_or_else(|e| die(e));
            let c = Container::load(inp).unwrap_or_else(|e| die(e));
            let oldt = Tape::from_file(inp).unwrap_or_else(|e| die(e));
            if newt.archives.len() != oldt.archives.len() {
                die(format!(
                    "tape has {} archives, {} has {}",
                    newt.archives.len(),
                    inp,
                    oldt.archives.len()
                ));
            }
            for (i, (na, oa)) in newt.archives.iter().zip(oldt.archives.iter()).enumerate() {
                if na.packets.len() != oa.packets.len() {
                    die(format!(
                        "archive {}: tape has {} ticks, {} has {}. A tape is injected into a \
                         container of the same length -- use `ghost trim` to change the length.",
                        i,
                        na.packets.len(),
                        inp,
                        oa.packets.len()
                    ));
                }
            }
            let enc = if has(rest, "--verbatim") { Encoding::Verbatim } else { Encoding::Explicit };
            let body = newt.splice_into(c.body(), enc).unwrap_or_else(|e| die(e));
            container::write_gbx(&c.gbx, body, out).unwrap_or_else(|e| die(e));
            // control: read it straight back and require the tape to match
            let back = Tape::from_file(out).unwrap_or_else(|e| die(e));
            let mut bad = 0usize;
            for (na, ba) in newt.archives.iter().zip(back.archives.iter()) {
                for (i, (p, q)) in na.packets.iter().zip(ba.packets.iter()).enumerate() {
                    if p.steer != q.steer
                        || p.accel != q.accel
                        || p.brake != q.brake
                        || p.respawn() != q.respawn()
                        || p.mode != q.mode
                        || p.mouse != q.mouse
                        || p.tri != q.tri
                    {
                        if bad < 5 {
                            eprintln!("  tick {}: written back differently", i);
                        }
                        bad += 1;
                    }
                }
            }
            if bad > 0 {
                die(format!("read-back control FAILED on {} ticks -- {} is not trustworthy", bad, out));
            }
            let n = newt.n();
            let writable = back.archives[0].packets.iter().filter(|p| !p.vsame).count();
            println!(
                "wrote {}  ({} ticks, {} explicitly coded, read-back control OK)",
                out, n, writable
            );
            let dec = gbx::record::decode_ghost(out).ok();
            if let Some(d) = dec {
                if !d.samples.is_empty() && !has(rest, "--allow-telemetry-mismatch") {
                    println!(
                        "NOTE: this file still carries {} telemetry samples from BEFORE the edit. \
                         They describe the old inputs. Run `ghost regen` to rebuild them, or the \
                         file will render the old run.",
                        d.samples.len()
                    );
                }
            }
        }
        "expand" => {
            let inp = rest.first().unwrap_or_else(|| die("ghost tape expand IN OUT"));
            let out = rest.get(1).unwrap_or_else(|| die("ghost tape expand IN OUT"));
            let state_too = rest.iter().any(|s| s == "--state");
            let c = Container::load(inp).unwrap_or_else(|e| die(e));
            let mut t = Tape::from_file(inp).unwrap_or_else(|e| die(e));
            let before = t.archives[0].packets.iter().filter(|p| p.vsame).count();
            // A repeated STATE word carries no literal, and the respawn input
            // is bit 31 of one, so on a tape like 153527's -- 566 327 packets,
            // 547 329 of them repeats -- a respawn is unwritable at 96.7% of
            // ticks. `--state` turns each repeat into the literal the decoder
            // derives from it anyway.
            let mut states = 0;
            if state_too {
                for a in t.archives.iter_mut() {
                    for p in a.packets.iter_mut() {
                        if !matches!(p.state, gbx::tape::StateEnc::Lit(_)) {
                            p.state = gbx::tape::StateEnc::Lit(gbx::tape::literal_for(p.word0, p.flags));
                            states += 1;
                        }
                    }
                }
            }
            let body = t.splice_into(c.body(), Encoding::Explicit).unwrap_or_else(|e| die(e));
            container::write_gbx(&c.gbx, body, out).unwrap_or_else(|e| die(e));
            let back = Tape::from_file(out).unwrap_or_else(|e| die(e));
            for (p, q) in t.archives[0].packets.iter().zip(back.archives[0].packets.iter()) {
                if p.steer != q.steer || p.accel != q.accel || p.brake != q.brake {
                    die("expansion changed an input value -- refusing");
                }
                // With --state the point of the exercise is that the state word
                // survives the rewrite, so it is checked rather than assumed.
                if state_too && (p.word0 != q.word0 || p.flags != q.flags) {
                    die("state expansion changed the state word -- refusing");
                }
            }
            println!(
                "wrote {}  ({} same-as-previous packets expanded{}; every one of {} ticks is now writable)",
                out,
                before,
                if state_too { format!(", {} repeated state words turned into literals", states) } else { String::new() },
                t.n()
            );
        }
        "diff" => {
            let x = Tape::from_text(&std::fs::read_to_string(&rest[0]).unwrap()).unwrap_or_else(|e| die(e));
            let y = Tape::from_text(&std::fs::read_to_string(&rest[1]).unwrap()).unwrap_or_else(|e| die(e));
            let n = x.n().min(y.n());
            let mut d = 0;
            for i in 0..n {
                let (p, q) = (&x.archives[0].packets[i], &y.archives[0].packets[i]);
                if p.steer != q.steer || p.accel != q.accel || p.brake != q.brake || p.respawn() != q.respawn() {
                    if d < 80 {
                        println!(
                            "t={:<6} steer {:>4} -> {:<4} accel {} -> {}  brake {} -> {}  respawn {} -> {}",
                            i,
                            p.steer_i8(),
                            q.steer_i8(),
                            p.accel,
                            q.accel,
                            p.brake,
                            q.brake,
                            p.respawn() as u8,
                            q.respawn() as u8
                        );
                    }
                    d += 1;
                }
            }
            println!("{} ticks differ (of {} / {})", d, x.n(), y.n());
            if x.n() != y.n() || d > 0 {
                std::process::exit(1);
            }
        }
        "stats" => {
            let t = if rest[0].ends_with(".gtape") {
                Tape::from_text(&std::fs::read_to_string(&rest[0]).unwrap()).unwrap_or_else(|e| die(e))
            } else {
                Tape::from_file(&rest[0]).unwrap_or_else(|e| die(e))
            };
            let a0 = &t.archives[0];
            let n = a0.packets.len();
            // --from / --to WINDOW the counts to a race window, because the
            // question that matters about an input is almost always "what did
            // the driver do DURING THE RACE" and a tape starts about 1.52 s
            // before it. Without a window, a run held flat out from the lights
            // to the line reads as "the throttle varied" on the strength of the
            // countdown alone -- which is exactly the reading that refused two
            // correct regenerations (byte 15, the gas echo, on 279209 and
            // 197047).
            let from = flag(rest, "--from").and_then(|v| v.parse::<i64>().ok());
            let to = flag(rest, "--to").and_then(|v| v.parse::<i64>().ok());
            let inw = |i: usize| -> bool {
                let ms = t.race_ms(i);
                from.map(|f| ms >= f).unwrap_or(true) && to.map(|x| ms <= x).unwrap_or(true)
            };
            let idx: Vec<usize> = (0..n).filter(|i| inw(*i)).collect();
            let mut ev = 0;
            for w in idx.windows(2) {
                let (p, q) = (&a0.packets[w[0]], &a0.packets[w[1]]);
                if p.steer != q.steer || p.accel != q.accel || p.brake != q.brake {
                    ev += 1;
                }
            }
            if from.is_some() || to.is_some() {
                println!("window         {} .. {}  ({} of {} ticks)",
                    from.map(|v| secs(v)).unwrap_or_else(|| "start".into()),
                    to.map(|v| secs(v)).unwrap_or_else(|| "end".into()),
                    idx.len(), n);
            }
            println!("ticks          {}", idx.len());
            println!("tape span      {} .. {}", secs(a0.start_offset_ms as i64), secs(a0.start_offset_ms as i64 + 10 * n as i64));
            println!("input events   {}", ev);
            println!("respawn ticks  {}", idx.iter().filter(|i| a0.packets[**i].respawn()).count());
            println!("same-as-prev   {}", idx.iter().filter(|i| a0.packets[**i].vsame).count());
            println!("accel on       {}", idx.iter().filter(|i| a0.packets[**i].accel != 0).count());
            println!("brake on       {}", idx.iter().filter(|i| a0.packets[**i].brake != 0).count());
            println!("mouse packets  {}", idx.iter().filter(|i| a0.packets[**i].mouse.is_some()).count());
            // PER CHANNEL: constant or not, in this window. This is the fact
            // `ghost regen`'s liveness check reads, printed so it can be
            // checked by hand rather than taken on trust.
            let steer: Vec<i64> = idx.iter().map(|i| a0.packets[*i].steer_i8() as i64).collect();
            let accel: Vec<i64> = idx.iter().map(|i| a0.packets[*i].accel as i64).collect();
            let brake: Vec<i64> = idx.iter().map(|i| a0.packets[*i].brake as i64).collect();
            for (name, v) in [("steer", &steer), ("accel", &accel), ("brake", &brake)] {
                match v.first() {
                    None => println!("{name:<14} no tick in the window"),
                    Some(f) if v.iter().all(|x| x == f) => {
                        println!("{name:<14} CONSTANT at {f} for all {} ticks -- a constant echo of it in the record is CORRECT", v.len())
                    }
                    Some(_) => {
                        let (lo, hi) = (v.iter().min().unwrap(), v.iter().max().unwrap());
                        println!("{name:<14} varies, {lo} .. {hi}")
                    }
                }
            }
        }
        "recinputs" => {
            let f = &rest[0];
            let t = Tape::from_file(f).unwrap_or_else(|e| die(e));
            let d = gbx::record::decode_ghost(f).unwrap_or_else(|e| die(e));
            let raw = raw_vehicle_samples(f).unwrap_or_else(|e| die(e));
            let a0 = &t.archives[0];
            let so = a0.start_offset_ms as i64;
            let (ss, r) = &raw;
            if has(rest, "--check") {
                // The FIT CONTROL for the recorded input channels: predict the
                // three telemetry bytes from the tape and count exact hits.
                let (mut n, mut h14, mut h15, mut h18) = (0usize, 0usize, 0usize, 0usize);
                let mut seen_lift = 0usize;
                let mut seen_brake = 0usize;
                for (i, s) in d.samples.iter().enumerate() {
                    let idx = (s.time_ms as i64 - so) / 10;
                    if idx < 0 || idx >= a0.packets.len() as i64 {
                        continue;
                    }
                    let p = &a0.packets[idx as usize];
                    let d0 = &r[i * ss..(i + 1) * ss];
                    n += 1;
                    if d0[14] == regen::steer_byte(p.steer_i8()) {
                        h14 += 1;
                    }
                    if d0[15] == regen::pedal_byte(p.accel) {
                        h15 += 1;
                    }
                    if d0[18] == regen::pedal_byte(p.brake) {
                        h18 += 1;
                    }
                    if p.accel == 0 {
                        seen_lift += 1;
                    }
                    if p.brake != 0 {
                        seen_brake += 1;
                    }
                }
                println!(
                    "{} samples  steer {:.2}%  gas {:.2}%  brake {:.2}%   (lift ticks {}, brake ticks {})",
                    n,
                    100.0 * h14 as f64 / n.max(1) as f64,
                    100.0 * h15 as f64 / n.max(1) as f64,
                    100.0 * h18 as f64 / n.max(1) as f64,
                    seen_lift,
                    seen_brake
                );
                // Where do the misses come from -- the byte encoding or the
                // tick the sample is paired with? Score every phase from -2 to
                // +2 ticks; if a neighbouring phase is perfect, the encoding is
                // right and the pairing was wrong.
                for ph in -2i64..=2 {
                    let (mut m, mut hit) = (0usize, 0usize);
                    for (i, s) in d.samples.iter().enumerate() {
                        let idx = (s.time_ms as i64 - so) / 10 + ph;
                        if idx < 0 || idx >= a0.packets.len() as i64 {
                            continue;
                        }
                        let p = &a0.packets[idx as usize];
                        let d0 = &r[i * ss..(i + 1) * ss];
                        m += 1;
                        if d0[14] == regen::steer_byte(p.steer_i8()) {
                            hit += 1;
                        }
                    }
                    println!("   phase {:+}: steer {:.2}%", ph, 100.0 * hit as f64 / m.max(1) as f64);
                }
                if h14 < n || h15 < n || h18 < n {
                    std::process::exit(1);
                }
                return;
            }
            println!("{:>8} {:>6} {:>5} {:>5}   {:>4} {:>4} {:>4}", "t_ms", "steer", "gas", "brk", "b14", "b15", "b18");
            let step = (d.samples.len() / 40).max(1);
            for (i, s) in d.samples.iter().enumerate() {
                if i % step != 0 {
                    continue;
                }
                let idx = (s.time_ms as i64 - so) / 10;
                if idx < 0 || idx >= a0.packets.len() as i64 {
                    continue;
                }
                let p = &a0.packets[idx as usize];
                let d0 = &r[i * ss..(i + 1) * ss];
                println!(
                    "{:>8} {:>6} {:>5} {:>5}   {:>4} {:>4} {:>4}",
                    s.time_ms, p.steer_i8(), p.accel, p.brake, d0[14], d0[15], d0[18]
                );
            }
        }
        "sync-record" => {
            // Write the RECORDED input channels from the tape. Useful on its
            // own: after `ghost tape inject`, the telemetry's steer / gas /
            // brake bytes are still the old run's even though they are fully
            // determined by the tape.
            let inp = rest.first().unwrap_or_else(|| die("ghost tape sync-record IN OUT"));
            let out = rest.get(1).unwrap_or_else(|| die("ghost tape sync-record IN OUT"));
            match regen::write_input_channels(inp, out) {
                Err(e) => die(e),
                Ok((w, sk)) => {
                    println!("wrote {} ({} samples rewritten, {} outside the tape)", out, w, sk);
                    if let Some((k, _, lag, n)) = verify::tape_record_agreement(out) {
                        println!(
                            "  tape/record agreement is now kappa {:.3} over {} samples (lag {} ms)",
                            k, n, lag
                        );
                    }
                }
            }
        }
        "sweep" => sweep::cmd(rest),
        "csv" => {
            // The `inputs/*.inputs.csv` a map page publishes as "the run
            // itself". They were written by a tool that no longer exists, so
            // until now nothing could regenerate one from its ghost -- or, the
            // reason this exists, prove that a ghost about to be filmed carries
            // the tape the page publishes. Two files on this project's store
            // re-simulate to 12.759 on the same map and their tapes differ.
            let src = rest
                .first()
                .unwrap_or_else(|| die("ghost tape csv FILE|TAPE.gtape [--out F.csv] [--from MS] [--to MS]"));
            let t = if src.ends_with(".gtape") {
                Tape::from_text(&std::fs::read_to_string(src).unwrap_or_else(|e| die(format!("{src}: {e}"))))
                    .unwrap_or_else(|e| die(e))
            } else {
                Tape::from_file(src).unwrap_or_else(|e| die(e))
            };
            let from = num(rest, "--from");
            let to = num(rest, "--to");
            let steer = t.steer_i8s();
            let accel = t.accels();
            let brake = t.brakes();
            let mut s = String::from("race_ms,steer,accel,brake\n");
            let mut rows = 0usize;
            for i in 0..steer.len() {
                let ms = t.race_ms(i);
                if from.is_some_and(|f| ms < f) || to.is_some_and(|x| ms > x) {
                    continue;
                }
                s.push_str(&format!("{},{},{},{}\n", ms, steer[i], accel[i], brake[i]));
                rows += 1;
            }
            match flag(rest, "--out") {
                Some(o) => {
                    std::fs::write(o, &s).unwrap_or_else(|e| die(format!("{o}: {e}")));
                    println!("wrote {o}  ({rows} ticks, race {} .. {})",
                             secs(t.race_ms(0)), secs(t.race_ms(steer.len().saturating_sub(1))));
                }
                None => print!("{s}"),
            }
        }
        "bits" => cmd_bits(rest),
        o => die(format!("unknown `ghost tape` operation {:?}", o)),
    }
}

/// The state literal is 33 or 34 bits and only a few of them have names. This
/// counts, over a whole corpus, which bits ever vary -- so "unnamed" is an
/// enumerated set rather than a shrug.
fn cmd_bits(a: &[String]) {
    let files: Vec<&String> = a.iter().filter(|s| !s.starts_with("--")).collect();
    let show_events = has(a, "--events");
    let mut ones = [0u64; 34];
    let mut zeros = [0u64; 34];
    let mut lits = 0u64;
    for f in &files {
        let t = match Tape::from_file(f) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  skip {}: {}", f, e);
                continue;
            }
        };
        for (archive_index, ar) in t.archives.iter().enumerate() {
            for (tick, p) in ar.packets.iter().enumerate() {
                if let tape::StateEnc::Lit(l) = p.state {
                    lits += 1;
                    if show_events {
                        println!(
                            "state\tfile={}\tarchive={}\ttick={}\trace_ms={}\tliteral=0x{l:09x}\tword0=0x{:08x}\tflags=0x{:06x}",
                            f,
                            archive_index,
                            tick,
                            ar.start_offset_ms + tick as i32 * 10,
                            p.word0,
                            p.flags
                        );
                    }
                    for b in 0..34 {
                        if l >> b & 1 == 1 {
                            ones[b] += 1
                        } else {
                            zeros[b] += 1
                        }
                    }
                }
            }
        }
    }
    println!("{} explicit state literals over {} files", lits, files.len());
    println!("bit  ones      zeros     what it is");
    for b in 0..34 {
        if ones[b] == 0 && zeros[b] == 0 {
            continue;
        }
        let name = match b {
            0..=3 => "mode (word0 & 0xF)",
            31 => "RESPAWN",
            5..=26 => "flags",
            _ => "",
        };
        let varies = ones[b] > 0 && zeros[b] > 0;
        println!(
            "{:>3}  {:>9} {:>9}  {}{}",
            b,
            ones[b],
            zeros[b],
            name,
            if varies { "   <-- varies" } else { "" }
        );
    }
}

fn cmd_map(a: &[String]) {
    let what = a.first().map(|s| s.as_str()).unwrap_or_else(|| die("ghost map <show|extract|set>"));
    let rest = &a[1..];
    match what {
        "show" => {
            let c = Container::load(&rest[0]).unwrap_or_else(|e| die(e));
            match c.embedded_map() {
                None => {
                    println!("NO embedded map chunk 0x03093002.");
                    println!("--map is REAL for this file: the server reads UserData/Maps.");
                    let mut u: Vec<String> = c.uids().into_iter().map(|(_, s)| s).collect();
                    u.sort();
                    u.dedup();
                    println!("declared map uid(s): {:?}", u);
                }
                Some((o, n)) => {
                    println!("EMBEDDED MAP: {} B at body offset {}", n, o);
                    println!("The dedicated server simulates THIS copy. --map, UserData/Maps and");
                    println!("the uid in the header are decoration for this file.");
                    let bytes = c.embedded_map_bytes().unwrap();
                    println!("carried map uid: {:?}", map_uid_of(&bytes));
                    let mut u: Vec<String> = c.uids().into_iter().map(|(_, s)| s).collect();
                    u.sort();
                    u.dedup();
                    println!("uid literals in the file: {:?}", u);
                }
            }
        }
        "extract" => {
            let c = Container::load(&rest[0]).unwrap_or_else(|e| die(e));
            let out = need(rest, "--out");
            match c.embedded_map_bytes() {
                None => die("this file carries no embedded map (nothing to extract)"),
                Some(b) => {
                    std::fs::write(out, &b).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
                    println!("wrote {} ({} B), uid {:?}", out, b.len(), map_uid_of(&b));
                }
            }
        }
        "rebind" => {
            // Change which map a PURE GHOST is bound to, by rewriting the uid
            // it declares (chunk 0x03092010 and its copies).
            //
            // This is the right operation for a file with no embedded map, and
            // the WRONG one for a file that carries one: there, rewriting the
            // uid makes the file CLAIM the other map while the server goes on
            // simulating the copy inside it. So it refuses.
            let inp = &rest[0];
            let out = &rest[1];
            let mapf = need(rest, "--map");
            let c = Container::load(inp).unwrap_or_else(|e| die(e));
            if c.embedded_map().is_some() && !has(rest, "--force") {
                die(
                    "REFUSED: this file carries an embedded map, so the server will simulate THAT \
                     copy whatever uid the file declares. Rewriting the uid here would produce a \
                     file that claims one map and runs another -- the exact failure this API \
                     exists to prevent. Use `ghost map set` to replace the carried map.",
                );
            }
            let mapdata = std::fs::read(mapf).unwrap_or_else(|e| die(format!("{}: {}", mapf, e)));
            let newuid = map_uid_of(&mapdata).unwrap_or_else(|| die("no uid in that map"));
            let olds: Vec<String> = {
                let mut v: Vec<String> = c.uids().into_iter().map(|(_, s)| s).collect();
                v.sort();
                v.dedup();
                v
            };
            if olds.is_empty() {
                die("this file declares no map uid at all");
            }
            if olds.len() > 1 {
                die(format!("this file declares more than one uid ({:?}); refusing to guess", olds));
            }
            let old = &olds[0];
            if old.len() != newuid.len() {
                die(format!(
                    "uid length differs ({} vs {}): rewriting would change the chunk size",
                    old.len(),
                    newuid.len()
                ));
            }
            let mut body = c.body().to_vec();
            let ob = old.as_bytes();
            let nb = newuid.as_bytes();
            let mut n = 0;
            let mut i = 0usize;
            while i + 4 + ob.len() <= body.len() {
                if u32::from_le_bytes(body[i..i + 4].try_into().unwrap()) as usize == ob.len()
                    && &body[i + 4..i + 4 + ob.len()] == ob
                {
                    body[i + 4..i + 4 + ob.len()].copy_from_slice(nb);
                    n += 1;
                    i += 4 + ob.len();
                    continue;
                }
                i += 1;
            }
            container::write_gbx(&c.gbx, body, out).unwrap_or_else(|e| die(e));
            let c2 = Container::load(out).unwrap_or_else(|e| die(e));
            let after: Vec<String> = {
                let mut v: Vec<String> = c2.uids().into_iter().map(|(_, s)| s).collect();
                v.sort();
                v.dedup();
                v
            };
            if after != vec![newuid.clone()] {
                die(format!("read-back control FAILED: uids are now {:?}", after));
            }
            println!("wrote {}", out);
            println!("  uid {} -> {}  ({} literal copies rewritten, read-back OK)", old, newuid, n);
            println!(
                "  PROVE IT: validate with ONLY {} in UserData/Maps. The control that makes the \n\
                 \x20         answer mean something is the same tape UNREBOUND against that map -- \n\
                 \x20         it must return nothing at all."
                , mapf
            );
        }
        "set" => {
            let inp = &rest[0];
            let out = &rest[1];
            let mapf = need(rest, "--map");
            let newmap = std::fs::read(mapf).unwrap_or_else(|e| die(format!("{}: {}", mapf, e)));
            if newmap.len() < 16 || &newmap[0..3] != b"GBX" {
                die(format!("{} is not a GBX map file", mapf));
            }
            let c = Container::load(inp).unwrap_or_else(|e| die(e));
            let newuid = map_uid_of(&newmap).unwrap_or_else(|| die("no uid in the replacement map"));
            let body = set_embedded_map(&c, &newmap, &newuid).unwrap_or_else(|e| die(e));
            container::write_gbx(&c.gbx, body, out).unwrap_or_else(|e| die(e));
            // control: read it back
            let c2 = Container::load(out).unwrap_or_else(|e| die(e));
            match c2.embedded_map_bytes() {
                None => die("wrote a file with no embedded map -- refusing to claim success"),
                Some(b) => {
                    if b != newmap {
                        die("the map read back is not the map written -- refusing");
                    }
                }
            }
            let u: Vec<String> = {
                let mut v: Vec<String> = c2.uids().into_iter().map(|(_, s)| s).collect();
                v.sort();
                v.dedup();
                v
            };
            println!("wrote {}", out);
            println!("  embedded map replaced: {} B -> {} B, uid {}", c.embedded_map().map(|x| x.1).unwrap_or(0), newmap.len(), newuid);
            println!("  uid literals now: {:?}", u);
            println!("  PROVE IT with an EMPTY UserData/Maps: `ghost verify {} --empty-maps`", out);
        }
        o => die(format!("unknown `ghost map` operation {:?}", o)),
    }
}


/// `ghost samples FILE [--every MS]`: the telemetry record as CSV — one row a
/// sample: `t_s,x,y,z,speed_kmh,yaw,pitch,roll,ground,gas,brake,steer,wetness`.
/// The trajectory the CLIENT replays for a ghost (it plays these positions
/// back; only record validation re-simulates the tape), so the file's own
/// answer to "where does this ghost's car go" needs no engine.
fn cmd_samples(args: &[String]) {
    let Some(path) = args.first() else {
        eprintln!("ghost samples FILE [--every MS]");
        std::process::exit(2);
    };
    let every: i32 = args.iter().position(|a| a == "--every").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(0);
    let d = match gbx::record::decode_ghost(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{path}: no telemetry record ({e})");
            std::process::exit(1);
        }
    };
    println!("t_s,x,y,z,speed_kmh,yaw,pitch,roll,ground,gas,brake,steer,wetness");
    let mut next = i32::MIN;
    for s in &d.samples {
        if every > 0 && s.time_ms < next {
            continue;
        }
        next = s.time_ms + every;
        println!(
            "{:.3},{:.3},{:.3},{:.3},{:.1},{:.3},{:.3},{:.3},{},{:.2},{:.2},{:.3},{:.2}",
            s.time_ms as f64 / 1000.0, s.x, s.y, s.z, s.speed_kmh, s.yaw, s.pitch, s.roll,
            if s.is_ground_contact { 1 } else { 0 }, s.gas, s.brake, s.steer, s.wetness
        );
    }
}
