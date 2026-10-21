//! `ghost selftest` -- the whole suite, from one command.
//!
//! Three tiers, each printed with its verdict:
//!
//!   PURE    format and codec checks against the checked-in fixtures. No
//!           server, no engine, milliseconds.
//!   ORACLE  the dedicated server, validating files this run just wrote.
//!   ENGINE  the real physics engine, regenerating a trajectory and being
//!           checked against a file that already knew the answer. Minutes.
//!
//! Every check states what it proves. A check that cannot run says SKIP and
//! why, and `--strict` turns a SKIP into a failure so a green run in CI cannot
//! be a green run that tested nothing.

use gbx::container::{secs, Container};
use crate::ident::{self, Role};
use crate::oracle::{self, MapsMode};
use gbx::tape::{Encoding, StateEnc, Tape};
use crate::cli::{flag, has};
use std::path::{Path, PathBuf};

pub struct Suite {
    pub data: PathBuf,
    pub work: PathBuf,
    pub server: Option<PathBuf>,
    pub results: Vec<(String, &'static str, String)>,
    pub strict: bool,
}

impl Suite {
    fn pass(&mut self, name: &str, detail: impl Into<String>) {
        println!("  PASS {:<26} {}", name, detail.into());
        self.results.push((name.into(), "PASS", String::new()));
    }
    fn fail(&mut self, name: &str, detail: impl Into<String>) {
        let d = detail.into();
        println!("  FAIL {:<26} {}", name, d);
        self.results.push((name.into(), "FAIL", d));
    }
    fn skip(&mut self, name: &str, detail: impl Into<String>) {
        let d = detail.into();
        println!("  SKIP {:<26} {}", name, d);
        self.results.push((name.into(), "SKIP", d));
    }
    fn check(&mut self, name: &str, ok: bool, detail: impl Into<String>) {
        if ok {
            self.pass(name, detail)
        } else {
            self.fail(name, detail)
        }
    }
    fn f(&self, n: &str) -> String {
        self.data.join(n).to_string_lossy().to_string()
    }
    fn w(&self, n: &str) -> String {
        self.work.join(n).to_string_lossy().to_string()
    }
}

const GHOSTS: [&str; 2] = ["human_22730.Ghost.Gbx", "human_23013.Ghost.Gbx"];
const MAP: &str = "map2.Map.Gbx";
const REPLAY: &str = "replay_kacky_7241.Replay.Gbx";
const POISONED: &str = "poisoned_searchtape.Ghost.Gbx";
/// The declared times the fixtures carry, and what the plain oracle returns.
const DONOR_MS: i64 = 22730;
const REPLAY_MS: i64 = 7241;

/// Checked into the repo at `tools/testdata`, shared by every crate in the
/// workspace so there is one corpus and not one per tool.
const DEFAULT_TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../testdata");

pub fn cmd(a: &[String]) {
    // The fixture directory is resolved AT COMPILE TIME from this crate's own
    // manifest path, not searched for relative to the cwd. The old ladder
    // ("ghostapi/testdata", then "testdata", then "rs/ghostapi/testdata")
    // silently picked a different answer depending on where you stood, and the
    // last rung named a tree that has not existed since the crate moved into
    // this repo — so running the suite from anywhere but one directory failed
    // with "No such file or directory" on every fixture rather than saying the
    // path was wrong.
    let data = PathBuf::from(flag(a, "--data").unwrap_or(DEFAULT_TESTDATA));
    if !data.is_dir() {
        eprintln!("ghost selftest: no fixture directory at {}", data.display());
        std::process::exit(2);
    }
    let work = std::env::temp_dir().join(format!("ghost-selftest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let sd = oracle::server_dir(flag(a, "--server"));
    let server = if sd.join("TrackmaniaServer").exists() { Some(sd) } else { None };
    let mut s = Suite {
        data,
        work,
        server,
        results: Vec::new(),
        strict: has(a, "--strict"),
    };
    println!("ghost selftest");
    println!("  fixtures  {}", s.data.display());
    println!(
        "  server    {}",
        s.server.as_ref().map(|p| p.display().to_string()).unwrap_or("none".into())
    );
    println!("\nPURE -- format, codec and refusals");
    pure_tier(&mut s);
    println!("\nORACLE -- the dedicated server on files this run wrote");
    oracle_tier(&mut s);
    if has(a, "--engine") {
        println!("\nENGINE -- the real physics engine (minutes)");
        engine_tier(&mut s);
    } else {
        println!("\nENGINE -- skipped, pass --engine to run it (it takes several minutes)");
        s.skip("engine.tier", "not requested");
    }

    let n = s.results.len();
    let f = s.results.iter().filter(|r| r.1 == "FAIL").count();
    let k = s.results.iter().filter(|r| r.1 == "SKIP").count();
    println!("\n{} checks: {} passed, {} failed, {} skipped", n, n - f - k, f, k);
    if f > 0 || (s.strict && k > 0) {
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------

/// The GBX header: the second container, and the one no check read.
///
/// These are PURE — no server, no map, no engine — because the defect they
/// guard is a whole family of fields living somewhere the tool did not look,
/// and a check that needs a 30 MB binary to run is a check that gets skipped.
fn header_checks(s: &mut Suite) {
    use gbx::header::{build_user_data, parse_user_data, replace_frames};

    // A plain .Ghost.Gbx has no replay header at all. This is the SCOPE of the
    // whole defect, so it is measured rather than remembered: it is why none
    // of the plain ghosts this project has published can carry it.
    let plain = Container::load(&s.f(GHOSTS[0])).unwrap();
    s.check(
        "header.plain_ghost_has_none",
        parse_user_data(&plain.gbx.user_data).is_none(),
        format!(
            "a plain .Ghost.Gbx has no header chunk table ({} B of user data), so it cannot carry \
             a header identity or a header copy of the declared time",
            plain.gbx.user_data.len()
        ),
    );

    let rp = s.f(REPLAY);
    let replay = Container::load(&rp).unwrap();
    let Some(chunks) = parse_user_data(&replay.gbx.user_data) else {
        s.fail("header.replay_has_one", "the replay fixture has no header chunk table");
        return;
    };
    s.pass(
        "header.replay_has_one",
        format!("the replay fixture carries {} header chunks", chunks.len()),
    );

    // parse -> build is the identity, or every edit below is writing into a
    // structure this code does not actually understand.
    s.check(
        "header.chunks_roundtrip",
        build_user_data(&chunks) == replay.gbx.user_data,
        "parsing the header chunk table and rebuilding it reproduces the bytes exactly",
    );

    // Declining every frame must change nothing. The rewrite is
    // `Fn(&str) -> Option<String>` precisely so that a frame the caller does
    // not recognise is copied through, and this is that promise as a test.
    let mut ident = true;
    for c in &chunks {
        let (out, n) = replace_frames(&c.data, |_| None);
        ident &= n == 0 && out == c.data;
    }
    s.check(
        "header.decline_is_identity",
        ident,
        "a rewrite that recognises no string leaves every header chunk byte-identical",
    );

    // The driver's identity and the run's own time, rewritten -- and the MAP's
    // attribution left alone. On this fixture they are different people, which
    // is the easy case; on 173691 they are the same person and the same 22
    // bytes, which is why the rule is positional.
    let before_ids = crate::hdr::header_driver_identity(&replay);
    let before_map_author = parse_user_data(&replay.gbx.user_data)
        .and_then(|cs| cs.iter().find(|c| c.id == 0x0309_3000).and_then(|c| crate::hdr::parse_3000(&c.data)))
        .map(|p| p.map_author.text);
    match crate::hdr::rewrite(&replay, true, Some("TAS"), Some(1234)) {
        None => s.fail("header.anonymise", "the header rewrite refused a replay"),
        Some(e) => {
            let out = s.w("hdr_anon.Replay.Gbx");
            let mut g = replay.gbx.clone();
            g.user_data = e.user_data;
            gbx::container::write_gbx(&g, replay.body().to_vec(), &out).unwrap();
            let after = Container::load(&out).unwrap();
            let ids = crate::hdr::header_driver_identity(&after);
            let left: Vec<String> =
                ids.iter().filter(|(_, v)| v != "TAS").map(|(w, v)| format!("{} {:?}", w, v)).collect();
            s.check(
                "header.anonymise",
                !ids.is_empty() && left.is_empty(),
                format!(
                    "{} header driver field(s) were {:?} and are now all \"TAS\"",
                    ids.len(),
                    before_ids.iter().map(|(_, v)| v.as_str()).collect::<Vec<_>>()
                ),
            );
            let times = crate::hdr::header_declared_ms(&after);
            s.check(
                "header.declare",
                !times.is_empty() && times.iter().all(|(_, v)| *v == 1234),
                format!(
                    "all {} header copies of the race time now read 1.234 ({:?} before)",
                    times.len(),
                    crate::hdr::header_declared_ms(&replay)
                        .iter()
                        .map(|(_, v)| gbx::container::secs(*v as i64))
                        .collect::<Vec<_>>()
                ),
            );
            let map_author_after = parse_user_data(&after.gbx.user_data)
                .and_then(|cs| {
                    cs.iter().find(|c| c.id == 0x0309_3000).and_then(|c| crate::hdr::parse_3000(&c.data))
                })
                .map(|p| p.map_author.text);
            s.check(
                "header.map_author_untouched",
                map_author_after.is_some() && map_author_after == before_map_author,
                format!(
                    "the MAP's own author is still {:?} -- it is the map's, not the driver's, and \
                     laundering it would be a misattribution the other way",
                    map_author_after.clone().unwrap_or_default()
                ),
            );
        }
    }
}

fn pure_tier(s: &mut Suite) {
    header_checks(s);
    // --- the codec is an identity -------------------------------------------
    for g in GHOSTS.iter().chain([REPLAY, POISONED].iter()) {
        let p = s.f(g);
        match Tape::from_file(&p) {
            Err(e) => s.fail(&format!("codec.identity[{}]", g), e),
            Ok(t) => match t.verbatim_is_identity() {
                Ok(()) => s.pass(
                    &format!("codec.identity[{}]", g),
                    format!(
                        "a verbatim re-encode of {} ticks reproduces the file's own bitstream, byte for byte",
                        t.n()
                    ),
                ),
                Err(e) => s.fail(&format!("codec.identity[{}]", g), e),
            },
        }
    }

    // --- extract -> inject -> extract ---------------------------------------
    for g in GHOSTS.iter() {
        let src = s.f(g);
        let t1 = s.w(&format!("{}.1.gtape", g));
        let out = s.w(&format!("{}.rt.Ghost.Gbx", g));
        let t2 = s.w(&format!("{}.2.gtape", g));
        let tape = Tape::from_file(&src).unwrap();
        std::fs::write(&t1, tape.to_text(&src)).unwrap();
        let re = Tape::from_text(&std::fs::read_to_string(&t1).unwrap());
        match re {
            Err(e) => s.fail(&format!("tape.roundtrip[{}]", g), e),
            Ok(re) => {
                let c = Container::load(&src).unwrap();
                let body = re.splice_into(c.body(), Encoding::Verbatim).unwrap();
                gbx::container::write_gbx(&c.gbx, body, &out).unwrap();
                let back = Tape::from_file(&out).unwrap();
                std::fs::write(&t2, back.to_text(&src)).unwrap();
                let a = std::fs::read_to_string(&t1).unwrap();
                let b = std::fs::read_to_string(&t2).unwrap();
                // the #source line names the file, so compare from the body on
                let strip = |x: &str| -> String {
                    x.lines().filter(|l| !l.starts_with("#source")).collect::<Vec<_>>().join("\n")
                };
                s.check(
                    &format!("tape.roundtrip[{}]", g),
                    strip(&a) == strip(&b),
                    "extract -> inject -> extract is identical, every field of every tick",
                );
                // and the bitstream the injection wrote is the file's own
                let src_bits = Tape::from_file(&src).unwrap().archives[0].orig_bitstream.clone();
                let out_bits = back.archives[0].orig_bitstream.clone();
                s.check(
                    &format!("tape.bitidentity[{}]", g),
                    src_bits == out_bits,
                    format!(
                        "the injected file's input bitstream is byte-identical to the original's ({} B)",
                        src_bits.len()
                    ),
                );
            }
        }
    }

    // --- expansion of "same as the previous tick" packets --------------------
    {
        let src = s.f(GHOSTS[0]);
        let t = Tape::from_file(&src).unwrap();
        let before = t.archives[0].packets.iter().filter(|p| p.vsame).count();
        let c = Container::load(&src).unwrap();
        let out = s.w("expanded.Ghost.Gbx");
        let body = t.splice_into(c.body(), Encoding::Explicit).unwrap();
        gbx::container::write_gbx(&c.gbx, body, &out).unwrap();
        let back = Tape::from_file(&out).unwrap();
        let after = back.archives[0].packets.iter().filter(|p| p.vsame).count();
        let same_values = t.archives[0]
            .packets
            .iter()
            .zip(back.archives[0].packets.iter())
            .all(|(p, q)| p.steer == q.steer && p.accel == q.accel && p.brake == q.brake);
        s.check(
            "tape.expand",
            before > 0 && after == 0 && same_values,
            format!(
                "{} one-bit \"same as previous tick\" packets expanded to explicit fields; {} left; every input value unchanged",
                before, after
            ),
        );
        s.check(
            "tape.expand.writable",
            back.archives[0].packets.len() == t.n(),
            format!("all {} ticks are now individually writable", t.n()),
        );
    }

    // --- a respawn is an editable input --------------------------------------
    {
        let src = s.f(GHOSTS[0]);
        let mut t = Tape::from_file(&src).unwrap();
        // find a packet with an explicit literal and set its respawn bit
        let idx = t.archives[0]
            .packets
            .iter()
            .position(|p| matches!(p.state, StateEnc::Lit(_)));
        match idx {
            None => s.skip("tape.respawn", "this fixture has no explicit state literal"),
            Some(i) => {
                let before = t.archives[0].packets[i].respawn();
                if let StateEnc::Lit(l) = t.archives[0].packets[i].state {
                    t.archives[0].packets[i].state = StateEnc::Lit(l | (1 << 31));
                }
                let c = Container::load(&src).unwrap();
                let out = s.w("respawn.Ghost.Gbx");
                let body = t.splice_into(c.body(), Encoding::Explicit).unwrap();
                gbx::container::write_gbx(&c.gbx, body, &out).unwrap();
                let back = Tape::from_file(&out).unwrap();
                s.check(
                    "tape.respawn",
                    !before && back.archives[0].packets[i].respawn(),
                    format!("respawn written at tick {} and read back (bit 31 of the state literal)", i),
                );
            }
        }
        // and a respawn on a repeated word is refused rather than dropped
        let txt = "#gtape 1\n@archive 0 format_version=12 field0=0 start_offset_ms=0 packets=1 bitstream_bytes=0 bits_used=0\nt=0 mode=2 w=prev respawn=1 mouse=none vsame=0 steer=0 accel=0 brake=0\n";
        s.check(
            "tape.respawn.refused",
            Tape::from_text(txt).is_err(),
            "respawn=1 on a w=prev packet is REFUSED, not silently dropped -- the bit only exists in a literal",
        );
    }

    // --- the recorded steer byte -------------------------------------------
    {
        let cases: [(i8, u8); 7] = [(0, 127), (127, 255), (-127, 0), (60, 187), (4, 131), (-101, 26), (-22, 105)];
        let ok = cases.iter().all(|(v, b)| crate::regen::steer_byte(*v) == *b);
        s.check(
            "record.steerbyte",
            ok,
            "byte 14 = floor((steer + 127) * 255 / 254) -- the FLOOR and the 254 both matter, and both were measured",
        );
    }

    // --- identity ------------------------------------------------------------
    {
        let c = Container::load(&s.f(GHOSTS[0])).unwrap();
        let f = ident::scan(&c);
        let get = |r: Role| f.iter().find(|x| x.role == r).map(|x| x.s.clone());
        let ok = get(Role::Nickname).as_deref() == Some("hobbi.")
            && get(Role::Trigram).as_deref() == Some("HOB")
            && get(Role::Zone).as_deref() == Some("World|Europe|Germany")
            && get(Role::AccountId).as_deref() == Some("I0PKZ6d8R8iDvXT5nG8KNw")
            && get(Role::Skin).map_or(false, |x| x.contains("Hans Sub Red"))
            && get(Role::Locator).map_or(false, |x| x.starts_with("https://core.trackmania"));
        s.check(
            "identity.read",
            ok,
            "display name, trigram, zone, skin, locator URL and the inline account id are all found",
        );
        // the account id lives in a NON-skippable chunk; a scan that only walks
        // skippable chunks misses it, which is how one survives an anonymiser
        s.check(
            "identity.inline_chunk",
            get(Role::AccountId).is_some() && !c.chunks().iter().any(|k| k.0 == 0x0309200F),
            "the account id is found in inline chunk 0x0309200F, which has no PIKS marker and is not in the chunk table",
        );
    }

    // --- the map is inside the replay ---------------------------------------
    {
        let rc = Container::load(&s.f(REPLAY)).unwrap();
        let gc = Container::load(&s.f(GHOSTS[0])).unwrap();
        s.check(
            "map.embedded.detect",
            rc.embedded_map().is_some() && gc.embedded_map().is_none(),
            format!(
                "the replay carries a {} B map (chunk 0x03093002, no PIKS marker); the pure ghost carries none",
                rc.embedded_map().map(|x| x.1).unwrap_or(0)
            ),
        );
        // put the carried map back and require the body to be unchanged
        let mo = rc.embedded_map().unwrap();
        let mp = s.w("carried.Map.Gbx");
        std::fs::write(&mp, &rc.body()[mo.0..mo.0 + mo.1]).unwrap();
        let out = s.w("same.Replay.Gbx");
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["map", "set", &s.f(REPLAY), &out, "--map", &mp])
            .output()
            .unwrap();
        let same = Container::load(&out).map(|c2| c2.body() == rc.body()).unwrap_or(false);
        s.check(
            "map.set.roundtrip",
            st.status.success() && same,
            "replacing the carried map with itself leaves the body byte-identical",
        );
    }

    // --- trim ---------------------------------------------------------------
    {
        let src = s.f(GHOSTS[0]);
        let out = s.w("trim15.Ghost.Gbx");
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["trim", &src, &out, "--to", "15000", "--no-oracle"])
            .output()
            .unwrap();
        if !st.status.success() {
            s.fail("trim.coherent", String::from_utf8_lossy(&st.stderr).to_string());
        } else {
            let t = Tape::from_file(&out).unwrap();
            let c = Container::load(&out).unwrap();
            let d = gbx::record::decode_ghost(&out).unwrap();
            let a0 = &t.archives[0];
            let last_tick_ms = a0.start_offset_ms as i64 + 10 * (a0.packets.len() as i64 - 1);
            let decl: Vec<u32> = c.declared_times().into_iter().map(|x| x.1).collect();
            let cps = c.splits();
            let ok = last_tick_ms <= 15000
                && d.samples.last().map_or(false, |x| x.time_ms <= 15000)
                && d.end_ms <= 15000
                && decl.iter().all(|v| *v == 15000)
                && cps.iter().all(|v| *v <= 15000);
            s.check(
                "trim.coherent",
                ok,
                format!(
                    "tape ends {}, last sample {}, record span ends {}, declared {:?}, checkpoints {:?} -- all inside the window",
                    secs(last_tick_ms),
                    secs(d.samples.last().map(|x| x.time_ms).unwrap_or(0) as i64),
                    secs(d.end_ms as i64),
                    decl.iter().map(|v| secs(*v as i64)).collect::<Vec<_>>(),
                    cps.iter().map(|v| secs(*v as i64)).collect::<Vec<_>>()
                ),
            );
        }
    }

    // --- trim, the other way: LENGTHEN --------------------------------------
    //
    // `u02 extend` had 15 callsites and no coherence at all; when u02 was
    // deleted the capability went with it, and the 173691 landing work needed a
    // 7000-tick tape to give the car room to brake after touchdown. The claim
    // being checked here is not "it got longer" -- it is that everything ELSE
    // stayed put: the appended ticks hold the last recorded input, carry no
    // respawn, and the declared time, the splits and the telemetry are
    // untouched, because ticks after the finish change none of them.
    {
        let src = s.f(GHOSTS[0]);
        let t0 = Tape::from_file(&src).unwrap();
        let c0 = Container::load(&src).unwrap();
        let d0 = gbx::record::decode_ghost(&src).unwrap();
        let a0 = &t0.archives[0];
        // a window that ends 7000 ticks after the tape starts
        let want = 7000i64;
        let to = a0.start_offset_ms as i64 + 10 * (want - 1);
        let out = s.w("extend7000.Ghost.Gbx");
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["trim", &src, &out, "--to", &to.to_string(), "--no-oracle"])
            .output()
            .unwrap();
        if !st.status.success() {
            s.fail("trim.extend", String::from_utf8_lossy(&st.stderr).to_string());
        } else {
            let t = Tape::from_file(&out).unwrap();
            let c = Container::load(&out).unwrap();
            let d = gbx::record::decode_ghost(&out).unwrap();
            let a = &t.archives[0];
            let added = want as usize - a0.packets.len();
            let src_last = &a.packets[a0.packets.len() - 1];
            let held = a.packets[a0.packets.len()..].iter().all(|p| {
                p.steer == src_last.steer
                    && p.accel == src_last.accel
                    && p.brake == src_last.brake
                    && !p.respawn()
            });
            let ok = a.packets.len() == want as usize
                && a.start_offset_ms == a0.start_offset_ms
                && held
                && t.verbatim_is_identity().is_ok()
                && c.splits() == c0.splits()
                && c.declared_times().iter().map(|x| x.1).collect::<Vec<_>>()
                    == c0.declared_times().iter().map(|x| x.1).collect::<Vec<_>>()
                && d.samples.len() == d0.samples.len()
                && d.end_ms == d0.end_ms;
            s.check(
                "trim.extend",
                ok,
                format!(
                    "{} -> {} ticks (+{}), the appended ticks hold steer {} accel {} brake {} with no respawn, \
                     codec identity OK, declared and splits unchanged, telemetry {} samples spanning to {} untouched",
                    a0.packets.len(),
                    a.packets.len(),
                    added,
                    src_last.steer_i8(),
                    src_last.accel,
                    src_last.brake,
                    d.samples.len(),
                    secs(d.end_ms as i64)
                ),
            );
        }
    }

    // --- declare --cps: the NUMBER of split entries --------------------------
    //
    // `--time` writes every copy of the declared time; it could not change how
    // MANY checkpoints the file claims, so a container borrowed from another
    // map went on declaring the donor's checkpoint list. Changing the count
    // changes the chunk's LENGTH, which moves every byte after it -- so the
    // check is that the file still reads as a file afterwards, in both
    // directions, and that the count, the entry list and the race time agree.
    {
        let src = s.f(GHOSTS[0]);
        let c0 = Container::load(&src).unwrap();
        let before = c0.splits().len();
        let time = c0.declared_times().first().map(|x| x.1).unwrap_or(0) as i64;
        let mut detail = Vec::new();
        let mut ok = true;
        for n in [before + 2, 1, before] {
            let out = s.w(&format!("cps{}.Ghost.Gbx", n));
            let st = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["declare", &src, &out, "--time", &time.to_string(), "--cps", &n.to_string()])
                .output()
                .unwrap();
            if !st.status.success() {
                ok = false;
                detail.push(format!("--cps {} FAILED: {}", n, String::from_utf8_lossy(&st.stderr)));
                continue;
            }
            let c = Container::load(&out).unwrap();
            let r = c.result().expect("a result chunk");
            let good = r.entries.len() == n
                && r.race_ms as i64 == time
                && r.checkpoints().last() == Some(&(time as i32))
                && Tape::from_file(&out).map(|t| t.n()).unwrap_or(0) == Tape::from_file(&src).unwrap().n();
            ok &= good;
            detail.push(format!("{} -> {} entries, race {}", before, r.entries.len(), secs(r.race_ms as i64)));
        }
        s.check("declare.cps", ok, detail.join("; "));
    }

    // --- the tape/record agreement separates the two populations -------------
    {
        let good = crate::verify::tape_record_agreement(&s.f(GHOSTS[0]));
        let bad = crate::verify::tape_record_agreement(&s.f(POISONED));
        match (good, bad) {
            (Some((kg, _, _, _)), Some((kb, _, _, _))) => {
                s.check(
                    "record.agreement",
                    kg > 0.9 && kb < 0.3,
                    format!(
                        "kappa {:.3} on a recording that belongs to its tape, {:.3} on the file this project itself labelled DO_NOT_PUBLISH",
                        kg, kb
                    ),
                );
            }
            _ => s.fail("record.agreement", "could not measure one of the two fixtures"),
        }
    }

    // --- the oracle parser, on canned server transcripts ---------------------
    //
    // These need no server and no map, which is the point: the parser is the
    // one piece of this tool that had a live bug, and a check that needs a
    // 30 MB binary and a running process is a check that gets skipped.
    //
    // BOTH FIXTURES ARE ASYMMETRIC ON PURPOSE. On a file that passes, the
    // simulated and declared times are equal, so an equal-number transcript
    // cannot fail whatever the parser does -- it would pin nothing.
    {
        let finish = r#"[
{
  "ValidatedResult" : {
    "NbCheckpoints" : 3,
    "NbRespawns" : 0,
    "Time" : 23081,
    "Score" : 0
  },
  "Desc" : "race finished, time is worse. (22963 < 23081)\n",
  "IsValid" : false,
  "DeclaredResult" : {
    "NbCheckpoints" : 3,
    "NbRespawns" : 0,
    "Time" : 22963,
    "Score" : 0
  },
  "Inputs" : "31_151C101E",
  "MapUid" : "someuid",
  "FileName" : "a.Ghost.Gbx"
}
]"#;
        let v = oracle::parse_many(finish);
        s.check(
            "oracle.parse.finish",
            v.len() == 1
                && v[0].time_ms == Some(23081)
                && v[0].declared_ms == Some(22963)
                && v[0].file == "a.Ghost.Gbx"
                && !v[0].declaration_holds(),
            "a transcript that SIMULATES 23.081 and DECLARES 22.963 reads as 23.081 -- the world, not the claim",
        );

        let dnf = r#"[
{
  "ValidatedResult" : null,
  "Desc" : "wrong simu, but reached some checkpoints (2 out of 4)\n",
  "IsValid" : false,
  "DeclaredResult" : {
    "NbCheckpoints" : 4,
    "NbRespawns" : 0,
    "Time" : 23074,
    "Score" : 0
  },
  "Inputs" : "31_151C",
  "FileName" : "b.Ghost.Gbx"
}
]"#;
        let v = oracle::parse_many(dnf);
        s.check(
            "oracle.parse.dnf",
            v.len() == 1 && v[0].time_ms.is_none() && v[0].declared_ms == Some(23074),
            "a null ValidatedResult with a DeclaredResult of 23.074 reads as DNF, never as a 23.074 finish",
        );
        s.check(
            "oracle.parse.dnf_cps",
            v.len() == 1 && v[0].cps == Some(2),
            "and the checkpoint count comes out of the Desc, which is the only place it exists on a DNF -- a search ladder needs it",
        );

        // The "never crossed the line" sentinel: a huge u32 in a time field.
        // Read as a value it is a finish at 4 294 967.295 s -- a DNF reported
        // as the slowest run in history, and worse, a Some() where every caller
        // tests for None. `tmmaps` cleaned it in its own driver for exactly
        // this reason; the check moved here with the parser.
        let sentinel = r#"[
{
  "ValidatedResult" : {
    "NbCheckpoints" : 0,
    "Time" : 4294967295,
    "Score" : 0
  },
  "Desc" : "wrong simu\n",
  "IsValid" : false,
  "DeclaredResult" : {
    "NbCheckpoints" : 3,
    "Time" : -1,
    "Score" : 0
  },
  "FileName" : "c.Ghost.Gbx"
}
]"#;
        let v = oracle::parse_many(sentinel);
        s.check(
            "oracle.parse.sentinel",
            v.len() == 1 && v[0].time_ms.is_none() && v[0].declared_ms.is_none(),
            "a 4294967295 time is the server's never-crossed sentinel, not a finish 49 days long, and a negative declared time is not a time either",
        );

        let two = format!("{}\n{}", finish, dnf);
        let v = oracle::parse_many(&two);
        s.check(
            "oracle.parse.batch",
            v.len() == 2 && v[0].file == "a.Ghost.Gbx" && v[1].file == "b.Ghost.Gbx" && v[1].time_ms.is_none(),
            "two files in one transcript stay keyed to their own names, and one file's state does not leak into the next",
        );
    }

    // --- a whole-file image, uncompressed and deterministic -------------------
    {
        let src = s.f(GHOSTS[0]);
        let c = Container::load(&src).unwrap();
        let t = Tape::from_file(&src).unwrap();
        let a = t.inject_into(&c, Encoding::Explicit).unwrap();
        let b = t.inject_into(&c, Encoding::Explicit).unwrap();
        s.check(
            "tape.inject_into",
            a == b && a.len() > 1000 && a[7] == b'U',
            format!(
                "the whole file as {} bytes, body UNCOMPRESSED ('U'), and the same tape twice gives the same bytes -- which is what makes a patchable base image possible",
                a.len()
            ),
        );
    }
    {
        // injecting a tape of the wrong length must be refused
        let src = s.f(GHOSTS[0]);
        let t = Tape::from_file(&src).unwrap();
        let mut short = t.clone();
        short.archives[0].packets.truncate(100);
        let tp = s.w("short.gtape");
        std::fs::write(&tp, short.to_text("short")).unwrap();
        let out = s.w("short.Ghost.Gbx");
        let st = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["tape", "inject", &src, &out, "--tape", &tp])
            .output()
            .unwrap();
        s.check(
            "refuse.length_mismatch",
            !st.status.success(),
            "injecting a 100-tick tape into a 2432-tick container is refused, not padded",
        );
        // rebinding a file that carries its own map must be refused
        let out2 = s.w("rebound.Replay.Gbx");
        let st2 = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["map", "rebind", &s.f(REPLAY), &out2, "--map", &s.f(MAP)])
            .output()
            .unwrap();
        s.check(
            "refuse.rebind_replay",
            !st2.status.success(),
            "rewriting the uid of a file that CARRIES a map is refused -- it would claim one map and run another",
        );
    }
}

// ---------------------------------------------------------------------------

fn oracle_tier(s: &mut Suite) {
    let Some(server) = s.server.clone() else {
        s.skip("oracle.tier", "no dedicated server (set TM_SERVER or --server DIR)");
        return;
    };
    let map = s.f(MAP);
    let mapp = Path::new(&map);

    // O1 the donor validates to what it declares
    let r = oracle::validate(&server, Path::new(&s.f(GHOSTS[0])), MapsMode::One(mapp), "o1");
    match r {
        Ok(v) => s.check(
            "oracle.donor",
            v.time_ms == Some(DONOR_MS),
            format!("the donor re-simulates to {} (declared {})", v.secs(), secs(DONOR_MS)),
        ),
        Err(e) => s.fail("oracle.donor", e),
    }

    // O2 expansion is a semantic no-op
    {
        let out = s.w("o2.Ghost.Gbx");
        run_self(&["tape", "expand", &s.f(GHOSTS[0]), &out]);
        match oracle::validate(&server, Path::new(&out), MapsMode::One(mapp), "o2") {
            Ok(v) => s.check(
                "oracle.expand_noop",
                v.time_ms == Some(DONOR_MS),
                format!("expanding every same-as-previous packet leaves the run at {}", v.secs()),
            ),
            Err(e) => s.fail("oracle.expand_noop", e),
        }
    }

    // O3 a full extract -> inject round trip is a no-op
    {
        let tp = s.w("o3.gtape");
        let out = s.w("o3.Ghost.Gbx");
        run_self(&["tape", "extract", &s.f(GHOSTS[0]), "--out", &tp]);
        run_self(&["tape", "inject", &s.f(GHOSTS[0]), &out, "--tape", &tp]);
        match oracle::validate(&server, Path::new(&out), MapsMode::One(mapp), "o3") {
            Ok(v) => s.check(
                "oracle.inject_noop",
                v.time_ms == Some(DONOR_MS),
                format!("extract -> inject reproduces the run exactly: {}", v.secs()),
            ),
            Err(e) => s.fail("oracle.inject_noop", e),
        }
    }

    // O4 an edited tick actually changes the run (the writer is not a no-op),
    // and the stale declaration it leaves behind is caught
    {
        let src = s.f(GHOSTS[0]);
        let mut t = Tape::from_file(&src).unwrap();
        // a one-unit steer nudge over 300 ms: small enough to still finish,
        // large enough that the finish moves
        for p in t.archives[0].packets.iter_mut().skip(1800).take(30) {
            p.steer = (p.steer_i8().saturating_add(1)) as u8 as u32;
        }
        let c = Container::load(&src).unwrap();
        let out = s.w("o4.Ghost.Gbx");
        let body = t.splice_into(c.body(), Encoding::Explicit).unwrap();
        gbx::container::write_gbx(&c.gbx, body, &out).unwrap();
        let r = oracle::validate(&server, Path::new(&out), MapsMode::One(mapp), "o4");
        match r {
            Ok(ref v) => s.check(
                "oracle.edit_bites",
                v.time_ms == Some(22738),
                format!(
                    "one steer unit for 300 ms moves the finish from {} to {} -- the writer is not a no-op",
                    secs(DONOR_MS),
                    v.secs()
                ),
            ),
            Err(ref e) => s.fail("oracle.edit_bites", e.clone()),
        }
        // THE SERVER PRINTS TWO RESULTS AND THE SECOND IS THE FILE'S OWN CLAIM.
        // This file simulates 22.738 and still declares 22.730, so it is the
        // fixture that catches a parser reading the claim instead of the answer
        // -- which this tool did, until this check was written. NOTE THAT THE
        // TWO NUMBERS MUST DIFFER: on a file that passes they are equal, and no
        // equal-number fixture can fail, whatever the parser does.
        let declared: Vec<u32> = Container::load(&out).unwrap().declared_times().into_iter().map(|x| x.1).collect();
        let r2 = oracle::validate(&server, Path::new(&out), MapsMode::One(mapp), "o4b");
        match r2 {
            Err(e) => s.fail("oracle.reads_the_world", e),
            Ok(v) => {
                s.check(
                    "oracle.reads_the_world",
                    v.time_ms == Some(22738) && v.declared_ms == Some(22730) && declared == vec![22730],
                    format!(
                        "the server reports BOTH: simulated {} and declared {}. They differ, so a parser that took the second would fail this check",
                        v.secs(),
                        v.declared_ms.map(secs).unwrap_or("none".into())
                    ),
                );
                s.check(
                    "oracle.declaration_holds",
                    !v.declaration_holds(),
                    "and `declaration_holds()` says so, which is the one comparison the search layer needs",
                );
            }
        }
        let vr = crate::verify::run(&out, &["--map".into(), map.clone(), "--server".into(), server.to_string_lossy().to_string()]);
        s.check(
            "verify.stale_declaration",
            vr.failed(),
            "and `ghost verify` REFUSES that file, because a run that declares one time and does another is exactly the container bug this project keeps paying for",
        );
    }

    // O4c THE OTHER ASYMMETRIC SHAPE: a DNF whose DeclaredResult still carries
    // a time. `ValidatedResult` is null and `DeclaredResult` says 15.000, so a
    // parser that reads any `"Time"` reports a 15.000 finish for a run that
    // never finished. The trimmed file is exactly that fixture.
    {
        let out = s.w("o4c.Ghost.Gbx");
        run_self(&["trim", &s.f(GHOSTS[0]), &out, "--to", "15000", "--no-oracle"]);
        match oracle::validate(&server, Path::new(&out), MapsMode::One(mapp), "o4c") {
            Err(e) => s.fail("oracle.dnf_with_declared_time", e),
            Ok(v) => s.check(
                "oracle.dnf_with_declared_time",
                v.time_ms.is_none() && v.declared_ms == Some(15000),
                format!(
                    "ValidatedResult is null and DeclaredResult says {} -- reported as DNF, not as a finish ({})",
                    v.declared_ms.map(secs).unwrap_or("none".into()),
                    v.desc
                ),
            ),
        }
    }

    // O4d BATCH VALIDATION. The server validates in batches and the per-launch
    // cost dominates the per-file cost, so this is the entry point the search
    // layer uses; it has to key every result to the right file.
    {
        let a = s.f(GHOSTS[0]);
        let b = s.f(GHOSTS[1]);
        let c = s.w("o4.Ghost.Gbx");
        let files: Vec<&Path> = vec![Path::new(&a), Path::new(&b), Path::new(&c)];
        match oracle::validate_many(&server, &files, MapsMode::One(mapp), "o4d") {
            Err(e) => s.fail("oracle.batch", e),
            Ok(v) => {
                let by = |n: &str| v.iter().find(|r| r.file.contains(n)).and_then(|r| r.time_ms);
                s.check(
                    "oracle.batch",
                    v.len() == 3
                        && by("human_22730") == Some(DONOR_MS)
                        && by("human_23013") == Some(23013)
                        && by("o4") == Some(22738),
                    format!(
                        "three files in ONE server launch, each keyed to its own name: {}",
                        v.iter().map(|r| format!("{} {}", r.file, r.secs())).collect::<Vec<_>>().join(", ")
                    ),
                );
                // The engine echoes the tape it decoded. Two files with the same
                // tape must produce the same echo, and the edited one must not.
                let inp: Vec<&str> = v.iter().map(|r| r.inputs.as_str()).collect();
                s.check(
                    "oracle.inputs_echo",
                    inp.len() == 3 && !inp[0].is_empty() && inp[0] != inp[1],
                    "the server's own echo of the decoded tape comes back with each result",
                );
            }
        }
    }

    // O5 THE MAP IS INSIDE THE REPLAY -- the empty-Maps control
    {
        match oracle::validate(&server, Path::new(&s.f(REPLAY)), MapsMode::Empty, "o5a") {
            Ok(v) => s.check(
                "oracle.map_inside_replay",
                v.time_ms == Some(REPLAY_MS),
                format!(
                    "the replay validates to {} with ZERO maps on disk -- the map it runs on comes out of the file",
                    v.secs()
                ),
            ),
            Err(e) => s.fail("oracle.map_inside_replay", e),
        }
        match oracle::validate(&server, Path::new(&s.f(GHOSTS[0])), MapsMode::Empty, "o5b") {
            Ok(v) => s.check(
                "oracle.pure_ghost_needs_map",
                v.time_ms.is_none(),
                "the pure ghost returns nothing with an empty Maps directory -- for it, --map is real",
            ),
            Err(e) => s.fail("oracle.pure_ghost_needs_map", e),
        }
    }

    // O6 changing the carried map actually moves the recording
    {
        let out = s.w("o6same.Replay.Gbx");
        let mp = s.w("carried.Map.Gbx");
        run_self(&["map", "set", &s.f(REPLAY), &out, "--map", &mp]);
        match oracle::validate(&server, Path::new(&out), MapsMode::Empty, "o6a") {
            Ok(v) => s.check(
                "oracle.map_set_roundtrip",
                v.time_ms == Some(REPLAY_MS),
                format!("putting the carried map back leaves the answer at {}", v.secs()),
            ),
            Err(e) => s.fail("oracle.map_set_roundtrip", e),
        }
        let out2 = s.w("o6moved.Replay.Gbx");
        run_self(&["map", "set", &s.f(REPLAY), &out2, "--map", &map]);
        match oracle::validate(&server, Path::new(&out2), MapsMode::Empty, "o6b") {
            Ok(v) => s.check(
                "oracle.map_set_moves",
                v.time_ms != Some(REPLAY_MS),
                format!(
                    "swapping a DIFFERENT map in changes the answer to {} -- on a box where the original map does not exist at all",
                    v.secs()
                ),
            ),
            Err(e) => s.fail("oracle.map_set_moves", e),
        }
    }

    // O7 identity edits are cosmetic
    {
        let out = s.w("o7.Ghost.Gbx");
        let st = run_self(&[
            "identity", "set", &s.f(GHOSTS[0]), &out, "--name", "TASvjeux", "--trigram", "VJX",
            "--anonymise", "--map", &map,
        ]);
        let v = oracle::validate(&server, Path::new(&out), MapsMode::One(mapp), "o7");
        let idok = Container::load(&out)
            .map(|c| {
                let f = ident::scan(&c);
                f.iter().find(|x| x.role == Role::Trigram).map(|x| x.s.clone()) == Some("VJX".into())
                    && !f.iter().any(|x| x.role == Role::Locator && !x.s.is_empty())
            })
            .unwrap_or(false);
        s.check(
            "oracle.identity_noop",
            st && idok && v.map(|x| x.time_ms) == Ok(Some(DONOR_MS)),
            "renaming the driver, setting a trigram and dropping the account id and locator URL leaves the run at 22.730",
        );
    }

    // O8 trimming
    {
        let out = s.w("o8.Ghost.Gbx");
        run_self(&["trim", &s.f(GHOSTS[0]), &out, "--to", "15000", "--no-oracle"]);
        match oracle::validate(&server, Path::new(&out), MapsMode::One(mapp), "o8a") {
            Ok(v) => s.check(
                "oracle.trim_partial_dnf",
                v.time_ms.is_none(),
                "a run cut at 15.000 does not finish -- which is the truth, and the tool says so instead of declaring a time",
            ),
            Err(e) => s.fail("oracle.trim_partial_dnf", e),
        }
        let out2 = s.w("o8b.Ghost.Gbx");
        run_self(&["trim", &s.f(GHOSTS[0]), &out2, "--to", "22730", "--no-oracle"]);
        match oracle::validate(&server, Path::new(&out2), MapsMode::One(mapp), "o8b") {
            Ok(v) => s.check(
                "oracle.trim_keeps_finish",
                v.time_ms == Some(DONOR_MS),
                format!("cutting the post-finish tail keeps the finish: {}", v.secs()),
            ),
            Err(e) => s.fail("oracle.trim_keeps_finish", e),
        }
        // O8c LENGTHENING. The control for the whole extension path, and the
        // one thing a read-back gate cannot check: ticks appended after the
        // finish must not change the run. 2432 -> 7000 ticks, and the world
        // must still say 22.730.
        let out3 = s.w("o8c.Ghost.Gbx");
        let so = Tape::from_file(&s.f(GHOSTS[0])).unwrap().archives[0].start_offset_ms as i64;
        let to = so + 10 * (7000 - 1);
        run_self(&["trim", &s.f(GHOSTS[0]), &out3, "--to", &to.to_string(), "--no-oracle"]);
        let n3 = Tape::from_file(&out3).map(|t| t.n()).unwrap_or(0);
        match oracle::validate(&server, Path::new(&out3), MapsMode::One(mapp), "o8c") {
            Ok(v) => s.check(
                "oracle.extend_keeps_finish",
                v.time_ms == Some(DONOR_MS) && n3 == 7000,
                format!(
                    "a tape lengthened to {} ticks -- 45 s of held input after a 22.730 finish -- \
                     still re-simulates to {}",
                    n3,
                    v.secs()
                ),
            ),
            Err(e) => s.fail("oracle.extend_keeps_finish", e),
        }
    }

    // O8d the declared split COUNT, against the server rather than against a
    // belief about it.
    //
    // `--cps` was asked for because a container borrowed from another map
    // declares the donor's checkpoint count, and the server was believed to
    // refuse such a file as `wrong simu` WITHOUT SIMULATING IT. It does not:
    // measured here, on the file that validates at 22.730, a declared count of
    // 5 and a declared count of 1 both come back IsValid with the same time,
    // and so do zeroed intermediate splits. `wrong simu` is what the server
    // says when the simulation does not reproduce the DECLARED RESULT -- on a
    // partial run it even reports how far it got.
    //
    // This check exists so that the belief cannot come back: if a future build
    // does start gating on the count, this is the line that turns red.
    {
        let src = s.f(GHOSTS[0]);
        let mut ok = true;
        let mut detail = Vec::new();
        for n in [5usize, 1] {
            let out = s.w(&format!("o8d{}.Ghost.Gbx", n));
            run_self(&["declare", &src, &out, "--time", "22730", "--cps", &n.to_string()]);
            match oracle::validate(&server, Path::new(&out), MapsMode::One(mapp), "o8d") {
                Ok(v) => {
                    ok &= v.time_ms == Some(DONOR_MS) && v.declared_cps == Some(n as u32);
                    detail.push(format!(
                        "declared {} checkpoints -> validated {} (server echoes {:?})",
                        n,
                        v.secs(),
                        v.declared_cps
                    ));
                }
                Err(e) => {
                    ok = false;
                    detail.push(format!("--cps {}: {}", n, e));
                }
            }
        }
        s.check("oracle.cps_does_not_gate", ok, detail.join("; "));
    }

    // O9 rebinding a pure ghost: the uid IS the binding, proved both ways
    {
        let carried = s.w("carried.Map.Gbx");
        let away = s.w("o9away.Ghost.Gbx");
        let back = s.w("o9back.Ghost.Gbx");
        let ok1 = run_self(&["map", "rebind", &s.f(GHOSTS[0]), &away, "--map", &carried]);
        let ok2 = run_self(&["map", "rebind", &away, &back, "--map", &map]);
        let a = oracle::validate(&server, Path::new(&away), MapsMode::One(mapp), "o9a");
        let b = oracle::validate(&server, Path::new(&back), MapsMode::One(mapp), "o9b");
        match (a, b) {
            (Ok(av), Ok(bv)) => s.check(
                "oracle.rebind",
                ok1 && ok2 && av.time_ms.is_none() && bv.time_ms == Some(DONOR_MS),
                format!(
                    "rebinding the donor AWAY from this map makes it stop validating on it ({}), and rebinding it BACK restores the run exactly ({}) -- so the uid is the binding, and this command sets it",
                    av.secs(),
                    bv.secs()
                ),
            ),
            _ => s.fail("oracle.rebind", "the oracle did not run"),
        }
    }
}

fn engine_tier(s: &mut Suite) {
    if s.server.is_none() {
        s.skip("engine.tier", "no dedicated server");
        return;
    }
    let map = s.f(MAP);
    let src = s.f(GHOSTS[0]);
    // E1 the engine's own run of this tape must match the recording in it
    match crate::regen::engine_trajectory_agreement(&src, &map) {
        Err(e) => s.fail("engine.trajectory", e),
        Ok((mean, worst, n, shift)) => s.check(
            "engine.trajectory",
            !shift && mean < 0.05,
            format!(
                "running the fixture's own tape through the real engine reproduces its recorded trajectory to {:.4} m mean / {:.4} m worst over {} samples{}",
                mean,
                worst,
                n,
                if shift { " -- BUT a whole sample out of phase" } else { "" }
            ),
        ),
    }
    // E2 THE LOCATE IS DETERMINISTIC. Two regenerations of the same file must
    // produce the same trajectory, bit for bit. This is the check that would
    // have failed on the old locate, which found the car about one run in six
    // and wrote a different answer the rest of the time.
    let a = s.w("e2a.Ghost.Gbx");
    let b = s.w("e2b.Ghost.Gbx");
    let ok1 = run_self(&["regen", &src, &a, "--map", &map]);
    let ok2 = run_self(&["regen", &src, &b, "--map", &map]);
    if !ok1 || !ok2 {
        s.fail("engine.determinism", "a regeneration did not pass its own gate");
        return;
    }
    let (da, db) = (
        gbx::record::decode_ghost(&a).ok(),
        gbx::record::decode_ghost(&b).ok(),
    );
    match (da, db) {
        (Some(x), Some(y)) => {
            let n = x.samples.len().min(y.samples.len());
            let mut worst = 0.0f64;
            for i in 0..n {
                let (p, q) = (&x.samples[i], &y.samples[i]);
                worst = worst.max(
                    (((p.x - q.x) as f64).powi(2)
                        + ((p.y - q.y) as f64).powi(2)
                        + ((p.z - q.z) as f64).powi(2))
                    .sqrt(),
                );
            }
            s.check(
                "engine.determinism",
                worst == 0.0 && n > 100,
                format!(
                    "two independent regenerations of the same file agree to {:.6} m over {} samples",
                    worst, n
                ),
            );
        }
        _ => s.fail("engine.determinism", "could not decode one of the two regenerations"),
    }
}

fn run_self(args: &[&str]) -> bool {
    // Spawn failure is REPORTED, not a panic. `current_exe()` is this binary's
    // path, and a path can stop resolving underneath a long run -- rebuilding
    // the workspace while a selftest is in flight makes cargo replace the file,
    // so every later spawn fails with NotFound. That took down the whole engine
    // tier through a bare `unwrap()`, with nothing said about why.
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("       (cannot find my own binary: {e})");
            return false;
        }
    };
    let st = match std::process::Command::new(&exe).args(args).output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "       (could not run `ghost {}`: {e} -- {} {})",
                args[0],
                exe.display(),
                if exe.exists() { "exists" } else { "IS GONE (rebuilt mid-run?)" }
            );
            return false;
        }
    };
    if !st.status.success() {
        eprintln!("       (ghost {} failed: {})", args[0], String::from_utf8_lossy(&st.stderr).trim());
    }
    st.status.success()
}
