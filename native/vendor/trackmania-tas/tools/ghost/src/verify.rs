//! `ghost verify` -- the acceptance gate.
//!
//! Nothing here reads a number off the command line and compares it to itself:
//! every check reads both of its operands out of the FILE or out of the world.
//! A check that cannot get its second operand says NA and says why; it never
//! passes by default.

use gbx::container::{secs, Container};
use crate::ident::{self, Role};
use crate::oracle::{self, MapsMode};
use gbx::tape::Tape;
use crate::cli::{die, flag, has, num};

#[derive(PartialEq, Clone, Copy)]
pub enum Verdict {
    Pass,
    Fail,
    Na,
    Warn,
}

pub struct Check {
    pub id: &'static str,
    pub verdict: Verdict,
    pub msg: String,
}

pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    fn add(&mut self, id: &'static str, v: Verdict, m: impl Into<String>) {
        self.checks.push(Check { id, verdict: v, msg: m.into() });
    }
    pub fn failed(&self) -> bool {
        self.checks.iter().any(|c| c.verdict == Verdict::Fail)
    }
    pub fn print(&self) {
        for c in &self.checks {
            let tag = match c.verdict {
                Verdict::Pass => "PASS",
                Verdict::Fail => "FAIL",
                Verdict::Na => "  NA",
                Verdict::Warn => "WARN",
            };
            println!("{} {:<4} {}", tag, c.id, c.msg);
        }
    }

    /// The same report as JSON, for callers rather than people.
    ///
    /// The prose above is a human rendering and has always been treated as
    /// one -- but `clip`'s audit was READING it, line by line, splitting on
    /// the literal strings "kappa ", "file: " and "re-simulated" to recover
    /// numbers this struct already holds. Reword a gate message and that
    /// caller silently reports `None` instead of failing, which is the worst
    /// way for a check to break.
    ///
    /// Emitted by hand: this workspace has one third-party dependency and a
    /// serialiser is not worth becoming the second. Strings are escaped for
    /// the characters a gate message can actually contain.
    pub fn json(&self) -> String {
        let esc = |s: &str| -> String {
            let mut o = String::with_capacity(s.len() + 8);
            for ch in s.chars() {
                match ch {
                    '"' => o.push_str("\\\""),
                    '\\' => o.push_str("\\\\"),
                    '\n' => o.push_str("\\n"),
                    '\t' => o.push_str("\\t"),
                    '\r' => {}
                    c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                    c => o.push(c),
                }
            }
            o
        };
        let mut o = String::from("{\"checks\": [");
        for (i, c) in self.checks.iter().enumerate() {
            if i > 0 {
                o.push_str(", ");
            }
            let v = match c.verdict {
                Verdict::Pass => "pass",
                Verdict::Fail => "fail",
                Verdict::Na => "na",
                Verdict::Warn => "warn",
            };
            o.push_str(&format!(
                "{{\"id\": \"{}\", \"verdict\": \"{}\", \"message\": \"{}\"}}",
                c.id,
                v,
                esc(&c.msg)
            ));
        }
        let n = |v: Verdict| self.checks.iter().filter(|c| c.verdict == v).count();
        o.push_str(&format!(
            "], \"pass\": {}, \"fail\": {}, \"warn\": {}, \"na\": {}}}",
            n(Verdict::Pass),
            n(Verdict::Fail),
            n(Verdict::Warn),
            n(Verdict::Na)
        ));
        o
    }
}

/// V6: does the RECORDING agree with the TAPE the same file carries?
///
/// One `.Ghost.Gbx` holds the driver's inputs twice: the input chunk at 10 ms,
/// and byte 14 of every 50 ms telemetry sample. A synthesised tape grafted onto
/// a donor carries the DONOR's telemetry, and no positional check can see that
/// -- but these two channels stop agreeing immediately, and comparing them
/// needs no reference file at all.
///
/// The statistic is COHEN'S KAPPA on the exact byte, not a raw hit rate, and
/// the difference matters: a bang-bang keyboard tape sits at three steer values
/// for most of a run, so two UNRELATED runs of it agree on 40 % of samples by
/// chance alone, and a raw rate cannot tell that from a real recording. Kappa
/// divides out that chance agreement -- `(p_obs - p_chance) / (1 - p_chance)`
/// with `p_chance` computed from each channel's own marginals -- so a genuine
/// recording scores near 1 and a foreign one scores near 0 whatever the tape
/// looks like.
///
/// Byte 14 is `floor((steer_i8 + 127) * 255 / 254)`, measured against the whole
/// corpus rather than assumed.
///
/// Returns (kappa, raw hit rate, best lag ms, samples compared).
pub fn tape_record_agreement(path: &str) -> Option<(f64, f64, i64, usize)> {
    let t = Tape::from_file(path).ok()?;
    let d = gbx::record::decode_ghost(path).ok()?;
    if d.samples.is_empty() {
        return None;
    }
    let (ss, raw) = crate::regen::raw_vehicle_samples(path).ok()?;
    let a0 = &t.archives[0];
    let so = a0.start_offset_ms as i64;
    let mut best: Option<(f64, f64, i64, usize)> = None;
    for lag in -10i64..=10 {
        let mut pairs: Vec<(u8, u8)> = Vec::new();
        for (i, s) in d.samples.iter().enumerate() {
            let idx = (s.time_ms as i64 + lag * 10 - so) / 10;
            if idx < 0 || idx >= a0.packets.len() as i64 || (i + 1) * ss > raw.len() {
                continue;
            }
            pairs.push((
                crate::regen::steer_byte(a0.packets[idx as usize].steer_i8()),
                raw[i * ss + 14],
            ));
        }
        let n = pairs.len();
        if n < 20 {
            continue;
        }
        let mut ma = [0u32; 256];
        let mut mb = [0u32; 256];
        let mut hit = 0usize;
        for (x, y) in &pairs {
            ma[*x as usize] += 1;
            mb[*y as usize] += 1;
            if x == y {
                hit += 1;
            }
        }
        let p_obs = hit as f64 / n as f64;
        let p_exp: f64 = (0..256)
            .map(|v| (ma[v] as f64 / n as f64) * (mb[v] as f64 / n as f64))
            .sum();
        let kappa = if (1.0 - p_exp).abs() < 1e-12 { 1.0 } else { (p_obs - p_exp) / (1.0 - p_exp) };
        if best.map_or(true, |b| kappa > b.0) {
            best = Some((kappa, p_obs, lag * 10, n));
        }
    }
    best
}

pub fn run(path: &str, a: &[String]) -> Report {
    let mut r = Report { checks: Vec::new() };
    let c = match Container::load(path) {
        Ok(c) => c,
        Err(e) => {
            r.add("V0", Verdict::Fail, e);
            return r;
        }
    };

    // ---- V1 codec identity ------------------------------------------------
    match Tape::from_file(path) {
        Err(e) => r.add("V1", Verdict::Na, format!("no input tape in this file ({})", e)),
        Ok(t) => match t.verbatim_is_identity() {
            Ok(()) => r.add(
                "V1",
                Verdict::Pass,
                format!(
                    "codec identity: a verbatim re-encode of all {} ticks reproduces the file's own bitstream",
                    t.n()
                ),
            ),
            Err(e) => r.add("V1", Verdict::Fail, format!("codec identity: {}", e)),
        },
    }

    // ---- V2 declared-time census -----------------------------------------
    // THE HEADER COUNTS TOO. A .Replay.Gbx holds the race time twice more, in
    // header chunk 0x03093000 and in the header XML, and this census used to
    // read only the body -- so it printed "1 copies, all 36.049" about a file
    // whose header still said 49.958. A count of a set you cannot see all of
    // is worse than no count.
    let dts = c.declared_times();
    let hdr_ts = crate::hdr::header_declared_ms(&c);
    let mut vals: Vec<u32> = dts.iter().map(|x| x.1).chain(hdr_ts.iter().map(|x| x.1)).collect();
    let total = vals.len();
    vals.sort();
    vals.dedup();
    let where_ = format!(
        "{} in the body, {} in the header",
        dts.len(),
        hdr_ts.len()
    );
    if total == 0 {
        r.add("V2", Verdict::Na, "this container declares no race time");
    } else if vals.len() > 1 {
        let detail: Vec<String> = dts
            .iter()
            .map(|(o, v)| format!("body@{} {}", o, secs(*v as i64)))
            .chain(hdr_ts.iter().map(|(w, v)| format!("{} {}", w, secs(*v as i64))))
            .collect();
        r.add(
            "V2",
            Verdict::Fail,
            format!(
                "declared-time census: {} copies ({}) carrying {} DIFFERENT times -- {}",
                total,
                where_,
                vals.len(),
                detail.join(", ")
            ),
        );
    } else {
        r.add(
            "V2",
            Verdict::Pass,
            format!("declared-time census: {} copies ({}), all {}", total, where_, secs(vals[0] as i64)),
        );
    }

    // ---- V3 container identity -------------------------------------------
    let fields = ident::scan(&c);
    // FOUR IDENTIFIERS THAT NAME A PERSON, in the body and in the header.
    //
    // The account id and the locator uuid were the known two. The RANKED BADGE
    // (`Prestige=Yes&Level=...&Medal=...`) and the ZONE (a country) were found
    // on 2026-08-22 by raw-stringing published files: 16 files across 5 maps
    // carry a stranger's badge and 21 carry a stranger's country, neither on
    // anyone's strip-list.
    //
    // The badge is cleared by `--anonymise`. The ZONE DELIBERATELY IS NOT: it
    // is the landmark this scanner locates the trigram and the club tag by
    // (`World|...` is the only self-identifying string in that block), so
    // blanking it makes both unfindable -- the suite caught that immediately,
    // asking for trigram VJX and reading back None. Trading one named leak for
    // two silent ones is a bad deal, so the zone is REPORTED here instead: a
    // leak that is named is a decision, a leak that is not is an accident.
    let mut foreign: Vec<String> = fields
        .iter()
        .filter(|f| {
            matches!(f.role, Role::AccountId | Role::Locator | Role::Prestige) && !f.s.is_empty()
        })
        .map(|f| format!("body {} {:?}", f.role.label(), f.s))
        .collect();
    let zone: Vec<String> = fields
        .iter()
        .filter(|f| f.role == Role::Zone && !f.s.is_empty())
        .map(|f| format!("{:?}", f.s))
        .collect();
    // ...and the driver identity in the header, which this check could not see.
    // The map's own author block is deliberately NOT in this list: it is the
    // map's, it stays, and laundering it would be a misattribution the other
    // way. `hdr::header_driver_identity` draws that line structurally.
    let hdr_id = crate::hdr::header_driver_identity(&c);
    for (wh, v) in &hdr_id {
        if !v.is_empty() && v != "TAS" && !v.chars().all(|ch| ch == 'x') {
            foreign.push(format!("header {} {:?}", wh, v));
        }
    }
    if foreign.is_empty() && zone.is_empty() {
        r.add(
            "V3",
            Verdict::Pass,
            format!(
                "container identity: no account id, locator, badge or zone in the body{}",
                if hdr_id.is_empty() {
                    " (this container has no replay header to carry one)".to_string()
                } else {
                    format!(", and the {} header driver field(s) are ours", hdr_id.len())
                }
            ),
        );
    } else if foreign.is_empty() {
        r.add(
            "V3",
            Verdict::Warn,
            format!(
                "container identity: no account id, locator or badge, but this file still \
                 declares a zone ({}) -- the container donor's country. `--anonymise` leaves it \
                 because it is the landmark the trigram and club tag are found by; clear it \
                 explicitly with `--zone \"\"` only if you have checked those still read back.",
                zone.join(", ")
            ),
        );
    } else {
        r.add(
            "V3",
            Verdict::Warn,
            format!(
                "container identity: this file still carries {} -- run `ghost identity set IN OUT --anonymise` before publishing it as ours",
                foreign.join(", ")
            ),
        );
    }

    // ---- V4 which map does this actually run on --------------------------
    match c.embedded_map() {
        Some((_, n)) => {
            let mut m = format!("embedded map: {} B -- the server simulates THIS copy, --map is decoration", n);
            if let Some(b) = c.embedded_map_bytes() {
                if let Some(u) = crate::map_uid_of(&b) {
                    m.push_str(&format!(" (uid {})", u));
                }
            }
            r.add("V4", Verdict::Warn, m);
        }
        None => r.add("V4", Verdict::Pass, "no embedded map: --map is real for this file"),
    }

    // ---- V10 the record's OWN declared span ---------------------------------
    //
    // V5 asks whether the car's last SAMPLE outlives the declared time. That
    // misses the case that cost a whole afternoon: the samples are right and
    // the RECORD NODE's own span is the carrier's.
    //
    // 286279's published `BEST_218812` declares `span 0.000 .. 441.000` for a
    // 218.812 run whose car stops at 217.95, because the record was
    // regenerated in the container of Bald_tm's 441.002 recording and the span
    // was inherited. Two symptoms, one cause, neither of them obviously about a
    // span: the MediaTracker renders a clip as long as its longest block, so
    // the video came out 441 s -- twice the run -- and when the camera's target
    // entity ran out at 218 s the camera drifted to the top of the map and
    // stayed there for the remaining half of the clip. It was reported as "the
    // camera flies away".
    //
    // The same shape hides a foreign entity: a carrier's non-vehicle entity
    // (0x2D001000, 13 bytes a sample) also inherits its own length, and 8820
    // samples of it is the 441 s that keeps the scene alive after our car has
    // gone. `ghost record rebuild` drops those; this is the check that says
    // when it is needed.
    match gbx::record::decode_ghost(path) {
        Err(_) => {}
        Ok(d) => {
            let last = d.samples.last().map(|s| s.time_ms).unwrap_or(0) as i64;
            let end = d.end_ms as i64;
            let over: Vec<String> = d
                .ents
                .iter()
                .filter(|e| e.t_last.unwrap_or(0) as i64 > last + 2000)
                .map(|e| {
                    format!(
                        "0x{:08X} runs to {}",
                        e.class_id.unwrap_or(0),
                        secs(e.t_last.unwrap_or(0) as i64)
                    )
                })
                .collect();
            if end > last + 2000 || !over.is_empty() {
                r.add(
                    "V10",
                    Verdict::Fail,
                    format!(
                        "the record declares a span of {} .. {} but the car's last sample is at \
                         {}{}. A scene built from this file outlives the run: the render is as \
                         long as the longest block and the camera loses its target when the car \
                         ends. Rebuild the record with `ghost record rebuild IN OUT --span {}`.",
                        secs(d.start_ms as i64),
                        secs(end),
                        secs(last),
                        if over.is_empty() {
                            String::new()
                        } else {
                            format!(", and {} other entity/entities outlive it ({})", over.len(), over.join(", "))
                        },
                        last
                    ),
                );
            } else {
                r.add(
                    "V10",
                    Verdict::Pass,
                    format!(
                        "the record's span ends at {} and the car's last sample is at {} -- \
                         nothing in this file outlives the run",
                        secs(end),
                        secs(last)
                    ),
                );
            }
        }
    }

    // ---- V11 the container's LIVE non-vehicle records ----------------------
    //
    // THE ONE DEFECT IN THIS GATE THAT NO HEADLESS TEST CAN SEE — and a
    // WARNING rather than a failure, because the rule is not established and
    // this project's own corpus refutes the obvious form of it.
    //
    // MEASURED, on the render box 2026-08-23, every run behind a same-session
    // control that read `scene ready`: 173691's film file, regenerated, crashed
    // the game client on import three times across two revisions; **grafting
    // the container's one live `0x2D001000` record back made it import**, twice,
    // in two variants; the container itself imports untouched; and three other
    // single-field repairs — the car entity's `u01`, the declared checkpoint
    // count, and the ghost-result race time — each crashed with the graft
    // absent. 227654's `TAS_57482` is a second file of the same shape that
    // crashes.
    //
    // AND WHAT REFUTES THE OBVIOUS RULE: `TAS_67319` has no live non-vehicle
    // record either, and imports cleanly. So "a ghost needs one" is FALSE. The
    // honest statement is narrower: **restoring one repaired both files that
    // crash, and nothing yet says what makes those two different from the
    // sixty-odd published files in the same shape that are fine.** A gate that
    // failed here would fail all of them.
    //
    // No oracle can help: the dedicated server re-simulates the input chunk and
    // never reads the scene, so `TAS_57482` passes every other check here and
    // re-simulates to 57.482. A client import is the only instrument there is.
    match gbx::record::decode_ghost(path) {
        Err(_) => {}
        Ok(d) => {
            let live: Vec<String> = d
                .ents
                .iter()
                .filter(|e| {
                    e.class_id != Some(gbx::record::CLASS_CSCENEVEHICLEVIS) && e.n_samples > 0
                })
                .map(|e| format!("0x{:08X} x{}", e.class_id.unwrap_or(0), e.n_samples))
                .collect();
            if live.is_empty() {
                r.add(
                    "V11",
                    Verdict::Warn,
                    "no live non-vehicle record in this file's telemetry. That is the shape of \
                     the two ghosts known to CRASH the game client on import, and grafting one \
                     back is what repaired both — but TAS_67319 is in the same shape and imports \
                     cleanly, so this is NOT a rule and not a failure. If an import kills the \
                     game, try it first: `ghost record graft-scene IN OUT --from CARRIER`. \
                     Nothing headless can see this defect."
                        .to_string(),
                );
            } else {
                r.add(
                    "V11",
                    Verdict::Pass,
                    format!(
                        "the container's live non-vehicle record(s) are present ({})",
                        live.join(", ")
                    ),
                );
            }
        }
    }

    // ---- V12 the declared SPLIT TABLE --------------------------------------
    //
    // The intermediate checkpoint times, checked for the things a single file
    // can actually establish about them: the last entry IS the race time (true
    // of every reference ghost in this corpus), the list does not go backwards,
    // and no intermediate sits at or past the finish.
    //
    // ⚠ READ THIS BEFORE TRUSTING A PASS. **These checks cannot detect the one
    // defect that has actually shipped here — a split table INHERITED from the
    // donor.** 287431's whole lineage is seeded from ITZYNO1FAN's 24.092 ghost;
    // every tape it produced declared his six splits beside its own finish, and
    // that table is monotonic, ends at the declared time, and sits entirely
    // inside the run. It passes all three checks below and is somebody else's
    // measurement. A published clip said "CP6 at 16.945" on that basis.
    //
    // Nor can the trajectory settle it. The obvious gate -- "was the car near
    // checkpoint k at split k" -- was built and MEASURED on this map before
    // being rejected: our line tracks his to about 2 m through CP4, so at his
    // split times our car is 1.2 / 2.9 / 2.4 / 2.6 m from his position, well
    // inside the 20-46 m that a checkpoint block's ORIGIN sits from its own
    // gate. Every one of the six passes. A test any outcome satisfies is
    // decoration, so it is not here.
    //
    // WHERE THE DEFECT IS CAUGHT INSTEAD: `ghost declare`, which is the only
    // place that sees the run BEFORE and AFTER. When the declared time changes,
    // the intermediates timed the old run and are zeroed. Hence the zero rule
    // below -- a blank table is the honest state, not a broken one.
    {
        let sp: Vec<i64> = c.splits().iter().map(|v| *v as i64).collect();
        let decl = vals.first().copied().unwrap_or(0) as i64;
        if sp.is_empty() {
            r.add("V12", Verdict::Na, "this container declares no split table");
        } else {
            let inter = &sp[..sp.len() - 1];
            let last = *sp.last().unwrap();
            let blank = inter.iter().all(|v| *v == 0);
            let mut bad: Vec<String> = Vec::new();
            if decl > 0 && last != decl {
                bad.push(format!(
                    "the last split is {} and the file declares {}",
                    secs(last),
                    secs(decl)
                ));
            }
            if !blank {
                for w in inter.windows(2) {
                    if w[1] < w[0] {
                        bad.push(format!("the splits go backwards: {} then {}", secs(w[0]), secs(w[1])));
                        break;
                    }
                }
                if let Some(v) = inter.iter().find(|v| **v >= last && last > 0) {
                    bad.push(format!(
                        "an intermediate split ({}) is at or past the finish ({})",
                        secs(*v),
                        secs(last)
                    ));
                }
            }
            if !bad.is_empty() {
                r.add("V12", Verdict::Fail, format!("split table: {}", bad.join("; ")));
            } else if blank {
                r.add(
                    "V12",
                    Verdict::Pass,
                    format!(
                        "the {} intermediate split(s) are blank (0.000) and the last is the \
                         race time {}. Blank is HONEST, not missing: it says this container \
                         does not know its own checkpoint crossings. Measure them and write \
                         them with `ghost declare --splits`.",
                        inter.len(),
                        secs(last)
                    ),
                );
            } else {
                r.add(
                    "V12",
                    Verdict::Warn,
                    format!(
                        "split table {} is self-consistent -- BUT NOTHING HERE SHOWS IT IS THIS \
                         RUN'S. An inherited table passes every check in V12 (see the comment). \
                         If this file came from a search or a regen, confirm the splits were \
                         MEASURED for it; if they were not, `ghost declare` will blank them.",
                        inter.iter().map(|v| secs(*v)).collect::<Vec<_>>().join(" ")
                    ),
                );
            }
        }
    }

    // ---- V5 telemetry span -------------------------------------------------
    match gbx::record::decode_ghost(path) {
        Err(e) => r.add("V5", Verdict::Na, format!("no telemetry record ({})", e)),
        Ok(d) => {
            let last = d.samples.last().map(|s| s.time_ms).unwrap_or(0) as i64;
            let decl = vals.first().copied().unwrap_or(0) as i64;
            if decl > 0 && last > decl + 2000 {
                r.add(
                    "V5",
                    Verdict::Fail,
                    format!(
                        "telemetry: the last sample is at {} but the file declares {} -- \
                         the record outlives the run it claims to be",
                        secs(last),
                        secs(decl)
                    ),
                );
            } else if d.samples.is_empty() {
                r.add("V5", Verdict::Fail, "telemetry: the record has no vehicle samples at all");
            } else {
                r.add(
                    "V5",
                    Verdict::Pass,
                    format!(
                        "telemetry: {} samples, {} .. {}, span {} .. {}",
                        d.samples.len(),
                        secs(d.samples[0].time_ms as i64),
                        secs(last),
                        secs(d.start_ms as i64),
                        secs(d.end_ms as i64)
                    ),
                );
            }
        }
    }

    // ---- V6 does the recording agree with the tape? -----------------------
    match tape_record_agreement(path) {
        None => r.add("V6", Verdict::Na, "tape/record agreement: this file has no telemetry to compare"),
        Some((kappa, rate, lag, n)) => {
            let thr: f64 = flag(a, "--agree-thr").and_then(|v| v.parse().ok()).unwrap_or(0.60);
            // THE BAND WHERE THIS GATE CANNOT DISCRIMINATE, and why it is a
            // WARN rather than a PASS.
            //
            // Measured, by swapping one corpus ghost's telemetry for another's
            // on the SAME map -- a forgery this gate exists to catch:
            //
            //     correct telemetry   kappa 0.938, 1.000, 1.000, 1.000
            //     FOREIGN telemetry   kappa 0.584, 0.692, 0.703, 0.806
            //
            // Three of those four forgeries clear a 0.60 threshold. The gate
            // caught one in four and said PASS to the rest.
            //
            // Raising the threshold is NOT the fix, and the note at V9 says
            // why: this project's own `SEARCHTAPE_..._DO_NOT_PUBLISH` files
            // score 0.83, the same as a genuine human recording. The
            // distributions OVERLAP, so no threshold separates them -- the
            // metric saturates, and only V9 (the engine re-simulating the
            // tape) settles it.
            //
            // What CAN be fixed is the report. A bare "PASS V6 kappa 0.644"
            // reads as "the telemetry is verified", and it is not: it means
            // "not obviously foreign". Above HIGH the agreement is as good as
            // a clean recording; between `thr` and HIGH it is a band that
            // contains both honest files and known-bad ones, and the reader
            // is told to run the engine.
            const HIGH: f64 = 0.90;
            if kappa >= HIGH {
                r.add(
                    "V6",
                    Verdict::Pass,
                    format!(
                        "tape/record agreement: kappa {:.3} ({:.1}% of {} samples exact, best lag {} ms)",
                        kappa, 100.0 * rate, n, lag
                    ),
                );
            } else if kappa >= thr {
                r.add(
                    "V6",
                    Verdict::Warn,
                    format!(
                        "tape/record agreement: kappa {:.3} ({:.1}% of {} samples exact, best lag {} ms) \
                         -- ABOVE the {:.2} floor but BELOW {:.2}, which is the band this gate cannot \
                         see into. Foreign telemetry swapped between two runs of one map scores \
                         0.58-0.81 here, and this project's own poisoned search tapes score 0.83, the \
                         same as a clean human recording. Passing this is not evidence the telemetry \
                         belongs to the tape. Run `ghost verify --engine --map M` (V9), which \
                         re-simulates the tape and compares trajectories.",
                        kappa, 100.0 * rate, n, lag, thr, HIGH
                    ),
                );
            } else {
                r.add(
                    "V6",
                    Verdict::Fail,
                    format!(
                        "tape/record agreement: kappa {:.3} ({:.1}% of {} samples exact, best of 21 lags at {} ms). \
                         Chance-corrected agreement this low means the recording in this file is NOT this \
                         tape's run -- it is carrying another run. Rebuild it with `ghost regen`, or name \
                         the file so that nobody renders it.",
                        kappa, 100.0 * rate, n, lag
                    ),
                );
            }
        }
    }

    // ---- V7 the plain oracle re-simulating THE WRITTEN FILE ---------------
    let sd = flag(a, "--server").map(|s| s.to_string()).or_else(|| std::env::var("TM_SERVER").ok());
    let server = oracle::server_dir(sd.as_deref());
    if !server.join("TrackmaniaServer").exists() {
        r.add("V7", Verdict::Na, format!("no dedicated server at {} (set TM_SERVER or --server)", server.display()));
    } else if has(a, "--no-oracle") {
        r.add("V7", Verdict::Na, "--no-oracle");
    } else if flag(a, "--map").is_none() && c.embedded_map().is_none() {
        // NO MAP ANYWHERE IS AN INSTRUMENT STATE, NOT A VERDICT ABOUT THE RUN.
        //
        // Without `--map` the Maps directory is emptied, and a file that does
        // not carry its own map then cannot finish whatever is on its tape --
        // the server has nothing to drive on. Running it anyway produced
        // "FAIL V7 oracle: DNF (cps None) on this file ... not publishable",
        // which reads as a judgement on the run and is a judgement on the
        // command line. A correct 239.133 was refused this way once.
        //
        // `--empty-maps` (V8) is the deliberate zero-map control, and it is for
        // files that DO carry a map; it is unaffected.
        r.add(
            "V7",
            Verdict::Na,
            "no map: this file carries none and no --map was given, so the oracle would have \
             nothing to drive on and would report DNF whatever is on the tape. Pass --map."
                .to_string(),
        );
    } else {
        let mode = match flag(a, "--map") {
            Some(m) => MapsMode::One(std::path::Path::new(m)),
            None => MapsMode::Empty,
        };
        match oracle::validate(&server, std::path::Path::new(path), mode, "verify") {
            Err(e) => r.add("V7", Verdict::Na, format!("oracle: {}", e)),
            Ok(res) => {
                let expect_dnf = flag(a, "--expect") == Some("dnf");
                let want = num(a, "--expect-ms").or(vals.first().map(|v| *v as i64));
                match (res.time_ms, expect_dnf) {
                    (None, true) => r.add(
                        "V7",
                        Verdict::Pass,
                        format!("oracle on the written file: DNF at cps {:?}, as expected for a partial run", res.cps),
                    ),
                    (Some(t), true) => r.add(
                        "V7",
                        Verdict::Fail,
                        format!("oracle returned {} but --expect dnf was asked for", secs(t)),
                    ),
                    _ => match (res.time_ms, want) {
                        (Some(t), Some(w)) if t == w => r.add(
                            "V7",
                            Verdict::Pass,
                            format!("oracle re-simulated the written file: {} == the declared time", secs(t)),
                        ),
                        (Some(t), Some(w)) => r.add(
                            "V7",
                            Verdict::Fail,
                            format!(
                                "oracle re-simulated the written file to {} but it declares {}",
                                secs(t),
                                secs(w)
                            ),
                        ),
                        (Some(t), None) => {
                            r.add("V7", Verdict::Pass, format!("oracle re-simulated the written file: {}", secs(t)))
                        }
                        (None, _) => r.add(
                            "V7",
                            Verdict::Fail,
                            format!(
                                "oracle: DNF (cps {:?}) on this file. If this is a deliberately \
                                 partial run -- a trimmed clip, a search tape -- say so with \
                                 `--expect dnf`; a file that declares a finish and does not \
                                 finish is not publishable.",
                                res.cps
                            ),
                        ),
                    },
                }
            }
        }
    }

    // ---- V8 the empty-Maps control ----------------------------------------
    if has(a, "--empty-maps") {
        let server = oracle::server_dir(sd.as_deref());
        match oracle::validate(&server, std::path::Path::new(path), MapsMode::Empty, "empty") {
            Err(e) => r.add("V8", Verdict::Na, format!("empty-Maps control: {}", e)),
            Ok(res) => {
                let carried = c.embedded_map().is_some();
                match (res.time_ms, carried) {
                    (Some(t), true) => r.add(
                        "V8",
                        Verdict::Pass,
                        format!(
                            "empty-Maps control: {} with ZERO maps on disk, uid {:?} -- confirmed, the map \
                             this file runs on comes out of the file",
                            secs(t),
                            res.map_uid
                        ),
                    ),
                    (None, false) => r.add(
                        "V8",
                        Verdict::Pass,
                        "empty-Maps control: no result with zero maps on disk -- confirmed, this file needs a map from disk",
                    ),
                    (Some(t), false) => r.add(
                        "V8",
                        Verdict::Fail,
                        format!("empty-Maps control: returned {} with no map on disk, but no embedded map was found", secs(t)),
                    ),
                    (None, true) => r.add(
                        "V8",
                        Verdict::Warn,
                        "empty-Maps control: this file carries a map but did not validate against it",
                    ),
                }
            }
        }
    }
    // ---- V9 the ENGINE re-simulating this tape ---------------------------
    //
    // V6 has a floor it cannot see past, and the floor is measured: a search
    // tape that differs from its template by a few per cent of ticks still
    // agrees with the template's recording on the other 97 %, and three of
    // this project's own files labelled `SEARCHTAPE_..._DO_NOT_PUBLISH` score
    // kappa 0.83 -- the same as a human recording. The only thing that settles
    // it is running the tape through the real engine and comparing the
    // trajectory the engine produces with the trajectory the file claims.
    if has(a, "--engine") {
        match flag(a, "--map") {
            None => r.add("V9", Verdict::Na, "--engine needs --map"),
            Some(m) => match crate::regen::engine_trajectory_agreement(path, m) {
                Err(e) => r.add("V9", Verdict::Na, format!("engine trajectory: {}", e)),
                Ok((mean, worst, n, shift)) => {
                    let thr: f64 = flag(a, "--traj-thr").and_then(|v| v.parse().ok()).unwrap_or(0.05);
                    if shift {
                        r.add(
                            "V9",
                            Verdict::Fail,
                            format!(
                                "engine trajectory: the recording is a WHOLE SAMPLE out of phase with the \
                                 engine's own run of this tape (mean {:.4} m over {} samples). A solo clip \
                                 cannot look wrong from this; every frame-synchronous comparison is.",
                                mean, n
                            ),
                        );
                    } else if mean <= thr {
                        r.add(
                            "V9",
                            Verdict::Pass,
                            format!(
                                "engine trajectory: the engine's own run of this tape matches the recording \
                                 to {:.4} m mean / {:.4} m worst over {} samples",
                                mean, worst, n
                            ),
                        );
                    } else {
                        r.add(
                            "V9",
                            Verdict::Fail,
                            format!(
                                "engine trajectory: running THIS tape through the real engine puts the car \
                                 {:.3} m from where this file says it was (worst {:.3} m over {} samples). \
                                 The recording is not this tape's run.",
                                mean, worst, n
                            ),
                        );
                    }
                }
            },
        }
    }
    // ---- V10 the raw-bytes backstop --------------------------------------
    //
    // Every check above is STRUCTURED: it reads a field this crate knows how
    // to find. The whole class of defect this gate exists for is a field
    // nobody thought of -- so the last check does not think. It scans the
    // file's bytes for anything that LOOKS like somebody's identity or like a
    // race time that is not this run's, and fails on a hit.
    //
    // It is dumb on purpose. The structured checks tell you WHICH field; this
    // one tells you THAT THERE IS ONE, which is the part we kept getting
    // wrong: the header defect that motivated it sat in front of a green V2
    // and a green V3 for a day.
    //
    // The embedded map's own byte range is excluded by POSITION. Everything in
    // there is the map's -- its author block, its name, its own uid -- and it
    // stays. That exclusion is the only thing in this check that knows
    // anything, and it is a range, not a value.
    {
        let raw = std::fs::read(path).unwrap_or_default();
        let hdr_len = raw.len().saturating_sub(c.body().len());
        // The map sits at a body offset; convert to a file offset.
        let map_range = c.embedded_map().map(|(o, n)| (hdr_len + o, hdr_len + o + n));
        // ...and the map's own attribution in the replay header, by position.
        let mut allow: Vec<(usize, usize)> = crate::hdr::legitimate_map_ranges(&c, hdr_len);
        if let Some(m) = map_range {
            allow.push(m);
        }
        let inside_any = |at: usize| allow.iter().any(|(a, b)| at >= *a && at < *b);
        let ours: Vec<String> = {
            let mut v: Vec<String> = vec!["TAS".into()];
            v.extend(c.uids().into_iter().map(|(_, u)| u));
            v
        };
        let own_ms: Vec<String> = {
            let mut s: Vec<String> =
                c.declared_times().iter().map(|(_, v)| v.to_string()).collect();
            s.extend(crate::hdr::header_declared_ms(&c).iter().map(|(_, v)| v.to_string()));
            s.sort();
            s.dedup();
            s
        };
        let mut hits: Vec<String> = Vec::new();
        // (a) storage-object locator URLs and skin paths carrying a uuid
        for pat in ["storageObjects/", "core.trackmania.nadeo.live"] {
            for at in find_all(&raw, pat.as_bytes()) {
                if inside_any(at) {
                    continue;
                }
                hits.push(format!("a locator URL at {} ({:?})", at, snippet(&raw, at, 72)));
            }
        }
        // (b) 22-character base64url tokens: the shape of an account id
        for (at, tok) in b64_tokens(&raw) {
            if inside_any(at) || ours.iter().any(|o| *o == tok) {
                continue;
            }
            if tok.chars().all(|ch| ch == 'x') {
                continue;
            }
            hits.push(format!("an account-id-shaped token at {}: {:?}", at, tok));
        }
        // (c) a `best="..."` in the header XML that is not this run's time
        for at in find_all(&raw, b"best=\"") {
            let v: String = raw[at + 6..]
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .map(|b| *b as char)
                .collect();
            if !v.is_empty() && !own_ms.contains(&v) {
                hits.push(format!("best=\"{}\" at {}, which is not this file's declared time", v, at));
            }
        }
        if hits.is_empty() {
            r.add(
                "V10",
                Verdict::Pass,
                format!(
                    "raw-bytes backstop: nothing that looks like another person's identity or \
                     another run's time, outside the embedded map's own {} B",
                    map_range.map_or(0, |(a2, b)| b - a2)
                ),
            );
        } else {
            r.add(
                "V10",
                Verdict::Fail,
                format!("raw-bytes backstop found {}: {}", hits.len(), hits.join("; ")),
            );
        }
    }
    r
}

fn find_all(hay: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    if needle.is_empty() || hay.len() < needle.len() {
        return out;
    }
    for i in 0..=hay.len() - needle.len() {
        if &hay[i..i + needle.len()] == needle {
            out.push(i);
        }
    }
    out
}

fn snippet(b: &[u8], at: usize, n: usize) -> String {
    b[at..(at + n).min(b.len())].iter().map(|c| if c.is_ascii_graphic() { *c as char } else { '.' }).collect()
}

/// Maximal runs of base64url characters, reported when they are exactly the
/// 22 characters a 16-byte account id encodes to.
fn b64_tokens(b: &[u8]) -> Vec<(usize, String)> {
    let ok = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'-';
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        if !ok(b[i]) {
            i += 1;
            continue;
        }
        let s = i;
        while i < b.len() && ok(b[i]) {
            i += 1;
        }
        if i - s == 22 {
            if let Ok(t) = std::str::from_utf8(&b[s..i]) {
                // an id has both cases or a digit; a 22-letter English word does not
                let has_digit = t.chars().any(|c| c.is_ascii_digit());
                let mixed = t.chars().any(|c| c.is_ascii_uppercase())
                    && t.chars().any(|c| c.is_ascii_lowercase());
                if has_digit && mixed {
                    out.push((s, t.to_string()));
                }
            }
        }
    }
    out
}

pub fn cmd(a: &[String]) {
    let path = a.first().unwrap_or_else(|| die("ghost verify FILE [--map M] [--server DIR] [-o json]"));
    // `-o human` (default) or `-o json`. The verdict and the exit code are
    // identical either way -- this changes the rendering, not the judgement.
    let json_out = matches!(flag(a, "-o"), Some("json"))
        || matches!(flag(a, "--output"), Some("json"));
    if !json_out {
        println!("verifying {}", path);
    }
    let r = run(path, a);
    if json_out {
        println!("{}", r.json());
    } else {
        r.print();
    }
    let n_fail = r.checks.iter().filter(|c| c.verdict == Verdict::Fail).count();
    let n_pass = r.checks.iter().filter(|c| c.verdict == Verdict::Pass).count();
    if n_fail > 0 {
        if !json_out {
            println!("\nREFUSED: {} check(s) failed", n_fail);
        }
        // EXIT 1: the verification ran and the answer is NO. Exit 2 is
        // reserved for "you called me wrong" -- see cli::refuse. This is the
        // single most important place for that distinction, because `verify`
        // is what a publish script branches on.
        std::process::exit(1);
    }
    // NOTHING CHECKED IS NOT A PASS.
    //
    // Every gate here reports NA when it has nothing to work with -- no
    // telemetry record, no tape, no server. A file that trips ALL of them
    // therefore reached the end with zero failures and printed OK: 67 bytes of
    // `GBX` + 0xFF verified clean. That is a false pass, and a false pass from
    // a verifier is worse than the panic it replaced, because a panic at least
    // stops the pipeline.
    //
    // Require at least one gate to have actually looked at something.
    // SOME GATES ARE ABSENCE CHECKS AND PASS VACUOUSLY.
    //
    // V3 ("no account id in the body"), V4 ("no embedded map") and V10 (the
    // raw-bytes backstop) are all of the form "this file does not contain
    // something bad". A file with no body contains nothing bad, so all three
    // pass -- and 67 bytes of `GBX` + 0xFF verified OK with every gate that
    // looks at an actual RUN reporting NA.
    //
    // A pass has to mean a gate looked at a run and liked it. These are the
    // gates that do: the tape, the declared time, the splits, the telemetry,
    // tape/record agreement, and the oracle.
    const SUBSTANTIVE: &[&str] = &["V1", "V2", "V5", "V6", "V7", "V12"];
    let n_substantive = r
        .checks
        .iter()
        .filter(|c| c.verdict == Verdict::Pass && SUBSTANTIVE.contains(&c.id))
        .count();
    if n_substantive == 0 {
        if !json_out {
            println!(
                "\nREFUSED: nothing could be checked -- {} gate(s) ran and none of them \
                 found a run to verify -- only absence checks passed, and those pass on \
                 any file. This is not a ghost.",
                r.checks.len()
            );
        }
        std::process::exit(1);
    }
    if !json_out {
        println!("\nOK");
    }
}
