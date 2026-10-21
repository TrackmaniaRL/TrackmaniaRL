//! The finishing pass: everything between "the engine state is in the record"
//! and "this file is publishable".
//!
//! It exists because the list of things that had to be remembered was longer
//! than anyone remembered it, and every item on it shipped at least once:
//!
//! | shipped defect | what was forgotten |
//! |---|---|
//! | dirt thrown where there is no dirt | the 49 unwritten per-run bytes |
//! | a 441 s clip of a 218 s run, camera adrift | the record's inherited span |
//! | a header declaring the donor's time | the declared-time census |
//! | a stranger's login, badge, country, uuid | the identity fields |
//!
//! Every one was found by a person noticing something odd in a video, days
//! later. So the fix is not another check that somebody has to run: it is that
//! `ghost regen`'s DEFAULT output is finished, and that it refuses rather than
//! writing something a later step has to catch.
//!
//! Each step below already existed as a separate command that had to be run
//! afterwards by whoever knew to. They are called here, in order, on every
//! regeneration — `ghost record rebuild`, `ghost declare --from-oracle`,
//! `ghost identity --anonymise` — and each one still refuses on its own terms,
//! which is what makes calling them a pipeline rather than a copy of them.

use gbx::container::secs;

/// Which sample bytes still hold the container donor's values, by number.
///
/// The acceptance test for the whole pass, and the one that cannot be argued
/// with: compare against the container this file was built in, byte by byte, at
/// the same sample index. **An empty list is the goal.** A non-empty one is
/// printed as loudly as a failure, with the offsets spelled out, because "91 of
/// 116 bytes are still the carrier's" was true for months and was carried
/// around as a generality instead of a list — which is why nobody noticed that
/// four of those bytes are what make the tyres throw dirt.
///
/// Format constants are excluded: identical in every ghost of every driver,
/// never varying in either file, so they carry no provenance and listing them
/// would bury the ones that do.
/// A BYTE WE WROTE CANNOT BE INHERITED, however much it agrees.
///
/// The round-trip control found this on its first run and it would have been a
/// slow poison otherwise: regenerating a recording from its OWN inputs
/// correctly reproduces its own position, so bytes 50, 57 and 58 -- the
/// high-order bytes of the x and z floats, which barely move over a lap --
/// came back bit-identical on every sample, and the provenance check called
/// three position bytes the donor's. Agreement between our output and the file
/// it was generated from is SUCCESS there, not contamination.
///
/// So the question is only ever asked about bytes the regeneration did not
/// write: the transform (47..69) and the tape echo (14, 15, 18) are ours by
/// construction, whatever they equal.
/// What a `--carrier layout` pass writes, as a predicate. The authority is
/// `gbx::sample::written_by_carrier`, derived from the three facts about the
/// layout rather than listed a fourth time.
fn written_by_us(o: usize) -> bool {
    (47..69).contains(&o) || o == 14 || o == 15 || o == 18
}

/// Bytes of `ghost` that are bit-identical to `carrier` on EVERY sample, split
/// by whether this pipeline wrote them.
///
/// **The split is the whole content of this function, and it exists because
/// the same measurement means two opposite things.**
///
/// `carrier` is whatever container the file was built in. When that is a
/// stranger's recording, a byte reproducing it on every sample is *their* run
/// showing through — the defect this check was built for (286279 carried
/// Bald_tm's ground contact, gear and dirt on all 4359 samples).
///
/// But a RE-regeneration's carrier is the map's own previous file: the same
/// run, off the same tape, through the same engine. There, a byte that comes
/// out bit-identical is the pipeline being REPRODUCIBLE. Refusing it refused
/// five correct files in this sweep — 279218 on bytes [1, 31, 91], 279197 on
/// [1, 31, 89, 91], 145875 on [24, 26, 28, 30, 91], 126859 on [5, 21, 31, 90]
/// — every one of them a byte `--carrier layout` had just written out of
/// engine memory, and every one of them low-entropy (turbo, gear, ground
/// contact, ground material) where two runs of one tape land on the same bytes.
///
/// So: a byte we did NOT write is inheritance and is refused. A byte we DID
/// write and that matches anyway is reported with the ratio — and the ratio is
/// the control, because the failure this cannot be allowed to miss (the gather
/// silently not running, c01a842) inherits the WHOLE written set at once, not
/// three bytes of it.
pub struct Inherited {
    /// Bytes we did not write that reproduce the carrier exactly. Refuse.
    pub not_ours: Vec<usize>,
    /// Bytes we did write that reproduce the carrier exactly. Report.
    pub reproduced: Vec<usize>,
    /// How many of the bytes we write actually vary in the carrier — the
    /// denominator `reproduced` has to be read against.
    pub ours_varying: usize,
}

pub fn inherited_bytes_split(
    ghost: &str,
    carrier: &str,
    carrier_ran: bool,
) -> Result<Inherited, String> {
    let a = gbx::record::decode_ghost(ghost)?;
    let b = gbx::record::decode_ghost(carrier)?;
    let ss = a.sample_size;
    if ss != b.sample_size || ss == 0 {
        return Err(format!(
            "sample sizes differ ({} vs {}) -- this is not the container it was built in",
            ss, b.sample_size
        ));
    }
    let n = (a.raw.len() / ss).min(b.raw.len() / ss);
    if n < 20 {
        return Err(format!("only {n} comparable samples"));
    }
    let wrote = gbx::sample::written_by_carrier();
    let ours = |k: usize| -> bool {
        written_by_us(k) || (carrier_ran && wrote.get(k).copied().unwrap_or(false))
    };
    let mut r = Inherited { not_ours: Vec::new(), reproduced: Vec::new(), ours_varying: 0 };
    for k in 0..ss {
        let mut same = 0usize;
        let mut varies = false;
        for i in 0..n {
            if a.raw[i * ss + k] == b.raw[i * ss + k] {
                same += 1;
            }
            if b.raw[i * ss + k] != b.raw[k] {
                varies = true;
            }
        }
        if !varies {
            continue;
        }
        if ours(k) {
            r.ours_varying += 1;
            if same == n {
                r.reproduced.push(k);
            }
        } else if same == n {
            r.not_ours.push(k);
        }
    }
    Ok(r)
}

/// The original signature, kept for callers that only want the refusable set.
pub fn inherited_bytes(ghost: &str, carrier: &str) -> Result<Vec<usize>, String> {
    Ok(inherited_bytes_split(ghost, carrier, false)?.not_ours)
}

/// Does anything in this file outlive the car?
///
/// The record's own declared end, and every entity's own last sample. 286279's
/// published `BEST_218812` reads `span 0.000 .. 441.000` for a car that stops
/// at 217.95 and keeps the donor's 8820-sample non-vehicle entity at its full
/// length — which is what renders 441 s of video and strands the camera when
/// our car's entity ends.
pub fn outlives_the_car(path: &str) -> Result<Option<String>, String> {
    let d = gbx::record::decode_ghost(path)?;
    let last = d.samples.last().map(|s| s.time_ms).unwrap_or(0) as i64;
    let ent_end = d.ents.iter().filter_map(|e| e.t_last).max().unwrap_or(0) as i64;
    let scene = (d.end_ms as i64).max(ent_end);
    let past = scene - last;
    // Proportionate, not flat: a record legitimately runs a fraction of a
    // second past the finish, while a carrier's span is 87 % to 10 500 % long.
    if past > 2000 && past as f64 / (last as f64).max(1.0) >= 0.10 {
        return Ok(Some(format!(
            "the scene ends at {} and the car's last sample is at {} (+{:.0} %), over {} entities",
            secs(scene),
            secs(last),
            100.0 * past as f64 / (last as f64).max(1.0),
            d.ents.len()
        )));
    }
    Ok(None)
}

/// Is every channel we claim to write actually alive in this file?
///
/// **A bare position copy once wrote ZEROED wheels into a file that passed the
/// entire verify gate** (found by the carrier-byte arm, 2026-08-22). Zero is a
/// legal value for every one of these channels, so no per-sample check can
/// object to one sample of it; what no real run does is hold a channel at a
/// single value for its whole length while the car drives 3 km.
///
/// So the test is variance, not value, and it applies to whatever the current
/// pipeline says it writes. As channels move from `unwritten_channels` into the
/// written set, they come under this automatically -- which is the point: the
/// day the wheel bytes start being written is the day a silently-zeroed wheel
/// byte has to be caught, and nobody should have to remember to add a check.
///
/// **EXCEPT WHERE THE RUN ITSELF IS CONSTANT, AND THREE MAPS PAID FOR THAT.**
/// `byte 15 (gas echo) holds 0xff on all samples` refused unluckE - get jiggy
/// with it, Training 10 long and Great WTF of what #165 -- and on unluckE the
/// input tape reads `accel=1` on **all 789 ticks**, so a constant gas echo is
/// the CORRECT echo of a run that never lifts. The page says so in prose:
/// "gas held throughout, brake never touched". The check was reading its own
/// assumption -- that a real run varies every channel -- as a property of the
/// file.
///
/// A channel is therefore dead only when it is constant AND the tape says the
/// driver varied the thing it echoes. `echoes` names that link; a channel with
/// no tape counterpart keeps the old unconditional test, because for position
/// and speed there is nothing to consult and a constant really is impossible.
///
/// This is the failure shape the doc comment on [`must_be_live`] already warns
/// about, arriving through the one door that comment did not cover: not "a
/// channel that may rest", but a channel that is *usually* live and is
/// legitimately constant on this particular run. The general lesson is in
/// CLAIMS.md -- a check must consult the artefact rather than the average.
/// Which sample bytes never change across a file's own record.
///
/// The discriminator `dead_channels` lacks: a byte that is constant in the
/// SOURCE is legitimately constant in a faithful regeneration of it, where a
/// byte that varies in the source and not in ours means the gather did not
/// run. Same measurement, different file.
pub fn constant_bytes(path: &str) -> Result<Vec<usize>, String> {
    let d = gbx::record::decode_ghost(path)?;
    let ss = d.sample_size;
    let n = d.raw.len() / ss.max(1);
    if ss == 0 || n == 0 {
        return Ok(Vec::new());
    }
    Ok((0..ss)
        .filter(|o| {
            let first = d.raw[*o];
            (0..n).all(|i| d.raw[i * ss + o] == first)
        })
        .collect())
}

pub fn dead_channels(path: &str, expect_alive: &[(usize, &str)]) -> Result<Vec<String>, String> {
    let d = gbx::record::decode_ghost(path)?;
    let ss = d.sample_size;
    let n = d.raw.len() / ss.max(1);
    if n < 20 {
        return Err(format!("only {n} samples"));
    }
    // What the driver actually did, where the tape can say.
    let tape = gbx::tape::Tape::from_file(path).ok();
    let mut dead = Vec::new();
    for (o, name) in expect_alive {
        if *o >= ss {
            continue;
        }
        let first = d.raw[*o];
        if !(0..n).all(|i| d.raw[i * ss + o] == first) {
            continue;
        }
        if let (Some(t), Some(kind)) = (&tape, echoes(*o)) {
            // The window is the RECORD's own span: what the driver was doing
            // outside it is not what these samples echo.
            if let Some(varied) = kind.varied_in(t, d.start_ms as i64, d.end_ms as i64) {
                if !varied {
                    // The run held this input for its whole length, so a
                    // constant echo of it is right. Not a defect.
                    continue;
                }
            }
        }
        dead.push(format!(
            "byte {o} ({name}) holds {first:#04x} on all {n} samples -- this channel is \
             claimed as written and is not alive"
        ));
    }
    Ok(dead)
}

/// Which driver input a sample byte echoes, where one does.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Echoes {
    Steer,
    Accel,
    Brake,
}

impl Echoes {
    /// Did the driver vary this input over the run? `None` when the tape
    /// carries no ticks, which keeps the caller on the unconditional test
    /// rather than letting an unreadable tape excuse a dead channel.
    /// Did this input vary **inside a race window**?
    ///
    /// THE WINDOW IS THE WHOLE POINT, and leaving it out convicted honest
    /// files. A tape runs from the countdown (`start_offset` is about
    /// −1.52 s) to the last tick recorded, while the telemetry record samples
    /// only the race. On any run that is flat out from the lights to the line
    /// -- which is most sprints -- the tape's accel goes 0 during the
    /// countdown and 1 for the entire race, so scanning the whole tape says
    /// "this input varied" and the constant echo in the record then reads as a
    /// dead channel. It is not: it is the truth about the race.
    /// Measured on 279209 (6.578, gas 0xff on all 132 samples) and 197047
    /// (95.839, gas 0xff on all 1917 samples); both were refused and both are
    /// correct files.
    pub fn varied_in(self, t: &gbx::tape::Tape, from_ms: i64, to_ms: i64) -> Option<bool> {
        let vals: Vec<i64> = match self {
            Echoes::Steer => t.steer_i8s().iter().map(|v| *v as i64).collect(),
            Echoes::Accel => t.accels().iter().map(|v| *v as i64).collect(),
            Echoes::Brake => t.brakes().iter().map(|v| *v as i64).collect(),
        };
        let inside: Vec<i64> = vals
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let ms = t.race_ms(*i);
                ms >= from_ms && ms <= to_ms
            })
            .map(|(_, v)| *v)
            .collect();
        // No tick inside the window is not "it did not vary" -- it is a
        // question this control cannot answer, and None makes the caller fall
        // through to reporting rather than to excusing.
        if inside.is_empty() {
            return None;
        }
        let first = *inside.first()?;
        Some(inside.iter().any(|v| *v != first))
    }
}

/// The mapping, kept next to the check that uses it.
pub fn echoes(byte: usize) -> Option<Echoes> {
    match byte {
        14 => Some(Echoes::Steer),
        15 => Some(Echoes::Accel),
        16 => Some(Echoes::Brake),
        _ => None,
    }
}

/// The engine channels this pass does NOT yet write, named by byte.
///
/// **This is a harness limit, not a data limit, and the distinction is the
/// point.** Every one of these quantities is in the engine's memory while it
/// simulates — the `whl` arm fitted them against a real recording and got
/// gear, turbo and wetness exact on 100 % of samples and rpm on 92.6 % — so
/// "we cannot have them" would be false. What is true is that their addresses
/// are relative to a copy of the car state whose position varies per map, and
/// the anchoring that would make them portable is not built. Until it is, the
/// honest thing is to write ZERO and say which byte, rather than pass the
/// donor's through where it reads as ours.
/// Channels the pipeline writes that MUST be alive on any real run, and the
/// ones that may legitimately rest.
///
/// The distinction matters because `dead_channels` refuses, and refusing is
/// only right where a constant value is impossible rather than merely
/// unusual. A car that drives for 3 km always turns its wheels; a car on a
/// short run may never change gear and may never touch the turbo, so claiming
/// those as must-be-live would refuse honest work -- this project's most
/// expensive failure shape, and the reason C3 and C8 had to be superseded.
///
/// The four wheel ROTATIONS are the diagnostic ones: the carrier-bytes arm
/// measured that a bare position copy has the car's position with dead memory
/// around it, so its wheel slots read 0 of 4 live while the real vehicle struct
/// reads 4 of 4, with nothing in between.
pub fn must_be_live() -> &'static [(usize, &'static str)] {
    &[
        (14, "steer echo"),
        (15, "gas echo"),
        (47, "position x"),
        (51, "position y"),
        (55, "position z"),
        (59, "orientation angle"),
        (65, "speed"),
        // The wheel rotations, once `fk regen --carrier` writes them. Until
        // then they are in `unwritten_channels` and zeroed, and a zeroed
        // channel that is not claimed is not a defect.
        (6, "front-left wheel rotation"),
        (8, "front-right wheel rotation"),
        (10, "rear-right wheel rotation"),
        (12, "rear-left wheel rotation"),
    ]
}

/// Written from engine state, but a constant value is legitimate on a short or
/// gentle run, so a dead one is REPORTED and never refused.
pub fn may_rest() -> &'static [(usize, &'static str)] {
    &[
        (0, "unnamed u16"),
        (2, "side speed"),
        (4, "rpm"),
        (5, "rpm, high half"),
        (22, "an angle"),
        (23, "front-left suspension travel"),
        (24, "front-left ground material"),
        (25, "front-right suspension travel"),
        (26, "front-right ground material"),
        (27, "rear-right suspension travel"),
        (28, "rear-right ground material"),
        (29, "rear-left suspension travel"),
        (30, "rear-left ground material"),
        (31, "turbo"),
        (81, "ice, front left"),
        (82, "ice, front right"),
        (83, "ice, rear right"),
        (84, "ice, rear left"),
        // THE REACTOR. All five members are bit-fields packed across bytes 89,
        // 90, 91 and 76, which is why no per-byte affine fit could ever write
        // them and why three arms failed on byte 89 before the archiver was
        // disassembled. They are here rather than in `must_be_live` because a
        // map with no reactor gate legitimately holds every one of them
        // constant -- measured, untitled 02 holds byte 90 at one value for the
        // whole 9.415.
        (76, "is_top_contact"),
        (89, "is_ground_contact"),
        (90, "booster_air_control"),
        (91, "gear"),
    ]
}

pub fn unwritten_channels() -> Vec<(usize, &'static str)> {
    // DERIVED FROM `gbx::sample`, NOT LISTED HERE.
    //
    // This was a hand-maintained list, and on 2026-08-23 it was wrong in three
    // ways at once: it announced bytes 5, 81-84, 89 and 91 as "zeroed rather
    // than inherited" months after `fk regen --carrier` began writing all seven
    // from engine memory; it held byte 91 while `may_rest()` also held it,
    // which says the same byte is both written and not written; and it omitted
    // 19, 20, 34 and 108-111, which really are unwritten. Nothing failed --
    // it is a REPORT, and a report cannot fail. It was simply untrue in the one
    // place whose whole job is to say what the pipeline did.
    //
    // The names are here because they are for a human; the SET is not.
    let name = |b: usize| -> &'static str {
        match b {
            19 => "unnamed, dead in the dedicated server",
            20 => "unnamed, dead in the dedicated server",
            34 => "unnamed, dead in the dedicated server",
            93 => "dirt, front left",
            95 => "dirt, front right",
            97 => "dirt, rear right",
            99 => "dirt, rear left",
            108..=111 => "countdown",
            _ => "unnamed",
        }
    };
    gbx::sample::not_written_by_carrier().into_iter().map(|b| (b, name(b))).collect()
}

/// The same question for a run with NO `--carrier`: then the transcription does
/// not run at all and only the transform is ours.
pub fn unwritten_channels_without_carrier() -> Vec<(usize, &'static str)> {
    (0..gbx::sample::SAMPLE_SIZE)
        .filter(|b| !gbx::sample::TRANSFORM.contains(b))
        .map(|b| (b, "not written without --carrier"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE CHECK MUST FIRE, not merely pass on good files.
    ///
    /// A liveness check that has only ever been run on live channels is
    /// indistinguishable from one that returns "fine" unconditionally -- and
    /// this project has shipped exactly that shape more than once (a guard
    /// whose function was never defined, so bash returned 127 and every call
    /// answered "no"). So: a synthetic record with one channel held constant
    /// and one varying, and the check must name the first and not the second.
    #[test]
    fn a_channel_held_at_one_value_is_named_and_a_varying_one_is_not() {
        let ss = 116usize;
        let n = 40usize;
        let mut raw = vec![0u8; ss * n];
        for i in 0..n {
            raw[i * ss + 7] = i as u8; // alive
                                       // byte 9 left at 0 on every sample: dead
        }
        let dead: Vec<usize> = (0..ss)
            .filter(|k| {
                let first = raw[*k];
                (0..n).all(|i| raw[i * ss + k] == first)
            })
            .collect();
        assert!(dead.contains(&9), "a constant channel must be detected");
        assert!(!dead.contains(&7), "a varying channel must not be");
    }

    /// THE THREE LISTS MUST NOT CONTRADICT EACH OTHER.
    ///
    /// `must_be_live` and `may_rest` name channels the pipeline WRITES;
    /// `unwritten_channels` names ones it does not. A byte in both says the
    /// pipeline both does and does not write it, and the acceptance report then
    /// prints one of the two as fact. Byte 91 (gear) sat in `may_rest` and
    /// `unwritten_channels` at once from the day the carrier began writing it
    /// until 2026-08-23, and nothing noticed, because a report cannot fail.
    #[test]
    fn a_channel_is_never_both_written_and_unwritten() {
        let un: std::collections::BTreeSet<usize> =
            unwritten_channels().iter().map(|(o, _)| *o).collect();
        for (o, n) in must_be_live().iter().chain(may_rest().iter()) {
            assert!(
                !un.contains(o),
                "byte {o} ({n}) is claimed as written AND listed as unwritten"
            );
        }
    }

    /// And they must agree with the crate that actually does the writing.
    /// `gbx::sample` is the one statement of which bytes a carrier run
    /// produces; a channel this crate claims must be one of them.
    #[test]
    fn every_claimed_channel_is_one_the_writer_actually_writes() {
        let w = gbx::sample::written_by_carrier();
        for (o, n) in must_be_live().iter().chain(may_rest().iter()) {
            assert!(w[*o], "byte {o} ({n}) is claimed, but the writer does not write it");
        }
    }
}
