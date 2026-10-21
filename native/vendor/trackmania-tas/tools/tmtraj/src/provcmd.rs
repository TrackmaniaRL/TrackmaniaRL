//! `tmtraj provenance` -- which of a sample's 116 bytes are OURS, byte by byte.
//!
//! A regenerated ghost is a stranger's recording with our engine state written
//! over part of it. `fk regen` writes 22 transform bytes and 3 input-echo
//! bytes; everything else stays exactly as the carrier had it. That is stated
//! in the regenerator's own output and it has been true all along, but nobody
//! could SEE it on a finished file -- so "the tyre effects fire at the wrong
//! moments" stayed a thing people argued about from watching a video.
//!
//! This settles it from the bytes. For each of the 116 byte positions it
//! reports what fraction of samples is bit-identical to the carrier at the same
//! sample index, and the answer separates cleanly into three populations:
//!
//! * **~0 %** -- we wrote it. The transform bytes 47..69 and the input echo
//!   14 / 15 / 18.
//! * **100 %** -- the carrier's, untouched. On a ghost whose carrier drove a
//!   different line on the same map, every per-run byte here is a picture of
//!   THEIR run: wheel rotation, suspension, rpm, and the surface channels that
//!   decide whether the tyres throw dirt.
//! * **100 % and constant** -- a format constant, identical in every ghost of
//!   every driver, carrying no provenance at all. Reporting those as
//!   "inherited" would be true and useless, so they are named separately.
//!
//! The comparison is at the same sample INDEX rather than the same timestamp,
//! because inheritance is a copy operation on the record array: byte k of
//! sample i was copied from byte k of sample i. Comparing by timestamp would
//! ask a different (and here, wrong) question.

use gbx::record::{find_entrecord_blob, load_body, parse_record_data};

/// The vehicle entity's raw samples: `(sample_size, flat bytes)`.
fn raw_samples(path: &str) -> Result<(usize, Vec<u8>), String> {
    let body = load_body(path)?;
    let (ver, blob) = find_entrecord_blob(&body)?;
    let rd = parse_record_data(&blob, ver)?;
    let e = rd
        .ents
        .iter()
        .filter(|e| e.sample_size >= 100 && !e.times.is_empty())
        .max_by_key(|e| e.times.len())
        .ok_or("no CSceneVehicleVis entity")?;
    Ok((e.sample_size, e.raw.clone()))
}

/// What each byte position is for, where this project has established it.
/// Anything not named prints as `?` -- a byte nobody has identified is exactly
/// what this report exists to surface, and giving it a confident label would
/// undo that.
fn role(o: usize) -> &'static str {
    match o {
        2 | 3 => "side speed (DERIVED)",
        5 => "rpm (DERIVED)",
        14 => "steer echo (VERIFIED, from the tape)",
        15 => "gas echo (VERIFIED, from the tape)",
        18 => "brake echo (VERIFIED, from the tape)",
        47..=58 => "POSITION x/y/z (VERIFIED, from the engine)",
        59..=60 => "orientation angle (VERIFIED, from the engine)",
        61..=64 => "orientation axis (VERIFIED, from the engine)",
        65..=66 => "log speed (VERIFIED, from the engine)",
        67..=68 => "velocity direction (VERIFIED, from the engine)",
        81..=84 => "ICE per wheel (GUESS)",
        89 => "ground contact bit 0 (DERIVED)",
        91 => "gear (DERIVED)",
        93..=99 => "DIRT per wheel (GUESS)",
        _ => "?",
    }
}

const USAGE: &str = "\
usage: tmtraj provenance GHOST --carrier CARRIER.Ghost.Gbx [--all]

Per-byte provenance of a regenerated ghost against the container it was built
in: what fraction of samples is bit-identical to the carrier at the same sample
index. ~0 % is ours, 100 % is the carrier's, 100 %-and-constant is a format
constant that carries no provenance. --all lists every byte; by default the
constants are summarised.

This is what answers \"why do the tyres throw dirt where there is no dirt\":
the surface channels are in the inherited population, so they describe the
CARRIER's run, on their line, at their moments.
";

pub fn cmd(argv: &[String]) -> i32 {
    let a = crate::cli::parse("tmtraj provenance", argv, &["all"]);
    let carrier = a.one("carrier").map(|s| s.to_string());
    let all = a.has("all");
    let a = a.finish(USAGE);
    let (Some(ghost), Some(carrier)) = (a.positional.first(), carrier) else {
        eprint!("{USAGE}");
        return 2;
    };

    let (ssa, ra) = match raw_samples(ghost) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{ghost}: {e}");
            return 2;
        }
    };
    let (ssb, rb) = match raw_samples(&carrier) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{carrier}: {e}");
            return 2;
        }
    };
    if ssa != ssb {
        eprintln!("sample sizes differ: {ssa} vs {ssb}");
        return 2;
    }
    let n = (ra.len() / ssa).min(rb.len() / ssb);
    if n < 20 {
        eprintln!("only {n} comparable samples");
        return 2;
    }
    println!("{ghost}");
    println!("  against carrier {carrier}");
    println!("  {n} samples compared at the same index, {ssa} bytes each\n");

    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    let mut constant = Vec::new();
    let mut mixed = Vec::new();
    for k in 0..ssa {
        let mut same = 0usize;
        let mut varies_a = false;
        let mut varies_b = false;
        let f = ra[k];
        let g = rb[k];
        for i in 0..n {
            let (x, y) = (ra[i * ssa + k], rb[i * ssb + k]);
            if x == y {
                same += 1;
            }
            if x != f {
                varies_a = true;
            }
            if y != g {
                varies_b = true;
            }
        }
        let pct = 100.0 * same as f64 / n as f64;
        // A byte that never varies in EITHER file is a format constant: it is
        // identical in every ghost of every driver, so "inherited" says nothing
        // about it.
        if !varies_a && !varies_b && same == n {
            constant.push((k, ra[k]));
        } else if pct >= 99.9 {
            theirs.push((k, pct, varies_b));
        } else if pct <= 1.0 {
            ours.push((k, pct));
        } else {
            mixed.push((k, pct));
        }
    }

    println!("OURS -- written by the regeneration ({} bytes)", ours.len());
    for (k, p) in &ours {
        println!("  byte {k:>3}  {p:5.1} % identical   {}", role(*k));
    }
    println!("\nTHE CARRIER'S -- inherited whole ({} bytes)", theirs.len());
    for (k, p, v) in &theirs {
        println!(
            "  byte {k:>3}  {p:5.1} % identical   {}{}",
            role(*k),
            if *v { "" } else { "   [constant in the carrier too -- inherited but inert]" }
        );
    }
    if !mixed.is_empty() {
        println!("\nPARTLY ONE AND PARTLY THE OTHER ({} bytes)", mixed.len());
        for (k, p) in &mixed {
            println!("  byte {k:>3}  {p:5.1} % identical   {}", role(*k));
        }
    }
    if all {
        println!("\nFORMAT CONSTANTS ({} bytes)", constant.len());
        for (k, v) in &constant {
            println!("  byte {k:>3}  = {v:#04x} in both, on every sample");
        }
    } else {
        println!(
            "\n{} further bytes are format constants (identical in both on every sample, so \
             they carry no provenance); --all lists them.",
            constant.len()
        );
    }

    // The headline, in the terms the question was asked in.
    let surf: Vec<usize> = (81..=84).chain(93..=99).chain([89usize, 91]).collect();
    let inherited_surf: Vec<usize> =
        surf.iter().copied().filter(|k| theirs.iter().any(|(t, _, v)| t == k && *v)).collect();
    println!();
    if inherited_surf.is_empty() {
        println!(
            "SURFACE AND CONTACT CHANNELS: none of them is a live inherited channel in this \
             file. Either they were regenerated or they were neutralised -- `tmtraj check` C5 \
             says which."
        );
    } else {
        println!(
            "SURFACE AND CONTACT CHANNELS: {} of them are the carrier's, live and varying \
             ({:?}). Every tyre effect and contact spark this file triggers is at the moment \
             THEIR run had it, not ours. This is not a rendering fault.",
            inherited_surf.len(),
            inherited_surf
        );
    }
    0
}
