//! `ghost roundtrip` -- the end-to-end control for the whole regeneration path.
//!
//! Take a ghost whose telemetry we know is its own — a recording the game
//! itself wrote — regenerate it from its own inputs plus the map, and require
//! the result to reproduce the trajectory it came in with.
//!
//! It is the strongest control available here because **the answer key is in
//! the file** and nothing about it can be tuned. Every other check on a
//! regenerated ghost asks whether the output is internally coherent, or how it
//! compares with a run somebody else drove. This one asks the only question
//! that matters — *did the pipeline reproduce a run it had already been given?*
//! — and a downloaded recording answers it for free, on any map, with no
//! reference to fetch.
//!
//! **A pass on a human recording is not a pass on a synthesised tape**, and the
//! difference is worth stating, because this control's failure mode is being
//! read as more than it is. What it proves: the locate found the car, the
//! engine reproduced the physics, the encoder wrote the bytes back correctly,
//! and the finishing pass did not corrupt them. What it cannot prove: that the
//! chooser picks the right object on a file whose own record is a DONOR's —
//! which is exactly when a regeneration is needed, and exactly when the
//! chooser's reference is contaminated. So this is a control on the machinery,
//! not a certificate for the corpus.
//!
//! What counts as reproduced: the project's own floor. A regeneration of a
//! recording made by the game client, re-simulated on a dedicated server, lands
//! at **0.48–0.52 mm** median position error on nine of ten maps — that is the
//! client-versus-server difference, not our error, and it is the same number
//! every arm that has measured it got. Bit-identity is reachable on some maps
//! (267859: 113 of 224 samples exact) and is reported when it happens, but
//! requiring it everywhere would fail on the floor rather than on a defect.

use crate::cli::{die, flag, has, num};
use gbx::container::secs;

/// Median and worst position error between two files' records, matched by
/// sample time, plus how many samples are bit-identical.
fn compare(a: &str, b: &str) -> Result<(f64, f64, usize, usize), String> {
    let x = gbx::record::decode_ghost(a)?;
    let y = gbx::record::decode_ghost(b)?;
    let by: std::collections::HashMap<i32, &gbx::record::Sample> =
        y.samples.iter().map(|s| (s.time_ms, s)).collect();
    let mut errs: Vec<f64> = Vec::new();
    let mut exact = 0usize;
    let ss = x.sample_size.min(y.sample_size);
    for (i, s) in x.samples.iter().enumerate() {
        let Some(t) = by.get(&s.time_ms) else { continue };
        let d = (((s.x - t.x) as f64).powi(2)
            + ((s.y - t.y) as f64).powi(2)
            + ((s.z - t.z) as f64).powi(2))
        .sqrt();
        if !d.is_finite() {
            return Err("the regenerated record has a non-finite position".into());
        }
        errs.push(d);
        // Bit-identity on the transform bytes only: the surrounding per-run
        // bytes are deliberately zeroed by the regeneration, so comparing the
        // whole sample would answer a different question.
        let (p, q) = (i * x.sample_size, i * y.sample_size);
        if ss >= 69
            && p + 69 <= x.raw.len()
            && q + 69 <= y.raw.len()
            && x.raw[p + 47..p + 69] == y.raw[q + 47..q + 69]
        {
            exact += 1;
        }
    }
    if errs.len() < 20 {
        return Err(format!("only {} shared sample instants", errs.len()));
    }
    let n = errs.len();
    errs.sort_by(|a, b| a.total_cmp(b));
    Ok((errs[n / 2], errs[n - 1], exact, n))
}

pub fn cmd(a: &[String]) {
    let subject = a.first().unwrap_or_else(|| {
        die("ghost roundtrip GHOST --map MAP [--bar MM] [--keep] [--out FILE]")
    });
    let map = flag(a, "--map").unwrap_or_else(|| die("--map MAP.Map.Gbx"));
    // 5 mm: an order of magnitude above the 0.48-0.52 mm client-vs-server floor
    // and two orders below the nearest decoy ever measured (the sub-sample
    // stale copy at 0.09 m). Nothing has ever landed in between.
    let bar_mm: f64 = num(a, "--bar").unwrap_or(5) as f64;
    let keep = has(a, "--keep");

    // PER PROCESS, because the subject does not name the run.
    //
    // The output used to be `<subject>.roundtrip.Ghost.Gbx`, so two round trips
    // on ONE subject -- which is exactly what comparing two locate settings
    // means -- wrote to the same path and to the same intermediates. Measured:
    // two runs at different anchor radii, both silently reading each other's
    // half-written grid, neither finishing. A tool whose scratch names come
    // from its input cannot be run twice on that input, and nothing said so.
    let out = flag(a, "--out").map(|s| s.to_string()).unwrap_or_else(|| {
        format!(
            "{}.roundtrip-{}.Ghost.Gbx",
            subject.trim_end_matches(".Ghost.Gbx"),
            std::process::id()
        )
    });
    println!("== round trip: {subject}");
    println!("   regenerating it from its own inputs, then requiring its own trajectory back\n");

    let mut args: Vec<String> = vec![subject.to_string(), out.clone(), "--map".into(), map.to_string()];
    for k in ["--server", "--tries", "--jobs", "--spawn-ref"] {
        if let Some(v) = flag(a, k) {
            args.push(k.into());
            args.push(v.to_string());
        }
    }
    // The subject IS the spawn reference: it is a real recording of this map,
    // so its own first sample is the answer key.
    if flag(a, "--spawn-ref").is_none() {
        args.push("--spawn-ref".into());
        args.push(subject.to_string());
    }
    crate::regen::cmd(&args);

    match compare(subject, &out) {
        Err(e) => {
            if !keep {
                let _ = std::fs::remove_file(&out);
            }
            die(format!("the two records could not be compared: {e}"));
        }
        Ok((med, worst, exact, n)) => {
            println!("\n== the answer key");
            println!(
                "   median {:.6} m, worst {:.6} m over {} shared instants",
                med, worst, n
            );
            println!(
                "   {exact} of {n} samples reproduce the original's 22 transform bytes EXACTLY"
            );
            let ok = med * 1000.0 <= bar_mm;
            if ok {
                println!(
                    "\nPASS: the pipeline reproduced a run it was given, to {:.3} mm. The floor \
                     for this comparison is 0.48-0.52 mm (a client recording re-simulated on a \
                     dedicated server), so anything at that scale is the two engines differing, \
                     not us.",
                    med * 1000.0
                );
            } else {
                if !keep {
                    let _ = std::fs::remove_file(&out);
                }
                die(format!(
                    "FAILED: {:.4} m median against a {:.0} mm bar. The pipeline did not \
                     reproduce a run it was handed, so nothing it produces on a tape we do NOT \
                     have the answer for can be trusted.{}",
                    med,
                    bar_mm,
                    if keep { format!(" {out} kept.") } else { String::new() }
                ));
            }
            if !keep {
                let _ = std::fs::remove_file(&out);
            } else {
                println!("   kept: {out}");
            }
        }
    }
    let _ = secs(0);
}
