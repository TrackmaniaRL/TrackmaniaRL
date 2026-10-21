//! `ghost declare` -- write the time the file ACTUALLY does into every copy of
//! it, asked of the plain oracle.
//!
//! Moved out of `main.rs` because `regen`'s finishing pass calls it on every
//! run: a regenerated file's declared time is the CARRIER's until something
//! rewrites it, and "something" was a person remembering. A CLI entry point
//! that only main.rs can reach is a step that can be skipped.

use crate::cli::{die, flag, has, num};
use crate::{oracle, trim, Container};
use gbx::container;
use gbx::container::secs;

/// `ghost declare` -- set the time the file DECLARES.
///
/// After the inputs change, the declared time is the old run's. Every check
/// that compares "what the file says" with "what the file does" then compares a
/// stale number, and a file that declares somebody else's time while validating
/// its own is exactly the shape of the container bugs that cost this project
/// five withdrawn clips.
///
/// `--from-oracle` is the form to use: it asks the plain oracle what the file
/// actually does and writes THAT, so the number is never typed by a human and
/// never copied out of a search log.
pub fn cmd(a: &[String]) {
    let inp = a.first().unwrap_or_else(|| die("ghost declare IN OUT (--time MS | --from-oracle --map M) [--splits MS,MS,...] [--cps N]"));
    let out = a.get(1).unwrap_or_else(|| die("ghost declare IN OUT (--time MS | --from-oracle --map M) [--splits MS,MS,...] [--cps N]"));
    let c = Container::load(inp).unwrap_or_else(|e| die(e));
    let ms: i64 = if has(a, "--from-oracle") {
        let server = oracle::server_dir(flag(a, "--server"));
        let mode = match flag(a, "--map") {
            Some(m) => oracle::MapsMode::One(std::path::Path::new(m)),
            None => oracle::MapsMode::Empty,
        };
        match oracle::validate(&server, std::path::Path::new(inp), mode, "declare") {
            Err(e) => die(format!("the oracle could not run: {}", e)),
            Ok(r) => match r.time_ms {
                Some(t) => {
                    println!("the plain oracle re-simulated {} to {}", inp, secs(t));
                    t
                }
                None => die(format!(
                    "the oracle returns DNF (cps {:?}) for this file, so there is no time to \
                     declare. A partial run should keep the window's end: use --time MS.",
                    r.cps
                )),
            },
        }
    } else {
        num(a, "--time").unwrap_or_else(|| die("give --time MS or --from-oracle --map M"))
    };
    let mut body = c.body().to_vec();
    trim::set_all_declared(&mut body, ms as u32);
    // The ghost-result chunk carries the race time AND the checkpoint list, and
    // the server compares the LENGTH of that list with the map's checkpoint
    // count before it simulates anything.
    let want_cps = num(a, "--cps");
    // --splits a,b,c: the intermediate checkpoint times, in seconds or ms.
    //
    // The gap this fills: `ghost declare` wrote the declared TIME at every site
    // and had nothing that wrote the SPLIT LIST, so a repaired file could carry
    // its own time and the carrier's checkpoints -- which is exactly what
    // 238835's two NORETRY files do, and the page has to say so. A file whose
    // splits are somebody else's is not fully ours however good its trajectory
    // is.
    //
    // The times are PASSED IN rather than derived, and that is deliberate: the
    // plain oracle reports `NbCheckpoints` and not the crossing times, so this
    // toolchain has no way to measure them from the file alone. Inventing them
    // would be worse than leaving them -- `--cps` already writes 0.000, which
    // reads as "this container does not know its splits" where a donor's number
    // reads as a measurement. When a caller HAS the times (from the search that
    // produced the run, or from `tmmaps`' per-checkpoint segment maps), this is
    // how they get into the file.
    let want_splits: Option<Vec<i64>> = flag(a, "--splits").map(|v| {
        v.split(',')
            .map(|x| {
                let t = x.trim();
                let f: f64 = t.parse().unwrap_or_else(|_| die(format!("--splits: {t:?} is not a number")));
                // Seconds with a decimal is how this project writes times, so
                // "12.475" means 12475 ms; a bare integer is already ms.
                if t.contains('.') { (f * 1000.0).round() as i64 } else { f as i64 }
            })
            .collect()
    });
    if let (Some(sp), Some(n)) = (&want_splits, want_cps) {
        if sp.len() as i64 + 1 != n {
            die(format!(
                "--splits has {} intermediate checkpoint(s) and --cps says {} in total; the last \
                 entry is the finish and is taken from the declared time, so --splits wants {}",
                sp.len(),
                n,
                n - 1
            ));
        }
    }
    let before: Vec<i32> = c.splits();
    // Did the run this file describes just change? The old declared time is
    // read BEFORE any rewrite; see the stale-splits block below.
    let old_declared: Option<i64> =
        c.declared_times().first().map(|x| x.1 as i64);
    let stale_splits = match old_declared {
        Some(old) => old != ms,
        // A container that declares no time at all cannot vouch for its
        // splits either.
        None => true,
    };
    let body = trim::rewrite_result(&body, |r| {
        r.race_ms = ms as i32;
        if let Some(sp) = &want_splits {
            // THE SPLITS ARE A CLAIM, AND THEY WERE THE CARRIER'S.
            //
            // `--time` rewrites the race time and the LAST entry and leaves
            // every intermediate one alone -- so a searched tape on a borrowed
            // container declares its own finish beside the donor's checkpoint
            // times. On 134672 that reads 13.906 / 33.106 / 45.437 / 63.812 /
            // 67.200: four of another driver's splits and one of ours, in one
            // list, with nothing in the file to say which is which. The
            // deleted `u02 declare --splits` could write them and its
            // replacement could not, and every regenerated ghost since has
            // carried the gap.
            //
            // Measure them the way this toolchain measures a split -- one
            // verified segment map per checkpoint, `tmmaps segments` then
            // `tmmaps oracle` -- and pass them here. Milliseconds, like
            // `--time`. The read-back control below requires the last one to
            // BE the declared time, which is the one relation that holds on
            // every reference ghost in the corpus.
            r.entries = sp.iter().map(|t| (*t as i32, 1)).collect();
        } else if let Some(n) = want_cps {
            // Every intermediate entry becomes 0.000. This is the borrowed-
            // container case by construction -- the count only changes when the
            // file moved to a map with a different number of checkpoints -- and
            // every intermediate time in the list is then the DONOR map's,
            // measured on a route this file no longer drives. A zero says "this
            // container does not know its intermediate splits"; carrying the
            // donor's numbers forward would say something false and look right.
            //
            // WHAT THIS IS AND IS NOT FOR. The audit that asked for `--cps`
            // believed the server refused a count mismatch as `wrong simu`
            // without simulating. It does not: measured on two maps and six
            // counts, a finishing tape validates whatever its declared count
            // says, and `wrong simu` is what the server returns when the
            // simulation does not reproduce the DECLARED RESULT -- on a
            // partial run it even says how far it got (`wrong simu, but
            // reached some checkpoints (1 out of 2)`), which is a simulation,
            // not a pre-check. The count is a claim this toolchain reads:
            // `tmmaps` builds a segment map per declared split and refuses a
            // ghost whose count is not the map's.
            let n = n.clamp(1, 199) as usize;
            r.entries = (0..n)
                .map(|k| {
                    if k + 1 == n {
                        (ms as i32, 1)
                    } else {
                        match &want_splits {
                            Some(sp) => (sp.get(k).copied().unwrap_or(0) as i32, 1),
                            None => (0, 0),
                        }
                    }
                })
                .collect();
        } else if let Some(sp) = &want_splits {
            // No --cps: keep the list's length and write the times we were
            // given into it, the last entry being the race time.
            let n = r.entries.len();
            if n != sp.len() + 1 {
                die(format!(
                    "this file declares {} checkpoint(s) and --splits gives {} intermediate \
                     time(s). Pass --cps {} to change the count as well.",
                    n,
                    sp.len(),
                    sp.len() + 1
                ));
            }
            for (k, e) in r.entries.iter_mut().enumerate() {
                e.0 = if k + 1 == n { ms as i32 } else { sp[k] as i32 };
            }
        } else if let Some(last) = r.entries.last_mut() {
            // The final split is the race time. That holds on every reference
            // ghost in the corpus, so a `--time` that left it behind would
            // produce a file whose own last checkpoint disagrees with its own
            // declared time.
            last.0 = ms as i32;
        }
        // ---- STALE INTERMEDIATE SPLITS, ZEROED UNCONDITIONALLY -------------
        //
        // Everything above rewrites the FINAL time and, without `--splits`,
        // leaves the intermediate checkpoint times exactly as they were. When
        // the time just changed, those intermediates describe A DIFFERENT RUN
        // -- the donor's -- and nothing downstream can tell, because a
        // borrowed split table is a plausible list of increasing numbers that
        // no check compares against anything.
        //
        // This is the byte-73 defect one level up, and it reached a published
        // clip: 287431's whole lineage is seeded from ITZYNO1FAN's 24.092, and
        // every tape it has ever produced declared HIS six splits
        // (5.154 / 7.375 / 9.920 / 11.680 / 14.477 / 16.945) beside its own
        // finish. A viewer reads "this run reached CP6 at 16.945"; that is his
        // crossing, not ours.
        //
        // So: if the declared time CHANGED, the intermediates are stale by
        // construction, and a zero is written instead. `--cps` already
        // establishes the meaning -- 0.000 reads as "this container does not
        // know its intermediate splits", where a donor's number reads as a
        // measurement. Pass `--splits` when you have measured them and this
        // does not fire.
        //
        // The test is "the time changed" rather than "this is a synthesised
        // tape", because that is the part this code can actually know. Its one
        // blind spot: a searched tape whose time lands exactly on the donor's
        // keeps the donor's splits. Narrow, and the alternative -- zeroing the
        // splits of a genuine recording being re-declared to its own time --
        // destroys good data on every honest file.
        if want_splits.is_none() && stale_splits {
            let n = r.entries.len();
            for (k, e) in r.entries.iter_mut().enumerate() {
                if k + 1 != n {
                    *e = (0, 0);
                }
            }
        }
    })
    .unwrap_or_else(|e| die(e));
    let stage = format!("{}.declare-stage", out);
    container::write_gbx(&c.gbx, body, &stage).unwrap_or_else(|e| die(e));
    // The telemetry record declares its own span, separately from the samples.
    // Leaving it at the old run's is the same defect one level down, and
    // `ghost verify` reports it, so fix it here rather than print it later.
    let mut span_note = String::new();
    if gbx::recwrite::find_rec_site(&Container::load(&stage).unwrap().gbx.body).is_ok() {
        let r = gbx::recwrite::rewrite_ghost(&stage, out, |rd| {
            let last = rd.ents.iter().filter_map(|e| e.times.last().copied()).max().unwrap_or(0);
            rd.end_ms = (ms as i32).max(last);
            Ok(())
        });
        match r {
            Ok(_) => {
                let _ = std::fs::remove_file(&stage);
                span_note = format!("  the record's own span now ends at {}", secs(ms));
            }
            Err(e) => {
                std::fs::rename(&stage, out).ok();
                span_note = format!("  the record span was left alone: {}", e);
            }
        }
    } else {
        std::fs::rename(&stage, out).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
    }
    let c2 = Container::load(out).unwrap_or_else(|e| die(e));
    let dt: Vec<u32> = c2.declared_times().into_iter().map(|x| x.1).collect();
    if dt.iter().any(|v| *v as i64 != ms) {
        die(format!("read-back control FAILED: declared copies are {:?}", dt));
    }
    let after: Vec<i32> = c2.splits();
    if want_splits.is_none() && stale_splits {
        // Read-back control for the zeroing above, and the notice a human
        // needs: this file's intermediate splits are now BLANK, on purpose,
        // and they will stay blank until somebody measures them.
        let kept: Vec<i32> = after.iter().take(after.len().saturating_sub(1)).copied().collect();
        if kept.iter().any(|v| *v != 0) {
            die(format!(
                "read-back control FAILED: the declared time changed, so the intermediate \
                 splits should have been zeroed, and the file still declares {:?}",
                kept
            ));
        }
        if !before.is_empty() {
            println!(
                "  the declared time changed ({} -> {}), so the {} intermediate split(s) are \
                 STALE -- they timed the old run. They were {:?} and are now 0.000.",
                match old_declared {
                    Some(o) => secs(o),
                    None => "none".to_string(),
                },
                secs(ms),
                before.len().saturating_sub(1),
                before
                    .iter()
                    .take(before.len().saturating_sub(1))
                    .map(|v| secs(*v as i64))
                    .collect::<Vec<_>>()
            );
            println!(
                "  0.000 means \"this container does not know its intermediate splits\". \
                 To fill them in, measure each checkpoint crossing and pass --splits."
            );
        }
    }
    if let Some(sp) = &want_splits {
        if after.iter().map(|v| *v as i64).collect::<Vec<i64>>() != *sp {
            die(format!(
                "read-back control FAILED: asked for splits {:?}, the file declares {:?}",
                sp, after
            ));
        }
        match sp.last() {
            Some(last) if *last as i64 == ms => {}
            Some(last) => die(format!(
                "the last split is {} and the declared time is {}. On every reference ghost in \
                 this corpus the final entry IS the race time, so a file written this way would \
                 contradict itself.",
                secs(*last as i64),
                secs(ms)
            )),
            None => die("--splits was empty"),
        }
    }
    if let Some(n) = want_cps {
        if after.len() as i64 != n.clamp(1, 199) {
            die(format!(
                "read-back control FAILED: asked for {} checkpoints, the file declares {}",
                n,
                after.len()
            ));
        }
    }
    // The split list has to come back as asked, and it has to be MONOTONIC and
    // inside the run: a list that is not increasing, or whose last intermediate
    // is past the finish, is another run's however it got there. That is the
    // cheapest test for the defect this flag exists to repair, and it costs one
    // read of the file we just wrote.
    if let Some(sp) = &want_splits {
        let got: Vec<i32> = after.iter().copied().collect();
        let want: Vec<i32> = sp.iter().map(|v| *v as i32).chain(std::iter::once(ms as i32)).collect();
        if got != want {
            die(format!(
                "read-back control FAILED: splits are {:?}, asked for {:?}",
                got.iter().map(|v| secs(*v as i64)).collect::<Vec<_>>(),
                want.iter().map(|v| secs(*v as i64)).collect::<Vec<_>>()
            ));
        }
    }
    if after.windows(2).any(|w| w[1] <= w[0]) || after.last().map_or(false, |l| *l as i64 != ms) {
        println!(
            "  NOTE: the checkpoint list {:?} is not strictly increasing up to the declared time \
             {}. On a repaired container that is the DONOR's list surviving; pass --splits with \
             this run's own crossing times, or --cps N to zero them honestly.",
            after.iter().map(|s| secs(*s as i64)).collect::<Vec<_>>(),
            secs(ms)
        );
    }
    if let Some(r) = c2.result() {
        if r.race_ms as i64 != ms {
            die(format!(
                "read-back control FAILED: the result chunk declares {} and the header {}",
                secs(r.race_ms as i64),
                secs(ms)
            ));
        }
    }
    println!("wrote {}", out);
    println!("  declared {} in {} copies, all equal (read-back control OK)", secs(ms), dt.len());
    println!("  the ghost-result chunk's race time was set to the same value");
    if want_splits.is_some() {
        println!(
            "  checkpoints {:?}  (was {:?})",
            after.iter().map(|s| secs(*s as i64)).collect::<Vec<_>>(),
            before.iter().map(|s| secs(*s as i64)).collect::<Vec<_>>()
        );
    }
    if want_cps.is_some() {
        println!(
            "  checkpoints {:?}  (was {:?})",
            after.iter().map(|s| secs(*s as i64)).collect::<Vec<_>>(),
            before.iter().map(|s| secs(*s as i64)).collect::<Vec<_>>()
        );
        println!(
            "  NOTE: the intermediate times are written as 0.000 because this file's old ones\n\
             \x20       were another map's, and 0.000 reads as \"this container does not know its\n\
             \x20       splits\" where a donor's number would read as a measurement. MEASURED on\n\
             \x20       the dedicated server (build 2026-05-15): neither the COUNT nor the VALUES\n\
             \x20       gate validation -- counts of 1, 2, 3, 5 and 9 on a 4-checkpoint map all\n\
             \x20       come back IsValid true at the same time, and so do zeroed splits. The\n\
             \x20       count matters to THIS toolchain: a segment map is built against the\n\
             \x20       ghost's declared splits, and tmmaps refuses when there are the wrong\n\
             \x20       number of them."
        );
    }
    if !span_note.is_empty() {
        println!("{}", span_note);
    }
}
