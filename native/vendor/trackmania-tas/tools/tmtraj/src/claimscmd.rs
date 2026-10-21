//! `tmtraj corpus claims` — does a map's page agree with the files in its own
//! directory?
//!
//! This project's known failure mode is a headline contradicted by the
//! directory it sits in: a number in a README that no file supports, a file
//! whose NAME says one time and whose HEADER says another, a page that lists
//! files it does not have. `227654`'s `RESULT.md` and `RESULTS-entry.md` are
//! the worked example, and the standing instruction — *read a store directory
//! by mtime, never by filename* — is a workaround for not having this check.
//!
//! The scan is deliberately narrow. It only asks questions whose answer is a
//! fact about two artefacts that are both in the repo, so it can never be the
//! thing that manufactures doubt:
//!
//! | check | question | why it is not arguable |
//! |---|---|---|
//! | `NAME-VS-HEADER` | does `TAS_39183.Ghost.Gbx` declare 39.183? | both numbers are in the repo |
//! | `MISSING-FILE` | the page links `replays/X`; is `X` there? | ditto |
//! | `UNLISTED-FILE` | `X` is there; does the page ever name it? | ditto |
//! | `HEADLINE-UNBACKED` | the headline claims a time faster than anything the directory declares | ditto |
//!
//! ## `NAME-VS-HEADER` IS A QUESTION, NOT A VERDICT
//!
//! **A header is not the authority on what a tape does. The oracle is.** A
//! searched tape is built inside a carrier and routinely inherits that
//! carrier's declared time, which `ghost declare --from-oracle` rewrites and
//! which changes no physics.
//!
//! This scan flagged 146612's `TAS_39183` and `KEYBOARD_39706` as declaring
//! 39.555 apiece, and the audit read that as "these times are not supported by
//! anything published" and rewrote two pages around it. Then the oracle was
//! actually asked:
//!
//! ```text
//! TAS_39183.Ghost.Gbx        PASS V7   oracle re-simulated the written file: 39.183
//! KEYBOARD_39706.Ghost.Gbx   PASS V7   oracle re-simulated the written file: 39.706
//! ```
//!
//! The names were right and the headers were stale — and `tools/LINEAGE.md` had
//! already recorded exactly that for this directory, before the scan ran. So a
//! flag from this command means **go and ask the oracle**, and if the map is not
//! to hand, say the question is open rather than answering it from the header.
//!
//! It says nothing about whether a declared time is *true* — that is the
//! oracle's job, and the oracle needs the map, which this repo does not
//! redistribute. A clean `claims` run means the page and the directory agree,
//! not that either is right.
//!
//! ## The one deliberate exception
//!
//! A file named `..._declares_<ms>` is telling you on purpose that its header
//! carries somebody else's time — the convention from 134672, where zeroing
//! the field would read as "declares 0.000" to every tool we own. Those are
//! reported as `BY-NAME` and are not failures.

use crate::corpuscmd::{base, MapDir};
use crate::fmt::secs;
use gbx::record;

/// Times in a filename are milliseconds, 4 to 7 digits, delimited by `_`, `-`
/// or the extension. `TAS_57493.Ghost.Gbx` -> 57493.
fn times_in_name(name: &str) -> Vec<i64> {
    let stem = name.split(".Ghost").next().unwrap_or(name);
    let stem = stem.split(".Replay").next().unwrap_or(stem);
    let mut out = Vec::new();
    for tok in stem.split(|c: char| !c.is_ascii_digit()) {
        if tok.len() >= 4 && tok.len() <= 7 {
            if let Ok(v) = tok.parse::<i64>() {
                out.push(v);
            }
        }
    }
    out
}

/// `**12.345**` in the page, as milliseconds.
///
/// A bolded **delta** (`**−0.646**`, `**+7.018**`) is not a time and must not
/// be compared against a file. Missing that produced 14 false flags on the
/// first run of this scan, which would have been 14 invitations to hedge a
/// page that was right.
fn bold_times(line: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let b: Vec<&str> = line.split("**").collect();
    // odd indices are the bolded spans
    for (i, span) in b.iter().enumerate() {
        if i % 2 == 0 {
            continue;
        }
        let ch: Vec<char> = span.chars().collect();
        let mut j = 0usize;
        while j < ch.len() {
            if !ch[j].is_ascii_digit() {
                j += 1;
                continue;
            }
            let start = j;
            while j < ch.len() && (ch[j].is_ascii_digit() || ch[j] == '.') {
                j += 1;
            }
            // signed => a delta, not a time
            let signed = start > 0 && matches!(ch[start - 1], '-' | '+' | '\u{2212}' | '\u{00b1}');
            if signed {
                continue;
            }
            let tok: String = ch[start..j].iter().collect();
            if let Some((a, f)) = tok.split_once('.') {
                if f.len() == 3 && !a.is_empty() && a.len() <= 4 {
                    if let (Ok(s), Ok(ms)) = (a.parse::<i64>(), f.parse::<i64>()) {
                        out.push(s * 1000 + ms);
                    }
                }
            }
        }
    }
    out
}

/// A headline line that is about somebody ELSE's time is not a claim about our
/// files, and must not be compared against them.
fn is_about_others(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    [
        "author time", "author medal", " at ", "| at", "at ", "wr", "world record",
        "human", "record", "medal", "rank", "leaderboard", "by nobody",
    ]
    .iter()
    .any(|k| l.contains(k))
}

pub fn claims(maps: &[MapDir], root: &str) -> i32 {
    println!("map\tcheck\tsubject\tdetail");
    let mut flags = 0usize;
    let mut byname = 0usize;
    let mut checked = 0usize;

    // map id -> its directory, from the same layout `walk` uses.
    let dirs: Vec<(String, String)> = std::fs::read_dir(root)
        .map(|rd| {
            let mut v: Vec<(String, String)> = rd
                .filter_map(|e| e.ok())
                .map(|e| e.path().to_string_lossy().to_string())
                .filter(|p| std::path::Path::new(p).join("replays").is_dir())
                .map(|p| {
                    let n = base(&p).to_string();
                    (n.split('-').next().unwrap_or(&n).to_string(), p)
                })
                .collect();
            v.sort();
            v
        })
        .unwrap_or_default();

    for m in maps {
        let Some((_, dir)) = dirs.iter().find(|(id, _)| *id == m.id) else {
            continue;
        };
        let readme = std::path::Path::new(dir).join("README.md");
        let page = std::fs::read_to_string(&readme).unwrap_or_default();
        if page.is_empty() {
            println!("{}\tNO-PAGE\tREADME.md\tno README.md in this map directory", m.id);
            flags += 1;
            continue;
        }

        let mut dir_best: Option<i64> = None;
        let all: Vec<&String> = m.ours.iter().chain(m.refs.iter()).collect();

        // ---- NAME-VS-HEADER, and the directory's own best declared time ----
        for f in &all {
            let name = base(f);
            let Ok(d) = record::decode_ghost(f) else {
                println!("{}\tDECODE-FAIL\t{}\tcannot read this file", m.id, name);
                flags += 1;
                continue;
            };
            let Some(hdr) = d.race_time_ms.map(|v| v as i64) else {
                println!("{}\tNO-DECLARED-TIME\t{}\tthe header carries no race time", m.id, name);
                flags += 1;
                continue;
            };
            checked += 1;
            if !crate::corpuscmd::is_reference_pub(name) {
                dir_best = Some(dir_best.map_or(hdr, |b: i64| b.min(hdr)));
            }
            let named = times_in_name(name);
            if named.is_empty() {
                continue;
            }
            if named.contains(&hdr) {
                continue;
            }
            let lower = name.to_ascii_lowercase();
            if lower.contains("declares_") {
                // deliberate, and the name says so
                println!(
                    "{}\tBY-NAME\t{}\theader {} — the name declares this on purpose",
                    m.id,
                    name,
                    secs(hdr)
                );
                byname += 1;
                continue;
            }
            println!(
                "{}\tNAME-VS-HEADER\t{}\tthe name says {} — the header says {} (ASK THE ORACLE: a stale declaration is common and harmless)",
                m.id,
                name,
                named.iter().map(|v| secs(*v)).collect::<Vec<_>>().join(" / "),
                secs(hdr)
            );
            flags += 1;
        }

        // ---- MISSING-FILE / UNLISTED-FILE ----
        let mut linked: Vec<String> = Vec::new();
        for line in page.lines() {
            // A struck-through link is the page telling you the file is gone
            // ON PURPOSE — 279218 withdraws `TAS_5345_starttrick` that way,
            // with the reason beside it. That is the behaviour this scan wants
            // to encourage, so flagging it would be exactly backwards.
            if line.contains("~~") {
                continue;
            }
            for tok in line.split("replays/").skip(1) {
                let end = tok.find(|c: char| {
                    c == ')' || c == '`' || c == ' ' || c == '|' || c == '\n' || c == '"'
                });
                let n = &tok[..end.unwrap_or(tok.len())];
                if n.ends_with(".Ghost.Gbx") || n.ends_with(".Replay.Gbx") {
                    linked.push(n.to_string());
                }
            }
        }
        linked.sort();
        linked.dedup();
        for n in &linked {
            if !std::path::Path::new(dir).join("replays").join(n).exists() {
                println!("{}\tMISSING-FILE\t{}\tthe page links it; it is not in replays/", m.id, n);
                flags += 1;
            }
        }
        for f in &all {
            let n = base(f);
            if !page.contains(n) {
                println!("{}\tUNLISTED-FILE\t{}\tin replays/; the page never names it", m.id, n);
                flags += 1;
            }
        }

        // ---- HEADLINE-UNBACKED ----
        // A bolded time in the page's headline block, on a line that is not
        // about a human or an author medal, that is FASTER than anything this
        // directory declares. That is a headline no file here supports.
        if let Some(best) = dir_best {
            for line in page.lines().take(40) {
                if is_about_others(line) {
                    continue;
                }
                for t in bold_times(line) {
                    if t < best && t > 500 {
                        println!(
                            "{}\tHEADLINE-UNBACKED\t{}\tthe fastest time this directory declares is {}",
                            m.id,
                            secs(t),
                            secs(best)
                        );
                        flags += 1;
                    }
                }
            }
        }
    }

    println!();
    println!(
        "{} files read, {} flags, {} deliberate BY-NAME",
        checked, flags, byname
    );
    if flags == 0 {
        println!("every page agrees with its own directory. This says nothing about whether");
        println!("the declared times are TRUE -- that needs the oracle, and the oracle needs");
        println!("the map, which this repo does not redistribute.");
    }
    if flags > 0 {
        1
    } else {
        0
    }
}
