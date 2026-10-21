//! `tmtraj corpus shipped` — is the number the ROOT README claims for a map
//! backed by a file anybody can download?
//!
//! `corpus claims` asks whether a map's own page agrees with its own
//! directory. This asks the question one level up, which is the one a reader
//! of the front page actually has: **the root README's `this TAS` column names
//! a time — is that run in `replays/`?**
//!
//! Two maps in this repo are known to answer no on purpose (Spaghetti Nights 2
//! holds a 38.968 it does not ship; Fall 2024 - 25's 95.507 is a search tape
//! that is not renderable), and each says so in its own row. The point of the
//! scan is that "held but not shipped" is a *different status* from "beaten",
//! and nothing in the repo computed which maps were in it — it was two
//! sentences of prose that had to be remembered.
//!
//! ## What counts as backing a claim
//!
//! A published file backs the claim if **its header declares that time, or its
//! filename states it**. Both, because neither alone is authoritative here:
//!
//! * a searched tape is built inside a carrier and inherits the carrier's
//!   declared time until `ghost declare --from-oracle` rewrites it — 146612's
//!   `TAS_39183` declares 39.555 and re-simulates to 39.183, and the *name* is
//!   the true one (see `claimscmd.rs`);
//! * a filename is a claim by a person, so a file whose header agrees with the
//!   root README is backing regardless of what its name says.
//!
//! Neither is the oracle. This scan says whether a file exists that *claims to
//! be* the headline run; only re-simulation against the map says whether it is.
//! Reference recordings (`wr|human|rank|author` in the name) are somebody
//! else's runs and never back one of our claims.

use crate::corpuscmd::{base, is_reference_pub, MapDir};
use crate::fmt::{delta, secs};
use gbx::record;

/// Times in a filename are milliseconds, 4 to 7 digits, delimited by anything
/// that is not a digit. `TAS_57493.Ghost.Gbx` -> 57493.
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

/// `12.345`, `**12.345**`, `` `12.345` `` as milliseconds. A signed token is a
/// delta, not a time, and is refused — the root README's tables carry a
/// `vs AT` column full of them.
fn cell_ms(tok: &str) -> Option<i64> {
    let t = tok.trim().trim_matches(|c| c == '*' || c == '`' || c == ' ');
    if t.starts_with('-') || t.starts_with('+') || t.starts_with('\u{2212}') || t.starts_with('\u{b1}')
    {
        return None;
    }
    let t: String = t.chars().filter(|c| *c != ',').collect();
    let (a, f) = t.split_once('.')?;
    if f.len() != 3 || a.is_empty() || !a.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(a.parse::<i64>().ok()? * 1000 + f.parse::<i64>().ok()?)
}

/// One entry of the root README. The front page used to be two tables and is
/// now one line per map; both shapes are read, because a scan that only
/// understands the current layout stops being a check the moment somebody
/// reformats the page.
///
/// The **line** contract, which is what the front page uses today:
///
/// ```text
/// **[Name](dir)** — author time `AT` · ours **TAS** (…) · best human WR (holder) · N records
/// ```
///
/// * it is **one physical line** — a wrapped entry is invisible here;
/// * `author time` is what marks a line as an entry at all, so prose that
///   happens to open with a link (`**[▶ Drive one of them.](trainer/)**`) is
///   not mistaken for a claim;
/// * `ours` introduces this project's claim and **the time must be the first
///   token after it**. That is not fussiness: Underwater's entry reads
///   `ours — *no completion; the page's 36.049 is a landing, not a lap*`, and
///   a parser that took the first time-shaped token anywhere in the field
///   would publish 36.049 as a claim this repo does not make. It is the
///   negative control in the tests below.
/// * the human record is `best human …`, or in group 3
///   `beaten by a human: …` / `equalled by a human: …`.
///
/// The **table** contract, kept for the older shape: map, records, author
/// time, best human, this TAS in the first five columns.
pub struct RootClaim {
    pub dir: String,
    pub name: String,
    pub at_ms: Option<i64>,
    pub human_ms: Option<i64>,
    pub tas_ms: Option<i64>,
}

/// The first token of `rest`, as a time. Leading markdown and a leading colon
/// are skipped; anything else that is not itself a time — `—`, `*none*`,
/// `UNKNOWN` — reads as no claim, which is the point.
fn lead_ms(rest: &str) -> Option<i64> {
    let t = rest.trim_start_matches(|c| c == ' ' || c == '*' || c == '`' || c == ':');
    cell_ms(t.split_whitespace().next()?)
}

/// `**[Name](dir)**` at the head of a line.
fn link_at_head(l: &str) -> Option<(String, String)> {
    let open = l.find('[')?;
    let link = l.find("](")?;
    if link < open {
        return None;
    }
    let close = l[link + 2..].find(')')?;
    Some((l[link + 2..link + 2 + close].to_string(), l[open + 1..link].to_string()))
}

fn line_claim(l: &str) -> Option<RootClaim> {
    if !l.starts_with("**[") || !l.contains("author time") {
        return None;
    }
    let (dir, name) = link_at_head(l)?;
    let mut c = RootClaim { dir, name, at_ms: None, human_ms: None, tas_ms: None };
    for field in l.split('\u{b7}') {
        let f = field.trim();
        if let Some(i) = f.find("author time") {
            c.at_ms = c.at_ms.or_else(|| lead_ms(&f[i + "author time".len()..]));
        }
        if f.starts_with("ours") {
            c.tas_ms = c.tas_ms.or_else(|| lead_ms(&f["ours".len()..]));
        }
        if let Some(i) = f.find("best human") {
            c.human_ms = c.human_ms.or_else(|| lead_ms(&f[i + "best human".len()..]));
        }
        if let Some(i) = f.find("by a human:") {
            c.human_ms = c.human_ms.or_else(|| lead_ms(&f[i + "by a human:".len()..]));
        }
    }
    Some(c)
}

pub fn root_claims(readme: &str) -> Vec<RootClaim> {
    let mut out = Vec::new();
    for line in readme.lines() {
        let l = line.trim();
        if let Some(c) = line_claim(l) {
            out.push(c);
            continue;
        }
        if !l.starts_with("| [") {
            continue;
        }
        let cells: Vec<&str> = l.trim_matches('|').split('|').collect();
        if cells.len() < 5 {
            continue;
        }
        let first = cells[0].trim();
        let (Some(open), Some(link)) = (first.find('['), first.find("](")) else { continue };
        let Some(close) = first[link + 2..].find(')') else { continue };
        out.push(RootClaim {
            dir: first[link + 2..link + 2 + close].to_string(),
            name: first[open + 1..link].to_string(),
            at_ms: cell_ms(cells[2]),
            human_ms: cell_ms(cells[3]),
            tas_ms: cell_ms(cells[4]),
        });
    }
    out
}

pub fn shipped(maps: &[MapDir], root: &str) -> i32 {
    let readme = match std::fs::read_to_string(std::path::Path::new(root).join("README.md")) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tmtraj corpus shipped: read {}/README.md: {}", root, e);
            return 2;
        }
    };
    let claims = root_claims(&readme);
    if claims.is_empty() {
        eprintln!("tmtraj corpus shipped: no result-table rows in {}/README.md", root);
        return 2;
    }

    println!("map\tdir\tclaim\tverdict\tbest_header\tbest_name\tfiles\tdetail");
    let mut unbacked = 0usize;
    let mut backed = 0usize;
    let mut noclaim = 0usize;
    let mut behind = 0usize;

    for c in &claims {
        let id = c.dir.split('-').next().unwrap_or(&c.dir).to_string();
        let Some(m) = maps.iter().find(|m| m.id == id) else {
            // No `replays/` at all. That is only an unbacked CLAIM if the
            // front page claims a time here: P-Found - Pokeuuu has no
            // directory and claims nothing, and reading that as a failure was
            // the scan calling an honest absence a defect.
            let Some(t) = c.tas_ms else {
                println!(
                    "{}\t{}\t—\tNO-CLAIM\t—\t—\t0\tthe root README claims no time on this map, and it has no replays/ directory",
                    id, c.dir
                );
                noclaim += 1;
                continue;
            };
            println!(
                "{}\t{}\t{}\tNO-REPLAYS-DIR\t—\t—\t0\tthe root README claims this time; the map has no replays/ directory",
                id,
                c.dir,
                secs(t)
            );
            unbacked += 1;
            continue;
        };

        // Ours only: a reference recording is somebody else's run.
        let ours: Vec<&String> = m.ours.iter().filter(|f| !is_reference_pub(base(f))).collect();

        let Some(claim) = c.tas_ms else {
            println!(
                "{}\t{}\t—\tNO-CLAIM\t—\t—\t{}\tthe root README claims no time on this map",
                id,
                c.dir,
                ours.len()
            );
            noclaim += 1;
            continue;
        };

        let mut best_hdr: Option<(i64, String)> = None;
        let mut best_name: Option<(i64, String)> = None;
        let mut backer: Option<(String, &'static str)> = None;
        for f in &ours {
            let name = base(f).to_string();
            let named = times_in_name(&name);
            // A file that says in its own name that it is not the lap -- a
            // segment, a fragment, a tape whose header is somebody else's --
            // is not a candidate for "the fastest thing published here". The
            // repo names those on purpose so a reader cannot mistake them.
            let lower = name.to_ascii_lowercase();
            let not_a_lap = lower.contains("do_not_publish")
                || lower.contains("segment")
                || lower.contains("declares_");
            if !not_a_lap {
                if let Some(t) = named.iter().copied().min() {
                    if best_name.as_ref().map_or(true, |(b, _)| t < *b) {
                        best_name = Some((t, name.clone()));
                    }
                }
                if named.contains(&claim) && backer.is_none() {
                    backer = Some((name.clone(), "name"));
                }
            }
            let Ok(d) = record::decode_ghost(f) else { continue };
            let Some(hdr) = d.race_time_ms.map(|v| v as i64) else { continue };
            if !not_a_lap {
                if best_hdr.as_ref().map_or(true, |(b, _)| hdr < *b) {
                    best_hdr = Some((hdr, name.clone()));
                }
                if hdr == claim {
                    backer = Some((name.clone(), "header"));
                }
            }
        }

        let show = |v: &Option<(i64, String)>| match v {
            Some((t, f)) => format!("{} ({})", secs(*t), f),
            None => "—".to_string(),
        };

        if ours.is_empty() {
            println!(
                "{}\t{}\t{}\tNO-FILES\t—\t—\t0\treplays/ holds no run of ours",
                id, c.dir, secs(claim)
            );
            unbacked += 1;
            continue;
        }

        if let Some((file, how)) = backer {
            // A published file can also be FASTER than the front page's
            // number: 134672 ships a 67.200 under a root README that still
            // says 67.319, and 267460 ships an 18.234 under one that still
            // says 21.022. That is a stale front page, and it is invisible to
            // every per-directory check.
            let ahead = best_hdr.as_ref().filter(|(t, _)| *t < claim);
            if let Some((t, f)) = ahead {
                println!(
                    "{}\t{}\t{}\tFRONT-PAGE-BEHIND\t{}\t{}\t{}\treplays/ holds a faster run than the root README claims: {} declares {} ({} better)",
                    id,
                    c.dir,
                    secs(claim),
                    show(&best_hdr),
                    show(&best_name),
                    ours.len(),
                    f,
                    secs(*t),
                    delta(t - claim)
                );
                behind += 1;
                continue;
            }
            println!(
                "{}\t{}\t{}\tBACKED\t{}\t{}\t{}\tthe claim is stated by this file's {}: {}",
                id,
                c.dir,
                secs(claim),
                show(&best_hdr),
                show(&best_name),
                ours.len(),
                how,
                file
            );
            backed += 1;
            continue;
        }

        // Nothing states the claim. Report the closest thing that IS
        // published, against BOTH readings of the directory: a header always
        // states a whole run's time (and can be a carrier's), a name states
        // what a person meant by the file (and can name a fragment). Quoting
        // one of them alone is how "the fastest run published here" gets to be
        // a segment.
        if best_hdr.is_none() && best_name.is_none() {
            println!(
                "{}\t{}\t{}\tUNREADABLE\t—\t—\t{}\tno file in replays/ states a time at all",
                id,
                c.dir,
                secs(claim),
                ours.len()
            );
            unbacked += 1;
            continue;
        }
        let gap = |v: &Option<(i64, String)>| match v {
            Some((t, _)) => delta(claim - t),
            None => "—".to_string(),
        };
        println!(
            "{}\t{}\t{}\tUNSHIPPED\t{}\t{}\t{}\tno published file states the front page's time; against the fastest header here {}, against the fastest name here {}",
            id,
            c.dir,
            secs(claim),
            show(&best_hdr),
            show(&best_name),
            ours.len(),
            gap(&best_hdr),
            gap(&best_name)
        );
        unbacked += 1;
    }

    println!();
    println!(
        "{} claims read, {} backed by a published file, {} not, {} behind a faster published file, {} maps claim no time",
        claims.len(),
        backed,
        unbacked,
        behind,
        noclaim
    );
    println!("A BACKED row means a file states that time, not that the time is true:");
    println!("only the oracle re-simulating the file against the map says that.");
    if unbacked > 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape the front page uses today: one line per map, three groups,
    /// no tables. Three things are load-bearing and each has its own
    /// assertion below.
    #[test]
    fn reads_the_one_line_front_page() {
        let readme = "\
**[▶ Drive one of them.](trainer/)** The 6.323 on unluckE is 23 steer events\n\
\n\
**[Tap water 01](173636-tap-water-01)** — author time `23.325` · **beaten by a human: 23.298 (Lukrecja666)** · ours **22.072** (−1.253) · 655 records\n\
**[[Turtle Trial] Angustus](238835-turtle-trial-angustus)** — author time `462.982` · ours **239.133** (−223.849) · best human 1964.933 (Quantiks) · 1 record\n\
**[untitled 01](276874-untitled-01)** — author time `23.839` · ours **12.759** (−11.080) · best human *none — the board is empty* · 0 records\n\
**[Spring 2023 - 15 (Underwater)](173691-spring-2023-15-underwater)** — author time `2672.290` · **beaten by a human: 1571.209 (Maionez77)** · ours — *no completion; the page's 36.049 is a landing, not a lap* · 1 record\n\
**[Fall 2025 - 18 CP1 End](270053-fall-2025-18-cp1-end)** — author time `4.492` · ours **4.492** (±0 — an exact tie takes the medal) · best human 4.495 (AffiTM, six players tied) · 1101 records\n";
        let rows = root_claims(readme);

        // The trainer link opens with `**[` and is not a claim. A line is an
        // entry only if it states an author time.
        assert_eq!(rows.len(), 5);

        assert_eq!(rows[0].dir, "173636-tap-water-01");
        assert_eq!(rows[0].name, "Tap water 01");
        assert_eq!(rows[0].at_ms, Some(23325));
        assert_eq!(rows[0].human_ms, Some(23298)); // "beaten by a human:"
        assert_eq!(rows[0].tas_ms, Some(22072));

        // A name with its own brackets must not truncate at the inner one.
        assert_eq!(rows[1].name, "[Turtle Trial] Angustus");
        assert_eq!(rows[1].dir, "238835-turtle-trial-angustus");
        assert_eq!(rows[1].tas_ms, Some(239133));
        assert_eq!(rows[1].human_ms, Some(1964933));

        // An empty board is an absence, not a time.
        assert_eq!(rows[2].human_ms, None);
        assert_eq!(rows[2].tas_ms, Some(12759));

        // THE NEGATIVE CONTROL. Underwater claims nothing, and its `ours`
        // field mentions a 36.049 that is a landing rather than a lap. A
        // parser that scanned the field for a time instead of reading the
        // token after `ours` would publish a claim this repo does not make.
        assert_eq!(rows[3].tas_ms, None);
        assert_eq!(rows[3].human_ms, Some(1571209));
        assert_eq!(rows[3].at_ms, Some(2672290));

        // Equalling is a claim of the author time, not an absence of one.
        assert_eq!(rows[4].tas_ms, Some(4492));
        assert_eq!(rows[4].human_ms, Some(4495));
    }

    /// The two real table shapes, and the negative control that matters: the
    /// `vs AT` column is full of signed deltas and a delta is not a time. The
    /// predecessor of this parser (in `claimscmd`) produced 14 false flags by
    /// reading them as times.
    #[test]
    fn reads_both_root_tables_and_refuses_the_delta_columns() {
        let readme = "\
| map | records | author time | best human | **this TAS** | vs AT |\n\
|---|---|---|---|---|---|\n\
| [Tap water 01](173636-tap-water-01) | 602 | 23.325 | 23.298 | **22.072** | −1.253 |\n\
| [untitled 01](276874-untitled-01) | **0** | 23.839 | *none* | **12.759** | **−46.5 %** |\n\
\n\
| map | records | author time | best human | **this TAS** | short of AT by |\n\
| [P-Found - Pokeuuu](153527-p-found-pokeuuu) | 1 | 939.283 | 5661.335 | — | — |\n\
not a table row at all\n";
        let rows = root_claims(readme);
        assert_eq!(rows.len(), 3);

        assert_eq!(rows[0].dir, "173636-tap-water-01");
        assert_eq!(rows[0].name, "Tap water 01");
        assert_eq!(rows[0].at_ms, Some(23325));
        assert_eq!(rows[0].human_ms, Some(23298));
        assert_eq!(rows[0].tas_ms, Some(22072));

        // `*none*` is an absence, not a time; the claim is still read.
        assert_eq!(rows[1].human_ms, None);
        assert_eq!(rows[1].tas_ms, Some(12759));

        // A map with no claim of ours must come back as no claim, never as 0.
        assert_eq!(rows[2].dir, "153527-p-found-pokeuuu");
        assert_eq!(rows[2].tas_ms, None);
    }

    #[test]
    fn a_signed_token_is_never_a_time() {
        assert_eq!(cell_ms("**−1.253**"), None);
        assert_eq!(cell_ms("+4.134"), None);
        assert_eq!(cell_ms("-0.021"), None);
        assert_eq!(cell_ms("±0"), None);
        assert_eq!(cell_ms("**22.072**"), Some(22072));
        assert_eq!(cell_ms("2,540.641"), Some(2540641));
        assert_eq!(cell_ms("—"), None);
    }

    /// A filename states milliseconds; `28ms` in the middle of a name is two
    /// digits and must not be read as one.
    #[test]
    fn times_in_a_filename() {
        assert_eq!(times_in_name("TAS_57493.Ghost.Gbx"), vec![57493]);
        assert_eq!(times_in_name("UTURN_28ms_east_at_6990.Ghost.Gbx"), vec![6990]);
        assert!(times_in_name("BEST.Ghost.Gbx").is_empty());
    }
}
