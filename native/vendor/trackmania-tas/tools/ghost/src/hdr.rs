//! The GBX **header** user-data of a `.Replay.Gbx`: its identity strings and
//! its own copies of the declared time — the second container that no check in
//! this toolchain read.
//!
//! ## What was wrong
//!
//! `ident.rs` classifies the identity strings in the **body**. `cmd_declare`
//! rewrites the declared time in the **body**. `verify`'s V2 and V3 read the
//! **body**. A `.Replay.Gbx` — the container kind every map-carrying recording
//! in this project uses — keeps its own copies of the driver's login,
//! nickname, zone and account id in the header, plus **two more copies of the
//! race time**: a raw `u32` in chunk `0x03093000` and `<times best="...">` in
//! the XML of chunk `0x03093001`.
//!
//! Measured on this project's 173691 landing file *after* a successful
//! `identity set --anonymise` and a successful `declare --from-oracle`, with
//! `ghost verify` reporting `V2 declared-time census: 1 copies, all 36.049`
//! and no foreign identity:
//!
//! ```text
//! header chunk 0x03093000  nickname "GothMommyTM"  login "3Awx2_MzSdaCJZjZOht51A"
//! header chunk 0x03093000  time 49958
//! header chunk 0x03093001  <times best="49958" .../>
//! header chunk 0x03093002  login/nickname/zone, all his
//! ```
//!
//! A check that is precise and wrong is worse than no check: "1 copies, all
//! 36.049" is a *count*, and it was counting a set whose other members it
//! could not see.
//!
//! ## Legitimate versus foreign — the rule, stated
//!
//! There are two people in one of these files and the tool must not confuse
//! them:
//!
//! * **the DRIVER** — whose run this is. On a synthesised tape that is us, and
//!   every driver field must say so.
//! * **the MAP's AUTHOR** — whose map this is. On 173691 that is
//!   GothMommyTM, it stays GothMommyTM, and laundering it would be a
//!   misattribution in the other direction.
//!
//! They are told apart **structurally, by position in the chunk** — never by
//! value, because here they are the same person and the same 22 bytes:
//!
//! | occurrence | whose | what happens |
//! |---|---|---|
//! | `0x03093000` meta triple (uid, environment, author) | the map's | untouched |
//! | `0x03093000` nickname and login, after the time word | the driver's | anonymised |
//! | `0x03093000` race-time `u32` | this run's claim | rewritten |
//! | `0x03093001` XML `<map … author= authorzone=/>` | the map's | untouched |
//! | `0x03093001` XML `<times best=…/>` | this run's claim | rewritten |
//! | `0x03093002` login, nickname, zone | the driver's | anonymised |
//! | inside the embedded map's bytes in the body | the map's | untouched |
//!
//! ## Chunk `0x03093000`, as measured
//!
//! ```text
//! u32  version                       (8 here)
//! u32  lookback version              (3)
//! meta uid          lookback string  <- the map's
//! meta environment  lookback index   (26 = Stadium)
//! meta author       lookback string  <- the MAP AUTHOR's login
//! u32  race time                     <- a copy of the declared time
//! str  nickname                      <- the DRIVER
//! str  login                         <- the DRIVER
//! u8
//! lookback title id                  ("TMStadium")
//! ```
//!
//! A lookback string is a `u32`: `0x40000000` means "a string follows", a small
//! value is a predefined collection id. The first lookback in a chunk is
//! preceded once by the lookback version word.

use gbx::header::{build_user_data, parse_user_data, string_frames, HeaderChunk};
use gbx::Container;

/// Where a string lives inside a chunk, and what it says.
#[derive(Clone, Debug)]
pub struct At {
    /// Offset of the LENGTH word inside the chunk's data.
    pub off: usize,
    pub text: String,
}

/// The driver-and-times fields of replay header chunk `0x03093000`.
#[derive(Clone, Debug)]
pub struct Chunk3000 {
    pub uid: At,
    pub map_author: At,
    pub time_off: usize,
    pub time_ms: u32,
    pub nickname: At,
    pub login: At,
}

struct R<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> R<'a> {
    fn u32(&mut self) -> Option<u32> {
        let v = u32::from_le_bytes(self.b.get(self.o..self.o + 4)?.try_into().ok()?);
        self.o += 4;
        Some(v)
    }
    fn string(&mut self) -> Option<At> {
        let off = self.o;
        let n = self.u32()? as usize;
        if n > 1 << 20 || self.o + n > self.b.len() {
            return None;
        }
        let text = std::str::from_utf8(&self.b[self.o..self.o + n]).ok()?.to_string();
        self.o += n;
        Some(At { off, text })
    }
    /// A lookback string: `0x40000000` then a string, or a predefined index.
    fn lookback(&mut self) -> Option<Option<At>> {
        let idx = self.u32()?;
        if idx == 0x4000_0000 {
            Some(Some(self.string()?))
        } else {
            Some(None)
        }
    }
}

/// Parse chunk `0x03093000`. `None` if it is not the shape above — never a
/// partial answer, because a partial answer here is a wrong offset and a wrong
/// offset is a corrupted file.
pub fn parse_3000(data: &[u8]) -> Option<Chunk3000> {
    let mut r = R { b: data, o: 0 };
    let _version = r.u32()?;
    let _lookback_version = r.u32()?;
    let uid = r.lookback()??;
    let _env = r.lookback()?;
    let map_author = r.lookback()??;
    let time_off = r.o;
    let time_ms = r.u32()?;
    let nickname = r.string()?;
    let login = r.string()?;
    Some(Chunk3000 { uid, map_author, time_off, time_ms, nickname, login })
}

/// The driver block of chunk `0x03093002`: `u32 version, u32 authorVersion,
/// str login, str nickname, str zone, str extra`.
#[derive(Clone, Debug)]
pub struct Chunk3002 {
    pub login: At,
    pub nickname: At,
    pub zone: At,
}

pub fn parse_3002(data: &[u8]) -> Option<Chunk3002> {
    let mut r = R { b: data, o: 0 };
    let _version = r.u32()?;
    let _author_version = r.u32()?;
    let login = r.string()?;
    let nickname = r.string()?;
    let zone = r.string()?;
    Some(Chunk3002 { login, nickname, zone })
}

fn attr(xml: &str, name: &str) -> Option<String> {
    let pat = format!("{}=\"", name);
    let i = xml.find(&pat)? + pat.len();
    let j = xml[i..].find('"')? + i;
    Some(xml[i..j].to_string())
}

/// The `<map …/>` element's byte range, so its attribution is skipped by
/// POSITION rather than by value.
fn map_element(xml: &str) -> Option<(usize, usize)> {
    let i = xml.find("<map ")?;
    let j = xml[i..].find("/>").map(|k| i + k + 2)?;
    Some((i, j))
}

/// Rewrite `best="…"` outside the `<map/>` element. Everything else, including
/// every attribute that belongs to the map, is copied through.
fn rewrite_xml_best(xml: &str, ms: u32) -> String {
    let (mi, mj) = map_element(xml).unwrap_or((usize::MAX, usize::MAX));
    let pat = "best=\"";
    let mut out = String::with_capacity(xml.len() + 8);
    let mut cur = 0usize;
    while let Some(rel) = xml[cur..].find(pat) {
        let i = cur + rel;
        let vs = i + pat.len();
        let Some(rel2) = xml[vs..].find('"') else { break };
        let ve = vs + rel2;
        out.push_str(&xml[cur..vs]);
        if i >= mi && i < mj {
            out.push_str(&xml[vs..ve]); // the map's, not ours
        } else {
            out.push_str(&ms.to_string());
        }
        cur = ve;
    }
    out.push_str(&xml[cur..]);
    out
}

/// The XML chunk's text, if the header has one.
pub fn xml_of(c: &Container) -> Option<String> {
    let chunks = parse_user_data(&c.gbx.user_data)?;
    let ch = chunks.iter().find(|c| c.id == 0x0309_3001)?;
    string_frames(&ch.data).into_iter().find(|f| f.text.starts_with("<header")).map(|f| f.text)
}

/// Every copy of the race time the HEADER holds, with where it lives.
///
/// This is the census `verify` V2 could not take. It returns an empty vector
/// for a container with no replay header — which is *why* the plain
/// `.Ghost.Gbx` files in this repo never carried the defect, and is a fact
/// about the container kind, not a pass.
pub fn header_declared_ms(c: &Container) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    let Some(chunks) = parse_user_data(&c.gbx.user_data) else { return out };
    for ch in &chunks {
        if ch.id == 0x0309_3000 {
            if let Some(p) = parse_3000(&ch.data) {
                out.push(("header chunk 0x03093000 u32".to_string(), p.time_ms));
            }
        }
    }
    if let Some(x) = xml_of(c) {
        if let Some(v) = attr(&x, "best").and_then(|v| v.parse::<u32>().ok()) {
            out.push(("header XML times best=".to_string(), v));
        }
    }
    out
}

/// Every identity string the HEADER holds that belongs to the DRIVER, i.e.
/// that must be ours on a file we publish as ours. The map's own attribution
/// is deliberately not in this list.
pub fn header_driver_identity(c: &Container) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(chunks) = parse_user_data(&c.gbx.user_data) else { return out };
    for ch in &chunks {
        match ch.id {
            0x0309_3000 => {
                if let Some(p) = parse_3000(&ch.data) {
                    out.push(("0x03093000 nickname".into(), p.nickname.text));
                    out.push(("0x03093000 login".into(), p.login.text));
                }
            }
            0x0309_3002 => {
                if let Some(p) = parse_3002(&ch.data) {
                    out.push(("0x03093002 login".into(), p.login.text));
                    out.push(("0x03093002 nickname".into(), p.nickname.text));
                    // The zone is deliberately NOT in this list, and not
                    // anonymised. It is a country, not an identifier, and this
                    // toolchain has never stripped it (`identity set` touches
                    // it only on an explicit `--zone`). It is also the ANCHOR
                    // the body scan uses to find the trigram and the club tag
                    // -- clearing it made `ghost identity set --trigram` pass
                    // its write and fail its own read-back, which the suite
                    // caught. A field that other checks navigate by is not a
                    // field to blank on a hunch.
                }
            }
            _ => {}
        }
    }
    out
}

/// File-offset ranges that are the MAP's own attribution and are therefore
/// legitimate on a file we publish as ours.
///
/// This is the allowlist the raw-bytes backstop needs. It is computed
/// STRUCTURALLY — the meta-triple author in header chunk `0x03093000`, and the
/// `author=` / `authorzone=` attribute values in the header XML — so that a
/// driver login which happens to be the same 22 bytes as the map author's is
/// still caught everywhere it is the driver's. A value-based allowlist would
/// launder exactly the case this project keeps meeting.
///
/// `header_len` is the file offset at which the body begins.
pub fn legitimate_map_ranges(c: &Container, header_len: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let Some(chunks) = parse_user_data(&c.gbx.user_data) else { return out };
    // where user_data starts in the file
    let ud_off = match header_len
        .checked_sub(4 + c.gbx.ref_table.len() + c.gbx.user_data.len())
    {
        Some(v) => v,
        None => return out,
    };
    let table = 4 + 8 * chunks.len();
    let mut data_off = ud_off + table;
    for ch in &chunks {
        if ch.id == 0x0309_3000 {
            if let Some(p) = parse_3000(&ch.data) {
                let a = data_off + p.map_author.off;
                out.push((a, a + 4 + p.map_author.text.len()));
            }
        }
        if ch.id == 0x0309_3001 {
            if let Some(f) =
                string_frames(&ch.data).into_iter().find(|f| f.text.starts_with("<header"))
            {
                let xml_at = data_off + f.off + 4;
                for k in ["author", "authorzone"] {
                    let pat = format!("{}=\"", k);
                    if let Some(i) = f.text.find(&pat) {
                        let vs = i + pat.len();
                        if let Some(rel) = f.text[vs..].find('"') {
                            out.push((xml_at + vs, xml_at + vs + rel));
                        }
                    }
                }
            }
        }
        data_off += ch.data.len();
    }
    out
}

/// What a header rewrite did.
pub struct HeaderEdit {
    pub user_data: Vec<u8>,
    pub log: Vec<String>,
}

fn put_string(data: &mut Vec<u8>, at: &At, new: &str) {
    let end = at.off + 4 + at.text.len();
    let mut repl = (new.len() as u32).to_le_bytes().to_vec();
    repl.extend_from_slice(new.as_bytes());
    data.splice(at.off..end, repl);
}

/// Rewrite the header's DRIVER identity and this run's own declared time.
///
/// Returns `None` when the container has no replay header chunk table — a
/// plain `.Ghost.Gbx`, which has none, and for which this is a no-op rather
/// than a failure.
///
/// Edits are applied back-to-front within a chunk so that every offset stays
/// valid while lengths change.
pub fn rewrite(
    c: &Container,
    anonymise: bool,
    name: Option<&str>,
    best_ms: Option<u32>,
) -> Option<HeaderEdit> {
    let chunks = parse_user_data(&c.gbx.user_data)?;
    let who = name.unwrap_or("TAS");
    let mut log = Vec::new();
    let mut out: Vec<HeaderChunk> = Vec::new();
    for ch in &chunks {
        let mut data = ch.data.clone();
        match ch.id {
            0x0309_3000 => {
                if let Some(p) = parse_3000(&data) {
                    // back to front: login, nickname, then the fixed-width time
                    if anonymise && p.login.text != who {
                        log.push(format!("  header 0x03093000 login    {:?} -> {:?}", p.login.text, who));
                        put_string(&mut data, &p.login, who);
                    }
                    if anonymise && p.nickname.text != who {
                        log.push(format!("  header 0x03093000 nickname {:?} -> {:?}", p.nickname.text, who));
                        put_string(&mut data, &p.nickname, who);
                    }
                    if let Some(ms) = best_ms {
                        if p.time_ms != ms {
                            log.push(format!(
                                "  header 0x03093000 time     {} -> {}",
                                gbx::container::secs(p.time_ms as i64),
                                gbx::container::secs(ms as i64)
                            ));
                            data[p.time_off..p.time_off + 4].copy_from_slice(&ms.to_le_bytes());
                        }
                    }
                    log.push(format!(
                        "  header 0x03093000 map author {:?} LEFT ALONE (it is the map's, not the driver's)",
                        p.map_author.text
                    ));
                }
            }
            0x0309_3001 => {
                if let Some(f) =
                    string_frames(&data).into_iter().find(|f| f.text.starts_with("<header"))
                {
                    let mut new = f.text.clone();
                    if let Some(ms) = best_ms {
                        new = rewrite_xml_best(&new, ms);
                    }
                    if new != f.text {
                        let old_best = attr(&f.text, "best").unwrap_or_default();
                        let new_best = attr(&new, "best").unwrap_or_default();
                        log.push(format!(
                            "  header XML       times best=\"{}\" -> \"{}\"",
                            old_best, new_best
                        ));
                        put_string(&mut data, &At { off: f.off, text: f.text.clone() }, &new);
                    }
                }
            }
            0x0309_3002 => {
                if anonymise {
                    if let Some(p) = parse_3002(&data) {
                        // back to front
                        if p.nickname.text != who {
                            log.push(format!("  header 0x03093002 nickname {:?} -> {:?}", p.nickname.text, who));
                            put_string(&mut data, &p.nickname, who);
                        }
                        if p.login.text != who {
                            log.push(format!("  header 0x03093002 login    {:?} -> {:?}", p.login.text, who));
                            put_string(&mut data, &p.login, who);
                        }
                    }
                }
            }
            _ => {}
        }
        out.push(HeaderChunk { id: ch.id, heavy: ch.heavy, data });
    }
    Some(HeaderEdit { user_data: build_user_data(&out), log })
}

/// `ghost header show FILE` — the chunk table, the driver fields, the map's
/// own attribution, and every copy of the race time the header holds.
pub fn cmd_show(path: &str) {
    let c = Container::load(path).unwrap_or_else(|e| crate::cli::die(e));
    let Some(chunks) = parse_user_data(&c.gbx.user_data) else {
        println!(
            "{}: NO replay header chunk table ({} B of user data).\n  A plain .Ghost.Gbx has no \
             replay header at all, so it cannot carry a header identity or a header copy of the \
             declared time. This is a fact about the container kind, not a pass.",
            path,
            c.gbx.user_data.len()
        );
        return;
    };
    println!("{}: {} header chunks, {} B of user data", path, chunks.len(), c.gbx.user_data.len());
    for ch in &chunks {
        println!("  0x{:08X}  {:>5} B{}", ch.id, ch.data.len(), if ch.heavy { "  heavy" } else { "" });
    }
    for ch in &chunks {
        if ch.id == 0x0309_3000 {
            if let Some(p) = parse_3000(&ch.data) {
                println!("\nchunk 0x03093000");
                println!("  map uid              {:?}", p.uid.text);
                println!("  MAP AUTHOR           {:?}   (the map's -- left alone)", p.map_author.text);
                println!("  race time            {}", gbx::container::secs(p.time_ms as i64));
                println!("  DRIVER nickname      {:?}", p.nickname.text);
                println!("  DRIVER login         {:?}", p.login.text);
            }
        }
        if ch.id == 0x0309_3002 {
            if let Some(p) = parse_3002(&ch.data) {
                println!("\nchunk 0x03093002 (the driver)");
                println!("  login                {:?}", p.login.text);
                println!("  nickname             {:?}", p.nickname.text);
                println!("  zone                 {:?}", p.zone.text);
            }
        }
    }
    if let Some(x) = xml_of(&c) {
        println!("\nheader XML");
        for k in ["author", "authorzone", "best", "uid", "name"] {
            if let Some(v) = attr(&x, k) {
                let whose = match k {
                    "author" | "authorzone" | "uid" | "name" => "the map's -- left alone",
                    _ => "this run's claim",
                };
                println!("  {:<11} {:?}   ({})", k, v, whose);
            }
        }
    }
    let times = header_declared_ms(&c);
    if !times.is_empty() {
        println!("\nrace-time copies IN THE HEADER (invisible to the body census)");
        for (where_, ms) in times {
            println!("  {:<32} {}", where_, gbx::container::secs(ms as i64));
        }
    }
}
