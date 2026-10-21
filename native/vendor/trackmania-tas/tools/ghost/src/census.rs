//! `ghost census` -- WHICH RUN'S TIME IS STORED IN THIS FILE, AT HOW MANY
//! SITES, AND WHERE.
//!
//! The standing question about a synthesised tape is "N copies of its own
//! validated time and ZERO of any other time, N read from the file" -- and the
//! reason it has to be *read* is that N is not a constant: there are six
//! declared-time sites on 173636 and five on 199100. A check that knows the
//! number in advance is comparing a command-line operand to itself.
//!
//! WHAT THIS IS NOT. It is not an identity scan: `ghost verify`'s V10 is the
//! raw-bytes backstop for logins, account ids and locator URLs, and
//! `ghost header show` reads the replay header. Duplicating either of those
//! here would put a second reader of the same bytes in the tree, which is the
//! disease this crate exists to cure. What V10 does NOT do is enumerate the
//! millisecond values stored as BINARY in the body: it only reads the header's
//! `best="..."`. That gap is this command.
//!
//! TWO RESTRICTIONS ON THE SCAN, both about not manufacturing evidence.
//!
//! ALIGNED WORDS ONLY. An unaligned walk of an N-byte file offers N-3 windows
//! instead of N/4, and most of the extra ones are two halves of adjacent
//! floats: it quadruples every count and invents coincidences faster than it
//! finds sites.
//!
//! NOT THE OPAQUE BLOBS. The zlib telemetry record and a carried map are
//! compressed or dense binary. A 780 kB map contains any particular 4-byte
//! value about 190 000 / 2^32 of the time per site, and across a few hundred
//! candidate times that is a steady drizzle of hits that mean nothing. What
//! this walks is where a TIME IS STORED AS A TIME. The regions it skipped are
//! printed, because a census whose denominator is not stated is not a census.

use gbx::container::Container;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Region {
    HeaderUserData,
    Body,
}

impl Region {
    pub fn label(self) -> &'static str {
        match self {
            Region::HeaderUserData => "header/userdata",
            Region::Body => "body",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TimeSite {
    pub region: Region,
    pub at: usize,
    pub ms: u32,
    /// A little-endian u32, or text inside the header XML.
    pub how: &'static str,
}

pub struct Report {
    pub file: String,
    pub bytes: usize,
    pub has_header_xml: bool,
    /// Regions the census did NOT walk, and why.
    pub skipped: Vec<String>,
    pub times: Vec<TimeSite>,
}

pub fn scan(path: &str) -> Result<Report, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let c = Container::load(path)?;
    let body = c.body();
    let hdr_len = raw.len().saturating_sub(body.len());
    let ud = &raw[..hdr_len];

    let mut times: Vec<TimeSite> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let plausible = |v: u32| (500..=20 * 60 * 1000).contains(&v);
    let mut scan_u32 = |b: &[u8], base: usize, region: Region, out: &mut Vec<TimeSite>| {
        let mut i = 0usize;
        while i + 4 <= b.len() {
            let v = u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
            if plausible(v) {
                out.push(TimeSite { region, at: base + i, ms: v, how: "u32" });
            }
            i += 4;
        }
    };
    scan_u32(ud, 0, Region::HeaderUserData, &mut times);

    let mut holes: Vec<(usize, usize)> = Vec::new();
    if let Some((a, b)) = c.embedded_map() {
        holes.push((a, b));
        skipped.push(format!("the carried map, {} B at body@{}", b - a, a));
    }
    if let Ok(s) = gbx::recwrite::find_rec_site(body) {
        let end = (s.hdr + 64 + s.csize).min(body.len());
        holes.push((s.hdr, end));
        skipped.push(format!("the zlib telemetry record, {} B at body@{}", end - s.hdr, s.hdr));
    }
    holes.sort();
    let mut cur = 0usize;
    for (a, b) in &holes {
        if *a > cur {
            scan_u32(&body[cur..*a], cur, Region::Body, &mut times);
        }
        cur = (*b).max(cur);
    }
    if cur < body.len() {
        scan_u32(&body[cur..], cur, Region::Body, &mut times);
    }

    // The header's own TEXT copy of the declared time, which no chunk walk can
    // reach. Read through `hdr`, the one reader of that block.
    let has_header_xml = crate::hdr::xml_of(&c).is_some();
    for (site, ms) in crate::hdr::header_declared_ms(&c) {
        times.push(TimeSite { region: Region::HeaderUserData, at: 0, ms, how: Box::leak(site.into_boxed_str()) });
    }

    Ok(Report { file: path.to_string(), bytes: raw.len(), has_header_xml, skipped, times })
}

pub fn render(r: &Report, own_ms: Option<u32>, others: &[u32]) -> String {
    let mut s = String::new();
    s.push_str(&format!("{}  ({} B)\n", r.file, r.bytes));
    s.push_str(&format!(
        "  header       {}\n",
        if r.has_header_xml {
            "XML USER-DATA PRESENT -- a map-carrying container; ghost verify V10 covers it"
        } else {
            "none (a plain .Ghost.Gbx has no user data, so it cannot carry the header defect)"
        }
    ));
    for x in &r.skipped {
        s.push_str(&format!("  census skips {}\n", x));
    }
    let mut tally: std::collections::BTreeMap<u32, Vec<&TimeSite>> = Default::default();
    for t in &r.times {
        tally.entry(t.ms).or_default().push(t);
    }
    let sites = |v: &Vec<&TimeSite>| -> String {
        v.iter()
            .map(|t| {
                if t.how == "u32" {
                    format!("{}@{}", t.region.label(), t.at)
                } else {
                    format!("{} ({})", t.region.label(), t.how)
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    match own_ms {
        None => s.push_str("  own          no --expect-ms given: the census is not judged\n"),
        Some(m) => {
            let v = tally.get(&m);
            s.push_str(&format!(
                "  own          {} aligned site(s) hold this run's own {}{}\n",
                v.map_or(0, |x| x.len()),
                gbx::container::secs(m as i64),
                v.map(|x| format!("  --  {}", sites(x))).unwrap_or_default()
            ));
        }
    }
    if others.is_empty() {
        s.push_str(
            "  foreign      no --other given: NOT CHECKED for another run's time, which is not\n\
             \x20              the same as clean. Pass every human record and sibling time of this map.\n",
        );
    } else {
        let mut hits = 0usize;
        for o in others {
            if own_ms == Some(*o) {
                continue;
            }
            if let Some(v) = tally.get(o) {
                hits += 1;
                s.push_str(&format!(
                    "  FOREIGN      {} appears {} time(s)  --  {}\n",
                    gbx::container::secs(*o as i64),
                    v.len(),
                    sites(v)
                ));
            }
        }
        if hits == 0 {
            s.push_str(&format!(
                "  foreign      none of the {} other time(s) named for this map appears anywhere\n",
                others.len()
            ));
        }
    }
    s
}

pub fn cmd(a: &[String]) {
    // A FLAG'S VALUE IS NOT A FILENAME. Taking positionals as "anything not
    // starting with --" swallowed `--expect-ms 7241`'s 7241 and printed
    // `7241: No such file or directory` under every good report.
    let mut files: Vec<String> = Vec::new();
    let mut own: Option<u32> = None;
    let mut others: Vec<u32> = Vec::new();
    let mut i = 0usize;
    while i < a.len() {
        match a[i].as_str() {
            "--expect-ms" => {
                own = a.get(i + 1).and_then(|v| v.parse().ok());
                i += 2;
            }
            "--other" => {
                if let Some(v) = a.get(i + 1) {
                    others.extend(v.split(',').filter_map(|x| x.trim().parse::<u32>().ok()));
                }
                i += 2;
            }
            x if x.starts_with("--") => crate::cli::die(format!("unknown flag {:?}", x)),
            x => {
                files.push(x.to_string());
                i += 1;
            }
        }
    }
    if files.is_empty() {
        crate::cli::die(
            "usage: ghost census FILE... [--expect-ms MS] [--other MS,MS,...]\n\n\
             Every millisecond value stored in the file as a TIME -- aligned words in the\n\
             body outside the record and the carried map, plus the header's own text copy.\n\
             The count comes out of the file: N is six on one map and five on another, so a\n\
             check that knows it in advance is comparing a command-line operand to itself.\n\n\
             \x20 --expect-ms MS   this run's own validated time\n\
             \x20 --other MS,...   times that must NOT appear -- every human record and sibling\n\
             \x20                  of this map. Without it the file is UNCHECKED, not clean.\n\n\
             For identity (logins, account ids, locator URLs) use `ghost verify` V10, and for\n\
             the replay header `ghost header show`. This command deliberately does neither.",
        );
    }
    for f in &files {
        match scan(f) {
            Ok(r) => print!("{}", render(&r, own, &others)),
            Err(e) => println!("{}  UNMEASURED: {}", f, e),
        }
    }
}
