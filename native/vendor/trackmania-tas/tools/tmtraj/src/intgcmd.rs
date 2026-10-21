//! `tmtraj intg` -- GHOST INTEGRITY: is this file's telemetry its own?
//!
//! THE TEST
//! --------
//! A shared bit-identical PREFIX between two runs proves nothing. The
//! simulation is deterministic, so two tapes with the same opening inputs give
//! identical f32 positions for as long as their inputs agree, and our own
//! sibling tapes do this routinely (67 % of samples on one 203072 pair).
//!
//! The proof is RE-CONVERGENCE. Once two runs are metres apart they are
//! different physical states, and no input sequence returns them to EXACTLY
//! 0.000000 m -- the same f32 bits in all three components. So the sequence
//!
//! ```text
//! bit-identical  ->  diverge past a threshold  ->  bit-identical again
//! ```
//!
//! cannot be produced by driving. It is a splice: two runs' samples in one
//! record. That is what a partially-regenerated ghost is.
//!
//! WHAT THE VERDICT IS WORTH
//! -------------------------
//! CONTAMINATED is a positive finding and it is reference-free in the sense
//! that it needs no model -- but WHICH reference matters. Against a HUMAN
//! recording the finding is clean. Against one of our own siblings it is not,
//! because two of our own files can re-converge for the honest reason that they
//! share the contaminating pipeline: both inherited the same donor's samples in
//! the same places. `--kind` labels the reference so the audit can keep the two
//! populations apart, and this command never merges them.
//!
//! CLEAN here means "no splice against THIS reference". It is not a
//! certificate; the NO-REFERENCE case is reported as its own verdict and must
//! never be collapsed into CLEAN.


use gbx::record;
use crate::whlcmd::{decode, R};

/// One maximal run of bit-identical samples.
#[derive(Clone, Debug)]
pub struct ZeroRun {
    pub i0: usize,
    pub i1: usize, // inclusive
    pub ms0: i64,
}

impl ZeroRun {
    pub fn n(&self) -> usize {
        self.i1 - self.i0 + 1
    }
}

pub struct Pair {
    /// samples present in both records, by time
    pub n: usize,
    pub d: Vec<f64>,
    pub ident: Vec<bool>,
    pub runs: Vec<ZeroRun>,
    pub max_d: f64,
    /// biggest divergence strictly BETWEEN two bit-identical runs
    pub gap_d: f64,
}

/// Bit-identical position: the same twelve bytes, not "close".
fn pos_ident(a: &R, b: &R) -> bool {
    if a.raw.len() < 59 || b.raw.len() < 59 {
        return false;
    }
    a.raw[47..59] == b.raw[47..59]
}

fn dist(a: &R, b: &R) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let q = a.pos[k] - b.pos[k];
        s += q * q;
    }
    s.sqrt()
}

/// Align two records on their shared sample instants and profile the distance.
pub fn pair(a: &[R], b: &[R]) -> Pair {
    let bm: std::collections::HashMap<i64, &R> = b.iter().map(|r| (r.ms, r)).collect();
    let mut times = Vec::new();
    let mut d = Vec::new();
    let mut ident = Vec::new();
    for x in a {
        let Some(y) = bm.get(&x.ms) else { continue };
        times.push(x.ms);
        d.push(dist(x, y));
        ident.push(pos_ident(x, y));
    }
    let n = times.len();
    // maximal runs of bit-identical samples
    let mut runs: Vec<ZeroRun> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if !ident[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < n && ident[j + 1] {
            j += 1;
        }
        runs.push(ZeroRun { i0: i, i1: j, ms0: times[i] });
        i = j + 1;
    }
    let mut max_d = 0.0f64;
    for k in 0..n {
        if d[k].is_finite() && d[k] > max_d {
            max_d = d[k];
        }
    }
    // The divergence that precedes an identical block -- the load-bearing
    // number. Measured over everything before the FIRST non-head identical
    // run, so a file whose donor block starts at sample 1 is scored too. Non-finite counts as infinite separation: a NaN sample is not
    // "close" to anything, and 270051's files diverge that way.
    let mut gap_d = 0.0f64;
    if let Some(r) = runs.iter().find(|r| r.i0 > 0) {
        for k in 0..r.i0 {
            let v = if d[k].is_finite() { d[k] } else { f64::INFINITY };
            if v > gap_d {
                gap_d = v;
            }
        }
    }
    Pair {
        n,
        d,
        ident,
        runs,
        max_d,
        gap_d,
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Verdict {
    Contaminated,
    Clean,
    Identical,
    NoComparison,
}

impl Verdict {
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Contaminated => "CONTAMINATED",
            Verdict::Clean => "CLEAN",
            Verdict::Identical => "IDENTICAL-TO-REFERENCE",
            Verdict::NoComparison => "NO-COMPARISON",
        }
    }
}

/// `minsep` is how far apart two runs must get before returning to the same
/// f32 bits is impossible rather than merely unlikely. Default 5 m: the car is
/// 2 m long, the sample grid is 50 ms, and at racing speed one tick of
/// divergence is already metres. Every finding this project has made is two
/// orders of magnitude above it (147 m .. 2361 m).
pub fn verdict(p: &Pair, minsep: f64) -> Verdict {
    if p.n < 5 {
        return Verdict::NoComparison;
    }
    let all_ident = p.runs.len() == 1 && p.runs[0].n() == p.n;
    if all_ident {
        return Verdict::Identical;
    }
    // THE GENERAL RULE, and the narrower re-convergence rule is one case of it.
    //
    // To be bit-identical at instant t, two runs must hold the same physical
    // state at t. Determinism can give that from the START -- same tape prefix,
    // same f32s -- so a HEAD-ANCHORED identical block is honest and common
    // (our tapes are seeded from human tapes). But once the two runs are
    // metres apart they are different states, and no input sequence returns
    // them to the same f32 bits. So ANY identical block that is preceded by
    // real separation is a splice: donor samples sitting in our record.
    //
    // This was written as "identical -> apart -> identical", which requires TWO
    // identical runs and MISSES the commonest real case: a file whose very
    // first sample differs (a wrong origin, a clock offset, one repaired
    // sample) and whose remaining 364 of 365 samples are the donor's. Measured
    // on 227654's TAS_57503 -- the single most contaminated file in the corpus
    // -- which the two-run spelling passed as CLEAN.
    for r in &p.runs {
        if r.i0 == 0 {
            continue; // head-anchored: determinism, not a splice
        }
        let before = (0..r.i0)
            .map(|k| if p.d[k].is_finite() { p.d[k] } else { f64::INFINITY })
            .fold(0.0f64, f64::max);
        if before >= minsep {
            return Verdict::Contaminated;
        }
    }
    Verdict::Clean
}



/// `tmtraj gate` — the publish gate.
///
/// Exit 0 publishable, 2 refused, 3 UNMEASURED — and **3 is never folded into
/// 0**: an input the gate could not read is not a verdict about the ghost.
///
/// WHAT WAS DELETED FROM HERE, AND WHY. This module used to expose eighteen
/// subcommands. Fourteen were the manual faces of an investigation — `pair`,
/// `sweep`, `lag`, `stale`, `c3`, `c12`, `echo`, `c11b`, `selfsim`, `qrule`,
/// `poison`, `corrupt`, `tapecsv`, `md5`. The ENGINES behind the load-bearing
/// ones are still here and the gate calls them; the CLIs are gone, because a
/// command that exists only so a person can eyeball one number during one
/// night's work is not an API.
///
/// Three of them were worse than unused:
///
/// * **`c12` was a correct check wired to nothing.** It plugs B-contam's
///   documented blind spot — the near-copy that is never byte-equal — and no
///   pipeline ran it. It is now a gate check.
/// * **`c3` was the corrected teleport test, and the gate ran the old one.**
///   Its speedometer rule is now C3 in `tmtraj check`, which is what the gate
///   shells out to.
/// * **`c11b`'s CLI could only ever print NO-VERDICT**, because it called
///   `c11b_verdict` with a hard-coded `control = None`, so the
///   MATCHES-THE-GAME and STALE-BUFFER arms were unreachable from the command
///   line. Its lesson survives in C-route, which scans the lag: *a magnitude
///   cannot see which side of a tick a file is on.* 227654's record reads
///   0.5485 m at lag 0 and 0.0000 m at lag −1, because 0.5485 m is exactly how
///   far that car travels in one 10 ms tick, and the first version of the check
///   convicted an honest file inside an hour.
pub fn cmd(args: &[String]) {
    // Asking for help is exit 0; forgetting the arguments is exit 2. The two
    // were collapsed into one branch, so `--help` looked like a usage error.
    if args[0..].iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        std::process::exit(0);
    }
    if args.is_empty() {
        print_usage();
        std::process::exit(2);
    }
    cmd_gate(args);
}

fn print_usage() {
    print!(
        "\
usage: tmtraj gate GHOST... --race S --refs refs.tsv --mapid ID
                   [--map M.Map.Gbx --server DIR] [--source SECOND_GENERATION]
                   [--route route.csv | --route-dir D] [--manifest F]
                   [--require-manifest] [--minsep M]

The publish gate. Exit 0 publishable, 2 REFUSED, 3 UNMEASURED -- and 3 is never
folded into 0: an input the gate could not read is not a verdict about the file.

  A  C1-C10    is this a physically coherent run of a car   (via `tmtraj check`)
  B  B-contam  bit-exact against every human recording held for the map
     C12       and, where the tapes differ, the trajectories must part
  C  C-oracle  does the dedicated server re-simulate THE WRITTEN BYTES
     C-header  does the file declare the time it actually does
     C-ident   our login, no account id
  F  C-spawn   the first in-race sample at the map's spawn, FACING the way
               every run on it faces
  G  C-route   the record against the engine's own trajectory, read by an
               instrument that never touches the record
  E  E-stale   is this a physics tick behind a second independent generation
  D  D-manifest  does the file's own account of how it was made hold up

A shared PREFIX proves nothing -- the simulation is deterministic. Only
RE-CONVERGENCE to exactly 0.000000 m after a real separation cannot be driven.
"
    );
}



// ---------------------------------------------------------------------------
// md5, so a row can carry the identity of the bytes it describes. No crate: the
// workspace is built --offline against a vendored set that has none.
// ---------------------------------------------------------------------------

pub fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let mut k = [0u32; 64];
    for (i, v) in k.iter_mut().enumerate() {
        *v = ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32;
    }
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_le_bytes());
    for chunk in msg.chunks(64) {
        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f2 = f
                .wrapping_add(a)
                .wrapping_add(k[i])
                .wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f2.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = String::new();
    for w in [a0, b0, c0, d0] {
        for byte in w.to_le_bytes() {
            out.push_str(&format!("{:02x}", byte));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// `intg audit` -- the whole published corpus, against a reference table.
// ---------------------------------------------------------------------------

/// A reference recording, and what it is. The distinction is load-bearing: a
/// HUMAN recording was made by the game client on a machine of ours and shares
/// nothing with our pipeline, so re-convergence against it is a finding. A
/// SIBLING is one of our own files, which can re-converge for the honest reason
/// that both inherited the same donor's samples -- so a sibling finding is
/// reported, never merged, and never on its own called contamination.
pub struct Ref {
    pub map: String,
    pub kind: String,
    pub path: String,
}

pub fn load_refs(path: &str) -> Result<Vec<Ref>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut out = Vec::new();
    for l in s.lines() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() < 3 {
            return Err(format!("bad reference line: {:?}", l));
        }
        out.push(Ref { map: f[0].into(), kind: f[1].into(), path: f[2].into() });
    }
    Ok(out)
}

/// Does the ghost's own manifest list the ground-contact byte among the fields
/// INHERITED from the carrier?
///
/// Scoped to the `fields_inherited` array on purpose: a bare substring search
/// over the whole manifest would also match the word in an `engine_note` or a
/// tool argument, and an exemption granted by prose is not an exemption.
fn manifest_says_contact_inherited(ghost: &str, override_: Option<&str>) -> bool {
    let p = override_
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}.manifest.json", ghost));
    let Ok(s) = std::fs::read_to_string(p) else { return false };
    let Some(i) = s.find("\"fields_inherited\"") else { return false };
    let Some(j) = s[i..].find(']') else { return false };
    let span = s[i..i + j].to_ascii_lowercase();
    span.contains("contact")
}

fn map_of(p: &std::path::Path) -> String {
    // <corpus>/<mapdir>/replays/<file>
    p.parent()
        .and_then(|d| d.parent())
        .and_then(|d| d.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// The strongest evidence against one file: over all references of a kind, the
/// pair with the largest separation BETWEEN two bit-identical runs.
struct Worst {
    v: Verdict,
    refname: String,
    n_common: usize,
    n_ident: usize,
    n_runs: usize,
    head_n: usize,
    tail_n: usize,
    tail_ms: i64,
    gap: f64,
    maxd: f64,
    minsep_seen: f64,
    /// COVERAGE. A verdict is a claim about the references it was actually
    /// compared against, and those three numbers are almost never equal:
    /// references are HELD, some fail to decode, and some share no instant
    /// with the file. "CLEAN" against 1 of 4 held references and "CLEAN"
    /// against 4 of 4 print identically unless the denominators travel with
    /// the row, and tonight has cost hours to claims of the first shape read
    /// as the second.
    n_refs_held: usize,
    n_refs_decoded: usize,
    n_refs_compared: usize,
}

/// Decoded references, kept between files. Decoding 186935's 48 835-sample
/// reference once per audited file is most of an eight-minute run; once per
/// audit it is nothing.
type Cache = std::collections::HashMap<String, Option<std::rc::Rc<Vec<R>>>>;

fn cached<'c>(cache: &'c mut Cache, path: &str) -> Option<std::rc::Rc<Vec<R>>> {
    if !cache.contains_key(path) {
        let rm = declared_race_ms(path);
        let v = decode(path).ok().map(|r| std::rc::Rc::new(in_race(&r, rm)));
        cache.insert(path.to_string(), v);
    }
    cache.get(path).cloned().flatten()
}

fn worst_over(a: &[R], refs: &[&Ref], minsep: f64, cache: &mut Cache) -> Option<Worst> {
    let rank = |v: Verdict| match v {
        Verdict::Contaminated => 3,
        Verdict::Identical => 2,
        Verdict::Clean => 1,
        Verdict::NoComparison => 0,
    };
    let mut best: Option<Worst> = None;
    let n_held = refs.len();
    let mut n_decoded = 0usize;
    let mut n_compared = 0usize;
    for r in refs {
        let Some(b) = cached(cache, &r.path) else { continue };
        n_decoded += 1;
        let p = pair(a, &b);
        if p.n > 0 {
            n_compared += 1;
        }
        let v = verdict(&p, minsep);
        let nid = p.ident.iter().filter(|x| **x).count();
        let head = p.runs.first().filter(|x| x.i0 == 0).map_or(0, |x| x.n());
        let (tail_n, tail_ms) = p
            .runs
            .last()
            .filter(|x| p.n > 0 && x.i1 == p.n - 1)
            .map_or((0, 0), |x| (x.n(), x.ms0));
        // rank by verdict first, then by how much evidence there is
        let cand = Worst {
            v,
            refname: std::path::Path::new(&r.path)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            n_common: p.n,
            n_ident: nid,
            n_runs: p.runs.len(),
            head_n: head,
            tail_n,
            tail_ms,
            gap: p.gap_d,
            maxd: p.max_d,
            minsep_seen: p.d.iter().cloned().fold(f64::INFINITY, f64::min),
            n_refs_held: n_held,
            n_refs_decoded: 0,
            n_refs_compared: 0,
        };
        let better = match &best {
            None => true,
            Some(b0) => {
                rank(cand.v) > rank(b0.v)
                    || (rank(cand.v) == rank(b0.v) && cand.n_ident > b0.n_ident)
            }
        };
        if better {
            best = Some(cand);
        }
    }
    // the counts belong to the SWEEP, not to whichever reference won
    if let Some(b) = best.as_mut() {
        b.n_refs_decoded = n_decoded;
        b.n_refs_compared = n_compared;
    }
    best
}

pub(crate) fn cmd_audit(args: &[String]) {
    let flag = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let corpus = flag("--corpus").expect("--corpus");
    let refs_p = flag("--refs").expect("--refs");
    let minsep: f64 = flag("--minsep").and_then(|v| v.parse().ok()).unwrap_or(5.0);
    let refs = load_refs(&refs_p).unwrap();
    let mut cache: Cache = Cache::new();

    // every ghost the corpus publishes
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let root = std::path::Path::new(&corpus);
    let mut dirs: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
    while let Some(d) = dirs.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().map_or(false, |n| n == ".git") {
                    continue;
                }
                dirs.push(p);
            } else if p.to_string_lossy().ends_with(".Ghost.Gbx") {
                files.push(p);
            }
        }
    }
    files.sort();

    println!(
        "map\tfile\tmd5\tn_samples\tspan_s\thuman_ref\tn_common\tn_ident\tn_runs\thead_n\ttail_n\t\
         tail_start_s\tgap_sep_m\tmax_sep_m\tmin_sep_m\tverdict\tsibling_ref\tsibling_verdict\t\
         sibling_gap_m\tsibling_ident\trefs_held\trefs_decoded\trefs_compared"
    );
    // COVERAGE. Denominators for every claim this sweep will make.
    let (mut n_seen, mut n_decode_fail, mut n_no_ref, mut n_no_common) = (0usize, 0, 0, 0);
    let mut hist: std::collections::BTreeMap<String, usize> = Default::default();
    let mut one_ref_only = 0usize;
    for f in &files {
        n_seen += 1;
        let fp = f.to_string_lossy().to_string();
        let map = map_of(f);
        let bytes = std::fs::read(f).unwrap_or_default();
        let md5 = md5_hex(&bytes);
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let race_ms = declared_race_ms(&fp);
        let a = match decode(&fp).map(|v| in_race(&v, race_ms)) {
            Ok(v) => v,
            Err(e) => {
                n_decode_fail += 1;
                println!(
                    "{}\t{}\t{}\t0\t-\t-\t0\t0\t0\t0\t0\t-\t-\t-\t-\tDECODE-FAIL ({})\t-\t-\t-\t-\t{}\t0\t0",
                    map, name, md5, e, 0
                );
                continue;
            }
        };
        let span = if a.is_empty() {
            0.0
        } else {
            (a[a.len() - 1].ms - a[0].ms) as f64 / 1000.0
        };
        // map id is the leading digits of the directory name
        let mapid: String = map.chars().take_while(|c| c.is_ascii_digit()).collect();
        let humans: Vec<&Ref> =
            refs.iter().filter(|r| r.map == mapid && r.kind == "human").collect();
        // siblings: every OTHER published file of the same map
        let sibs: Vec<Ref> = files
            .iter()
            .filter(|g| map_of(g) == map && g.to_string_lossy() != fp)
            .map(|g| Ref {
                map: mapid.clone(),
                kind: "sibling".into(),
                path: g.to_string_lossy().to_string(),
            })
            .collect();
        let sib_refs: Vec<&Ref> = sibs.iter().collect();

        // References are windowed to THEIR OWN race too: a reference's tail is
        // no more a run than ours is.
        let hw = worst_over(&a, &humans, minsep, &mut cache);
        let sw = worst_over(&a, &sib_refs, minsep, &mut cache);
        let (rheld, rdec, rcmp) = match &hw {
            None => (humans.len(), 0usize, 0usize),
            Some(w) => (w.n_refs_held, w.n_refs_decoded, w.n_refs_compared),
        };
        let (hv, hr, hn, hi, hru, hh, ht, htm, hg, hm, hmin) = match &hw {
            None => (
                "NO-HUMAN-REFERENCE".to_string(),
                "-".to_string(),
                0,
                0,
                0,
                0,
                0,
                "-".to_string(),
                "-".to_string(),
                "-".to_string(),
                "-".to_string(),
            ),
            Some(w) => (
                if w.v == Verdict::NoComparison {
                    "NO-COMMON-INSTANTS".to_string()
                } else {
                    w.v.word().to_string()
                },
                w.refname.clone(),
                w.n_common,
                w.n_ident,
                w.n_runs,
                w.head_n,
                w.tail_n,
                if w.tail_n > 0 { fmt_s(w.tail_ms) } else { "-".into() },
                format!("{:.4}", w.gap),
                format!("{:.4}", w.maxd),
                format!("{:.6}", w.minsep_seen),
            ),
        };
        let (sv, sr, sg, si) = match &sw {
            None => ("-".to_string(), "-".to_string(), "-".to_string(), 0),
            Some(w) => (
                w.v.word().to_string(),
                w.refname.clone(),
                format!("{:.4}", w.gap),
                w.n_ident,
            ),
        };
        if hv.starts_with("NO-HUMAN-REFERENCE") {
            n_no_ref += 1;
        } else if hv.starts_with("NO-COMMON-INSTANTS") {
            n_no_common += 1;
        }
        if rcmp == 1 {
            one_ref_only += 1;
        }
        *hist.entry(hv.split(' ').next().unwrap_or("?").to_string()).or_insert(0) += 1;
        println!(
            "{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            map,
            name,
            md5,
            a.len(),
            span,
            hr,
            hn,
            hi,
            hru,
            hh,
            ht,
            htm,
            hg,
            hm,
            hmin,
            hv,
            sr,
            sv,
            sg,
            si,
            rheld,
            rdec,
            rcmp
        );
    }
    // ---- COVERAGE, on stderr so the TSV stays a TSV ----------------------
    eprintln!("\nCOVERAGE of this sweep -- the denominators behind every row above");
    eprintln!("  {:>5}  ghost files found under the corpus", n_seen);
    eprintln!("  {:>5}  refused to decode (no verdict is possible for these)", n_decode_fail);
    eprintln!("  {:>5}  had NO human reference held for their map", n_no_ref);
    eprintln!("  {:>5}  had a reference but NO instant in common with it", n_no_common);
    eprintln!(
        "  {:>5}  were compared against exactly ONE reference -- those rows are CLEAN-vs-1-REF, \
         never CLEAN",
        one_ref_only
    );
    eprintln!("  verdicts:");
    for (k, v) in &hist {
        eprintln!("  {:>5}  {}", v, k);
    }
    eprintln!(
        "\n  A verdict is a claim about the references it was compared against. \
         'CLEAN' with refs_compared=1 and 'CLEAN' with refs_compared=4 are different claims."
    );
}

// ===========================================================================
// `intg gate` -- THE PUBLISH GATE. One command, exits non-zero, run BEFORE a
// ghost reaches a README.
//
// Every quality instrument this project owns was built after the artefacts it
// should have gated: `tmtraj check` was written the day ~170 ghosts were
// already published, and the re-convergence test an hour after that. That
// ordering is the defect this command exists to end.
//
// Three independent families, and a file must clear all three:
//
//   A. THE C-CHECKS (upstream `tmtraj check`) -- the file against itself and
//      the map: finite, moves, no teleport, no post-finish tail, the contact
//      and surface fields against free fall, the wheels, the input echo.
//
//   B. CONTAMINATION -- the bit-exact re-convergence test, against every human
//      recording we hold for the map. Identical -> metres apart -> identical
//      again cannot be driven, only spliced.
//
//   C. THE ORACLE -- the dedicated server re-simulating THE WRITTEN FILE and
//      returning the declared time. Not the tape it was built from: the file
//      that would be published, as it sits on disk.
//
// SANCTIONED EXCEPTIONS. A gate that refuses genuine files teaches people to
// override gates, so the narrow, named withdrawals are honoured -- and every
// application is LOGGED by name, so the exceptions are visible in the record
// rather than folded into a pass:
//
//   * lone C5  -- ground contact ON while provably airborne; the carrier's
//     byte on the two untitled ghosts, a known withdrawal.
//   * lone C10 -- the contact byte's flight claim; understood and unfixed.
//   * lone C8 ONLY when the file passes C8b below.
//
// Any second finding alongside an exception refuses. Any other failure
// refuses. Keep this list exactly this narrow.
//
// C8b -- WHY C8 AS WRITTEN REFUSES GENUINE FILES. Upstream C8 asks what share
// of a file's rolling steps imply a radius in 0.30..0.45 m. That band is the
// STADIUM car's wheel. On a snow map the car is swapped by the gameplay items
// and its wheel is 0.4700 m -- measured twice from opposite directions, on a
// downloaded Nadeo recording that re-simulates to its own exact millisecond,
// and by locating the wheel block in engine memory (0.4701 m). So the band is
// an automatic refusal on every non-Stadium car: snow, desert, rally. Same
// family as the free-fall constant that was carried at -22.3 for weeks.
//
// C8b measures the radius PER FILE and tests CONSISTENCY -- does the wheel
// behave like a wheel of SOME fixed size -- instead of equality with a
// constant. A foreign wheel byte-set still fails it: those bytes turn to
// another run's motion, so paired with THIS file's positions they imply no
// consistent radius at all.
// ===========================================================================

pub struct WheelFit {
    pub radius: f64,
    pub share: f64,
    pub n: usize,
}

/// Implied radii from the rolling steps, and how tightly they cluster.
pub fn wheel_fit(v: &[R], cls: &[crate::whlcmd::Cls]) -> Option<WheelFit> {
    use crate::whlcmd::Cls;
    let turns = |x: &R| -> f64 { x.b(7) as f64 + x.b(6) as f64 / 255.0 };
    let mut rr: Vec<f64> = Vec::new();
    for i in 1..v.len() {
        if cls.get(i) != Some(&Cls::Supported) {
            continue;
        }
        let mut dt = turns(&v[i]) - turns(&v[i - 1]);
        while dt < -128.0 {
            dt += 256.0;
        }
        while dt > 128.0 {
            dt -= 256.0;
        }
        let mut d = 0.0;
        for k in 0..3 {
            let q = v[i].pos[k] - v[i - 1].pos[k];
            d += q * q;
        }
        let d = d.sqrt();
        let dt = dt.abs();
        if dt > 1e-4 && d > 0.05 {
            rr.push(d / (dt * std::f64::consts::TAU));
        }
    }
    if rr.len() < 5 {
        return None;
    }
    // The mode over the whole PHYSICALLY POSSIBLE range, not a Stadium band.
    // 0.15 .. 0.90 m spans every car this game ships; the point is only to
    // exclude the wheelspin pile at ~0.05 and the locked-wheel tail above 1 m,
    // both of which are real driving and neither of which is the free-rolling
    // radius.
    let cand: Vec<f64> = rr.iter().cloned().filter(|x| (0.15..=0.90).contains(x)).collect();
    if cand.is_empty() {
        return Some(WheelFit { radius: f64::NAN, share: 0.0, n: rr.len() });
    }
    let mut b: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for x in &cand {
        *b.entry((x * 100.0).round() as i64).or_insert(0) += 1;
    }
    let k = *b.iter().max_by_key(|(_, v)| **v).unwrap().0;
    let mut c: Vec<f64> = cand
        .iter()
        .cloned()
        .filter(|x| ((x * 100.0).round() as i64) == k)
        .collect();
    c.sort_by(|a, b| a.total_cmp(b));
    let radius = c[c.len() / 2];
    // CONSISTENCY: what share of all rolling steps sit within 15 % of that one
    // radius. A real wheel returns to its free-rolling value constantly; a
    // foreign byte-set never settles anywhere.
    let lo = radius * 0.85;
    let hi = radius * 1.15;
    let share = rr.iter().filter(|x| **x >= lo && **x <= hi).count() as f64 / rr.len() as f64;
    Some(WheelFit { radius, share, n: rr.len() })
}

/// THE ORACLE IS `ghost::oracle`, AND THIS GATE NO LONGER HAS A PARSER OF ITS
/// OWN.
///
/// It had one, and the parser was wrong. `parse_oracle` handled the TIME
/// correctly -- an explicit `"ValidatedResult" : null` test first -- and then
/// read the checkpoint count with
/// `grab_i("NbCheckpoints", "\"ValidatedResult\"")`, a helper that finds
/// `"ValidatedResult"` and scans FORWARD for the key. On a DNF there is no
/// `NbCheckpoints` inside it -- the whole object is `null` -- so the scan ran
/// on into `DeclaredResult` and returned THE FILE'S OWN CLAIM. Measured on the
/// captured transcript in `tools/testdata/oracle_transcript.json`:
/// `sim_time=None, cps=Some(4), declared=Some(19538)` — the gate reported four
/// validated checkpoints for a run the server refused outright.
///
/// That was the FIFTH copy of the dedicated-server driver in this tree, and the
/// fix is not a better copy. `ghost::oracle` is the one driver and the one
/// parser: it tracks which result block it is inside instead of scanning
/// forward from a key, it keeps the validated and declared numbers in separate
/// fields so their disagreement is a value rather than a bug, and on a DNF it
/// takes the checkpoint count out of the server's own prose (`reached some
/// checkpoints (2 out of 4)`), which is the only place a DNF's count exists.
///
/// `tests/oracle_gate.rs` pins both halves against that transcript -- and runs
/// the WRONG parser on the same bytes and requires the wrong answer, so the
/// fixture cannot quietly stop being a test.
pub fn oracle_run(server: &str, map: &str, file: &str) -> Result<ghost::oracle::SimResult, String> {
    ghost::oracle::validate(
        std::path::Path::new(server),
        std::path::Path::new(file),
        ghost::oracle::MapsMode::One(std::path::Path::new(map)),
        "intg-gate",
    )
}

/// The gate's verdict on one file.
pub struct GateOut {
    pub code: i32,
    pub lines: Vec<String>,
}


pub fn route_compare(ghost: &str, route: &str) -> Result<(usize, i64, f64, f64, f64, f64, f64), String> {
    let rows = std::fs::read_to_string(route).map_err(|e| format!("{}: {}", route, e))?;
    let mut hdr = true;
    let mut rmap: std::collections::HashMap<i64, [f64; 3]> = std::collections::HashMap::new();
    let mut mag = 0.0f64;
    for l in rows.lines() {
        if hdr {
            hdr = false;
            continue;
        }
        let f: Vec<&str> = l.split(',').collect();
        if f.len() < 4 {
            continue;
        }
        let (t, x, y, z) = (
            f[0].parse::<i64>().ok(),
            f[1].parse::<f64>().ok(),
            f[2].parse::<f64>().ok(),
            f[3].parse::<f64>().ok(),
        );
        if let (Some(t), Some(x), Some(y), Some(z)) = (t, x, y, z) {
            mag = mag.max(x.abs()).max(y.abs()).max(z.abs());
            rmap.insert(t, [x, y, z]);
        }
    }
    if rmap.is_empty() {
        return Err(format!("{} has no usable rows", route));
    }
    let d = record::decode_ghost(ghost).map_err(|e| format!("{}: {}", ghost, e))?;
    // SCAN INTEGER TICK OFFSETS. Comparing at lag 0 and reporting a MAGNITUDE is
    // how the first version of this check convicted an honest file: 227654 reads
    // 0.5485 m at lag 0 and 0.0000 m at lag -1, because 0.5485 m is how far that
    // car travels in one 10 ms tick. The same failure C11b is documented as
    // having. A tick offset is a property of the RUN and a solo clip cannot look
    // wrong from it; a genuinely different trajectory collapses to zero at NO lag.
    let at = |lag: i64| -> Option<(usize, f64, f64, f64)> {
        let mut ds: Vec<f64> = Vec::new();
        for s in &d.samples {
            if let Some(p) = rmap.get(&(s.time_ms as i64 + lag * 10)) {
                ds.push(((s.x as f64 - p[0]).powi(2) + (s.y as f64 - p[1]).powi(2) + (s.z as f64 - p[2]).powi(2)).sqrt());
            }
        }
        if ds.is_empty() { return None; }
        let n = ds.len();
        let mean = ds.iter().sum::<f64>() / n as f64;
        let mx = ds.iter().cloned().fold(0.0f64, f64::max);
        ds.sort_by(|a, b| a.total_cmp(b));
        Some((n, ds[n / 2], mean, mx))
    };
    let zero = at(0).map(|z| z.1).unwrap_or(f64::NAN);
    let mut best: Option<(i64, usize, f64, f64, f64)> = None;
    for lag in -5..=5i64 {
        if let Some((n, med, mean, mx)) = at(lag) {
            if best.map(|b| med < b.2).unwrap_or(true) { best = Some((lag, n, med, mean, mx)); }
        }
    }
    let Some((lag, n, med, mean, mx)) = best else {
        return Err(format!(
            "the record and {} share no instant at any tick offset -- the route was made from a \
             different run, or at a probe tick past the record's span",
            route
        ));
    };
    let quant = (mag * 1e-6 * 2.0).max(1e-6);
    Ok((n, lag, med, mean, mx, quant, zero))
}

pub fn gate_one(
    ghost: &str,
    race: i64,
    refs: &[Ref],
    mapid: &str,
    server: Option<&str>,
    map: Option<&str>,
    minsep: f64,
    cache: &mut Cache,
    manifest_override: Option<&str>,
    require_manifest: bool,
    source_ref: Option<&str>,
    flag_route: Option<String>,
    flag_route_dir: Option<String>,
) -> GateOut {
    let mut lines: Vec<String> = Vec::new();
    let mut hard = 0usize;
    // Inputs the gate could not read. NEVER folded into `hard`: an operator
    // error is not a property of the ghost.
    let mut unmeasured = 0usize;

    // ---- A. the C-checks -------------------------------------------------
    // Does the file's OWN manifest declare the ground-contact byte inherited
    // from the carrier? `ghost regen` writes 22 of the 116 bytes in a sample
    // from the engine plus three from the tape; the other 91, byte 89's
    // contact flag among them, stay the carrier's. C6 and C10 read byte 89, so
    // on such a file they are measuring the CARRIER -- and this gate's own
    // rule is that an input it cannot read is never a verdict about the
    // subject. The file has to say so itself: no manifest, no exemption.
    let contact_inherited = manifest_says_contact_inherited(ghost, manifest_override);
    let (ccode, cfails, cwarns, clines) = run_checks(ghost, race, contact_inherited);
    for l in &clines {
        lines.push(format!("  {}", l));
    }
    // which checks failed, by id
    let failed: Vec<String> = clines
        .iter()
        .filter(|l| l.starts_with("FAIL"))
        .map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string())
        .collect();
    let _ = cwarns;
    if ccode == 2 {
        // the narrow sanctioned withdrawals, each logged BY NAME
        let lone = |id: &str| failed.len() == 1 && failed[0] == id;
        if lone("C5") {
            lines.push("EXCEPTION C5 applied -- lone C5 (contact ON while airborne), a named withdrawal".into());
        } else if lone("C10") {
            lines.push("EXCEPTION C10 applied -- lone C10 (the flight claim), understood and unfixed".into());
        } else if lone("C8") {
            // C8b decides: is there a consistent wheel of SOME size?
            match decode(ghost) {
                Ok(r) => {
                    let v: Vec<R> = r.into_iter().filter(|x| race <= 0 || x.ms <= race).collect();
                    let c = crate::whlcmd::classify(&v, crate::whlcmd::G_DEFAULT, 2.0, 5.0, 3);
                    match wheel_fit(&v, &c.cls) {
                        Some(w) if c8b_accepts(w.share, w.radius) => {
                            lines.push(format!(
                                "EXCEPTION C8 applied -- C8b PASSES: {:.0} % of {} rolling steps sit within 15 % of one radius, {:.4} m. \
                                 Upstream C8's 0.30-0.45 m band is the STADIUM wheel; a snow car measures 0.4700 m.",
                                100.0 * w.share, w.n, w.radius
                            ));
                        }
                        Some(w) => {
                            hard += 1;
                            lines.push(format!(
                                "FAIL   C8b    no consistent wheel: only {:.0} % of {} rolling steps cluster (mode {:.4} m)",
                                100.0 * w.share, w.n, w.radius
                            ));
                        }
                        None => {
                            hard += 1;
                            lines.push("FAIL   C8b    too few rolling steps to test the wheel".into());
                        }
                    }
                }
                Err(e) => {
                    hard += 1;
                    lines.push(format!("FAIL   C8b    {}", e));
                }
            }
        } else {
            hard += 1;
            lines.push(format!(
                "FAIL   C-checks  {} failing ({}) -- only a LONE C5, C10 or C8 is sanctioned",
                cfails,
                failed.join(",")
            ));
        }
    }

    // ---- B. contamination, against every human recording we hold ---------
    let a = decode(ghost);
    match &a {
        Err(e) => {
            hard += 1;
            lines.push(format!("FAIL   B-contam  no readable vehicle record: {}", e));
        }
        Ok(a) => {
            let humans: Vec<&Ref> =
                refs.iter().filter(|r| r.map == mapid && r.kind == "human").collect();
            if humans.is_empty() {
                // NOT a pass. An untested file is untested.
                lines.push(format!(
                    "n/a    B-contam  NO HUMAN REFERENCE for map {} -- this file is UNTESTED for contamination, not clean",
                    mapid
                ));
            } else {
                match worst_over(a, &humans, minsep, cache) {
                    Some(w) if w.v == Verdict::Contaminated => {
                        hard += 1;
                        lines.push(format!(
                            "FAIL   B-contam  SPLICE against {}: {} of {} samples bit-identical in {} runs, \
                             {:.1} m apart between them. Identical -> apart -> identical cannot be driven.",
                            w.refname, w.n_ident, w.n_common, w.n_runs, w.gap
                        ));
                    }
                    Some(w) if w.v == Verdict::Identical => {
                        hard += 1;
                        lines.push(format!(
                            "FAIL   B-contam  this file's telemetry IS {} -- all {} shared samples bit-identical. \
                             It is a human recording published under our name.",
                            w.refname, w.n_ident
                        ));
                    }
                    Some(w) => {
                        let big_head = w.head_n * 4 >= w.n_common && w.head_n >= 10;
                        if big_head {
                            // NOT a refusal, and not a clean pass either. A
                            // head-anchored identical block is what determinism
                            // produces when two runs share an opening tape --
                            // and it is ALSO what an inherited donor prefix
                            // looks like, which is the exact shape `fk regen`
                            // used to produce from a late handover. The two are
                            // not separable from the file alone. What separates
                            // them is the MANIFEST: whether those samples were
                            // written from engine state. Say so, loudly, rather
                            // than pick one.
                            lines.push(format!(
                                "WARN   B-contam  {} of {} samples are bit-identical to {} in a HEAD-ANCHORED block ({:.0} %). \
                                 Determinism produces this from a shared opening tape; so does an inherited donor prefix. \
                                 The file cannot tell you which -- its manifest can.",
                                w.head_n, w.n_common, w.refname,
                                100.0 * w.head_n as f64 / w.n_common.max(1) as f64
                            ));
                        } else {
                            lines.push(format!(
                                "PASS   B-contam  no splice against {} human recording(s); closest {} \
                                 ({} of {} samples bit-identical, max separation {:.1} m)",
                                humans.len(), w.refname, w.n_ident, w.n_common, w.maxd
                            ));
                        }
                    }
                    None => lines.push(format!(
                        "n/a    B-contam  {} human reference(s) named for map {}, none decodable",
                        humans.len(), mapid
                    )),
                }
            }
        }
    }

    // ---- B2. C12: after the tapes part, the trajectories MUST part -------
    //
    // B-contam is bit-exact, and it says so: a near-copy that tracks the
    // reference without ever being byte-equal passes it by construction. This
    // is the check that plugs that hole, and until now it was implemented in
    // this file and WIRED TO NOTHING — reachable only from `intg c12`, which
    // no pipeline ran. A correct check nobody calls is not coverage.
    //
    // It asks the physics question exactness cannot: the inputs diverge at some
    // tick, so the trajectories must separate. Staying together is the anomaly.
    // The bar is GROWTH relative to the file's own pre-divergence separation,
    // not a distance — on 279218 five independent clean runs sit 0.29-0.62 m
    // from the human, so a metre bar refuses honest work, while the two
    // populations here are six orders of magnitude apart (0.000509 m against
    // 1113 m) and a factor of ten is a gulf rather than a tuned edge.
    if let Ok(a) = &a {
        let humans: Vec<&Ref> =
            refs.iter().filter(|r| r.map == mapid && r.kind == "human").collect();
        let mut ran = false;
        for h in &humans {
            let Some(fd) = first_sustained_diff(ghost, &h.path) else { continue };
            let Ok(b) = decode(&h.path) else { continue };
            let Some(d) = divergence_growth(a, &b, fd) else { continue };
            ran = true;
            let growth = d.sep_after / d.sep_before.max(1e-9);
            if c12_is_near_copy(d.sep_before, d.sep_after) {
                hard += 1;
                lines.push(format!(
                    "FAIL   C12       NEAR-COPY of {}: the tapes part at {} and differ over the \
                     {} samples after it, yet the trajectories stay {:.6} m apart ({:.1}x the \
                     {:.6} m they were already apart). Bit-exact tests cannot see this.",
                    std::path::Path::new(&h.path)
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    crate::fmt::secs(d.first_diff_ms),
                    d.n_after,
                    d.sep_after,
                    growth,
                    d.sep_before
                ));
            } else if growth <= C12_GROWTH_BAR {
                // Low growth, but they ARE in different places. Reported, not
                // refused: this is what a pair that was already apart before
                // the tapes parted looks like, and the ratio alone convicted it.
                lines.push(format!(
                    "note   C12       {}: low growth ({:.1}x) after the tapes part at {}, but the \
                     trajectories are {:.4} m apart -- past the {:.2} m at which \"they never \
                     separated\" is a description rather than an opinion.",
                    std::path::Path::new(&h.path)
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    growth,
                    crate::fmt::secs(d.first_diff_ms),
                    d.sep_after,
                    C12_TOGETHER_M
                ));
            }
        }
        if !ran {
            lines.push(
                "n/a    C12       no human reference whose tape differs from this one, so the \
                 near-copy test could not run"
                    .into(),
            );
        } else if !lines.iter().any(|l| l.starts_with("FAIL   C12")) {
            lines.push(
                "PASS   C12       the trajectories part after the inputs part, as they must".into(),
            );
        }
    }

    // ---- C. the oracle re-simulating THE WRITTEN FILE --------------------
    match (server, map) {
        (Some(s), Some(m)) => match oracle_run(s, m, ghost) {
            Ok(o) => {
                let st = o.time_ms;
                match st {
                    Some(t) if race > 0 && t == race => lines.push(format!(
                        "PASS   C-oracle  the server re-simulates THIS FILE to {:.3} s, the declared time (cps {}, IsValid {})",
                        t as f64 / 1000.0,
                        o.cps.map(|v| v as i64).unwrap_or(-1),
                        o.is_valid.unwrap_or(false)
                    )),                    Some(t) if race > 0 => {
                        hard += 1;
                        lines.push(format!(
                            "FAIL   C-oracle  the server returns {:.3} s but the file is published as {:.3} s (delta {:+} ms). \
                             A 1 ms delta is the `u02 truncate` signature -- check the tape had 40 ticks of headroom.",
                            t as f64 / 1000.0,
                            race as f64 / 1000.0,
                            t - race
                        ));
                    }
                    Some(t) => lines.push(format!(
                        "n/a    C-oracle  the server returns {:.3} s; no --race given to check it against",
                        t as f64 / 1000.0
                    )),
                    None => {
                        hard += 1;
                        lines.push(format!(
                            "FAIL   C-oracle  the server REFUSES this file: {} (declared {:?})",
                            if o.desc.is_empty() { "DNF / not validated".into() } else { o.desc.clone() },
                            o.declared_ms
                        ));
                    }
                }
                // ---- C-header: WHOSE FILE THE GAME THINKS THIS IS ----------
                //
                // 173636's regeneration is clean against 31 human recordings
                // -- every position in it is ours -- and it still carries the
                // rank-1 human's account id and login, inherited from the
                // carrier its template was built on. NO POSITIONAL CHECK CAN
                // SEE THAT, however many references you throw at it.
                //
                // Two measurable statements, both taken from the game's own
                // parser rather than from a manifest's claim:
                //
                //   1. the header's declared time must equal what the server
                //      validates the file to. 173636's declares 23.638 -- the
                //      human's time -- and validates to 22.072. Our published
                //      TAS_22072 declares 22.072 and validates to 22.072, and
                //      is the control that makes this real.
                //   2. the file must carry no account id at all. Our own files
                //      carry none and report Login "TAS".
                //
                // Point 1 caught 173636 BY ACCIDENT: had the carrier's declared
                // time happened to match, IsValid would have read true and the
                // file would have shipped under a real player's account. Point
                // 2 is the check that would have caught it on purpose.
                match (o.declared_ms, o.time_ms) {
                    (Some(d), Some(v)) if d != v => {
                        // WHOSE time is it? A header declaring another RUN's
                        // time is a borrowed container. A header declaring the
                        // map's AUTHOR time is a different field entirely and
                        // no claim about anyone's driving.
                        //
                        // 146612's KEYBOARD_39706 and TAS_39183 both declare
                        // 39.555, which is on no leaderboard we hold -- the
                        // census reads 40223 x0 and 40226 x0 in both -- and
                        // both files carry their OWN time at three sites
                        // (+92 +148) with 39555 at two more, 7 kB away, at a
                        // +60 spacing that appears nowhere in the six-site
                        // group. Two distinct fields, not one half-overwritten.
                        // Rewriting those two sites would assert that the map's
                        // author time equals our run: manufacturing a false
                        // claim to make a check go green.
                        //
                        // CONTROL: 191465's WIP_keyboard declares 13.081, which
                        // IS the map's rank-1 human time, and must still refuse.
                        // The two cases are separated by whether the declared
                        // value is a human record we hold -- not by how far it
                        // is from the validated time.
                        let foreign = refs
                            .iter()
                            .filter(|r| r.map == mapid && r.kind == "human")
                            .any(|r| declared_race_ms(&r.path) == Some(d));
                        if foreign {
                            hard += 1;
                            lines.push(format!(
                                "FAIL   C-header  the header declares {:.3} s -- a HUMAN RECORD of this map -- but the server \
                                 validates this file to {:.3} s. A container whose header was inherited and never rewritten.",
                                d as f64 / 1000.0,
                                v as f64 / 1000.0
                            ));
                        } else {
                            lines.push(format!(
                                "note   C-header  the header declares {:.3} s, which is NOT a human record of this map \
                                 (most likely the map's author time) while the server validates this file to {:.3} s. \
                                 Its own time is present at its own sites; no claim about anyone's driving is involved.",
                                d as f64 / 1000.0,
                                v as f64 / 1000.0
                            ));
                        }
                    }
                    (Some(d), Some(_)) => lines.push(format!(
                        "PASS   C-header  the header's declared time and the validated time agree at {:.3} s",
                        d as f64 / 1000.0
                    )),
                    _ => lines.push(
                        "n/a    C-header  the server returned no declared/validated pair".into(),
                    ),
                }
                // `ghost::oracle` reports these as plain strings, empty when
                // the server did not print the field at all -- which is the
                // state OUR files are supposed to be in.
                match (o.account_id.as_str(), o.login.as_str()) {
                    (a, l) if !a.is_empty() => {
                        hard += 1;
                        lines.push(format!(
                            "FAIL   C-ident   this file carries a player account id ({}{}). Ours carry NONE. \
                             The file says it is somebody's; that is a claim about a person, not about a trajectory.",
                            a,
                            if l.is_empty() { String::new() } else { format!(", login {}", l) }
                        ));
                    }
                    (_, l) if l != "TAS" && !l.is_empty() => {
                        hard += 1;
                        lines.push(format!(
                            "FAIL   C-ident   this file's login is {:?}, not \"TAS\". Ours report \"TAS\" and no account id.",
                            l
                        ));
                    }
                    _ => lines.push(
                        "PASS   C-ident   no account id; the login is ours".into(),
                    ),
                }
            }
            Err(e) => {
                // AN INPUT WE COULD NOT READ IS NOT A VERDICT ABOUT THE GHOST.
                //
                // This used to be `hard += 1` and a FAIL line, which meant that
                // typing the wrong map path REFUSED the subject -- a file the
                // gate had never read. Every other silent-wrong-answer tonight
                // produced a false NEGATIVE (`0.0 m travelled`, `M-time`
                // comparing two command-line numbers, curl scoring a 404 green).
                // This one produced a false REFUSAL, and false refusals are how
                // a gate dies: refuse honest work twice and the third time
                // somebody reaches for the override, after which it catches
                // nothing at all.
                //
                // The rule, as a class: any input the gate could not read yields
                // UNMEASURED naming THE INPUT, never a verdict about the subject.
                unmeasured += 1;
                lines.push(format!(
                    "UNMEASURED  C-oracle  the oracle could not run -- {} -- so THIS FILE WAS NOT \
                     RE-SIMULATED. This says nothing about the ghost: it is an input the gate \
                     could not read.",
                    e
                ));
            }
        },
        _ => lines.push(
            "n/a    C-oracle  no --server/--map given -- THE FILE HAS NOT BEEN RE-SIMULATED".into(),
        ),
    }

    // ---- E. the STALE-BUFFER check, against the source this file was made
    //      from. Only runs when a source is named: it compares two renderings
    //      of the SAME run, which is the only case where "one tick behind" is
    //      a meaningful statement.
    if let Some(src) = source_ref {
        match (decode(ghost), decode(src)) {
            (Ok(a), Ok(b)) => match stale_check(&a, &b) {
                Some(v) => {
                    let (code, word) = stale_verdict(&v);
                    if code != 0 {
                        hard += 1;
                        lines.push(format!(
                            "FAIL   E-stale    {} (median {:.4} m, max {:.4} m vs {})",
                            word,
                            v.median_m,
                            v.max_m,
                            std::path::Path::new(src)
                                .file_name()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default()
                        ));
                    } else {
                        lines.push(format!("PASS   E-stale    {}", word));
                    }
                }
                None => lines.push(
                    "n/a    E-stale    too few shared instants with the source to compare".into(),
                ),
            },
            _ => lines.push("n/a    E-stale    the source could not be decoded".into()),
        }
    } else {
        lines.push(
            "n/a    E-stale    no --source given -- THIS FILE HAS NOT BEEN CHECKED against the \
             one-tick-stale buffer copy, which passes every other check in this gate"
                .into(),
        );
    }

    // ---- F. C-SPAWN: is the car pointing where every car on this map points?
    //
    // Every run of a map begins at the same spawn, in the same attitude. So the
    // FIRST in-race sample of an honest recording equals a downloaded human
    // recording's first in-race sample -- in POSITION and in ORIENTATION -- and
    // a file whose first sample is somewhere else, or facing somewhere else, is
    // not this map's run whatever its coverage says.
    //
    // WHY ORIENTATION IS HERE, AND WHY NO OTHER CHECK CATCHES IT. The writer
    // encodes the engine's rotation, and the engine object it locates stores
    // that rotation in one of three layouts (`fk regen`'s probe prints
    // `orient kind N`). Which one it finds VARIES BETWEEN RUNS of the same
    // command. Get it wrong and every position is still exact, so C1-C10, the
    // oracle, the tape md5 and the whole contamination family pass while the
    // car faces the wrong way for the entire render. That is exactly what
    // 197047's withdrawn clip was: identity spawn quaternion, correct path.
    // Measured on this map's own answer key, one run, two encodings:
    // the probe's own kind -> 0.0068 deg against the human's recorded bytes,
    // the "settled convention" pinned over it -> 90.68 deg.
    //
    // The bars are wide on purpose. This separates 0.004 deg from 90 deg; it is
    // not a precision instrument and must never be read as one.
    // A COMMON INSTANT, NOT EACH FILE'S OWN FIRST SAMPLE. "Every run of this
    // map starts in the same place facing the same way" is a statement about
    // RACE ZERO, and a carrier record does not always begin there: 227654's
    // rank-1 recording starts at race 1.310 s doing 66 km/h. Comparing each
    // file's own first in-race sample then measures a car on the start block
    // against a car already down the road -- 11.2 m and 48.9 deg -- and the
    // check INVERTS, refusing exactly the repaired files it exists to admit
    // while passing a file 11.2 m along. So: take the reference's first
    // in-race sample, find the SUBJECT sample nearest that instant, and if
    // there is none within 60 ms say the records do not overlap at the start
    // rather than pronouncing on them. (Found by 6d743a47; the same fix is in
    // `ghost regen`'s G2.)
    {
        let firstpos = |d: &record::Decoded| -> Option<record::Sample> {
            d.samples.iter().find(|s| s.time_ms >= 0).cloned()
        };
        let mut done = false;
        for r in refs.iter().filter(|r| r.map == mapid && r.kind == "human") {
            let (a, b) = match (
                record::decode_ghost(ghost),
                record::decode_ghost(&r.path),
            ) {
                (Ok(a), Ok(b)) => (a, b),
                _ => continue,
            };
            // COMPARE AT A COMMON INSTANT, not at "each file's first sample".
            //
            // "Every run of this map starts in the same place" is true, and it
            // is a statement about RACE ZERO. A carrier's record does not
            // always begin there: 227654's rank-1 recording starts at race
            // 1.310 s doing 66 km/h, so a regenerated run that correctly begins
            // on the start block reads 11.2 m and 49 deg out and is refused,
            // while a file 11.2 m down the road would pass. The check inverted,
            // on exactly the class of file this gate exists to admit once it is
            // repaired.
            //
            // So take the reference's first in-race sample and the SUBJECT's
            // sample nearest that same instant. When there is none within one
            // sample period the two records do not overlap at the start at all,
            // and the honest answer is that the spawn is unchecked -- never a
            // refusal computed against the wrong moment.
            let Some(sb) = firstpos(&b) else { continue };
            let Some(sa) = a
                .samples
                .iter()
                .filter(|s| s.time_ms >= 0)
                .min_by_key(|s| (s.time_ms - sb.time_ms).abs())
                .cloned()
            else {
                continue;
            };
            if (sa.time_ms - sb.time_ms).abs() > 60 {
                lines.push(format!(
                    "n/a    C-spawn   UNMEASURED: the reference's first in-race sample is at                      {:.3} s and this record has none within 60 ms of it, so there is no common                      instant. THE SPAWN AND THE FACING ARE UNCHECKED.",
                    sb.time_ms as f64 / 1000.0
                ));
                unmeasured += 1;
                done = true;
                break;
            }
            let nm = std::path::Path::new(&r.path)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let dpos =
                (((sa.x - sb.x) as f64).powi(2) + ((sa.y - sb.y) as f64).powi(2) + ((sa.z - sb.z) as f64).powi(2)).sqrt();
            // rotation angle between two unit quaternions: 2 acos |a.b|
            let dot = (sa.qx as f64 * sb.qx as f64 + sa.qy as f64 * sb.qy as f64 + sa.qz as f64 * sb.qz as f64 + sa.qw as f64 * sb.qw as f64).abs();
            let dang = 2.0 * dot.min(1.0).acos() * 180.0 / std::f64::consts::PI;
            if dpos > 0.5 || dang > 5.0 {
                hard += 1;
                lines.push(format!(
                    "FAIL   C-spawn   at {:.3} s this run is {:.4} m and {:.3} deg from {}'s sample \
                     at the same instant. Every run of this map starts in the same place facing \
                     the same way; {} is another object, or the orientation encoding is permuted.",
                    sa.time_ms as f64 / 1000.0, dpos, dang, nm,
                    if dpos > 0.5 { "this position" } else { "this attitude" }
                ));
            } else {
                lines.push(format!(
                    "PASS   C-spawn   at {:.3} s, {:.4} m and {:.3} deg from {}'s sample at {:.3} s",
                    sa.time_ms as f64 / 1000.0,
                    dpos,
                    dang,
                    nm,
                    sb.time_ms as f64 / 1000.0
                ));
            }
            done = true;
            break;
        }
        if !done {
            lines.push(
                "n/a    C-spawn   no human recording of this map could be decoded -- THE SPAWN \
                 AND THE FACING ARE UNCHECKED, and a permuted orientation passes every other \
                 check in this gate"
                    .into(),
            );
        }
    }

    // ---- G. C-ROUTE: the record against the ENGINE, read by a different
    //      instrument.
    //
    // THIS IS THE ONLY CHECK IN THE GATE THAT DOES NOT READ THE RECORD PATH.
    // Everything above interrogates the written record, or the tape, or the
    // file's own header, and on 227654 every one of them passed a file whose
    // telemetry is the CONTAINER's run and not ours:
    //
    //   * B-contam passed -- the record is 0.000511 m from ailiei.'s, which is
    //     the client-vs-server floor, not bit-identical.
    //   * C-spawn passed -- the first sample is the map's spawn, because his
    //     run starts there too.
    //   * C-oracle passed -- the oracle re-simulates the TAPE, and the tape is
    //     ours.
    //   * E-stale passed -- two independent generations agreed exactly.
    //
    // They agreed with each other because they were all reading one poisoned
    // source. What convicted the file was asking a DIFFERENT instrument the
    // same question: `fk btraj2` re-simulates the tape and dumps the car's
    // per-tick position without going near the record. Our car was 0.7249 m
    // mean / 1.8607 m max from the line the record claims, over 301 instants --
    // and two of our tapes that differ at race 11.270 s, inside the recorded
    // window, produced BIT-IDENTICAL records, which two different simulations
    // cannot do.
    //
    // The bar is derived from the instrument, not chosen: `fk btraj2` writes
    // six significant digits, so at coordinates around 1400 its own quantum is
    // ~0.01 m. Anything within a small multiple of that is agreement to the
    // resolution of the reading; 0.55 m is fifty times it. Nothing in the gap.
    //
    // NO ROUTE IS A REFUSAL, NOT AN `n/a`. A file that cannot be route-checked
    // is precisely the file this hole hides in.
    {
        let route = flag_route.clone().or_else(|| {
            flag_route_dir.as_ref().map(|d| {
                let stem = std::path::Path::new(ghost)
                    .file_name()
                    .map(|s| s.to_string_lossy().replace(".Ghost.Gbx", ""))
                    .unwrap_or_default();
                format!("{}/route_{}.csv", d, stem)
            })
        });
        match route {
            None => {
                hard += 1;
                lines.push(
                    "FAIL   C-route   no --route given. The record has NOT been checked against \
                     the engine by any instrument other than the one that wrote it -- every other \
                     check in this gate READS the record, so none of them can disagree with it \
                     about where the car was. \
                     Produce one with `fk btraj2 --template THIS FILE ... --out route.csv`."
                        .into(),
                );
            }
            Some(rp) => match route_compare(ghost, &rp) {
                Err(e) => {
                    unmeasured += 1;
                    lines.push(format!(
                        "UNMEASURED  C-route   {} -- the engine's own trajectory could not be \
                         read, so THIS FILE IS NEITHER CLEAN NOR CONVICTED on the one axis the \
                         rest of the gate cannot see.",
                        e
                    ));
                }
                Ok((n, lag, med, mean, mx, quant, zero)) => {
                    let bar = (20.0 * quant).max(0.02);
                    if med > bar {
                        hard += 1;
                        lines.push(format!(
                            "FAIL   C-route   the record is {:.4} m from where the engine put this \
                             tape's car, AT THE BEST OF ELEVEN TICK OFFSETS ({:+} ticks; median over \
                             {} shared instants, mean {:.4}, max {:.4}; route quantum {:.4} m, bar \
                             {:.4}). A time shift collapses to zero at some lag and this does not. \
                             THIS RECORD IS NOT THIS RUN.",
                            med, lag, n, mean, mx, quant, bar
                        ));
                    } else if lag != 0 {
                        lines.push(format!(
                            "PASS   C-route   the record matches the engine's own trajectory for this \
                             tape to {:.4} m over {} shared instants -- at a lag of {:+} ticks ({:.4} m \
                             at lag 0, which is just how far this car travels in {} ms). Tick \
                             alignment is a property of the run; check it against the map's own \
                             control before reading anything into it.",
                            med, n, lag, zero, lag.abs() * 10
                        ));
                    } else {
                        lines.push(format!(
                            "PASS   C-route   the record matches the engine's own trajectory for \
                             this tape to {:.4} m over {} shared instants, at lag 0 (route quantum \
                             {:.4} m)",
                            med, n, quant
                        ));
                    }
                }
            },
        }
    }

    // ---- D. the file against its own MANIFEST ---------------------------
    //
    // Last, and it is the one that certifies. A, B and C interrogate the
    // bytes; D asks whether the account of how those bytes were made holds up.
    // Where B can say nothing -- no human recording held, or the donor IS the
    // reference -- D is the only evidence there is.
    let mpath = manifest_override
        .map(|s| s.to_string())
        .unwrap_or_else(|| crate::manifest::manifest_path(ghost));
    match crate::manifest::read(&mpath) {
        Ok(m) => {
            let (mbad, mlines) = crate::manifest::verify(ghost, &m);
            for l in mlines {
                lines.push(format!("  {}", l));
            }
            hard += mbad;
        }
        Err(e) => {
            if require_manifest {
                hard += 1;
                lines.push(format!("FAIL   D-manifest  no readable manifest: {}", e));
            } else {
                lines.push(format!(
                    "n/a    D-manifest  no manifest at {} -- THIS FILE IS UNPROVENANCED. \
                     It may be perfectly good; nothing in it says so. \
                     (--require-manifest makes this a refusal.)",
                    mpath
                ));
            }
        }
    }

    // 2 = refused, 3 = could not be measured, 0 = publishable. Unmeasured is
    // NOT clean and NOT refused; it is its own outcome and its own exit code.
    GateOut { code: if hard > 0 { 2 } else if unmeasured > 0 { 3 } else { 0 }, lines }
}

pub fn cmd_gate(args: &[String]) {
    let flag = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let mut files: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        if args[i].starts_with("--") {
            if matches!(
                args[i].as_str(),
                "--race" | "--refs" | "--map" | "--server" | "--minsep" | "--mapid" | "--manifest" | "--source" | "--route" | "--route-dir"
            ) {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        files.push(args[i].clone());
        i += 1;
    }
    if files.is_empty() {
        print!(
            "\
usage: tmtraj intg gate GHOST... --race MS [--refs refs.tsv] [--mapid ID]
                        [--server DIR --map MAP.Map.Gbx] [--minsep M]

THE PUBLISH GATE. Exit 0 = publishable, 2 = REFUSED.

  A  the C-checks (tmtraj check)
  B  the bit-exact contamination test against every human recording held
  C  the dedicated server re-simulating THE WRITTEN FILE to its declared time

Sanctioned, and logged by name every time one is applied: a LONE C5, a LONE
C10, and a LONE C8 whose file passes C8b (a consistent wheel of any size --
0.36 m is the Stadium car's, a snow car is 0.4700 m). Anything else refuses.

Without --server/--map the file is NOT re-simulated and the gate says so.
Without a human reference the file is UNTESTED for contamination, never clean.
"
        );
        std::process::exit(2);
    }
    let race: i64 = flag("--race").and_then(|v| v.parse().ok()).unwrap_or(-1);
    let minsep: f64 = flag("--minsep").and_then(|v| v.parse().ok()).unwrap_or(5.0);
    let refs = flag("--refs").map(|p| load_refs(&p).unwrap()).unwrap_or_default();
    let server = flag("--server");
    let map = flag("--map");
    let manifest = flag("--manifest");
    let source = flag("--source");
    let require_manifest = args.iter().any(|a| a == "--require-manifest");
    let mut cache: Cache = Cache::new();
    let mut worst = 0;
    for f in &files {
        // the map id: given, or the leading digits of the containing map dir
        let mapid = flag("--mapid").unwrap_or_else(|| {
            let d = map_of(std::path::Path::new(f));
            d.chars().take_while(|c| c.is_ascii_digit()).collect()
        });
        let g = gate_one(
            f,
            race,
            &refs,
            &mapid,
            server.as_deref(),
            map.as_deref(),
            minsep,
            &mut cache,
            manifest.as_deref(),
            require_manifest,
            source.as_deref(),
            flag("--route"),
            flag("--route-dir"),
        );
        println!(
            "=== {}  --  {}",
            f,
            match g.code { 0 => "PUBLISHABLE", 3 => "UNMEASURED (an input could not be read; this is not a verdict about the file)", _ => "REFUSED" }
        );
        for l in &g.lines {
            println!("{}", l);
        }
        worst = worst.max(g.code);
    }
    std::process::exit(worst);
}

/// Run the upstream `tmtraj check` as a SUBPROCESS and read its verdict.
///
/// Deliberately not a library call into `checkcmd`. Two reasons, and the
/// second is the real one:
///
///   * `checkcmd.rs` arrived from another arm and this gate does not modify
///     files it did not write -- a gate that quietly forks the checker it
///     claims to run is the same disease as a filename that claims a run;
///   * the gate then exercises THE COMMAND A PERSON WOULD RUN, byte for byte.
///     If `tmtraj check` and the gate ever disagree, that is a bug in one of
///     them, and a library call would hide it.
fn run_checks(ghost: &str, race: i64, contact_inherited: bool) -> (i32, usize, usize, Vec<String>) {
    let exe = std::env::current_exe().unwrap_or_else(|_| "tmtraj".into());
    let mut c = std::process::Command::new(exe);
    c.arg("check").arg(ghost);
    if race > 0 {
        c.arg("--race").arg(race.to_string());
    }
    // The contact byte is one of the 91 per-sample bytes `ghost regen` leaves
    // as the carrier's. When the file's OWN manifest declares it inherited,
    // a check that reads it is measuring the carrier, not the subject -- and
    // an input the gate cannot read is never a verdict about the file.
    if contact_inherited {
        c.arg("--contact-inherited");
    }
    let out = match c.output() {
        Ok(o) => o,
        Err(e) => {
            return (2, 1, 0, vec![format!("FAIL   C0     could not run tmtraj check: {}", e)])
        }
    };
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let lines: Vec<String> = text
        .lines()
        .map(|l| l.trim_end().to_string())
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("PASS ") || t.starts_with("FAIL ") || t.starts_with("WARN ") || t.starts_with("n/a ")
        })
        .map(|l| l.trim_start().to_string())
        .collect();
    let fails = lines.iter().filter(|l| l.starts_with("FAIL")).count();
    let warns = lines.iter().filter(|l| l.starts_with("WARN")).count();
    let code = out.status.code().unwrap_or(2);
    (code, fails, warns, lines)
}

// ===========================================================================
// `intg dup` -- THE CROSS-FILE CHECK. No per-file gate can see this class.
//
// The corpus pass produced four 227654 ghosts that were each internally
// flawless -- 365 of 365 samples written, nothing missing in the race, the
// oracle exact, every C-check passing -- and all four carried ONE trajectory,
// bit-identical to each other at median, p90, p99 and max. Coverage, oracle
// agreement, tape md5 and C1..C10 are all properties of a SINGLE file, so
// every one of them passed. Only a comparison BETWEEN outputs can see it.
//
// The rule needs both halves:
//
//   identical telemetry + DIFFERENT inputs  -> REFUSE. Two different drives
//       cannot produce the same trajectory; one of them is the other's.
//   identical telemetry + IDENTICAL inputs  -> correct, and must not be
//       refused. 227654's TAS_57493 and TAS_57498 have the same inputs inside
//       the recorded window, so one trajectory is the right answer for both.
//       A naive "no two files may match" rule condemns them.
//
// So the comparison is against the INPUT ARCHIVE, never the telemetry alone.
// The input key used here is the validator's own decoded `Inputs` string --
// the engine's account of what it simulated, which is exactly the thing that
// has to differ for the telemetry to differ. Without a map to run the oracle
// on, the pair is reported as INPUTS-UNCHECKED and never as a pass.
//
// Why this happens, which is the part that constrains regeneration generally:
// the chooser identifies the car by grading candidates against the template's
// OWN recorded positions. On a contaminated file that record is the donor's,
// and all five 227654 files share one donor -- so it picks the donor's object
// every time and emits one trajectory every time. THE VERY CONDITION THAT
// MAKES A FILE WORTH REGENERATING IS THE CONDITION THAT MAKES THE CHOOSER
// UNSAFE. (`FK_NO_CHOOSER=1` is the enforcement upstream.)
// ===========================================================================

pub(crate) fn cmd_dup(args: &[String]) {
    let flag = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let corpus = flag("--corpus").expect("--corpus");
    let server = flag("--server");
    let maps: std::collections::HashMap<String, String> = match flag("--maps") {
        Some(p) => std::fs::read_to_string(&p)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                if f.len() >= 2 { Some((f[0].to_string(), f[1].to_string())) } else { None }
            })
            .collect(),
        None => Default::default(),
    };

    // group the corpus by map
    let mut by_map: std::collections::BTreeMap<String, Vec<std::path::PathBuf>> = Default::default();
    let mut dirs = vec![std::path::PathBuf::from(&corpus)];
    while let Some(d) = dirs.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().map_or(false, |n| n == ".git") {
                    continue;
                }
                dirs.push(p);
            } else if p.to_string_lossy().ends_with(".Ghost.Gbx") {
                by_map.entry(map_of(&p)).or_default().push(p);
            }
        }
    }

    println!("map\tfile_a\tfile_b\tn_common\tn_ident\tlongest_run\tmax_sep_m\tinputs\tverdict");
    let mut refused = 0usize;
    // Warn once, loudly, if the input comparison cannot run at all.
    let mut warned_unavailable = false;
    // Pairs whose "identical positions across differing inputs" could not be
    // attributed to inert inputs or to a splice from the files alone.
    let mut unresolved = 0usize;
    for (map, files) in &by_map {
        if files.len() < 2 {
            continue;
        }
        let _mapid: String = map.chars().take_while(|c| c.is_ascii_digit()).collect();
        let mut files = files.clone();
        files.sort();
        // decode once per file
        let dec: Vec<Option<Vec<R>>> =
            files.iter().map(|f| decode(&f.to_string_lossy()).ok()).collect();
        // the input key is computed PER PAIR, over the identical run
        let inputs: Vec<Option<String>> = vec![None; files.len()];
        let _ = &inputs;
        let _ = &server;
        let _ = &maps;
        for i in 0..files.len() {
            for j in i + 1..files.len() {
                let (Some(a), Some(b)) = (&dec[i], &dec[j]) else { continue };
                let p = pair(a, b);
                if p.n < 5 {
                    continue;
                }
                let nid = p.ident.iter().filter(|x| **x).count();
                // NOT "every sample identical". 134672 publishes two files that
                // share 986 CONSECUTIVE bit-identical positions -- 49.3 s of
                // driving agreeing to the last f32 bit -- out of 1349, and an
                // all-or-nothing rule passed them. What makes a pair one run is
                // a RUN of identical samples long enough that coincidence is
                // not an explanation, at ANY alignment: the same reasoning as
                // `intg lag`, and it subsumes the 100 % case.
                let hits = lag_scan(a, b, 20.min(p.n));
                let (best_run, best_lag, best_free_run) = hits
                    .first()
                    .map_or((0, 0, 0), |h| (h.best_run, h.lag, h.best_free_run));
                let all_ident = best_run >= 10;
                // Reported, not acted on: see LagHit::best_free_run. If these
                // two differ the identical run began at a shared restore point,
                // which is what a respawn does and not what a graft does.
                let free_note = if best_free_run < best_run {
                    format!(" (only {} of it unanchored by a respawn)", best_free_run)
                } else {
                    String::new()
                };
                // THE INPUT KEY IS TAKEN OVER THE IDENTICAL RUN ITSELF, not over
                // the whole recorded window. Two of our tapes seeded from one
                // human tape share a DETERMINISTIC PREFIX -- identical f32
                // positions for as long as their inputs agree -- and that is
                // honest. What is impossible is identical positions across a
                // span where the inputs DIFFER. Keyed on the whole window the
                // run rule produced 284 refusals, nearly all of them shared
                // prefixes; keyed on the run itself, only the impossible ones
                // remain. A non-zero best lag means the two files come from
                // different recording sessions, where no input window is
                // comparable -- and a long identical run there is already
                // impossible, so no key is needed.
                // A DIFFERENCE MUST HAVE HAD TIME TO PROPAGATE. It is not
                // enough that the inputs differ somewhere inside the span where
                // the positions agree: a difference in the LAST few ticks of
                // that span has not moved the car yet. 126859 publishes two
                // tapes that are input-identical until race 18.900 and
                // bit-identical in position through exactly 18.900 --
                // determinism, working. A span digest calls that a splice, and
                // did, on 163 pairs of mine. The honest quantity is the FIRST
                // instant the inputs differ: positions may agree up to there and
                // a little past it; agreeing far beyond it cannot happen.
                let run_span = if all_ident && best_lag == 0 {
                    find_run_span(a, b, best_lag, best_run)
                } else {
                    None
                };
                // How far past the first input divergence the positions may still
                // agree before the agreement is impossible. Two bands, because
                // the evidence is not the same strength at both ends: every
                // splice this corpus actually contains overshoots by tens or
                // hundreds of seconds (227654 by 16.4 s, 238835 by 111 s,
                // 286279 by 403 s), while a sub-second overshoot has an honest
                // explanation available -- an input that is INERT where it
                // differs, which on a Trial map with a wedged car is common.
                // Refusing those would be the C8 mistake again: a gate that
                // refuses genuine files teaches people to override gates.
                const REVIEW_MS: i64 = 150;
                const PROPAGATE_MS: i64 = 2000;
                let (inp, verdict) = if !all_ident {
                    ("-".to_string(), "DISTINCT".to_string())
                } else if best_lag != 0 {
                    // different recording sessions: no input window is
                    // comparable, and a long identical run is already impossible
                    refused += 1;
                    ("cross-session".to_string(), "REFUSE-ONE-RUN-TWICE".to_string())
                } else {
                    // THE FIRST DIFFERING TICK IS NOT THE DIVERGENCE. A single tick
                    // inside a no-authority window carries no information -- the
                    // car is airborne, or wedged, or the value is inert -- so a
                    // differencer that treats all ticks alike manufactures
                    // distinctions. Measured on 165922: TAS_15240 differs from
                    // TAS_15290 at exactly ONE inert tick at 2.250 s and then not
                    // again until 3.37 s, and keying on 2.250 made a pair whose
                    // positions legitimately agree to 4.350 s read as one run
                    // published twice.
                    //
                    // Key on the first SUSTAINED divergence instead: the first
                    // tick after which the tapes keep differing.
                    match first_sustained_diff_checked(
                        &files[i].to_string_lossy(),
                        &files[j].to_string_lossy(),
                    ) {
                        // THE INSTRUMENT IS NOT AVAILABLE. Unchecked is a
                        // refusal, not a pass -- see `TapeDiff`. Before this,
                        // a missing `fk` made every pair in the corpus read
                        // `identical-tapes / EXPECTED-SAME-INPUTS`.
                        TapeDiff::Unavailable(why) => {
                            refused += 1;
                            if !warned_unavailable {
                                warned_unavailable = true;
                                eprintln!("tmtraj corpus dup: the input comparison is UNAVAILABLE -- {why}");
                                eprintln!("  every pair is reported UNCHECKED-NO-TAPEDIFF, not clean.");
                                eprintln!("  `fk` is excluded from this workspace: build it from tools/fk and");
                                eprintln!("  put it beside this binary or on PATH, then re-run.");
                            }
                            (
                                "unavailable".to_string(),
                                "UNCHECKED-NO-TAPEDIFF".to_string(),
                            )
                        }
                        TapeDiff::Identical => (
                            "identical-tapes".to_string(),
                            "EXPECTED-SAME-INPUTS".to_string(),
                        ),
                        TapeDiff::At(d) => match run_span {
                            // ONE RUN TWICE IS A STATEMENT ABOUT THE RECORDED
                            // MOTION, AND IT IS EXACT: the two files carry the
                            // same trajectory. If the trajectories separate
                            // ANYWHERE, they are two different recorded
                            // motions and the pair cannot be one run published
                            // twice, whatever the tapes did.
                            //
                            // Adjudicated over the whole corpus 2026-08-22, and
                            // the split is not a threshold — it is a gap:
                            // three pairs at max_sep EXACTLY 0.000000 m, and
                            // the next one at 0.44 m.
                            //
                            //   0.000000  186935 CUT_795034 / ONE_ATTEMPT_DELETED
                            //   0.000000  227654 TAS_57518  / TAS_57537
                            //   0.000000  227654 TAS_57518  / TAS_57577
                            //   0.440097  145875 BEST_KEYBOARD_6323 / KEYBOARD_23ev_6323
                            //
                            // POSITIVE CONTROL FOR THIS RULE: the 227654 page
                            // already says, by hand, that TAS_57537 and
                            // TAS_57577 are "one trajectory, not two runs".
                            // The rule rediscovers a defect the corpus had
                            // documented independently — and flags TAS_57518
                            // as a third member of that set.
                            //
                            // The other 43 all separate, most of them by
                            // metres. They are NOT duplicates; see below.
                            // THE ADJUDICATION, 2026-08-22. This arm used to
                            // refuse outright, and 46 pairs landed in it. The
                            // question it could not answer is WHY two files
                            // hold bit-identical positions across a stretch
                            // where their tapes disagree, and there are two
                            // answers with opposite meanings:
                            //
                            //   (a) the differing inputs had NO AUTHORITY
                            //       there -- countdown, wedge, ballistic
                            //       flight, SpecialNoSteering road. Innocent.
                            //   (b) one file is CARRYING THE OTHER'S RECORDING
                            //       for that stretch. A defect.
                            //
                            // Separation later does NOT decide it -- a partial
                            // splice separates too, and an early version of
                            // this rule excused 38 pairs on exactly that
                            // mistake. What is reported instead is the count of
                            // differing ticks INSIDE the identical stretch,
                            // which is the quantity the two answers disagree
                            // about, and the verdict names the open question
                            // rather than picking a side.
                            //
                            // Worked example of why (a) alone will not do:
                            // 203330 is the ONE map with a per-tick authority
                            // map, measured by overwriting one tick at a time
                            // (race -1.500..2.970 inert, 2.980..3.330 live).
                            // Its `an330_13984` / `kb330_12ev_13986` pair has
                            // 72 differing ticks in the inert window -- and
                            // EIGHT in the live one, while the positions stay
                            // bit-identical to 5.000 s. So even there the
                            // innocent story does not cover the evidence.
                            // 126859's `ALPHABET153_23545` / `TAS_23416` is
                            // starker: 335 ticks of throttle difference across
                            // the tapes and 378 consecutive bit-identical
                            // samples, 18.9 s of them.
                            //
                            // SETTLING IT needs the map: re-simulate each
                            // tape and ask whether the engine reproduces that
                            // file's own record over the shared stretch
                            // (`fk btraj2`, the C-route check). This repo does
                            // not redistribute maps, so it cannot be done from
                            // a clean checkout -- which is why this is an open
                            // task with a named test and not a verdict.
                            Some((_, t1)) if t1 > d + PROPAGATE_MS && p.max_d > 0.0 => {
                                let inside = tape_diffs_in_window(
                                    &files[i].to_string_lossy(),
                                    &files[j].to_string_lossy(),
                                    d,
                                    t1,
                                )
                                .unwrap_or(0);
                                unresolved += 1;
                                (
                                    format!(
                                        "diverge@{:.3}s,identical_to@{:.3}s,{} differing ticks INSIDE,sep={:.3}m",
                                        d as f64 / 1000.0,
                                        t1 as f64 / 1000.0,
                                        inside,
                                        p.max_d
                                    ),
                                    "UNRESOLVED-INERT-OR-SPLICE".to_string(),
                                )
                            }
                            Some((_, t1)) if t1 > d + PROPAGATE_MS => {
                                refused += 1;
                                (
                                    format!(
                                        "diverge@{:.3}s,agree_to@{:.3}s,over={:.3}s,sep=0",
                                        d as f64 / 1000.0,
                                        t1 as f64 / 1000.0,
                                        (t1 - d) as f64 / 1000.0
                                    ),
                                    "REFUSE-ONE-RUN-TWICE".to_string(),
                                )
                            }
                            Some((_, t1)) if t1 > d + REVIEW_MS => (
                                format!(
                                    "diverge@{:.3}s,agree_to@{:.3}s,over={:.3}s",
                                    d as f64 / 1000.0,
                                    t1 as f64 / 1000.0,
                                    (t1 - d) as f64 / 1000.0
                                ),
                                "REVIEW-SHORT-OVERSHOOT".to_string(),
                            ),
                            Some(_) => (
                                format!("diverge@{:.3}s", d as f64 / 1000.0),
                                "EXPECTED-SHARED-PREFIX".to_string(),
                            ),
                            None => {
                                refused += 1;
                                (
                                    "unchecked".to_string(),
                                    "IDENTICAL-INPUTS-UNCHECKED".to_string(),
                                )
                            }
                        },
                    }
                };
                if all_ident {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
                        map,
                        files[i].file_name().unwrap().to_string_lossy(),
                        files[j].file_name().unwrap().to_string_lossy(),
                        p.n,
                        nid,
                        best_run,
                        p.max_d,
                        inp,
                        format!("{}{}", verdict, free_note)
                    );
                }
            }
        }
    }
    if unresolved > 0 {
        eprintln!();
        eprintln!(
            "{unresolved} pair(s) UNRESOLVED-INERT-OR-SPLICE: bit-identical positions across a"
        );
        eprintln!("  stretch where the tapes disagree. Innocent if those inputs had no authority");
        eprintln!("  there; a splice if one file carries the other's recording. The count of");
        eprintln!("  differing ticks INSIDE the identical stretch is on each row. Settle it by");
        eprintln!("  re-simulating each tape and asking whether the engine reproduces that");
        eprintln!("  file's own record over the shared stretch -- which needs the map.");
    }
    std::process::exit(if refused > 0 { 2 } else { 0 });
}



// ===========================================================================
// `intg lag` -- alignment-free comparison, for a reference from ANOTHER
// recording session.
//
// WHY THE TIME-ALIGNED TEST IS NOT ENOUGH. `intg pair` matches samples by their
// recorded instant, which is right when both files descend from the same
// carrier -- that is how the whole-file donor copies were found. It is USELESS
// against an independent recording: a ghost's sample times are SESSION times,
// not race times, so 227654's rank-2 human recording labels its samples
// 93680.. while our files label theirs 1310.. . Zero instants in common, and
// the honest verdict is NO-COMPARISON, not CLEAN. Nine files came back that
// way and none of them was tested by it.
//
// This is the reference-free form, and it is the rule the `nan` arm arrived at
// from the other direction: DO NOT ASK WHETHER TWO RUNS ARE CLOSE, ASK WHETHER
// ONE IS THE OTHER, SHIFTED. Slide the file along the reference over every
// integer sample lag and count BIT-IDENTICAL positions. A copy of another run
// -- however its timestamps were relabelled -- lands on a lag where the count
// is large. Two genuinely different drives never do: the same twelve f32 bytes
// by coincidence, repeatedly, does not happen.
//
// It costs O(n*m) position compares, which for 365 x 8550 is nothing.
// ===========================================================================

pub struct LagHit {
    pub lag: i64,
    pub ident: usize,
    pub best_run: usize,
    /// The longest identical run that is NOT anchored at a shared restore
    /// point. A respawn puts both cars at the same trigger, so a run that
    /// begins there is what the game does, not what a graft does -- and this is
    /// the number that separates them. It is REPORTED beside `best_run` rather
    /// than substituted for it: changing which one the rule reads is a change
    /// to who gets refused, and that needs a control this arm did not run.
    pub best_free_run: usize,
}

/// Bit-identical position counts at every integer lag with enough overlap.
pub fn lag_scan(a: &[R], b: &[R], min_overlap: usize) -> Vec<LagHit> {
    fn key(r: &R) -> Option<&[u8]> {
        if r.raw.len() >= 59 { Some(&r.raw[47..59]) } else { None }
    }
    let mut out = Vec::new();
    let lo = -(b.len() as i64) + min_overlap as i64;
    let hi = a.len() as i64 - min_overlap as i64;
    for lag in lo..=hi {
        let mut ident = 0usize;
        let mut overlap = 0usize;
        let mut run = 0usize;
        let mut best_run = 0usize;
        let mut best_free_run = 0usize;
        let mut run_start = 0usize;
        for i in 0..a.len() {
            let j = i as i64 - lag;
            if j < 0 || j as usize >= b.len() {
                continue;
            }
            overlap += 1;
            // A STATIONARY CAR IS BIT-IDENTICAL IN EVERY RECORDING OF THE MAP.
            // Every run starts on the same spawn, held still, so the pad
            // samples match at EVERY lag -- and a long enough countdown makes
            // that look like a shared trajectory. Measured: 238835's embedded
            // author ghost scored 23 consecutive identical positions against
            // rank 1 at a lag of -1410 s, which is two cars parked. A shared
            // trajectory means shared MOTION, so a sample only counts when the
            // car has moved since the previous one.
            let moved = i > 0 && {
                let mut d = 0.0f64;
                for k in 0..3 {
                    let q = a[i].pos[k] - a[i - 1].pos[k];
                    d += q * q;
                }
                d > 1e-6
            };
            match (key(&a[i]), key(&b[j as usize])) {
                (Some(x), Some(y)) if x == y && moved => {
                    ident += 1;
                    if run == 0 {
                        run_start = i;
                    }
                    run += 1;
                    best_run = best_run.max(run);
                    // The restore can sit INSIDE the run as well as before it: the
                    // last pre-respawn sample often matches too (both cars are
                    // at the trigger). So a run is anchored if a gap occurs
                    // anywhere from just before it to here.
                    if !anchored_within(a, run_start, i, 50) {
                        best_free_run = best_free_run.max(run);
                    }
                }
                (Some(x), Some(y)) if x == y => {}
                _ => run = 0,
            }
        }
        if overlap >= min_overlap && ident > 0 {
            out.push(LagHit { lag, ident, best_run, best_free_run });
        }
    }
    out.sort_by(|x, y| y.ident.cmp(&x.ident));
    out
}

/// Is an identical run ANCHORED AT A SHARED RESTORE POINT?
///
/// THE CORRECTION THIS ENCODES. "An identical block preceded by real
/// separation is a splice" assumes that once two runs are metres apart their
/// states differ for good. A RESPAWN BREAKS THAT ASSUMPTION: it restores the
/// car to a fixed checkpoint state, so two runs that were far apart are
/// suddenly in the SAME state, and from there identical inputs give
/// bit-identical positions -- the head-anchored determinism case, anchored at
/// a respawn instead of at the start.
///
/// Measured on 238835, a Trial map: the map's own embedded AUTHOR ghost and
/// the rank-1 human recording share 23 consecutive bit-identical positions
/// (1.15 s) beginning at the sample straight after a 1050 ms gap in the
/// author's record -- a respawn to (1044.8, 106, 624), from which both cars
/// accelerate away identically. Two unrelated humans, no contamination.
///
/// A respawn shows as a gap in the record's own sample times, so this needs
/// nothing but the timestamps.
fn anchored_within(a: &[R], start: usize, end: usize, grid_ms: i64) -> bool {
    // `start <= 1`, not `== 0`: a run only counts samples where the car MOVED,
    // and sample 0 has no predecessor to have moved from, so a head-anchored
    // block always begins at index 1. Testing for 0 made every determinism
    // prefix read as unanchored -- `intg lag` called 145875 BEST_6322
    // CONTAMINATED on a 27-sample block that `intg pair`, on the same pair of
    // files, correctly reports as HEAD samples [0..27] and CLEAN.
    if start <= 1 {
        return true; // the race start is the original shared restore point
    }
    for k in start..=end.min(a.len() - 1) {
        if k == 0 {
            continue;
        }
        if a[k].ms - a[k - 1].ms > grid_ms * 2 {
            return true;
        }
    }
    false
}



/// The first race instant at which two tapes' decoded inputs differ.
/// `None` when the two tapes are identical, or the answer is unavailable.
/// **THE INSTRUMENT MUST NOT FAIL TOWARD CLEAN.**
///
/// `fk tapediff` is a separate binary — `tools/fk` is excluded from this
/// workspace and is built from its own directory — so on any box where the
/// workspace has been built but `fk` has not, every one of these calls used to
/// return `None`, and `None` means *the tapes are identical*. The whole corpus
/// then reads `identical-tapes / EXPECTED-SAME-INPUTS`: the duplicate detector
/// excuses every pair it exists to catch, and reports **0 refusals**.
///
/// Measured on 228607, 2026-08-22, with `fk` absent: `SPLICE_24854` and
/// `TAS_19907` were reported `identical-tapes` while their trajectories are
/// **357 m** apart and `ghost tape diff` puts their first input difference at
/// tick 72. Nine pairs on that map, all excused.
///
/// This is the fourth instance in this project of *an instrument fell silent
/// and the pipeline read the silence as a clean result*, and the second one
/// caused specifically by `fk` not being on `PATH` (`tools/search/SEARCH.md`
/// records the first, where 24 attempts "failed to find the car"). The port
/// from shell to Rust carried the bug across, because `.ok()?` is `2>/dev/null`
/// with a nicer spelling.
///
/// So: three outcomes, and the caller must handle all three.
pub enum TapeDiff {
    /// `fk` ran and found the first sustained divergence here.
    At(i64),
    /// `fk` ran and the tapes really are identical.
    Identical,
    /// `fk` could not be run, or said something we could not parse. **Not
    /// clean.**
    Unavailable(String),
}

/// How many ticks differ between two tapes STRICTLY INSIDE a race-time window.
///
/// This is the number that adjudicates a `dup` refusal, and neither the first
/// differing tick nor the eventual separation can stand in for it:
///
/// * **0 differences inside the identical stretch** — an ordinary shared
///   prefix. The tapes part at the end of it and the trajectories follow.
///   Nothing to explain.
/// * **many differences inside it** — the two files hold bit-identical
///   positions across a stretch where their inputs disagree. That is either
///   inputs with no authority (a countdown, a wedge, a ballistic flight, a
///   `SpecialNoSteering` road) **or one file carrying the other's recording for
///   that stretch**. Those two look identical from here and the distinction
///   matters, so this function reports the count and refuses to guess.
pub fn tape_diffs_in_window(a: &str, b: &str, from_ms: i64, to_ms: i64) -> Result<usize, String> {
    let ta = gbx::tape::Tape::from_file(a).map_err(|e| format!("{a}: {e}"))?;
    let tb = gbx::tape::Tape::from_file(b).map_err(|e| format!("{b}: {e}"))?;
    let oa = ta.archives.first().map(|x| x.start_offset_ms).unwrap_or(0) as i64;
    let (sa, sb) = (ta.steer_i8s(), tb.steer_i8s());
    let (aa, ab) = (ta.accels(), tb.accels());
    let (ba, bb) = (ta.brakes(), tb.brakes());
    // A RESPAWN IS AN INPUT -- bit 31 of the packet state literal, and this
    // project edits it deliberately (it can pin a finish time). Leaving it out
    // makes two tapes differing only in a respawn read as identical, which on
    // the trial maps here is the difference that matters most.
    let (ra, rb) = (ta.respawns(), tb.respawns());
    let mut n = 0usize;
    for i in 0..sa.len().min(sb.len()) {
        let race = i as i64 * 10 + oa;
        if race < from_ms.max(0) || race > to_ms {
            continue;
        }
        if sa[i] != sb[i] || aa.get(i) != ab.get(i) || ba.get(i) != bb.get(i) || ra.get(i) != rb.get(i) {
            n += 1;
        }
    }
    Ok(n)
}

fn run_tapediff(a: &str, b: &str, from_ms: Option<i64>) -> Result<Option<i64>, String> {
    // IN PROCESS, ON PURPOSE. This used to shell out to `fk tapediff`, and
    // `fk tapediff` DOES NOT EXIST — it is not a command this repo's `fk` has,
    // at any build. So the call failed every time, `.ok()?` turned the failure
    // into `None`, and `None` means "the tapes are identical". The comparison
    // has therefore never run in this lineage, and every `corpus dup` over
    // this corpus reported `identical-tapes / EXPECTED-SAME-INPUTS` for all
    // 117 pairs and exited 0.
    //
    // `gbx::tape` is already a dependency of this crate and reads the same
    // input chunk `ghost tape diff` reads, so there is no reason for a
    // subprocess, a `PATH`, or a failure mode at all.
    let ta = gbx::tape::Tape::from_file(a).map_err(|e| format!("{a}: {e}"))?;
    let tb = gbx::tape::Tape::from_file(b).map_err(|e| format!("{b}: {e}"))?;
    let (oa, _ob) = (
        ta.archives.first().map(|x| x.start_offset_ms).unwrap_or(0) as i64,
        tb.archives.first().map(|x| x.start_offset_ms).unwrap_or(0) as i64,
    );
    let (sa, sb) = (ta.steer_i8s(), tb.steer_i8s());
    let (aa, ab) = (ta.accels(), tb.accels());
    let (ba, bb) = (ta.brakes(), tb.brakes());
    // A RESPAWN IS AN INPUT -- bit 31 of the packet state literal, and this
    // project edits it deliberately (it can pin a finish time). Leaving it out
    // makes two tapes differing only in a respawn read as identical, which on
    // the trial maps here is the difference that matters most.
    let (ra, rb) = (ta.respawns(), tb.respawns());
    if sa.is_empty() || sb.is_empty() {
        return Err(format!("no input archive in {}", if sa.is_empty() { a } else { b }));
    }
    // Compare by RACE TIME, not by tick index: two tapes with different
    // start offsets are not aligned by index, and 203072's is -1560 ms.
    //
    // AND SKIP THE COUNTDOWN. `start_offset_ms` is negative on every tape
    // here, so ticks before race 0 are input the car cannot act on — it is on
    // the start line and held. Counting them made the first run of this
    // comparison report **35 REFUSE-ONE-RUN-TWICE** across the corpus, nearly
    // all of them `diverge@-1.52s` — that is not a divergence, it is two
    // drivers holding different keys during the lights. The same mistake the
    // comment above this function warns about, made one window earlier.
    let n = sa.len().min(sb.len());
    for i in 0..n {
        let race = i as i64 * 10 + oa;
        if race < 0 {
            continue;
        }
        if let Some(f) = from_ms {
            if race < f {
                continue;
            }
        }
        if sa[i] != sb[i] || aa.get(i) != ab.get(i) || ba.get(i) != bb.get(i) || ra.get(i) != rb.get(i) {
            return Ok(Some(race));
        }
    }
    // One tape longer than the other is a difference at the point it runs out.
    if sa.len() != sb.len() {
        let race = n as i64 * 10 + oa;
        if from_ms.is_none_or(|f| race >= f) {
            return Ok(Some(race));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tapediff_control {
    use super::run_tapediff;

    fn repo(p: &str) -> String {
        format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), p)
    }

    /// THE POSITIVE CONTROL FOR THIS COMPARISON.
    ///
    /// A null from `run_tapediff` means "these tapes are identical", and the
    /// whole duplicate detector reads that as clean — so a version of it that
    /// cannot see any difference passes every corpus silently. That is exactly
    /// what shipped: it called `fk tapediff`, which is not a command `fk` has,
    /// and returned `None` every time.
    ///
    /// So the test is in two halves and BOTH have to hold: a file against
    /// itself must come back identical (or the comparison is inventing
    /// differences), and two files that are known to be different runs must
    /// come back with a divergence (or it is blind).
    #[test]
    fn identical_to_itself_and_different_from_a_different_run() {
        let a = repo("145875-unlucke-get-jiggy-with-it/replays/tas_6330.Ghost.Gbx");
        let b = repo("145875-unlucke-get-jiggy-with-it/replays/tas_6333.Ghost.Gbx");
        if !std::path::Path::new(&a).exists() {
            return; // corpus not beside the tools; nothing to control against
        }
        assert_eq!(
            run_tapediff(&a, &a, None),
            Ok(None),
            "a tape must be identical to itself"
        );
        let d = run_tapediff(&a, &b, None);
        assert!(
            matches!(d, Ok(Some(_))),
            "two runs 3 ms apart must differ somewhere; got {d:?} -- \
             a blind comparison reports every pair in the corpus as clean"
        );
    }
}

pub fn first_input_diff(a: &str, b: &str) -> Option<i64> {
    run_tapediff(a, b, None).ok().flatten()
}

/// The file's own declared race time, in ms.
///
/// Needed because THE CONTAMINATION WINDOW IS THE RACE. Two independent runs
/// that share a carrier agree exactly in the carrier's TAIL -- neither is
/// driving there, so the samples are the same graft. Measured on genuine
/// rank-1 downloads: 199100 shares 28 consecutive positions and 228811 shares
/// 38, and BOTH blocks sit entirely past the finish (50.850..52.200 against a
/// race of 50.738; 20.820..22.670 against 20.237), with ZERO identical samples
/// in race, 0 of 1015 and 0 of 405. That is a tail problem, which C3 and C4
/// already flag and which the tail cut removes -- not a provenance problem.
/// Counting it as contamination inflates the finding and, worse, spends
/// somebody's evening on a benign file.
pub fn declared_race_ms(path: &str) -> Option<i64> {
    let d = record::decode_ghost(path).ok()?;
    d.race_time_ms
        .map(|v| v as i64)
        .or_else(|| d.checkpoints_ms.last().map(|v| *v as i64))
}

/// Keep only the samples inside the race.
pub fn in_race(r: &[R], race_ms: Option<i64>) -> Vec<R> {
    match race_ms {
        // +60 ms of slack: the finish sample itself is the run's, and the
        // 50 ms grid does not land exactly on the declared millisecond.
        Some(t) => r.iter().filter(|x| x.ms <= t + 60).cloned().collect(),
        None => r.to_vec(),
    }
}

// ===========================================================================
// C3b -- IS THAT JUMP A TELEPORT, OR IS THE CAR SIMPLY FAST?
//
// The distance-based C3 asks whether a step is far beyond the run's own step
// distribution. On a map where the car is genuinely fast that is a false
// refusal: 126859's biggest step implies 223.7 m/s and the car's own
// speedometer reads 223.4 m/s -- 805 km/h on a Kacky map, and entirely real.
// A bar in metres cannot tell "fast" from "moved without travelling".
//
// The file already carries the answer. Every sample has the engine's own
// recorded speed, so compare the displacement the positions imply with the
// speed the car says it had:
//
//   ratio ~ 1.00                real driving, at any speed
//   recorded speed exactly 0.0  a RESPAWN -- normal gameplay, and the whole
//                               point of a Trial map (267460, 238835, 286279,
//                               186935 are all Trial-shaped)
//   ratio in the thousands      a SPLICE (227654's TAS_57503: 50 090 m/s
//                               implied against 19.2 recorded)
//
// THE SPEEDOMETER IS NOT ALWAYS ZEROED ON A RESPAWN -- on at least one file it
// holds its pre-respawn value, so a rule that only tests "speed == 0" misses
// it and calls a respawn a splice. The robust discriminator needs no speed at
// all: after a respawn the car RESUMES DRIVING from where it landed, so the
// steps in the second following the jump are ordinary. After a splice the
// record simply continues in another run, and the jump is one discontinuity
// in a stream that was never interrupted. So: measure the jump against the
// motion that FOLLOWS it, from positions and times alone.
// ===========================================================================

pub struct JumpVerdict {
    pub ms: i64,
    pub dist: f64,
    pub recorded: f64,
    pub kind: &'static str,
}

/// Every step whose implied speed is far above what the car reports, classified.
pub fn jumps(r: &[R]) -> Vec<JumpVerdict> {
    let mut out = Vec::new();
    for i in 1..r.len() {
        let dt = (r[i].ms - r[i - 1].ms) as f64 / 1000.0;
        if dt <= 0.0 {
            continue;
        }
        let mut d2 = 0.0;
        for k in 0..3 {
            let q = r[i].pos[k] - r[i - 1].pos[k];
            d2 += q * q;
        }
        if !d2.is_finite() {
            continue;
        }
        let dist = d2.sqrt();
        let implied = dist / dt;
        // the faster of the two endpoints' own speedometers
        let recorded = r[i].speed.max(r[i - 1].speed);
        // a step the car's own speed explains is not a jump at all
        if implied <= recorded.max(1.0) * 1.5 + 2.0 {
            continue;
        }
        // ...nor is a step that goes nowhere. Without an absolute floor a 0.20 m
        // shuffle at 0.8 m/s trips the ratio and gets classified as a splice.
        if dist < 5.0 {
            continue;
        }
        // THE WORLD ORIGIN IS NOT A PLACE THE CAR WAS. A first sample at
        // (0,0,0) is a placeholder -- `ghostqc`'s ORIGIN class -- and the step
        // away from it is the record starting, not the car moving.
        let at_origin = |p: &[f64; 3]| p.iter().all(|v| v.abs() < 1e-6);
        if at_origin(&r[i - 1].pos) || at_origin(&r[i].pos) {
            out.push(JumpVerdict {
                ms: r[i].ms,
                dist,

                recorded,

                kind: "ORIGIN PLACEHOLDER (a sample at the world origin, not a position)",
            });
            continue;
        }
        // what does the car do in the SECOND AFTER the jump? Median step speed
        // over the following 20 samples, positions and times only.
        let mut after: Vec<f64> = Vec::new();
        for j in i + 1..(i + 21).min(r.len()) {
            let dt2 = (r[j].ms - r[j - 1].ms) as f64 / 1000.0;
            if dt2 <= 0.0 {
                continue;
            }
            let mut s2 = 0.0;
            for k in 0..3 {
                let q = r[j].pos[k] - r[j - 1].pos[k];
                s2 += q * q;
            }
            after.push(s2.sqrt() / dt2);
        }
        after.sort_by(|a, b| a.total_cmp(b));
        let after_med = if after.is_empty() { f64::NAN } else { after[after.len() / 2] };
        // THE DISCRIMINATOR IS WHERE THE CAR LANDS, not how fast it is going.
        // "Ordinary motion resumes after the jump" does NOT separate the two --
        // measured on 238835, a respawn lands and drives away at 9.4 m/s, and
        // so does the far side of a splice. What only a respawn does is RETURN
        // THE CAR SOMEWHERE IT HAS ALREADY BEEN: a checkpoint it crossed
        // earlier in this same record. A splice puts it on another run's path,
        // which this record has never visited. Positions only, no speed.
        let landed = &r[i].pos;
        let mut been_there = false;
        for k in 0..i.saturating_sub(2) {
            let mut s2 = 0.0;
            for c in 0..3 {
                let q = r[k].pos[c] - landed[c];
                s2 += q * q;
            }
            if s2 < 100.0 {
                been_there = true;
                break;
            }
        }
        // 153-213 m/s implied over 8-11 m, and the car STOPPED on the far side:
        // that is a respawn placing the car and holding it, which is ordinary
        // gameplay on a Trial map. A splice into another run continues at that
        // run's pace -- it does not arrive at a standstill. (This is the class
        // the old 200 m/s distance bar sat inside, which is why it refused
        // HUMANCUT_236972, a genuine human run.)
        let stopped_after = after_med.is_finite() && after_med < 1.0;
        let kind = if recorded < 0.5 {
            "RESPAWN (the car's own speed is zero)"
        } else if stopped_after {
            "RESPAWN (the car arrives at a standstill)"
        } else if been_there {
            // the speedometer is not always zeroed on a respawn -- on at least
            // one file it holds its pre-respawn value -- so this branch is what
            // catches those.
            "RESPAWN (the car returns to a point it already occupied)"
        } else {
            "SPLICE (the record continues somewhere this run has never been)"
        };
        let _ = (implied, after_med);
        out.push(JumpVerdict { ms: r[i].ms, dist, recorded, kind });
    }
    out
}



pub fn new_cache() -> Cache {
    Cache::new()
}

/// What the instruments say about a file TODAY: the verdict word, the
/// references actually tested (path + md5), and the limits that apply.
///
/// Both the time-aligned and the alignment-free test are run against every
/// reference held, because they answer different questions: `pair` finds a
/// shared carrier, `lag` finds a copy of another recording session. A file
/// that passes one and not the other is not clean.
pub fn certify_now(
    ghost: &str,
    humans: &[&Ref],
    race_ms: i64,
    cache: &mut Cache,
) -> (String, Vec<(String, String)>, Vec<String>) {
    let mapid = humans.first().map(|r| r.map.clone()).unwrap_or_default();
    let mut limits: Vec<String> = Vec::new();
    let a = match decode(ghost).map(|v| in_race(&v, Some(race_ms).filter(|t| *t > 0))) {
        Ok(v) => v,
        Err(e) => {
            limits.push(format!(
                "THIS FILE HAS NO READABLE VEHICLE RECORD ({}). It cannot be checked and it \
                 cannot be filmed: there is no car in it.",
                e
            ));
            return ("UNCERTIFIED".into(), Vec::new(), limits);
        }
    };
    if humans.is_empty() {
        limits.push(crate::manifest::limit_text("no-human-reference", &mapid));
        return ("UNCERTIFIED".into(), Vec::new(), limits);
    }
    let mut tested: Vec<(String, String)> = Vec::new();
    let mut worst = "CLEAN".to_string();
    for r in humans {
        let Some(b) = cached(cache, &r.path) else { continue };
        tested.push((
            r.path.clone(),
            md5_hex(&std::fs::read(&r.path).unwrap_or_default()),
        ));
        let p = pair(&a, &b);
        match verdict(&p, 5.0) {
            Verdict::Contaminated => worst = "CONTAMINATED".into(),
            Verdict::Identical if worst != "CONTAMINATED" => {
                worst = "CONTAMINATED".into();
            }
            _ => {}
        }
        // alignment-free: sample times are session times, so a copy of another
        // recording shares no instants and `pair` cannot see it at all.
        if worst != "CONTAMINATED" {
            let hits = lag_scan(&a, &b, 20.min(a.len().max(1)));
            if hits.first().map_or(0, |h| h.best_run) >= 10 {
                worst = "CONTAMINATED".into();
            }
        }
        if p.runs.first().map_or(false, |x| x.i0 == 0 && x.n() >= 10) {
            limits.push(crate::manifest::limit_text("head-anchored-block", &mapid));
        }
    }
    if tested.is_empty() {
        limits.push(crate::manifest::limit_text("no-human-reference", &mapid));
        return ("UNCERTIFIED".into(), Vec::new(), limits);
    }
    limits.sort();
    limits.dedup();
    (worst, tested, limits)
}



// ===========================================================================
// E -- THE STALE-BUFFER CHECK. Is this file the LIVE car, or a copy of it one
// tick in the past?
//
// The engine double-buffers the vehicle state and BOTH COPIES PASS EVERY
// STRUCTURAL TEST: the stale one is finite, moves, is self-consistent
// (d(pos)/dt matches the velocity stored beside it), has a unit quaternion,
// re-simulates to the exact declared millisecond, and is 100 % covered. A
// regeneration that locks onto it produces a file that passes coverage, the
// oracle, the manifest, C2 AND the contamination test, and is still wrong.
//
// Measured on three unrelated maps in one pass tonight -- 285268, 249521 and
// 227969 -- each 10.0 ms from its published copy to three digits, with the
// regeneration BEHIND on 100 % of samples.
//
// The discriminator is the `nan` arm's and it needs no reference recording:
// A BUFFER HOLDS THE PAST, so the live copy is the one FURTHEST ALONG ITS OWN
// VELOCITY at the same clock value. Project (this - other) onto the car's
// velocity: consistently negative means this file is the stale one.
//
// WHY IT IS SEPARATE FROM THE CONTAMINATION TEST. That test asks whose run
// this is; this asks which copy of our own run it is. A stale file is our
// driving, correct in every claim the page makes, displayed 10 ms late -- so
// this is a REFUSAL for a file we are about to write, and NOT a reason to
// withdraw one already published. Keep those two consequences apart.
// ===========================================================================

pub struct StaleVerdict {
    pub median_m: f64,
    pub max_m: f64,
    pub implied_ms: f64,
    pub ahead: usize,
    pub behind: usize,
}

/// Compare this file against a reference of THE SAME RUN (its own source, or a
/// sibling regeneration). Only meaningful when the two are the same driving.
pub fn stale_check(a: &[R], b: &[R]) -> Option<StaleVerdict> {
    let bm: std::collections::HashMap<i64, &R> = b.iter().map(|r| (r.ms, r)).collect();
    let mut ds: Vec<f64> = Vec::new();
    let mut ahead = 0usize;
    let mut behind = 0usize;
    let mut ratio = 0.0f64;
    let mut rn = 0usize;
    for x in a {
        let Some(y) = bm.get(&x.ms) else { continue };
        let d = [x.pos[0] - y.pos[0], x.pos[1] - y.pos[1], x.pos[2] - y.pos[2]];
        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !dist.is_finite() {
            continue;
        }
        ds.push(dist);
        // project onto the reference's velocity: positive => `a` is ahead
        let sp = y.speed;
        if sp > 1.0 {
            let proj = (d[0] * y.vel[0] + d[1] * y.vel[1] + d[2] * y.vel[2]) / sp;
            if proj > 0.0 {
                ahead += 1;
            } else if proj < 0.0 {
                behind += 1;
            }
            ratio += dist / sp;
            rn += 1;
        }
    }
    if ds.len() < 5 {
        return None;
    }
    ds.sort_by(|x, y| x.total_cmp(y));
    Some(StaleVerdict {

        median_m: ds[ds.len() / 2],
        max_m: ds[ds.len() - 1],
        implied_ms: if rn == 0 { f64::NAN } else { 1000.0 * ratio / rn as f64 },
        ahead,
        behind,
    })
}


/// The rule. A file is refused when it is consistently BEHIND its reference by
/// about one tick.
/// THE PRECONDITION. This check compares two renderings of the SAME RUN and
/// asks which is one tick behind. It is MEANINGLESS when the reference is a
/// file whose telemetry is not that run's: on a contaminated source the
/// comparison measures our fresh regeneration against a STRANGER, and duly
/// reported "STALE BUFFER, behind by 8.642 ms" on 279218's repaired file,
/// which is demonstrably correct.
///
/// Two guards, because the failure was visible twice over and I read neither:
///   * the caller must state that the reference carries its own telemetry;
///   * 8.642 ms IS NOT A CLEAN TICK. The whole signature of this defect is
///     EXACTLY one physics tick on every sample. A ragged offset means "this
///     is not the defect I test for", not "a weaker version of it".
pub fn stale_verdict(v: &StaleVerdict) -> (i32, String) {
    let tot = v.ahead + v.behind;
    // Zero difference is not "too few samples to tell" -- it is the strongest
    // possible answer, and printing INDETERMINATE for it made an identity
    // control pass for the wrong reason.
    if v.max_m == 0.0 {
        return (0, "IDENTICAL to the reference -- no buffer question arises".into());
    }
    if tot < 5 {
        return (0, "INDETERMINATE (too few moving samples)".into());
    }
    let behind_frac = v.behind as f64 / tot as f64;
    // The tg control: a correct regeneration reproduces a ghost's own recorded
    // bytes to ~0.5 mm. Anything at centimetre scale or beyond is a different
    // copy of the state, not numerical noise.
    if v.median_m < 0.01 {
        return (0, format!("LIVE (median {:.6} m -- within the client-vs-server floor)", v.median_m));
    }
    // Tick-shaped or it is not this defect. The band is TIGHT on purpose: every
    // real instance measured tonight sits at 9.959, 10.004, 10.020 or 10.033 ms
    // -- a physics tick is exactly 10 -- while 279218's false positive read
    // 8.642. A loose 8..12 band admits that false positive, which is how I
    // nearly rejected a correct file.
    let ragged = !(9.4..=10.6).contains(&v.implied_ms) && v.implied_ms.is_finite();
    if behind_frac >= 0.95 && ragged {
        return (
            3,
            format!(
                "NOT THIS DEFECT -- behind on {:.0} % of samples, but by {:.3} ms, which is not one \
                 physics tick (10 ms). The stale-buffer signature is EXACTLY a tick on every \
                 sample; a ragged offset means the reference is a different run, not an earlier \
                 copy of this one. Check that the reference carries its OWN telemetry.",
                100.0 * behind_frac, v.implied_ms
            ),
        );
    }
    if behind_frac >= 0.95 {
        (
            2,
            format!(
                "STALE BUFFER -- behind on {:.0} % of samples by {:.3} ms. This is the same run, \
                 one physics tick in the past: the engine double-buffers the car state and both \
                 copies pass every other check. REFUSE and re-locate.",
                100.0 * behind_frac,
                v.implied_ms
            ),
        )
    } else if v.ahead as f64 / tot as f64 >= 0.95 {
        (
            0,
            format!(
                "AHEAD of the reference on {:.0} % of samples -- this file is the LIVE copy and \
                 the REFERENCE is the stale one",
                100.0 * v.ahead as f64 / tot as f64
            ),
        )
    } else {
        (
            0,
            format!(
                "DIFFERENT RUN or mixed ({} ahead / {} behind, median {:.3} m) -- not a buffer copy",
                v.ahead, v.behind, v.median_m
            ),
        )
    }
}

/// Bit-equality at lag 1: does `a[t]` hold exactly what `b[t-1]` held?
///
/// SHARPER THAN THE VELOCITY PROJECTION and it supersedes it as the screen.
/// The projection is a statistic -- it can tie, and it needs the car to be
/// moving fast enough for the sign to be meaningful. Bit-equality at lag 1
/// either holds or it does not. The car-state arm established that the two
/// buffers' roles NEVER SWAP (zero swaps in 2399 ticks; behind on
/// 1562/2377/2377 decidable ticks, ahead on zero), so a single-pass majority
/// is not merely evidence, it is the answer.
///
/// The projection stays as a fallback for the case this cannot serve: two
/// renderings that are near-copies but not bit-equal at any lag -- which the
/// same arm has now measured on 267460, where two of six candidate objects sit
/// 0.494 mm from the car and neither rule catches them.


// ===========================================================================
// `intg poison` -- THE DECLARED-TIME DETECTOR. One field, no simulation.
//
// A synthesised tape carries its TEMPLATE's telemetry, and the header's
// declared time comes with it. So a file whose DECLARED time is some other
// run's time, while its filename (and its tape) say ours, is carrying somebody
// else's record. Found in the store as `203330/best/an330_13984.Ghost.Gbx`:
// declares 14018 -- the human WR's time -- with 281 samples byte-identical to
// that WR, while the repo's copy of the same run declares 13984 with 280
// samples of its own telemetry. Same input tape in both.
//
// This costs one header read. It is the cheapest instrument in the set and it
// catches the poisoning AT INGEST, where every other check we have needs the
// file to reach publication first.
//
// It is a NECESSARY condition, not a sufficient one: a poisoned file whose
// donor happened to run the same time would pass. Pair it with the
// contamination test, which is the sufficient one and costs a reference.
// ===========================================================================


// ===========================================================================
// `intg selfsim` -- IS THIS TELEMETRY THIS CAR'S? (the `csvcmp` requirement)
//
// The strongest of the three screens, and the one the other two cannot reach.
// It compares a ghost's RECORDED telemetry against a live re-simulation of
// THAT SAME TAPE's own inputs. For a real recording the two are the same
// object and must agree to the noise floor: a downloaded human control reads
// 0.000407 m median. A tape carrying its TEMPLATE's telemetry reads metres --
// 19.271 m median, 138.8 m max on 146612 -- because the recorded block belongs
// to the graft carrier, not to the run the inputs describe.
//
// It is the one quantity a transplanted telemetry block cannot fake, and it
// passes nothing else: on that map `nan gate` said PASS on all five gates and
// `tail scan` returned statistics identical to the human control TO FOUR
// DECIMALS.
//
// Implemented here on top of `fk clean`, which is what this workspace has:
// clean runs the tape as an ordinary validation and samples the engine's own
// state per tick, so its positions ARE the re-simulated route.
//
// THE CONTROL IS MANDATORY AND MUST RUN IN THE SAME BATCH. My own attempt at
// this measurement earlier tonight read 12 m on a known-good file, and the
// only reason I did not report a clean corpus as contaminated is that I ran a
// downloaded human ghost through the identical procedure and it read 11.96 m
// too. A number from this command means nothing without its control.
// ===========================================================================


// ===========================================================================
// C11b -- WHICH of the three telemetry defects is this?
//
// C11 compares a ghost's stored telemetry against the route re-simulated from
// its own inputs and refuses above 0.01 m. That bar cannot separate two
// defects whose consequences are OPPOSITE, because one is speed-dependent and
// the other is not:
//
//   own telemetry      < 0.001 m                     publish
//   STALE BUFFER       exactly speed x 0.010 m       refuse to WRITE; do NOT
//                                                    withdraw what is published
//   TEMPLATE POISONING metres to hundreds, no tick   withdraw
//
// On a slow map one tick is 5 cm and passes the bar; on a fast map it is 0.5 m
// and reads as poisoning. Measured: 267460's published file reads 0.487 m
// against its own route -- which C11 calls FOREIGN-TELEMETRY -- at a median
// speed of 39.92 m/s, so the implied offset is 10.024 ms over 236 samples.
// One tick. Our car, our run, displayed 10 ms late; the real 267460 poisoning
// that forced a withdrawal read 17.05 m median and 53.19 m max, forty times
// larger and not a whole tick.
//
// So divide by the car's OWN SPEED and ask whether the answer is a tick. That
// turns a threshold into a test of identity -- the same move as replacing the
// velocity projection with bit-equality at lag 1.
//
// The verdict text names which of the three it found, because
// "FOREIGN-TELEMETRY" on a stale file is a verdict that costs a withdrawal.
// ===========================================================================








// ===========================================================================
// COVERAGE ACCOUNTING FOR NEGATIVE-RETURNING TOOLS.
//
// Tonight's largest bug hid in exactly this gap: `locate_candidates` had
// `take(64)` on its shortlist, dropped 6161 of 6637 addresses WITHOUT PROBING
// THEM, and reported `0 passing` as though it had looked. Four hypotheses were
// burned before anyone counted.
//
// Every tool here that can return a negative -- the triage sampler, the lag
// scan, the poison sweep -- has the same shape, so each now reports what it
// EXAMINED beside what it FOUND, and a zero finding with a nonzero drop count
// is a hard error naming the knob rather than a clean-looking result.
//
// The abort path is instrumented too, and that is the half people forget: a
// tool that produces NOTHING and a tool that produces the WRONG THING are both
// silent failures, and the first is worse because it reads as bad luck. On
// 227654 four of five regenerations produced no file at all, and a retry loop
// would have turned that 50 % failure rate into a 100 % ship rate with no
// signal whatsoever.
// ===========================================================================


/// The first race instant after which two tapes differ and KEEP differing.
///
/// `fk tapediff` reports the first differing tick, which is the wrong quantity:
/// one inert tick proves nothing about the trajectory, and 165922 has exactly
/// that case. This asks `fk tapediff` for the first difference and then, if the
/// tapes reconverge, walks forward — implemented as a windowed re-query so it
/// costs one extra subprocess rather than a decoder of its own.
pub fn first_sustained_diff(a: &str, b: &str) -> Option<i64> {
    match first_sustained_diff_checked(a, b) {
        TapeDiff::At(v) => Some(v),
        _ => None,
    }
}

/// The three-outcome form. Use this anywhere a silent `None` would be read as
/// a clean result — see `TapeDiff`.
pub fn first_sustained_diff_checked(a: &str, b: &str) -> TapeDiff {
    let mut from = match run_tapediff(a, b, None) {
        Err(e) => return TapeDiff::Unavailable(e),
        Ok(None) => return TapeDiff::Identical,
        Ok(Some(v)) => v,
    };
    // A divergence is sustained if the tapes also differ shortly after it. Step
    // forward past isolated ticks until the next difference is close behind.
    for _ in 0..64 {
        let next = match run_tapediff(a, b, Some(from + 10)) {
            Err(e) => return TapeDiff::Unavailable(e),
            // nothing after it: the divergence we already have is the last one
            Ok(None) => return TapeDiff::At(from),
            Ok(Some(v)) => v,
        };
        if next - from <= 200 {
            return TapeDiff::At(from);
        }
        from = next;
    }
    TapeDiff::At(from)
}

fn first_input_diff_after(a: &str, b: &str, from_ms: i64) -> Option<i64> {
    run_tapediff(a, b, Some(from_ms)).ok().flatten()
}

// ===========================================================================
// C12 -- IF THE INPUTS DIVERGED, THE POSITIONS MUST DIVERGE TOO.
//
// THE HOLE THIS FILLS. Every contamination test here keys on BIT-EQUALITY --
// `pair`, `lag`, `dup` -- which is what makes them free of thresholds and is
// also a precise blind spot: A NEAR-COPY DEFEATS THEM ALL. A file 0.9 mm from
// the human recording is not bit-identical to it, so all three say CLEAN.
//
// Measured: 227654's regenerated TAS_57573 passes spawn, C2, bit-exact
// contamination and the oracle, and sits **0.0009 m** from ailiei.'s 147.031
// across all 365 samples -- while `fk tapediff` puts the first input
// divergence at 3.050 s with 1782 differing ticks after it. Two runs whose
// inputs part at 3.050 s and stay nine tenths of a millimetre apart for the
// next sixteen seconds is not physics. It is the 0.494 mm shadow class, at the
// same order of magnitude.
//
// The fix is NOT to loosen the exact tests -- that reintroduces every threshold
// problem the night has been about. It is to ask the physics question that
// exactness cannot: after the inputs diverge, the trajectories MUST separate.
// Staying together is the anomaly, and it needs no threshold on "how close is
// too close", only the observation that the separation fails to grow.
// ===========================================================================

pub struct Diverge {
    pub first_diff_ms: i64,
    pub sep_before: f64,
    pub sep_after: f64,
    pub n_after: usize,
}

pub fn divergence_growth(a: &[R], b: &[R], first_diff_ms: i64) -> Option<Diverge> {
    let bm: std::collections::HashMap<i64, &R> = b.iter().map(|r| (r.ms, r)).collect();
    let mut before: Vec<f64> = Vec::new();
    let mut after: Vec<f64> = Vec::new();
    for x in a {
        let Some(y) = bm.get(&x.ms) else { continue };
        let mut s = 0.0;
        for k in 0..3 {
            let q = x.pos[k] - y.pos[k];
            s += q * q;
        }
        let d = s.sqrt();
        if !d.is_finite() {
            continue;
        }
        // one second of grace: a divergence takes a moment to show in position
        if x.ms < first_diff_ms {
            before.push(d);
        } else if x.ms > first_diff_ms + 1000 {
            after.push(d);
        }
    }
    if after.len() < 5 {
        return None;
    }
    let med = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let n_after = after.len();
    Some(Diverge {
        first_diff_ms,
        sep_before: if before.is_empty() { 0.0 } else { med(&mut before) },
        sep_after: med(&mut after),
        n_after,
    })
}


/// The C8b acceptance rule, as a function, so it can be tested without a
/// fixture that fails C8.
///
/// C8 upstream asks whether the implied wheel radius sits in a 0.30-0.45 m
/// band. That band is the STADIUM wheel: a snow car measures 0.4700 m, and the
/// check therefore refused files for driving the wrong car. C8b asks the
/// question the band was standing in for -- *is there a consistent wheel of
/// SOME size* -- and the bar is the share of rolling steps that agree on one
/// radius to within 15 %.
///
/// The 0.15 bar is deliberately low. It is not "most steps look like a wheel";
/// it is "enough steps agree that a wheel exists at all", on maps where the car
/// spends most of the run sideways (134672 has 24-35 % of ticks within 3 % of
/// pure rolling, against 96.1 % on 173636). Raising it turns a permissive
/// exception into a refusal of exactly the files it was written to admit.
pub fn c8b_accepts(share: f64, radius: f64) -> bool {
    share >= C8B_MIN_SHARE && radius.is_finite()
}

/// The share of rolling steps that must agree on one radius for C8b to accept.
pub const C8B_MIN_SHARE: f64 = 0.15;

/// C12: after the tapes part, has the trajectory stayed with the reference?
///
/// GROWTH relative to the pre-divergence separation, never a distance. A metre
/// bar refuses honest files -- on 279218 five independent clean runs sit
/// 0.29-0.62 m from the human, and 270053's clean file departs by 0.363 m over
/// a 4.5 s segment. What separates a copy from a run is not how far it got but
/// whether it moved AT ALL once the inputs stopped being shared, and the two
/// populations are six orders of magnitude apart.
pub fn c12_is_near_copy(sep_before: f64, sep_after: f64) -> bool {
    sep_after / sep_before.max(1e-9) <= C12_GROWTH_BAR && sep_after <= C12_TOGETHER_M
}

pub const C12_GROWTH_BAR: f64 = 10.0;

/// AND THEY MUST STILL BE IN THE SAME PLACE. Growth alone convicts a pair that
/// was already far apart before the tapes parted: 285268's TAS_49278 ends up
/// **2.907 m** from the human world record, five times the 0.575 m they were
/// apart beforehand, and a ratio of 5 is under the bar. Two runs three metres
/// apart are not one run, whatever the ratio says.
///
/// The two populations, measured rather than chosen:
///
/// | | separation after the tapes part |
/// |---|---|
/// | copies -- 267460's three files against Wirtual's WR | **0.000633 m** over 375 samples |
/// | 227654's regenerated TAS_57573 against ailiei.'s run | **0.0009 m** over 365 samples |
/// | honest divergence -- 267859 TAS_10859 | 0.160 m |
/// | honest divergence -- 279218 best_pF | 0.487 m |
/// | honest divergence -- 279209 kb_gasfull | 0.721 m |
/// | honest divergence -- 285268 TAS_49278 | 2.908 m |
///
/// The gap is a factor of 178 with nothing in it. This bar is eleven times the
/// largest copy and sixteen times below the smallest honest divergence: a
/// gulf, not a tuned edge.
///
/// WHAT SETTLES A DISPUTED CASE is not this function but the engine. 267460's
/// three files reproduce their recorded trajectory from their own tapes to
/// 0.000000 m, and the server has no second car in the process -- so the
/// engine, driving our tape, is what puts the car 0.6 mm from Wirtual's line.
/// On a Trial map an input that differs where the car is wedged is inert, and
/// that is the case this comparison cannot resolve from two files alone.
pub const C12_TOGETHER_M: f64 = 0.01;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whlcmd::R;

    fn r(ms: i64, x: f64, speed: f64) -> R {
        R { ms, pos: [x, 0.0, 0.0], vel: [speed, 0.0, 0.0], speed, raw: vec![0u8; 116] }
    }

    /// The three classes C3 has to tell apart, built by hand so the test does
    /// not depend on a corpus file happening to contain each one.
    ///
    /// The distinction is what emptied a work queue of 24 files across 8 maps
    /// that were mostly never broken: a bar on distance, or even on implied
    /// speed alone, cannot tell a teleport from fast driving, and these cars
    /// are very fast.
    #[test]
    fn a_teleport_a_respawn_and_fast_driving_are_three_different_things() {
        // Fast but real: 8 m per 50 ms sample is 160 m/s, well past any fixed
        // distance bar and past the 5 m floor -- and the car's own speedometer
        // agrees. The fastest thing in this corpus is a 546 km/h reactor run.
        let fast: Vec<R> = (0..10).map(|i| r(i * 50, i as f64 * 8.0, 160.0)).collect();
        assert!(
            jumps(&fast).is_empty(),
            "ordinary fast driving is not a jump: {:?}",
            jumps(&fast).iter().map(|j| j.kind).collect::<Vec<_>>()
        );

        // a teleport: 400 m in one 50 ms sample while the speedometer reads 19.2
        let mut spliced = fast.clone();
        spliced.push(r(500, 900.0, 19.2));
        let js = jumps(&spliced);
        assert_eq!(js.len(), 1, "one jump expected");
        assert!(js[0].kind.starts_with("SPLICE"), "got {}", js[0].kind);

        // a respawn: the same jump, but the car reports EXACTLY zero speed and
        // then stays put. Normal on Trial maps; not a defect.
        let mut resp = fast.clone();
        for k in 0..8 {
            resp.push(r(500 + k * 50, 900.0, 0.0));
        }
        let js = jumps(&resp);
        assert_eq!(js.len(), 1);
        assert!(js[0].kind.starts_with("RESPAWN"), "got {}", js[0].kind);

        // the world origin is not a place the car was: the step away from a
        // (0,0,0) placeholder first sample is the record starting, not motion.
        let mut orig = vec![r(0, 0.0, 0.0)];
        orig.extend((1..10).map(|i| r(i * 50, 1000.0 + i as f64 * 8.0, 160.0)));
        let js = jumps(&orig);
        assert_eq!(js.len(), 1);
        assert!(js[0].kind.starts_with("ORIGIN"), "got {}", js[0].kind);
    }

    #[test]
    fn the_c8b_bar_admits_a_drift_map_and_refuses_noise() {
        // 134672 drives permanently sideways: 24-35 % of its ticks are within
        // 3 % of pure rolling. It must be admitted.
        assert!(c8b_accepts(0.24, 0.3643));
        // a snow car's 0.4700 m is a real wheel and outside upstream C8's band
        assert!(c8b_accepts(0.50, 0.4700));
        // no agreement at all is not a wheel
        assert!(!c8b_accepts(0.05, 0.3643));
        // and a radius that is not a number is not a radius
        assert!(!c8b_accepts(0.90, f64::NAN));
        // The bar itself, pinned: moving it to 0.95 refuses the drift map,
        // which is the file the exception exists for.
        assert_eq!(C8B_MIN_SHARE, 0.15);
    }

    #[test]
    fn c12_growth_bar_separates_the_two_populations() {
        // A run that drives its own tape separates by three orders of
        // magnitude after the inputs part; a near-copy tracks the reference.
        // Measured populations: 0.000509 m against 1113 m.
        assert!(c12_is_near_copy(0.000509, 0.000512));
        assert!(!c12_is_near_copy(0.000509, 1113.0));
        // and an honest short segment (270053 departs 0.363 m over 4.5 s from
        // a 0.0005 m start) must NOT be called a copy
        assert!(!c12_is_near_copy(0.0005, 0.363));
    }
}

fn fmt_s(ms: i64) -> String {
    format!("{:.3}", ms as f64 / 1000.0)
}

fn find_run_span(a: &[R], b: &[R], lag: i64, want: usize) -> Option<(i64, i64)> {
    fn key(r: &R) -> Option<&[u8]> {
        if r.raw.len() >= 59 { Some(&r.raw[47..59]) } else { None }
    }
    let mut run = 0usize;
    let mut start = 0usize;
    for i in 0..a.len() {
        let j = i as i64 - lag;
        let same = if j < 0 || j as usize >= b.len() {
            false
        } else {
            matches!((key(&a[i]), key(&b[j as usize])), (Some(x), Some(y)) if x == y)
        };
        if same {
            if run == 0 {
                start = i;
            }
            run += 1;
            if run == want {
                return Some((a[start].ms, a[i].ms));
            }
        } else {
            run = 0;
        }
    }
    None
}
