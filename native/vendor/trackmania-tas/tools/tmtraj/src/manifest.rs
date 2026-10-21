//! `tmtraj intg manifest` -- THE PROVENANCE MANIFEST.
//!
//! WHY THIS IS THE PRIMARY EVIDENCE, not a nicety
//! ----------------------------------------------
//! The contamination test compares a file against a human recording we happen
//! to hold. That is corroboration, and it has three holes we know about:
//!
//!   * 134672 has no downloaded human recording, and 276874/276877 are
//!     zero-record maps that never will;
//!   * on 227654 the human recording IS the donor, so the test is blind by
//!     construction -- the file agrees with the reference because both are the
//!     same stranger;
//!   * a head-anchored identical block is what determinism produces from a
//!     shared opening tape AND what an inherited donor prefix looks like, and
//!     the file cannot tell you which.
//!
//! So CLEAN can only ever mean "no splice found against the references named
//! in this row". What actually certifies a file is a record of HOW IT WAS
//! MADE: which bytes came from a live engine run, which were inherited, from
//! which donor, and what the oracle said about the result. That record is this
//! manifest. A filename is a claim and nothing can contradict it; a manifest
//! is a claim that a machine can check, and `intg gate --manifest` checks it.
//!
//! WHAT IT MUST SURVIVE
//! --------------------
//! Every defect this project shipped would have been visible in one of these
//! fields before publication:
//!
//!   telemetry_coverage  the 36 part-carrier files in the v2 pass (one at 1
//!                       sample of 365) -- a per-FILE fraction, never
//!                       files-per-corpus
//!   donor + donor_md5   the eleven ghosts carrying a human recording whole
//!   fields_regenerated  the rpm/gear/wheel bytes that are still the carrier's
//!   oracle_time         the file on disk, re-simulated, against declared_time
//!   tools[]             `u02 truncate` silently rewrites the last 40 packets
//!                       with a locator signature; with under 40 ticks of
//!                       headroom that lands on LIVE ticks (measured 18.702 ->
//!                       18.703 from a call that truncated nothing). So the
//!                       record carries every tool, its arguments, and for
//!                       truncate the headroom it actually had.
//!
//! FORMAT: JSON, hand-written -- the workspace builds --offline against a
//! vendored set with no serde. One manifest per ghost, `<ghost>.manifest.json`
//! beside it. Unknown fields are preserved on read so a later arm can add its
//! own without this tool dropping them.

use crate::intgcmd::md5_hex;

/// Minimal JSON value, enough to read and write a manifest without serde.
#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, x)| x),
            _ => None,
        }
    }
    pub fn s(&self, k: &str) -> Option<&str> {
        match self.get(k) {
            Some(J::Str(s)) => Some(s),
            _ => None,
        }
    }
    pub fn n(&self, k: &str) -> Option<f64> {
        match self.get(k) {
            Some(J::Num(x)) => Some(*x),
            _ => None,
        }
    }
    pub fn i(&self, k: &str) -> Option<i64> {
        self.n(k).map(|x| x as i64)
    }
    pub fn set(&mut self, k: &str, v: J) {
        if let J::Obj(o) = self {
            match o.iter_mut().find(|(n, _)| n == k) {
                Some(e) => e.1 = v,
                None => o.push((k.into(), v)),
            }
        }
    }
    pub fn write(&self, out: &mut String, ind: usize) {
        let pad = "  ".repeat(ind);
        match self {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Num(x) => {
                if x.fract() == 0.0 && x.abs() < 9e15 {
                    out.push_str(&format!("{}", *x as i64));
                } else {
                    out.push_str(&format!("{}", x));
                }
            }
            J::Str(s) => {
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\t' => out.push_str("\\t"),
                        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            J::Arr(v) if v.is_empty() => out.push_str("[]"),
            J::Arr(v) => {
                out.push_str("[\n");
                for (i, x) in v.iter().enumerate() {
                    out.push_str(&"  ".repeat(ind + 1));
                    x.write(out, ind + 1);
                    if i + 1 < v.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push(']');
            }
            J::Obj(v) if v.is_empty() => out.push_str("{}"),
            J::Obj(v) => {
                out.push_str("{\n");
                for (i, (k, x)) in v.iter().enumerate() {
                    out.push_str(&"  ".repeat(ind + 1));
                    J::Str(k.clone()).write(out, ind + 1);
                    out.push_str(": ");
                    x.write(out, ind + 1);
                    if i + 1 < v.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push('}');
            }
        }
    }
    pub fn to_string_pretty(&self) -> String {
        let mut s = String::new();
        self.write(&mut s, 0);
        s.push('\n');
        s
    }
}

// ---- a small recursive-descent parser, so a manifest round-trips ----------

pub fn parse(s: &str) -> Result<J, String> {
    let b: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    let v = pval(&b, &mut i)?;
    Ok(v)
}

fn ws(b: &[char], i: &mut usize) {
    while *i < b.len() && b[*i].is_whitespace() {
        *i += 1;
    }
}

fn pval(b: &[char], i: &mut usize) -> Result<J, String> {
    ws(b, i);
    if *i >= b.len() {
        return Err("unexpected end".into());
    }
    match b[*i] {
        '{' => {
            *i += 1;
            let mut o = Vec::new();
            loop {
                ws(b, i);
                if *i < b.len() && b[*i] == '}' {
                    *i += 1;
                    break;
                }
                let J::Str(k) = pval(b, i)? else { return Err("object key".into()) };
                ws(b, i);
                if *i >= b.len() || b[*i] != ':' {
                    return Err("expected :".into());
                }
                *i += 1;
                o.push((k, pval(b, i)?));
                ws(b, i);
                if *i < b.len() && b[*i] == ',' {
                    *i += 1;
                }
            }
            Ok(J::Obj(o))
        }
        '[' => {
            *i += 1;
            let mut a = Vec::new();
            loop {
                ws(b, i);
                if *i < b.len() && b[*i] == ']' {
                    *i += 1;
                    break;
                }
                a.push(pval(b, i)?);
                ws(b, i);
                if *i < b.len() && b[*i] == ',' {
                    *i += 1;
                }
            }
            Ok(J::Arr(a))
        }
        '"' => {
            *i += 1;
            let mut s = String::new();
            while *i < b.len() && b[*i] != '"' {
                if b[*i] == '\\' && *i + 1 < b.len() {
                    *i += 1;
                    match b[*i] {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'u' => {
                            let h: String = b[*i + 1..(*i + 5).min(b.len())].iter().collect();
                            if let Ok(c) = u32::from_str_radix(&h, 16) {
                                if let Some(c) = char::from_u32(c) {
                                    s.push(c);
                                }
                            }
                            *i += 4;
                        }
                        c => s.push(c),
                    }
                } else {
                    s.push(b[*i]);
                }
                *i += 1;
            }
            *i += 1;
            Ok(J::Str(s))
        }
        't' => {
            *i += 4;
            Ok(J::Bool(true))
        }
        'f' => {
            *i += 5;
            Ok(J::Bool(false))
        }
        'n' => {
            *i += 4;
            Ok(J::Null)
        }
        _ => {
            let st = *i;
            while *i < b.len()
                && (b[*i].is_ascii_digit() || "+-.eE".contains(b[*i]))
            {
                *i += 1;
            }
            let t: String = b[st..*i].iter().collect();
            t.parse().map(J::Num).map_err(|_| format!("bad number {:?}", t))
        }
    }
}

pub const SCHEMA: &str = "tm-ghost-provenance/2";
/// `/1` manifests stay valid and are NOT retrofitted: one that was true when
/// it was written stays true. `/2` adds only the `repair` provenance kind.
pub const SCHEMA_V1: &str = "tm-ghost-provenance/1";

// ===========================================================================
// THE `repair` PROVENANCE KIND (schema /2, 2026-08-21)
//
// A container-only repair has NO telemetry coverage to declare, because the
// person repairing it did not generate the telemetry. Four published files had
// their declared time rewritten from the carrier's to their own tonight; their
// samples were never touched. Writing `262/262` for those would assert that
// every sample came from an engine instant in a run I never made -- the exact
// shape of claim this whole format exists to stop -- so the manifest refused,
// correctly, on a missing coverage field.
//
// The fix is not to relax the coverage rule. It is to say what is actually
// known, which is STRONGER:
//
//     coverage: not-applicable: container-only repair; telemetry untouched
//               and verified bit-identical to <md5>
//
// A coverage ratio asserts something about a private past that nobody can
// re-inspect. A bit-identity hash against the artefact that was already
// published asserts something anyone can check tonight.
// ===========================================================================


pub fn read(path: &str) -> Result<J, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    parse(&s)
}

pub fn manifest_path(ghost: &str) -> String {
    format!("{}.manifest.json", ghost)
}

fn now_iso() -> String {
    // No chrono in the vendored set. Unix seconds is unambiguous and sorts.
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}", t)
}

/// Build a manifest from what the producer knows.
#[allow(clippy::too_many_arguments)]
pub fn build(
    ghost: &str,
    map_id: &str,
    donor: Option<(&str, &str)>, // (path, md5)
    inputs_from: Option<(&str, &str)>,
    declared_ms: i64,
    oracle_ms: Option<i64>,
    oracle_cps: Option<i64>,
    coverage: Option<(usize, usize)>, // (regenerated, total)
    fields_regen: &[String],
    fields_inherited: &[String],
    engine: Option<J>,
    tools: Vec<J>,
) -> J {
    let bytes = std::fs::read(ghost).unwrap_or_default();
    let name = std::path::Path::new(ghost)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut m = J::Obj(vec![]);
    m.set("schema", J::Str(SCHEMA.into()));
    m.set("written_unix", J::Str(now_iso()));
    m.set("file", J::Str(name));
    m.set("md5", J::Str(md5_hex(&bytes)));
    m.set("bytes", J::Num(bytes.len() as f64));
    m.set("map_id", J::Str(map_id.into()));
    m.set("declared_time_ms", J::Num(declared_ms as f64));
    m.set(
        "oracle_time_ms",
        oracle_ms.map(|v| J::Num(v as f64)).unwrap_or(J::Null),
    );
    m.set(
        "oracle_checkpoints",
        oracle_cps.map(|v| J::Num(v as f64)).unwrap_or(J::Null),
    );
    // THE CONTAINER DONOR. Every ghost this project produces is a downloaded
    // human ghost with our input chunks grafted in -- the only native container
    // for map X is a ghost a human recorded on X. So naming the donor is not
    // optional bookkeeping: it is the identity of every byte we did not write.
    m.set(
        "container_donor",
        match donor {
            Some((p, h)) => {
                let mut d = J::Obj(vec![]);
                d.set("path", J::Str(p.into()));
                d.set("md5", J::Str(h.into()));
                d
            }
            None => J::Null,
        },
    );
    m.set(
        "source_inputs",
        match inputs_from {
            Some((p, h)) => {
                let mut d = J::Obj(vec![]);
                d.set("path", J::Str(p.into()));
                d.set("md5", J::Str(h.into()));
                d
            }
            None => J::Null,
        },
    );
    m.set("engine_run", engine.unwrap_or(J::Null));
    // COVERAGE IS PER FILE. `samples_regenerated / samples_in_record` -- the
    // metric whose absence let 36 files ship as "regenerated" when eight were
    // under 50 % and two were at one sample of 365.
    m.set(
        "telemetry_coverage",
        match coverage {
            Some((r, t)) => {
                let mut c = J::Obj(vec![]);
                c.set("samples_regenerated", J::Num(r as f64));
                c.set("samples_in_record", J::Num(t as f64));
                c.set(
                    "fraction",
                    J::Num(if t == 0 { 0.0 } else { r as f64 / t as f64 }),
                );
                c
            }
            None => J::Null,
        },
    );
    m.set(
        "fields_regenerated",
        J::Arr(fields_regen.iter().map(|s| J::Str(s.clone())).collect()),
    );
    // WHAT IS STILL THE CARRIER'S, named. rpm, gear, wheel rotation, suspension
    // dampen, turbo, ice, dirt and wetness were never regenerated; a file that
    // does not say so is claiming more than it is.
    m.set(
        "fields_inherited",
        J::Arr(fields_inherited.iter().map(|s| J::Str(s.clone())).collect()),
    );
    m.set("tools", J::Arr(tools));
    m
}

/// One tool invocation, for the `tools` array.
pub fn tool(name: &str, args: &str, note: Option<&str>) -> J {
    let mut t = J::Obj(vec![]);
    t.set("tool", J::Str(name.into()));
    t.set("args", J::Str(args.into()));
    if let Some(n) = note {
        t.set("note", J::Str(n.into()));
    }
    t
}

/// Verify a file against its own manifest. Returns (hard failures, lines).
pub fn verify(ghost: &str, m: &J) -> (usize, Vec<String>) {
    let mut bad = 0usize;
    let mut out: Vec<String> = Vec::new();
    // A BACKFILLED manifest cannot prove what it never witnessed. The donor,
    // the engine run and the coverage are not recoverable from a ghost's bytes,
    // so on a RECONSTRUCTED manifest their absence is a declared limit rather
    // than a false claim -- and the file is UNCERTIFIED, which is not a pass
    // either. Failing it instead would make every already-published file look
    // defective and teach people to ignore the gate.
    let reconstructed = m.s("provenance") == Some("RECONSTRUCTED");
    let fail = |o: &mut Vec<String>, b: &mut usize, s: String| {
        *b += 1;
        o.push(format!("FAIL   M-{}", s));
    };

    match m.s("schema") {
        Some(SCHEMA) => out.push(format!("PASS   M-schema  {}", SCHEMA)),
        Some(SCHEMA_V1) => out.push(format!(
            "PASS   M-schema  {} (an earlier schema: still valid, not retrofitted)",
            SCHEMA_V1
        )),
        Some(x) => fail(&mut out, &mut bad, format!("schema  unknown schema {:?}", x)),
        None => fail(&mut out, &mut bad, "schema  no schema field".into()),
    }

    // THE MANIFEST MUST DESCRIBE THIS FILE. Without this, a manifest is just
    // another filename -- a claim beside the bytes rather than about them.
    let bytes = std::fs::read(ghost).unwrap_or_default();
    let have = md5_hex(&bytes);
    match m.s("md5") {
        Some(x) if x == have => out.push(format!("PASS   M-md5     the manifest describes THIS file ({})", have)),
        Some(x) => fail(
            &mut out,
            &mut bad,
            format!("md5     the manifest is for md5 {} but this file is {}", x, have),
        ),
        None => fail(&mut out, &mut bad, "md5     the manifest names no md5".into()),
    }

    // COVERAGE. Anything under 100 % means donor telemetry is in the file.
    // A `repair` manifest has none to declare and says so with the stronger
    // statement instead: the telemetry is bit-identical to a named artefact.
    let repair_of = match m.get("repaired_from_md5") {
        Some(J::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    if let Some(src) = &repair_of {
        out.push(format!(
            "PASS   M-cover   not-applicable: container-only repair; telemetry untouched and \
             verified bit-identical to {}",
            src
        ));
    } else {
    match m.get("telemetry_coverage") {
        Some(J::Obj(_)) => {
            let c = m.get("telemetry_coverage").unwrap();
            let r = c.i("samples_regenerated").unwrap_or(-1);
            let t = c.i("samples_in_record").unwrap_or(-1);
            if r < 0 || t <= 0 {
                fail(&mut out, &mut bad, "coverage  malformed coverage block".into());
            } else if r == t {
                out.push(format!("PASS   M-cover   all {} samples in the record were regenerated", t));
            } else {
                fail(
                    &mut out,
                    &mut bad,
                    format!(
                        "cover   only {} of {} samples ({:.1} %) were regenerated -- {} carry the DONOR's telemetry",
                        r,
                        t,
                        100.0 * r as f64 / t as f64,
                        t - r
                    ),
                );
            }
        }
        _ if reconstructed => out.push(
            "NOTE   M-cover   coverage UNKNOWABLE: this manifest was backfilled, and how much of a \
             published file is ours cannot be recovered from its bytes"
                .into(),
        ),
        _ => fail(
            &mut out,
            &mut bad,
            "cover   no telemetry_coverage -- the file does not say how much of it is ours".into(),
        ),
    }
    }

    // THE DONOR MUST BE NAMED. "No donor" is a positive claim (a natively
    // recorded file) and is accepted only when spelled that way.
    match m.get("container_donor") {
        Some(J::Obj(_)) => {
            let d = m.get("container_donor").unwrap();
            match (d.s("path"), d.s("md5")) {
                (Some(p), Some(h)) if !h.is_empty() => {
                    out.push(format!("PASS   M-donor   container donor named: {} ({})", p, h))
                }
                _ => fail(&mut out, &mut bad, "donor   donor block without a path and md5".into()),
            }
        }
        Some(J::Null) => out.push(
            "PASS   M-donor   no container donor claimed -- this file is asserted natively recorded".into(),
        ),
        _ if reconstructed => out.push(
            "NOTE   M-donor   the container donor is not recorded in a ghost and cannot be \
             recovered after the fact"
                .into(),
        ),
        _ => fail(
            &mut out,
            &mut bad,
            "donor   no container_donor field -- every synthesised ghost has one".into(),
        ),
    }

    // THE ORACLE MUST AGREE WITH THE DECLARED TIME.
    match (m.i("declared_time_ms"), m.i("oracle_time_ms")) {
        (Some(d), Some(o)) if d == o => out.push(format!(
            "PASS   M-time    declared and oracle both {:.3} s",
            d as f64 / 1000.0
        )),
        (Some(d), Some(o)) => fail(
            &mut out,
            &mut bad,
            format!(
                "time    declared {:.3} s but the oracle returned {:.3} s ({:+} ms)",
                d as f64 / 1000.0,
                o as f64 / 1000.0,
                o - d
            ),
        ),
        (Some(_), None) if reconstructed => out.push(
            "NOTE   M-time    no oracle run recorded: backfilled. The gate's own C-oracle family \
             re-simulates the file, so this is covered there, not here."
                .into(),
        ),
        (Some(_), None) => fail(
            &mut out,
            &mut bad,
            "time    no oracle_time_ms -- the written file was never re-simulated".into(),
        ),
        _ => fail(&mut out, &mut bad, "time    no declared_time_ms".into()),
    }

    // INHERITED FIELDS MUST BE DECLARED, not discovered later.
    match m.get("fields_inherited") {
        Some(J::Arr(v)) => {
            if v.is_empty() {
                out.push("PASS   M-fields  the manifest claims every field is regenerated".into());
            } else {
                let names: Vec<String> = v
                    .iter()
                    .map(|x| match x {
                        J::Str(s) => s.clone(),
                        _ => "?".into(),
                    })
                    .collect();
                out.push(format!(
                    "PASS   M-fields  {} field(s) declared INHERITED from the carrier: {}",
                    names.len(),
                    names.join(", ")
                ));
            }
        }
        _ => fail(
            &mut out,
            &mut bad,
            "fields  no fields_inherited list -- the file does not say which bytes are the carrier's".into(),
        ),
    }

    // THE TOOL LOG, and the truncate headroom rule.
    match m.get("tools") {
        Some(J::Arr(v)) => {
            out.push(format!("PASS   M-tools   {} tool invocation(s) recorded", v.len()));
            for t in v {
                let name = t.s("tool").unwrap_or("");
                if name.contains("truncate") {
                    // `u02 truncate` unconditionally rewrites the last 40
                    // packets with a pseudorandom locator signature, on the
                    // assumption they are post-finish and never simulated.
                    // With under 40 ticks of headroom that lands on LIVE ticks:
                    // measured 18.702 -> 18.703 on a tape with 10 ticks spare,
                    // and 0 ms on a sibling only because the garbage fell in a
                    // straight. A 1 ms discrepancy is exactly this signature.
                    match t.i("headroom_ticks") {
                        Some(h) if h >= 40 => out.push(format!(
                            "PASS   M-trunc   truncate had {} ticks of headroom (>= 40)",
                            h
                        )),
                        Some(h) => fail(
                            &mut out,
                            &mut bad,
                            format!(
                                "trunc   truncate ran with only {} ticks of headroom -- its 40-packet signature landed on {} LIVE ticks",
                                h,
                                40 - h
                            ),
                        ),
                        None => fail(
                            &mut out,
                            &mut bad,
                            "trunc   a truncate call is recorded with no headroom_ticks -- unprovable".into(),
                        ),
                    }
                }
            }
        }
        _ if reconstructed => out.push(
            "NOTE   M-tools   the tool history of an already-published file is unrecoverable".into(),
        ),
        _ => fail(
            &mut out,
            &mut bad,
            "tools   no tools list -- which tools touched this file is unrecorded".into(),
        ),
    }

    // THE CERTIFICATION BLOCK IS MANDATORY. Absent, it reads as clean.
    match m.get("certification") {
        Some(J::Obj(_)) => {
            let c = m.get("certification").unwrap();
            let v = c.s("contamination_verdict").unwrap_or("");
            let nrefs = match c.get("references_tested") {
                Some(J::Arr(a)) => a.len(),
                _ => 0,
            };
            let limits: Vec<String> = match c.get("limits") {
                Some(J::Arr(a)) => a
                    .iter()
                    .map(|x| match x {
                        J::Str(s) => s.clone(),
                        _ => String::new(),
                    })
                    .collect(),
                _ => Vec::new(),
            };
            match v {
                "CONTAMINATED" => fail(
                    &mut out,
                    &mut bad,
                    "cert    the manifest itself records this file as CONTAMINATED".into(),
                ),
                "CLEAN" if nrefs > 0 => out.push(format!(
                    "PASS   M-cert    no splice found against {} named reference(s)",
                    nrefs
                )),
                "UNCERTIFIED" => out.push(format!(
                    "NOTE   M-cert    UNCERTIFIED -- {} reference(s) tested. This is NOT a pass.",
                    nrefs
                )),
                x => fail(
                    &mut out,
                    &mut bad,
                    format!("cert    contamination_verdict {:?} with {} reference(s) named", x, nrefs),
                ),
            }
            for l in &limits {
                out.push(format!("LIMIT  M-cert    {}", l));
            }
        }
        _ => fail(
            &mut out,
            &mut bad,
            "cert    no certification block -- an absent verdict reads as a clean one".into(),
        ),
    }

    (bad, out)
}

pub fn cmd(args: &[String]) {
    let flag = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print!(
            "\
tmtraj intg manifest -- provenance that travels with a ghost

  tmtraj intg manifest new GHOST --map-id ID --declared MS
        [--donor PATH] [--inputs-from PATH] [--oracle MS] [--cps N]
        [--coverage R/T] [--regen a,b,c] [--inherited a,b,c]
        [--engine-note TEXT] [--tool 'name:args'] ...   [--out PATH]

  tmtraj intg manifest verify GHOST [--manifest PATH]
        Check the file against its own manifest. Exit 0 ok, 2 refused.

  tmtraj intg manifest show GHOST [--manifest PATH]

The manifest is the PRIMARY evidence a ghost is ours. A contamination test
compares against a human recording we happen to hold; where we hold none, or
where the donor IS the reference, it can say nothing. The manifest records how
the file was made, and `intg gate --manifest` checks the file against it.
"
        );
        // Exit 0 when help was ASKED FOR, 2 when the arguments were simply
        // missing -- see the same split in intgcmd::cmd.
        std::process::exit(if args.iter().any(|a| a == "--help" || a == "-h") { 0 } else { 2 });
    }
    match args[0].as_str() {
        "new" => {
            let ghost = args
                .iter()
                .skip(1)
                .find(|a| !a.starts_with("--"))
                .cloned()
                .expect("GHOST");
            let map_id = flag("--map-id").unwrap_or_default();
            let declared: i64 = flag("--declared").and_then(|v| v.parse().ok()).unwrap_or(-1);
            let oracle: Option<i64> = flag("--oracle").and_then(|v| v.parse().ok());
            let cps: Option<i64> = flag("--cps").and_then(|v| v.parse().ok());
            let md5_of = |p: &str| md5_hex(&std::fs::read(p).unwrap_or_default());
            let donor = flag("--donor");
            let inputs = flag("--inputs-from");
            let cov = flag("--coverage").and_then(|s| {
                let (a, b) = s.split_once('/')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            });
            let split = |s: Option<String>| -> Vec<String> {
                s.map(|v| v.split(',').filter(|x| !x.is_empty()).map(|x| x.trim().to_string()).collect())
                    .unwrap_or_default()
            };
            let mut engine = J::Obj(vec![]);
            if let Some(n) = flag("--engine-note") {
                engine.set("note", J::Str(n));
            }
            engine.set("host", J::Str(std::env::var("HOSTNAME").unwrap_or_default()));
            let mut tools: Vec<J> = Vec::new();
            let mut i = 0usize;
            while i < args.len() {
                if args[i] == "--tool" {
                    if let Some(v) = args.get(i + 1) {
                        let (n, a) = v.split_once(':').unwrap_or((v.as_str(), ""));
                        tools.push(tool(n, a, None));
                    }
                    i += 2;
                    continue;
                }
                i += 1;
            }
            let mut m = build(
                &ghost,
                &map_id,
                donor.as_deref().map(|p| (p, md5_of(p))).as_ref().map(|(p, h)| (*p, h.as_str())),
                inputs.as_deref().map(|p| (p, md5_of(p))).as_ref().map(|(p, h)| (*p, h.as_str())),
                declared,
                oracle,
                cps,
                cov,
                &split(flag("--regen")),
                &split(flag("--inherited")),
                Some(engine),
                tools,
            );
            // `--repair-of MD5`: this file is a CONTAINER-ONLY repair of an
            // existing artefact. It declares no coverage, because whoever
            // repairs a container did not generate its telemetry; it declares
            // the bit-identity instead, which is checkable.
            if let Some(src) = flag("--repair-of") {
                m.set("provenance_kind", J::Str("repair".into()));
                m.set("repaired_from_md5", J::Str(src));
            }
            // The certification block: the verdict, the references actually
            // tested, and the named limits. Never omitted -- an absent
            // verdict reads as a clean one.
            //
            // `--certify-refs TSV` makes the manifest MEASURE instead of being
            // told: it runs the same contamination test the gate runs, over
            // every human recording held for this map, and records the verdict
            // with the references named. Without it the block defaulted to
            // UNCERTIFIED with an empty reference list even on a file the gate
            // had just cleared against five human recordings -- a manifest that
            // under-reports costs the next reader a regeneration for nothing,
            // and "UNCERTIFIED" is supposed to mean "nobody looked", not
            // "somebody looked and the manifest was not listening".
            let mut refs: Vec<(String, String)> = args
                .iter()
                .enumerate()
                .filter(|(_, a)| *a == "--tested-ref")
                .filter_map(|(i, _)| args.get(i + 1))
                .map(|p| (p.clone(), md5_of(p)))
                .collect();
            let mut limits: Vec<String> = args
                .iter()
                .enumerate()
                .filter(|(_, a)| *a == "--limit")
                .filter_map(|(i, _)| args.get(i + 1))
                .map(|k| limit_text(k, &map_id))
                .collect();
            let mut measured: Option<String> = None;
            if let Some(rp) = flag("--certify-refs") {
                match crate::intgcmd::load_refs(&rp) {
                    Ok(all) => {
                        let humans: Vec<&crate::intgcmd::Ref> = all
                            .iter()
                            .filter(|r| r.map == map_id && r.kind == "human")
                            .collect();
                        let mut cache = crate::intgcmd::new_cache();
                        let (v, tested, mut lim) =
                            crate::intgcmd::certify_now(&ghost, &humans, declared, &mut cache);
                        println!(
                            "certify: {} against {} reference(s) held for map {}",
                            v,
                            tested.len(),
                            map_id
                        );
                        refs = tested;
                        limits.append(&mut lim);
                        measured = Some(v);
                    }
                    Err(e) => println!("certify: could not read {}: {}", rp, e),
                }
            }
            let verdict = flag("--contam").or(measured).unwrap_or_else(|| {
                if refs.is_empty() { "UNCERTIFIED".into() } else { "CLEAN".into() }
            });
            m.set("certification", certification(&verdict, &refs, &limits));
            let out = flag("--out").unwrap_or_else(|| manifest_path(&ghost));
            std::fs::write(&out, m.to_string_pretty()).expect("write manifest");
            println!("wrote {}", out);
        }
        "backfill" => cmd_backfill(&args[1..]),
        "verify" | "show" => {
            let ghost = args
                .iter()
                .skip(1)
                .find(|a| !a.starts_with("--"))
                .cloned()
                .expect("GHOST");
            let mp = flag("--manifest").unwrap_or_else(|| manifest_path(&ghost));
            let m = match read(&mp) {
                Ok(m) => m,
                Err(e) => {
                    println!("=== {}  --  NO MANIFEST ({})", ghost, e);
                    println!("  A ghost with no manifest is UNPROVENANCED. It may be perfectly");
                    println!("  good; nothing in it says so.");
                    std::process::exit(2);
                }
            };
            if args[0] == "show" {
                print!("{}", m.to_string_pretty());
                return;
            }
            let (bad, lines) = verify(&ghost, &m);
            // An UNCERTIFIED file must not print a bare "VERIFIED" headline:
            // the headline is the only line most readers will see.
            let uncert = lines.iter().any(|l| l.contains("UNCERTIFIED"));
            println!(
                "=== {}  --  {}",
                ghost,
                if bad > 0 {
                    "MANIFEST REFUSED"
                } else if uncert {
                    "MANIFEST CONSISTENT, FILE UNCERTIFIED"
                } else {
                    "MANIFEST VERIFIED"
                }
            );
            for l in &lines {
                println!("  {}", l);
            }
            std::process::exit(if bad == 0 { 0 } else { 2 });
        }
        x => {
            eprintln!("unknown manifest subcommand {:?}", x);
            std::process::exit(2);
        }
    }
}

// ===========================================================================
// THE UNCERTIFIABLE CASE.
//
// Some files cannot be certified and never will be. 134672 has no downloaded
// human recording; 276874 and 276877 are zero-record maps where no human has
// ever driven, so no reference can exist even in principle; and on 227654 the
// human recording IS the donor, so the test is blind by construction.
//
// The temptation is to leave the certification block off those files. That is
// the whitewash: an absent field reads as a clean one to every human who scans
// a table, and this project has already shipped that exact mistake -- fourteen
// files whose own results table recorded `moved_m = 0.0000` and nobody read it.
//
// So a manifest MUST carry a certification block, it must NAME the references
// actually tested, and where it could test nothing it must say so in a word
// that cannot be mistaken for a pass. `verify` refuses a manifest with no
// certification block at all, and prints an UNCERTIFIED file's limits as
// prominently as any failure -- without failing it, because "we cannot test
// this" is not the same finding as "this is wrong".
// ===========================================================================

/// A certification block. `refs` are the references actually compared against;
/// `limits` are the named reasons certification is incomplete.
pub fn certification(verdict: &str, refs: &[(String, String)], limits: &[String]) -> J {
    let mut c = J::Obj(vec![]);
    c.set("contamination_verdict", J::Str(verdict.into()));
    c.set(
        "references_tested",
        J::Arr(
            refs.iter()
                .map(|(p, h)| {
                    let mut r = J::Obj(vec![]);
                    r.set("path", J::Str(p.clone()));
                    r.set("md5", J::Str(h.clone()));
                    r
                })
                .collect(),
        ),
    );
    c.set(
        "limits",
        J::Arr(limits.iter().map(|s| J::Str(s.clone())).collect()),
    );
    c
}

/// The standing limits, by name, so every manifest words them the same way.
pub fn limit_text(kind: &str, map_id: &str) -> String {
    match kind {
        "no-human-reference" => format!(
            "NO HUMAN RECORDING IS HELD FOR MAP {}. This file has not been tested for \
             contamination against any independent recording. It is UNTESTED, not clean.",
            map_id
        ),
        "zero-record-map" => format!(
            "MAP {} HAS NO HUMAN RECORDS AT ALL. No reference recording can exist, so this \
             file can never be certified by comparison. Its only evidence is this manifest.",
            map_id
        ),
        "donor-is-the-reference" => format!(
            "THE ONLY HUMAN RECORDING HELD FOR MAP {} IS THIS FILE'S OWN CONTAINER DONOR. \
             A comparison against it cannot distinguish 'our telemetry' from 'the donor's \
             telemetry': the test is blind by construction and its result means nothing.",
            map_id
        ),
        "head-anchored-block" =>
            "A LEADING BLOCK OF SAMPLES IS BIT-IDENTICAL TO THE REFERENCE. Determinism \
             produces this from a shared opening tape, and so does an inherited donor \
             prefix; the file cannot distinguish them. Only telemetry_coverage can."
                .into(),
        "fields-inherited" =>
            "SOME SAMPLE BYTES ARE THE CARRIER'S, LISTED IN fields_inherited. Those fields \
             describe the donor's drive, not ours."
                .into(),
        x => format!("UNNAMED LIMIT: {}", x),
    }
}

// ===========================================================================
// `intg manifest backfill` -- a manifest for every file already published.
//
// A backfilled manifest is WEAKER THAN ONE WRITTEN AT BIRTH and must say so.
// At birth the producer knows the donor it grafted into, the engine run that
// sampled the state, the coverage it achieved, and every tool it called. After
// the fact, none of that is recoverable from the bytes: a ghost does not
// record where its container came from. What CAN be established after the fact
// is what the file is now -- its md5, its declared time, what the oracle says
// about it today, and what every contamination instrument we have says when
// pointed at it.
//
// So a backfilled manifest is marked `provenance: RECONSTRUCTED`, its unknown
// fields are `null` rather than guessed, and `certification` carries the
// limits that apply. Writing "container_donor: unknown" is honest; inventing
// one, or leaving the field out so it reads as "none", is the disease.
// ===========================================================================

pub fn cmd_backfill(args: &[String]) {
    let flag = |n: &str| -> Option<String> {
        args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned()
    };
    let corpus = flag("--corpus").expect("--corpus");
    let refs_p = flag("--refs");
    let dry = args.iter().any(|a| a == "--dry-run");
    let refs = refs_p
        .as_deref()
        .map(|p| crate::intgcmd::load_refs(p).unwrap())
        .unwrap_or_default();

    let mut files: Vec<std::path::PathBuf> = Vec::new();
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
                files.push(p);
            }
        }
    }
    files.sort();
    let mut cache = crate::intgcmd::new_cache();
    println!("file\tmap\tdeclared_s\tverdict\tlimits\tmanifest");
    for f in &files {
        let fp = f.to_string_lossy().to_string();
        let mapdir = f
            .parent()
            .and_then(|d| d.parent())
            .and_then(|d| d.file_name())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let mapid: String = mapdir.chars().take_while(|c| c.is_ascii_digit()).collect();
        let declared = crate::intgcmd::declared_race_ms(&fp).unwrap_or(-1);
        let humans: Vec<&crate::intgcmd::Ref> = refs
            .iter()
            .filter(|r| r.map == mapid && r.kind == "human")
            .collect();

        // What the instruments say TODAY. This is establishable after the fact;
        // the production history is not.
        let (verdict, tested, mut limits) =
            crate::intgcmd::certify_now(&fp, &humans, declared, &mut cache);
        limits.push(limit_text("fields-inherited", &mapid));

        let mut m = build(
            &fp,
            &mapid,
            None, // the donor is NOT recoverable from the bytes
            None,
            declared,
            None, // no oracle run in a backfill; the gate does that
            None,
            None, // coverage is unknowable after the fact
            &[],
            &[
                "rpm".into(),
                "gear".into(),
                "wheel_rotation".into(),
                "suspension_dampen".into(),
                "turbo".into(),
                "ice".into(),
                "dirt".into(),
                "wetness".into(),
            ],
            None,
            vec![],
        );
        m.set("provenance", J::Str("RECONSTRUCTED".into()));
        m.set(
            "provenance_note",
            J::Str(
                "Backfilled after publication. The container donor, the engine run and the \
                 telemetry coverage are NOT recoverable from a ghost's bytes and are null here, \
                 not unknown-but-probably-fine. What this manifest establishes is what the file \
                 IS: its md5, its declared time, and what every contamination instrument says \
                 about it. A manifest written at birth is strictly stronger."
                    .into(),
            ),
        );
        m.set("certification", certification(&verdict, &tested, &limits));
        let out = manifest_path(&fp);
        if !dry {
            std::fs::write(&out, m.to_string_pretty()).expect("write manifest");
        }
        println!(
            "{}\t{}\t{:.3}\t{}\t{}\t{}",
            f.file_name().unwrap().to_string_lossy(),
            mapdir,
            declared as f64 / 1000.0,
            verdict,
            limits.len(),
            if dry { "(dry run)".into() } else { out }
        );
    }
}
