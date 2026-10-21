//! `ghost synth` -- build a ghost container by EMITTING it, not by copying it.
//!
//! The question this answers is "could we generate a ghost from scratch
//! instead of transplanting a game-written one?". The honest test of that is
//! not "does a rebuilt file validate" -- a memcpy of a donor validates too. It
//! is: decompose a real ghost into named parts, then re-emit every byte from
//! PARSED VALUES and built-in constants, and see whether the result is
//! byte-identical to what the game wrote.
//!
//! What openplanet.dev gives us (next.openplanet.dev/Game/<Class>) is the
//! authority for what those parts MEAN: the class ids are exactly our chunk-id
//! prefixes -- CGameGhost 0x0303F000, CGameCtnGhost 0x03092000,
//! CPlugEntRecordData 0x0911F000 -- and CGameCtnGhost's 28 documented members
//! (RaceTime, NbRespawns, Validate_ChallengeUid, Validate_ExeChecksum, ...)
//! are the fields those chunks serialise.
//!
//! A ghost body turns out to be ELEVEN parts: five skippable chunks carrying
//! ~99-100 % of the bytes, and six gaps totalling 123 bytes of non-skippable
//! chunks. Every one of those 123 bytes is identified below.

use crate::cli::die;
use gbx::container::{Gbx, SKIP_MAGIC};

/// The `0x0303F006` chunk (CGameGhost). Byte-identical in every ghost the
/// corpus has -- five maps, times from 10.6 s to 235.6 s -- so it is a
/// constant of the format on this build, not something a run determines. It is
/// a u32 version, a u32, and a 12-byte length-prefixed zlib blob.
const CHUNK_0303F006: &[u8] = &[
    0x06, 0xF0, 0x03, 0x03, // chunk id 0x0303F006
    0x01, 0x00, 0x00, 0x00, // version
    0x04, 0x00, 0x00, 0x00, //
    0x0C, 0x00, 0x00, 0x00, // 12 bytes follow
    0x78, 0x9C, 0xFB, 0xFF, 0xFF, 0xFF, 0x7F, 0x00, 0x09, 0xFA, 0x03, 0xFD,
];

/// The GBX body end marker.
const FACADE: u32 = 0xFACA_DE01;

#[derive(Debug)]
enum Part {
    /// A skippable chunk: id, `PIKS`, u32 length, payload.
    Skip { id: u32, payload: Vec<u8> },
    /// A run of non-skippable chunks, kept as parsed items.
    Gap(Vec<Item>),
}

#[derive(Debug)]
enum Item {
    /// The constant CGameGhost chunk, emitted from the built-in above.
    Const0303F006,
    /// A chunk id followed by a fixed-width payload we understand.
    Fixed { id: u32, payload: Vec<u8> },
    /// `[u32 flag][u32 len][len bytes]` -- a bare lookback string.
    LenString { lead: u32, s: String },
    /// A chunk whose payload is an MwId (lookback string): chunk id, the
    /// lookback marker, and the string. `0x03092010` is the map uid.
    MwId { id: u32, lead: u32, s: String },
    /// The 0xFACADE01 end marker.
    End,
    /// Bytes we have not named yet. Their presence is the honest measure of
    /// how far from "from scratch" we still are.
    Unknown(Vec<u8>),
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Split a body into the five skippable spans and the gaps between them.
///
/// `all_skip_chunks` byte-scans for `PIKS`, so inside a NON-skippable chunk's
/// payload it happily reports chunks that are not there -- that is where the
/// old "27 chunks" picture came from, and several of those phantom payloads
/// decode to text fragments landing mid-string. Merging overlapping spans
/// collapses the phantoms back into their real container.
fn split(body: &[u8]) -> Vec<(usize, usize, u32)> {
    let mut spans: Vec<(usize, usize, u32)> = gbx::container::all_skip_chunks(body)
        .into_iter()
        .map(|(id, off, poff, sz)| (off, poff + sz, id))
        .collect();
    spans.sort_by_key(|s| s.0);
    let mut merged: Vec<(usize, usize, u32)> = Vec::new();
    for (s, e, id) in spans {
        match merged.last_mut() {
            // STRICTLY inside the previous span = a phantom found by the byte
            // scan inside someone's payload; swallow it. ADJACENT (s == end) is
            // a real neighbouring chunk and must stay separate -- merging on
            // `<=` glued 0x0303F007 (a 4-byte chunk) onto 0x03092000 (10726 B)
            // and made the whole body look like five giant spans.
            Some(last) if s < last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e, id)),
        }
    }
    merged
}

/// Parse a gap -- a run of non-skippable chunks -- into named items.
fn parse_gap(b: &[u8]) -> Vec<Item> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        if b.len() - i >= CHUNK_0303F006.len() && &b[i..i + CHUNK_0303F006.len()] == CHUNK_0303F006
        {
            out.push(Item::Const0303F006);
            i += CHUNK_0303F006.len();
            continue;
        }
        if b.len() - i >= 4 {
            let id = u32_at(b, i);
            if id == FACADE {
                out.push(Item::End);
                i += 4;
                continue;
            }
            // The chunks that appear in a ghost's gaps, with the payload width
            // each one carries. 0x0309200E and 0x0309201C are hashes we copy
            // rather than compute -- see the note in `emit`.
            let width = match id {
                0x0309_200C => Some(4),
                0x0309_200E => Some(4),
                0x0309_200F => Some(4),
                0x0309_201C => Some(32),
                _ => None,
            };
            if let Some(w) = width {
                if b.len() - i >= 4 + w {
                    out.push(Item::Fixed {
                        id,
                        payload: b[i + 4..i + 4 + w].to_vec(),
                    });
                    i += 4 + w;
                    continue;
                }
            }
            // `0x03092010` carries an MwId, and openplanet names it:
            // CGameCtnGhost::Validate_ChallengeUid. An MwId serialises as a
            // GBX lookback string -- a u32 marker (0x40000000 = "a new entry
            // follows"), then a length-prefixed byte string. That is the map
            // uid, in the clear.
            if id == 0x0309_2010 && b.len() - i >= 12 {
                let lead = u32_at(b, i + 4);
                let n = u32_at(b, i + 8) as usize;
                if n > 0 && n <= 64 && b.len() - i >= 12 + n {
                    let rawu = &b[i + 12..i + 12 + n];
                    if rawu.iter().all(|c| c.is_ascii_graphic()) {
                        out.push(Item::MwId {
                            id,
                            lead,
                            s: String::from_utf8_lossy(rawu).into_owned(),
                        });
                        i += 12 + n;
                        continue;
                    }
                }
            }
            // `[u32 flag][u32 len][len bytes of ASCII]` -- a bare lookback.
            if b.len() - i >= 8 {
                let n = u32_at(b, i + 4) as usize;
                if n > 0 && n <= 64 && b.len() - i >= 8 + n {
                    let raw = &b[i + 8..i + 8 + n];
                    if raw.iter().all(|c| c.is_ascii_graphic()) {
                        out.push(Item::LenString {
                            lead: id,
                            s: String::from_utf8_lossy(raw).into_owned(),
                        });
                        i += 8 + n;
                        continue;
                    }
                }
            }
        }
        // Unnamed: take the rest of this gap in one lump so the count of
        // unknown BYTES is honest rather than split across many items.
        out.push(Item::Unknown(b[i..].to_vec()));
        break;
    }
    out
}

fn emit(parts: &[Part], zero_hashes: bool) -> Vec<u8> {
    let mut o = Vec::new();
    for p in parts {
        match p {
            Part::Skip { id, payload } => {
                o.extend_from_slice(&id.to_le_bytes());
                o.extend_from_slice(SKIP_MAGIC);
                o.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                o.extend_from_slice(payload);
            }
            Part::Gap(items) => {
                for it in items {
                    match it {
                        Item::Const0303F006 => o.extend_from_slice(CHUNK_0303F006),
                        Item::End => o.extend_from_slice(&FACADE.to_le_bytes()),
                        Item::Fixed { id, payload } => {
                            o.extend_from_slice(&id.to_le_bytes());
                            // The two values in a ghost we cannot DERIVE: the
                            // u32 after 0x0309200E and the 32 bytes after
                            // 0x0309201C. Both look like hashes. The dedicated
                            // server does not check either -- flipping every
                            // bit of both still validates -- so `--zero-hashes`
                            // exists to find out whether anything else does.
                            let hashish =
                                matches!(*id, 0x0309_200E | 0x0309_201C);
                            if zero_hashes && hashish {
                                o.extend(std::iter::repeat(0u8).take(payload.len()));
                            } else {
                                o.extend_from_slice(payload);
                            }
                        }
                        Item::LenString { lead, s } => {
                            o.extend_from_slice(&lead.to_le_bytes());
                            o.extend_from_slice(&(s.len() as u32).to_le_bytes());
                            o.extend_from_slice(s.as_bytes());
                        }
                        Item::MwId { id, lead, s } => {
                            o.extend_from_slice(&id.to_le_bytes());
                            o.extend_from_slice(&lead.to_le_bytes());
                            o.extend_from_slice(&(s.len() as u32).to_le_bytes());
                            o.extend_from_slice(s.as_bytes());
                        }
                        Item::Unknown(b) => o.extend_from_slice(b),
                    }
                }
            }
        }
    }
    o
}

pub fn cmd(a: &[String]) {
    // IN OUT, like every other writing command. This shipped as
    // `synth OUT --from DONOR`, which inverts the convention: the output was
    // the positional and the input arrived as a flag, so `ghost synth a b`
    // would have silently meant something different here than everywhere else.
    let inp = a
        .first()
        .unwrap_or_else(|| die("ghost synth IN OUT [--zero-hashes]"));
    let out = a
        .get(1)
        .unwrap_or_else(|| die("ghost synth IN OUT [--zero-hashes]"));
    let from = inp;
    let zero_hashes = a.iter().any(|x| x == "--zero-hashes");

    let raw = std::fs::read(from).unwrap_or_else(|e| die(format!("{}: {}", from, e)));
    let g = Gbx::parse(&raw);
    let body = g.body.clone();

    let spans = split(&body);
    let mut parts: Vec<Part> = Vec::new();
    let mut prev = 0usize;
    for (s, e, id) in &spans {
        if *s > prev {
            parts.push(Part::Gap(parse_gap(&body[prev..*s])));
        }
        // Re-frame the chunk from (id, payload) rather than copying its bytes.
        parts.push(Part::Skip {
            id: *id,
            payload: body[*s + 12..*e].to_vec(),
        });
        prev = *e;
    }
    if prev < body.len() {
        parts.push(Part::Gap(parse_gap(&body[prev..])));
    }

    println!("{} -> {} parts", from, parts.len());
    let mut unknown = 0usize;
    for p in &parts {
        match p {
            Part::Skip { id, payload } => {
                println!("  chunk 0x{:08X}  {:>7} B  (re-framed)", id, payload.len())
            }
            Part::Gap(items) => {
                for it in items {
                    match it {
                        Item::Const0303F006 => println!(
                            "  chunk 0x0303F006  {:>7} B  (built-in constant)",
                            CHUNK_0303F006.len()
                        ),
                        Item::Fixed { id, payload } => println!(
                            "  chunk 0x{:08X}  {:>7} B  (parsed{})",
                            id,
                            payload.len(),
                            if matches!(*id, 0x0309_200E | 0x0309_201C) {
                                ", hash -- copied, not derived"
                            } else {
                                ""
                            }
                        ),
                        Item::LenString { s, .. } => {
                            println!("  lookback string   {:>7} B  \"{}\"", s.len(), s)
                        }
                        Item::MwId { id, s, .. } => println!(
                            "  chunk 0x{:08X}  {:>7} B  MwId Validate_ChallengeUid \"{}\"",
                            id,
                            s.len(),
                            s
                        ),
                        Item::End => println!("  end marker              4 B  0xFACADE01"),
                        Item::Unknown(b) => {
                            unknown += b.len();
                            println!("  UNNAMED           {:>7} B  {}", b.len(), hex(b));
                        }
                    }
                }
            }
        }
    }

    let nb = emit(&parts, zero_hashes);
    let same = nb == body;
    println!(
        "\nemitted body {} B (donor {} B) -- byte-identical: {}",
        nb.len(),
        body.len(),
        same
    );
    // ACCOUNT FOR THE EMBEDDED MAP before reporting what is unexplained.
    //
    // A bare count of unnamed bytes is alarming and usually wrong about what
    // it means. A replay carries the whole .Map.Gbx inside it -- 771,380 of
    // 781,044 bytes in this project's own replay fixture -- and synth does not
    // parse maps, correctly: that is a different format with its own tooling.
    // Reporting "bytes still unnamed: 774657" invites the reading that the
    // container is 99 % unparseable and therefore broken, when the file is
    // perfectly healthy and splices fine.
    //
    // That misreading cost a real investigation: a peer reported 443,299
    // unnamed of 453,169 on a file they could not splice, I took it as an
    // unparseable container, and it was an embedded map -- benign, and not the
    // cause of their failure at all.
    let embedded = gbx::container::embedded_map_in(&body).map(|(_, n)| n).unwrap_or(0);
    let unexplained = unknown.saturating_sub(embedded);
    if embedded > 0 {
        println!(
            "bytes still unnamed: {} -- of which {} are an EMBEDDED MAP (a whole .Map.Gbx \
             inside this file, which synth does not parse and does not need to). \
             Unexplained: {}.",
            unknown, embedded, unexplained
        );
    } else {
        println!("bytes still unnamed: {}", unknown);
    }
    if !same {
        if let Some(k) = (0..nb.len().min(body.len())).find(|&k| nb[k] != body[k]) {
            println!(
                "first difference at body+{}: emitted {:02x}, donor {:02x}",
                k, nb[k], body[k]
            );
        }
    }
    if zero_hashes {
        println!("--zero-hashes: 0x0309200E and 0x0309201C zeroed");
    }

    gbx::container::write_gbx(&g, nb, out).unwrap_or_else(|e| die(e));
    println!("wrote {}", out);
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}
