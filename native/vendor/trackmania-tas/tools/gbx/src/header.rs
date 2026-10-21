//! The GBX **header** user-data block: chunks, string frames, and safe edits.
//!
//! ## Why this exists
//!
//! Everything this toolchain calls "the container" lived in the *body*: the
//! declared-time chunks, the identity strings, the tape, the record. So every
//! check we own reads the body. A `.Replay.Gbx` carries a second, complete copy
//! of the same facts in its **header user-data**, and nothing read it.
//!
//! Measured on this project's own 173691 landing file, after
//! `ghost identity set --anonymise` and `ghost declare --from-oracle` had both
//! reported success and `ghost verify` had passed it:
//!
//! ```text
//! V2  declared-time census: 1 copies, all 36.049        <- body only
//! V3  container identity: ... (nothing foreign)         <- body only
//! ```
//!
//! and in the header, untouched:
//!
//! ```text
//! GothMommyTM
//! 3Awx2_MzSdaCJZjZOht51A
//! <times best="49958" .../>          <- the CARRIER's time, not this run's
//! ```
//!
//! A check that is precise and wrong is worse than no check: "1 copy, all
//! 36.049" is a *count*, and it was counting a set it could not see the rest
//! of. So the header is parsed here, in the format crate, once — not with a
//! grep in whichever tool noticed.
//!
//! ## The layout, as measured
//!
//! ```text
//! user_data := u32 n_chunks
//!              n_chunks x { u32 chunk_id, u32 size_with_heavy_bit }
//!              the chunk bodies, concatenated, in table order
//! ```
//!
//! The top bit of the size word is the "heavy" flag and is not part of the
//! size. Editing a string inside a chunk therefore means editing that chunk's
//! size word too, which is why this is a parse and not a byte patch —
//! `Gbx::header_bytes_u` already writes `user_data.len()`, so the outer size
//! takes care of itself.
//!
//! ## String frames
//!
//! A GBX string is `u32 length` followed by that many bytes. The header's
//! chunk layouts differ per class and per version, and this crate does not
//! model them; instead a chunk is scanned for byte ranges that *are* a string
//! frame — a plausible length, in bounds, whose content is printable UTF-8.
//!
//! **A false positive here cannot corrupt anything**, because nothing is
//! rewritten unless the caller recognises the text: the rewrite is
//! `Fn(&str) -> Option<String>`, and a frame the caller does not recognise is
//! copied through byte for byte. The scan is used to FIND candidates; the
//! caller decides. That is the opposite of a heuristic that edits.

/// One header chunk: its id, its heavy flag, and its bytes.
#[derive(Clone, Debug)]
pub struct HeaderChunk {
    pub id: u32,
    pub heavy: bool,
    pub data: Vec<u8>,
}

const HEAVY: u32 = 0x8000_0000;

/// Parse the `user_data` blob into chunks. `None` if it is not a chunk table
/// (an empty header, or a class whose header this does not describe) — never a
/// guess, and never a partial parse the caller could mistake for a whole one.
pub fn parse_user_data(ud: &[u8]) -> Option<Vec<HeaderChunk>> {
    if ud.len() < 4 {
        return None;
    }
    let n = u32::from_le_bytes(ud[0..4].try_into().ok()?) as usize;
    if n == 0 || n > 64 {
        return None;
    }
    let table_end = 4 + 8 * n;
    if ud.len() < table_end {
        return None;
    }
    let mut sizes = Vec::with_capacity(n);
    let mut total = 0usize;
    for i in 0..n {
        let o = 4 + 8 * i;
        let id = u32::from_le_bytes(ud[o..o + 4].try_into().ok()?);
        let sz = u32::from_le_bytes(ud[o + 4..o + 8].try_into().ok()?);
        let heavy = sz & HEAVY != 0;
        let size = (sz & !HEAVY) as usize;
        total += size;
        sizes.push((id, heavy, size));
    }
    if table_end + total != ud.len() {
        // The table must account for every byte. Anything else and this is not
        // the structure we think it is; say so rather than truncate.
        return None;
    }
    let mut out = Vec::with_capacity(n);
    let mut o = table_end;
    for (id, heavy, size) in sizes {
        out.push(HeaderChunk { id, heavy, data: ud[o..o + size].to_vec() });
        o += size;
    }
    Some(out)
}

/// Rebuild a `user_data` blob from chunks, with every size word recomputed.
pub fn build_user_data(chunks: &[HeaderChunk]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    for c in chunks {
        out.extend_from_slice(&c.id.to_le_bytes());
        let mut sz = c.data.len() as u32;
        if c.heavy {
            sz |= HEAVY;
        }
        out.extend_from_slice(&sz.to_le_bytes());
    }
    for c in chunks {
        out.extend_from_slice(&c.data);
    }
    out
}

/// A candidate string frame inside a chunk: `u32 len` at `off`, then `len`
/// bytes of text.
#[derive(Clone, Debug)]
pub struct StrFrame {
    /// Offset of the LENGTH word within the chunk's data.
    pub off: usize,
    pub text: String,
}

/// The longest sensible header string. The replay XML blob is about 700 bytes;
/// a megabyte cap is generous and keeps the scan from adopting a length word
/// that is really a float.
const MAX_STR: usize = 1 << 20;

fn plausible(bytes: &[u8]) -> Option<String> {
    // Reject control characters outright: real header strings are logins,
    // nicknames, zone paths, uids and one XML document.
    if bytes.iter().any(|b| *b < 0x09 || (*b > 0x0d && *b < 0x20)) {
        return None;
    }
    std::str::from_utf8(bytes).ok().map(|s| s.to_string())
}

/// Every byte range in `data` that reads as a string frame.
///
/// Overlapping interpretations are resolved by taking the first and skipping
/// past it, which is what a real reader does.
pub fn string_frames(data: &[u8]) -> Vec<StrFrame> {
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 4 <= data.len() {
        let n = u32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as usize;
        if n > 0 && n <= MAX_STR && o + 4 + n <= data.len() {
            if let Some(text) = plausible(&data[o + 4..o + 4 + n]) {
                out.push(StrFrame { off: o, text });
                o += 4 + n;
                continue;
            }
        }
        o += 1;
    }
    out
}

/// Rewrite the string frames a caller recognises, fixing each frame's own
/// length word. Returns the new bytes and how many frames changed.
///
/// `f` sees every candidate frame's text and returns `Some(replacement)` only
/// for the ones it recognises. A frame it declines is copied byte for byte, so
/// `replace_frames(x, |_| None)` is the identity — which is a test.
pub fn replace_frames<F: Fn(&str) -> Option<String>>(data: &[u8], f: F) -> (Vec<u8>, usize) {
    let frames = string_frames(data);
    let mut out = Vec::with_capacity(data.len());
    let mut cur = 0usize;
    let mut changed = 0usize;
    for fr in frames {
        let Some(new) = f(&fr.text) else { continue };
        if new == fr.text {
            continue;
        }
        out.extend_from_slice(&data[cur..fr.off]);
        out.extend_from_slice(&(new.len() as u32).to_le_bytes());
        out.extend_from_slice(new.as_bytes());
        cur = fr.off + 4 + fr.text.len();
        changed += 1;
    }
    out.extend_from_slice(&data[cur..]);
    (out, changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u32).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }

    #[test]
    fn a_chunk_table_round_trips() {
        let chunks = vec![
            HeaderChunk { id: 0x03093000, heavy: false, data: frame("abc") },
            HeaderChunk { id: 0x03093002, heavy: true, data: frame("GothMommyTM") },
        ];
        let ud = build_user_data(&chunks);
        let back = parse_user_data(&ud).expect("parses");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].id, 0x03093000);
        assert!(back[1].heavy);
        assert_eq!(build_user_data(&back), ud);
    }

    #[test]
    fn a_table_that_does_not_account_for_every_byte_is_refused() {
        let mut ud = build_user_data(&[HeaderChunk { id: 1, heavy: false, data: frame("x") }]);
        ud.push(0); // one byte nobody claims
        assert!(parse_user_data(&ud).is_none());
    }

    #[test]
    fn declining_every_frame_is_the_identity() {
        let mut d = frame("hello");
        d.extend_from_slice(&frame("world"));
        let (out, n) = replace_frames(&d, |_| None);
        assert_eq!(n, 0);
        assert_eq!(out, d);
    }

    #[test]
    fn a_replacement_of_a_different_length_fixes_its_own_length_word() {
        let mut d = frame("GothMommyTM");
        d.extend_from_slice(&frame("keep me"));
        let (out, n) =
            replace_frames(&d, |s| if s == "GothMommyTM" { Some("TAS".into()) } else { None });
        assert_eq!(n, 1);
        let frames = string_frames(&out);
        let texts: Vec<&str> = frames.iter().map(|f| f.text.as_str()).collect();
        assert!(texts.contains(&"TAS"), "{:?}", texts);
        assert!(texts.contains(&"keep me"), "{:?}", texts);
        assert_eq!(out.len(), d.len() - 8);
    }

    #[test]
    fn a_length_word_that_runs_past_the_end_is_not_a_string() {
        let d = [0xff, 0xff, 0x00, 0x00, b'a', b'b'];
        assert!(string_frames(&d).iter().all(|f| f.text != "ab"));
    }
}
