//! The TM2020 ghost INPUT TAPE: a lossless, editable, per-tick text format.
//!
//! WHAT THE PACKET ACTUALLY CARRIES
//! --------------------------------
//! Chunk `0x0309201D` holds one or more *input archives*. Each archive is a
//! bit-packed stream of one packet per 10 ms tick. A packet is, in order:
//!
//!   1. a STATE word, coded three ways:
//!        `1`              -> repeat the previous packet's word
//!        `0 1 x y`        -> repeat it, overriding flag bits 0 and 1
//!        `0 0 <literal>`  -> an explicit 33-bit (format 11) or 34-bit
//!                            (format 12) literal
//!      The literal unpacks into `word0` and 22 `flags` bits. `word0 & 0xF` is
//!      the packet MODE, which decides what the rest of the packet contains,
//!      and **bit 31 of the literal is the RESPAWN input** (it lands in
//!      `word0` bit 5).
//!   2. a MOUSE segment: `1` for none, or `0` plus two 16-bit axes.
//!   3. the VEHICLE fields, per mode:
//!        mode 2, 4  : same-bit, else steer:8, accel:1, brake:1
//!        mode 12    : same-bit, else steer:32, accel:1, brake:1
//!        mode 13    : same-bit, else steer:32
//!        mode 0     : nothing
//!        otherwise  : same-bit, else four 2-bit trigger fields
//!
//! THE "SAME AS PREVIOUS TICK" BIT IS NOT A FROZEN TICK. It is one bit with no
//! fields behind it. To write a different input there you EXPAND the packet
//! into its explicit form. `Tape::encode(Encoding::Explicit)` does exactly
//! that, and `Encoding::Verbatim` reproduces the original bitstream byte for
//! byte -- which is the round-trip control this module is tested with.

use crate::bits::{BitReader, BitWriter};
use crate::container::{Gbx, SKIP_MAGIC};

pub const INPUTS_CHUNK_ID: u32 = 0x0309201D;

/// How the state word was coded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateEnc {
    /// `0 0 <literal>`
    Lit(u64),
    /// `1`
    Prev,
    /// `0 1 x y`
    Prev2(u8, u8),
}

#[derive(Clone, Debug)]
pub struct Packet {
    pub word0: u32,
    pub flags: u32,
    pub mode: u32,
    pub state: StateEnc,
    /// `None` = the one-bit "no mouse" form; `Some((a, b))` = two 16-bit axes.
    pub mouse: Option<(u16, u16)>,
    /// True when the vehicle fields were coded as the one-bit "same as the
    /// previous packet" form.
    pub vsame: bool,
    /// Raw steer field value (8 or 32 bits wide, per mode).
    pub steer: u32,
    pub accel: u32,
    pub brake: u32,
    /// Four 2-bit trigger fields, for the modes that carry them.
    pub tri: Option<[u8; 4]>,
}

impl Packet {
    fn blank() -> Packet {
        Packet {
            word0: 0,
            flags: 0,
            mode: 2,
            state: StateEnc::Prev,
            mouse: None,
            vsame: false,
            steer: 0,
            accel: 0,
            brake: 0,
            tri: None,
        }
    }
    /// The respawn input: bit 31 of the state literal, which unpacks into
    /// `word0` bit 5. Only an explicit literal can carry it; a repeated word
    /// always reads 0, so setting it on a repeated packet forces an expansion.
    pub fn respawn(&self) -> bool {
        self.word0 & 0x20 != 0
    }
    /// Steer as the game reads it for an 8-bit field: a signed i8 over 127.
    pub fn steer_i8(&self) -> i8 {
        (self.steer & 0xFF) as u8 as i8
    }
}

/// `unpack_word`, for code that BUILDS a packet rather than decoding one.
///
/// A synthesizer choosing a state literal needs the `word0`/`flags` the decoder
/// will derive from it, and it must be the same arithmetic — a second
/// implementation of this unpacking is how the two halves drift apart and a
/// tape reads back as a mode nobody wrote.
pub fn unpack_word_pub(n: u64) -> (u32, u32) {
    unpack_word(n)
}

/// `unpack_word`: literal (33/34-bit int) -> (word0, flags).
fn unpack_word(n: u64) -> (u32, u32) {
    let lo = (n & 0xFFFF_FFFF) as u32;
    let hi = ((n >> 32) & 0xFFFF_FFFF) as u32;
    let flags = ((n >> 5) & 0x3F_FFFF) as u32;
    let lo_s = lo as i32 as i64;
    let word0 = ((((((hi & 2) << 5) | (hi & 1)) << 6) as u64)
        | ((lo_s >> 20) as u64 & 1920)
        | ((n >> 26) & 0x20)
        | (n & 0x1F)) as u32;
    (word0, flags)
}

/// `pack_prev`: rebuild a literal from the previous word0/flags. Lossy by
/// design -- this is what the format itself does for a repeated word.
fn pack_prev(word0: u32, flags: u32) -> u64 {
    (((flags & 0x3F_FFFF) as u64) << 5) | ((word0 & 0xF) as u64)
}

/// The literal a repeated state word is equivalent to.
///
/// A `w=prev` / `w=prev2` packet carries no state literal, and the respawn
/// input lives in bit 31 of one -- so on a tape whose words are mostly
/// repeated (566 327 packets, 18 997 input events) a respawn cannot be
/// WRITTEN at all, which is the harness gap `FK.md` G1 names from the other
/// side. This turns a repeat into the literal the decoder itself derives from
/// it, after which the bit is writable. It is the format's own arithmetic, so
/// re-decoding the result reproduces the same word0/flags.
pub fn literal_for(word0: u32, flags: u32) -> u64 {
    pack_prev(word0, flags)
}

#[derive(Clone, Debug)]
pub struct Archive {
    pub format_version: u32,
    pub field0: u32,
    pub start_offset_ms: i32,
    pub packets: Vec<Packet>,
    /// Byte length of the archive's bitstream as it was read.
    pub orig_bitstream_len: usize,
    /// How many BITS the packets actually consumed. Real files have trailing
    /// bytes past this that the packet count does not cover; a verbatim
    /// re-encode has to carry them through untouched or it is not an identity.
    pub orig_bits_used: usize,
    /// The bitstream bytes from `orig_bits_used >> 3` to the end: the partial
    /// last byte plus whatever the game wrote after the packets.
    pub tail: Vec<u8>,
    /// The original bitstream, kept so `Encoding::Verbatim` can be proved
    /// against it byte for byte.
    pub orig_bitstream: Vec<u8>,
}

impl Archive {
    fn state_bits(&self) -> usize {
        if self.format_version == 12 {
            34
        } else {
            33
        }
    }
}

/// A whole `0x0309201D` chunk: the version word and every archive in it.
#[derive(Clone, Debug)]
pub struct Tape {
    pub chunk_version: u32,
    pub archives: Vec<Archive>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Reproduce the original coding decisions exactly (same-bits kept as
    /// same-bits). Byte-identical to the input when nothing was edited.
    Verbatim,
    /// Write every vehicle field explicitly, so every tick is patchable and no
    /// input silently inherits the previous tick's.
    Explicit,
}

fn decode_archive(a: &mut Archive, bitstream: &[u8], packet_count: usize) -> Result<(), String> {
    if a.format_version != 11 && a.format_version != 12 {
        return Err(format!(
            "unsupported ghost input format version {} (this tool knows 11 and 12)",
            a.format_version
        ));
    }
    let sb = a.state_bits();
    let mut r = BitReader::new(bitstream);
    let mut prev = Packet::blank();
    let mut out = Vec::with_capacity(packet_count);
    for _ in 0..packet_count {
        let mut p = Packet::blank();
        if r.bit() == 1 {
            p.state = StateEnc::Prev;
            let (w, f) = unpack_word(pack_prev(prev.word0, prev.flags));
            p.word0 = w;
            p.flags = f;
        } else if r.bit() == 0 {
            let lit = r.bits(sb);
            p.state = StateEnc::Lit(lit);
            let (w, f) = unpack_word(lit);
            p.word0 = w;
            p.flags = f;
        } else {
            let (w, f) = unpack_word(pack_prev(prev.word0, prev.flags));
            p.word0 = w;
            p.flags = f;
            let x = r.bit() as u8;
            let y = r.bit() as u8;
            p.state = StateEnc::Prev2(x, y);
            p.flags = (p.flags & !1) | (x as u32);
            p.flags = (p.flags & !2) | ((y as u32 & 1) * 2);
        }

        if r.bit() == 0 {
            let a14 = r.bits(16) as u16;
            let a16 = r.bits(16) as u16;
            p.mouse = Some((a14, a16));
        } else {
            p.mouse = None;
        }

        p.mode = p.word0 & 0xF;
        match p.mode {
            2 | 4 | 12 => {
                p.vsame = r.bit() != 0;
                if p.vsame {
                    p.steer = prev.steer;
                    p.accel = prev.accel;
                    p.brake = prev.brake;
                } else {
                    let nb = if p.mode == 12 { 32 } else { 8 };
                    p.steer = r.bits(nb) as u32;
                    p.accel = r.bit() as u32;
                    p.brake = r.bit() as u32;
                }
            }
            13 => {
                p.vsame = r.bit() != 0;
                if p.vsame {
                    p.steer = prev.steer;
                } else {
                    p.steer = r.bits(32) as u32;
                }
            }
            0 => {}
            _ => {
                p.vsame = r.bit() != 0;
                if p.vsame {
                    p.tri = prev.tri.or(Some([0, 0, 0, 0]));
                } else {
                    p.tri = Some([
                        r.bits(2) as u8,
                        r.bits(2) as u8,
                        r.bits(2) as u8,
                        r.bits(2) as u8,
                    ]);
                }
            }
        }
        prev = p.clone();
        out.push(p);
    }
    a.packets = out;
    a.orig_bits_used = r.pos;
    a.tail = bitstream[(r.pos >> 3).min(bitstream.len())..].to_vec();
    Ok(())
}

/// Re-encode one archive. `enc` decides whether the "same as previous" coding
/// is kept (Verbatim) or expanded away (Explicit).
///
/// A packet whose value no longer equals the previous packet's is ALWAYS
/// written explicitly, whatever `enc` says: keeping a same-bit there would
/// silently discard the edit, which is the exact failure this API exists to
/// make impossible.
pub fn encode_archive(a: &Archive, enc: Encoding) -> Vec<u8> {
    let sb = a.state_bits();
    let mut w = BitWriter::new();
    // The decoder starts from a BLANK packet, not from "no packet": a first
    // packet coded as "same as the previous tick" inherits zeros. The encoder
    // has to agree, or the very first same-bit is expanded and every byte after
    // it shifts.
    let blank = Packet::blank();
    let mut prev: &Packet = &blank;
    for p in &a.packets {
        match p.state {
            StateEnc::Prev => w.bit(1),
            StateEnc::Lit(l) => {
                w.bit(0);
                w.bit(0);
                w.bits(l, sb);
            }
            StateEnc::Prev2(x, y) => {
                w.bit(0);
                w.bit(1);
                w.bit(x as u64);
                w.bit(y as u64);
            }
        }
        match p.mouse {
            None => w.bit(1),
            Some((x, y)) => {
                w.bit(0);
                w.bits(x as u64, 16);
                w.bits(y as u64, 16);
            }
        }
        let inherits = {
            let q = prev;
            match p.mode {
                2 | 4 | 12 => p.steer == q.steer && p.accel == q.accel && p.brake == q.brake,
                13 => p.steer == q.steer,
                0 => true,
                _ => p.tri == q.tri.or(Some([0, 0, 0, 0])),
            }
        };
        let same = enc == Encoding::Verbatim && p.vsame && inherits;
        match p.mode {
            2 | 4 | 12 => {
                if same {
                    w.bit(1);
                } else {
                    w.bit(0);
                    w.bits(p.steer as u64, if p.mode == 12 { 32 } else { 8 });
                    w.bit(p.accel as u64);
                    w.bit(p.brake as u64);
                }
            }
            13 => {
                if same {
                    w.bit(1);
                } else {
                    w.bit(0);
                    w.bits(p.steer as u64, 32);
                }
            }
            0 => {}
            _ => {
                if same {
                    w.bit(1);
                } else {
                    w.bit(0);
                    for v in p.tri.unwrap_or([0, 0, 0, 0]) {
                        w.bits(v as u64, 2);
                    }
                }
            }
        }
        prev = p;
    }
    // Trailing bytes. A real recording's bitstream is longer than its packets
    // need -- 11 to 33 bytes on the files measured here -- and those bytes are
    // part of the file. Verbatim carries them through unchanged, which is what
    // makes the round-trip an identity rather than an approximation. The
    // explicit form cannot: every bit position after the first expansion has
    // moved, so the tail is dropped (the reader stops after `packet_count`
    // packets, which is why the game and the server accept such a file).
    if enc == Encoding::Verbatim && a.orig_bits_used > 0 && w.pos == a.orig_bits_used && !a.tail.is_empty() {
        let base = a.orig_bits_used & !7usize;
        let total = base + a.tail.len() * 8;
        let mut b = w.pos;
        while b % 8 != 0 && b < total {
            let v = (a.tail[(b - base) >> 3] >> (b & 7)) & 1;
            w.bits(v as u64, 1);
            b += 1;
        }
        if b < total {
            w.buf.extend_from_slice(&a.tail[(b - base) / 8..]);
            w.pos = total;
        }
    }
    w.buf
}

/// EVERY input chunk in a body, in file order: (chunk offset, payload offset,
/// size). A solo ghost has one. A SERVER RECORDING has one per life -- this
/// map's practice replay has 46 -- and `find_inputs_chunk` returns only the
/// first, which is how a naive reader silently reads life 0 and calls it the
/// run. The walk advances past each payload it accepts, so a chunk id that
/// happens to occur inside an accepted payload cannot be counted twice.
pub fn find_inputs_chunks(body: &[u8]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 12 <= body.len() {
        if u32::from_le_bytes(body[i..i + 4].try_into().unwrap()) == INPUTS_CHUNK_ID
            && &body[i + 4..i + 8] == SKIP_MAGIC
        {
            let size = u32::from_le_bytes(body[i + 8..i + 12].try_into().unwrap()) as usize;
            if i + 12 + size <= body.len() {
                out.push((i, i + 12, size));
                i += 12 + size;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Where the input chunk sits in a body: (chunk offset, payload offset, size).
pub fn find_inputs_chunk(body: &[u8]) -> Option<(usize, usize, usize)> {
    let mut i = 0usize;
    while i + 12 <= body.len() {
        if u32::from_le_bytes(body[i..i + 4].try_into().unwrap()) == INPUTS_CHUNK_ID
            && &body[i + 4..i + 8] == SKIP_MAGIC
        {
            let size = u32::from_le_bytes(body[i + 8..i + 12].try_into().unwrap()) as usize;
            if i + 12 + size <= body.len() {
                return Some((i, i + 12, size));
            }
        }
        i += 1;
    }
    None
}

impl Tape {
    /// Decode every archive of a ghost's input chunk.
    pub fn from_body(body: &[u8]) -> Result<Tape, String> {
        Tape::from_body_nth(body, 0)
    }

    /// Decode the Nth input chunk of a body (0 = the first, what every stock
    /// reader takes).
    pub fn from_body_nth(body: &[u8], nth: usize) -> Result<Tape, String> {
        let all = find_inputs_chunks(body);
        if all.is_empty() {
            return Err("no 0x0309201D input chunk in this file".into());
        }
        if nth >= all.len() {
            return Err(format!("--nth {} but this file has {} input chunks", nth, all.len()));
        }
        let (_, payload_off, payload_size) = all[nth];
        let pay = &body[payload_off..payload_off + payload_size];
        let ver = u32::from_le_bytes(pay[0..4].try_into().unwrap());
        if ver > 4 {
            return Err(format!("unsupported input chunk version {}", ver));
        }
        let count = u32::from_le_bytes(pay[4..8].try_into().unwrap()) as usize;
        let mut o = 8usize;
        let mut archives = Vec::new();
        for _ in 0..count {
            let fv = u32::from_le_bytes(pay[o..o + 4].try_into().unwrap());
            let f0 = u32::from_le_bytes(pay[o + 4..o + 8].try_into().unwrap());
            let so = i32::from_le_bytes(pay[o + 8..o + 12].try_into().unwrap());
            let pc = u32::from_le_bytes(pay[o + 12..o + 16].try_into().unwrap()) as usize;
            let bl = u32::from_le_bytes(pay[o + 16..o + 20].try_into().unwrap()) as usize;
            o += 20;
            let mut a = Archive {
                format_version: fv,
                field0: f0,
                start_offset_ms: so,
                packets: Vec::new(),
                orig_bitstream_len: bl,
                orig_bits_used: 0,
                tail: Vec::new(),
                orig_bitstream: pay[o..o + bl].to_vec(),
            };
            decode_archive(&mut a, &pay[o..o + bl], pc)?;
            archives.push(a);
            o += bl;
        }
        Ok(Tape { chunk_version: ver, archives })
    }

    pub fn from_file(path: &str) -> Result<Tape, String> {
        Tape::from_file_nth(path, 0)
    }

    pub fn from_file_nth(path: &str, nth: usize) -> Result<Tape, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        let g = Gbx::parse(&data);
        Tape::from_body_nth(&g.body, nth)
    }

    /// Serialise the whole chunk payload.
    pub fn to_payload(&self, enc: Encoding) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.chunk_version.to_le_bytes());
        out.extend_from_slice(&(self.archives.len() as u32).to_le_bytes());
        for a in &self.archives {
            let bs = encode_archive(a, enc);
            out.extend_from_slice(&a.format_version.to_le_bytes());
            out.extend_from_slice(&a.field0.to_le_bytes());
            out.extend_from_slice(&a.start_offset_ms.to_le_bytes());
            out.extend_from_slice(&(a.packets.len() as u32).to_le_bytes());
            out.extend_from_slice(&(bs.len() as u32).to_le_bytes());
            out.extend_from_slice(&bs);
        }
        out
    }

    /// The index of the first packet whose verbatim re-encode diverges from the
    /// original, found by re-encoding growing prefixes. For diagnosing a codec
    /// gap on a file the round-trip control rejects.
    pub fn first_divergent_packet(&self, ai: usize) -> Option<usize> {
        let a = self.archives.get(ai)?;
        let orig = &a.orig_bitstream;
        let mut lo = 0usize;
        let mut hi = a.packets.len();
        // A prefix of k packets must be a bit-prefix of the original stream.
        let ok = |k: usize| -> bool {
            let mut sub = a.clone();
            sub.packets.truncate(k);
            let re = encode_archive(&sub, Encoding::Verbatim);
            if re.len() > orig.len() {
                return false;
            }
            // the last byte of a prefix is partial; compare all but it
            let n = re.len().saturating_sub(1);
            re[..n] == orig[..n]
        };
        if ok(hi) {
            return None;
        }
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if ok(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(lo)
    }

    /// True when a Verbatim re-encode of every archive reproduces the bytes it
    /// was read from. This is the codec's own identity control and every write
    /// path asserts it before it is allowed to touch a file.
    pub fn verbatim_is_identity(&self) -> Result<(), String> {
        for (i, a) in self.archives.iter().enumerate() {
            if a.orig_bitstream.is_empty() {
                continue; // a tape parsed from text has nothing to compare with
            }
            let re = encode_archive(a, Encoding::Verbatim);
            if re != a.orig_bitstream {
                let n = re.len().min(a.orig_bitstream.len());
                let bad = (0..n).find(|k| re[*k] != a.orig_bitstream[*k]);
                return Err(format!(
                    "archive {}: verbatim re-encode is {} B against the file's {} B{}",
                    i,
                    re.len(),
                    a.orig_bitstream.len(),
                    match bad {
                        Some(k) => format!(", first differing byte {} ({:02x} vs {:02x})", k, re[k], a.orig_bitstream[k]),
                        None => String::new(),
                    }
                ));
            }
        }
        Ok(())
    }

    /// Write this tape back into `body`, returning the new body.
    pub fn splice_into(&self, body: &[u8], enc: Encoding) -> Result<Vec<u8>, String> {
        self.splice_into_nth(body, enc, 0)
    }

    pub fn splice_into_nth(&self, body: &[u8], enc: Encoding, nth: usize) -> Result<Vec<u8>, String> {
        let all = find_inputs_chunks(body);
        if nth >= all.len() {
            return Err(format!("--nth {} but this file has {} input chunks", nth, all.len()));
        }
        let (chunk_off, payload_off, payload_size) = all[nth];
        let pay = self.to_payload(enc);
        let mut out = Vec::with_capacity(body.len() + pay.len());
        out.extend_from_slice(&body[..chunk_off + 8]);
        out.extend_from_slice(&(pay.len() as u32).to_le_bytes());
        out.extend_from_slice(&pay);
        out.extend_from_slice(&body[payload_off + payload_size..]);
        Ok(out)
    }

    pub fn n(&self) -> usize {
        self.archives.first().map(|a| a.packets.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// The text format
// ---------------------------------------------------------------------------

pub const TAPE_FORMAT: &str = "gtape 1";

fn fmt_packet(p: &Packet) -> String {
    let mut s = String::with_capacity(96);
    s.push_str(&format!("mode={}", p.mode));
    match p.state {
        StateEnc::Lit(l) => s.push_str(&format!(" w=lit:0x{:09X}", l)),
        StateEnc::Prev => s.push_str(" w=prev"),
        StateEnc::Prev2(x, y) => s.push_str(&format!(" w=prev2:{},{}", x, y)),
    }
    s.push_str(&format!(" respawn={}", p.respawn() as u8));
    match p.mouse {
        None => s.push_str(" mouse=none"),
        Some((a, b)) => s.push_str(&format!(" mouse={},{}", a, b)),
    }
    s.push_str(&format!(" vsame={}", p.vsame as u8));
    match p.mode {
        2 | 4 => s.push_str(&format!(
            " steer={} accel={} brake={}",
            p.steer_i8(),
            p.accel,
            p.brake
        )),
        12 => s.push_str(&format!(
            " steer32=0x{:08X} accel={} brake={}",
            p.steer, p.accel, p.brake
        )),
        13 => s.push_str(&format!(" steer32=0x{:08X}", p.steer)),
        0 => {}
        _ => {
            let t = p.tri.unwrap_or([0, 0, 0, 0]);
            s.push_str(&format!(" tri={},{},{},{}", t[0], t[1], t[2], t[3]));
        }
    }
    s.push_str(&format!(" flags=0x{:06X}", p.flags));
    s
}

impl Tape {
    /// The full-fidelity text rendering: one line per tick, every field the
    /// packet carries, named.
    pub fn to_text(&self, src: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!("#{}\n", TAPE_FORMAT));
        s.push_str(&format!("#source {}\n", src));
        s.push_str(&format!("#chunk_version {}\n", self.chunk_version));
        s.push_str(
            "# t = tick index (10 ms each). race time ms = (t * 10) + start_offset_ms.\n\
             # w      how the state word is coded: lit:<hex literal> | prev | prev2:<x>,<y>\n\
             # respawn  bit 31 of the state literal -- a real, editable input\n\
             # vsame  1 = the vehicle fields were coded as the one-bit \"same as the\n\
             #        previous tick\" form. Editing the values below expands it.\n\
             # flags  the 22 state flag bits, derived from w (informational)\n",
        );
        for (ai, a) in self.archives.iter().enumerate() {
            s.push_str(&format!(
                "@archive {} format_version={} field0={} start_offset_ms={} packets={} bitstream_bytes={} bits_used={}\n",
                ai,
                a.format_version,
                a.field0,
                a.start_offset_ms,
                a.packets.len(),
                a.orig_bitstream_len,
                a.orig_bits_used
            ));
            if !a.tail.is_empty() {
                let hex: String = a.tail.iter().map(|b| format!("{:02x}", b)).collect();
                s.push_str(&format!("@tail {}\n", hex));
            }
            for (i, p) in a.packets.iter().enumerate() {
                s.push_str(&format!("t={} {}\n", i, fmt_packet(p)));
            }
        }
        s
    }

    /// Parse the text rendering back. Every field is authoritative; `respawn`
    /// wins over the literal's bit 31, so setting it is enough to change the
    /// input (an expansion is forced when the word was a repeat).
    pub fn from_text(txt: &str) -> Result<Tape, String> {
        let mut chunk_version = 0u32;
        let mut archives: Vec<Archive> = Vec::new();
        let mut saw_header = false;
        for (lineno, line) in txt.lines().enumerate() {
            let l = line.trim();
            let err = |m: String| format!("line {}: {}", lineno + 1, m);
            if l.is_empty() {
                continue;
            }
            if let Some(rest) = l.strip_prefix('#') {
                let r = rest.trim();
                if r.starts_with("gtape ") {
                    saw_header = true;
                    if r != TAPE_FORMAT {
                        return Err(err(format!(
                            "tape format {:?}, this build writes {:?}",
                            r, TAPE_FORMAT
                        )));
                    }
                } else if let Some(v) = r.strip_prefix("chunk_version ") {
                    chunk_version = v.trim().parse().map_err(|_| err("chunk_version".into()))?;
                }
                continue;
            }
            if let Some(rest) = l.strip_prefix("@archive ") {
                let kv = parse_kv(rest);
                archives.push(Archive {
                    format_version: kv_num(&kv, "format_version").map_err(&err)? as u32,
                    field0: kv_num(&kv, "field0").map_err(&err)? as u32,
                    start_offset_ms: kv_num(&kv, "start_offset_ms").map_err(&err)? as i32,
                    packets: Vec::new(),
                    orig_bitstream_len: kv_num(&kv, "bitstream_bytes").unwrap_or(0) as usize,
                    orig_bits_used: kv_num(&kv, "bits_used").unwrap_or(0) as usize,
                    tail: Vec::new(),
                    orig_bitstream: Vec::new(),
                });
                continue;
            }
            if let Some(rest) = l.strip_prefix("@tail ") {
                let h = rest.trim();
                let mut t = Vec::with_capacity(h.len() / 2);
                let bytes = h.as_bytes();
                let mut k = 0;
                while k + 1 < bytes.len() {
                    t.push(
                        u8::from_str_radix(&h[k..k + 2], 16)
                            .map_err(|_| err("@tail is not hex".into()))?,
                    );
                    k += 2;
                }
                archives.last_mut().ok_or_else(|| err("@tail before @archive".into()))?.tail = t;
                continue;
            }
            if !l.starts_with("t=") {
                return Err(err(format!("unrecognised line {:?}", l)));
            }
            let a = archives.last_mut().ok_or_else(|| err("a t= line before any @archive".into()))?;
            let kv = parse_kv(l);
            let idx = kv_num(&kv, "t").map_err(&err)? as usize;
            if idx != a.packets.len() {
                return Err(err(format!(
                    "tick out of order: t={} but this archive has {} packets so far",
                    idx,
                    a.packets.len()
                )));
            }
            let mut p = Packet::blank();
            p.mode = kv_num(&kv, "mode").map_err(&err)? as u32;
            let w = kv.iter().find(|(k, _)| k == "w").map(|(_, v)| v.clone()).ok_or_else(|| err("no w=".into()))?;
            p.state = if w == "prev" {
                StateEnc::Prev
            } else if let Some(h) = w.strip_prefix("lit:0x") {
                StateEnc::Lit(u64::from_str_radix(h, 16).map_err(|_| err("w=lit".into()))?)
            } else if let Some(xy) = w.strip_prefix("prev2:") {
                let (x, y) = xy.split_once(',').ok_or_else(|| err("w=prev2".into()))?;
                StateEnc::Prev2(
                    x.parse().map_err(|_| err("w=prev2 x".into()))?,
                    y.parse().map_err(|_| err("w=prev2 y".into()))?,
                )
            } else {
                return Err(err(format!("unknown w={:?}", w)));
            };
            let respawn = kv_num(&kv, "respawn").unwrap_or(0) != 0;
            let mouse = kv.iter().find(|(k, _)| k == "mouse").map(|(_, v)| v.clone());
            p.mouse = match mouse.as_deref() {
                None | Some("none") => None,
                Some(v) => {
                    let (x, y) = v.split_once(',').ok_or_else(|| err("mouse".into()))?;
                    Some((
                        x.parse().map_err(|_| err("mouse x".into()))?,
                        y.parse().map_err(|_| err("mouse y".into()))?,
                    ))
                }
            };
            p.vsame = kv_num(&kv, "vsame").unwrap_or(0) != 0;
            if let Ok(v) = kv_num(&kv, "steer") {
                p.steer = (v as i64 as i8) as u8 as u32;
            }
            if let Some((_, v)) = kv.iter().find(|(k, _)| k == "steer32") {
                let h = v.strip_prefix("0x").unwrap_or(v);
                p.steer = u32::from_str_radix(h, 16).map_err(|_| err("steer32".into()))?;
            }
            p.accel = kv_num(&kv, "accel").unwrap_or(0) as u32;
            p.brake = kv_num(&kv, "brake").unwrap_or(0) as u32;
            if let Some((_, v)) = kv.iter().find(|(k, _)| k == "tri") {
                let n: Vec<u8> = v.split(',').filter_map(|x| x.parse().ok()).collect();
                if n.len() != 4 {
                    return Err(err("tri needs four values".into()));
                }
                p.tri = Some([n[0], n[1], n[2], n[3]]);
            }
            // Respawn is authoritative. Bit 31 of the literal is the only place
            // it can live, so asking for one on a repeated word expands it.
            match p.state {
                StateEnc::Lit(l) => {
                    let nl = if respawn { l | (1 << 31) } else { l & !(1u64 << 31) };
                    p.state = StateEnc::Lit(nl);
                }
                _ if respawn => {
                    return Err(err(
                        "respawn=1 on a w=prev / w=prev2 packet: the bit lives in the state \
                         literal, so rewrite that tick as w=lit:<hex> first (`ghost tape expand` \
                         does it for the whole tape)"
                            .into(),
                    ));
                }
                _ => {}
            }
            a.packets.push(p);
        }
        if !saw_header {
            return Err("not a gtape file (no `#gtape 1` header)".into());
        }
        if archives.is_empty() {
            return Err("no @archive line".into());
        }
        // resolve word0/flags exactly as the decoder does, so `mode` in the
        // file is checked rather than trusted
        for a in archives.iter_mut() {
            let mut prevw = (0u32, 0u32);
            for (i, p) in a.packets.iter_mut().enumerate() {
                let (w, f) = match p.state {
                    StateEnc::Lit(l) => unpack_word(l),
                    StateEnc::Prev => unpack_word(pack_prev(prevw.0, prevw.1)),
                    StateEnc::Prev2(x, y) => {
                        let (w, mut f) = unpack_word(pack_prev(prevw.0, prevw.1));
                        f = (f & !1) | (x as u32);
                        f = (f & !2) | ((y as u32 & 1) * 2);
                        (w, f)
                    }
                };
                if w & 0xF != p.mode {
                    return Err(format!(
                        "tick {}: the state word decodes to mode {} but the line says mode={}",
                        i,
                        w & 0xF,
                        p.mode
                    ));
                }
                p.word0 = w;
                p.flags = f;
                prevw = (w, f);
            }
        }
        Ok(Tape { chunk_version, archives })
    }
}

fn parse_kv(s: &str) -> Vec<(String, String)> {
    s.split_whitespace()
        .filter_map(|tok| tok.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect()
}

fn kv_num(kv: &[(String, String)], key: &str) -> Result<i64, String> {
    let v = kv
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
        .ok_or_else(|| format!("missing {}=", key))?;
    v.parse::<i64>().map_err(|_| format!("{}={:?} is not an integer", key, v))
}


impl Tape {
    /// The decoded per-tick input channels of archive 0, as plain slices.
    /// Callers that only want "what did the driver do on tick i" should not
    /// have to know how a packet is coded.
    pub fn steer_i8s(&self) -> Vec<i8> {
        self.archives.first().map(|a| a.packets.iter().map(|p| p.steer_i8()).collect()).unwrap_or_default()
    }
    pub fn accels(&self) -> Vec<u8> {
        self.archives.first().map(|a| a.packets.iter().map(|p| p.accel as u8).collect()).unwrap_or_default()
    }
    pub fn brakes(&self) -> Vec<u8> {
        self.archives.first().map(|a| a.packets.iter().map(|p| p.brake as u8).collect()).unwrap_or_default()
    }
    pub fn respawns(&self) -> Vec<bool> {
        self.archives.first().map(|a| a.packets.iter().map(|p| p.respawn()).collect()).unwrap_or_default()
    }
    /// Race time in ms of tick `i` of archive 0. The offset is why: most of
    /// this project's incumbents are countdown-prefixed, so tick 0 is not race 0.
    pub fn race_ms(&self, i: usize) -> i64 {
        self.archives.first().map(|a| a.start_offset_ms as i64 + 10 * i as i64).unwrap_or(0)
    }
}

impl Tape {
    /// The WHOLE FILE, as bytes, with this tape spliced in.
    ///
    /// The body is written UNCOMPRESSED. That is not a shortcut: the dedicated
    /// server accepts a `U` body, and it is what makes a candidate a byte
    /// buffer a caller can patch in place instead of a file that has to be
    /// re-encoded. The search writes tens of thousands of candidates a second
    /// by building one explicit-encoded image here and then patching the
    /// steer / accel / brake bits of that image; an LZO round trip in this
    /// function would make that impossible and force a second writer to exist.
    ///
    /// `Encoding::Explicit` is what a patchable base image needs: every vehicle
    /// field is written out, so every tick has bits at a fixed position. It is
    /// deterministic -- the same tape always produces the same bytes.
    pub fn inject_into(&self, c: &crate::container::Container, enc: Encoding) -> Result<Vec<u8>, String> {
        self.inject_into_nth(c, enc, 0)
    }

    pub fn inject_into_nth(&self, c: &crate::container::Container, enc: Encoding, nth: usize) -> Result<Vec<u8>, String> {
        let body = self.splice_into_nth(c.body(), enc, nth)?;
        let mut file = c.gbx.header_bytes_u();
        file.extend_from_slice(&body);
        Ok(file)
    }
}
