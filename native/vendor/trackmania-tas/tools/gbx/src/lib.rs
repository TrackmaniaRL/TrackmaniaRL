//! `gbx` — the TM2020 GBX file format, in one place.
//!
//! WHY THIS CRATE EXISTS. Three tools in this project used to carry their own
//! copy of this format: `tmtraj` (the trajectory decoder), `tools/ghost` (the
//! ghost/replay API) and `tmsite` (the visualiser). Two implementations of one
//! file format is how a project gets silent corruption — a fix lands in one
//! reader and the other keeps decoding the old way, and nothing fails. There is
//! now exactly one implementation and it is here; everything else calls it.
//!
//! The split of responsibility across the workspace:
//!
//! * **`gbx`** — what the bytes ARE. Container, chunks, the input tape, the
//!   `CPlugEntRecordData` telemetry record and the meaning of the 116-byte
//!   vehicle sample. Reads and writes; no verdicts, no analysis, no I/O beyond
//!   the file in front of it.
//! * **`tmtraj`** — what the bytes MEAN for a run: trajectory analysis,
//!   comparison between runs, the publish gate's physics checks, racing-line
//!   clustering. Read-only; it never writes a ghost.
//! * **`ghost`** — every MUTATION of a ghost or replay, plus the oracle and the
//!   publish decision.
//! * **`tmsite`** — presentation: the 3D page and the TICK input-script export.

pub mod bits;
pub mod container;
pub mod header;
pub mod manifest;
pub mod name;
pub mod record;
pub mod recwrite;
pub mod sample;
pub mod sha;
pub mod tape;

pub use container::{all_skip_chunks, lzo_init, Container, Gbx, SKIP_MAGIC};
pub use record::{
    decode_ghost, decode_body, load_body, Decoded, Ent, EntInfo, RecordData, Sample,
    CLASS_CPLUGENTRECORDDATA, CLASS_CSCENEVEHICLEVIS,
};

/// The map uid a `.Map.Gbx` declares.
///
/// Two encodings, because there are two kinds of map file: one written by the
/// editor, which carries an XML header, and the copy carried inside a replay,
/// which has a binary header instead. Returning `None` for the second and
/// calling it "no uid" is a harness limit reported as a fact about the file.
pub fn map_uid_of(data: &[u8]) -> Option<String> {
    let head = &data[..data.len().min(60000)];
    let s = String::from_utf8_lossy(head);
    if let Some(i) = s.find("uid=\"") {
        let rest = &s[i + 5..];
        if let Some(j) = rest.find('"') {
            return Some(rest[..j].to_string());
        }
    }
    let mut i = 0usize;
    while i + 31 <= head.len() {
        if u32::from_le_bytes(head[i..i + 4].try_into().unwrap()) == 27 {
            if let Ok(t) = std::str::from_utf8(&head[i + 4..i + 31]) {
                if t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                    return Some(t.to_string());
                }
            }
        }
        i += 1;
    }
    None
}
