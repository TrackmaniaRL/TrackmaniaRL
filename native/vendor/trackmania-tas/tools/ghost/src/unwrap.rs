//! `ghost unwrap IN OUT` -- the `CGameCtnGhost` a `.Replay.Gbx` carries, written
//! out as a standalone `.Ghost.Gbx`.
//!
//! WHY (2026-09-09, tiny campaign): the client keeps the run a player finished in
//! the editor's test mode as `ProgramData/Trackmania/MediaTrackerCache/
//! MTAuthorGhost<map name>.Ghost.gbx` -- the MediaTracker's "Ref. Ghost: Author
//! ghost", looked up BY MAP NAME. Despite the extension the file is a
//! `CGameCtnReplayRecord` (class 0x03093000): the whole map embedded in chunk
//! 0x03093002 (9-18 MB for a tiny map), then chunk 0x03093014 whose one node
//! reference is the ghost. vjeux's playtest runs live only there, and the route
//! search wants them as plain ghosts.
//!
//! WHAT THE GAME WROTE, measured on the 03 cache file against a game-written
//! standalone ghost (`testdata/human_22730.Ghost.Gbx`): the ghost node's chunk
//! stream inside the replay is BYTE-IDENTICAL in shape to a standalone ghost
//! body -- it opens with the constant 0x0303F006, its first lookback string
//! carries the version marker (nothing before it in the replay body reads a
//! string: the map is a raw blob, 0x03093014 is two words and a node index),
//! and it closes with 0x0309202E + 0xFACADE01. The ONE difference is the node
//! index of the nested `CPlugEntRecordData` (the telemetry): node 2 inside the
//! replay, node 1 in a file whose main node is the ghost. So the conversion is
//! a slice and a four-byte rewrite, and the control below proves exactly that.

use crate::cli::die;
use gbx::container::{write_gbx, Container, Gbx, SKIP_MAGIC};

const CLASS_REPLAY: u32 = 0x0309_3000;
const CLASS_GHOST: u32 = 0x0309_2000;
const CHUNK_GHOSTS: u32 = 0x0309_3014;
const FACADE: u32 = 0xFACA_DE01;
/// CPlugEntRecordData: class id, then its one chunk id -- the eight bytes that
/// follow the node index we rewrite.
const RECORD_NODE: [u8; 8] = [0x00, 0xF0, 0x11, 0x09, 0x00, 0xF0, 0x11, 0x09];

fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

/// Where the ghost node's chunk stream starts in a replay body (just past its
/// class id), and which node index the replay gave it.
fn ghost_start(body: &[u8]) -> Result<(usize, u32), String> {
    // The ghost list follows the embedded map; search from there so a byte
    // pattern inside the map cannot be mistaken for the chunk id.
    let from = gbx::container::embedded_map_in(body).map(|(o, n)| o + n).unwrap_or(0);
    let pat = CHUNK_GHOSTS.to_le_bytes();
    let mut i = from;
    while i + 20 <= body.len() {
        if body[i..i + 4] == pat {
            let version = u32_at(body, i + 4).unwrap();
            let count = u32_at(body, i + 8).unwrap();
            let index = u32_at(body, i + 12).unwrap();
            let class = u32_at(body, i + 16).unwrap();
            if count == 0 {
                return Err("chunk 0x03093014 lists no ghost".into());
            }
            if class != CLASS_GHOST {
                return Err(format!(
                    "chunk 0x03093014 (version {version}): first node reference {index} is not followed by a \
                     CGameCtnGhost class id (found 0x{class:08X}) -- a node defined earlier in the file, \
                     which this converter does not follow"
                ));
            }
            if count > 1 {
                eprintln!("note: the replay lists {count} ghosts; taking the first");
            }
            return Ok((i + 20, index));
        }
        i += 1;
    }
    Err("no ghost list chunk 0x03093014 after the embedded map".into())
}

/// The end of a ghost node's chunk stream: the offset just past its 0xFACADE01.
///
/// Walked, not searched: the nested record node ends with its own marker inside
/// the skippable 0x03092000, and a byte scan for the marker would stop there.
fn ghost_end(body: &[u8], start: usize) -> Result<usize, String> {
    let mut o = start;
    loop {
        let id = u32_at(body, o).ok_or("ran off the body inside the ghost")?;
        if id == FACADE {
            return Ok(o + 4);
        }
        if body.get(o + 4..o + 8) == Some(&SKIP_MAGIC[..]) {
            let size = u32_at(body, o + 8).ok_or("truncated skippable chunk")? as usize;
            o += 12 + size;
            continue;
        }
        // The five non-skippable chunks a CGameCtnGhost body carries on this build.
        match id {
            0x0303_F006 => {
                // u32 version, u32, u32 length, length bytes (28 bytes in every ghost seen)
                let len = u32_at(body, o + 12).ok_or("truncated 0x0303F006")? as usize;
                o += 16 + len;
            }
            0x0309_200C | 0x0309_200E => {
                // one u32 each (0x0309200E is the hash GHOSTFORMAT.md copies)
                o += 8;
            }
            0x0309_201C => {
                // the 32-byte hash GHOSTFORMAT.md copies
                o += 4 + 32;
            }
            0x0309_200F => {
                // GhostLogin: a plain length-prefixed string
                let len = u32_at(body, o + 4).ok_or("truncated 0x0309200F")? as usize;
                o += 8 + len;
            }
            0x0309_2010 => {
                // Validate_ChallengeUid: a lookback string. The version marker was
                // consumed by the first MwId inside 0x03092000, so this is the flag
                // word and, when the flag names a NEW string, its length and bytes.
                let flag = u32_at(body, o + 4).ok_or("truncated 0x03092010")?;
                o += 8;
                if flag & 0x3FFF_FFFF == 0 && flag != 0 {
                    let len = u32_at(body, o).ok_or("truncated 0x03092010 string")? as usize;
                    o += 4 + len;
                }
            }
            other => {
                return Err(format!(
                    "unknown non-skippable chunk 0x{other:08X} at body offset {o} -- this converter knows \
                     0x0303F006, 0x0309200C, 0x0309200E, 0x0309200F, 0x03092010 and 0x0309201C; a new build may have added one"
                ));
            }
        }
    }
}

/// The ghost node's bytes, rebased so that its record data is node 1.
pub fn unwrap_body(body: &[u8]) -> Result<(Vec<u8>, usize, usize, u32, usize), String> {
    let (start, index) = ghost_start(body)?;
    let end = ghost_end(body, start)?;
    let mut ghost = body[start..end].to_vec();
    let hits: Vec<usize> = (0..ghost.len().saturating_sub(8))
        .filter(|&i| ghost[i..i + 8] == RECORD_NODE)
        .collect();
    if hits.len() != 1 {
        return Err(format!(
            "expected exactly one CPlugEntRecordData node in the ghost, found {}",
            hits.len()
        ));
    }
    let at = hits[0] - 4;
    let had = u32_at(&ghost, at).unwrap();
    if had == 0 || had > 64 {
        return Err(format!("the record node index before the record class reads {had}: not a node index"));
    }
    ghost[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
    Ok((ghost, start, end, index, at))
}

pub fn cmd(a: &[String]) {
    if a.len() < 2 {
        die("ghost unwrap IN.Replay.Gbx OUT.Ghost.Gbx");
    }
    let (inp, out) = (&a[0], &a[1]);
    let c = Container::load(inp).unwrap_or_else(|e| die(e));
    if c.gbx.class_id != CLASS_REPLAY {
        die(format!(
            "{inp}: class 0x{:08X} is not a CGameCtnReplayRecord (0x03093000); a plain ghost needs no unwrapping",
            c.gbx.class_id
        ));
    }
    let (ghost, start, end, index, at) = unwrap_body(c.body()).unwrap_or_else(|e| die(format!("{inp}: {e}")));
    let g = Gbx {
        version: 6,
        format: b'B',
        ref_comp: b'U',
        unknown: Some(b'R'),
        class_id: CLASS_GHOST,
        user_data: Vec::new(),
        num_nodes: 2,
        ref_table: 0u32.to_le_bytes().to_vec(),
        body: Vec::new(),
    };
    write_gbx(&g, ghost.clone(), out).unwrap_or_else(|e| die(e));

    // CONTROL: the written file is the sliced node with the one rewrite and
    // nothing else, and it parses as the same run.
    let back = Container::load(out).unwrap_or_else(|e| die(format!("read-back: {e}")));
    let diffs: Vec<usize> = back.body().iter().zip(c.body()[start..end].iter()).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect();
    if back.body().len() != end - start || diffs.iter().any(|&i| i < at || i >= at + 4) {
        die(format!(
            "control failed: the read-back body differs from the replay's ghost node at byte(s) {diffs:?} (only the node index at {at}..{} may differ)",
            at + 4
        ));
    }
    let (r_in, r_out) = (c.result(), back.result());
    if r_in.is_none() || r_in != r_out {
        die("control failed: the result chunk (race time, splits) does not read back the same");
    }
    let (t_in, t_out) = (c.declared_times().into_iter().map(|(_, ms)| ms).collect::<Vec<_>>(), back.declared_times().into_iter().map(|(_, ms)| ms).collect::<Vec<_>>());
    if t_in.is_empty() || t_in != t_out {
        die(format!("control failed: declared times {t_in:?} vs {t_out:?}"));
    }
    let tape_in = gbx::tape::Tape::from_body(c.body()).unwrap_or_else(|e| die(format!("input tape: {e}")));
    let tape_out = gbx::tape::Tape::from_body(back.body()).unwrap_or_else(|e| die(format!("output tape: {e}")));
    if format!("{:?}", tape_in) != format!("{:?}", tape_out) {
        die("control failed: the input tape does not read back identically");
    }
    let r = r_out.unwrap();
    println!(
        "wrote {out}: {} B body (replay node {index} at body {start}..{end}, record node 2 -> 1), race {}, {} checkpoints, {} ticks",
        ghost.len(),
        gbx::container::secs(r.race_ms as i64),
        r.entries.len(),
        tape_out.archives.iter().map(|a| a.packets.len()).sum::<usize>()
    );
}
