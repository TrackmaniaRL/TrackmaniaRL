//! The PLAIN ORACLE: the TM2020 dedicated server, validating files exactly the
//! way it would validate a submitted replay.
//!
//! This is the only oracle driver in the toolchain. Everything that needs a
//! time from the server comes through here, because the two things below are
//! easy to get wrong once and then wrong everywhere:
//!
//! **THE SERVER PRINTS TWO RESULTS PER FILE AND THE SECOND IS THE FILE'S OWN
//! CLAIM.** `ValidatedResult` is what it simulated; `DeclaredResult` is what the
//! file says. Both are objects with a `"Time"`. A parser that takes `"Time"`
//! lines as they come reports the file's declaration as the world's answer --
//! measured on a tape that simulates 22.738 and declares 22.730, where the
//! naive parse returned 22.730 and made a stale declaration look correct. This
//! parser tracks WHICH block it is inside and keeps both, so the disagreement
//! is a value you can read rather than a bug you can have.
//!
//! **THE SERVER ONLY LOOKS AT FILES WITH THE RIGHT EXTENSION.** A candidate
//! called `out.try3` is not read at all and the result is indistinguishable
//! from a run that did not finish. Every file is linked in under a name the
//! server will read.
//!
//! It validates in BATCHES, and the per-launch cost dominates a single file, so
//! `validate_many` is the real entry point and `validate` is the one-file case
//! of it.

use std::path::{Path, PathBuf};
use std::process::Command;

static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The server's "never crossed the line" sentinel, as it appears in a time
/// field: a huge u32 read as an i64.
///
/// Inherited from `tmmaps`'s own driver, which cleaned every time it parsed
/// through a `clean()` of exactly this shape, and kept when the two parsers
/// were merged -- a sentinel read as a value would be a finish at 4 294 967.295
/// seconds, which is a DNF reported as the best run in history.
pub const BAD_TIME_MS: i64 = 4_294_967_000;

/// A time the server can actually have simulated, or `None`.
fn sane_time(v: Option<i64>) -> Option<i64> {
    match v {
        Some(t) if t > BAD_TIME_MS || t < 0 => None,
        other => other,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MapsMode<'a> {
    /// Exactly one map linked in.
    One(&'a Path),
    /// A Maps directory containing zero files. If a file still validates, the
    /// map it ran on came out of the file itself -- the one-command proof
    /// behind "the map is inside the replay".
    Empty,
}

/// One file's answer from the server, with everything the server said about it.
#[derive(Debug, Clone, Default)]
pub struct SimResult {
    pub file: String,
    /// What the server SIMULATED. `None` is a DNF.
    pub time_ms: Option<i64>,
    pub cps: Option<u32>,
    pub respawns: Option<u32>,
    /// What the FILE CLAIMS. Present even when the simulation was a DNF, which
    /// is exactly the case a careless parser reports as a finish.
    pub declared_ms: Option<i64>,
    pub declared_cps: Option<u32>,
    pub is_valid: Option<bool>,
    /// The server's own explanation, e.g. `race finished, time is worse.`
    pub desc: String,
    /// The engine's echo of the input tape it decoded. An identity control that
    /// costs nothing: two files with the same tape produce the same string.
    pub inputs: String,
    pub account_id: String,
    pub login: String,
    pub map_uid: String,
    pub game_build: String,
}

impl SimResult {
    pub fn secs(&self) -> String {
        match self.time_ms {
            None => "DNF".into(),
            Some(t) => gbx::container::secs(t),
        }
    }
    /// Does the file's own declaration match what it actually does?
    ///
    /// The search layer needs exactly this and nothing else: a file that
    /// declares one time and does another is the container bug this project
    /// keeps paying for, and it is one comparison of two numbers that both came
    /// out of the world.
    pub fn declaration_holds(&self) -> bool {
        match (self.time_ms, self.declared_ms) {
            (Some(a), Some(b)) => a == b,
            (None, None) => true,
            _ => false,
        }
    }
}

pub fn server_dir(explicit: Option<&str>) -> PathBuf {
    if let Some(d) = explicit {
        return PathBuf::from(d);
    }
    if let Ok(d) = std::env::var("TM_SERVER") {
        return PathBuf::from(d);
    }
    PathBuf::from("/tmp/tmoracle/server")
}

fn link(target: &Path, at: &Path) {
    if std::fs::symlink_metadata(at).is_err() {
        let _ = std::os::unix::fs::symlink(target, at);
    }
}

/// The name a file must be linked under for the server to read it at all.
fn server_name(orig: &str) -> String {
    if orig.ends_with(".Ghost.Gbx") || orig.ends_with(".Replay.Gbx") {
        orig.to_string()
    } else {
        format!("{}.Ghost.Gbx", orig.replace(['/', ' '], "_"))
    }
}

/// Validate a batch. One server launch for the whole set, which is where the
/// throughput is: the per-launch cost dominates the per-file cost.
///
/// Results come back keyed by the name the server used, in the order it
/// reported them. A file the server never mentioned is absent from the result,
/// which is a fact worth having rather than a silent DNF.
pub fn validate_many(
    server: &Path,
    ghosts: &[&Path],
    maps: MapsMode,
    tag: &str,
) -> Result<Vec<SimResult>, String> {
    let bin = server.join("TrackmaniaServer");
    if !bin.exists() {
        return Err(format!(
            "no dedicated server at {} -- set TM_SERVER or pass --server DIR",
            server.display()
        ));
    }
    if ghosts.is_empty() {
        return Ok(Vec::new());
    }
    let root = std::env::temp_dir().join(format!(
        "ghost-oracle-{}-{}-{}",
        std::process::id(),
        tag,
        // UNIQUE PER CALL. The first thing done to this directory is
        // `remove_dir_all`, so two batches running concurrently in one process
        // -- which a test binary and `tmmaps`'s parallel driver both do -- would
        // wipe each other's staged files and validate the wrong set. A pid and
        // a tag are not enough; a counter is.
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&root);
    let replays = root.join("UserData").join("Replays");
    let mapsdir = root.join("UserData").join("Maps");
    std::fs::create_dir_all(&replays).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&mapsdir).map_err(|e| e.to_string())?;
    link(&server.join("Packs"), &root.join("Packs"));
    link(&bin, &root.join("TrackmaniaServer"));
    if let MapsMode::One(m) = maps {
        let abs = std::fs::canonicalize(m).map_err(|e| format!("{}: {}", m.display(), e))?;
        link(&abs, &mapsdir.join(abs.file_name().unwrap()));
    }
    // Two files can share a base name; the link name has to stay unique or the
    // batch silently validates one of them twice.
    let mut names: Vec<String> = Vec::with_capacity(ghosts.len());
    for (i, g) in ghosts.iter().enumerate() {
        let abs = std::fs::canonicalize(g).map_err(|e| format!("{}: {}", g.display(), e))?;
        let base = server_name(abs.file_name().unwrap().to_string_lossy().as_ref());
        let name = if names.contains(&base) {
            format!("{}_{}", i, base)
        } else {
            base
        };
        link(&abs, &replays.join(&name));
        names.push(name);
    }
    let out = Command::new("./TrackmaniaServer")
        .args(["/nodaemon", "/validatepath=."])
        .current_dir(&root)
        .output()
        .map_err(|e| format!("launching the server: {}", e))?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let _ = std::fs::remove_dir_all(&root);
    Ok(parse_many(&text))
}

/// The one-file case.
pub fn validate(
    server: &Path,
    ghost: &Path,
    maps: MapsMode,
    tag: &str,
) -> Result<SimResult, String> {
    let v = validate_many(server, &[ghost], maps, tag)?;
    v.into_iter().next().ok_or_else(|| {
        format!(
            "the server reported nothing for {} -- it did not read the file",
            ghost.display()
        )
    })
}

/// Which result block the parser is inside.
#[derive(PartialEq, Clone, Copy)]
enum Block {
    None,
    Validated,
    Declared,
}

/// Parse a whole batch. One record per `"FileName"`.
pub fn parse_many(text: &str) -> Vec<SimResult> {
    let mut out = Vec::new();
    let mut cur = SimResult::default();
    let mut block = Block::None;
    let field = |t: &str| -> String {
        t.splitn(2, ':')
            .nth(1)
            .map(|s| s.trim().trim_end_matches(',').trim_matches('"').to_string())
            .unwrap_or_default()
    };
    let numf = |t: &str| -> Option<i64> {
        t.splitn(2, ':').nth(1).and_then(|s| s.trim().trim_end_matches(',').parse::<i64>().ok())
    };
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("\"ValidatedResult\"") {
            block = if t.contains("null") { Block::None } else { Block::Validated };
        } else if t.starts_with("\"DeclaredResult\"") {
            block = if t.contains("null") { Block::None } else { Block::Declared };
        } else if t.starts_with("\"Time\"") {
            match block {
                Block::Validated => cur.time_ms = sane_time(numf(t)),
                Block::Declared => cur.declared_ms = sane_time(numf(t)),
                Block::None => {}
            }
        } else if t.starts_with("\"NbCheckpoints\"") {
            match block {
                Block::Validated => cur.cps = numf(t).map(|v| v as u32),
                Block::Declared => cur.declared_cps = numf(t).map(|v| v as u32),
                Block::None => {}
            }
        } else if t.starts_with("\"NbRespawns\"") {
            if block == Block::Validated {
                cur.respawns = numf(t).map(|v| v as u32);
            }
        } else if t.starts_with("\"IsValid\"") {
            cur.is_valid = Some(t.contains("true"));
            block = Block::None;
        } else if t.starts_with("\"Desc\"") {
            cur.desc = field(t).replace("\\n", " ").trim().to_string();
            block = Block::None;
        } else if t.starts_with("\"Inputs\"") {
            cur.inputs = field(t);
        } else if t.starts_with("\"AccountId\"") {
            cur.account_id = field(t);
        } else if t.starts_with("\"Login\"") {
            cur.login = field(t);
        } else if t.starts_with("\"GameBuild\"") {
            cur.game_build = field(t);
        } else if t.starts_with("\"MapUid\"") {
            cur.map_uid = field(t);
        } else if t.starts_with("\"FileName\"") {
            cur.file = field(t);
            // On a DNF the checkpoint count is only in the prose.
            if cur.cps.is_none() {
                cur.cps = cps_from_desc(&cur.desc);
            }
            out.push(std::mem::take(&mut cur));
            block = Block::None;
        }
    }
    out
}

/// The checkpoint count the server mentions in its own explanation.
///
/// On a DNF there is no `ValidatedResult`, so `NbCheckpoints` never appears --
/// but the `Desc` says `reached some checkpoints (2 out of 2)`, and a search
/// ladder that ranks DNFs by depth needs exactly that number. Reading it out of
/// the sentence is not elegant; losing it is worse.
fn cps_from_desc(desc: &str) -> Option<u32> {
    let i = desc.find("checkpoint")?;
    let rest = &desc[i..];
    let open = rest.find('(')?;
    let close = rest[open..].find(')')? + open;
    rest[open + 1..close]
        .split_whitespace()
        .next()
        .and_then(|v| v.parse::<u32>().ok())
}
