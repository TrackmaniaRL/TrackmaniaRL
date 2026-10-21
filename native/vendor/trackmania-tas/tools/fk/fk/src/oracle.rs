//! The PLAIN ORACLE, and the one thing `fk` is allowed to conclude from it.
//!
//! `ghost::oracle` runs the dedicated server; this is a thin layer over it that
//! exists to hold one rule:
//!
//! > **A fork-reported time is a measurement. Only the plain oracle, run on the
//! > file as written to disk, is a result.**
//!
//! The fork server was exact on 4700 of 4700 candidates that perturbed a human
//! reference by a few ticks late in the run. Outside that regime, 0 of 312
//! fork-reported finishes survived a full `/validatepath` of the byte-identical
//! bitstream. Both numbers are real; the difference between them is the regime,
//! and nothing inside the fork can see which regime it is in.
//!
//! There is deliberately no `fk validate` subcommand. Validating a file is
//! `ghost verify` / `ghost declare --from-oracle`, and having two commands that
//! both "ask the oracle" is how a project ends up with two answers.

use ghost::oracle::{validate_many, MapsMode, SimResult};
use std::path::Path;

/// Validate a batch of files the way the server does it — several ghosts in one
/// launch.
///
/// Batched on purpose. Nearly all of a validation's cost is the server's own
/// start-up, so a one-file-at-a-time baseline would inflate every speedup `fk`
/// reports by exactly the number it is trying to measure. `fk server bench`
/// compares against this.
pub fn validate_batch(
    server: &Path,
    map: &Path,
    files: &[&Path],
    tag: &str,
) -> Result<Vec<SimResult>, String> {
    // Not belt-and-braces. The server silently ignores a file whose name does
    // not end in `.Ghost.Gbx` / `.Replay.Gbx` and reports a plain DNF you
    // cannot tell from a real one: the ghost arm lost 32 consecutive GOOD
    // regenerations to exactly this before anyone noticed the gate was refusing
    // files that were fine.
    for f in files {
        crate::tape::check_oracle_readable_name(f)?;
    }
    validate_many(server, files, MapsMode::One(map), tag)
}

/// A reusable handle for repeated batch validations of the same (server, map).
///
/// `tag` becomes part of the scratch root, so two of these in one process — or
/// two jobs of one harness — never share a `UserData/Replays`. That is not
/// hygiene: worker directories named by index are how two concurrent searches
/// came to validate each other's candidates and credit the time to the local
/// one.
pub struct Batch {
    server: std::path::PathBuf,
    map: std::path::PathBuf,
    tag: String,
}

impl Batch {
    pub fn new(server: &Path, map: &Path, tag: &str) -> Batch {
        Batch {
            server: server.to_path_buf(),
            map: map.to_path_buf(),
            tag: format!("{}-{}", tag, std::process::id()),
        }
    }

    /// Validate these files. A server that could not be launched is an empty
    /// result and a message on stderr — never a fabricated set of DNFs, because
    /// a DNF is an answer and "I could not ask" is not.
    pub fn times(&self, files: &[std::path::PathBuf]) -> Vec<SimResult> {
        let refs: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
        match validate_batch(&self.server, &self.map, &refs, &self.tag) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("fk: oracle: {}", e);
                Vec::new()
            }
        }
    }
}
