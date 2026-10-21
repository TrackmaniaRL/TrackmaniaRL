//! The tape, as the ENGINE sees it: three arrays of per-tick inputs.
//!
//! # This is the only file in `fk` that knows a file format exists
//!
//! Everything else in this crate takes `&[u8]` steer / accel / brake and a
//! `start_offset_ms`. That is deliberate. `fk` used to carry its own copy of
//! the `0x0309201D` input codec (`tmsearch::ghost::Factory`), which made two
//! implementations of one bit-packed format in one project — and two
//! implementations of one file format is how this project got silent
//! corruption before. The codec lives in `ghost` now and this module is the
//! adapter.
//!
//! # What moving to `ghost`'s codec fixed
//!
//! The old `Factory` re-encoded the archive with explicit vehicle fields so
//! individual ticks could be patched in place — **except** mode-12 packets
//! carrying the "same as the previous tick" bit, which it emitted as a one-bit
//! `Slot::FROZEN` and could not patch at all. A write to such a tick was
//! silently dropped. Template `seed_rank10000` has three (ticks 0, 1, 2); they
//! sat below every resume boundary, so the defect never bit, and nothing would
//! have said so if it had.
//!
//! (The codec calls the type `SAME_INPUTS`, never "frozen": "frozen" reads as a
//! claim about the game, and a stretch of them then looks like a physics
//! constraint rather than a gap in the encoder.)
//!
//! `ghost`'s writer expands every such packet, so every tick of a tape written
//! through here is writable. `tests/suite.rs` pins it.

use gbx::tape::{Encoding, Tape as GTape};
use std::path::Path;

/// A ghost's input tape, decoded into the channels the engine consumes, plus
/// enough of the container to write a modified copy back out.
#[derive(Clone)]
pub struct Tape {
    /// Steer per tick, as the raw byte the format stores. The engine reads it
    /// as `(byte as i8) as f32 / 127.0`.
    pub steer: Vec<u8>,
    pub accel: Vec<u8>,
    pub brake: Vec<u8>,
    /// Race time of tick `t` is `t * 10 + start_offset_ms`. It is usually
    /// NEGATIVE: most of this project's incumbents are countdown-prefixed, so
    /// tick 0 is well before race 0 and "the first tick" is not "the start".
    /// Check it before interpreting any tick-0 edit.
    pub start_offset_ms: i32,
    /// What the file CLAIMS it achieved — not what it does. The two disagree on
    /// every synthesised tape until `ghost declare --from-oracle` is run on it,
    /// which is why nothing in `fk` ever scores anything against this.
    pub declared_ms: Option<u32>,
    inner: GTape,
    body: Vec<u8>,
    path: String,
}

impl Tape {
    /// Decode archive 0 of `path`.
    ///
    /// A file with more than one input archive is refused rather than guessed
    /// at: every tool in this project has only ever meant archive 0, and a
    /// silent choice between two of them produces a correct-looking measurement
    /// of the wrong run.
    pub fn load(path: &str) -> Result<Tape, String> {
        let inner = GTape::from_file(path)?;
        if inner.archives.len() != 1 {
            return Err(format!(
                "{} has {} input archives; fk only ever means archive 0 and will not guess \
                 which one you want",
                path,
                inner.archives.len()
            ));
        }
        let c = ghost::Container::load(path)?;
        let body = c.body().to_vec();
        Ok(Tape {
            steer: inner.steer_i8s().into_iter().map(|s| s as u8).collect(),
            accel: inner.accels(),
            brake: inner.brakes(),
            start_offset_ms: inner.archives[0].start_offset_ms,
            declared_ms: c.declared_times().first().map(|d| d.1),
            inner,
            body,
            path: path.to_string(),
        })
    }

    pub fn n(&self) -> usize {
        self.steer.len()
    }

    /// Race time of tick `t`, in milliseconds.
    pub fn race_ms(&self, t: usize) -> i64 {
        t as i64 * 10 + self.start_offset_ms as i64
    }

    /// The codec's own control: re-encoding this tape verbatim reproduces the
    /// file's original bitstream byte for byte. Run it before trusting any
    /// candidate written from this tape — if the decode lost something, every
    /// candidate carries the loss and every comparison between them still
    /// agrees.
    pub fn codec_is_lossless(&self) -> Result<(), String> {
        self.inner.verbatim_is_identity()
    }

    /// Write a copy of this file carrying `steer` / `accel` / `brake` instead of
    /// its own inputs.
    pub fn write_candidate(
        &self,
        steer: &[u8],
        accel: &[u8],
        brake: &[u8],
        out: &Path,
    ) -> Result<(), String> {
        check_oracle_readable_name(out)?;
        let n = self.n();
        if steer.len() != n || accel.len() != n || brake.len() != n {
            return Err(format!(
                "candidate has {}/{}/{} ticks, the tape has {}",
                steer.len(),
                accel.len(),
                brake.len(),
                n
            ));
        }
        let mut t = self.inner.clone();
        for (i, p) in t.archives[0].packets.iter_mut().enumerate() {
            p.steer = (steer[i] as i8) as u32 & 0xFF;
            p.accel = accel[i] as u32;
            p.brake = brake[i] as u32;
            p.vsame = false;
        }
        let body = t.splice_into(&self.body, Encoding::Explicit)?;
        let g = gbx::container::Gbx::parse(&std::fs::read(&self.path).map_err(|e| e.to_string())?);
        gbx::container::write_gbx(&g, body, &out.to_string_lossy())
    }

    /// Write this tape back out unmodified, under a new name.
    ///
    /// Used to stage the reference file a fork server runs. It is a re-encode
    /// rather than a copy on purpose: the reference the server simulates is then
    /// produced by exactly the path a candidate is, so a broken encoder cannot
    /// hide by breaking both sides of the comparison equally. The remaining way
    /// it could hide is caught by checking the reference against the file's own
    /// DECLARED time, which `fk server check` does on its first line.
    pub fn write_reference(&self, out: &Path) -> Result<(), String> {
        self.write_candidate(&self.steer, &self.accel, &self.brake, out)
    }

    /// The per-tick records the engine holds, for the tape from `from` onwards.
    /// This is what a resume writes into engine memory.
    pub fn tail_records(&self, from: usize) -> Vec<forkoracle::forksrv::Rec> {
        (from..self.n())
            .map(|t| forkoracle::forksrv::rec_of(self.steer[t], self.accel[t], self.brake[t]))
            .collect()
    }
}

/// The dedicated server silently ignores a file whose name does not end in
/// `.Ghost.Gbx` or `.Replay.Gbx` and reports a plain DNF — indistinguishable
/// from a run that genuinely did not finish. Every path in `fk` that hands a
/// file to the oracle goes through here, so the mistake is an error rather than
/// a wrong measurement.
pub fn check_oracle_readable_name(p: &Path) -> Result<(), String> {
    let name = p.to_string_lossy();
    if name.ends_with(".Ghost.Gbx") || name.ends_with(".Replay.Gbx") {
        return Ok(());
    }
    Err(format!(
        "{}: a file handed to the oracle must be named *.Ghost.Gbx or *.Replay.Gbx -- the \
         dedicated server ignores anything else and returns a bare DNF you cannot tell from a \
         genuine one",
        name
    ))
}
