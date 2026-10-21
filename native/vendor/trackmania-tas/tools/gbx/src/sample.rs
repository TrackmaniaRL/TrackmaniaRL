//! The 116-byte `CSceneVehicleVis` record sample, as a vocabulary.
//!
//! # Why this is in `gbx` and not next to the code that writes it
//!
//! Two crates need to agree about what a sample byte IS. `fk` writes them
//! (`fk::vislayout` transcribes the game's own writer, and `fk regen` puts the
//! result in a file); `ghost` judges them (`ghost::finish` refuses a file whose
//! claimed channels are dead). `ghost` cannot depend on `fk` — `fk` depends on
//! `ghost` for the input codec — so before this module the two crates each kept
//! their own hand-maintained lists of which bytes the pipeline writes.
//!
//! **They drifted, and the drift was invisible.** On 2026-08-23 `fk` wrote
//! bytes 5, 81–84, 89 and 91 from engine memory while `ghost::finish`'s
//! `unwritten_channels()` still announced all seven as "zeroed rather than
//! inherited" in the acceptance report of every run; byte 91 was
//! simultaneously in `may_rest()` (written, may legitimately be constant) and
//! `unwritten_channels()` (not written at all), which cannot both be true; and
//! bytes 19, 20, 34 and 108–111, which really are not written, were in neither
//! list. Nothing failed. The report was simply wrong for months, in a file
//! whose whole purpose is to say what the pipeline did.
//!
//! A fact stated twice is a fact that will disagree with itself. This module is
//! the one statement; both crates read it.

/// A TM2020 vehicle sample. 103 is the v30 floor the decoder documents; 116 is
/// what v33 (TM2020) carries.
pub const SAMPLE_SIZE: usize = 116;

/// The bytes the transform encoder owns: position, orientation, speed and the
/// velocity direction, packed together at 47..69.
pub const TRANSFORM: std::ops::Range<usize> = 47..69;

/// Bytes the state transcription CANNOT predict, because producing them needs
/// something the vehicle state does not hold.
///
/// 59..65 are the orientation words — the transcription reads `Loc`'s 3x3
/// rotation and the encoder wants the game's own quantisation of it, so these
/// come from the transform encoder instead (they ARE written; they are simply
/// not the transcription's). 108..112 is the countdown, which needs the race
/// clock rather than the car.
pub const UNPREDICTED: &[usize] = &[59, 60, 61, 62, 63, 64, 108, 109, 110, 111];

/// Bytes that read identically zero in the DEDICATED SERVER, whatever the car
/// is doing — the client computes them and the server does not.
///
/// Writing them would put a confident zero where a real value belongs, which is
/// strictly worse than leaving the carrier's: a zero looks measured. Byte 34
/// was claimed as the reactor on a four-file correlation and refuted by a
/// seven-gate recording that holds it constant; 93/95/97/99 are the four dirt
/// slots, and a clip once ran on dirt tyres because they were the donor's.
pub const DEAD_IN_SERVER: &[usize] = &[19, 20, 34, 93, 95, 97, 99];

/// Which bytes a `--carrier layout` regeneration writes, and which it leaves.
///
/// DERIVED, not listed: the transcription writes everything except
/// `UNPREDICTED` and `DEAD_IN_SERVER`, and the transform encoder then writes
/// `TRANSFORM` — which is what puts 59..65 back. Anything a caller wants to
/// know about coverage is a question about these three facts, so asking it as a
/// function is what stops a fourth list existing.
pub fn written_by_carrier() -> Vec<bool> {
    let mut w = vec![true; SAMPLE_SIZE];
    for &b in UNPREDICTED {
        w[b] = false;
    }
    for &b in DEAD_IN_SERVER {
        w[b] = false;
    }
    for b in TRANSFORM {
        w[b] = true;
    }
    w
}

/// The bytes a `--carrier layout` regeneration does NOT write, in order.
pub fn not_written_by_carrier() -> Vec<usize> {
    let w = written_by_carrier();
    (0..SAMPLE_SIZE).filter(|b| !w[*b]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE DERIVATION MUST REPRODUCE WHAT THE TOOL ACTUALLY DID.
    ///
    /// Not a restatement of the constants above in another form — that would
    /// pass however wrong they are. This is the list `fk regen --carrier
    /// layout` printed for untitled 01 on 2026-08-23, copied from the run:
    /// `NOT WRITTEN: [19, 20, 34, 93, 95, 97, 99, 108, 109, 110, 111]`.
    #[test]
    fn the_unwritten_set_is_the_one_a_real_regeneration_reported() {
        assert_eq!(
            not_written_by_carrier(),
            vec![19, 20, 34, 93, 95, 97, 99, 108, 109, 110, 111]
        );
    }

    /// A byte cannot be both written and not written. `ghost::finish` held 91
    /// in two lists that mean exactly that, and nothing noticed.
    #[test]
    fn no_byte_is_both_written_and_unwritten() {
        let w = written_by_carrier();
        for b in not_written_by_carrier() {
            assert!(!w[b], "byte {b} is in both");
        }
    }

    /// The orientation words are UNPREDICTED by the transcription and still
    /// written, by the transform encoder. If the derivation ever stops putting
    /// them back, every regenerated ghost faces the wrong way.
    #[test]
    fn the_orientation_words_are_written_even_though_the_transcription_cannot_predict_them() {
        let w = written_by_carrier();
        for b in 59..65 {
            assert!(w[b], "orientation byte {b} would not be written");
        }
    }
}
