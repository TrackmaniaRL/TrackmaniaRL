//! `fk probe` — find a named channel in the car's memory by matching it
//! against a recording that already knows the answer.
//!
//! The per-tick readout this driver ships gathers 44 bytes: the clock, the
//! quaternion, the position and the velocity. That is a decision somebody
//! made, not a limit — the engine computes far more than that and the game's
//! own recording carries 116 bytes per sample. Anything in the telemetry
//! record exists somewhere in the process; the question is only where.
//!
//! So this asks the question the cheap way round. Gather a WIDE window around
//! the located car, take the recording's own series for the channel you want,
//! and report every offset in the window whose bytes reproduce it. It is a
//! search with a ground-truth answer key, so it cannot talk itself into a
//! wrong answer the way a self-consistency argument can — and the tell that it
//! has found something real rather than a coincidence is UNIQUENESS plus a
//! decoding that is exact rather than merely correlated.
//!
//! # The encoding checklist
//!
//! Every assumption below was made silently by an earlier version of this
//! command and cost a channel. The next person to add a channel type will
//! find a fourth; add it here.
//!
//! 1. **The scale.** `u8`, `u8/255`, `f32` and an affine `a*u8+b` are all
//!    scored. Scoring a gear as a 0..1 quantity gave 0.00 % exact beside a
//!    0.9953 correlation -- the shape of a right answer told the wrong
//!    question.
//! 2. **The rounding.** Round-to-nearest and truncation differ by 17
//!    percentage points on wetness (83 % against 96 %). Both are scored.
//! 3. **The range.** The f32 branch once filtered candidates to 0..1, because
//!    the first channel it looked for was a fraction -- so it could not see a
//!    wheel rotation that runs to 1607. It filters to the REFERENCE's range.
//! 4. **The exactness test.** A channel the record does not quantise needs
//!    agreement to the reference's printed precision, not to half a u8 step.
//!
//! Two traps this is shaped around, both of which cost this project time
//! before:
//!
//! * **A channel that barely varies matches everywhere.** A byte that is 0 for
//!   90 % of a run is reproduced by any other mostly-zero byte, so the report
//!   states the reference's own variation and refuses to rank anything when
//!   there is not enough of it.
//! * **A resumed fork is not the same run.** `fk trace`'s own header says so.
//!   The comparison is therefore on the ticks the fork actually simulated, in
//!   race time, and never row against row.

use crate::locate::{gather_ticks, locate_v2};
use crate::session::{Checkpoint, Engine, Session};
use crate::tape::Tape;
use crate::traj;

pub struct ProbeOpts {
    /// The recording whose decoded telemetry holds the answer.
    pub reference: String,
    /// Which column of it to look for, e.g. `wetness` or `gear`.
    pub channel: String,
    /// How far either side of the located position to gather, in bytes.
    pub span: u32,
    /// Print this many best offsets.
    pub top: usize,
    /// Also score `a*u8 + b`, for a channel the record stores packed.
    pub affine: Option<(f64, f64)>,
}

/// One candidate offset's agreement with the reference series.
struct Hit {
    off: i64,
    /// Fraction of compared ticks where the byte, scaled, equals the reference
    /// to within half a quantisation step.
    exact: f64,
    /// Pearson correlation, for the case where the scale is not /255.
    corr: f64,
    distinct: usize,
    /// How the bytes were read: "u8", "u8/255" or "f32".
    enc: &'static str,
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for i in 0..a.len() {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    num / (da * db).sqrt()
}

pub fn run(engine: &Engine, tape: Tape, at: Checkpoint, o: ProbeOpts) -> Result<(), String> {
    let affine = o.affine;
    let reference = traj::Reference::load(&o.reference)?;
    let want = crate::traj::Reference::channel_from(&o.reference, &o.channel)

        .ok_or_else(|| format!("the reference has no column {}", o.channel))?;
    let bounds = reference.bounds(400.0);

    let mut s = Session::start(engine, tape, at)?;
    let probe = s.probe_tick()?;
    let recs = s.tape.tail_records(probe);
    let layout = locate_v2(
        &mut s.srv,
        probe,
        &recs,
        s.tape.start_offset_ms,
        bounds,
        2000,
        4000,
        true,
    )?;

    // A window centred on the car, plus the clock so every gathered tick can be
    // placed in race time.
    let span = o.span;
    let base = layout.pos.saturating_sub(span as u64);
    let width = span * 2;
    let segs = vec![(layout.clock, 4u32), (base, width)];
    let ticks = (s.tape.n() - probe + 200) as u32;
    let rows = gather_ticks(&mut s.srv, probe, &recs, &segs, ticks, 200_000, (0, 4));
    println!(
        "probe: {} ticks, {} bytes per tick around {:#x} (car at +{})",
        rows.len(),
        width,
        base,
        span
    );

    // Reference values at the race times the fork actually produced.
    let mut pairs: Vec<(usize, f64)> = Vec::new(); // (row index, wanted value)
    for (i, t) in rows.iter().enumerate() {
        let race = t.clock as i64 - layout.clock_bias;
        if let Some(v) = want.at(race) {
            pairs.push((i, v));
        }
    }
    if pairs.len() < 50 {
        return Err(format!(
            "only {} of {} gathered ticks have a reference value -- nothing to match against",
            pairs.len(),
            rows.len()
        ));
    }
    // STEADY TICKS ONLY. The record is on a 50 ms grid and the fork reports
    // every 10 ms, so on a tick where the channel is mid-transition the two are
    // comparing different instants and a disagreement says nothing about the
    // offset. Dropping those is not tuning: it is refusing to score a
    // comparison that was never valid. The count is printed so the reader can
    // see how much was dropped.
    let steady: Vec<usize> = (0..pairs.len())
        .filter(|&k| {
            let v = pairs[k].1;
            let lo = k.saturating_sub(1);
            let hi = (k + 2).min(pairs.len());
            (lo..hi).all(|j| (pairs[j].1 - v).abs() < 1e-9)
        })
        .collect();
    let pairs: Vec<(usize, f64)> = steady.iter().map(|&k| pairs[k]).collect();
    println!("steady ticks (channel not mid-transition): {}", pairs.len());
    let refvals: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for v in &refvals {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    let mut sorted = refvals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    println!(
        "reference {}: {} values over {} ticks, range {:.3}..{:.3}, {} distinct",
        o.channel,
        pairs.len(),
        rows.len(),
        lo,
        hi,
        sorted.len()
    );
    if sorted.len() < 4 || (hi - lo) < 1e-6 {
        return Err(format!(
            "the reference's own {} barely varies here ({} distinct values) -- ANY \
             near-constant byte would reproduce it, so a match would mean nothing. \
             Probe a window of the run where it moves.",
            o.channel,
            sorted.len()
        ));
    }

    // Every byte offset in the window, as a u8 scaled to /255 -- the encoding
    // the telemetry record uses for this family of channels -- and as a raw
    // value for the correlation.
    let mut hits: Vec<Hit> = Vec::new();
    let reclen = rows[0].rec.len();
    // THREE ENCODINGS, not one. The telemetry record quantises this family of
    // channels to a u8, but the record is a WIRE FORMAT -- the engine's own
    // variable is whatever it is, and on a 0..1 quantity that is usually an
    // f32. Probing only the wire encoding would find nothing and prove nothing.
    let getf32 = |b: &[u8], o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64;
    for off in 4..reclen {
        let raw: Vec<f64> = pairs.iter().map(|(i, _)| rows[*i].rec[off] as f64).collect();
        let mut distinct: Vec<u8> = pairs.iter().map(|(i, _)| rows[*i].rec[off]).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let scaled: Vec<f64> = raw.iter().map(|v| v / 255.0).collect();
        let exact = scaled
            .iter()
            .zip(&refvals)
            .filter(|(a, b)| (*a - *b).abs() <= 1.0 / 510.0)
            .count() as f64
            / refvals.len() as f64;
        hits.push(Hit {
            off: off as i64 - span as i64 - 4,
            exact,
            corr: pearson(&raw, &refvals),
            distinct: distinct.len(),
            enc: "u8/255",
        });
    }
    // An AFFINE u8: `a*v + b`, the encoding a small integer takes when the
    // record stores it packed -- gear is `4*u8 + 1`. Passed in rather than
    // guessed, because a fitted coefficient absorbs a wrong offset.
    if let Some((a, b)) = affine {
        for off in 4..reclen {
            let raw: Vec<f64> =
                pairs.iter().map(|(i, _)| a * rows[*i].rec[off] as f64 + b).collect();
            let mut distinct: Vec<u8> = pairs.iter().map(|(i, _)| rows[*i].rec[off]).collect();
            distinct.sort_unstable();
            distinct.dedup();
            let exact = raw.iter().zip(&refvals).filter(|(x, y)| (*x - *y).abs() < 1e-9).count()
                as f64
                / refvals.len() as f64;
            hits.push(Hit {
                off: off as i64 - span as i64 - 4,
                exact,
                corr: pearson(&raw, &refvals),
                distinct: distinct.len(),
                enc: "affine",
            });
        }
    }

    // A RAW u8, unscaled. Not every channel is a 0..1 quantity: a gear or a
    // count is a small integer, and scaling it by 1/255 makes an exact match
    // impossible -- the first version of this scored gear at 0.00 % while its
    // correlation sat at 0.9953, which is the shape of a right answer being
    // told the wrong question.
    for off in 4..reclen {
        let raw: Vec<f64> = pairs.iter().map(|(i, _)| rows[*i].rec[off] as f64).collect();
        let mut distinct: Vec<u8> = pairs.iter().map(|(i, _)| rows[*i].rec[off]).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let exact = raw.iter().zip(&refvals).filter(|(a, b)| (*a - *b).abs() < 1e-9).count()
            as f64
            / refvals.len() as f64;
        hits.push(Hit {
            off: off as i64 - span as i64 - 4,
            exact,
            corr: pearson(&raw, &refvals),
            distinct: distinct.len(),
            enc: "u8",
        });
    }
    for off in (4..reclen.saturating_sub(4)).step_by(1) {
        let raw: Vec<f64> = pairs.iter().map(|(i, _)| getf32(&rows[*i].rec, off)).collect();
        // Plausible means "in the reference's own range", not "in 0..1". The
        // first version hardcoded 0..1 because the first channel it looked for
        // was a fraction, and it then silently could not see a wheel rotation
        // that runs to 1607. An assumption about a channel's range is the same
        // class of mistake as an assumption about its encoding.
        let pad = (hi - lo).abs().max(1e-6) * 0.5;
        if raw.iter().any(|v| !v.is_finite() || *v < lo - pad || *v > hi + pad) {
            continue;
        }
        let mut distinct: Vec<u64> = raw.iter().map(|v| (v * 1e6) as u64).collect();
        distinct.sort_unstable();
        distinct.dedup();
        // The record quantises to a u8, so an f32 source agrees with it only to
        // within that step. Half a step is the right tolerance and anything
        // looser would accept a merely-correlated neighbour.
        // The record quantises to a u8, and WHICH ROUNDING it uses is not
        // something to assume: round-to-nearest and truncation give answers
        // 17 percentage points apart on this channel, so both are scored and
        // the better one is reported. Assuming one of them would have turned a
        // located channel into a "not found".
        let q = |v: f64, trunc: bool| {
            if trunc {
                (v * 255.0).floor() / 255.0
            } else {
                (v * 255.0).round() / 255.0
            }
        };
        // For a channel the record does NOT quantise to a u8 -- a rotation, a
        // length -- the right test is agreement to the printed precision of
        // the reference, not to half a u8 step.
        let direct = raw
            .iter()
            .zip(&refvals)
            .filter(|(a, b)| (*a - *b).abs() <= 1e-5 * b.abs().max(1.0))
            .count() as f64
            / refvals.len() as f64;
        let hit = |trunc: bool| {
            raw.iter()
                .zip(&refvals)
                .filter(|(a, b)| (q(**a, trunc) - **b).abs() <= 1e-6)
                .count() as f64
                / refvals.len() as f64
        };
        let exact = hit(false).max(hit(true)).max(direct);
        hits.push(Hit {
            off: off as i64 - span as i64 - 4,
            exact,
            corr: pearson(&raw, &refvals),
            distinct: distinct.len(),
            enc: "f32",
        });
    }
    hits.sort_by(|a, b| {
        b.exact
            .partial_cmp(&a.exact)
            .unwrap()
            .then(b.corr.abs().partial_cmp(&a.corr.abs()).unwrap())
    });

    println!("\ncar_offset\tas\texact\tcorr\tdistinct");
    for h in hits.iter().take(o.top) {
        println!(
            "{:+}\t{}\t{:.4}\t{:+.4}\t{}",
            h.off,
            h.enc,
            h.exact,
            h.corr,
            h.distinct
        );
    }
    let best = &hits[0];
    let runner = hits.get(1).map(|h| h.exact).unwrap_or(0.0);
    println!();
    // The bar is a GAP, not a level. A resumed fork is not bit-identical to the
    // recording it is compared against -- this driver says so in `fk trace` --
    // so demanding 100 % would reject a correctly located channel. What cannot
    // happen by chance is one offset reproducing the series far better than
    // every other offset in a two-kilobyte window.
    if best.exact > 0.90 && best.exact - runner > 0.25 {
        println!(
            "FOUND: {} is at car{:+} as {} -- {:.2}% exact, next best {:.2}%",
            o.channel,
            best.off,
            best.enc,
            100.0 * best.exact,
            100.0 * runner
        );
    } else if best.exact > 0.90 {
        println!(
            "AMBIGUOUS: {} of {} offsets reproduce {} at {:.2}% or better. A unique \
             answer is the evidence; several is a channel that does not distinguish them.",
            hits.iter().filter(|h| h.exact > 0.90).count(),
            hits.len(),
            o.channel,
            100.0 * best.exact
        );
    } else {
        println!(
            "NOT FOUND in this window: the best offset reproduces {} on {:.2}% of ticks. \
             Widen --span, or the channel is not a u8/255 at a fixed offset from the car.",
            o.channel,
            100.0 * best.exact
        );
    }
    Ok(())
}
