//! `fk carrier events` — find the memory slot that changes WHEN SOMETHING
//! HAPPENS, using the map as the ground truth instead of a reference recording.
//!
//! # Why this exists
//!
//! `fk carrier scan` fits engine memory against a recording's own sample bytes.
//! That is the stronger test when it can run, and it has a hard precondition:
//! **an answer key** — a recording the GAME wrote for this tape, whose sample
//! bytes hold the channel you are hunting. Two ways that precondition fails,
//! both hit on 276874 (`untitled 01`) while looking for the reactor:
//!
//! * the only recordings of that map are search tapes on a stranger's
//!   container, so the scan cannot even locate the car against them (their
//!   telemetry is 500 m from where the tape goes); and
//! * our own regenerated file has our run, but `--neutralise` ZEROES the very
//!   bytes we are hunting, so the fit has nothing to fit — `0 channels beat
//!   both their permutation floor and a constant`, from an empty column.
//!
//! And no third file exists: 276874 is the one map in this repo with reactor
//! gates *and* zero human records. The empirical route is closed for want of a
//! key, which is `CARRIER.md`'s third verdict — "could not be tested" — in its
//! most annoying form, because the channel plainly IS live (byte 34 takes 137
//! distinct values there and is constant across 201 files of four other maps).
//!
//! # What this does instead
//!
//! **The map is the answer key.** A boost gate is at a known position; the
//! engine run gives the car's position per instant; so the times at which
//! something must happen to the car are known WITHOUT any recording. A slot
//! that holds reactor state changes at those instants and rarely between them.
//!
//! So: for every offset in the gathered window, build the per-instant series,
//! mark where it changes, and score how well those change-points line up with
//! the events. Rank by alignment.
//!
//! # What it is NOT
//!
//! **A proposal, exactly like a scan's.** Alignment is a correlation and this
//! command cannot test itself: with 1.3 M offsets, some will align by accident,
//! and the printed `noise` column is the only thing separating a slot that
//! tracks the event from a slot that changes constantly. A survivor is a
//! candidate to CONFIRM — on another map, or by predicting the value and
//! scoring it — never a location. The floor is printed for the same reason a
//! scan prints its permutation floor: a score with nothing to beat is not a
//! result.

use crate::record;

/// One offset's alignment with the events.
struct Hit {
    off: usize,
    width: usize,
    hits: usize,
    noise: usize,
    distinct: usize,
}

pub fn run(a: &[String]) -> Result<(), String> {
    let dump = flag(a, "--dump").ok_or("--dump FILE is required (a gather's raw window)")?;
    let reclen = num(a, "--reclen", 0) as usize;
    if reclen == 0 {
        return Err("--reclen N is required: the gather printed it".into());
    }
    let bias = num(a, "--bias", 0);
    let win = num(a, "--window", 150);
    let at: Vec<i64> = flag(a, "--at")
        .ok_or("--at MS,MS,... is required: the instants something must happen")?
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    if at.is_empty() {
        return Err("--at parsed to no times".into());
    }
    let top = num(a, "--top", 40) as usize;
    let threads = num(
        a,
        "--threads",
        std::thread::available_parallelism().map(|v| v.get() as i64).unwrap_or(8),
    ) as usize;

    let recs = record::read_samples_pair(&dump, reclen);
    if recs.len() < 8 {
        return Err(format!("{} holds {} instants, too few", dump, recs.len()));
    }
    let ms: Vec<i64> = recs.iter().map(|(c, _, _)| *c as i64 - bias).collect();
    println!(
        "{} instants, {} .. {} ms, reclen {}, {} events, window +/-{} ms",
        recs.len(),
        ms.first().copied().unwrap_or(0),
        ms.last().copied().unwrap_or(0),
        reclen,
        at.len(),
        win
    );

    // An instant is "near an event" if any event is within the window. This is
    // computed once and shared: it is the same for every offset, and it is what
    // separates a hit from noise.
    let near: Vec<bool> = ms
        .iter()
        .map(|t| at.iter().any(|e| (t - e).abs() <= win))
        .collect();
    let n_near = near.iter().filter(|b| **b).count();
    if n_near == 0 {
        return Err(format!(
            "no instant is within {} ms of any event -- the events and the gather do not \
             overlap, which is a phase or bias error and not an absence",
            win
        ));
    }
    println!(
        "{} of {} instants are near an event ({:.1} %) -- a slot that changed at random \
         would score about that",
        n_near,
        ms.len(),
        100.0 * n_near as f64 / ms.len() as f64
    );

    // Both writes of the tick are gathered; the car's own fields have been
    // measured on either, so both are swept and the width is reported.
    let series = |off: usize, w: usize, r: &(u32, Vec<u8>, Vec<u8>)| -> u64 {
        let b = &r.2;
        match w {
            1 => b[off] as u64,
            2 => u16::from_le_bytes([b[off], b[off + 1]]) as u64,
            _ => u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]) as u64,
        }
    };

    let mut all: Vec<Hit> = Vec::new();
    for width in [1usize, 4] {
        let chunk = reclen.div_ceil(threads);
        let found: Vec<Vec<Hit>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|k| {
                    let recs = &recs;
                    let near = &near;
                    let ms = &ms;
                    s.spawn(move || {
                        let lo = k * chunk;
                        let hi = ((k + 1) * chunk).min(reclen.saturating_sub(width));
                        let mut out: Vec<Hit> = Vec::new();
                        for off in lo..hi {
                            let mut prev = series(off, width, &recs[0]);
                            let mut distinct = 1usize;
                            let mut seen: u64 = prev;
                            let mut hits = 0usize;
                            let mut noise = 0usize;
                            let mut hit_at: Vec<bool> = vec![false; recs.len()];
                            for i in 1..recs.len() {
                                let v = series(off, width, &recs[i]);
                                if v != seen {
                                    distinct = 2;
                                    seen = v;
                                }
                                if v != prev {
                                    if near[i] {
                                        hit_at[i] = true;
                                    } else {
                                        noise += 1;
                                    }
                                }
                                prev = v;
                            }
                            if distinct < 2 {
                                continue; // constant: nothing to align
                            }
                            // Count EVENTS covered, not instants: a slot that
                            // twitches five times inside one event window has
                            // found one event, not five.
                            for e in 0..1 {
                                let _ = e;
                            }
                            hits = hit_at.iter().enumerate().filter(|(_, h)| **h).fold(
                                (0usize, -1i64),
                                |(n, last), (i, _)| {
                                    let t = ms[i];
                                    if last < 0 || (t - last).abs() > 2 * win {
                                        (n + 1, t)
                                    } else {
                                        (n, last)
                                    }
                                },
                            ).0;
                            if hits == 0 {
                                continue;
                            }
                            out.push(Hit { off, width, hits, noise, distinct });
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for f in found {
            all.extend(f);
        }
    }

    // Rank: every event covered first, then the quietest between them.
    all.sort_by(|a, b| b.hits.cmp(&a.hits).then(a.noise.cmp(&b.noise)));
    let perfect = all.iter().filter(|h| h.hits == at.len() && h.noise == 0).count();
    println!(
        "\n{} offsets change at least once near an event; {} cover ALL {} events with ZERO \
         change between them",
        all.len(),
        perfect,
        at.len()
    );
    if perfect > 40 {
        println!(
            "  {} is a large tie, and a tie is not a location: intersect it with the same \
             command on another map before believing any member of it",
            perfect
        );
    }
    println!("\n{:>10} {:>6} {:>6} {:>7} {:>9}", "offset", "width", "hits", "noise", "distinct");
    for h in all.iter().take(top) {
        println!(
            "{:>10} {:>6} {:>6} {:>7} {:>9}",
            h.off, h.width, h.hits, h.noise, h.distinct
        );
    }
    if all.iter().all(|h| h.hits < at.len()) {
        println!(
            "\nNo offset covers every event. That is a real answer and not a failure: either \
             the quantity is not in this window, or it does not change at these times."
        );
    }
    Ok(())
}

fn flag(a: &[String], n: &str) -> Option<String> {
    a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned()
}
fn num(a: &[String], n: &str, d: i64) -> i64 {
    flag(a, n).map(|v| v.parse().expect("a number")).unwrap_or(d)
}

// ---------------------------------------------------------------------------
// THE FIRST RUN, AND ITS PERMUTATION FLOOR
//
// 276874 `untitled 01`, 260 instants on the recording's 50 ms grid, the window
// gathered around the car by `fk carrier scan --dump`. Events: the run's two
// crossings of the boost gates' own plane z = 752, interpolated by
// `tmtraj route --cross` (race 1.423 and 5.988).
//
//     offset   width  hits  noise      relative to the car (record +1046380)
//     1046347      1     2     13      car - 33
//     1045483      1     2     13      car - 897   = (car-33) - 864
//     1044819      1     2     13      car - 1561  = (car-33) - 1728
//     1046395      1     2     15      car + 15
//     1045531      1     2     15      car + 15    - 864
//
// The stride-864 chain is real structure — 864 is the stride of the array of
// copies of the vehicle state — so those rows are one field seen in three
// copies, not three findings.
//
// AND IT DOES NOT CLEAR ITS FLOOR. The same command on times where nothing
// happens (3200/8400, 700/10200, 2500/9100) also finds offsets covering both
// events, at noise 15, 26 and 48. The real events score 13. **Thirteen against
// a floor of fifteen is not a location**, and reporting `car - 33` from it
// would be exactly the "agreement is not confirmation" failure this project
// keeps writing up.
//
// The reason is power, not method: two events over 260 instants means 3.1 % of
// the record is "near an event", and 2000 offsets change somewhere in a window
// that wide. What would give this test teeth:
//
//   * MORE EVENTS. A map whose run crosses many gates, or a longer run.
//   * A FINER GRID. The gather uses the recording's 50 ms sampling because a
//     320 KB window at 10 ms fills a disk — but a narrow window (a few KB
//     around the car) at 10 ms is affordable and multiplies the instants by 5.
//   * A SHAPE, not just a change. Reactor state rises at the gate and DECAYS;
//     scoring that shape is far sharper than scoring "something changed".
