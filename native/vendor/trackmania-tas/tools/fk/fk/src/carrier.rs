//! Naming the sample bytes a regenerated ghost inherits from its carrier.
//!
//! A `CSceneVehicleVis` sample is 116 bytes. `fk regen` writes 22 of them (the
//! transform) and `--inputs` writes 3 more (the tape echo); the other **91 are
//! still the donor's**, which is why `ghost regen` names them every time it
//! writes a file. They are not unavailable — the engine computes every one of
//! them and they are in its memory — so this module is the measurement that
//! turns "unnamed" into either a named byte or an enumerated gap.
//!
//! # The three ways this measurement lies, and what is done about each
//!
//! **1. A single answer key manufactures entries.** With one recording as the
//! key a byte-map fit once reported 6 of 116 bytes available at 94–99 % exact;
//! with five keys on the same map only 2 survived. So a fit is a HYPOTHESIS
//! here and nothing else: [`scan`] proposes, and a proposal is only a result
//! once [`confirm`] has scored it — at the offset and with the coefficients the
//! scan froze, with **no refit** — on other recordings.
//!
//! **2. A best-of-N scan lands high by chance.** At ~460 paired instants the
//! best of 33 000 candidates sits about four standard deviations above the mean
//! for free. Every scan therefore runs itself a second time against a
//! **row-permuted copy of the same target column** and reports the best score
//! that got. A channel whose real best does not clear its own permutation best
//! is noise, and the table says so on the same line rather than in a footnote.
//!
//! **3. A correlation with a monotone target proves nothing.** Wheel rotation,
//! dirt and race time all rise together, so *any* clock in the address space
//! correlates with them. Everything here is therefore computed on **increments**
//! (`v[i+1] - v[i]`), where a clock has zero variance and drops out by
//! construction, and never on levels.
//!
//! # The encoding
//!
//! One form covers every channel measured so far:
//!
//! ```text
//!     target == (floor(k * slot + c)) mod M
//! ```
//!
//! with `M` = 256 for a byte and 65536 for a two-byte channel. A wheel rotation
//! is an angle that wraps, which is what the modulus is for; rpm does not wrap,
//! and the same form fits it with a `k` small enough that it never has to.
//! `k` is fitted by regression of the wrapped target increments on the slot
//! increments — through the origin, so `c` cannot absorb a slope — and `c` is
//! then the modal residual, which is the value that maximises exact agreement
//! rather than one that minimises a square.



/// A quantity in the recorded sample: one byte, or two bytes read as a
/// little-endian `u16`.
///
/// The pairs are not a guess of mine: `gbx::record`'s field table has carried
/// `fl_wheel_rot` as "bytes 6,7: rotation + rotation count" since the decoder
/// was ported from GBX.NET. Reading them as one `u16` is what makes a wrapping
/// angle a single linear channel instead of two ragged ones.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Channel {
    Byte(usize),
    U16(usize),
}

impl Channel {
    pub fn modulus(self) -> u32 {
        match self {
            Channel::Byte(_) => 256,
            Channel::U16(_) => 65536,
        }
    }
    pub fn value(self, sample: &[u8]) -> Option<u32> {
        match self {
            Channel::Byte(b) => sample.get(b).map(|v| *v as u32),
            Channel::U16(b) => match (sample.get(b), sample.get(b + 1)) {
                (Some(lo), Some(hi)) => Some(*lo as u32 | ((*hi as u32) << 8)),
                _ => None,
            },
        }
    }
    pub fn name(self) -> String {
        match self {
            Channel::Byte(b) => format!("b{}", b),
            Channel::U16(b) => format!("u16@{}", b),
        }
    }
    pub fn parse(s: &str) -> Option<Channel> {
        if let Some(r) = s.strip_prefix("u16@") {
            return r.parse().ok().map(Channel::U16);
        }
        s.strip_prefix('b').and_then(|r| r.parse().ok()).map(Channel::Byte)
    }
}

/// Which write of a tick a channel is read from, NAMED RELATIVE TO THE CAR.
///
/// The engine writes the vehicle state more than once inside a 10 ms tick, and
/// the recorder captured one of those instants. Which one, in absolute terms,
/// is not a property of the game: it depends on how many writes the gather
/// happened to keep, and it MOVES between runs — the car matched the first
/// write on five keys here and the last on three, on the same binary.
///
/// So a table cannot say "first". It says `car` — the same instant the
/// position that identified the car came from — or `other`. Naming it
/// absolutely is what made a frozen table score 100 % on five keys and 1 % on
/// three: on those three every offset was right and every instant was wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Write {
    /// The write the car's own position was identified on.
    Car,
    /// The other one.
    Other,
}

impl Write {
    pub fn name(self) -> &'static str {
        match self {
            Write::Car => "car",
            Write::Other => "other",
        }
    }
    pub fn parse(s: &str) -> Option<Write> {
        match s {
            "car" => Some(Write::Car),
            "other" => Some(Write::Other),
            _ => None,
        }
    }
}

/// The engine and the recording, at the same instants.
pub struct Paired {
    pub ms: Vec<i64>,
    /// The recorded 116-byte sample at each instant.
    pub sample: Vec<Vec<u8>>,
    /// Gathered engine bytes, TRANSPOSED, indexed by [`Write`]: `cols[0]` is
    /// the write the car was identified on and `cols[1]` is the other.
    ///
    /// Transposed because both passes below walk one memory offset across every
    /// instant, and the row-major dump makes that one cache miss per instant.
    pub cols: [Vec<u8>; 2],
    pub reclen: usize,
    /// Where the CHOSEN car's position sits inside a gathered record. Every
    /// offset this module reports is relative to it, because that is the only
    /// anchor that transfers: the copies of the car are an array at stride 864
    /// and a locate lands on an arbitrary member.
    pub pos_off: usize,
}

impl Paired {
    pub fn n(&self) -> usize {
        self.ms.len()
    }
    fn col(&self, w: Write, o: usize) -> &[u8] {
        let n = self.n();
        let c = &self.cols[w as usize];
        &c[o * n..o * n + n]
    }
    /// The f32 at record offset `o`, across every instant.
    pub fn f32col(&self, w: Write, o: usize) -> Vec<f64> {
        let n = self.n();
        let (b0, b1, b2, b3) =
            (self.col(w, o), self.col(w, o + 1), self.col(w, o + 2), self.col(w, o + 3));
        (0..n)
            .map(|i| f32::from_le_bytes([b0[i], b1[i], b2[i], b3[i]]) as f64)
            .collect()
    }
    /// The channel's recorded values, across every instant.
    pub fn target(&self, ch: Channel) -> Option<Vec<u32>> {
        self.sample.iter().map(|s| ch.value(s)).collect()
    }
}

/// A fitted encoding: `target == (floor(k * slot + c)) mod M`.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub k: f64,
    /// REAL, not an integer step. `c` is where the quantiser's grid sits, and
    /// forcing it onto integers forces a choice between `floor` and `round`
    /// that nothing justifies: rpm fitted 81.3 % exact with an integer `c` and
    /// 92.7 % once `c` was allowed its own fraction, on the same slot with the
    /// same `k`. The project has been bitten by this before from the other
    /// side — the input echo was a `round` where the game writes a `floor`,
    /// worth a Cohen's kappa of 0.467 against 1.000 — so the grid offset is
    /// fitted rather than assumed.
    pub c: f64,
    pub exact: usize,
    pub n: usize,
}

impl Fit {
    pub fn rate(&self) -> f64 {
        self.exact as f64 / self.n.max(1) as f64
    }
}

/// Score an encoding whose coefficients are already known. No fitting happens
/// here, which is the whole point of the confirmation pass.
pub fn score(v: &[f64], t: &[u32], m: u32, k: f64, c: f64) -> Fit {
    let mut exact = 0usize;
    let mut n = 0usize;
    for (vi, ti) in v.iter().zip(t.iter()) {
        let x = k * vi + c;
        if !x.is_finite() || x.abs() > 1e15 {
            continue;
        }
        n += 1;
        if (x.floor() as i64).rem_euclid(m as i64) as u32 == *ti {
            exact += 1;
        }
    }
    Fit { k, c, exact, n }
}

/// The correlation of the WRAPPED target increments on the slot increments.
///
/// This is the cheap filter, and it is computed on increments rather than
/// levels for the reason in the module header: on levels, every counter in the
/// address space matches every monotone field. A constant slot has no increment
/// variance and returns `None` rather than a correlation of nothing.
pub fn incr_corr(v: &[f64], t: &[u32], m: u32) -> Option<f64> {
    let n = v.len().min(t.len());
    if n < 8 {
        return None;
    }
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0.0, 0usize);
    for i in 1..n {
        let dx = v[i] - v[i - 1];
        if !dx.is_finite() {
            return None;
        }
        let dy = wrapped_delta(t[i - 1], t[i], m);
        sx += dx;
        sy += dy;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
        cnt += 1;
    }
    let c = cnt as f64;
    let vx = sxx - sx * sx / c;
    let vy = syy - sy * sy / c;
    if vx <= 0.0 || vy <= 0.0 {
        return None;
    }
    Some((sxy - sx * sy / c) / (vx * vy).sqrt())
}

/// `b - a` taken into `(-M/2, M/2]`, which is what makes a wrapping angle
/// differentiable.
fn wrapped_delta(a: u32, b: u32, m: u32) -> f64 {
    let d = (b as i64 - a as i64).rem_euclid(m as i64);
    (if d > m as i64 / 2 { d - m as i64 } else { d }) as f64
}

/// `k` from the wrapped target increments, robustly.
///
/// The through-the-origin regression in [`fit`] is the right estimator when the
/// SLOT is a plain accumulator, and it is destroyed when the slot is itself an
/// angle that wraps: at 100 km/h a wheel turns twice between two 50 ms samples,
/// so a slot holding a phase wraps on nearly every increment and a least-square
/// slope fitted through those is meaningless. The median of the per-increment
/// ratios is not: a wrap corrupts an increment, and corrupted increments have to
/// be a MAJORITY before they move a median.
///
/// Restricted to increments whose slot movement is above the median magnitude,
/// because a ratio with a near-zero denominator is noise however robust the
/// estimator around it is.
fn slope_median(v: &[f64], t: &[u32], m: u32) -> Option<f64> {
    let n = v.len().min(t.len());
    if n < 16 {
        return None;
    }
    let mut mags: Vec<f64> = Vec::with_capacity(n);
    for i in 1..n {
        let d = (v[i] - v[i - 1]).abs();
        if !d.is_finite() {
            return None;
        }
        mags.push(d);
    }
    let mid = mags.len() / 2;
    let mut sorted = mags.clone();
    sorted.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let cut = sorted[mid].max(f64::MIN_POSITIVE);
    let mut r: Vec<f64> = Vec::with_capacity(mid + 1);
    for i in 1..n {
        let dx = v[i] - v[i - 1];
        if dx.abs() < cut {
            continue;
        }
        r.push(wrapped_delta(t[i - 1], t[i], m) / dx);
    }
    if r.len() < 8 {
        return None;
    }
    let h = r.len() / 2;
    r.select_nth_unstable_by(h, |a, b| a.total_cmp(b));
    Some(r[h]).filter(|k| k.is_finite() && *k != 0.0)
}

/// Fit `k` and `c`, then count exact agreement.
///
/// Two estimators are tried for `k` — a through-the-origin regression of the
/// wrapped target increments on the slot increments, and the median of the
/// per-increment ratios — because they fail on opposite things: the regression
/// is the better estimator on a clean accumulator and is destroyed by a slot
/// that wraps, and the median survives the wrap and is coarser. Whichever
/// AGREES EXACTLY more often wins, which is a decision made on the data rather
/// than on an assumption about what the slot is.
///
/// A local refinement follows, over a small multiplicative neighbourhood. It
/// exists because `floor` is unforgiving: a `k` a thousandth of a per cent out
/// still correlates at 0.99999 and agrees exactly on almost nothing once the
/// product has grown past a few hundred.
pub fn fit(v: &[f64], t: &[u32], m: u32) -> Option<Fit> {
    let n = v.len().min(t.len());
    if n < 8 {
        return None;
    }
    let (mut sxx, mut sxy) = (0.0f64, 0.0f64);
    for i in 1..n {
        let dx = v[i] - v[i - 1];
        if !dx.is_finite() {
            return None;
        }
        sxx += dx * dx;
        sxy += dx * wrapped_delta(t[i - 1], t[i], m);
    }
    let mut best: Option<Fit> = None;
    for k0 in [
        (sxx > 0.0).then(|| sxy / sxx).filter(|k| k.is_finite() && *k != 0.0),
        slope_median(v, t, m),
    ]
    .into_iter()
    .flatten()
    .filter(|k| plausible_k(*k))
    {
        let mut b: Option<Fit> = None;
        // +-0.2 % in 81 steps, then +-0.005 % in 81 steps around the winner.
        // Two decades of refinement for 162 evaluations rather than a fine
        // sweep of the whole range.
        for (span, steps) in [(2e-3f64, 40i64), (5e-5, 40)] {
            let centre = b.map(|x: Fit| x.k).unwrap_or(k0);
            for s in -steps..=steps {
                let k = centre * (1.0 + span * s as f64 / steps as f64);
                let f = fit_c(v, t, m, k);
                if b.map_or(true, |x| f.exact > x.exact) {
                    b = Some(f);
                }
            }
        }
        if let Some(f) = b {
            if plausible_k(f.k) && best.map_or(true, |x: Fit| f.exact > x.exact) {
                best = Some(f);
            }
        }
    }
    best
}

/// With `k` fixed, the `c` that maximises exact agreement, EXACTLY.
///
/// Each instant is satisfied by an interval of `c` one grid step wide —
/// `floor(k*v + c) == t` holds precisely for `c` in `[t - k*v, t - k*v + 1)` —
/// so the best `c` is the point covered by the most unit intervals on a circle
/// of circumference `M`. Sorting the interval starts and sweeping finds it in
/// one pass, and there is nothing left to guess: no rounding convention, no
/// half-step, no "close enough".
///
/// This is the inner loop of the whole sweep — tens of millions of calls — so
/// it sorts rather than hashing and allocates one vector.
fn fit_c(v: &[f64], t: &[u32], m: u32, k: f64) -> Fit {
    let mm = m as f64;
    let mut s: Vec<f64> = Vec::with_capacity(v.len());
    for (vi, ti) in v.iter().zip(t.iter()) {
        let x = k * vi;
        if !x.is_finite() || x.abs() > 1e15 {
            continue;
        }
        s.push((*ti as f64 - x).rem_euclid(mm));
    }
    let n = s.len();
    if n == 0 {
        return Fit { k, c: 0.0, exact: 0, n: 0 };
    }
    s.sort_unstable_by(f64::total_cmp);
    // Sweep: for each interval start, how many starts lie in [s_i, s_i + 1) on
    // the circle. The winner's own start is the leftmost `c` that satisfies
    // them all, and the midpoint of the covered stretch is the most robust
    // representative of it — a `c` sitting exactly on a boundary is one ULP
    // from disagreeing on the instant that put it there.
    let (mut best, mut bi, mut bj) = (0usize, 0usize, 0.0f64);
    let mut j = 0usize;
    for i in 0..n {
        while j < i + n && s[j % n] + if j >= n { mm } else { 0.0 } < s[i] + 1.0 {
            j += 1;
        }
        if j - i > best {
            best = j - i;
            bi = i;
            // the last start inside the window: c may sit anywhere in
            // [last_start, first_start + 1)
            bj = s[(j - 1) % n] + if j - 1 >= n { mm } else { 0.0 };
        }
    }
    let c = ((bj + (s[bi] + 1.0)) / 2.0).rem_euclid(mm);
    Fit { k, c, exact: best, n }
}

// ===========================================================================
// The sweep
// ===========================================================================

/// One channel's best candidate, with the number that says whether to believe
/// it.
#[derive(Clone, Debug)]
pub struct Cand {
    pub ch: Channel,
    pub write: Write,
    /// Record offset RELATIVE TO THE CHOSEN CAR's position.
    pub rel: i64,
    pub kind: Kind,
    pub fit: Fit,
    /// The best score the same sweep reached against a row-permuted copy of
    /// this channel. The multiple-comparison floor, measured rather than
    /// assumed.
    pub null: f64,
    /// The rate of this channel's single commonest value: what a constant
    /// scores. A candidate below it has learnt nothing.
    pub baseline: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// The sample byte is a copy of an engine byte.
    Raw,
    /// `target == (floor(k * f32@rel + c)) mod M`.
    Affine,
    /// `target == (floor(k * u8@rel + c)) mod M` — the slot is a small INTEGER,
    /// not a float.
    ///
    /// This exists because without it the fitter reports a lookup table as a
    /// law. Gear is stored as an integer and recorded as `4 * gear + 1`; read
    /// as an f32 that integer is a DENORMAL of about 1e-45, and the affine fit
    /// obligingly returned `k = 2.85e45` — a perfect 100 % on all eight keys,
    /// with a coefficient that means nothing and would transfer to nothing.
    /// A number that large is the fitter telling you it has been handed the
    /// wrong type.
    AffineU8,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Raw => "raw",
            Kind::Affine => "affine",
            Kind::AffineU8 => "affineu8",
        }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "raw" => Some(Kind::Raw),
            "affine" => Some(Kind::Affine),
            "affineu8" => Some(Kind::AffineU8),
            _ => None,
        }
    }
}

/// A slope this large is not a coefficient, it is the fitter compensating for
/// having read an integer as a float (see [`Kind::AffineU8`]); one this small is
/// it compensating for a slot that barely moves. Neither transfers to another
/// run, so neither is allowed out of a fit.
fn plausible_k(k: f64) -> bool {
    k.is_finite() && k.abs() >= 1e-9 && k.abs() <= 1e9
}

/// Deterministic row permutation. A shuffled copy of a target column has the
/// same marginal distribution and no relationship to the engine at all, so the
/// best score a sweep reaches on it is exactly the multiple-comparison floor
/// for that channel, that many candidates and that many instants.
fn permute<T: Copy>(v: &[T], seed: u64) -> Vec<T> {
    let mut out = v.to_vec();
    let mut s = seed | 1;
    for i in (1..out.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.swap(i, (s % (i as u64 + 1)) as usize);
    }
    out
}

fn modal_rate(t: &[u32]) -> f64 {
    let mut h: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for v in t {
        *h.entry(*v).or_insert(0) += 1;
    }
    h.values().copied().max().unwrap_or(0) as f64 / t.len().max(1) as f64
}

/// Sweep every gathered offset against every channel, on both writes of the
/// tick, and return the best candidates per channel — beside the score the same
/// sweep got on permuted data.
///
/// `top` candidates are kept per channel rather than one, because the best is
/// often not alone and the runners-up are the evidence. Two of the four
/// ground-contact bytes picked the SAME engine byte on the first key here, for
/// the honest reason that both wheels were on the same surface for the whole
/// run; only the runners-up show that each has its own slot at stride 44. A
/// table with one row per channel would have recorded that coincidence as the
/// answer.
///
/// `threads` splits the offset range. Nothing is shared mutably: each worker
/// keeps its own table and the tables are merged at the end, so the answer does
/// not depend on how many cores ran it.
pub fn sweep(
    p: &Paired,
    channels: &[Channel],
    threads: usize,
    seed: u64,
    top: usize,
    floor: Option<&std::collections::HashMap<String, usize>>,
) -> Vec<Cand> {
    let n = p.n();
    let targets: Vec<(Channel, Vec<u32>, Vec<u32>, f64)> = channels
        .iter()
        .filter_map(|ch| {
            let t = p.target(*ch)?;
            // A channel that never changes has nothing to find and would
            // otherwise match every constant in the address space.
            if t.iter().all(|v| *v == t[0]) {
                return None;
            }
            let base = modal_rate(&t);
            let sh = permute(&t, seed ^ (ch.name().len() as u64) ^ (t[0] as u64) << 8);
            Some((*ch, t, sh, base))
        })
        .collect();

    let hi = p.reclen.saturating_sub(4);
    let chunk = hi.div_ceil(threads.max(1));
    let parts: Vec<Vec<(Vec<Cand>, f64)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.max(1))
            .map(|w| {
                let targets = &targets;
                s.spawn(move || {
                    let lo = w * chunk;
                    let end = ((w + 1) * chunk).min(hi);
                    let mut best: Vec<(Vec<Cand>, f64)> =
                        (0..targets.len()).map(|_| (Vec::new(), 0.0)).collect();
                    // The FLOOR turns "the best offset" into "every offset that
                    // does as well as the best". Which matters more than it
                    // sounds: b24 agrees exactly at hundreds of offsets on one
                    // key, so reporting one of them reports an arbitrary member
                    // of a large set, and two keys then look like they disagree
                    // when they are both right. The intersection ACROSS keys is
                    // the discriminator, so each key has to contribute its whole
                    // set.
                    let fl = |ci: usize, e: usize| -> bool {
                        match floor {
                            None => true,
                            Some(m) => e >= *m.get(&targets[ci].0.name()).unwrap_or(&usize::MAX),
                        }
                    };
                    if lo >= end {
                        return best;
                    }
                    for wr in [Write::Car, Write::Other] {
                        // RAW: a byte-for-byte copy. Every offset, not just the
                        // aligned ones.
                        for o in lo..end {
                            let col = p.col(wr, o);
                            for (ci, (ch, t, sh, base)) in targets.iter().enumerate() {
                                if !matches!(ch, Channel::Byte(_)) {
                                    continue;
                                }
                                let mut e = 0usize;
                                let mut en = 0usize;
                                for i in 0..n {
                                    if col[i] as u32 == t[i] {
                                        e += 1;
                                    }
                                    if col[i] as u32 == sh[i] {
                                        en += 1;
                                    }
                                }
                                let rn = en as f64 / n as f64;
                                if rn > best[ci].1 {
                                    best[ci].1 = rn;
                                }
                                if fl(ci, e) {
                                    keep(&mut best[ci].0, top, Cand {
                                        ch: *ch,
                                        write: wr,
                                        rel: o as i64 - p.pos_off as i64,
                                        kind: Kind::Raw,
                                        fit: Fit { k: 1.0, c: 0.0, exact: e, n },
                                        null: 0.0,
                                        baseline: *base,
                                    });
                                }
                            }
                            // The same byte column read as an INTEGER, for the
                            // fields the engine stores as one. Every channel,
                            // not just the single-byte ones: a two-byte channel
                            // can be an integer field too.
                            let iv: Vec<f64> = col.iter().map(|b| *b as f64).collect();
                            if iv.iter().any(|x| *x != iv[0]) {
                            for (ci, (ch, t, sh, base)) in targets.iter().enumerate() {
                                let m = ch.modulus();
                                for (tt, is_null) in [(t, false), (sh, true)] {
                                    // The same two shortlists as the f32 pass,
                                    // and for the same reason: the 162-step
                                    // refinement is affordable per candidate
                                    // and not per offset.
                                    let sharp =
                                        incr_corr(&iv, tt, m).map_or(false, |r| r.abs() >= 0.9);
                                    if !sharp {
                                        let Some(k1) = slope_median(&iv, tt, m) else { continue };
                                        if fit_c(&iv, tt, m, k1).rate() <= base + 0.02 {
                                            continue;
                                        }
                                    }
                                    let Some(f) = fit(&iv, tt, m) else { continue };
                                    if is_null {
                                        if f.rate() > best[ci].1 {
                                            best[ci].1 = f.rate();
                                        }
                                    } else if fl(ci, f.exact) {
                                        keep(&mut best[ci].0, top, Cand {
                                            ch: *ch,
                                            write: wr,
                                            rel: o as i64 - p.pos_off as i64,
                                            kind: Kind::AffineU8,
                                            fit: f,
                                            null: 0.0,
                                            baseline: *base,
                                        });
                                    }
                                }
                            }
                            }
                        }
                        // AFFINE: a linear function of an f32 slot, floored and
                        // wrapped. Two stages -- an increment correlation to
                        // shortlist, then the exact fit, because the fit costs
                        // 162 evaluations and there are tens of thousands of
                        // slots.
                        let alo = lo.next_multiple_of(4);
                        for o in (alo..end).step_by(4) {
                            let v = p.f32col(wr, o);
                            for (ci, (ch, t, sh, base)) in targets.iter().enumerate() {
                                let m = ch.modulus();
                                for (tt, is_null) in [(t, false), (sh, true)] {
                                    // TWO SHORTLISTS, because they fail on
                                    // opposite things. The increment
                                    // correlation is the cheap and sharp one
                                    // and it is destroyed by a slot that wraps
                                    // — a wheel phase turns over twice between
                                    // two 50 ms samples at racing speed. The
                                    // one-shot fit at the median slope survives
                                    // that and costs a sort. A slot has to
                                    // clear one of them to earn the 162-step
                                    // refinement.
                                    let sharp =
                                        incr_corr(&v, tt, m).map_or(false, |r| r.abs() >= 0.9);
                                    if !sharp {
                                        let Some(k1) = slope_median(&v, tt, m) else { continue };
                                        if fit_c(&v, tt, m, k1).rate() <= base + 0.02 {
                                            continue;
                                        }
                                    }
                                    let Some(f) = fit(&v, tt, m) else { continue };
                                    let rate = f.rate();
                                    if is_null {
                                        if rate > best[ci].1 {
                                            best[ci].1 = rate;
                                        }
                                    } else if fl(ci, f.exact) {
                                        let _ = rate;
                                        keep(&mut best[ci].0, top, Cand {
                                            ch: *ch,
                                            write: wr,
                                            rel: o as i64 - p.pos_off as i64,
                                            kind: Kind::Affine,
                                            fit: f,
                                            null: 0.0,
                                            baseline: *base,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    best
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut out: Vec<Cand> = Vec::new();
    for ci in 0..targets.len() {
        let mut merged: Vec<Cand> = Vec::new();
        let mut null = 0.0f64;
        for part in &parts {
            if part[ci].1 > null {
                null = part[ci].1;
            }
            for c in &part[ci].0 {
                keep(&mut merged, top, c.clone());
            }
        }
        for mut c in merged {
            c.null = null;
            out.push(c);
        }
    }
    out.sort_by(|a, b| {
        b.fit.rate().total_cmp(&a.fit.rate()).then(a.ch.name().cmp(&b.ch.name()))
    });
    out
}

/// Keep the best `top` candidates, best first. A tiny insertion sort: `top` is
/// four and this is called once per surviving slot.
fn keep(v: &mut Vec<Cand>, top: usize, c: Cand) {
    if v.len() >= top && v.last().map_or(false, |l| l.fit.rate() >= c.fit.rate()) {
        return;
    }
    let at = v.partition_point(|x| x.fit.rate() >= c.fit.rate());
    v.insert(at, c);
    v.truncate(top);
}

// ===========================================================================
// The frozen table
// ===========================================================================

/// A row of a frozen table.
pub struct Row {
    pub ch: Channel,
    pub write: Write,
    pub rel: i64,
    pub kind: Kind,
    pub k: f64,
    pub c: f64,
}

pub fn read_table(path: &str) -> Result<Vec<Row>, String> {
    // `--carrier layout` is not a file. It asks for the game's OWN writer --
    // `vislayout::pack`, transcribed from the archiver at 0x9cfed0 -- instead of
    // a table of coefficients somebody fitted one channel at a time. It is
    // represented as a single row with `rel == i64::MIN` so that every caller's
    // plumbing (the `--carrier` flag, `must_be_live`, the gather's width
    // calculation) keeps working unchanged, and the field gather recognises the
    // sentinel and packs the whole 116 bytes.
    //
    // Prefer it. The table's rows were the only way to write these channels
    // before the writer was read out of the binary, and five of the 23 turned
    // out to encode the wrong law -- each scoring 100.00 % once read the way the
    // game writes it, against 92-97 % as fitted.
    if path == "layout" {
        return Ok(vec![Row {
            ch: Channel::Byte(0),
            write: Write::Car,
            rel: i64::MIN,
            kind: Kind::Raw,
            k: 0.0,
            c: 0.0,
        }]);
    }
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut out = Vec::new();
    for l in s.lines() {
        // `#` is a comment, and the checked-in table opens with four of them
        // defining what `car` is. An offset without its anchor is not a
        // measurement, so the definition travels with the numbers.
        if l.starts_with('#') || l.starts_with("channel\t") {
            continue;
        }
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() < 6 {
            continue;
        }
        out.push(Row {
            ch: Channel::parse(f[0]).ok_or(format!("bad channel {:?}", f[0]))?,
            write: Write::parse(f[1]).ok_or(format!("bad write {:?}", f[1]))?,
            rel: f[2].parse().map_err(|_| "bad rel")?,
            kind: Kind::parse(f[3]).ok_or(format!("bad kind {:?}", f[3]))?,
            k: f[4].parse().map_err(|_| "bad k")?,
            c: f[5].parse().map_err(|_| "bad c")?,
        });
    }
    Ok(out)
}

