//! `tmtraj geom envelope` — what the ROUTE is worth, measured from the field's
//! own driving rather than from its lap times.
//!
//! WHY (arm `ksi2`, 134672, 2026-08-22)
//! ------------------------------------
//! The standing bound on 134672 is a sum of best SECTORS — five numbers on a
//! 67 s lap — and it says the route is worth 63.263 against an author time of
//! 58.687. Five numbers can only see a swap between whole sectors, and every
//! run in this field throws away time in a different place *inside* a sector.
//!
//! This module asks the same question at a 10 m grain, and it asks it in a
//! coordinate where the question is meaningful.
//!
//! **Arclength along a run's OWN path is not a place.** A run that spins or
//! loops travels further to reach the same point, so its "speed at 1500 m" is
//! a speed somewhere else entirely — four of this map's fifteen records have
//! paths 200-1200 m longer than the ribbon. So every sample is projected onto
//! one reference centreline first, and the speed that matters is
//! **d(reference arclength)/dt: how fast the run is getting down the route**,
//! which charges a detour for being a detour and charges a slide for nothing.
//!
//! Then two envelopes, and the difference between them is the point:
//!
//! * **raw** — the fastest anyone has ever gone through each 10 m of the
//!   route. Optimistic beyond achievability: it stitches together cars in
//!   states that cannot be reached from one another.
//! * **feasible** — the same envelope after a forward-backward pass under
//!   acceleration limits *measured from this field on this map*, so a bin
//!   cannot claim a speed the car could not have got to, or got away from.
//!
//! ## The control
//!
//! Both passes reduce, on one run's own data, to reconstructing that run's own
//! lap. `--self-control` does exactly that for each input file and prints the
//! reconstruction against the file's real time. A projection bug, an
//! integration bug or a wrong reference all show up there as a number that is
//! not the run's own time — and a bound whose instrument cannot reproduce a
//! known lap is not evidence about anything.

use gbx::record::{self, Decoded, Sample};

pub struct Run {
    pub name: String,
    /// reference-arclength of each sample, metres
    pub s: Vec<f64>,
    /// sample time, seconds
    pub t: Vec<f64>,
    /// median distance from the reference line -- how well this run even
    /// belongs on this route
    pub median_miss: f64,
}

/// A reference centreline: positions and their cumulative arclength.
pub struct Ref {
    pub p: Vec<[f64; 3]>,
    pub s: Vec<f64>,
}

fn d3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (x * x + y * y + z * z).sqrt()
}

impl Ref {
    pub fn from(d: &Decoded) -> Ref {
        let p: Vec<[f64; 3]> =
            d.samples.iter().filter(|s| s.time_ms >= 0).map(|s| [s.x as f64, s.y as f64, s.z as f64]).collect();
        let mut s = vec![0.0];
        for i in 1..p.len() {
            s.push(s[i - 1] + d3(&p[i - 1], &p[i]));
        }
        Ref { p, s }
    }

    pub fn total(&self) -> f64 {
        *self.s.last().unwrap_or(&0.0)
    }

    /// Index of the reference sample at or just before `arc`.
    pub fn index_at(&self, arc: f64) -> usize {
        self.seg_at(arc)
    }

    fn seg_at(&self, arc: f64) -> usize {
        match self.s.binary_search_by(|x| x.partial_cmp(&arc).unwrap()) {
            Ok(i) => i.min(self.p.len() - 2),
            Err(i) => i.saturating_sub(1).min(self.p.len() - 2),
        }
    }

}

/// Project one run onto the reference, returning (reference arclength, time).
///
/// Two constraints make this measurement rather than fiction, and both were
/// paid for by a wrong answer:
///
/// * **Monotone.** The index only advances, so a car that spins, stops or
///   reverses banks no route progress until it is past where it was.
/// * **No faster than the car moved.** The advance in reference arclength is
///   capped at the distance the car actually travelled between the two
///   samples. You cannot get down a route faster than you move through space,
///   whatever the geometry says. Before this cap, a run momentarily nearest to
///   the next fold of the sausage projected metres forward for free, and the
///   envelope came out at 42 s — sixteen seconds under an author time that
///   nobody has ever approached. The acceleration table gave it away first:
///   5225 m/s².
///
/// `median_miss` reports how far this run sits from the reference line, which
/// is the number that says whether it is on this route at all.
/// Project one run onto the reference by **banded monotone alignment** (DTW).
///
/// Three greedy projections were tried first and all three failed a control:
///
/// * nearest-point with a wide window jumped to the next fold of the sausage
///   and bought free metres (envelope 41.5 s, acceleration table 5225 m/s²);
/// * nearest-point capped by the car's own displacement DEADLOCKED at a
///   hairpin, where the point nearest a car at the apex is on the leg it came
///   in on — every run in the field stalled at the same 500 m;
/// * a predictor-corrector lost lock and drifted 17-22 m off the line.
///
/// A greedy rule decides each sample from local information, and on a folded
/// route local information is ambiguous. The alignment is a global decision:
/// one monotone correspondence between the run's samples and the reference's,
/// minimising total distance, computed by dynamic programming. Monotone by
/// construction, and it cannot jump a fold because a jump makes the samples
/// on either side of it expensive.
///
/// The band keeps it O(n·w): a run and the reference are both driving the same
/// route, so the fraction of the route completed cannot differ from the
/// fraction of the run's own path completed by more than `BAND` metres of
/// reference. That is an assumption, and it is checked — `median_miss` says
/// how far the aligned pairs actually are apart, and a run that does not
/// belong on this route shows up there.
pub fn project_run(name: &str, d: &Decoded, r: &Ref) -> Run {
    const BAND: f64 = 220.0;
    let sm: Vec<&Sample> = d.samples.iter().filter(|x| x.time_ms >= 0).collect();
    let n = sm.len();
    let m = r.p.len();
    if n < 2 || m < 2 {
        return Run { name: name.to_string(), s: vec![0.0], t: vec![0.0], median_miss: 0.0 };
    }
    let q: Vec<[f64; 3]> = sm.iter().map(|x| [x.x as f64, x.y as f64, x.z as f64]).collect();
    // the run's own cumulative arclength, as the alignment's prior
    let mut own = vec![0.0f64];
    for i in 1..n {
        own.push(own[i - 1] + d3(&q[i - 1], &q[i]));
    }
    let own_total = *own.last().unwrap();
    let ref_total = r.total();
    let band_of = |i: usize| -> (usize, usize) {
        let guess = if own_total > 0.0 { own[i] / own_total * ref_total } else { 0.0 };
        let lo = r.index_at((guess - BAND).max(0.0));
        let hi = r.index_at((guess + BAND).min(ref_total));
        (lo, hi.max(lo))
    };
    // cost[j] for the current row, with back-pointers packed per row
    let mut prev_lo = 0usize;
    let mut prev: Vec<f64> = Vec::new();
    let mut back: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut lows: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let (lo, hi) = band_of(i);
        let w = hi - lo + 1;
        let mut cur = vec![f64::MAX; w];
        let mut bk = vec![0u8; w];
        for jj in 0..w {
            let j = lo + jj;
            let local = d3(&q[i], &r.p[j]);
            if i == 0 {
                // the run must start at the start of the route
                cur[jj] = local + (r.s[j]) * 0.5;
                bk[jj] = 3;
                continue;
            }
            // three predecessors: (i-1,j) hold, (i-1,j-1) step, (i,j-1) skip
            let mut best = f64::MAX;
            let mut which = 0u8;
            let pj = |j: usize| -> f64 {
                if j >= prev_lo && j < prev_lo + prev.len() {
                    prev[j - prev_lo]
                } else {
                    f64::MAX
                }
            };
            let a = pj(j);
            if a < best {
                best = a;
                which = 1;
            }
            if j > 0 {
                let b = pj(j - 1);
                if b < best {
                    best = b;
                    which = 2;
                }
                if jj > 0 && cur[jj - 1] < best {
                    best = cur[jj - 1];
                    which = 3;
                }
            }
            if best == f64::MAX {
                continue;
            }
            cur[jj] = best + local;
            bk[jj] = which;
        }
        prev_lo = lo;
        prev = cur;
        back.push(bk);
        lows.push(lo);
    }
    // backtrack from the best end cell
    let mut j = {
        let mut b = (f64::MAX, 0usize);
        for (jj, c) in prev.iter().enumerate() {
            if *c < b.0 {
                b = (*c, prev_lo + jj);
            }
        }
        b.1
    };
    let mut align = vec![usize::MAX; n];
    let mut i = n - 1;
    loop {
        if align[i] == usize::MAX {
            align[i] = j;
        }
        if i == 0 {
            break;
        }
        let jj = j - lows[i];
        match back[i][jj] {
            2 => {
                i -= 1;
                j = j.saturating_sub(1);
            }
            3 => {
                j = j.saturating_sub(1);
            }
            _ => {
                i -= 1;
            }
        }
    }
    for a in align.iter_mut() {
        if *a == usize::MAX {
            *a = 0;
        }
    }
    // enforce monotone non-decreasing arclength (backtracking gives it, but a
    // truncated path must not leave holes)
    let mut s: Vec<f64> = align.iter().map(|j| r.s[*j]).collect();
    for i in 1..n {
        if s[i] < s[i - 1] {
            s[i] = s[i - 1];
        }
    }
    let t: Vec<f64> = sm.iter().map(|x| x.time_ms as f64 / 1000.0).collect();
    let mut miss: Vec<f64> = (0..n).map(|i| d3(&q[i], &r.p[align[i]])).collect();
    miss.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = miss[n / 2];
    Run { name: name.to_string(), s, t, median_miss: med }
}

/// Time each run spends crossing each bin of the route, by interpolation of
/// the projected arclength. `None` where the run never crossed that bin.
///
/// **A bin's value is a TIME, not a speed.** The first version took the
/// maximum sample speed inside each bin, which is biased fast — a single run
/// scored 3.1 s under its own lap, because the fastest of the five samples in
/// a bin is not the rate the bin was crossed at. Measuring the crossing time
/// makes the one-run case an identity: the bins of a run sum to its own lap,
/// and the self-control is exact rather than approximately right.
pub fn bin_times(r: &Run, total: f64, bin: f64) -> Vec<Option<f64>> {
    let n = (total / bin).ceil() as usize;
    // time at which the run first reaches each bin boundary
    let mut cross: Vec<Option<f64>> = vec![None; n + 1];
    let mut k = 0usize;
    for i in 1..r.s.len() {
        while k <= n {
            let edge = (k as f64 * bin).min(total);
            if r.s[i] < edge {
                break;
            }
            if r.s[i] > r.s[i - 1] {
                let f = (edge - r.s[i - 1]) / (r.s[i] - r.s[i - 1]);
                cross[k] = Some(r.t[i - 1] + f * (r.t[i] - r.t[i - 1]));
            } else {
                cross[k] = Some(r.t[i]);
            }
            k += 1;
        }
    }
    (0..n)
        .map(|b| match (cross[b], cross[b + 1]) {
            (Some(a), Some(c)) if c > a => Some(c - a),
            _ => None,
        })
        .collect()
}

pub struct Bins {
    pub bin: f64,
    pub n: usize,
    /// shortest observed crossing time of each bin, seconds
    pub tmin: Vec<Option<f64>>,
    pub who: Vec<usize>,
}

impl Bins {
    /// Mean speed implied by the shortest crossing, m/s.
    pub fn vmax(&self) -> Vec<f64> {
        self.tmin
            .iter()
            .map(|t| match t {
                Some(x) if *x > 0.0 => self.bin / x,
                _ => 0.0,
            })
            .collect()
    }
}

/// The raw envelope: the shortest time anyone takes to cross each bin.
pub fn envelope(runs: &[Run], total: f64, bin: f64) -> Bins {
    let n = (total / bin).ceil() as usize;
    let mut tmin: Vec<Option<f64>> = vec![None; n];
    let mut who = vec![usize::MAX; n];
    for (k, r) in runs.iter().enumerate() {
        let bt = bin_times(r, total, bin);
        for i in 0..n {
            if let Some(x) = bt[i] {
                if tmin[i].map(|c| x < c).unwrap_or(true) {
                    tmin[i] = Some(x);
                    who[i] = k;
                }
            }
        }
    }
    Bins { bin, n, tmin, who }
}

/// Longitudinal acceleration limits, measured from the field on this map.
///
/// Returned as (accel_max, decel_max) in m/s^2, both positive, per speed bin
/// of `vstep` m/s. Taken as a high quantile rather than the maximum: one
/// sample of a car hitting a wall is not a braking capability, and one sample
/// of a boost pad is not an engine.
pub fn accel_limits(runs: &[Run], vstep: f64, nv: usize, q: f64) -> (Vec<f64>, Vec<f64>) {
    let mut acc: Vec<Vec<f64>> = vec![Vec::new(); nv];
    let mut dec: Vec<Vec<f64>> = vec![Vec::new(); nv];
    for r in runs {
        for i in 2..r.s.len() {
            let dt1 = r.t[i - 1] - r.t[i - 2];
            let dt2 = r.t[i] - r.t[i - 1];
            if dt1 <= 0.0 || dt2 <= 0.0 {
                continue;
            }
            let v1 = (r.s[i - 1] - r.s[i - 2]) / dt1;
            let v2 = (r.s[i] - r.s[i - 1]) / dt2;
            if v1 < 0.0 || v2 < 0.0 {
                continue;
            }
            let a = (v2 - v1) / dt2;
            let k = ((v1 / vstep) as usize).min(nv - 1);
            if a > 0.0 {
                acc[k].push(a);
            } else {
                dec[k].push(-a);
            }
        }
    }
    let quant = |mut v: Vec<f64>| -> f64 {
        if v.len() < 20 {
            // Too few observations in this speed band to call it a limit.
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q) as usize]
    };
    let mut a: Vec<f64> = acc.into_iter().map(quant).collect();
    let mut e: Vec<f64> = dec.into_iter().map(quant).collect();
    // A speed band this field never visited has no measured limit. Carrying
    // the nearest measured value forward is the honest fill; leaving a zero
    // there silently caps the forward pass at a standstill, which is how the
    // first version of this returned 'feasible laps' of 690 s.
    fill(&mut a);
    fill(&mut e);
    (a, e)
}

fn fill(v: &mut [f64]) {
    let mut last = f64::NAN;
    for x in v.iter_mut() {
        if x.is_nan() {
            *x = last;
        } else {
            last = *x;
        }
    }
    let mut last = f64::NAN;
    for x in v.iter_mut().rev() {
        if x.is_nan() {
            *x = last;
        } else {
            last = *x;
        }
    }
}

/// Forward-backward pass: no bin may claim a speed the car could not reach
/// from the bin before it, nor one it could not shed before the bin after.
pub fn feasible(v: &[f64], bin: f64, acc: &[f64], dec: &[f64], vstep: f64) -> Vec<f64> {
    let nv = acc.len();
    let at = |tab: &[f64], s: f64| tab[((s / vstep) as usize).min(nv - 1)];
    let mut f = v.to_vec();
    for i in 1..f.len() {
        let a = at(acc, f[i - 1]).max(0.01);
        let cap = (f[i - 1] * f[i - 1] + 2.0 * a * bin).sqrt();
        if f[i] > cap {
            f[i] = cap;
        }
    }
    for i in (0..f.len() - 1).rev() {
        let d = at(dec, f[i + 1]).max(0.01);
        let cap = (f[i + 1] * f[i + 1] + 2.0 * d * bin).sqrt();
        if f[i] > cap {
            f[i] = cap;
        }
    }
    f
}

/// Sum the envelope over the route. Bins nobody crossed are UNPRICED and are
/// reported, never silently treated as free or as infinite: an unpriced bin is
/// a hole in the evidence, and a bound that hides one is not a bound.
pub fn integrate(v: &[f64], bin: f64, total: f64) -> (f64, usize) {
    let mut t = 0.0;
    let mut unpriced = 0;
    for (i, s) in v.iter().enumerate() {
        let lo = i as f64 * bin;
        let hi = ((i + 1) as f64 * bin).min(total);
        if hi <= lo {
            break;
        }
        if *s <= 0.1 {
            unpriced += 1;
            continue;
        }
        t += (hi - lo) / s;
    }
    (t, unpriced)
}

pub fn load(path: &str) -> Option<Decoded> {
    record::decode_ghost(path).ok()
}

/// Fill bins nobody crossed by linear interpolation of their neighbours.
///
/// A hole is not a standstill. Left as a zero it caps the forward pass at
/// nothing and the "feasible lap" comes out at 690 s — which is not a
/// conservative answer, it is a broken one.
pub fn interpolate_holes(v: &mut [f64]) {
    let n = v.len();
    let mut i = 0;
    while i < n {
        if v[i] > 0.0 {
            i += 1;
            continue;
        }
        let lo = if i == 0 { None } else { Some(i - 1) };
        let mut j = i;
        while j < n && v[j] <= 0.0 {
            j += 1;
        }
        let hi = if j < n { Some(j) } else { None };
        let fill = match (lo, hi) {
            (Some(a), Some(b)) => {
                let (va, vb) = (v[a], v[b]);
                for k in i..j {
                    let f = (k - a) as f64 / (b - a) as f64;
                    v[k] = va + f * (vb - va);
                }
                None
            }
            (Some(a), None) => Some(v[a]),
            (None, Some(b)) => Some(v[b]),
            (None, None) => Some(1.0),
        };
        if let Some(x) = fill {
            for k in i..j {
                v[k] = x;
            }
        }
        i = j;
    }
}
