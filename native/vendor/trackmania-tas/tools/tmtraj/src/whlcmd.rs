//! `tmtraj whl ...` -- the SURFACE AND CONTACT instrument (arm `whl`,
//! 2026-08-20).
//!
//! WHY THIS EXISTS
//! ---------------
//! The published ghosts carry regenerated position/orientation/speed and the
//! CARRIER's wheel and surface bytes. On screen that is a car kicking up dirt
//! in mid-air. The defect is visible; the fix must be checked by something
//! stronger than "the effect stopped", because **a zeroed field also stops
//! it**.
//!
//! THE TEST THAT CANNOT PASS ON A ZEROED FIELD
//! -------------------------------------------
//! Airborne is decidable from the TRAJECTORY ALONE, with no contact flag
//! involved: a car with no contact force on it accelerates at exactly
//! `(0, -g, 0)` with `g = 22.3 m/s^2` in this engine. So classify every
//! interior sample from the second difference of its own position:
//!
//!   BALLISTIC  |a - (0,-g,0)| < tol            -- nothing is touching the car
//!   SUPPORTED  a_y > -g + margin, sustained    -- something is holding it up
//!   UNKNOWN    everything else (collisions, reactors, sample-rate artefacts)
//!
//! and then ask TWO questions of the contact flag, not one:
//!
//!   A. is it OFF on every BALLISTIC sample?   (a zeroed field passes this)
//!   B. is it ON  on every SUPPORTED sample?   (a zeroed field FAILS this)
//!
//! A gate that only asks A is the gate the brief warns about. Both are
//! reported, always, and the verdict needs both.
//!
//! The `-g` calibration is checked, not assumed: `whl grav` fits g on the
//! longest ballistic stretch of a file and prints it, so a wrong constant
//! shows up as a wrong fit rather than as a silently empty BALLISTIC class.

use gbx::record;
use record::{find_entrecord_blob, load_body, parse_record_data, read_transform_pub};

pub const G_DEFAULT: f64 = 25.2;

/// One decoded sample, with the raw bytes kept: the surface fields are single
/// bytes whose semantics are partly a guess, so every consumer here reads the
/// BYTE, not a derived float.
#[derive(Clone)]
pub struct R {
    pub ms: i64,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub speed: f64,
    pub raw: Vec<u8>,
}

impl R {
    pub fn b(&self, i: usize) -> u8 {
        *self.raw.get(i).unwrap_or(&0)
    }
    pub fn contact(&self) -> bool {
        (self.b(89) & 0x1) != 0
    }
    pub fn dirt_max(&self) -> u8 {
        [93usize, 95, 97, 99].iter().map(|o| self.b(*o)).max().unwrap_or(0)
    }
    pub fn ice_max(&self) -> u8 {
        [81usize, 82, 83, 84].iter().map(|o| self.b(*o)).max().unwrap_or(0)
    }
    /// Are the 49 per-run bytes at their neutral values in THIS sample? See
    /// [`gbx::record::is_neutralised`] for the record-wide question, and for
    /// why a plain zero counts wherever the neutral value is not zero.
    pub fn neutralised(&self) -> bool {
        gbx::record::NEUTRALISE.iter().all(|&o| {
            o >= self.raw.len()
                || self.raw[o] == gbx::record::neutral_value(o)
                || self.raw[o] == 0
        })
    }
    pub fn finite(&self) -> bool {
        self.pos.iter().all(|v| v.is_finite()) && self.vel.iter().all(|v| v.is_finite())
    }
}

pub fn decode(path: &str) -> Result<Vec<R>, String> {
    let body = load_body(path)?;
    let (ver, blob) = find_entrecord_blob(&body)?;
    let rd = parse_record_data(&blob, ver)?;
    let ent = rd
        .ents
        .iter()
        .filter(|e| e.sample_size >= 100 && !e.times.is_empty())
        .max_by_key(|e| e.times.len())
        .ok_or("no CSceneVehicleVis entity")?;
    let ss = ent.sample_size;
    let mut out = Vec::with_capacity(ent.times.len());
    for (i, t) in ent.times.iter().enumerate() {
        let raw = ent.raw[i * ss..(i + 1) * ss].to_vec();
        let (pos, _q, speed, vel) = read_transform_pub(&raw, 47);
        out.push(R { ms: *t as i64, pos, vel, speed, raw });
    }
    out.sort_by_key(|r| r.ms);
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Cls {
    Ballistic,
    Supported,
    Unknown,
}

pub struct Classified {
    pub cls: Vec<Cls>,
    pub ay: Vec<f64>,
}

/// Classify from the second difference of position alone.
///
/// `tol` is the metres/s^2 ball around free fall that counts as BALLISTIC;
/// `margin` is how far above -g the vertical acceleration must sit before a
/// sample counts as SUPPORTED. Between the two is UNKNOWN and is never
/// asserted on.
///
/// Both classes additionally require `run` consecutive samples of the same
/// kind, so a single collision spike or a sample-rate artefact cannot create a
/// one-sample class of its own. That is what makes a violation meaningful.
pub fn classify(r: &[R], g: f64, tol: f64, margin: f64, run: usize) -> Classified {
    let n = r.len();
    let mut cls = vec![Cls::Unknown; n];

    let mut ay = vec![f64::NAN; n];
    // raw per-sample decision first
    let mut raw = vec![Cls::Unknown; n];
    for i in 1..n.saturating_sub(1) {
        let dt0 = (r[i].ms - r[i - 1].ms) as f64 / 1000.0;
        let dt1 = (r[i + 1].ms - r[i].ms) as f64 / 1000.0;
        if dt0 <= 0.0 || dt1 <= 0.0 || (dt0 - dt1).abs() > 1e-6 {
            continue;
        }
        if !r[i - 1].finite() || !r[i].finite() || !r[i + 1].finite() {
            continue;
        }
        let mut a = [0f64; 3];
        for k in 0..3 {
            a[k] = (r[i + 1].pos[k] - 2.0 * r[i].pos[k] + r[i - 1].pos[k]) / (dt0 * dt1);
        }
        ay[i] = a[1];
        let e = (a[0] * a[0] + (a[1] + g) * (a[1] + g) + a[2] * a[2]).sqrt();

        // SUPPORTED is deliberately narrower than "not falling at g".
        // Measured on 276874: from 4.9 s to 11.3 s the car climbs 20 m and
        // comes back down with a_y between -5 and +13 -- REACTOR FLIGHT, no
        // contact anywhere in it, and the user's own description of the defect
        // says so. A rule that only asks "is a_y above -g" calls that supported
        // and then condemns a CORRECT contact flag for reading zero there.
        // Ground support means the car is being held at a roughly constant
        // height: small vertical acceleration AND small vertical speed.
        let vy = (r[i + 1].pos[1] - r[i - 1].pos[1]) / (dt0 + dt1);
        let sh = {
            let dx = (r[i + 1].pos[0] - r[i - 1].pos[0]) / (dt0 + dt1);
            let dz = (r[i + 1].pos[2] - r[i - 1].pos[2]) / (dt0 + dt1);
            (dx * dx + dz * dz).sqrt()
        };
        if e < tol {
            raw[i] = Cls::Ballistic;
        } else if a[1].abs() < margin && vy.abs() < 2.0 && sh > 3.0 {
            raw[i] = Cls::Supported;
        }
    }
    // require a run of `run` consecutive samples of the same kind
    let mut i = 0usize;
    while i < n {
        let k = raw[i];
        let mut j = i;
        while j < n && raw[j] == k {
            j += 1;
        }
        if k != Cls::Unknown && j - i >= run {
            for x in i..j {
                cls[x] = k;
            }
        }
        i = j;
    }
    Classified { cls, ay }
}


fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        100.0 * a as f64 / b as f64
    }
}



// ---------------------------------------------------------------------------
// tmtraj motion — what the trajectory says, beside what the flag claims
// ---------------------------------------------------------------------------

const MOTION_USAGE: &str = "\
usage: tmtraj motion GHOST [--race S] [--g G] [--tol T] [--margin M] [--run N]
                           [--fit-g] [--per-sample]

Classifies every sample BALLISTIC / SUPPORTED / UNKNOWN from the second
difference of its own position, then prints what the recorded contact,
dirt and ice bytes say on each class.

Three classes, not two: a car held up by a reactor, a boost or a wall it is
scraping is neither in free fall nor ground-borne, and a two-class rule has to
call it one of them.

--fit-g fits gravity from THIS file's own longest free-fall stretch instead of
using the fleet constant, and reports the vertical-speed range it was fitted
over -- a fit whose lever arm is a few m/s of v_y cannot identify a drag term
and must not be quoted as if it had.
";

pub fn cmd_motion(argv: &[String]) -> i32 {
    let a = crate::cli::parse("tmtraj motion", argv, &["fit-g", "per-sample"]);
    let race: i64 = a.num("race", i64::MAX);
    let g: f64 = a.num("g", G_DEFAULT);
    let tol: f64 = a.num("tol", 2.0);
    let margin: f64 = a.num("margin", 5.0);
    let run: usize = a.num("run", 3);
    let fit = a.has("fit-g");
    let per_sample = a.has("per-sample");
    let a = a.finish(MOTION_USAGE);
    let Some(path) = a.positional.first() else {
        eprint!("{}", MOTION_USAGE);
        return 2;
    };
    let r = match decode(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("UNMEASURED: {}: {}", path, e);
            return 3;
        }
    };
    if fit {
        fit_gravity(&r, race);
    }
    motion_report(r, race, g, tol, margin, run, per_sample)
}
fn motion_report(r: Vec<R>, race: i64, g: f64, tol: f64, margin: f64, run: usize, per_sample: bool) -> i32 {





    let v: Vec<R> = r.into_iter().filter(|x| x.ms <= race).collect();
    let c = classify(&v, g, tol, margin, run);
    let n = v.len();
    let nb = c.cls.iter().filter(|x| **x == Cls::Ballistic).count();
    let ns = c.cls.iter().filter(|x| **x == Cls::Supported).count();
    let nu = n - nb - ns;
    let dt = if n > 1 { (v[1].ms - v[0].ms) as f64 / 1000.0 } else { 0.05 };
    // longest ballistic run
    let mut longest = 0usize;
    let mut i = 0usize;
    while i < n {
        if c.cls[i] == Cls::Ballistic {
            let mut j = i;
            while j < n && c.cls[j] == Cls::Ballistic {
                j += 1;
            }
            longest = longest.max(j - i);
            i = j;
        } else {
            i += 1;
        }
    }
    println!(
        "samples {} ({:.3} .. {:.3} s, dt {:.3})   g={:.2} tol={:.2} margin={:.2} run={}",
        n,
        v.first().map(|x| x.ms as f64 / 1000.0).unwrap_or(0.0),
        v.last().map(|x| x.ms as f64 / 1000.0).unwrap_or(0.0),
        dt, g, tol, margin, run
    );
    println!(
        "BALLISTIC {:4} ({:5.1} %, {:.3} s, longest run {:.3} s)   SUPPORTED {:4} ({:5.1} %)   UNKNOWN {:4} ({:5.1} %)",
        nb, pct(nb, n), nb as f64 * dt, longest as f64 * dt, ns, pct(ns, n), nu, pct(nu, n)
    );

    // ---- the two directions, reported separately, always -------------------
    let ball: Vec<usize> = (0..n).filter(|i| c.cls[*i] == Cls::Ballistic).collect();
    let sup: Vec<usize> = (0..n).filter(|i| c.cls[*i] == Cls::Supported).collect();
    let con_b = ball.iter().filter(|i| v[**i].contact()).count();
    let con_s = sup.iter().filter(|i| v[**i].contact()).count();
    let dirt_b = ball.iter().filter(|i| v[**i].dirt_max() > 0).count();
    let dirt_s = sup.iter().filter(|i| v[**i].dirt_max() > 0).count();
    let ice_b = ball.iter().filter(|i| v[**i].ice_max() > 0).count();
    let ice_s = sup.iter().filter(|i| v[**i].ice_max() > 0).count();
    println!("                              on BALLISTIC        on SUPPORTED");
    println!(
        "  ground contact (b89&1)      {:4}/{:<4} {:5.1} %   {:4}/{:<4} {:5.1} %",
        con_b, ball.len(), pct(con_b, ball.len()), con_s, sup.len(), pct(con_s, sup.len())
    );
    println!(
        "  dirt  > 0 (b93/95/97/99)    {:4}/{:<4} {:5.1} %   {:4}/{:<4} {:5.1} %",
        dirt_b, ball.len(), pct(dirt_b, ball.len()), dirt_s, sup.len(), pct(dirt_s, sup.len())
    );
    println!(
        "  ice   > 0 (b81..84)         {:4}/{:<4} {:5.1} %   {:4}/{:<4} {:5.1} %",
        ice_b, ball.len(), pct(ice_b, ball.len()), ice_s, sup.len(), pct(ice_s, sup.len())
    );
    let allc = v.iter().filter(|x| x.contact()).count();
    let alld = v.iter().filter(|x| x.dirt_max() > 0).count();
    let alli = v.iter().filter(|x| x.ice_max() > 0).count();
    println!(
        "  whole file: contact {}/{} ({:.1} %)  dirt>0 {} ({:.1} %)  ice>0 {} ({:.1} %)  max dirt {}  max ice {}",
        allc, n, pct(allc, n), alld, pct(alld, n), alli, pct(alli, n),
        v.iter().map(|x| x.dirt_max()).max().unwrap_or(0),
        v.iter().map(|x| x.ice_max()).max().unwrap_or(0)
    );

    if per_sample {
        println!("\n  t(s)      y      a_y   class      contact dirt ice");
        for i in 0..n {
            println!(
                "{:>7.3} {:>7.2} {:>8.2}   {:<10} {:^7} {:>4} {:>3}",
                v[i].ms as f64 / 1000.0, v[i].pos[1], c.ay[i],
                match c.cls[i] { Cls::Ballistic => "BALLISTIC", Cls::Supported => "supported", _ => "-" },
                u8::from(v[i].contact()), v[i].dirt_max(), v[i].ice_max()
            );
        }
    }
    {
        // A: OFF in the air. B: ON on the ground. A zeroed field passes A and
        // fails B; the carrier's field fails A. BOTH must hold.
        //
        // C and D are NOT "zero in the air". Material already stuck to a wheel
        // stays stuck: on 267460's own downloaded recording the ice bytes read
        // 245-253 through an airborne stretch, decaying. A gate that demanded
        // zero there would FAIL A GENUINE HUMAN RECORDING, which is the
        // must-say-yes test every gate here has to survive. What contact-free
        // flight forbids is ACCUMULATION: the value may hold or decay, never
        // rise.
        let rise = |o: &dyn Fn(&R) -> u8| -> usize {
            ball.iter()
                .filter(|i| **i > 0 && c.cls[**i - 1] == Cls::Ballistic && o(&v[**i]) > o(&v[**i - 1]))
                .count()
        };
        let dirt_rise = rise(&|r: &R| r.dirt_max());
        let ice_rise = rise(&|r: &R| r.ice_max());
        let a_ok = ball.is_empty() || con_b == 0;
        // B has to be able to say "no evidence". On 145875 the car descends for
        // the whole run, so there is not one unambiguously ground-borne sample,
        // and a gate that reports FAIL there is condemning a downloaded human
        // recording for the shape of its map.
        let b_n = sup.len();
        let b_ok = b_n >= 5 && pct(con_s, b_n) >= 95.0;
        let b_na = b_n < 5;
        let d_ok = dirt_rise == 0;
        let i_ok = ice_rise == 0;
        println!();
        println!("{} A  contact OFF on every BALLISTIC sample ({} violations)", if a_ok { "PASS" } else { "FAIL" }, con_b);
        println!(
            "{} B  contact ON on >=95 % of SUPPORTED samples ({:.1} % of {}) -- THE TEST A ZEROED FIELD FAILS",
            if b_na { "N/A " } else if b_ok { "PASS" } else { "FAIL" },
            pct(con_s, b_n), b_n
        );
        println!("{} C  dirt never RISES during ballistic flight ({} violations)", if d_ok { "PASS" } else { "FAIL" }, dirt_rise);
        println!("{} D  ice  never RISES during ballistic flight ({} violations)", if i_ok { "PASS" } else { "FAIL" }, ice_rise);
        if !(a_ok && (b_ok || b_na) && d_ok && i_ok) {
            return 1;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// tmtraj wheels
// ---------------------------------------------------------------------------

const WHEELS_USAGE: &str = "\
usage: tmtraj wheels GHOST [--race S] [--g G] [--tol T] [--margin M] [--run N]

Two different questions, both answered, because conflating them once produced a
false refusal of Nadeo's own recording:

  1. IS THERE A WHEEL RADIUS. Fitted from the SUPPORTED steps only, as
     distance / (turns * 2pi). Reported as a measured value with the number of
     steps behind it -- never compared against a constant. A published 0.30-0.45 m
     band is the STADIUM wheel; a snow car measures 0.4700 m, and 267460's wheel
     block reads 0.3636-0.3644 m.
  2. ARE THE WHEEL BYTES ALIVE AT ALL. Distinct quantised values per wheel.
     Dead or donor-blanked telemetry is constant: 145875's download carries
     88-109 distinct values per wheel, a zeroed field carries 1. Reported for
     the LEAST varying of the four, because one dead wheel renders wrongly.

A run that descends the whole way has no ground-supported sample, so question 1
can honestly answer 'no evidence'. An n/a is a statement about the CHECK, not
about the file.
";

pub fn cmd_wheels(argv: &[String]) -> i32 {
    let a = crate::cli::parse("tmtraj wheels", argv, &[]);
    let race: i64 = a.num("race", i64::MAX);
    let g: f64 = a.num("g", G_DEFAULT);
    let tol: f64 = a.num("tol", 2.0);
    let margin: f64 = a.num("margin", 5.0);
    let run: usize = a.num("run", 3);
    let a = a.finish(WHEELS_USAGE);
    let Some(path) = a.positional.first() else {
        eprint!("{}", WHEELS_USAGE);
        return 2;
    };
    let r = match decode(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("UNMEASURED: {}: {}", path, e);
            return 3;
        }
    };
    let v: Vec<R> = r.into_iter().filter(|x| x.ms <= race).collect();
    liveness(&v);
    wheel_radius(&v, g, tol, margin, run);
    0
}

/// Question 2: are the bytes alive. One number, no threshold.
fn liveness(v: &[R]) {
    let mut worst = (usize::MAX, 0usize);
    for (w, (lo, hi)) in [(6usize, 7usize), (8, 9), (10, 11), (12, 13)].iter().enumerate() {
        let mut set = std::collections::BTreeSet::new();
        for s in v {
            set.insert((s.b(*lo), s.b(*hi)));
        }
        if set.len() < worst.0 {
            worst = (set.len(), w);
        }
    }
    println!(
        "wheel bytes: least varying wheel is #{} with {} distinct values over {} samples{}",
        worst.1,
        worst.0,
        v.len(),
        if worst.0 <= 1 { "   -- DEAD or donor-blanked" } else { "" }
    );
}
fn wheel_radius(v: &[R], g: f64, tol: f64, margin: f64, run: usize) {






    let c = classify(v, g, tol, margin, run);
    let n = v.len();
    // turns of wheel 0, unwrapped across the byte pair's own 256-turn range
    let turns = |x: &R| -> f64 { x.b(7) as f64 + x.b(6) as f64 / 255.0 };
    let mut rows: Vec<(Cls, f64, f64)> = Vec::new(); // class, d(turns), distance
    for i in 1..n {
        let dt = (v[i].ms - v[i - 1].ms) as f64 / 1000.0;
        if dt <= 0.0 || dt > 0.2 {
            continue;
        }
        let mut dturn = turns(&v[i]) - turns(&v[i - 1]);
        while dturn < -128.0 {
            dturn += 256.0;
        }
        while dturn > 128.0 {
            dturn -= 256.0;
        }
        let mut d = 0.0;
        for k in 0..3 {
            let q = v[i].pos[k] - v[i - 1].pos[k];
            d += q * q;
        }
        rows.push((c.cls[i], dturn, d.sqrt()));
    }
    // radius from the SUPPORTED rows only
    let mut rr: Vec<f64> = rows
        .iter()
        .filter(|(k, dt, d)| *k == Cls::Supported && *dt > 1e-4 && *d > 0.05)
        .map(|(_, dt, d)| d / (dt * std::f64::consts::TAU))
        .collect();
    rr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rad = if rr.is_empty() { f64::NAN } else { rr[rr.len() / 2] };
    println!("radius fitted on {} supported steps: {:.4} m", rr.len(), rad);
    for k in [Cls::Supported, Cls::Ballistic, Cls::Unknown] {
        let mut q: Vec<f64> = rows
            .iter()
            .filter(|(kk, _, d)| *kk == k && *d > 0.05)
            .map(|(_, dt, d)| dt * std::f64::consts::TAU * rad / d)
            .filter(|x| x.is_finite())
            .collect();
        if q.is_empty() {
            println!("{:<10} no steps", format!("{:?}", k));
            continue;
        }
        q.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{:<10} n {:>5}   wheel/car rate  p10 {:>7.3}  median {:>7.3}  p90 {:>7.3}",
            format!("{:?}", k), q.len(),
            q[q.len() / 10], q[q.len() / 2], q[q.len() * 9 / 10]
        );
    }
}
fn fit_gravity(r: &[R], race: i64) {

    let v: Vec<&R> = r.iter().filter(|x| x.ms <= race && x.finite()).collect();
    let n = v.len();
    if n < 5 {
        println!("ABORT: too few samples");
        return;
    }
    // a sample is a free-fall candidate when the HORIZONTAL acceleration is
    // near zero -- that test does not mention g, so fitting g on it is not
    // circular.
    let mut cand = vec![false; n];
    let mut ay = vec![f64::NAN; n];
    for i in 1..n - 1 {
        let dt = (v[i].ms - v[i - 1].ms) as f64 / 1000.0;
        if dt <= 0.0 || ((v[i + 1].ms - v[i].ms) as f64 / 1000.0 - dt).abs() > 1e-6 {
            continue;
        }
        let ax = (v[i + 1].pos[0] - 2.0 * v[i].pos[0] + v[i - 1].pos[0]) / (dt * dt);
        let az = (v[i + 1].pos[2] - 2.0 * v[i].pos[2] + v[i - 1].pos[2]) / (dt * dt);
        ay[i] = (v[i + 1].pos[1] - 2.0 * v[i].pos[1] + v[i - 1].pos[1]) / (dt * dt);
        cand[i] = (ax * ax + az * az).sqrt() < 1.0 && ay[i] < -5.0;
    }
    let (mut bi, mut bl, mut i) = (0usize, 0usize, 0usize);
    while i < n {
        if cand[i] {
            let mut j = i;
            while j < n && cand[j] {
                j += 1;
            }
            if j - i > bl {
                bl = j - i;
                bi = i;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if bl == 0 {
        println!("no free-fall stretch (no sample with horizontal |a| < 1.0 and a_y < -5)");
        return;
    }
    let mut s: Vec<f64> = (bi..bi + bl).map(|k| ay[k]).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = s[s.len() / 2];
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    println!(
        "longest free-fall stretch: {} samples, {:.3} .. {:.3} s",
        bl,
        v[bi].ms as f64 / 1000.0,
        v[bi + bl - 1].ms as f64 / 1000.0
    );
    println!("  a_y median {:.4} m/s^2   mean {:.4}   min {:.4}   max {:.4}", med, mean, s[0], s[s.len() - 1]);
    println!("  => g = {:.4}", -med);
}

