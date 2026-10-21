//! `tmtraj blockdiff` — put two block censuses in ONE frame and print, per
//! block, what actually differs.
//!
//! WHY (arm `wtr_`, 284238, 2026-08-22)
//! ------------------------------------
//! A map that repeats one module under a symmetry, or a pair of maps that share
//! a module, is a source of ANSWER KEYS: a human line on one copy is a line on
//! every copy, provided the copies really are copies. The existing check
//! (`tmmaps bowl sym`) reports one aggregate — *"129 images land on a same-model
//! block, worst 0.846 m"* — which is enough to say the map IS a repetition and
//! not enough to say what one copy's launcher does differently from another's.
//!
//! On 284238 every geometric claim about the obstacle had been one number
//! quoted out of that aggregate ("the kicker is 1.00 m lower"), and the
//! experiment built on it moved ONE BLOCK. This command printed the
//! neighbourhood instead, and the kicker turned out to be a **four-block
//! assembly** carrying that offset rigidly — so raising "the kicker" raised a
//! quarter of it and built a 1 m step, which is why the decisive experiment of
//! two earlier arms could not have worked. Same command, run copy-against-copy,
//! then showed the three water launchers are internally identical to 1 mm and
//! differ only by a rigid placement, which closed a whole line of enquiry
//! ("maybe copy 2 is kinder") without an engine.
//!
//! It also has to be able to say NO: a block with no same-model counterpart
//! prints `NO-IMAGE` rather than being matched to something far away, and the
//! count of those is the last line.
//!
//! INPUT is a tab-separated census, either shape:
//!   `model  x  y  z  yaw  pitch  roll`                     (7+ fields)
//!   `Bnnn  model  x  y  z  yaw  pitch  roll  flags`        (8+ fields, id first)
//! which is what `tmmaps` writes for free blocks and for the baked census.

use crate::cli;

/// A screw: rotate by `angle` about the vertical axis through `(cx, cz)`, then
/// raise by `dy`. `pow(k)` is the transform applied k times, which is how a
/// repeated module's copy k is mapped back onto copy 0.
#[derive(Clone, Copy)]
pub struct Screw {
    pub cx: f64,
    pub cz: f64,
    pub angle: f64,
    pub dy: f64,
}

impl Screw {
    pub fn pow(&self, k: i32) -> impl Fn([f64; 3]) -> [f64; 3] + '_ {
        let a = self.angle * k as f64;
        let (c, s) = (a.cos(), a.sin());
        let dy = self.dy * k as f64;
        move |p: [f64; 3]| {
            let (x, z) = (p[0] - self.cx, p[2] - self.cz);
            [c * x - s * z + self.cx, p[1] + dy, s * x + c * z + self.cz]
        }
    }
}

pub struct Block {
    pub model: String,
    pub p: [f64; 3],
    pub yaw: f64,
}

/// Parse a census, tolerating both shapes. A line that does not carry three
/// finite coordinates is skipped rather than guessed at.
pub fn parse_census(text: &str) -> Vec<Block> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        // The id-first shape is only taken when the first field cannot be a
        // model name and the row is long enough to hold one.
        let (model, off) = if f.len() >= 8 && f[0].starts_with('B') && f[0][1..].chars().all(|c| c.is_ascii_digit()) {
            (f[1], 2)
        } else if f.len() >= 7 {
            (f[0], 1)
        } else {
            continue;
        };
        let g = |i: usize| -> Option<f64> { f.get(off + i)?.trim().parse().ok() };
        match (g(0), g(1), g(2)) {
            (Some(x), Some(y), Some(z)) => out.push(Block {
                model: model.to_string(),
                p: [x, y, z],
                yaw: g(3).unwrap_or(f64::NAN),
            }),
            _ => continue,
        }
    }
    out
}

pub struct Hit {
    pub model: String,
    pub a: [f64; 3],
    /// `None` when no block of the same model exists at all.
    pub d: Option<[f64; 3]>,
}

/// For every A block inside `bbox`, the nearest same-model B block's offset.
pub fn diff(a: &[Block], b: &[Block], bbox: [f64; 6]) -> Vec<Hit> {
    let inb = |p: [f64; 3]| {
        p[0] >= bbox[0] && p[0] <= bbox[1] && p[1] >= bbox[2] && p[1] <= bbox[3] && p[2] >= bbox[4] && p[2] <= bbox[5]
    };
    let mut out = Vec::new();
    for x in a.iter().filter(|x| inb(x.p)) {
        let mut best: Option<(f64, [f64; 3])> = None;
        for y in b.iter().filter(|y| y.model == x.model) {
            let d = [y.p[0] - x.p[0], y.p[1] - x.p[1], y.p[2] - x.p[2]];
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if best.as_ref().is_none_or(|(bn, _)| n < *bn) {
                best = Some((n, d));
            }
        }
        out.push(Hit { model: x.model.clone(), a: x.p, d: best.map(|(_, d)| d) });
    }
    out
}

const USAGE: &str = "\
usage: tmtraj blockdiff --a A.tsv --b B.tsv [flags]

  --a FILE --b FILE     two block censuses (tab separated)
  --ka K   --kb K       apply the screw K times to that side (default 0)
  --screw CX,CZ,DEG,DY  the screw that UNDOES one copy, applied k times by
                        --ka/--kb (default 772.2857,821.4282,120,56 -- the
                        284238/279008 module, whose own screw is -120,-56)
  --box x0,x1,y0,y1,z0,z1
                        only report A blocks whose transformed position is in
                        the box (default: everything)
  --model SUBSTR        only models containing SUBSTR
";

pub fn cmd(argv: &[String]) -> i32 {
    let a = cli::parse("tmtraj blockdiff", argv, &[]);
    let pa = a.one("a").unwrap_or_default().to_string();
    let pb = a.one("b").unwrap_or_default().to_string();
    let ka: i32 = a.num("ka", 0);
    let kb: i32 = a.num("kb", 0);
    let sc: Vec<f64> = {
        let v = a.many("screw");
        if v.is_empty() {
            vec![772.2857, 821.4282, 120.0, 56.0]
        } else {
            v.iter().filter_map(|s| s.parse().ok()).collect()
        }
    };
    let bx: Vec<f64> = {
        let v = a.many("box");
        if v.is_empty() {
            vec![f64::MIN, f64::MAX, f64::MIN, f64::MAX, f64::MIN, f64::MAX]
        } else {
            v.iter().filter_map(|s| s.parse().ok()).collect()
        }
    };
    let want = a.one("model").unwrap_or_default().to_string();
    let a = a.finish(USAGE);
    let _ = &a;
    if pa.is_empty() || pb.is_empty() || sc.len() != 4 || bx.len() != 6 {
        eprint!("{}", USAGE);
        return 2;
    }
    let rd = |p: &str| match std::fs::read_to_string(p) {
        Ok(t) => Some(parse_census(&t)),
        Err(e) => {
            eprintln!("tmtraj blockdiff: {}: {}", p, e);
            None
        }
    };
    let (ba, bb) = match (rd(&pa), rd(&pb)) {
        (Some(x), Some(y)) => (x, y),
        _ => return 2,
    };
    let screw = Screw { cx: sc[0], cz: sc[1], angle: sc[2].to_radians(), dy: sc[3] };
    let ta = screw.pow(ka);
    let tb = screw.pow(kb);
    let mut la: Vec<Block> = ba
        .iter()
        .map(|x| Block { model: x.model.clone(), p: ta(x.p), yaw: x.yaw })
        .collect();
    let lb: Vec<Block> = bb
        .iter()
        .map(|x| Block { model: x.model.clone(), p: tb(x.p), yaw: x.yaw })
        .collect();
    if !want.is_empty() {
        la.retain(|x| x.model.contains(&want));
    }
    let hits = diff(&la, &lb, [bx[0], bx[1], bx[2], bx[3], bx[4], bx[5]]);
    println!("# blockdiff  A {} (k {})  B {} (k {})", pa, ka, pb, kb);
    println!("# model\tAx\tAy\tAz\tdx\tdy\tdz\t|d|");
    let (mut miss, mut worst) = (0usize, 0.0f64);
    for h in &hits {
        match h.d {
            Some(d) => {
                let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                worst = worst.max(n);
                println!(
                    "{}\t{:.3}\t{:.3}\t{:.3}\t{:+.3}\t{:+.3}\t{:+.3}\t{:.3}",
                    h.model, h.a[0], h.a[1], h.a[2], d[0], d[1], d[2], n
                );
            }
            None => {
                miss += 1;
                println!("{}\t{:.3}\t{:.3}\t{:.3}\tNO-IMAGE\t\t\t", h.model, h.a[0], h.a[1], h.a[2]);
            }
        }
    }
    println!("# {} blocks compared, {} with no same-model counterpart, worst residual {:.3} m", hits.len(), miss, worst);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    // An asymmetric fixture: one block that matches exactly, one that is offset
    // by a known amount, and one with no counterpart at all. A fixture whose
    // two sides agree everywhere pins nothing -- it passes for a differ that
    // always prints zero.
    const A: &str = "Kicker\t10.0\t20.0\t30.0\t0.0\t0.0\t0.0\n\
                     Deck\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n\
                     WaterRamp\t5.0\t0.0\t5.0\t0.0\t0.0\t0.0\n";
    const B: &str = "Kicker\t10.2\t21.0\t30.5\t0.0\t0.0\t0.0\n\
                     Deck\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n";

    #[test]
    fn offsets_are_reported_per_axis_and_absence_is_reported_as_absence() {
        let a = parse_census(A);
        let b = parse_census(B);
        assert_eq!(a.len(), 3);
        let h = diff(&a, &b, [f64::MIN, f64::MAX, f64::MIN, f64::MAX, f64::MIN, f64::MAX]);
        assert_eq!(h.len(), 3);
        let k = h.iter().find(|x| x.model == "Kicker").unwrap().d.unwrap();
        assert!((k[0] - 0.2).abs() < 1e-9 && (k[1] - 1.0).abs() < 1e-9 && (k[2] - 0.5).abs() < 1e-9);
        assert_eq!(h.iter().find(|x| x.model == "Deck").unwrap().d.unwrap(), [0.0, 0.0, 0.0]);
        assert!(h.iter().find(|x| x.model == "WaterRamp").unwrap().d.is_none());
    }

    #[test]
    fn the_box_excludes_and_the_id_first_shape_parses() {
        let a = parse_census(A);
        let b = parse_census(B);
        let h = diff(&a, &b, [9.0, 11.0, f64::MIN, f64::MAX, f64::MIN, f64::MAX]);
        assert_eq!(h.len(), 1, "only the Kicker is inside x 9..11");
        let idfirst = parse_census("B146\tKicker\t10.0\t20.0\t30.0\t0.5\t0.0\t0.0\t0x20600000\t\n");
        assert_eq!(idfirst.len(), 1);
        assert_eq!(idfirst[0].model, "Kicker");
        assert_eq!(idfirst[0].p, [10.0, 20.0, 30.0]);
    }

    // The screw is what makes a repeated module comparable with itself. Three
    // applications of a -120 deg screw come back to the same x/z and three
    // times the drop, which is the property every copy-against-copy run relies
    // on; if it did not hold, "copy 2 is an image of copy 0" would be a
    // statement about the transform rather than about the map.
    #[test]
    fn three_screws_return_the_horizontal_position_and_stack_the_drop() {
        let s = Screw { cx: 772.2857, cz: 821.4282, angle: (-120.0f64).to_radians(), dy: -56.0 };
        let p = [900.0, 1873.0, 925.0];
        let q = s.pow(3)(p);
        assert!((q[0] - p[0]).abs() < 1e-6, "x returned: {}", q[0]);
        assert!((q[2] - p[2]).abs() < 1e-6, "z returned: {}", q[2]);
        assert!((q[1] - (p[1] - 168.0)).abs() < 1e-9);
    }
}
