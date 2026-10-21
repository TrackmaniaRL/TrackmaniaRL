//! `fk carrier` — name the sample bytes a regenerated ghost inherits.
//!
//! ```text
//! fk carrier scan    --template KEY --map M --out CAND.tsv   # propose
//! fk carrier confirm --template KEY --map M --table CAND.tsv # score, no refit
//! ```
//!
//! # Why it is two verbs and not one
//!
//! Because a fit and a test are different things and this project has already
//! paid for conflating them: a byte-map fitted on ONE recording reported six
//! bytes at 94–99 % exact, and four of the six were coincidences that died the
//! moment a second key was asked. `scan` fits; it is allowed to be wrong.
//! `confirm` takes the frozen table — offset, encoding, `k`, `c` — and scores
//! it on a recording that had no say in choosing any of them. Nothing here can
//! refit during a confirmation, which is why the two verbs do not share a flag
//! that would let it.
//!
//! # The gather is two runs, and that is the point
//!
//! A run of the engine locates the car by searching the gathered record for a
//! self-consistent vehicle state. That search is right over 452 bytes of
//! vehicle state and useless over 320 KB of anything — measured, on the first
//! wide run made here: it walked off the car and the self-check caught it at
//! `|q|-1 = 1.34e-1`. And the offsets it would report would not transfer
//! anyway, because the copies of the car are an array at stride 864 and a
//! locate lands on an arbitrary member.
//!
//! So: **run one is narrow and identifies the car**; run two centres its window
//! on the car run one found and does not search at all. Every offset reported
//! is relative to that car, which is the only anchor that means the same thing
//! in the next process.

use crate::carrier::{self, read_table, Cand, Channel, Kind, Paired, Row, Write};
use crate::record::{self, GatherOpts};
use crate::session::Ctx;

const USAGE: &str = "\
fk carrier -- name the 91 sample bytes a regenerated ghost inherits from its carrier.

  fk carrier scan     sweep engine memory against one recording   [PROPOSES]
  fk carrier confirm  score a frozen table on another recording   [DECIDES]
  fk carrier layout   score the DISASSEMBLED 116-byte writer, all bytes at once
                      -- no fit anywhere in it, so it can only be right or wrong
  fk carrier bytes    print that layout without running anything
  fk carrier rollup   collapse several layout runs into one row per byte
  fk carrier merge    intersect several scans into one frozen table
  fk carrier write    write the named bytes into a ghost, from engine memory

  --template FILE   the answer key: a recording whose telemetry the GAME wrote
  --map FILE        its map
  --server DIR      the dedicated-server install          [$TM_SERVER]
  --shim FILE       libforkshim.so                        [$FK_SHIM]
  --work DIR        scratch; per-process by default
  --dump FILE       where the gathered engine bytes go
  --back N          bytes of engine memory to gather before the car  [1048576]
  --fwd N           and after it                                     [262144]
  --threads N       sweep workers                                    [all cores]
  --out FILE        the candidate/result table (TSV)
  --table FILE      (confirm) the table to score, from a scan
  --tag NAME        a label for the key, written into the table

Every offset in a table is relative to THE CAR, never to the located anchor:
the copies of the vehicle state are an array at stride 864 and a locate lands
on an arbitrary member of it.
";

pub fn run(a: &[String]) -> Result<(), String> {
    let verb = a.first().map(|s| s.as_str()).unwrap_or("");
    if verb.is_empty() || verb == "--help" || verb == "-h" {
        print!("{}", USAGE);
        return Ok(());
    }
    let rest = &a[1..];
    match verb {
        "scan" => scan(rest),
        "confirm" => confirm(rest),
        "layout" => layout(rest),
        "bytes" => {
            println!("byte\tfield\tencoding");
            for d in crate::vislayout::DOC {
                println!("{}\t{}\t{}", d.byte, d.field, d.encoding);
            }
            Ok(())
        }
        "merge" => merge(rest),
        "rollup" => rollup(rest),        "write" => write(rest),
        x => Err(format!("fk carrier <scan|confirm|layout|bytes|merge|write>, got {:?}", x)),
    }
}

fn flag(a: &[String], n: &str) -> Option<String> {
    a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned()
}
fn num(a: &[String], n: &str, d: i64) -> i64 {
    flag(a, n).map(|v| v.parse().expect("a number")).unwrap_or(d)
}

fn ctx(a: &[String]) -> Result<Ctx, String> {
    Ok(Ctx {
        template: flag(a, "--template").ok_or("--template FILE is required")?,
        map: flag(a, "--map").ok_or("--map FILE is required")?,
        work: flag(a, "--work")
            .unwrap_or_else(|| crate::session::Engine::default_work().to_string_lossy().into()),
        shim: flag(a, "--shim")
            .or_else(|| std::env::var("FK_SHIM").ok())
            .or_else(|| crate::session::default_shim().map(|p| p.to_string_lossy().into()))
            .ok_or("no --shim: pass one, set FK_SHIM, or build tools/search")?,
        server: flag(a, "--server")
            .or_else(|| std::env::var("TM_SERVER").ok())
            .unwrap_or_else(|| "/tmp/tmoracle/server".into()),
        ckpt: num(a, "--ckpt", 0) as u64,
    })
}

// ---------------------------------------------------------------- the gather

/// Measure the clock bias and collect every anchor candidate the locate offers.
///
/// The anchor here is doing far less work than it does in `regen`: it only has
/// to land the gather window somewhere the car is INSIDE it, because the car is
/// then identified within the window against the recording's own path. That
/// matters, because the anchor is not reliable enough to do more. Measured
/// here: an anchor that passed every structural check in one server process
/// pointed at an object **1662.8 m** from the car in the next one, and on the
/// very next attempt the same locate returned a single candidate that was a
/// frozen slot. Neither is a problem when the window is wide and the car is
/// looked for rather than assumed.
fn anchors_for(c: &Ctx, verbose: bool) -> Result<Vec<record::Anchors>, String> {
    let f = crate::tape::Tape::load(&c.template)?;
    let n = f.steer.len() as i64;
    let mut ticks: Vec<i64> = vec![200.min(n / 3).max(60)];
    let mut k = n / 2;
    while k >= 60 {
        ticks.push(k);
        k /= 2;
    }
    ticks.retain(|t| *t >= 60 && *t < n - 20);
    ticks.dedup();

    let mut bias = 0i64;
    for t in &ticks {
        if let Ok(b) = record::measure_bias(c, &f, *t, verbose) {
            bias = b;
            println!("bias {} (tick {})", b, t);
            break;
        }
    }
    if bias == 0 {
        return Err("could not measure the clock bias at any checkpoint".into());
    }
    let mut anchors: Vec<record::Anchors> = Vec::new();
    for t in &ticks {
        if let Ok(mut b) = record::measure_anchors(c, &f, *t, verbose) {
            for a in b.iter_mut() {
                a.bias = bias;
            }
            anchors.append(&mut b);
        }
    }
    anchors.dedup_by_key(|a| (a.chain.clone(), a.member));
    if anchors.is_empty() {
        return Err("the locate offered no anchor at any checkpoint".into());
    }
    println!("{} distinct anchor candidates, bias {}", anchors.len(), bias);
    Ok(anchors)
}

/// One wide run per anchor candidate, until one of them contains the car.
fn gather_wide(
    c: &Ctx,
    back: i64,
    fwd: i64,
    dump: &str,
    verbose: bool,
) -> Result<Paired, String> {
    // THE GRID IS THE RECORDING'S, NOT A CONSTANT. A ghost samples every 50 ms
    // but not necessarily on a multiple of 50: 285885's own instants are
    // 20, 70, 120 ms, so a gate phased on zero matched 0 of its 1225 samples
    // and the run aborted with "too few to fit anything" — a phase error
    // wearing the clothes of an absent signal.
    let (times, _) = record::targets_from_ghost(&c.template)?;
    let (period, phase_ms) = record::grid_of(&times);
    println!("the recording's own grid: {} ms, phase {}", period, phase_ms);
    let anchors = anchors_for(c, verbose)?;
    let mut last = String::new();
    // A window that holds only shadows is widened rather than reported as an
    // absence: the car IS in the address space, and a search that stopped at
    // its first guess would be reporting the width of its own window as a fact
    // about the engine.
    for scale in [1i64, 4] {
      let (back, fwd) = (back * scale, fwd * scale);
      for a in &anchors {
        // The shim gathers at most eight segments and two are spoken for (the
        // race clock and the production window), so the ground is cut into six.
        let mut extra: record::ExtraSegs = Vec::new();
        let span = back + fwd;
        let each = (span as u32).div_ceil(6);
        let mut o = -back;
        while o < fwd {
            let l = (each as i64).min(fwd - o) as u32;
            if l > 0 {
                extra.push((o, l));
            }
            o += each as i64;
        }
        let g = GatherOpts {
            bias_override: Some(a.bias),
            anchors: Some(a),
            // The record's own grid: a ghost sample exists only every 50 ms,
            // and a 320 KB window at 10 ms fills a disk.
            period,
            phase_ms,
            verbose,
            // Dedup on the production window ONLY. On the whole record nothing
            // ever matches -- something in 320 KB changes on every lroundf
            // call -- and the first wide run made here wrote 9.8 GB in two
            // minutes and was still going.
            dedup: Some((0, 4 + record::win_len())),
            // Neither of these is wanted: the car is identified against the
            // recording's own path, which is a stronger test than either, and
            // both of them assume the car is at the centre of the window.
            choose_copy: false,
            self_check: false,
            extra,
            ..GatherOpts::production(dump)
        };
        let two = match record::run_clean_anch(c, &g) {
            Ok(v) => v,
            Err(e) => {
                println!("anchor {}: {}", a.chain, e);
                last = e;
                continue;
            }
        };
        println!(
            "anchor {}: {} instants ({} .. {} ms), reclen {}, validator Time {:?}, \
             region {:#x}..{:#x}",
            a.chain, two.instants, two.first_ms, two.last_ms, two.reclen, two.sim_time,
            two.pos_region.0, two.pos_region.1
        );
        match pair(c, &two, dump) {
            Ok(p) => return Ok(p),
            Err(e) => {
                println!("anchor {}: {}", a.chain, e);
                last = e;
            }
        }
      }
    }
    Err(format!("no anchor put the car inside a {} byte window: {}", 4 * (back + fwd), last))
}

/// Pair the gathered instants with the recording's own samples, and prove the
/// gather is on the right car before returning anything.
fn pair(c: &Ctx, two: &record::CleanOut, dump: &str) -> Result<Paired, String> {
    let (times, raws) = record::targets_from_ghost(&c.template)?;
    let recs = record::read_samples_pair(dump, two.reclen);
    let by_ms: std::collections::HashMap<i64, usize> = recs
        .iter()
        .enumerate()
        .map(|(i, (clk, _, _))| (*clk as i64 - two.bias, i))
        .collect();
    let race_end = two.sim_time.unwrap_or(i64::MAX);

    let mut ms = Vec::new();
    let mut sample = Vec::new();
    let mut idx = Vec::new();
    for (i, t) in times.iter().enumerate() {
        if *t < 0 || *t > race_end {
            continue;
        }
        let Some(j) = by_ms.get(t) else { continue };
        ms.push(*t);
        sample.push(raws[i].clone());
        idx.push(*j);
    }
    let n = ms.len();
    if n < 40 {
        return Err(format!(
            "only {} of {} recorded in-race instants have an engine instant -- too few to fit \
             anything",
            n,
            times.len()
        ));
    }

    // THE CAR IS IDENTIFIED, NOT ASSUMED.
    //
    // Run one found the car at a byte offset from the input-array base, and
    // that offset does NOT reliably survive a second server start: the heap
    // layout is bimodal run to run. Carried over blindly it put the window
    // **1662.8 m** from this recording's own path here, on the first attempt —
    // and every structural self-check passed, because what sits there is a
    // perfectly self-consistent state of something else.
    //
    // So the wide record is searched for the car, and the thing it is searched
    // against is the answer key's own recorded positions. That is not a fit and
    // not a heuristic: a key is a recording the GAME wrote, so its positions
    // are the run. The match is millimetres over hundreds of instants and the
    // runner-up is metres, so the identification is not close.
    let (po, car_write, med, runner) = find_car(&recs, &idx, &sample, two.reclen)?;
    println!(
        "car identified at record +{} on the {} write: median {:.6} m from the recording's own \
         path over {} paired instants (runner-up {:.6} m)",
        po, car_write.name(), med, n, runner
    );

    // Transpose: both sweeps walk one memory offset across every instant, and
    // the dump is row-major.
    // Plane 0 is THE CAR'S OWN WRITE, whichever of the two that turned out to
    // be, so every offset in a table is read at the instant the car was
    // identified at. See `Write`.
    let reclen = two.reclen;
    let mut cols = [vec![0u8; reclen * n], vec![0u8; reclen * n]];
    for (i, r) in idx.iter().enumerate() {
        let (a, b) = match car_write {
            Write::Car => (&recs[*r].1, &recs[*r].2),
            Write::Other => (&recs[*r].2, &recs[*r].1),
        };
        for (w, src) in [a, b].into_iter().enumerate() {
            for o in 0..reclen {
                cols[w][o * n + i] = src[o];
            }
        }
    }
    Ok(Paired { ms, sample, cols, reclen, pos_off: po })
}

/// Every sample byte a regenerated file does NOT write, plus the four
/// wheel-rotation pairs the field table documents.
fn channels(ss: usize) -> Vec<Channel> {
    let mut v: Vec<Channel> = Vec::new();
    for b in 0..ss.min(116) {
        // 47..69 is the transform `fk regen` writes from engine state; 14, 15
        // and 18 are the tape echo. Everything else is the carrier's.
        if (47..69).contains(&b) || b == 14 || b == 15 || b == 18 {
            continue;
        }
        v.push(Channel::Byte(b));
    }
    // EVERY ADJACENT PAIR, not the four the field table happens to document.
    // Reading a pair as one `u16` is what turns a quantity that wraps into a
    // single linear channel instead of two ragged ones, and which pairs do that
    // is a measurement rather than a guess: the table's documented wheel pairs
    // came out at 99-100 % and `side_speed` at bytes 2,3 came out at 100 % with
    // a coefficient of 65535/2000 exactly, but the corpus census also reported
    // bytes 0 and 1 wrapping together and nothing names them. A pair that is
    // not a `u16` simply fails, at the cost of one more column in a sweep that
    // already runs in a minute.
    for b in 0..ss.min(116).saturating_sub(1) {
        if (46..69).contains(&b) || (13..19).contains(&b) {
            continue;
        }
        v.push(Channel::U16(b));
    }
    v
}

// ------------------------------------------------------------------ the verbs

fn scan(a: &[String]) -> Result<(), String> {
    let c = ctx(a)?;
    let verbose = a.iter().any(|x| x == "--verbose");
    let dump = flag(a, "--dump").unwrap_or_else(|| format!("/tmp/fkcarrier-{}.bin", std::process::id()));
    let tag = flag(a, "--tag").unwrap_or_else(|| crate::record::name_of(&c.template));
    let threads = num(a, "--threads", std::thread::available_parallelism().map(|v| v.get() as i64).unwrap_or(8)) as usize;

    let p = gather_wide(&c, num(a, "--back", 1048576), num(a, "--fwd", 262144), &dump, verbose)?;
    let chs = channels(p.sample[0].len());
    println!("sweeping {} offsets x {} channels x 2 writes on {} threads", p.reclen, chs.len(), threads);
    let t0 = std::time::Instant::now();
    // TWO PASSES. The first finds, per channel, the best exact count anything
    // in the window reaches. The second collects EVERY offset that reaches it.
    // One pass reporting one winner reports an arbitrary member of what is
    // often a large tie — b24 agrees exactly at hundreds of offsets — and two
    // keys picking different members of the same tie look like two keys
    // disagreeing. The set is what gets intersected, so the set is what a scan
    // has to publish.
    let cap = num(a, "--cap", 4096) as usize;
    let best = carrier::sweep(&p, &chs, threads, 0x5eed, 1, None);
    let floor: std::collections::HashMap<String, usize> =
        best.iter().map(|c| (c.ch.name(), c.fit.exact)).collect();
    let cands = carrier::sweep(&p, &chs, threads, 0x5eed, cap, Some(&floor));
    println!("sweep: {:.1}s, {} rows", t0.elapsed().as_secs_f64(), cands.len());

    let out = flag(a, "--out");
    let mut tsv = String::from("channel\twrite\trel\tkind\tk\tc\texact\tn\trate\tnull\tbaseline\tkey\n");
    for cd in &cands {
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.12e}\t{:.9}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\n",
            cd.ch.name(), cd.write.name(), cd.rel, cd.kind.name(),
            cd.fit.k, cd.fit.c, cd.fit.exact, cd.fit.n,
            cd.fit.rate(), cd.null, cd.baseline, tag
        ));
    }
    let named = print_table(&cands);
    println!(
        "\n{} channels beat both their permutation floor and a constant. THESE ARE PROPOSALS: \
         a scan fits and cannot test itself, and a tie of hundreds is not a location. Intersect \
         several keys with `fk carrier merge`, then score the survivors on keys that chose none \
         of them with `fk carrier confirm`.",
        named
    );
    if let Some(o) = &out {
        std::fs::write(o, &tsv).map_err(|e| format!("{}: {}", o, e))?;
        println!("wrote {}", o);
    }
    Ok(())
}

/// One line per channel: its best row, and how many offsets tied with it.
///
/// The tie count is the honest part. A channel whose best score is reached at
/// six hundred offsets has not been located, however good the score is; the
/// number that locates it is the intersection with another key, and this column
/// is what says how much work that intersection has to do.
fn print_table(cands: &[Cand]) -> usize {
    println!(
        "\n{:>8} {:>6} {:>9} {:>7} {:>8} {:>8} {:>8} {:>6}  {}",
        "channel", "write", "rel", "kind", "exact%", "null%", "const%", "tied", "verdict"
    );
    let mut seen: Vec<String> = Vec::new();
    let mut named = 0usize;
    for cd in cands.iter() {
        let name = cd.ch.name();
        if seen.contains(&name) {
            continue;
        }
        let tied = cands.iter().filter(|x| x.ch == cd.ch && x.fit.exact == cd.fit.exact).count();
        seen.push(name);
        let ok = cd.fit.rate() > cd.null && cd.fit.rate() > cd.baseline;
        if ok {
            named += 1;
        }
        println!(
            "{:>8} {:>6} {:>9} {:>7} {:>7.2}% {:>7.2}% {:>7.2}% {:>6}  {}",
            cd.ch.name(), cd.write.name(), cd.rel, cd.kind.name(),
            100.0 * cd.fit.rate(), 100.0 * cd.null, 100.0 * cd.baseline, tied,
            if ok { "candidate" } else { "noise" }
        );
    }
    named
}

fn confirm(a: &[String]) -> Result<(), String> {
    let c = ctx(a)?;
    let verbose = a.iter().any(|x| x == "--verbose");
    let dump = flag(a, "--dump").unwrap_or_else(|| format!("/tmp/fkcarrier-{}.bin", std::process::id()));
    let tag = flag(a, "--tag").unwrap_or_else(|| crate::record::name_of(&c.template));
    let table = flag(a, "--table").ok_or("--table FILE is required")?;
    let rows = read_table(&table)?;
    if rows.is_empty() {
        return Err(format!("{} has no rows", table));
    }

    // The window has to be as wide as the SCAN's, not as wide as the table.
    // Narrowing it to the frozen offsets looks thrifty and is not: the car is
    // identified INSIDE the window, and a window a kilobyte wide often contains
    // only a shadow — three of eight keys either scored at their baseline or
    // refused outright when this was trimmed to what the table names. The
    // frozen offsets still bound what is READ; the width only bounds where the
    // car may be found.
    let need_back = num(a, "--back", 1048576).max(rows.iter().map(|r| -r.rel).max().unwrap_or(0) + 8);
    let need_fwd = num(a, "--fwd", 262144).max(rows.iter().map(|r| r.rel).max().unwrap_or(0) + 8);

    let p = gather_wide(&c, need_back, need_fwd, &dump, verbose)?;

    println!(
        "\n{:>8} {:>6} {:>9} {:>7} {:>8} {:>8}  {}",
        "channel", "write", "rel", "kind", "exact%", "const%", "verdict"
    );
    let mut tsv = String::from("channel\twrite\trel\tkind\tk\tc\texact\tn\trate\tnull\tbaseline\tkey\n");
    let (mut held, mut lost, mut no_power) = (0usize, 0usize, 0usize);
    for r in &rows {
        let Some(t) = p.target(r.ch) else { continue };
        let base = t.iter().fold(std::collections::HashMap::<u32, usize>::new(), |mut h, v| {
            *h.entry(*v).or_insert(0) += 1;
            h
        });
        let baseline = base.values().copied().max().unwrap_or(0) as f64 / t.len().max(1) as f64;
        let o = p.pos_off as i64 + r.rel;
        if o < 0 || o as usize + 4 > p.reclen {
            println!("{:>8} {:>6} {:>9} {:>7}   -- offset outside the gathered window", r.ch.name(), r.write.name(), r.rel, r.kind.name());
            continue;
        }
        let n = p.n();
        let byte = |i: usize| p.cols[r.write as usize][o as usize * n + i] as u32;
        let f = match r.kind {
            Kind::Raw => {
                let mut e = 0usize;
                for i in 0..n {
                    if byte(i) == t[i] {
                        e += 1;
                    }
                }
                carrier::Fit { k: 1.0, c: 0.0, exact: e, n }
            }
            Kind::Affine => {
                let v = p.f32col(r.write, o as usize);
                carrier::score(&v, &t, r.ch.modulus(), r.k, r.c)
            }
            Kind::AffineU8 => {
                let v: Vec<f64> = (0..n).map(|i| byte(i) as f64).collect();
                carrier::score(&v, &t, r.ch.modulus(), r.k, r.c)
            }
        };
        // THREE VERDICTS, NOT TWO. A channel that never moves on this key
        // cannot be shown to be right and cannot be shown to be wrong: the
        // constant scores whatever the channel scores. Calling that a failure
        // is as dishonest as calling it a pass, and it is common — the surface
        // bytes are 99 % constant on a run that stays on one surface. What
        // separates it from a real failure is the SHAPE: no power reads AT the
        // baseline, a wrong offset reads far below it.
        let verdict = if baseline >= 0.95 && (f.rate() - baseline).abs() < 1e-9 {
            no_power += 1;
            "no power (the channel never moves on this key)"
        } else if f.rate() > baseline {
            held += 1;
            "holds"
        } else {
            lost += 1;
            "FAILS"
        };
        println!(
            "{:>8} {:>6} {:>9} {:>7} {:>7.2}% {:>7.2}%  {}",
            r.ch.name(), r.write.name(), r.rel, r.kind.name(),
            100.0 * f.rate(), 100.0 * baseline, verdict
        );
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.12e}\t{:.9}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\n",
            r.ch.name(), r.write.name(), r.rel, r.kind.name(), r.k, r.c,
            f.exact, f.n, f.rate(), 0.0, baseline, tag
        ));
    }
    println!(
        "\n{} held, {} failed, {} could not be tested (the channel never moves on this key), \
         on a key that chose none of these offsets.",
        held, lost, no_power
    );
    if let Some(o) = flag(a, "--out") {
        std::fs::write(&o, &tsv).map_err(|e| format!("{}: {}", o, e))?;
        println!("wrote {}", o);
    }
    Ok(())
}

/// One instant of a gathered `CSceneVehicleVisState`, read straight out of the
/// engine-memory columns. `base` is the record offset of state+0, i.e. the
/// car's position offset minus `Loc.translation`'s 0x50.
struct Gathered<'a> {
    p: &'a Paired,
    w: Write,
    base: i64,
    i: usize,
}

impl Gathered<'_> {
    fn byte(&self, off: usize) -> u8 {
        let o = self.base + off as i64;
        if o < 0 || o as usize >= self.p.reclen {
            return 0;
        }
        let n = self.p.n();
        self.p.cols[self.w as usize][o as usize * n + self.i]
    }
    /// Is the whole of `[off, off+len)` inside the gathered window?
    ///
    /// A PARTIAL READ IS WORSE THAN NO READ. `f32` composes four `byte()`
    /// calls, and each one answers 0 outside the record — so a float straddling
    /// the edge comes back with two real bytes and two zeros, which is a finite,
    /// small, entirely plausible number. All-zero at least looks like nothing;
    /// this looks like a measurement. Callers that care ask.
    fn whole(&self, off: usize, len: usize) -> bool {
        let o = self.base + off as i64;
        o >= 0 && (o as usize).saturating_add(len) <= self.p.reclen
    }
}

impl crate::vislayout::State for Gathered<'_> {
    fn covers_state(&self) -> bool {
        self.whole(0, crate::vislayout::STATE_SIZE as usize)
    }
    fn f32(&self, off: usize) -> f32 {
        f32::from_le_bytes([
            self.byte(off),
            self.byte(off + 1),
            self.byte(off + 2),
            self.byte(off + 3),
        ])
    }
    fn u32(&self, off: usize) -> u32 {
        u32::from_le_bytes([
            self.byte(off),
            self.byte(off + 1),
            self.byte(off + 2),
            self.byte(off + 3),
        ])
    }
    fn u8(&self, off: usize) -> u8 {
        self.byte(off)
    }
}

/// Score the DISASSEMBLED layout: rebuild all 116 bytes from the engine state
/// and compare them, byte for byte, with the recording the game wrote.
///
/// This is the confirmation `confirm` cannot give. `confirm` scores a table of
/// per-channel coefficients that a sweep FITTED; this scores a transcription of
/// the game's own writer, in which there is no coefficient to fit and no offset
/// to choose. Every byte is either right or wrong, and the three verdicts of
/// `CARRIER.md` still apply: a byte the run never moves cannot be tested by it.
fn layout(a: &[String]) -> Result<(), String> {
    let c = ctx(a)?;
    let verbose = a.iter().any(|x| x == "--verbose");
    let dump = flag(a, "--dump")
        .unwrap_or_else(|| format!("/tmp/fkcarrier-{}.bin", std::process::id()));
    let tag = flag(a, "--tag").unwrap_or_else(|| crate::record::name_of(&c.template));
    // The state runs from car-0x50 to car-0x50+0x360, so the window only has to
    // reach 80 bytes back and 784 forward -- but the car is IDENTIFIED inside
    // the window, so the width stays the scan's. `confirm` paid for that lesson.
    let p = gather_wide(&c, num(a, "--back", 1048576), num(a, "--fwd", 262144), &dump, verbose)?;
    let n = p.n();
    let base = p.pos_off as i64 - crate::vislayout::POS_IN_STATE;

    println!("\n{} instants, state base at record offset {}", n, base);
    println!(
        "{:>5} {:>7} {:>8} {:>8}  {:<44} {}",
        "byte", "write", "exact%", "const%", "field", "verdict"
    );
    let mut tsv = String::from("byte\twrite\texact\tn\trate\tbaseline\tfield\tencoding\tverdict\tkey\n");
    let (mut held, mut lost, mut no_power, mut skipped) = (0usize, 0usize, 0usize, 0usize);
    // Predict once per write, then score every byte off the same prediction.
    let mut pred: [Vec<[u8; 116]>; 2] = [Vec::with_capacity(n), Vec::with_capacity(n)];
    for w in [Write::Car, Write::Other] {
        for i in 0..n {
            let g = Gathered { p: &p, w, base, i };
            pred[w as usize].push(crate::vislayout::pack(&g));
        }
    }
    for d in crate::vislayout::DOC {
        if crate::vislayout::UNPREDICTED.contains(&d.byte) {
            skipped += 1;
            println!(
                "{:>5} {:>7} {:>8} {:>8}  {:<44} not predicted here ({})",
                d.byte, "-", "-", "-", d.field, d.encoding
            );
            continue;
        }
        let t: Vec<u8> = p.sample.iter().map(|s| s[d.byte]).collect();
        let mut h = std::collections::HashMap::<u8, usize>::new();
        for v in &t {
            *h.entry(*v).or_insert(0) += 1;
        }
        let baseline = h.values().copied().max().unwrap_or(0) as f64 / n.max(1) as f64;
        // The write is not a free parameter to fit either: the table takes the
        // write the CAR was identified on, and the other one is printed only
        // when it disagrees, as evidence about the instant rather than a choice.
        let mut best = (0usize, Write::Car);
        for w in [Write::Car, Write::Other] {
            let e = (0..n).filter(|&i| pred[w as usize][i][d.byte] == t[i]).count();
            if e > best.0 {
                best = (e, w);
            }
        }
        let car_exact = (0..n).filter(|&i| pred[0][i][d.byte] == t[i]).count();
        let rate = car_exact as f64 / n.max(1) as f64;
        // A byte that is right at a SHIFTED instant is a pairing error, not a
        // wrong field: the engine writes the vehicle state more than once per
        // tick and some fields are updated by a later stage than the one the
        // recorder captured. Measuring the shift separates the two, and this
        // project has already been bitten by reading a tick-late field as a
        // refuted law.
        let mut shift = (car_exact, 0i64);
        for s in [-2i64, -1, 1, 2] {
            let e = (0..n)
                .filter(|&i| {
                    let j = i as i64 + s;
                    j >= 0 && (j as usize) < n && pred[0][j as usize][d.byte] == t[i]
                })
                .count();
            if e > shift.0 {
                shift = (e, s);
            }
        }
        let verdict = if baseline >= 0.95 && (rate - baseline).abs() < 1e-9 {
            no_power += 1;
            "no power (the byte never moves on this key)".to_string()
        } else if rate > baseline {
            held += 1;
            "holds".to_string()
        } else {
            lost += 1;
            let modal = |v: &[u8]| {
                let mut m = std::collections::HashMap::<u8, usize>::new();
                for x in v {
                    *m.entry(*x).or_insert(0) += 1;
                }
                m.into_iter().max_by_key(|&(_, c)| c).map(|(v, _)| v).unwrap_or(0)
            };
            let pv: Vec<u8> = (0..n).map(|i| pred[0][i][d.byte]).collect();
            // THE FOURTH VERDICT. A layout read out of the writer can be right
            // about the field and still score nothing, because the SOURCE SLOT
            // is not populated in this binary: the dedicated server runs the
            // simulation, not the presentation, and some of what a client
            // records is written by code the server never executes. That is a
            // different fact from a wrong offset, and the discriminator is
            // whether the prediction moves at all while the recording does.
            let dead = pv.iter().all(|&x| x == pv[0]) && baseline < 0.95;
            format!(
                "{} (other write {:.2}%, modal recorded {} vs predicted {}, best shift {:+} at {:.2}%)",
                if dead {
                    "SOURCE SLOT DEAD in this binary -- the prediction never moves, the recording does"
                } else {
                    "FAILS"
                },
                100.0 * best.0 as f64 / n.max(1) as f64,
                modal(&t),
                modal(&pv),
                shift.1,
                100.0 * shift.0 as f64 / n.max(1) as f64
            )
        };
        println!(
            "{:>5} {:>7} {:>7.2}% {:>7.2}%  {:<44} {}",
            d.byte, "car", 100.0 * rate, 100.0 * baseline, d.field, verdict
        );
        tsv.push_str(&format!(
            "{}\tcar\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\n",
            d.byte, car_exact, n, rate, baseline, d.field, d.encoding, verdict, tag
        ));
    }
    println!(
        "\n{} hold, {} fail, {} could not be tested (the byte never moves on this key), \
         {} not predicted here.",
        held, lost, no_power, skipped
    );

    // The packed bytes, field by field. A byte can pass on the strength of one
    // of the six quantities in it, and for the reactor that is exactly the
    // trap: byte 89 is 100 % on a run with no reactor because IsGroundContact
    // is 100 %.
    println!(
        "\n{:<30} {:>8} {:>8} {:>7}  {}",
        "packed field", "exact%", "const%", "values", "verdict"
    );
    for f in crate::vislayout::BITFIELDS {
        let t: Vec<u32> = p.sample.iter().map(|s| f.read(s)).collect();
        let mut h = std::collections::HashMap::<u32, usize>::new();
        for v in &t {
            *h.entry(*v).or_insert(0) += 1;
        }
        let baseline = h.values().copied().max().unwrap_or(0) as f64 / n.max(1) as f64;
        let e = (0..n).filter(|&i| f.read(&pred[0][i]) == t[i]).count();
        let rate = e as f64 / n.max(1) as f64;
        let verdict = if h.len() <= 1 {
            "the field never moves on this key -- untested"
        } else if rate > baseline {
            "holds"
        } else {
            "FAILS"
        };
        println!(
            "{:<30} {:>7.2}% {:>7.2}% {:>7}  {}",
            f.name,
            100.0 * rate,
            100.0 * baseline,
            h.len(),
            verdict
        );
        tsv.push_str(&format!(
            "{}\tcar\t{}\t{}\t{:.4}\t{:.4}\t{}\tbitfield\t{}\t{}\n",
            f.byte, e, n, rate, baseline, f.name, verdict, tag
        ));
    }
    if let Some(o) = flag(a, "--out") {
        std::fs::write(&o, &tsv).map_err(|e| format!("{}: {}", o, e))?;
        println!("wrote {}", o);
    }
    Ok(())
}

/// Collapse several `fk carrier layout` runs into one row per sample byte.
///
/// The column that decides is the WORST key that had power, exactly as
/// `CARRIER.md` reports its table: a byte that is 100 % on six keys and 3 % on
/// the seventh has not been named. The count of keys with power is printed
/// beside it, because a byte that no key exercises is neither confirmed nor
/// contradicted and saying so is not optional.
fn rollup(a: &[String]) -> Result<(), String> {
    let list = flag(a, "--tables").ok_or("--tables A.tsv,B.tsv,... is required")?;
    #[derive(Default, Clone)]
    struct Agg {
        field: String,
        power: Vec<(String, f64)>,
        no_power: usize,
        worst: f64,
        dead: usize,
    }
    let mut rows: std::collections::BTreeMap<String, Agg> = std::collections::BTreeMap::new();
    let mut keys: Vec<String> = Vec::new();
    for path in list.split(',') {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 10 {
                continue;
            }
            let byte: usize = f[0].parse().map_err(|_| format!("bad byte {:?}", f[0]))?;
            let rate: f64 = f[4].parse().unwrap_or(0.0);
            let baseline: f64 = f[5].parse().unwrap_or(0.0);
            let key = f[9].to_string();
            if !keys.contains(&key) {
                keys.push(key.clone());
            }
            // Bitfield rows and byte rows share a byte number, so the row
            // identity has to carry both.
            let id = if f[7] == "bitfield" {
                format!("f{:03}:{}", byte, f[6])
            } else {
                format!("b{:03}", byte)
            };
            let e = rows.entry(id).or_default();
            e.field = f[6].to_string();
            if f[8].starts_with("SOURCE SLOT DEAD") {
                e.dead += 1;
            }
            // "power" is the recording moving, not the prediction being right.
            if baseline >= 0.95 && (rate - baseline).abs() < 1e-9 {
                e.no_power += 1;
            } else {
                e.power.push((key, rate));
            }
        }
    }
    println!("{} keys: {}", keys.len(), keys.join(", "));
    println!(
        "\n{:>10} {:>7} {:>9} {:>8}  {:<44} {}",
        "row", "power", "worst%", "no-power", "field", "worst key"
    );
    for (id, e) in &rows {
        let worst = e
            .power
            .iter()
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .cloned();
        match worst {
            Some((k, r)) => println!(
                "{:>10} {:>7} {:>8.2}% {:>8}  {:<44} {}{}",
                id,
                e.power.len(),
                100.0 * r,
                e.no_power,
                e.field,
                k,
                if e.dead > 0 {
                    format!("   [source slot dead on {} key(s)]", e.dead)
                } else {
                    String::new()
                }
            ),
            None => println!(
                "{:>10} {:>7} {:>9} {:>8}  {:<44} -",
                id, 0, "-", e.no_power, e.field
            ),
        }
    }
    Ok(())
}

/// Find the offset in the gathered record whose f32 triple IS the recording's
/// own position, and say how far the next best one was.
///
/// The runner-up matters as much as the winner. A locate that reports "0.4 mm"
/// without it cannot tell the car from its own shadow, and this engine keeps
/// several: copies of the vehicle state at −2648, −3512, −4176 and −4792 from
/// the live one, each a valid state of the same car half a millimetre or one
/// tick away. The runner-up distance is what says whether the identification
/// had anything to choose between.
fn find_car(
    recs: &[(u32, Vec<u8>, Vec<u8>)],
    idx: &[usize],
    sample: &[Vec<u8>],
    reclen: usize,
) -> Result<(usize, Write, f64, f64), String> {
    let n = idx.len();
    let want: Vec<[f64; 3]> = sample
        .iter()
        .map(|s| gbx::record::read_transform_pub(s, 47).0)
        .collect();
    // A coarse pass on ONE instant first: 328 000 offsets times 455 instants is
    // 150 M distance evaluations, and all but a handful of offsets are ruled
    // out by a single one.
    let probe = n / 2;
    let hi = reclen.saturating_sub(12);
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8);
    let chunk = hi.div_ceil(threads);
    let mut scored: Vec<(f64, usize, Write)> = Vec::new();
    // BOTH WRITES OF THE TICK. The engine writes the vehicle state more than
    // once per tick and the recorder captured one of them; which one is a
    // measurable question, and getting it wrong is not a small error — it is
    // the one-tick shadow that a previous pass at this anchored its whole wheel
    // block inside, leaving every field it read with a latent one-tick offset.
    // The number that tells them apart is the median distance: the write the
    // recorder took is bit-identical or a micron away, the other is half a
    // millimetre.
    for wr in [Write::Car, Write::Other] {
        let shortlist: Vec<usize> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|w| {
                    let want = &want;
                    s.spawn(move || {
                        let (lo, end) = (w * chunk, ((w + 1) * chunk).min(hi));
                        let b = match wr {
                            Write::Car => &recs[idx[probe]].1,
                            Write::Other => &recs[idx[probe]].2,
                        };
                        let mut out = Vec::new();
                        for o in (lo..end).step_by(4) {
                            let mut d = 0.0;
                            for k in 0..3 {
                                let e = f32::from_le_bytes(
                                    b[o + k * 4..o + k * 4 + 4].try_into().unwrap(),
                                ) as f64;
                                d += (e - want[probe][k]).powi(2);
                            }
                            // 1 m at one instant: wide enough for a stale copy
                            // of the car, narrow enough to drop everything else.
                            if d < 1.0 {
                                out.push(o);
                            }
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        for o in shortlist {
            let mut e: Vec<f64> = Vec::with_capacity(n);
            for i in 0..n {
                let b = match wr {
                    Write::Car => &recs[idx[i]].1,
                    Write::Other => &recs[idx[i]].2,
                };
                let mut d = 0.0;
                for k in 0..3 {
                    let v =
                        f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap()) as f64;
                    d += (v - want[i][k]).powi(2);
                }
                e.push(d.sqrt());
            }
            e.sort_by(|a, b| a.total_cmp(b));
            scored.push((e[e.len() / 2], o, wr));
        }
    }
    if scored.is_empty() {
        return Err("no float triple in the gathered window is within 1 m of the recording's \
                    own position at the midpoint of the run"
            .into());
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("{} copies of the car in the window:", scored.len());
    for (d, o, w) in scored.iter().take(6) {
        println!("    +{:<9} {:>5}  {:.6} m", o, w.name(), d);
    }
    // THE CAR, NOT ONE OF ITS SHADOWS. This engine keeps several copies of the
    // vehicle state — a lag-one shadow, a back buffer, a bare position copy —
    // and every one of them is a valid state of the SAME car, so none of them
    // fails a structural test. They are not the same thing to measure from: a
    // field at a fixed offset from the car reads as being at that offset plus
    // 2648, or 3512, or 864, when the anchor is a shadow, and a table built
    // from several keys that landed on different members intersects to nothing.
    // That is exactly what happened here on the first pass: `u16@8` fitted at
    // 99.3-100 % on six keys with the same coefficient to seven figures, at six
    // apparently unrelated offsets.
    //
    // ONE BAR, and it is coarse on purpose: a millimetre, which every copy of
    // this car clears and no other object comes near. Distance does NOT decide
    // which copy is the car — the wheel block does, below — and it took two
    // wrong versions of this to learn why.
    //
    // The tempting rule is "the car reads at a micron and a shadow at half a
    // millimetre, so refuse anything above 0.0002". That is true of a GAME
    // RECORDING and false of anything else, because the distance is measured
    // against the file's own recorded positions. On a file whose transform this
    // project regenerated, those positions came from whichever copy the
    // regeneration read: the copy it read then matches at 0.000000 m and the
    // real vehicle struct is 0.000488 m away — the shadow number, wearing it
    // for the opposite reason. A distance bar refuses the car and accepts the
    // copy with nothing in it.
    const CANDIDATE: f64 = 1e-3;
    // AND THE ONE WITH A CAR AROUND IT.
    //
    // Several copies are bit-identical on position: this engine keeps a bare
    // position copy as well as the vehicle state, and the bare one is just as
    // close to the recording's own path with nothing but dead memory around it.
    // Landing on it is not a small error and it is not a loud one — every
    // offset in the table points at zeros, and a file written from it passes
    // the whole `ghost verify` gate, because none of these bytes affects the
    // simulation. Caught here doing exactly that: a two-pass regeneration wrote
    // ZEROED wheel rotations and gear into a file that then re-simulated to its
    // declared 22.730 and reported kappa 1.000.
    //
    // The discriminator is the WHEEL BLOCK, and it is physics rather than a
    // statistic: at car+92, +136, +180 and +224 sit four accumulating wheel
    // angles, so their increments must track the distance the car actually
    // travelled between the same two instants. A bare position copy has zeros
    // there. It needs no answer key beyond the positions already in hand, and
    // it is the reference-free signature of the vehicle struct that this
    // project has wanted since the wheel block was first found inside a shadow.
    // DERIVED. See `vislayout::wheel_rot_rel` -- one statement of these four.
    let wheels: [usize; 4] = crate::vislayout::wheel_rot_rel().map(|r| r as usize);
    let wheelness = |o: usize, w: Write| -> f64 {
        // DO THE FOUR WHEEL SLOTS HOLD ANYTHING AT ALL.
        //
        // That is the whole test, and it is deliberately not more than that.
        // Two cleverer versions were tried and both are wrong, for the same
        // reason: the slot is an ANGLE THAT WRAPS, and a wheel turns twice
        // between two 50 ms samples at racing speed.
        //
        //   * "each wheel's increments track the distance travelled" reads
        //     |corr| 0.016 on the real car, because the wraps dominate;
        //   * "the four wheels move together" reads 0.4967, because they wrap
        //     at slightly different instants when they are slipping.
        //
        // What actually separates the vehicle struct from a bare position copy
        // — the failure this guards, where a file came out with ZEROED wheel
        // rotations and gear and still passed the whole `ghost verify` gate,
        // because none of these bytes affects the simulation — is that the bare
        // copy has FOUR CONSTANTS there and the real one has four live floats.
        // 4 against 0, with nothing in between to tune.
        wheels
            .iter()
            .filter(|wo| {
                let q = o + *wo;
                if q + 4 > reclen {
                    return false;
                }
                let get = |r: usize| -> f64 {
                    let b = match w {
                        Write::Car => &recs[idx[r]].1,
                        Write::Other => &recs[idx[r]].2,
                    };
                    f32::from_le_bytes(b[q..q + 4].try_into().unwrap()) as f64
                };
                let first = get(0);
                first.is_finite() && (1..n).any(|i| get(i) != first && get(i).is_finite())
            })
            .count() as f64
    };
    {
        // EVERY copy that is the car, not every copy within a ratio of the
        // best. A relative window collapses exactly when it matters most: on a
        // file whose transform has already been regenerated the best copy
        // matches at 0.000000 m, `best * 1.5` is then zero, and the full
        // vehicle struct sitting at 0.000001 m is excluded from its own tie —
        // which is how a two-pass regeneration ended up refusing every anchor
        // in a 5 MB window while the car was in it three times over.
        let tied: Vec<(f64, usize, Write)> =
            scored.iter().copied().filter(|s| s.0 < CANDIDATE).collect();
        // NOTHING WITHIN A MILLIMETRE IS NOT A NEAR MISS, IT IS A FAILED
        // LOCATE. Falling through here scores the whole table against whatever
        // the closest object happened to be: on 286279 that was 142.3 m away
        // and the run reported "0 held, 18 FAILED", which reads as eighteen
        // refuted encodings and is one refused locate.
        if tied.is_empty() {
            return Err(format!(
                "the closest float triple in the window is {:.3} m from the recording's own \
                 path -- the car is not in this window, so nothing here can be scored",
                scored[0].0
            ));
        }
        {
            let mut ranked: Vec<(f64, (f64, usize, Write))> =
                tied.into_iter().map(|s| (wheelness(s.1, s.2), s)).collect();
            // live wheels first, then closeness -- in that order, because a
            // copy without the fields is not a worse answer, it is a different
            // object.
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1 .0.total_cmp(&b.1 .0)));
            if ranked.len() > 1 {
                println!("  tie on position; live wheel slots per copy:");
                for (l, (_, o, w)) in ranked.iter().take(4) {
                    println!("    +{:<9} {:>5}  {} of 4", o, w.name(), *l as u32);
                }
            }
            // 0.99: the wheel angles of the real struct track distance at
            // 0.9999 and a copy without a wheel block scores 0 exactly, so
            // there is nothing in between to tune.
            if ranked[0].0 < 0.99 {
                return Err(format!(
                    "the copy at +{} matches the recording's path to {:.6} m but its wheel                      angles do not track the distance travelled (|corr| {:.4}) -- this is a                      bare position copy, not the vehicle state, and every offset measured                      from it would read dead memory",
                    ranked[0].1 .1, ranked[0].1 .0, ranked[0].0
                ));
            }
            scored[0] = ranked[0].1;
        }
    }
    Ok((
        scored[0].1,
        scored[0].2,
        scored[0].0,
        scored.get(1).map(|s| s.0).unwrap_or(f64::INFINITY),
    ))
}

// ------------------------------------------------------------------- merge

/// Intersect several scans into one frozen table.
///
/// A scan is one key's opinion. This is where the rule "fit from three keys or
/// do not fit" is arithmetic rather than advice: a row survives only if the
/// SAME record offset, kind and write won on at least `--min-keys` independent
/// recordings, each beating its own permutation floor and its own constant.
/// The four per-wheel "dampen" entries in the previous attempt at this table
/// were exactly the rows that would not have survived it — they fitted one
/// recording by coincidence and died when a second was asked.
///
/// The coefficients are then the MEDIAN over the keys that agreed, not one
/// key's. A `k` that is a property of the game (a wheel radius, an rpm scale)
/// is the same number on every key and the median changes nothing; a `k` that
/// is really a property of one run shows up as a spread, which the table
/// prints.
fn merge(a: &[String]) -> Result<(), String> {
    let min_keys = num(a, "--min-keys", 3) as usize;
    let tables: Vec<String> = flag(a, "--tables")
        .ok_or("--tables A.tsv,B.tsv,... is required")?
        .split(',')
        .map(|s| s.to_string())
        .collect();

    #[derive(Default)]
    struct Agg {
        keys: Vec<String>,
        k: Vec<f64>,
        c: Vec<f64>,
        rate: Vec<f64>,
    }
    let mut agg: std::collections::BTreeMap<(String, String, i64, String), Agg> =
        Default::default();
    let mut all_keys: Vec<String> = Vec::new();
    for t in &tables {
        let s = std::fs::read_to_string(t).map_err(|e| format!("{}: {}", t, e))?;
        for l in s.lines().skip(1) {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() < 12 {
                continue;
            }
            let (rate, null, base): (f64, f64, f64) = (
                f[8].parse().unwrap_or(0.0),
                f[9].parse().unwrap_or(1.0),
                f[10].parse().unwrap_or(1.0),
            );
            let key = f[11].to_string();
            if !all_keys.contains(&key) {
                all_keys.push(key.clone());
            }
            // The same bar the scan printed: a row must beat the sweep's own
            // permutation floor AND a constant. Anything else is not evidence
            // that a second key should be asked to corroborate.
            if !(rate > null && rate > base) {
                continue;
            }
            let e = agg
                .entry((f[0].into(), f[1].into(), f[2].parse().unwrap_or(0), f[3].into()))
                .or_default();
            if e.keys.contains(&key) {
                continue;
            }
            e.keys.push(key);
            e.k.push(f[4].parse().unwrap_or(0.0));
            e.c.push(f[5].parse().unwrap_or(0.0));
            e.rate.push(rate);
        }
    }
    let med = |v: &[f64]| -> f64 {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    };
    let spread = |v: &[f64]| -> f64 {
        let m = med(v);
        if m == 0.0 {
            return 0.0;
        }
        v.iter().map(|x| ((x - m) / m).abs()).fold(0.0, f64::max)
    };

    let mut rows: Vec<(usize, String, f64)> = Vec::new();
    let mut tsv = String::from("channel\twrite\trel\tkind\tk\tc\texact\tn\trate\tnull\tbaseline\tkey\n");
    println!(
        "{:>8} {:>6} {:>9} {:>7} {:>5} {:>8} {:>10} {:>10}  {}",
        "channel", "write", "rel", "kind", "keys", "min rate", "k spread", "c spread", "keys that agreed"
    );
    for ((ch, wr, rel, kind), e) in &agg {
        if e.keys.len() < min_keys {
            continue;
        }
        let line = format!(
            "{:>8} {:>6} {:>9} {:>7} {:>5} {:>7.2}% {:>9.2e} {:>9.2e}  {}",
            ch,
            wr,
            rel,
            kind,
            e.keys.len(),
            100.0 * e.rate.iter().copied().fold(f64::INFINITY, f64::min),
            spread(&e.k),
            spread(&e.c),
            e.keys.join(",")
        );
        rows.push((e.keys.len(), line, e.rate.iter().copied().fold(f64::INFINITY, f64::min)));
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.12e}\t{:.9}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\n",
            ch, wr, rel, kind, med(&e.k), med(&e.c),
            0, 0,
            e.rate.iter().copied().fold(f64::INFINITY, f64::min), 0.0, 0.0,
            e.keys.join("+")
        ));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(b.2.total_cmp(&a.2)));
    for (_, l, _) in &rows {
        println!("{}", l);
    }
    println!(
        "\n{} rows on >= {} of {} keys.",
        rows.len(),
        min_keys,
        all_keys.len()
    );
    if let Some(o) = flag(a, "--out") {
        std::fs::write(&o, &tsv).map_err(|e| format!("{}: {}", o, e))?;
        println!("wrote {}", o);
    }
    Ok(())
}

// ------------------------------------------------------------------- write

/// Write the named bytes into a ghost, from engine memory.
///
/// This is the half of the exercise that makes the other half matter. `ghost
/// regen` names 91 inherited bytes every time it writes a file; every row of a
/// frozen table is a byte it no longer has to.
///
/// THE CONTROL IS LEAVE-ONE-OUT. Run this on an answer key with a table frozen
/// from OTHER keys, and compare what it wrote with what the game recorded for
/// the same run. The offsets and the coefficients then had no input from this
/// recording at all, and the only thing the recording contributes is which copy
/// of the car to read — which is a position match to a micron, not a fit.
/// `fk carrier write --layout` — write EVERY byte the writer's own transcription
/// predicts, instead of the hand-fitted table's rows.
///
/// # Why this replaces the table
///
/// The table is 23 rows, each one an offset and an affine coefficient somebody
/// fitted against a recording, one channel at a time, over weeks. Every new
/// channel cost another sweep, another answer key, and another argument about
/// whether a 92 % agreement was a location or a coincidence — and five of the
/// rows turned out to be the wrong ENCODING (`b22`'s constant is 255/2π and not
/// the wheel constant; `b31` is a 3-bit enum plus a flag and not a byte copy;
/// the ground materials substitute 13), each scoring 100.00 % once read the way
/// the writer writes it.
///
/// `vislayout::pack` is the writer, transcribed from the archiver at
/// `0x9cfed0` and the class descriptor at `0x9d2ea0`. It takes the state and
/// returns all 116 bytes. So there is nothing left to fit, nothing to tie, and
/// no per-byte dance: the packed bit-fields (the five reactor members across
/// bytes 89, 90, 91 and 76, which NO per-byte affine fit could ever represent)
/// come out with everything else.
///
/// # What it still refuses to do
///
/// Two classes of byte are left as the container's, and both are printed:
///
/// * **`UNPREDICTED`** — the orientation words (59..64), which need the
///   matrix-to-quaternion step, and the countdown (108..111), which needs the
///   archiver's caller-supplied timestamp.
/// * **DEAD IN THIS BINARY** — a slot the dedicated server never populates
///   while the container's own value moves. Byte 34, bytes 19/20 and the four
///   dirt slots are identically zero in the server, so writing them would
///   replace a real value with a confident zero. That is the failure mode this
///   whole command exists to avoid, and it is checked per byte rather than
///   assumed.
fn write_layout(a: &[String]) -> Result<(), String> {
    let c = ctx(a)?;
    let verbose = a.iter().any(|x| x == "--verbose");
    let dump = flag(a, "--dump")
        .unwrap_or_else(|| format!("/tmp/fkcarrier-{}.bin", std::process::id()));
    let outp = flag(a, "--out").ok_or("--out FILE is required")?;
    let p = gather_wide(&c, num(a, "--back", 1048576), num(a, "--fwd", 262144), &dump, verbose)?;
    let n = p.n();
    let base = p.pos_off as i64 - crate::vislayout::POS_IN_STATE;

    // Predict every instant once, on the write the car was identified on.
    let pred: Vec<[u8; 116]> = (0..n)
        .map(|i| {
            let g = Gathered { p: &p, w: Write::Car, base, i };
            crate::vislayout::pack(&g)
        })
        .collect();

    // Decide, per byte, whether we may write it. Three verdicts, all printed.
    let mut writable: Vec<usize> = Vec::new();
    let mut unpredicted: Vec<usize> = Vec::new();
    let mut dead: Vec<usize> = Vec::new();
    for b in 0..116 {
        if crate::vislayout::UNPREDICTED.contains(&b) {
            unpredicted.push(b);
            continue;
        }
        let ours_live = (1..n).any(|i| pred[i][b] != pred[0][b]);
        let theirs_live = p.sample.iter().any(|s| s[b] != p.sample[0][b]);
        if theirs_live && !ours_live {
            dead.push(b);
            continue;
        }
        writable.push(b);
    }
    let exact = |b: usize| {
        (0..n).filter(|&i| pred[i][b] == p.sample[i][b]).count() as f64 / n.max(1) as f64
    };
    println!(
        "\n{} instants. {} bytes writable, {} not predicted {:?}, {} dead in this binary {:?}",
        n,
        writable.len(),
        unpredicted.len(),
        unpredicted,
        dead.len(),
        dead
    );
    println!(
        "  agreement with the container, over the writable bytes: {} at 100 %, {} below",
        writable.iter().filter(|b| exact(**b) >= 1.0).count(),
        writable.iter().filter(|b| exact(**b) < 1.0).count()
    );

    let by_ms: std::collections::HashMap<i64, usize> =
        p.ms.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    let (mut wrote, mut skipped) = (0usize, 0usize);
    gbx::recwrite::rewrite_ghost(&c.template, &outp, |rd| {
        let ent = rd
            .ents
            .iter_mut()
            .filter(|e| e.sample_size >= 100 && !e.times.is_empty())
            .max_by_key(|e| e.times.len())
            .ok_or("no vehicle entity")?;
        let ss = ent.sample_size;
        for (si, t) in ent.times.clone().iter().enumerate() {
            let Some(i) = by_ms.get(&(*t as i64)).copied() else {
                skipped += 1;
                continue;
            };
            let s = &mut ent.raw[si * ss..(si + 1) * ss];
            for b in &writable {
                if *b < ss {
                    s[*b] = pred[i][*b];
                }
            }
            wrote += 1;
        }
        Ok(())
    })?;
    println!(
        "wrote {} ({} samples rewritten, {} left alone -- no engine instant)",
        outp, wrote, skipped
    );
    Ok(())
}

fn write(a: &[String]) -> Result<(), String> {
    if a.iter().any(|x| x == "--layout") {
        return write_layout(a);
    }
    let c = ctx(a)?;
    let verbose = a.iter().any(|x| x == "--verbose");
    let dump = flag(a, "--dump").unwrap_or_else(|| format!("/tmp/fkcarrier-{}.bin", std::process::id()));
    let outp = flag(a, "--out").ok_or("--out FILE is required")?;
    let table = flag(a, "--table").ok_or("--table FILE is required")?;
    let rows = read_table(&table)?;
    if rows.is_empty() {
        return Err(format!("{} has no rows", table));
    }
    let need_back = num(a, "--back", 1048576).max(rows.iter().map(|r| -r.rel).max().unwrap_or(0) + 8);
    let need_fwd = num(a, "--fwd", 262144).max(rows.iter().map(|r| r.rel).max().unwrap_or(0) + 8);
    let p = gather_wide(&c, need_back, need_fwd, &dump, verbose)?;

    // What each row computes, per instant, as the value the sample should hold.
    let n = p.n();
    let mut plan: Vec<(&Row, Vec<u32>)> = Vec::new();
    for r in &rows {
        let o = p.pos_off as i64 + r.rel;
        if o < 0 || o as usize + 4 > p.reclen {
            return Err(format!(
                "{} wants record offset {} and the gather is {} bytes wide",
                r.ch.name(), o, p.reclen
            ));
        }
        let m = r.ch.modulus();
        let vals: Vec<u32> = match r.kind {
            Kind::Raw => (0..n)
                .map(|i| p.cols[r.write as usize][o as usize * n + i] as u32)
                .collect(),
            Kind::Affine => p
                .f32col(r.write, o as usize)
                .into_iter()
                .map(|v| ((r.k * v + r.c).floor() as i64).rem_euclid(m as i64) as u32)
                .collect(),
            Kind::AffineU8 => (0..n)
                .map(|i| {
                    let v = p.cols[r.write as usize][o as usize * n + i] as f64;
                    ((r.k * v + r.c).floor() as i64).rem_euclid(m as i64) as u32
                })
                .collect(),
        };
        plan.push((r, vals));
    }

    // AGREEMENT WITH WHAT THE CONTAINER ALREADY HELD, per channel, before
    // anything is written. On an answer key this is the verdict. On a
    // transplanted ghost the container's bytes are a stranger's and
    // disagreement is the POINT. And on a NEUTRALISED container they are zeros,
    // so the column means nothing at all and says so rather than printing a
    // number somebody will read as a score.
    println!(
        "\n{:>8} {:>9} {:>9} {:>8}  {:>7}  agreement with what the container already held",
        "channel", "rel", "kind", "exact%", "live"
    );
    for (r, vals) in &plan {
        let Some(t) = p.target(r.ch) else { continue };
        let e = vals.iter().zip(t.iter()).filter(|(a, b)| a == b).count();
        let container_live = t.iter().any(|v| *v != t[0]);
        let ours_live = vals.iter().any(|v| *v != vals[0]);
        println!(
            "{:>8} {:>9} {:>9} {:>7.2}%  {:>7}  {}",
            r.ch.name(), r.rel, r.kind.name(),
            100.0 * e as f64 / n.max(1) as f64,
            if ours_live { "yes" } else { "NO" },
            if !container_live {
                "-- the container is constant here (neutralised, or a channel that never moves)"
            } else {
                ""
            }
        );
    }
    // A channel that comes out CONSTANT when the container's own is not is the
    // signature of reading dead memory, and it is the one failure of this
    // command that produces a plausible file: none of these bytes affects the
    // simulation, so the plain oracle and the whole `ghost verify` gate pass on
    // a file full of zeroed wheels.
    let dead: Vec<String> = plan
        .iter()
        .filter(|(r, vals)| {
            !vals.iter().any(|v| *v != vals[0])
                && p.target(r.ch).map_or(false, |t| t.iter().any(|v| *v != t[0]))
        })
        .map(|(r, _)| r.ch.name())
        .collect();
    if !dead.is_empty() {
        return Err(format!(
            "{:?} came out CONSTANT from the engine while the container's own values move --              the gathered slots are dead memory and nothing downstream would catch it",
            dead
        ));
    }

    // Now write. Only in-race instants the engine reached are touched; a sample
    // the clean run never saw keeps whatever it had, and the count is printed,
    // because a file that is quietly part-carrier is what this exercise exists
    // to end.
    let by_ms: std::collections::HashMap<i64, usize> =
        p.ms.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    let mut wrote = 0usize;
    let mut skipped = 0usize;
    let mut touched: Vec<usize> = Vec::new();
    for (r, _) in &plan {
        match r.ch {
            Channel::Byte(b) => touched.push(b),
            Channel::U16(b) => {
                touched.push(b);
                touched.push(b + 1);
            }
        }
    }
    touched.sort_unstable();
    touched.dedup();
    gbx::recwrite::rewrite_ghost(&c.template, &outp, |rd| {
        let ent = rd
            .ents
            .iter_mut()
            .filter(|e| e.sample_size >= 100 && !e.times.is_empty())
            .max_by_key(|e| e.times.len())
            .ok_or("no vehicle entity")?;
        let ss = ent.sample_size;
        for (si, t) in ent.times.clone().iter().enumerate() {
            let Some(i) = by_ms.get(&(*t as i64)).copied() else {
                skipped += 1;
                continue;
            };
            let s = &mut ent.raw[si * ss..(si + 1) * ss];
            for (r, vals) in &plan {
                let v = vals[i];
                match r.ch {
                    Channel::Byte(b) if b < ss => s[b] = v as u8,
                    Channel::U16(b) if b + 1 < ss => {
                        s[b] = v as u8;
                        s[b + 1] = (v >> 8) as u8;
                    }
                    _ => {}
                }
            }
            wrote += 1;
        }
        Ok(())
    })?;
    println!(
        "\nwrote {} ({} samples rewritten, {} left alone -- no engine instant), bytes {:?}",
        outp, wrote, skipped, touched
    );
    Ok(())
}

// ------------------------------------------------- the fields, for `fk regen`

/// Read the carrier fields for one run, against a trajectory that is ALREADY
/// KNOWN — the positions `fk regen`'s own clean run just measured, keyed by
/// race time.
///
/// # Why this is a second gather rather than a second command
///
/// The clean run gathers 452 bytes at every tick, which is what the transform
/// needs and is nowhere near wide enough for the fields: on this engine the
/// vehicle struct sits further than 8 KB from where the clock-first locate
/// lands, and every copy inside that window is a BARE POSITION COPY — the
/// car's own position with dead memory around it. Widening the clean run to
/// find the struct is not an option either: 1.25 MB at a 10 ms grid is
/// gigabytes.
///
/// So the fields come from a second gather, wide and on the record's own 50 ms
/// grid, inside the same command. The transform's gather is untouched, so a
/// regenerated trajectory is bit-identical to one from a run with no `--carrier`
/// at all.
///
/// # And this is what removes the ordering rule
///
/// A gather has to identify WHICH copy of the car it is looking at, and the
/// obvious reference — the file's own recorded positions — is the DONOR's on
/// exactly the files worth regenerating. Here it is not needed: the clean run
/// has just measured this run's own positions, per millisecond, from the engine.
/// Matching against those identifies the car with no recording involved, so
/// `--carrier` works on a transplanted container on the first pass and there is
/// no "regenerate the transform first" for anyone to get wrong.
/// What one instant of the vehicle struct holds: the transform, and the fields.
///
/// The transform is here because it MUST come from the same object as the
/// fields. Two copies of the car sit half a millimetre apart -- one tick's
/// travel -- and pairing a position from one with a wheel angle from the other
/// is a pure time shift, invisible in a solo clip and fatal in any
/// frame-synchronous comparison.
pub struct Instant {
    pub pos: [f32; 3],
    /// (x, y, z, w), whatever form the engine held it in.
    pub quat: [f64; 4],
    pub vel: [f64; 3],
    pub fields: Vec<(Channel, u32)>,
}

pub fn gather_fields(
    c: &Ctx,
    anchors: &record::Anchors,
    rows: &[Row],
    truth: &std::collections::HashMap<i64, [f64; 3]>,
    // `truth_q`: the recording's OWN orientation per instant, when there is
    // one. An answer key, used only to REPORT how each orientation candidate
    // scores -- it never chooses, because a chooser that reads the file it is
    // regenerating picks the donor's car on exactly the files worth
    // regenerating.
    truth_q: &std::collections::HashMap<i64, [f64; 4]>,
    period: i64,
    phase_ms: i64,
    dump: &str,
    back: i64,
    fwd: i64,
    // THE POINTER, WHEN THERE IS ONE. `Some(f)` resolves the address of the
    // vehicle state in the live halted engine (see `fk::ptr`), and the gather
    // is then 864 bytes AT THE CAR instead of 1.25 MB around an anchor with the
    // car somewhere inside it. Measured on 191465: 6 MB and 2.3 s against
    // 1.36 GB and eleven minutes, for the same bytes.
    //
    // It changes the WINDOW and nothing else. Every test below runs unchanged
    // -- the copy is still identified by matching the clean run's own measured
    // path, the four wheel slots must still be live, and the distance bar is
    // still 1e-3 m -- so a chain that has gone stale (a new binary, a changed
    // build) FAILS here and the caller falls back to the blind window.
    car: Option<&dyn Fn(i32, u64) -> Result<(u64, Vec<(i64, u32)>), String>>,
    verbose: bool,
) -> Result<std::collections::HashMap<i64, Instant>, String> {
    // The layout sentinel is `rel == i64::MIN`, so `-r.rel` OVERFLOWS and the
    // max of the reaches is meaningless. What the packer actually needs is the
    // whole vehicle state, which the class descriptor puts at 864 bytes with
    // `Loc.translation` at 0x50 -- so 0x50 behind the car and 0x310 ahead of it,
    // and not a byte more. Stated as a range rather than derived from rows,
    // because there are no rows in that mode.
    let layout_mode = rows.len() == 1 && rows[0].rel == i64::MIN;
    let (reach, behind) = if layout_mode {
        (0x310 + 8, 0x50 + 8)
    } else {
        (
            rows.iter().map(|r| r.rel).max().unwrap_or(0).max(0) + 8,
            rows.iter().map(|r| -r.rel).max().unwrap_or(0).max(0) + 8,
        )
    };
    // TWO PHASES, because the wide window is needed at ONE INSTANT and the
    // narrow one at all of them.
    //
    // What this used to do: gather 1.25 MB of memory at every one of ~260
    // instants, write all 1.36 GB of it to a file in /tmp, and read it back.
    // Twenty-four regeneration attempts run in parallel, so that is 32 GB
    // through the disk to extract about 200 KB of car state. It was I/O-bound
    // by three orders of magnitude, and on a busy box it took minutes.
    //
    // The width is only there to FIND the copy of the car that has the fields,
    // and the code below does that from a single probe instant (`probe = n/2`)
    // -- every other use is four wheel offsets and the state itself. So:
    //
    //   phase A   wide, but COARSE: one instant in `PHASE_A_STRIDE`, enough to
    //             find the copy and to take a median over. ~16 instants.
    //   phase B   NARROW -- the vehicle state is 864 bytes, stated by the class
    //             descriptor -- at every instant, positioned on what phase A
    //             found.
    //
    // Same answers, ~1/500th of the bytes. The dump also goes to a RAM-backed
    // tmpfs when one is available, so even the wide phase never reaches a disk.
    // The stride is chosen to leave ENOUGH INSTANTS, not to be as coarse as
    // possible. Phase A must still pair with the clean run (the guard below
    // wants 20 shared instants, and a median over a handful is not a median),
    // so it targets ~48 and never goes below that. A fixed stride of 16 left
    // 17 and tripped the guard -- correctly: "they are not the same run" is
    // exactly what too few shared instants cannot be distinguished from.
    const PHASE_A_TARGET: usize = 48;
    let mut extra: record::ExtraSegs = Vec::new();
    if car.is_some() {
        // The production window already covers car-192..car+256; the rest of
        // the struct is gathered immediately after it, so `po` stays inside a
        // contiguous run of bytes and every row's `car + rel` is in the record.
        //
        // SIZED FOR A COPY THAT IS NOT THE ONE THE POINTER NAMED. Measured on
        // untitled 01: the chain resolved a state whose translation the copy
        // rule then found 124 bytes further on, 0.000493 m from the run's own
        // path -- the half-millimetre shadow of CARRIER.md §6, a second object
        // rather than a second reading of one. The window has to hold the whole
        // 864-byte state of whichever of them wins, so its far edge is
        // `car + (win_fwd) + reach`, not `car + reach`. Getting this wrong does
        // not fail loudly: `GatheredRec` answers 0 outside the record, so the
        // transcription writes a confident zero into every byte past the edge,
        // and 116 bytes of a 116-byte sample came out constant.
        let win_fwd = record::win_len() as i64 - record::win_back();
        let hi = win_fwd + reach.max(0x310 + 8);
        if hi > win_fwd {
            extra.push((win_fwd, (hi - win_fwd) as u32));
        }
    } else {
        let span = back + fwd;
        let each = (span as u32).div_ceil(6);
        let mut o = -back;
        while o < fwd {
            let l = (each as i64).min(fwd - o) as u32;
            if l > 0 {
                extra.push((o, l));
            }
            o += each as i64;
        }
    }
    // Where a record offset came from, as a rel -- so phase B can ask for the
    // copy phase A found. The record is [0..4] clock, then the production
    // window, then the extras in the order they were requested.
    let rel_of = |po: usize, extra: &record::ExtraSegs| -> Option<i64> {
        let mut at = 4usize + record::win_len() as usize;
        for (rel, len) in extra {
            if po >= at && po < at + *len as usize {
                return Some(*rel + (po - at) as i64);
            }
            at += *len as usize;
        }
        None
    };
    let g = GatherOpts {
        bias_override: Some(anchors.bias),
        anchors: Some(anchors),
        period,
        phase_ms,
        verbose,
        dedup: Some((0, 4 + record::win_len())),
        choose_copy: false,
        self_check: false,
        extra: extra.clone(),
        pos_from: car,
        ..GatherOpts::production(dump)
    };
    // PHASE A -- WIDE, AND ONLY WHEN THERE IS NO POINTER.
    //
    // There WAS a calibration knob here, `FK_FIELD_REL`, on the reasoning that
    // the copy holding the fields sits at a fixed offset from the anchor for a
    // given binary and map, so a search with one right answer need not be re-run
    // on every fork. THE REASONING IS WRONG, and the run that proved it is worth
    // keeping: the offset is relative to the ANCHOR, and the anchor is chosen
    // per fork -- the same file on the same binary picked base-1574780 on one
    // attempt and base-872608 on the next. A rel measured against one is
    // meaningless against the other, and re-running with the printed value
    // gathered a 1332-byte record with 0 of 4 wheel slots live.
    //
    // The guards caught it, as designed. But a knob whose value is only ever
    // valid inside the process that printed it is a footgun, not a cache -- and
    // the POINTER is the real form of what it was reaching for: an address the
    // engine itself holds, resolved fresh in every fork. So with a pointer
    // there is no phase A at all, and without one there is no shortcut.
    //
    // How coarse phase A may be, when it runs. The gather is on the RECORDING's
    // grid, so the instant count is the ghost's sample count -- not the clean
    // run's, which is five times denser at 10 ms and made this over-stride by
    // 5x (48 asked for, 10 delivered, and the guard correctly refused to call
    // that the same run).
    let n_samples = (truth.len() as i64 * 10 / period.max(1)).max(1);
    // PHASE A'S STRIDE IS AN OPTIMISATION AND MUST NOT CHANGE THE ANSWER.
    //
    // A strided gather lands on instants the clean run may not have measured,
    // so the pairing is looser and the chosen copy scored 0.110236 m against
    // the 1e-3 m bar -- the guard refusing a correct copy because the probe was
    // sparse. The copy search is cheap once the record is in hand; it is the
    // GATHER that costs. So phase A stays dense by default and the stride is
    // opt-in via FK_PHASE_A_STRIDE for anyone who has measured that it is safe
    // on their run.
    let stride = std::env::var("FK_PHASE_A_STRIDE")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(1)
        .max(1)
        .min((n_samples / PHASE_A_TARGET as i64).max(1));
    let wide_period = if car.is_some() { period } else { period * stride };
    let g = GatherOpts { period: wide_period, ..g };
    let two = record::run_clean_anch(c, &g)?;
    let recs = record::read_samples_pair(dump, two.reclen);
    // WHERE THE RECORD'S BYTES CAME FROM, when a pointer placed them.
    //
    // A record offset is not `anchor + something`: the segments are separate
    // windows and the gatherer may sort, merge or clip them, so the only honest
    // map from an address to a record offset is `segs_abs`. Printing it is what
    // turns "the copy is at +320 and the anchor should be at +196" from a
    // mystery into a subtraction.
    if car.is_some() {
        let segs: Vec<String> =
            two.segs_abs.iter().map(|(a, l)| format!("{:#x}+{}", a, l)).collect();
        println!("  pointer window: record {} B from {}", two.reclen, segs.join(" "));
    }
    if verbose || car.is_none() {
        println!(
            "field gather phase A: {} instants at {} ms, record {} B ({:.1} MB gathered)",
            recs.len(),
            wide_period,
            two.reclen,
            recs.len() as f64 * two.reclen as f64 * 2.0 / 1e6
        );
    }

    // Pair the gathered instants with the clean run's own measured positions.
    let mut idx: Vec<usize> = Vec::new();
    let mut want: Vec<[f64; 3]> = Vec::new();
    let mut ms: Vec<i64> = Vec::new();
    for (i, (clk, _, _)) in recs.iter().enumerate() {
        let t = *clk as i64 - two.bias;
        if let Some(p) = truth.get(&t) {
            // A non-finite reference instant is not a reference. The clean run
            // records the countdown too, and the state before the car exists is
            // not a position.
            if !p.iter().all(|v| v.is_finite()) {
                continue;
            }
            idx.push(i);
            want.push(*p);
            ms.push(t);
        }
    }
    if ms.len() < 20 {
        return Err(format!(
            "the field gather and the clean run share only {} instants -- they are not the \
             same run",
            ms.len()
        ));
    }
    let sample: Vec<Vec<u8>> = Vec::new();
    let _ = sample;

    // Find the car, then step to the copy that has the fields. Same two rules
    // as `fk carrier write`, for the same reasons (see `find_car`), except that
    // the positions come from the engine rather than from a recording.
    let n = ms.len();
    let reclen = two.reclen;
    let probe = n / 2;
    let hi = reclen.saturating_sub(12);
    // IN LAYOUT MODE A COPY IS ONLY A CANDIDATE IF ITS WHOLE STATE IS IN THE
    // RECORD. `vislayout::pack` reads all 864 bytes of the struct and
    // `GatheredRec` answers 0 outside the window, so a copy whose tail runs off
    // the edge does not produce a partial sample -- it produces a confident
    // zero in every byte past the edge, which passes every acceptance test the
    // file has. Refusing it here is what makes the window's size a measurable
    // property instead of an assumption.
    let covered = |o: usize| -> bool {
        let (p, sz) = (crate::vislayout::POS_IN_STATE as usize, crate::vislayout::STATE_SIZE as usize);
        !layout_mode || (o >= p && o - p + sz <= reclen)
    };
    let mut cands: Vec<(f64, usize, Write)> = Vec::new();
    // PROBE AT SEVERAL INSTANTS, NOT ONE.
    //
    // A candidate had to sit on the car at ONE instant, the midpoint, and the
    // set it produced was therefore the set of objects that are the car THERE.
    // On a map that changes the vehicle under you -- 227654 is
    // DesertCar / SnowCar / Bobsleigh -- the car's live state is a different
    // object in each phase, so a midpoint probe can only ever see the middle
    // one and the other two are not candidates at all. Probing across the run
    // costs one pass of arithmetic per probe over a record already in memory,
    // and it is what makes the phases visible.
    let probes: Vec<usize> = {
        let mut v: Vec<usize> = (1..=15).map(|k| (n * k) / 16).collect();
        v.push(n / 2);
        v.sort_unstable();
        v.dedup();
        v.retain(|p| *p < n);
        v
    };
    let mut seen: std::collections::HashSet<(usize, u8)> = Default::default();
    for wr in [Write::Car, Write::Other] {
        for &probe in &probes {
            let b = match wr {
                Write::Car => &recs[idx[probe]].1,
                Write::Other => &recs[idx[probe]].2,
            };
            for o in (4..hi).step_by(4) {
                if !covered(o) || seen.contains(&(o, wr as u8)) {
                    continue;
                }
                let mut d = 0.0;
                for k in 0..3 {
                    let e = f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap())
                        as f64;
                    d += (e - want[probe][k]).powi(2);
                }
                // `!(d < eps)`, not `d >= eps`. A NaN fails BOTH comparisons, so
                // the `>=` form lets every offset in the window through as a
                // candidate -- measured here as "1395 copies of the car" and a
                // median error printed as NaN. A float filter has three outcomes.
                if !(d < 1e-6) {
                    continue;
                }
                seen.insert((o, wr as u8));
                let mut e: Vec<f64> = Vec::with_capacity(n);
                for i in 0..n {
                    let b = match wr {
                        Write::Car => &recs[idx[i]].1,
                        Write::Other => &recs[idx[i]].2,
                    };
                    let mut d = 0.0;
                    for k in 0..3 {
                        let v = f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap())
                            as f64;
                        d += (v - want[i][k]).powi(2);
                    }
                    e.push(d.sqrt());
                }
                let mut s = e.clone();
                s.sort_by(|a, b| a.total_cmp(b));
                cands.push((s[s.len() / 2], o, wr));
            }
        }
    }
    if cands.is_empty() {
        // SAY HOW FAR OFF IT WAS. With a blind window "nothing here holds the
        // trajectory" is the whole story; with a POINTER the window is the
        // struct the chain named, and the distance from the run's own path is
        // the difference between "the chain is stale" (metres) and "the chain
        // is right and something else is wrong" (sub-millimetre). A fallback
        // that does not print it leaves the next person guessing.
        let mut worst = String::new();
        if car.is_some() {
            let mut e: Vec<f64> = Vec::new();
            for (j, i) in idx.iter().enumerate() {
                let b = &recs[*i].1;
                let po = 4 + record::win_back() as usize;
                if po + 12 > reclen {
                    break;
                }
                let d: f64 = (0..3)
                    .map(|k| {
                        (f32::from_le_bytes(b[po + k * 4..po + k * 4 + 4].try_into().unwrap())
                            as f64
                            - want[j][k])
                            .powi(2)
                    })
                    .sum();
                e.push(d.sqrt());
            }
            e.sort_by(|a, b| a.total_cmp(b));
            if !e.is_empty() {
                worst = format!(
                    " -- the state the chain named is {:.6} m from it (median over {} instants)",
                    e[e.len() / 2],
                    e.len()
                );
            }
        }
        return Err(format!(
            "no copy in the field window holds the trajectory the clean run measured{}{}",
            worst,
            if layout_mode {
                format!(
                    " (layout mode also requires the copy's whole 864-byte state inside the \
                     {}-byte record, so a copy near either edge is not a candidate)",
                    reclen
                )
            } else {
                String::new()
            }
        ));
    }
    cands.sort_by(|a, b| a.0.total_cmp(&b.0));
    // ARE THE FOUR WHEEL-ROTATION SLOTS LIVE? That is what separates the copy
    // of the car that carries the fields from a bare position copy with dead
    // memory around it, and it is the guard that stops a file full of zeroed
    // wheels passing every acceptance test.
    //
    // The offsets come from the TABLE when there is one. In layout mode there
    // are no rows -- the writer's transcription replaces them -- so the filter
    // matched nothing, `live` returned 0 for every candidate, and the guard
    // refused every copy including the right one. It read exactly like "the
    // window does not reach the car" while the car was sitting at 0.000000 m.
    //
    // So in layout mode the offsets are stated from the measured structure
    // instead: `CARRIER.md`'s wheel record is `car + 88 + 44k` with the
    // rotation at +4, i.e. car+92, +136, +180, +224 — the exact four offsets
    // the table's `u16@6/8/10/12` rows carry, confirmed on eight keys. They are
    // CAR-relative already; deriving them from the class descriptor's
    // state-relative 0x88 and subtracting 0x50 puts them 32 bytes low, which
    // scores 3 of 4 on neighbouring live floats and is the kind of near-miss
    // that looks like a result.
    let wheel_rels: Vec<i64> = if layout_mode {
        crate::vislayout::wheel_rot_rel().to_vec()
    } else {
        rows.iter()
            .filter(|r| matches!(r.ch, Channel::U16(b) if (6..=12).contains(&b) && b % 2 == 0))
            .map(|r| r.rel)
            .collect()
    };
    let live = |o: usize, w: Write| -> usize {
        wheel_rels
            .iter()
            .filter(|rel| {
                let q = o as i64 + **rel;
                if q < 0 || q as usize + 4 > reclen {
                    return false;
                }
                let q = q as usize;
                let g = |i: usize| {
                    let b = match w {
                        Write::Car => &recs[idx[i]].1,
                        Write::Other => &recs[idx[i]].2,
                    };
                    f32::from_le_bytes(b[q..q + 4].try_into().unwrap())
                };
                let f = g(0);
                f.is_finite() && (1..n).any(|i| g(i) != f && g(i).is_finite())
            })
            .count()
    };
    let mut ranked: Vec<(usize, (f64, usize, Write))> =
        cands.iter().take(64).map(|c| (live(c.1, c.2), *c)).collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1 .0.total_cmp(&b.1 .0)));
    // THE TABLE, NOT JUST THE WINNER.
    //
    // The chooser ranks by live wheel slots first and distance second, and it
    // printed one line: the winner. On 227654 that line said `112.588863 m`
    // twenty-four times over -- and the one question it could not answer was
    // whether a copy at 0.000x m had been ranked BELOW it, which is the
    // difference between "the ranking is wrong" and "no copy here is the car".
    // A refusal that cannot be diagnosed from its own output costs a fork every
    // time somebody asks.
    let table = |head: &str| {
        println!("  {head}");
        println!("    {:>4}  {:>9}  {:>5}  {:>14}  {}", "rank", "record+", "wheel", "vs clean run", "write");
        for (i, (nl, (e, o, w))) in ranked.iter().take(10).enumerate() {
            println!(
                "    {:>4}  {:>9}  {:>3}/4  {:>12.6} m  {}",
                i, o, nl, e, w.name()
            );
        }
        if ranked.len() > 10 {
            println!("    ... {} more copies", ranked.len() - 10);
        }
    };
    if verbose {
        table("candidate copies, best first:");
        // WHAT THE DISTANCE IS MADE OF, for the copy that will be chosen.
        //
        // A median says how far; it cannot say what KIND of far, and the three
        // kinds want three different responses. A CONSTANT delta vector is one
        // object held in another frame (the fields on it are still this car's).
        // A delta whose size tracks speed is the same car read at another
        // instant. A delta that wanders is a different object. On 227654 the
        // copy that holds the live wheel slots sits 112.588863 m from the car
        // and the copies at 0.000000 m have no wheels at all, so which of the
        // three this is decides whether the run is recoverable or not.
        let (_, (_, po0, wr0)) = ranked[0];
        let mut dv: Vec<[f64; 3]> = Vec::with_capacity(n);
        for i in 0..n {
            let b = match wr0 {
                Write::Car => &recs[idx[i]].1,
                Write::Other => &recs[idx[i]].2,
            };
            let mut d = [0.0f64; 3];
            for k in 0..3 {
                d[k] = f32::from_le_bytes(b[po0 + k * 4..po0 + k * 4 + 4].try_into().unwrap())
                    as f64
                    - want[i][k];
            }
            dv.push(d);
        }
        let mut mag: Vec<f64> = dv
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            .collect();
        mag.sort_by(|a, b| a.total_cmp(b));
        let mean = |k: usize| dv.iter().map(|d| d[k]).sum::<f64>() / n as f64;
        let sd = |k: usize| {
            let m = mean(k);
            (dv.iter().map(|d| (d[k] - m).powi(2)).sum::<f64>() / n as f64).sqrt()
        };
        println!(
            "  the chosen copy's offset from the car, per instant: p0 {:.3} p10 {:.3} p50 {:.3} \
             p90 {:.3} p100 {:.3} m",
            mag[0],
            mag[n / 10],
            mag[n / 2],
            mag[(n * 9) / 10],
            mag[n - 1]
        );
        println!(
            "  as a vector: mean ({:.3}, {:.3}, {:.3}) sd ({:.3}, {:.3}, {:.3}) -- a constant \
             offset has sd ~0, a time offset does not",
            mean(0),
            mean(1),
            mean(2),
            sd(0),
            sd(1),
            sd(2)
        );
        // WHEN IS EACH COPY THE CAR?
        //
        // The whole gather assumes ONE copy holds the car for the WHOLE run, and
        // scores candidates by a median over all instants. If instead the car's
        // state moves between objects part-way through -- which is what a map
        // that changes the vehicle would do -- then no single copy has a small
        // median and the ranking is choosing between two halves of one run.
        // The intervals say which world this is: copies whose "on the car"
        // windows TILE the run are one car in several objects; copies that are
        // never on the car are other objects.
        println!("  when each copy is ON the car (within 1 mm), by race time:");
        for (nl, (_, o, w)) in ranked.iter().take(8) {
            let mut first: Option<i64> = None;
            let mut last: Option<i64> = None;
            let mut on = 0usize;
            for i in 0..n {
                let b = match w {
                    Write::Car => &recs[idx[i]].1,
                    Write::Other => &recs[idx[i]].2,
                };
                let mut d = 0.0;
                for k in 0..3 {
                    let v = f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap())
                        as f64;
                    d += (v - want[i][k]).powi(2);
                }
                if d.sqrt() < 1e-3 {
                    on += 1;
                    first.get_or_insert(ms[i]);
                    last = Some(ms[i]);
                }
            }
            println!(
                "    record+{:<9} {}/4 wheels  on the car at {} of {} instants{}",
                o,
                nl,
                on,
                n,
                match (first, last) {
                    (Some(a), Some(b)) => format!("  ({:.3} .. {:.3} s)", a as f64 / 1000.0, b as f64 / 1000.0),
                    _ => String::new(),
                }
            );
        }
    }
    let (mut nlive, (mut err, mut po, mut wr)) = ranked[0];
    // Which copy each instant's fields come from. One entry per instant so the
    // extraction below can be per-instant; on every map where one copy is the
    // car for the whole run, every entry is that copy and the file is
    // byte-for-byte what it was before this existed.
    let mut pick: Vec<(usize, Write)> = vec![(po, wr); n];
    let mut stitched = 0usize;
    println!(
        "field gather: the car is at record +{} on the {} write, {:.6} m from the clean run's \
         own measured path over {} instants, {} of 4 wheel slots live ({} copies)",
        po, wr.name(), err, n, nlive, cands.len()
    );
    // WHERE THE COPY WAS FOUND, as an offset from the anchor. Printed as a
    // diagnostic and nothing more: the anchor is chosen per fork, so this
    // number is only meaningful inside the process that printed it. It used to
    // be offered as a calibration to re-run with, and it is not one.
    if car.is_none() {
        if let Some(rel) = rel_of(po, &extra) {
            println!("  (the copy is anchor{:+}, for this fork only)", rel);
        }
    }
    // A MEASUREMENT WORTH READING, not a diagnostic.
    //
    // `err` is how far the copy that HAS THE FIELDS sits from the copy the
    // transform was read from. If the two were the same object it would be
    // zero. Measured on map 2 it is 0.000491 m -- which is the number this
    // project has called the "client-vs-server floor" for months, and which a
    // 2026-08-20 note already suspected: *~0.0005 m is the signature of the
    // shadow, not a measure of accuracy; a gather that found the car is
    // bit-identical or ~0.000001 m*.
    //
    // So the transform is being read from a copy half a millimetre from the one
    // the game itself recorded, and the fields would be paired with a position
    // one tick's worth away from them. Both are worth knowing and neither is
    // this command's to fix: `fk regen` owns the transform.
    if err > 1e-5 {
        println!(
            "  NOTE: the copy holding the fields is {:.6} m from the copy the transform was \
             read from. They are different objects, so the written sample would pair a \
             position with fields from another instant of it -- and {:.6} m is the shadow \
             signature, not an accuracy floor.",
            err, err
        );
    }
    if nlive < 4 || !(err < 1e-3) {
        // THE CAR IS NOT ALWAYS ONE OBJECT, AND ON SOME MAPS IT IS THREE.
        //
        // Everything above asks "which copy is the car" once, for the whole
        // run. On 227654 there is no such copy: every copy with four live wheel
        // slots is EXACT (0.000000 m) for 337 of 1159 instants and hundreds of
        // metres away for the rest, and every copy that holds the position for
        // the whole run has no wheels at all. The map is
        // DesertCar / SnowCar / Bobsleigh: it changes the vehicle under you,
        // each vehicle is its own `CSceneVehicleVisState`, and the boundaries
        // are the MAP's -- 19.500 and 36.300 s, the same instants where the
        // container's own recording breaks its entities, on a tape with no
        // respawn in it.
        //
        // So: choose per instant instead of per run. Each phase must clear the
        // SAME two bars inside its own window -- sub-millimetre against the
        // clean run, four live wheel slots -- and the phases must TILE the run,
        // because a gap here is a hole in the telemetry and an unnoticed hole
        // is a confident zero.
        //
        // It runs ONLY where the single-copy answer was refused, so a map that
        // regenerates today produces the same bytes tomorrow: the control is a
        // single-vehicle map re-run before and after, and it must be identical.
        let mask_of = |o: usize, w: Write| -> Vec<bool> {
            (0..n)
                .map(|i| {
                    let b = match w {
                        Write::Car => &recs[idx[i]].1,
                        Write::Other => &recs[idx[i]].2,
                    };
                    let mut d = 0.0;
                    for k in 0..3 {
                        let v = f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap())
                            as f64;
                        d += (v - want[i][k]).powi(2);
                    }
                    d.sqrt() < 1e-3
                })
                .collect()
        };
        // Liveness INSIDE the window, not over the run: a wheel that turns for
        // the seventeen seconds this copy is the car is alive, and the same
        // slot read over the whole run is dominated by the instants where this
        // object is not the car at all.
        let live_in = |o: usize, w: Write, mask: &[bool]| -> usize {
            wheel_rels
                .iter()
                .filter(|rel| {
                    let q = o as i64 + **rel;
                    if q < 0 || q as usize + 4 > reclen {
                        return false;
                    }
                    let q = q as usize;
                    let g = |i: usize| {
                        let b = match w {
                            Write::Car => &recs[idx[i]].1,
                            Write::Other => &recs[idx[i]].2,
                        };
                        f32::from_le_bytes(b[q..q + 4].try_into().unwrap())
                    };
                    let on: Vec<usize> = (0..n).filter(|i| mask[*i]).collect();
                    match on.first() {
                        None => false,
                        Some(f) => {
                            let v0 = g(*f);
                            v0.is_finite() && on.iter().any(|i| g(*i) != v0 && g(*i).is_finite())
                        }
                    }
                })
                .count()
        };
        let mut phases: Vec<(usize, Write, Vec<bool>)> = Vec::new();
        let mut got = vec![false; n];
        loop {
            let mut best: Option<(usize, usize, Write, Vec<bool>)> = None;
            for (_, (_, o, w)) in ranked.iter() {
                if phases.iter().any(|(po2, w2, _)| po2 == o && w2 == w) {
                    continue;
                }
                let m = mask_of(*o, *w);
                if live_in(*o, *w, &m) < 4 {
                    continue;
                }
                let gain = (0..n).filter(|i| m[*i] && !got[*i]).count();
                if gain > best.as_ref().map(|b| b.0).unwrap_or(0) {
                    best = Some((gain, *o, *w, m));
                }
            }
            match best {
                Some((_, o, w, m)) => {
                    for i in 0..n {
                        if m[i] {
                            got[i] = true;
                        }
                    }
                    phases.push((o, w, m));
                }
                None => break,
            }
        }
        let holes: Vec<usize> = (0..n).filter(|i| !got[*i]).collect();
        if !phases.is_empty() {
            println!(
                "  the car is {} object(s) over this run -- per-instant selection:",
                phases.len()
            );
            for (o, w, m) in &phases {
                let on: Vec<usize> = (0..n).filter(|i| m[*i]).collect();
                println!(
                    "    record+{:<9} {} write  {} instants  {:.3} .. {:.3} s",
                    o,
                    w.name(),
                    on.len(),
                    ms[*on.first().unwrap()] as f64 / 1000.0,
                    ms[*on.last().unwrap()] as f64 / 1000.0
                );
            }
        }
        if phases.len() > 1 && holes.is_empty() {
            // Each instant is taken from the FIRST phase that holds it: the
            // phases are added largest-gain first, so an instant two of them
            // both hold goes to the one that holds more of the run. An overlap
            // is a choice and it is made here, in the open, rather than by the
            // order of a hash map.
            for i in 0..n {
                for (o, w, m) in &phases {
                    if m[i] {
                        pick[i] = (*o, *w);
                        break;
                    }
                }
            }
            let (o0, w0, _) = phases[0];
            po = o0;
            wr = w0;
            nlive = 4;
            err = 0.0;
            stitched = phases.len();
        } else if !holes.is_empty() && !phases.is_empty() {
            println!(
                "  the phases do not tile the run: {} of {} instants are held by no copy with \
                 live wheels, the first at {:.3} s. A gap is a hole in the telemetry, so this \
                 is refused rather than filled.",
                holes.len(),
                n,
                ms[holes[0]] as f64 / 1000.0
            );
        }
    }
    if stitched == 0 && nlive < 4 {
        if !verbose {
            table("candidate copies, best first:");
        }
        return Err(format!(
            "no copy in the field window has all four wheel slots live (best {} of 4, at +{}) \
             -- these are bare position copies and every field read from one would be dead \
             memory written into a file that passes every acceptance test",
            nlive, po
        ));
    }
    if stitched == 0 && !(err < 1e-3) {
        if !verbose {
            table("candidate copies, best first:");
        }
        return Err(format!("the chosen copy is {:.6} m from the clean run's own path", err));
    }

    let mut out: std::collections::HashMap<i64, Instant> = Default::default();
    let gq = |b: &[u8], o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64;
    // WHERE IS THE ORIENTATION ON THIS COPY?
    //
    // Not necessarily where it is on the located copy. The two are not copies in
    // the sense that matters: one has a live wheel block and the other does not,
    // so this is a different struct that happens to contain a position, and an
    // offset measured on the other one has no reason to transfer. Measured: it
    // does not -- taking the anchor's relative offset makes the written
    // orientation bytes WORSE (2 of 455 identical against 237).
    //
    // So it is located the same way everything else here is: scan, then score
    // on something the answer key does not supply. Four consecutive floats that
    // form a UNIT quaternion and VARY, ranked by how tightly the body's forward
    // axis tracks the direction of travel over the whole run. A car drifts, so
    // the score is the SPREAD of that angle and not its size -- the rule the
    // locator already uses, for the reason it already has: a constant
    // quaternion satisfies a norm test and an identity rotation is
    // indistinguishable from a zeroed slot.
    // WITH A POINTER THERE IS NO ORIENTATION HUNT, AND THAT IS DELIBERATE.
    //
    // The hunt below searches +-4 KB around the car for four consecutive floats
    // that form a varying unit quaternion, because on a blind gather the copy
    // holding the fields is a different object from the one the anchor's
    // offsets were measured on. A pointer window is 864 bytes -- the struct
    // itself -- and the struct holds `Loc`'s 3x3 rotation, not a quaternion, so
    // the hunt has nothing to find and would fail the whole gather. The fields
    // do not depend on it (they are read at `car + rel`), and
    // `--transform-from-fields`, which does, is REFUSED with a pointer window
    // by `fk regen` rather than served with a guess.
    let (qoff, qk, qsign) = if car.is_some() || (layout_mode && stitched > 0) {
        // NO ORIENTATION HUNT ON A STITCHED RUN, and the reason is the same as
        // the pointer window's: the hunt scores ONE copy's quaternion over the
        // WHOLE run, and on a stitched run no copy is the car for the whole
        // run -- outside its own phase it would be scoring a stranger, and the
        // answer-key veto would then refuse a correct gather. Nothing is lost:
        // in layout mode the orientation bytes are `UNPREDICTED` and are not
        // written from here at all (`fk regen` owns the transform).
        (0usize, anchors.quat_kind, 1.0f64)
    } else {
        let vo = (po as i64 + anchors.vel_off) as usize;
        if vo + 12 > reclen {
            return Err("the field window does not cover this copy's velocity".into());
        }
        let at = |i: usize, o: usize| -> f64 {
            let b = match wr {
                Write::Car => &recs[idx[i]].1,
                Write::Other => &recs[idx[i]].2,
            };
            f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64
        };
        let rows_i: Vec<usize> = (0..n).step_by((n / 120).max(1)).collect();
        let score = |qo: usize, kind: u8| -> Option<f64> {
            let mut a: Vec<f64> = Vec::new();
            for &i in &rows_i {
                let q4: [f64; 4] = std::array::from_fn(|k| at(i, qo + k * 4));
                if !q4.iter().all(|v| v.is_finite()) {
                    return None;
                }
                let nq = q4.iter().map(|v| v * v).sum::<f64>().sqrt();
                if (nq - 1.0).abs() > 1e-3 {
                    return None;
                }
                let q = if kind == 0 { q4 } else { [q4[1], q4[2], q4[3], q4[0]] };
                let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
                let f = [
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ];
                let v: [f64; 3] = std::array::from_fn(|k| at(i, vo + k * 4));
                let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if nv < 5.0 {
                    continue;
                }
                a.push(
                    ((f[0] * v[0] + f[1] * v[1] + f[2] * v[2]) / nv).clamp(-1.0, 1.0).acos(),
                );
            }
            if a.len() < 20 {
                return None;
            }
            // and it must VARY: a constant unit quaternion passes every other test
            let q0: [f64; 4] = std::array::from_fn(|k| at(rows_i[0], qo + k * 4));
            if !rows_i.iter().any(|&i| (0..4).any(|k| at(i, qo + k * 4) != q0[k])) {
                return None;
            }
            let m = a.iter().sum::<f64>() / a.len() as f64;
            Some((a.iter().map(|x| (x - m).powi(2)).sum::<f64>() / a.len() as f64).sqrt())
        };
        let lo = (po as i64 - 4096).max(4) as usize;
        let hi = (po + 4096).min(reclen.saturating_sub(16));
        let mut best: Vec<(f64, usize, u8)> = Vec::new();
        for qo in (lo..hi).step_by(4) {
            for kind in [0u8, 1] {
                if let Some(sc) = score(qo, kind) {
                    best.push((sc, qo, kind));
                }
            }
        }
        if best.is_empty() {
            return Err(format!(
                "no varying unit quaternion within 4 KB of the car at record +{}",
                po
            ));
        }
        best.sort_by(|a, b| a.0.total_cmp(&b.0));
        // And, when the file carries the game's own orientation, how far each
        // candidate is FROM IT. This is the answer key and it only reports: the
        // reference-free spread has to be the thing that decides, or the method
        // does not work on the transplanted containers it exists for.
        let vs_key = |qo: usize, kind: u8| -> Option<(f64, f64)> {
            if truth_q.is_empty() {
                return None;
            }
            let mut a: Vec<f64> = Vec::new();
            for (j, &i) in rows_i.iter().enumerate() {
                let _ = j;
                let Some(t) = truth_q.get(&ms[i]) else { continue };
                let q4: [f64; 4] = std::array::from_fn(|k| at(i, qo + k * 4));
                let q = if kind == 0 { q4 } else { [q4[1], q4[2], q4[3], q4[0]] };
                // the angle between two rotations, via |<q1,q2>|
                let d: f64 = (0..4).map(|k| q[k] * t[k]).sum::<f64>().abs().clamp(0.0, 1.0);
                a.push(2.0 * d.acos());
            }
            if a.len() < 10 {
                return None;
            }
            a.sort_by(f64::total_cmp);
            // THE FRACTION AND THE TAIL, NOT THE MEDIAN.
            //
            // The median said 0.00000 rad and the written bytes disagreed on
            // 453 of 455 samples, which is not a contradiction: about half the
            // instants match exactly and about half do not, and a median of a
            // bimodal population reports the mode it happens to sit in. This
            // project has that written down -- *a bimodal population
            // masquerades as a refuted law; split before you quote a spread* --
            // and it cost an hour here anyway.
            Some((
                a.iter().filter(|x| **x < 1e-6).count() as f64 / a.len() as f64,
                a[(a.len() as f64 * 0.9) as usize],
            ))
        };
        println!("  orientation candidates near the car (spread of body-vs-travel angle):");
        for (sc, qo, k) in best.iter().take(6) {
            println!(
                "    car{:+<7} {}  spread {:.4} rad{}",
                *qo as i64 - po as i64,
                if *k == 0 { "(x,y,z,w)" } else { "(w,x,y,z)" },
                sc,
                match vs_key(*qo, *k) {
                    Some((f, p90)) => format!(
                        "   [exact on {:.1} % of instants, p90 {:.5} rad, vs the recording's own]",
                        100.0 * f,
                        p90
                    ),
                    None => String::new(),
                }
            );
        }
        // THE RANKING DECIDES; THE ANSWER KEY MAY ONLY VETO.
        //
        // The reference-free score picks the right one: measured on map 2, the
        // top candidate (car+2632, (w,x,y,z)) is **0.00000 rad** from the
        // orientation the game itself recorded, and the runner-up 1056 bytes
        // away is 0.181 rad out. So the rule works -- and its margin is thin,
        // 0.1102 against 0.1310, which is one calibration point and not enough
        // to set a threshold on. Demanding a factor of two refused the correct
        // answer.
        //
        // So the ranking is trusted, and where the container carries this run's
        // own orientation that is checked -- as a VETO, never as the chooser. A
        // chooser that read the file would pick the donor's car on exactly the
        // files this exists for; a veto only ever refuses.
        let (sc, qo, kind) = best[0];
        if let Some((frac, e)) = vs_key(qo, kind) {
            let _ = frac;
            // 0.02 rad is a degree, on the p90 -- not the median, which reports
            // whichever mode of a bimodal population it lands in.
            if e > 0.02 {
                return Err(format!(
                    "the best orientation candidate (car{:+}, spread {:.4} rad) is {:.5} rad \
                     from the orientation this container already carries -- the \
                     reference-free ranking and the answer key disagree, and a wrong \
                     quaternion writes a file that passes every positional check and faces \
                     the wrong way",
                    qo as i64 - po as i64,
                    sc,
                    e
                ));
            }
        }
        // AND THE SIGN. q and -q are the same rotation and DIFFERENT BYTES:
        // the record's encoder deliberately does not normalise (`recwrite`:
        // "the game writes the quaternion it holds, sign and all -- 143 of 474
        // samples carry qw < 0"). So a candidate can be exact as a rotation --
        // measured here at 0.00000 rad against the recording -- and still write
        // every orientation byte differently. That is precisely what the first
        // version of this did.
        //
        // The sign is taken from the container when it carries this run's own
        // orientation, and left alone otherwise. It is NOT solved
        // reference-free, and saying so is the point: an unsolved choice named
        // is a task, an unsolved choice guessed is a file that faces backwards
        // on half its samples.
        let sign = if truth_q.is_empty() {
            1.0
        } else {
            let mut acc = 0.0;
            for &i in &rows_i {
                let Some(t) = truth_q.get(&ms[i]) else { continue };
                let q4: [f64; 4] = std::array::from_fn(|k| at(i, qo + k * 4));
                let q = if kind == 0 { q4 } else { [q4[1], q4[2], q4[3], q4[0]] };
                acc += (0..4).map(|k| q[k] * t[k]).sum::<f64>();
            }
            if acc < 0.0 {
                println!("  orientation sign: the container holds -q; negating");
                -1.0
            } else {
                1.0
            }
        };
        (qo, kind, sign)
    };
    let _ = (qoff, qsign);
    for i in 0..n {
        // The copy this instant's fields come from: `pick` is uniform unless
        // the stitch above found the car in more than one object.
        let (po, wr) = pick[i];
        let b = match wr {
            Write::Car => &recs[idx[i]].1,
            Write::Other => &recs[idx[i]].2,
        };
        let mut v = Vec::with_capacity(rows.len());
        // THE WHOLE SAMPLE, FROM THE WRITER, when the caller asked for the
        // layout instead of a table of fitted rows.
        //
        // `--carrier layout` parses to a single sentinel row so every caller's
        // plumbing is unchanged; here it means "use `vislayout::pack`". That
        // transcription is the game's own archiver, so it produces every byte
        // at once -- including the packed bit-fields (the five reactor members
        // across bytes 89, 90, 91 and 76) that NO per-byte affine row could
        // ever express, which is why three arms failed on byte 89 and the
        // verdict called it closed.
        //
        // Bytes it must not touch are excluded here rather than downstream:
        // `UNPREDICTED` (the orientation words and the countdown, which need
        // inputs this transcription does not have) and `DEAD_IN_SERVER` --
        // byte 34, 19, 20 and the four dirt slots read identically zero in the
        // dedicated server, and writing them would put a confident zero where a
        // real value was.
        if layout_mode {
            let g = GatheredRec { b, base: po as i64 - crate::vislayout::POS_IN_STATE, reclen };
            let packed = crate::vislayout::pack_checked(&g)?;
            for ch in 0..116usize {
                if crate::vislayout::UNPREDICTED.contains(&ch)
                    || crate::vislayout::DEAD_IN_SERVER.contains(&ch)
                {
                    continue;
                }
                v.push((Channel::Byte(ch), packed[ch] as u32));
            }
        }
        for r in rows.iter().filter(|r| r.rel != i64::MIN) {
            let o = po as i64 + r.rel;
            if o < 0 || o as usize + 4 > reclen {
                return Err(format!(
                    "{} wants record offset {} and the field gather is {} bytes wide",
                    r.ch.name(),
                    o,
                    reclen
                ));
            }
            let o = o as usize;
            let m = r.ch.modulus();
            let x = match r.kind {
                Kind::Raw => b[o] as u32,
                Kind::Affine => {
                    let f = f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64;
                    ((r.k * f + r.c).floor() as i64).rem_euclid(m as i64) as u32
                }
                Kind::AffineU8 => ((r.k * b[o] as f64 + r.c).floor() as i64).rem_euclid(m as i64) as u32,
            };
            v.push((r.ch, x));
        }
        // The transform, from THIS copy. The quaternion and velocity sit at the
        // same offsets from the position as they do on the copy the anchor was
        // measured on -- they are copies of one struct.
        let (qo, vo) = (
            (po as i64 + anchors.quat_off) as usize,
            (po as i64 + anchors.vel_off) as usize,
        );
        if qo + 16 > reclen || vo + 12 > reclen {
            return Err("the field window does not cover this copy's quaternion or velocity".into());
        }
        // WHICH QUATERNION, AND IN WHICH ORDER. The offset is the anchor's --
        // the copies are one struct, so a relative offset transfers -- but the
        // ORDER does not: the layout probe reports the same offset every time
        // and its KIND flips between runs, which cost 165922 three files that
        // faced the wrong way. So the order is CHOSEN HERE, per gather, by the
        // only test that can tell them apart without a reference: the body's
        // forward axis must track the direction of travel. Both readings are
        // unit quaternions; only one of them is pointing where the car is going.
        let q = match qk {
            0 => [gq(b, qo), gq(b, qo + 4), gq(b, qo + 8), gq(b, qo + 12)],
            2 => {
                let m: [f64; 9] = std::array::from_fn(|k| gq(b, qo + k * 4));
                record::mat_to_quat(&m)
            }
            _ => [gq(b, qo + 4), gq(b, qo + 8), gq(b, qo + 12), gq(b, qo)],
        };
        out.insert(
            ms[i],
            Instant {
                pos: std::array::from_fn(|k| {
                    f32::from_le_bytes(b[po + k * 4..po + k * 4 + 4].try_into().unwrap())
                }),
                quat: q,
                vel: [gq(b, vo), gq(b, vo + 4), gq(b, vo + 8)],
                fields: v,
            },
        );
    }
    let _ = (reach, behind);
    Ok(out)
}

/// One gathered instant's bytes, as a `vislayout::State`.
///
/// The field gather holds each instant as a flat record; `base` is where the
/// vehicle state starts inside it (`car - 0x50`, since `Loc.translation` is at
/// `0x50` of the state). Out-of-window reads return zero rather than panicking:
/// a window that does not reach a field is a width bug, and the caller's
/// `DEAD_IN_SERVER`/agreement reporting is what surfaces it.
struct GatheredRec<'a> {
    b: &'a [u8],
    base: i64,
    reclen: usize,
}

impl GatheredRec<'_> {
    fn at(&self, off: usize) -> usize {
        let o = self.base + off as i64;
        if o < 0 || o as usize + 4 > self.reclen {
            usize::MAX
        } else {
            o as usize
        }
    }
}

impl crate::vislayout::State for GatheredRec<'_> {
    fn covers_state(&self) -> bool {
        self.base >= 0
            && (self.base as usize).saturating_add(crate::vislayout::STATE_SIZE as usize)
                <= self.reclen
    }
    fn f32(&self, off: usize) -> f32 {
        match self.at(off) {
            usize::MAX => 0.0,
            o => f32::from_le_bytes(self.b[o..o + 4].try_into().unwrap()),
        }
    }
    fn u32(&self, off: usize) -> u32 {
        match self.at(off) {
            usize::MAX => 0,
            o => u32::from_le_bytes(self.b[o..o + 4].try_into().unwrap()),
        }
    }
    fn u8(&self, off: usize) -> u8 {
        match self.at(off) {
            usize::MAX => 0,
            o => self.b[o],
        }
    }
}
