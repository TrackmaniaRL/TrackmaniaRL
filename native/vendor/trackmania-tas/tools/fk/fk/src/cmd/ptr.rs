//! `fk ptr` — find the engine's own pointer to the vehicle state, and use it.
//!
//! ```text
//! fk ptr find   one engine run: snapshot memory, find the car, walk backwards
//! fk ptr check  resolve a chain in a fresh run and grade what it lands on
//! ```
//!
//! `find` produces a MODULE-RELATIVE chain spec; `check` is the acceptance
//! test, and it is the same acceptance the field gather already applies: the
//! state the chain lands on must reproduce the recording's own path and must
//! have all four wheel-rotation slots live. Neither command writes a ghost.

use crate::ptr::{Kind, Snapshot};
use crate::record::{self, GatherOpts};
use crate::session::Ctx;
use std::cell::RefCell;
use std::collections::HashMap;

/// The wheel-rotation slots, relative to the position triple. Four live floats
/// at stride 44 is a car; four constants is a bare position copy, and a
/// regeneration anchored on one writes zeroed wheels into a file that passes
/// every acceptance test there is. See `CARRIER.md` §3.
///
/// DERIVED IN `vislayout`, which is the module that IS the structure. This was
/// one of five independent statements of the same four numbers.
pub fn wheel_rel() -> [i64; 4] {
    crate::vislayout::wheel_rot_rel()
}

/// `Loc.translation` is 0x50 into `CSceneVehicleVisState` (VEHICLEVISSTATE.md),
/// so the struct a pointer would name starts 0x50 before the position.
pub const POS_IN_STATE: u64 = crate::vislayout::POS_IN_STATE as u64;
pub const STATE_SIZE: u64 = crate::vislayout::STATE_SIZE as u64;

const USAGE: &str = "\
fk ptr -- the pointer that owns the vehicle state.

  fk ptr find  --template FILE --map FILE [--depth N] [--maxoff N] [--out TSV]
  fk ptr check --template FILE --map FILE --chain SPEC

find   one engine run. Snapshots every writable mapping while the engine is
       halted, identifies the car against the template's OWN recorded path,
       then walks the snapshot backwards to every chain of pointers that
       reaches it from the game binary's static data.
check  resolves a chain in a fresh server and grades the state it lands on
       against the template's recorded path and the four wheel slots.

  --depth N    how many pointers deep to walk          [4]
  --maxoff N   how far into an object a pointer may land, bytes  [0x400]
  --out TSV    write the chains found
";

pub fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(|s| s.as_str()).unwrap_or("") {
        "find" => find(&args[1..]),
        "check" => check(&args[1..]),
        "--help" | "-h" => {
            print!("{}", USAGE);
            Ok(())
        }
        v => Err(format!("unknown verb {:?}\n{}", v, USAGE)),
    }
}

fn flag(a: &[String], n: &str) -> Option<String> {
    a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned()
}

fn num(a: &[String], n: &str, d: u64) -> u64 {
    flag(a, n)
        .map(|v| {
            let v = v.trim().to_string();
            match v.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16).unwrap_or(d),
                None => v.parse().unwrap_or(d),
            }
        })
        .unwrap_or(d)
}

/// The template's own recorded positions, per race millisecond.
///
/// This is the identifying reference the whole exercise turns on, and it is
/// only a reference when the file is a RECORDING OF THIS RUN — a downloaded
/// human ghost, or a published file whose telemetry has already been
/// regenerated. On a transplanted container it is the donor's path and it
/// identifies the donor's car; `fk ptr` is a calibration command run on a
/// recording, so that case is refused rather than handled.
fn truth_of(template: &str) -> Result<(HashMap<i64, [f64; 3]>, i64, i64), String> {
    let (times, raws) = record::targets_from_ghost(template)?;
    let mut out = HashMap::new();
    for (i, t) in times.iter().enumerate() {
        let (p, _, _, _) = gbx::record::read_transform_pub(&raws[i], 47);
        if p.iter().all(|v| v.is_finite()) {
            out.insert(*t, p);
        }
    }
    if out.len() < 20 {
        return Err(format!(
            "the template carries only {} usable recorded instants -- it is not a recording of \
             a run and cannot identify a car",
            out.len()
        ));
    }
    // THE RECORD'S OWN GRID, not a constant. A gather on the wrong phase pairs
    // with NOTHING: 285885's recording sits off the 50 ms phase this command
    // first assumed, and the run came back "no offset holds the recording's
    // path" -- a phase error wearing the costume of a missing car.
    let (gp, gph) = record::grid_of(&times);
    Ok((out, gp, gph))
}

/// Score every 4-byte-aligned offset of a gathered record against the
/// recording: the median distance from the recorded path, and how many of the
/// four wheel slots hold a live float.
fn car_offsets(
    recs: &[(u32, Vec<u8>, Vec<u8>)],
    truth: &HashMap<i64, [f64; 3]>,
    bias: i64,
    reclen: usize,
) -> Vec<(f64, usize, usize)> {
    let paired: Vec<(usize, [f64; 3])> = recs
        .iter()
        .enumerate()
        .filter_map(|(i, (c, _, _))| truth.get(&(*c as i64 - bias)).map(|p| (i, *p)))
        .collect();
    if paired.len() < 20 {
        return Vec::new();
    }
    let g = |i: usize, o: usize| -> f64 {
        f32::from_le_bytes(recs[i].1[o..o + 4].try_into().unwrap()) as f64
    };
    let probe = paired.len() / 2;
    let mut out: Vec<(f64, usize, usize)> = Vec::new();
    for o in (0..reclen.saturating_sub(12)).step_by(4) {
        // A cheap gate on one instant first: the full median over hundreds of
        // instants at every offset of a 1.25 MB window is the cost this whole
        // exercise exists to remove.
        let (i0, p0) = paired[probe];
        let d0: f64 = (0..3).map(|k| (g(i0, o + k * 4) - p0[k]).powi(2)).sum();
        if !(d0 < 1e-6) {
            continue;
        }
        let mut e: Vec<f64> = Vec::with_capacity(paired.len());
        for (i, p) in &paired {
            let d: f64 = (0..3).map(|k| (g(*i, o + k * 4) - p[k]).powi(2)).sum();
            e.push(d.sqrt());
        }
        e.sort_by(|a, b| a.total_cmp(b));
        let live = wheel_rel()
            .iter()
            .filter(|rel| {
                let q = o as i64 + **rel;
                if q < 0 || q as usize + 4 > reclen {
                    return false;
                }
                let q = q as usize;
                let f = |i: usize| f32::from_le_bytes(recs[i].1[q..q + 4].try_into().unwrap());
                let a = f(paired[0].0);
                a.is_finite() && paired.iter().any(|(i, _)| f(*i) != a && f(*i).is_finite())
            })
            .count();
        out.push((e[e.len() / 2], o, live));
    }
    out.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.total_cmp(&b.0)));
    out
}

/// Grade ONE record offset against the recording: the distances, the count of
/// instants that could be compared, and how many wheel slots are live.
///
/// The pointer says where the state is, so this grades THERE and nowhere else.
/// Taking the best offset in the window instead would grade a neighbour and
/// call the pointer right.
fn grade_at(
    recs: &[(u32, Vec<u8>, Vec<u8>)],
    truth: &HashMap<i64, [f64; 3]>,
    bias: i64,
    off: usize,
) -> (Vec<f64>, usize) {    let mut e: Vec<f64> = Vec::new();
    for (c, first, _) in recs {
        let Some(p) = truth.get(&(*c as i64 - bias)) else { continue };
        let g = |k: usize| -> f64 {
            f32::from_le_bytes(first[off + k * 4..off + k * 4 + 4].try_into().unwrap()) as f64
        };
        let d: f64 = (0..3).map(|k| (g(k) - p[k]).powi(2)).sum();
        if d.is_finite() {
            e.push(d.sqrt());
        }
    }
    e.sort_by(|a, b| a.total_cmp(b));
    let live = wheel_rel()
        .iter()
        .filter(|rel| {
            let q = off as i64 + **rel;
            if q < 0 || q as usize + 4 > recs[0].1.len() {
                return false;
            }
            let q = q as usize;
            let f = |r: &(u32, Vec<u8>, Vec<u8>)| {
                f32::from_le_bytes(r.1[q..q + 4].try_into().unwrap())
            };
            let a = f(&recs[0]);
            a.is_finite() && recs.iter().any(|r| f(r) != a && f(r).is_finite())
        })
        .count();
    (e, live)
}

/// The bias and the anchor candidates, from the same ladder `fk regen` uses.
fn bias_and_anchors(
    c: &Ctx,
    f: &crate::tape::Tape,
    verbose: bool,
) -> Result<(i64, Vec<record::Anchors>), String> {
    let ticks = record::ladder_ticks(f.steer.len() as i64, 200);
    let mut bias = 0i64;
    for t in &ticks {
        match record::measure_bias(c, f, *t, verbose) {
            Ok(b) => {
                bias = b;
                println!("bias {} (tick {})", b, t);
                break;
            }
            Err(e) => println!("bias at tick {}: {}", t, e),
        }
    }
    if bias == 0 {
        return Err("could not measure the clock bias at any checkpoint".into());
    }
    let mut anchors: Vec<record::Anchors> = Vec::new();
    for t in &ticks {
        if let Ok(mut b) = record::measure_anchors(c, f, *t, verbose) {
            for a in b.iter_mut() {
                a.bias = bias;
            }
            anchors.append(&mut b);
        }
        if !anchors.is_empty() {
            break;
        }
    }
    anchors.dedup_by_key(|a| (a.chain.clone(), a.member));
    if anchors.is_empty() {
        return Err("no anchor at any checkpoint".into());
    }
    Ok((bias, anchors))
}

fn ctx_of(a: &[String]) -> Result<Ctx, String> {
    let template = flag(a, "--template").ok_or("--template FILE is required")?;
    let map = flag(a, "--map").ok_or("--map FILE is required")?;
    let work = flag(a, "--work")
        .unwrap_or_else(|| crate::session::Engine::default_work().to_string_lossy().into());
    let shim = flag(a, "--shim")
        .or_else(|| std::env::var("FK_SHIM").ok())
        .or_else(|| crate::session::default_shim().map(|p| p.to_string_lossy().into()))
        .ok_or("no --shim: pass one, set FK_SHIM, or build tools/search")?;
    let server = flag(a, "--server")
        .or_else(|| std::env::var("TM_SERVER").ok())
        .unwrap_or_else(|| "/tmp/tmoracle/server".into());
    Ok(Ctx {
        template,
        map,
        server,
        work,
        shim,
        ckpt: flag(a, "--ckpt").and_then(|v| v.parse().ok()).unwrap_or(0),
    })
}

// ===========================================================================

fn find(a: &[String]) -> Result<(), String> {
    let c = ctx_of(a)?;
    let verbose = a.iter().any(|x| x == "--verbose");
    let depth = num(a, "--depth", 4) as usize;
    let maxoff = num(a, "--maxoff", 0x400);
    let dump = flag(a, "--dump").unwrap_or_else(|| format!("/tmp/fkptr-{}.bin", std::process::id()));
    let (mut truth, gp, gph) = truth_of(&c.template)?;
    let f = crate::tape::Tape::load(&c.template)?;
    let (bias, anchors) = bias_and_anchors(&c, &f, verbose)?;
    // WHOSE PATH IDENTIFIES THE CAR.
    //
    // `--truth record` (the default) uses the file's own recorded positions,
    // which is right for a downloaded recording and WRONG for a transplanted
    // container, where the record is the donor's. `--truth engine` runs the
    // clean gather first and uses the positions the ENGINE just measured for
    // this tape -- the same reference `gather_fields` uses, and the only one
    // that can tell the simulated car from an object playing the file's record
    // back. The two disagree by 978 m on a transplant, which is how this
    // distinction was found rather than assumed.
    let engine_truth = flag(a, "--truth").as_deref() == Some("engine");
    let mut bar = 1e-5;
    // The 10-micron bar is for the copy the RECORDING was written from -- that
    // one matches its own file to the bit. The anchor copy is a different copy
    // of the same car, and CARRIER.md measures the gap between copies at about
    // half a millimetre, so 10 microns rejects it for being what it is: 203072
    // reports 0.000061 m, six times the bar and a hundredth of the real gap.
    // Under --bare, use the same millimetre bar `gather_fields` uses.
    if a.iter().any(|x| x == "--bare") {
        bar = 1e-3;
    }
    if engine_truth {
        let cdump = format!("{}.clean", dump);
        let g = GatherOpts {
            bias_override: Some(bias),
            anchors: Some(&anchors[0]),
            period: 10,
            phase_ms: 0,
            verbose,
            ..GatherOpts::production(&cdump)
        };
        let cl = record::run_clean_anch(&c, &g)?;
        let crecs = record::read_samples_pair(&cdump, cl.reclen);
        let _ = std::fs::remove_file(&cdump);
        truth = crecs
            .iter()
            .map(|(clk, fst, _)| {
                let g = |k: usize| {
                    f32::from_le_bytes(
                        fst[cl.pos_off + k * 4..cl.pos_off + k * 4 + 4].try_into().unwrap(),
                    ) as f64
                };
                (*clk as i64 - cl.bias, [g(0), g(1), g(2)])
            })
            .collect();
        // The clean run reads a copy of the car half a millimetre from the one
        // that has the fields (`CARRIER.md` §6), so the bar against ITS path is
        // the millimetre `gather_fields` uses, not the micron a recording gets.
        bar = 1e-3;
        println!("truth: the engine's own clean run, {} instants", truth.len());
    }

    // The window is the blind one this command exists to replace: 1 MB behind
    // the anchor and 256 KB ahead, which is where the field gather looks today.
    // It is used ONCE here, to learn the pointer.
    let (back, fwd) = (num(a, "--back", 1_048_576) as i64, num(a, "--fwd", 262_144) as i64);
    let mut extra: record::ExtraSegs = Vec::new();
    let each = ((back + fwd) as u32).div_ceil(6);
    let mut o = -back;
    while o < fwd {
        let l = (each as i64).min(fwd - o) as u32;
        if l > 0 {
            extra.push((o, l));
        }
        o += each as i64;
    }
    // `--at search` LOCATES THE CAR INSTEAD OF GUESSING AT IT.
    //
    // The ordinary path resolves `anchors[0]`, a CANDIDATE chain, and on a map
    // where no candidate is valid that lands wherever the bad walk ends --
    // float data, on 287431. The memory search does not guess: it scans for an
    // object whose motion matches the recording. It is slow, and for deriving
    // a chain ONCE that does not matter at all.
    //
    // The address and the snapshot must come from ONE process, so this takes
    // both together and goes straight to the walk.
    if flag(a, "--at").as_deref() == Some("search") {
        let tick = num(a, "--tick", 200) as i64;
        let (car, snap) = record::search_car_and_snapshot(&c, &f, tick, true)?;
        println!(
            "--at search: the car is at {:#x} (module{:+}), located in this process",
            car,
            car as i64 - snap.module as i64
        );
        return chains_to(&snap, car, a, None);
    }
    let snap: RefCell<Option<Snapshot>> = RefCell::new(None);
    // THE ANCHOR'S ADDRESS, IN THE SNAPSHOT'S OWN PROCESS.
    //
    // This closure has always been handed `pos` and has always thrown it away.
    // It is the one thing `--at` could not get from outside: an address is
    // only meaningful in the process that measured it, and every attempt to
    // carry a `base±N` in from a previous run failed for exactly that reason
    // (08798cc -- "NOT IN THE SNAPSHOT in this process", on a map whose chain
    // is known good).
    //
    // Here it is free. The gather resolves the anchor in the child, calls this
    // to snapshot the same child, and the address it passes belongs to those
    // very bytes.
    let anchor_pos: std::cell::Cell<u64> = std::cell::Cell::new(0);
    let take = |pid: i32, _pos: u64, _segs: &[(u64, u32)]| {
        anchor_pos.set(_pos);
        let t = std::time::Instant::now();
        match Snapshot::take(pid) {
            Ok(s) => {
                println!(
                    "snapshot: {} writable mappings, {:.1} MB, module at {:#x} ({}), {} unreadable, {:.2} s",
                    s.chunks.len(),
                    s.bytes as f64 / 1e6,
                    s.module,
                    s.module_path,
                    s.unread.len(),
                    t.elapsed().as_secs_f64()
                );
                *snap.borrow_mut() = Some(s);
            }
            Err(e) => println!("snapshot FAILED: {}", e),
        }
    };
    let g = GatherOpts {
        bias_override: Some(bias),
        anchors: Some(&anchors[0]),
        period: num(a, "--period", gp as u64) as i64,
        phase_ms: gph,
        verbose,
        dedup: Some((0, 4 + record::win_len())),
        choose_copy: false,
        self_check: false,
        extra,
        before_go: Some(&take),
        ..GatherOpts::production(&dump)
    };
    // WITH `--at` THE CLEAN RUN ONLY HAS TO TAKE THE SNAPSHOT.
    //
    // Its self-check is about the TRAJECTORY -- "the clock bias does not
    // belong to this process's counter" on 287431 -- and the backward walk
    // does not use the trajectory at all. It uses the snapshot, which the
    // gather has already taken by the time the check runs (the `take` closure
    // fires during the run). So when the caller has supplied the address, a
    // self-check failure is not fatal: it means the identification would have
    // failed, which is exactly why `--at` was passed.
    let at_given = flag(a, "--at").is_some();
    let out = match record::run_clean_anch(&c, &g) {
        Ok(v) => v,
        Err(e) if at_given && snap.borrow().is_some() => {
            println!("the clean run did not self-check ({}) -- with --at, only the snapshot is needed", e);
            record::CleanOut::default()
        }
        Err(e) => return Err(e),
    };
    // `read_samples_pair` slices by `reclen`, and an empty CleanOut has none.
    // Under --at there is nothing to read: the walk wants the snapshot only.
    let recs = if at_given && out.reclen == 0 {
        Vec::new()
    } else {
        record::read_samples_pair(&dump, out.reclen)
    };
    let _ = std::fs::remove_file(&dump);
    let snap = snap.into_inner().ok_or("no snapshot was taken")?;

    // `--at base±N` NAMES THE CAR INSTEAD OF IDENTIFYING IT.
    //
    // This command is two independent halves: everything to here answers
    // "which object is the car?" by matching candidates against the
    // recording's own path, and everything after walks backwards from that
    // answer to every chain that reaches it.
    //
    // Only the FIRST half fails on 287431. It gathers from tick 0, where that
    // map has not spawned its car yet, so it dies on the very next line with
    // "no offset in the gathered window holds the recording's own path" and
    // never reaches the walk. But the memory SEARCH finds an object there that
    // is the car for the whole run -- base-4012928, stable, the basis of the
    // regeneration that produces the bytes the client accepted.
    //
    // A stable copy EXISTS; only a chain to it is missing. So take the address
    // in the search's own notation and hand it straight to the walk. The
    // acceptance tests below are unchanged, so a wrong address fails there.
    // `--at anchor` uses the address the gather resolved in the snapshot's own
    // process -- the only address that is valid there.
    if flag(a, "--at").as_deref() == Some("anchor") {
        let car = anchor_pos.get();
        if car == 0 {
            return Err("--at anchor: the gather never reported an anchor address".into());
        }
        println!(
            "--at anchor: the car as the gather resolved it here, {:#x} (module{:+})",
            car,
            car as i64 - snap.module as i64
        );
        // THE ANCHOR IS NOT ALWAYS A CAR, and this path skips the acceptance
        // test that would normally say so. On 287431 it comes back as
        // 0x3f8000003f800538 -- `0x3f800000` is the float 1.0 bit pattern,
        // twice, so the chain read FLOAT DATA as a pointer. Walking to an
        // address like that produces confident nonsense: chains that "reach"
        // an object which does not exist.
        //
        // The snapshot is the check. It holds every writable mapping in this
        // process, so an address outside it is not an object at all.
        if !snap.contains(car) {
            return Err(format!(
                "--at anchor: {:#x} is not in any writable mapping of this process -- the                  anchor did not resolve to an object (a chain that reads float data as a                  pointer lands here: 0x3f800000 is 1.0f). There is no car to walk to.",
                car
            ));
        }
        return chains_to(&snap, car, a, None);
    }
    if let Some(v) = flag(a, "--at") {
        let d: i64 = v
            .trim_start_matches("base")
            .parse()
            .map_err(|_| format!("--at wants base±N, e.g. base-4012928, not {:?}", v))?;
        let car = (snap.module as i64 + d) as u64;
        println!(
            "--at {}: taking the car as given at {:#x} (module{:+}), skipping identification",
            v, car, d
        );
        return chains_to(&snap, car, a, None);
    }

    // THE CAR, identified the way everything else here identifies it.
    let scored = car_offsets(&recs, &truth, out.bias, out.reclen);
    let (err, off, live) = *scored
        .first()
        .ok_or("no offset in the gathered window holds the recording's own path")?;
    let car = out
        .addr_of(off)
        .ok_or_else(|| format!("record offset {} is outside the gathered segments", off))?;
    println!(
        "the car is at {:#x} (record +{}), {:.6} m from the recording's own path over {} \
         instants, {} of 4 wheel slots live; {} copies matched",
        car,
        off,
        err,
        recs.len(),
        live,
        scored.len()
    );
    // THE WHEELS ARE REQUIRED FOR THE FIELD GATHER, NOT FOR THE ANCHOR.
    //
    // There are two copies of the car in engine memory and they serve
    // different callers. The vis state carries the wheels and a 3x3 rotation,
    // and `--carrier` needs it. `measure_anchors` needs the OTHER one: a bare
    // position triple with a quaternion 16 B before it and a velocity 12 B
    // after. They are megabytes apart -- measured on 203072, the vis state at
    // ~base-3453700 and the anchor copies at base-307840 and base-306660.
    //
    // This command was written for the first caller and rejects the second
    // outright, which is why `locate_candidates` still has to search hundreds
    // of megabytes for something the engine has a pointer to. `--bare` asks
    // for a chain to the anchor copy instead: same identification against the
    // recording's own path, same distance bar, only the wheel requirement
    // dropped -- because on that copy the wheels are genuinely absent rather
    // than a sign of the wrong object.
    let want_bare = a.iter().any(|x| x == "--bare");
    if live < 4 && !want_bare {
        return Err(format!(
            "the best copy has {} of 4 wheel slots live -- this is a bare position copy and a \
             pointer to it would be a pointer to the wrong thing. Pass --bare if you are \
             looking for the ANCHOR copy (quaternion + velocity, no wheels), which is what \
             `measure_anchors` needs.",
            live
        ));
    }
    if live < 4 {
        println!(
            "--bare: accepting a copy with {} of 4 wheel slots -- this is the anchor copy, \
             not the vis state",
            live
        );
    }
    if !(err < bar) {
        return Err(format!(
            "the chosen copy is {:.6} m from the reference (bar {:.0e}) -- not the car",
            err, bar
        ));
    }
    // THE REFRAME THIS COMMAND NEEDS, and it is small.
    //
    // Everything ABOVE this line answers "which object is the car?" by
    // matching candidates against the recording's own path. Everything BELOW
    // walks backwards from that answer to every chain that reaches it. The two
    // halves are independent, and on a map where the first half cannot work
    // the second half is still exactly what is wanted.
    //
    // 287431 is that map. `ptr find` gathers from tick 0, where the car does
    // not exist yet -- "no offset in the gathered window holds the recording's
    // own path" -- so it never reaches the walk. But the memory SEARCH finds
    // an object there that is the car for the WHOLE run: base-4012928, stable,
    // accepted by every downstream check, used for the 169 s regeneration that
    // produces the bytes the client accepted.
    //
    // So a stable copy EXISTS on that map. Nothing needs to follow a moving
    // car -- the sampler-chain machinery in forkshim was solving a problem the
    // evidence says is not the problem. What is missing is only a CHAIN TO AN
    // ADDRESS WE ALREADY HAVE.
    //
    // Add `--at <addr>` (or a base-relative form) that skips the
    // identification and hands `car` straight to the walk below. The one piece
    // of plumbing needed is the fork server's base in this scope, so the
    // search's own `base±N` notation can be used verbatim.
    let copies: Vec<(u64, f64, usize)> = scored
        .iter()
        .filter_map(|(e, o, l)| out.addr_of(*o).map(|ad| (ad, *e, *l)))
        .collect();
    chains_to(&snap, car, a, Some(&copies))
}


/// Every chain of pointers that reaches `car`, from static data.
///
/// THE SECOND HALF OF `ptr find`, extracted so it can be reached two ways:
/// after identifying the car from the recording's own path, or with an address
/// supplied by `--at`. It uses nothing from the identification -- verified by
/// reading every line of it -- which is why the split is clean.
fn chains_to(
    snap: &Snapshot,
    car: u64,
    a: &[String],
    // The other copies of this car, when the caller identified it and so knows
    // them. `--at` supplies an address and no census, and the walk does not
    // need one -- this only feeds a report.
    copies_in: Option<&[(u64, f64, usize)]>,
) -> Result<(), String> {
    // Re-read the flag-derived knobs here: this function is now reached from
    // two places.
    let depth = num(a, "--depth", 4) as usize;
    let maxoff = num(a, "--maxoff", 0x400);
    let verbose = a.iter().any(|x| x == "--verbose");
    let state = car - POS_IN_STATE;
    println!(
        "the state runs {:#x}..{:#x} ({} bytes); it is {} in this process{}",
        state,
        state + STATE_SIZE,
        STATE_SIZE,
        snap.kind_of(state).map(|k| k.name()).unwrap_or("NOT IN THE SNAPSHOT"),
        match snap.class_of(state) {
            Some((vt, n)) => format!(", vtable mod{} = {}", crate::ptr::hexoff(vt), n),
            None => String::new(),
        }
    );

    // THE CONTROL, before the result. A scan that cannot see a pointer that is
    // there would report "no pointer" in exactly the same words.
    let (ok, n) = snap.recall_control(2000);
    println!("recall control: {} of {} planted pointers found by the same scan", ok, n);
    if ok != n || n == 0 {
        return Err(format!(
            "the scan recovered {} of {} pointers it was shown -- it cannot be trusted to \
             report an absence",
            ok, n
        ));
    }

    // EVERY OTHER COPY OF THIS CAR, because the question is which of them the
    // engine owns. They are the offsets that also reproduce the recording; the
    // one this command anchors on is the one that also has live wheels.
    let mut copies: Vec<(u64, f64, usize)> = copies_in.unwrap_or(&[]).to_vec();
    copies.sort_by_key(|c| c.0);
    println!("{} copies of this car in the gathered window:", copies.len());
    for (ad, e, l) in &copies {
        println!(
            "  {:#x}  car{}  {:.6} m  {} of 4 wheels live{}",
            ad,
            crate::ptr::hexoff(*ad as i64 - car as i64),
            e,
            l,
            if *ad == car { "   <- THE CAR" } else { "" }
        );
    }

    // WHAT POINTS AT IT, AND FROM HOW FAR IN FRONT.
    //
    // The first version of this looked for pointers into the 864 bytes of the
    // struct and found three, all on the stack. That is the wrong question when
    // the struct is a MEMBER of something bigger: a pointer to the owner lands
    // BEFORE the state, not inside it. So the range runs backwards as well, and
    // the delta is reported rather than assumed.
    let before = num(a, "--before", 0x8000);
    let lo = state.saturating_sub(before);
    let wide = snap.direct_pointers((lo, state + STATE_SIZE));
    println!(
        "{} slots point into [state-{:#x}, state+{:#x}]:",
        wide.len(),
        before,
        STATE_SIZE
    );
    let mut shown = 0;
    for (slot, value, kind) in wide.iter() {
        // A slot inside a vehicle state is that state's own field, not an owner.
        if copies.iter().any(|(ad, _, _)| {
            *slot >= *ad - POS_IN_STATE && *slot < *ad - POS_IN_STATE + STATE_SIZE
        }) {
            continue;
        }
        shown += 1;
        if shown > 40 {
            continue;
        }
        println!(
            "  {:#x} [{}{}]  ->  state{}{}",
            slot,
            kind.name(),
            if kind == &Kind::Static {
                format!(" mod{}", crate::ptr::hexoff(*slot as i64 - snap.module as i64))
            } else {
                String::new()
            },
            crate::ptr::hexoff(*value as i64 - state as i64),
            match snap.class_of(*value) {
                Some((vt, n)) => format!("   [{} at vtable mod{}]", n, crate::ptr::hexoff(vt)),
                None => String::new(),
            }
        );
    }
    if shown > 40 {
        println!("  ... {} more", shown - 40);
    }

    // THE OWNER, by counting. Every slot above that is not a vehicle state's
    // own field points at ONE of a handful of addresses below the car, and the
    // most-pointed-at of them is the object the state is a member of. Counting
    // is the whole method: an owner is the thing many independent places hold.
    let mut tally: std::collections::HashMap<u64, usize> = HashMap::new();
    for (slot, value, _) in wide.iter() {
        if copies.iter().any(|(ad, _, _)| {
            *slot >= *ad - POS_IN_STATE && *slot < *ad - POS_IN_STATE + STATE_SIZE
        }) {
            continue;
        }
        *tally.entry(*value).or_default() += 1;
    }
    let mut tal: Vec<(u64, usize)> = tally.into_iter().collect();
    tal.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
    println!("the addresses those slots name, by how many places hold them:");
    for (ad, n) in tal.iter().take(8) {
        println!(
            "  state{}  held by {} slots{}",
            crate::ptr::hexoff(*ad as i64 - state as i64),
            n,
            if *ad == state { "   (the state itself)" } else { "" }
        );
    }
    // A SIBLING ARRAY, if there is one: consecutive 8-byte slots each holding a
    // DIFFERENT one of these objects is the engine's list of vehicles, and that
    // is the structure this exercise was sent to find.
    let owners: std::collections::HashSet<u64> = tal.iter().map(|(a, _)| *a).collect();
    let mut runs: Vec<(u64, Vec<u64>)> = Vec::new();
    for (slot, value, _) in wide.iter() {
        if !owners.contains(value) {
            continue;
        }
        match runs.last_mut() {
            Some((st, v)) if *st + 8 * v.len() as u64 == *slot => v.push(*value),
            _ => runs.push((*slot, vec![*value])),
        }
    }
    runs.retain(|(_, v)| v.len() > 1 && v.iter().collect::<std::collections::HashSet<_>>().len() > 1);
    for (st, v) in runs.iter().take(8) {
        println!(
            "  ARRAY at {:#x} [{}]: {} consecutive slots holding {} distinct objects -- {}",
            st,
            snap.kind_of(*st).map(|k| k.name()).unwrap_or("?"),
            v.len(),
            v.iter().collect::<std::collections::HashSet<_>>().len(),
            v.iter()
                .map(|a| crate::ptr::hexoff(*a as i64 - state as i64))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    // Chains to the OWNER as well as to the state itself. A chain to the owner
    // is the better answer even when both resolve: it names the object the
    // engine keeps, and the state is a member of it at a fixed offset.
    let owner = tal.first().map(|(a, _)| *a).filter(|a| *a != state);
    let mut chains: Vec<crate::ptr::Chain> = Vec::new();
    if let Some(ow) = owner {
        let adj = state as i64 - ow as i64;
        println!(
            "chains to the OWNER at state{} (the state is its member at +{:#x}):",
            crate::ptr::hexoff(ow as i64 - state as i64),
            adj
        );
        let mut cs = snap.chains_to((ow, ow + 8), depth, maxoff, 200);
        for ch in cs.iter_mut() {
            // The chain lands on the owner; the state is `adj` bytes further in.
            if let Some(l) = ch.steps.last_mut() {
                l.delta += adj;
            }
        }
        for ch in &cs {
            println!("  depth {}  {}", ch.depth(), ch.spec());
        }
        if cs.is_empty() {
            println!("  none");
        }
        chains.extend(cs);
    }
    println!("chains to the state itself:");
    chains.extend(snap.chains_to((state, state + STATE_SIZE), depth, maxoff, 200));
    println!(
        "{} chains of at most {} pointers reach the car from static data (maxoff {:#x})",
        chains.len(),
        depth,
        maxoff
    );
    let mut rows: Vec<String> = Vec::new();
    for ch in &chains {
        println!("  depth {}  {}", ch.depth(), ch.spec());
        if verbose {
            for s in &ch.steps {
                println!(
                    "      {:#x} [{}] -> {:#x} (short by {})",
                    s.slot,
                    snap.kind_of(s.slot).map(|k| k.name()).unwrap_or("?"),
                    s.value,
                    s.delta
                );
            }
        }
        rows.push(format!("{}\t{}\t{}", ch.depth(), ch.root_kind.name(), ch.spec()));
    }
    if chains.is_empty() {
        println!(
            "NO STATIC CHAIN. The scan is sound (the control above), so this is a measurement: \
             within {} of pointer indirection and {:#x} of interior offset, nothing in the game \
             binary's own writable data reaches this object.",
            depth, maxoff
        );
    }
    if let Some(p) = flag(a, "--out") {
        let mut s = String::from("depth\troot\tchain\n");
        for r in &rows {
            s.push_str(r);
            s.push('\n');
        }
        std::fs::write(&p, s).map_err(|e| e.to_string())?;
        println!("wrote {}", p);
    }
    Ok(())
}


// ===========================================================================

fn check(a: &[String]) -> Result<(), String> {
    let c = ctx_of(a)?;
    let verbose = a.iter().any(|x| x == "--verbose");
    let spec = flag(a, "--chain").ok_or("--chain SPEC is required")?;
    let dump = flag(a, "--dump").unwrap_or_else(|| format!("/tmp/fkptrc-{}.bin", std::process::id()));
    let (truth, _, _) = truth_of(&c.template)?;
    let f = crate::tape::Tape::load(&c.template)?;
    let (bias, anchors) = bias_and_anchors(&c, &f, verbose)?;

    // THE DEREFERENCE. One `pread` per step, in the live halted engine, and
    // the window that follows is the struct itself.
    let spec2 = spec.clone();
    let landed: RefCell<Vec<u64>> = RefCell::new(Vec::new());
    let resolve = |pid: i32, _base: u64| -> Result<(u64, Vec<(i64, u32)>), String> {
        let (m, _) = crate::ptr::module_base(pid).ok_or("no module base for the live server")?;
        let states = crate::ptr::resolve_pool(pid, m, &spec2)?;
        *landed.borrow_mut() = states.clone();
        // The anchor is the first member; every other member is gathered
        // beside it, so the copy rule can choose between them.
        let anchor = states[0] + POS_IN_STATE;
        let ex: Vec<(i64, u32)> = states[1..]
            .iter()
            .map(|s| (*s as i64 - anchor as i64, (STATE_SIZE + 8) as u32))
            .collect();
        Ok((anchor, ex))
    };
    // The production window puts the position at record +196 and reaches
    // car+256; the rest of the 864-byte struct is gathered right after it, so
    // every member `VEHICLEVISSTATE.md` names is in the record.
    let g = GatherOpts {
        bias_override: Some(bias),
        anchors: Some(&anchors[0]),
        period: num(a, "--period", 10) as i64,
        phase_ms: 0,
        verbose,
        // The same dedup key production uses: one record per distinct vehicle
        // state rather than one per `lroundf` call that touched anything in the
        // window. Measured here: 60.8 MB against 2.6 MB for the same 1399
        // instants.
        dedup: Some((0, 4 + record::win_len())),
        choose_copy: false,
        self_check: false,
        extra: vec![(256, (STATE_SIZE - POS_IN_STATE - 256) as u32 + 8)],
        pos_from: Some(&resolve),
        ..GatherOpts::production(&dump)
    };
    let t0 = std::time::Instant::now();
    let out = record::run_clean_anch(&c, &g)?;
    let recs = record::read_samples_pair(&dump, out.reclen);
    let bytes = std::fs::metadata(&dump).map(|m| m.len()).unwrap_or(0);
    let _ = std::fs::remove_file(&dump);
    // Grade EVERY member of the pool the chain named, at the offset that
    // member's own window starts at, and take the best. One member of an array
    // of four is the live car and the others are not, and which one it is
    // varies by process -- so a spec names the array and the acceptance picks.
    if out.reclen < out.pos_off + 12 || recs.is_empty() {
        return Err(format!(
            "the chain resolved but the gather came back {} bytes wide in {} instants -- there \
             is nothing to grade",
            out.reclen,
            recs.len()
        ));
    }
    // Where each member of the pool landed in the record. Derived from the
    // segment list the gather actually used, not from the order they were
    // asked for: the clipper drops and truncates segments and an assumed
    // offset would grade the wrong bytes.
    let states = landed.borrow().clone();
    let offs: Vec<(usize, u64)> = states
        .iter()
        .filter_map(|s| out.off_of(s + POS_IN_STATE).map(|o| (o, *s)))
        .collect();
    if offs.is_empty() {
        return Err("no member of the pool was gathered -- every segment was clipped".into());
    }
    let mut graded: Vec<(Vec<f64>, usize, usize, u64)> = offs
        .iter()
        .filter(|(o, _)| o + 12 <= out.reclen)
        .map(|(o, st)| {
            let (e, l) = grade_at(&recs, &truth, out.bias, *o);
            (e, l, *o, *st)
        })
        .collect();
    graded.sort_by(|a, b| {
        b.1.cmp(&a.1).then(
            a.0.get(a.0.len() / 2)
                .unwrap_or(&f64::MAX)
                .total_cmp(b.0.get(b.0.len() / 2).unwrap_or(&f64::MAX)),
        )
    });
    for (e, l, _, st) in graded.iter() {
        println!(
            "  member {:#x}: {} of 4 wheels live, {:.6} m median over {} paired instants",
            st,
            l,
            e.get(e.len() / 2).copied().unwrap_or(f64::NAN),
            e.len()
        );
    }
    let (e, live, at, state_at) = graded.remove(0);
    let _ = at;
    println!(
        "{} instants, {} B per record in {} segments, {} B gathered, {:.1} s",
        recs.len(),
        out.reclen,
        out.segs_abs.len(),
        bytes,
        t0.elapsed().as_secs_f64()
    );
    if verbose {
        for (a, l) in &out.segs_abs {
            println!("  segment {:#x} + {}", a, l);
        }
    }
    if e.len() < 20 {
        return Err(format!(
            "the chain resolved and the run gathered {} instants, but only {} of them pair with \
             a recorded instant -- there is nothing to grade the pointer against",
            recs.len(),
            e.len()
        ));
    }
    let (med, p90, worst) = (e[e.len() / 2], e[(e.len() as f64 * 0.9) as usize], e[e.len() - 1]);
    println!(
        "CHAIN {} -> state {:#x}: {:.6} m median from the recording's own path (p90 {:.6}, worst \
         {:.6}) over {} paired instants, {} of 4 wheel slots live",
        spec,
        state_at,
        med,
        p90,
        worst,
        e.len(),
        live
    );
    // The two guards `gather_fields` already applies, and for the same reasons:
    // a wrong copy reads dead memory, and dead memory written into a ghost
    // passes every downstream test there is.
    if live < 4 || !(med < 1e-3) {
        // What the window DOES hold, so a failure says which of the two
        // failures it is: the chain landed somewhere else, or the run did.
        let alt = car_offsets(&recs, &truth, out.bias, out.reclen);
        return Err(format!(
            "the chain landed on a state {:.6} m from the recording with {} of 4 wheel slots \
             live -- REFUSED. The best offset anywhere in the gathered window is {:?}",
            med,
            live,
            alt.first().map(|(d, o, l)| (*d, *o as i64 - at as i64, *l))
        ));
    }
    println!("ACCEPTED");
    Ok(())
}

