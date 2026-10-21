//! `fk` — the command line.
//!
//! Every command takes the same five engine flags and one checkpoint selector.
//! There is no shared config bag: a flag a command accepts is a flag that
//! command uses.

use fk::session::{Checkpoint, Engine};
use fk::tape::Tape;
use std::path::PathBuf;

use fk::cmd;

const USAGE: &str = "\
fk -- the driver for the TM2020 dedicated server used as a physics oracle.

  fk server probe    where a fork server actually stopped, and the safe resume tick
  fk server check    fork resume vs full validation on the same candidates  [THE CONTROL]
  fk server bench    throughput against the batched plain oracle
  fk tree cost       what a savestate-tree branch costs, against every baseline  [Q1]
  fk tree exact      forward-only fork exactness, with both controls and a depth sweep
  fk tree scale      branch-evals per second with many servers side by side
  fk liveness        is this anchor the copy of the car that has the fields?
  fk probe           find a named telemetry channel in the car's memory
  fk trace           one fork -> the car's own state per tick, as a 29-column CSV
  fk watch           the early-abort watchdog: exactness, false positives, speedup
  fk resync          put an old recording's tape back on its own recorded line
  fk regen           rewrite a ghost's telemetry from engine state
  fk carrier         name the sample bytes a regenerated ghost inherits, and write them
  fk ptr             the engine's own pointer to the car: find it, check it

Engine flags, accepted by every command:
  --tape FILE        the .Ghost.Gbx / .Replay.Gbx whose inputs the engine runs
  --map FILE         the map (decoration for a .Replay.Gbx: it carries its own)
  --server DIR       the dedicated-server install       [$TM_SERVER]
  --shim FILE        libforkshim.so                     [$FK_SHIM, or beside fk, or
                     ../search/target/release/]
  --work DIR         scratch; per-process by default, and never shared

Where to stop the simulation (one of):
  --at tick:N        tape tick N, via clock = 36141 + 25.483 * race_ms
  --at clock:N       a raw lroundf call count
  --at frac:F        F of the way through the tape       [default frac:0.5]

Run `fk <command> --help` for a command's own flags.

A fork-reported time is a MEASUREMENT. Only the plain oracle, run on the file as
written to disk, is a RESULT -- see `ghost verify`. The fork server was exact on
4700 of 4700 candidates that perturbed a human reference late in the run, and
LIED on 312 of 312 outside that regime.
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print!("{}", USAGE);
        std::process::exit(if args.is_empty() { 2 } else { 0 });
    }
    if let Err(e) = dispatch(&args) {
        fk::abort(e);
    }
}

fn dispatch(a: &[String]) -> Result<(), String> {
    match a[0].as_str() {
        "server" => {
            let verb = a.get(1).map(|s| s.as_str()).unwrap_or("");
            let rest = &a[2.min(a.len())..];
            let (engine, tape, at) = common(rest)?;
            match verb {
                "probe" => cmd::server::probe(&engine, tape, at),
                "check" => {
                    let o = cmd::server::CheckOpts {
                        n: num(rest, "--n").unwrap_or(20) as usize,
                        seed: num(rest, "--seed").unwrap_or(1) as u64,
                        span: num(rest, "--span").unwrap_or(60) as usize,
                    };
                    match cmd::server::check(&engine, tape, at, o)? {
                        true => Ok(()),
                        false => Err("the fork server did not reproduce the full validation \
                                      on every candidate"
                            .into()),
                    }
                }
                "bench" => cmd::server::bench(
                    &engine,
                    tape,
                    at,
                    num(rest, "--n").unwrap_or(50) as usize,
                    num(rest, "--seed").unwrap_or(1) as u64,
                ),
                _ => Err("fk server <probe|check|bench>".into()),
            }
        }
        "liveness" => {
            let rest = &a[1..];
            let (engine, tape, at) = common(rest)?;
            cmd::liveness::run(
                &engine,
                tape,
                at,
                cmd::liveness::LivenessOpts {
                    reference: flag(rest, "--reference").map(|s| s.to_string()),
                    also: flag(rest, "--also")
                        .map(|s| s.split(',').map(|x| x.parse().expect("--also a,b,c")).collect())
                        .unwrap_or_default(),
                },
            )
        }
        "probe" => {
            let rest = &a[1..];
            let (engine, tape, at) = common(rest)?;
            cmd::probe::run(
                &engine,
                tape,
                at,
                cmd::probe::ProbeOpts {
                    reference: flag(rest, "--reference")
                        .ok_or("fk probe needs --reference CSV (the answer key)")?
                        .to_string(),
                    channel: flag(rest, "--channel").unwrap_or("wetness").to_string(),
                    span: num(rest, "--span").unwrap_or(512) as u32,
                    top: num(rest, "--top").unwrap_or(12) as usize,
                    affine: flag(rest, "--affine").map(|s| {
                        let v: Vec<f64> = s.split(',').map(|x| x.parse().expect("--affine a,b")).collect();
                        (v[0], v[1])
                    }),
                },
            )
        }
        "trace" => {
            let rest = &a[1..];
            let (engine, tape, at) = common(rest)?;
            cmd::trace::run(
                &engine,
                tape,
                at,
                cmd::trace::TraceOpts {
                    reference: flag(rest, "--reference").map(|s| s.to_string()),
                    out: flag(rest, "--out").map(|s| s.to_string()),
                    nth: num(rest, "--nth").unwrap_or(1).max(1) as usize,
                },
            )
        }
        // `watch` and `regen` take `--template` rather than `--tape`, and both
        // choose their own checkpoints from a ladder rather than being told
        // one, so neither goes through `common`. That is a real difference, not
        // an inconsistency to paper over: a harness that measures the watchdog
        // over a window is not the same shape of command as one that reads a
        // trajectory at a checkpoint you name.
        "resync" => {
            let rest = &a[1..];
            let (engine, tape, at) = common(rest)?;
            cmd::resync::run(
                &engine,
                tape,
                at,
                cmd::resync::Opts {
                    reference: flag(rest, "--reference")
                        .ok_or("fk resync needs --reference REC.csv")?
                        .to_string(),
                    tol: flag(rest, "--tol").map(|s| s.parse().unwrap()).unwrap_or(1.0f64),
                    evals: num(rest, "--evals").unwrap_or(400) as usize,
                    window: num(rest, "--window").unwrap_or(80) as usize,
                    seed: num(rest, "--seed").unwrap_or(1) as u64,
                    out: flag(rest, "--out").map(|s| s.to_string()),
                    control_break_tick: num(rest, "--control-break").map(|v| v as usize),
                    control_break_delta: num(rest, "--control-delta").unwrap_or(8) as i32,
                    maxdelta: num(rest, "--maxdelta").unwrap_or(24) as i32,
                    maxspan: num(rest, "--maxspan").unwrap_or(3) as usize,
                    onset: flag(rest, "--onset").map(|s| s.parse().unwrap()).unwrap_or(0.02f64),
                    minstep: num(rest, "--minstep").unwrap_or(0),
                    ctlticks: num(rest, "--ctlticks").unwrap_or(80) as usize,
                    pedals: rest.iter().any(|x| x == "--pedals"),
                    lo: num(rest, "--lo").map(|v| v as usize),
                    hi: num(rest, "--hi").map(|v| v as usize),
                },
            )
        }
        "tree" => {
            let verb = a.get(1).map(|s| s.as_str()).unwrap_or("");
            let rest = &a[2.min(a.len())..];
            let (engine, tape, at) = common(rest)?;
            let ns = |name: &str, dflt: &str| -> Vec<u64> {
                flag(rest, name)
                    .unwrap_or(dflt)
                    .split(',')
                    .map(|v| v.parse().unwrap_or_else(|_| fk::die(format!("{} wants numbers, got {:?}", name, v))))
                    .collect()
            };
            match verb {
                "cost" => cmd::tree::cost(
                    &engine,
                    tape,
                    at,
                    cmd::tree::CostOpts {
                        reps: num(rest, "--reps").unwrap_or(11) as usize,
                        ks: ns("--ks", "1,5,10,20,50,200,1000"),
                        depth: num(rest, "--depth").unwrap_or(50) as usize,
                        seed: num(rest, "--seed").unwrap_or(1) as u64,
                        load_limit: flag(rest, "--load-limit").map(|s| s.parse().unwrap()).unwrap_or(2.0),
                        allow_load: has(rest, "--allow-load"),
                        trace: has(rest, "--trace"),
                    },
                ),
                "exact" => match cmd::tree::exact(
                    &engine,
                    tape,
                    at,
                    cmd::tree::ExactOpts {
                        n: num(rest, "--n").unwrap_or(40) as usize,
                        depths: ns("--depths", "1,2,5,10,25,50").iter().map(|v| *v as usize).collect(),
                        k: num(rest, "--k").unwrap_or(10) as u64,
                        seed: num(rest, "--seed").unwrap_or(1) as u64,
                    },
                )? {
                    true => Ok(()),
                    false => Err("RUNG 0.5 DID NOT PASS -- see the table above".into()),
                },
                "scale" => cmd::tree::scale(
                    &engine,
                    tape,
                    at,
                    cmd::tree::ScaleOpts {
                        servers: num(rest, "--servers").unwrap_or(16) as usize,
                        secs: num(rest, "--secs").unwrap_or(30) as u64,
                        k: num(rest, "--k").unwrap_or(10) as u64,
                        seed: num(rest, "--seed").unwrap_or(1) as u64,
                        load_limit: flag(rest, "--load-limit").map(|s| s.parse().unwrap()).unwrap_or(2.0),
                        allow_load: has(rest, "--allow-load"),
                    },
                ),
                _ => Err("fk tree <cost|exact|scale>".into()),
            }
        }
        "watch" => cmd::watch::run(&a[1..]),
        "regen" => cmd::regen::run(&a[1..]),
        "carrier" => cmd::carrier::run(&a[1..]),
        "ptr" => cmd::ptr::run(&a[1..]),
        "events" => cmd::events::run(&a[1..]),
        x => Err(format!("unknown command {:?}\n\n{}", x, USAGE)),
    }
}

pub fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a str> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(|s| s.as_str())
}
pub fn num(a: &[String], name: &str) -> Option<i64> {
    flag(a, name).map(|v| {
        v.parse()
            .unwrap_or_else(|_| fk::die(format!("{} wants a number, got {:?}", name, v)))
    })
}
pub fn has(a: &[String], name: &str) -> bool {
    a.iter().any(|x| x == name)
}

/// The five engine flags and the checkpoint, parsed once.
///
/// Unknown flags are an ERROR, not a shrug. The old parser panicked on some and
/// silently ignored others depending on which command you were in, so a typo
/// could run a whole measurement against a default you did not mean.
fn common(a: &[String]) -> Result<(Engine, Tape, Checkpoint), String> {
    let tape_path = flag(a, "--tape").ok_or("--tape FILE is required")?;
    let tape = Tape::load(tape_path)?;
    let work = flag(a, "--work").map(PathBuf::from);
    let engine = Engine {
        server: flag(a, "--server")
            .map(PathBuf::from)
            .or_else(|| std::env::var("TM_SERVER").ok().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("/tmp/tmoracle/server")),
        map: PathBuf::from(flag(a, "--map").ok_or("--map FILE is required")?),
        shim: flag(a, "--shim")
            .map(PathBuf::from)
            .or_else(|| std::env::var("FK_SHIM").ok().map(PathBuf::from))
            .or_else(fk::session::default_shim)
            .ok_or("no --shim: pass one, set FK_SHIM, or build tools/search (which produces \
              libforkshim.so)")?,
        work_is_temporary: work.is_none(),
        work: work.unwrap_or_else(Engine::default_work),
    };
    let at = match flag(a, "--at") {
        None => Checkpoint::Fraction(0.5),
        Some(s) => match s.split_once(':') {
            Some(("tick", v)) => Checkpoint::Tick(v.parse().map_err(|_| "--at tick:N")?),
            Some(("clock", v)) => Checkpoint::Clock(v.parse().map_err(|_| "--at clock:N")?),
            Some(("frac", v)) => Checkpoint::Fraction(v.parse().map_err(|_| "--at frac:F")?),
            _ => return Err(format!("--at wants tick:N, clock:N or frac:F, got {:?}", s)),
        },
    };
    Ok((engine, tape, at))
}


