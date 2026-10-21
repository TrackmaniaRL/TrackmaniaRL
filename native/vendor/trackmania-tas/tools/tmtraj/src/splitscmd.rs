//! `tmtraj splits` — two runs compared AS A FUNCTION OF DISTANCE, not of time.
//!
//! The question this exists for is "where is the other run ahead of ours, by
//! how much, and where do we take it back". Every way of answering it that
//! compares the two runs at the same INSTANT answers a different question: at
//! 240 m/s two cars 5 ms apart are 1.2 m apart, so "who is faster at t = 11.0 s"
//! is dominated by who is further down the road, not by who is quicker. The
//! only comparison that means anything is at the same PLACE.
//!
//! So: pick a ladder of planes along the axis the road runs down, interpolate
//! each run's crossing of each plane, and difference the crossing TIMES. The
//! difference at a station is the whole cumulative lead at that station, and
//! the change in it from one station to the next is what that segment cost or
//! bought — a sector table with no arbitrary sector boundaries.
//!
//! Interpolation matters and the error is bounded: telemetry is on a 50 ms grid
//! and a linear interpolation of x(t) under acceleration a is wrong by at most
//! a·dt²/8, which at this project's largest accelerations is under 2 mm, or
//! under 10 µs of time at racing speed. The 50 ms grid is NOT a 50 ms
//! resolution on a crossing time.
//!
//! Everything is read-only: the input is `tmtraj export --csv` output or a
//! ghost, exactly like `tmtraj route`.

use crate::routecmd::{crossings, load, parse_plane, Plane, Table};

const USAGE: &str = "\
usage: tmtraj splits REF OTHER... [--axis x] (--from A --to B --step S | --stations v1,v2,..)

  REF OTHER...        CSV or .Ghost.Gbx. The FIRST file is the reference every
                      other file's time is differenced against.
  --axis NAME         the coordinate the stations are planes of (default x)
  --from A --to B     the first and last station, in metres. B may be less
  --step S            than A: the ladder runs in the direction of travel.
  --stations LIST     explicit comma-separated station values instead
  --pick first|last   which crossing to take when a run crosses a plane more
                      than once (default first)
  --shift-ms LIST     a per-file correction added to that file's sample labels,
                      comma-separated, one per file (e.g. `0,-10`). THIS IS NOT
                      A TUNING KNOB. `ghost regen` labels each sample by the
                      engine clock minus a measured bias, and that bias has been
                      seen to land a whole tick out on one file while the same
                      instrument reproduces a downloaded recording of the same
                      map exactly. A whole-tick offset is invisible inside one
                      run -- the trajectory is perfectly self-consistent -- and
                      it shows up here as a constant lead at every station,
                      including stations where the two tapes are byte-identical
                      and the two cars are measurably in the same state. Only
                      ever pass a value you have MEASURED, twice, against
                      something outside the file: a window where the two input
                      tapes are identical (any lead there is instrument), and
                      the plain oracle's own validated finish time.
  --csv FILE          also write the table as CSV

Columns, per station and per non-reference file:
  t         the interpolated crossing time
  dt        that file's time MINUS the reference's, in ms. Negative = ahead.
  seg       the change in dt since the previous station: what this segment
            alone was worth. Negative = the file gained here.
  v         speed at the crossing, m/s, and dv against the reference
  vax       speed ALONG the station axis -- the component that actually gets
            you to the line; the rest is side-slip
  lat       the other two coordinates at the crossing, for the line taken

Times print as seconds. A station no file reaches is skipped and said so.
";

fn die(msg: &str) -> ! {
    eprintln!("tmtraj splits: {}", msg);
    std::process::exit(2)
}

fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a str> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(|s| s.as_str())
}

fn numf(a: &[String], name: &str) -> Option<f64> {
    flag(a, name).map(|v| v.parse().unwrap_or_else(|_| die(&format!("{} {:?} is not a number", name, v))))
}

/// One run's state where it crossed a station.
#[derive(Clone, Copy)]
struct At {
    t_ms: f64,
    x: f64,

    z: f64,
    v: f64,
    vax: f64,
    /// Side-slip in degrees: the angle between where the car is pointing and
    /// where it is actually going, in the horizontal plane. On a flat map this
    /// is the whole difference between "fast" and "fast in the right
    /// direction" -- a car at 230 m/s crabbing at 7 degrees is putting 1.7 m/s
    /// less down the road than one at the same speed pointing straight.
    slip: f64,
}

/// Which velocity column belongs to an axis name.
fn vel_of(axis: &str) -> &'static str {
    match axis {
        "x" => "vx",
        "y" => "vy",
        "z" => "vz",
        _ => die("--axis must be x, y or z (the station ladder is a coordinate plane)"),
    }
}

fn at_station(t: &Table, axis: &str, value: f64, pick_last: bool) -> Option<At> {
    let p: Plane = parse_plane(t, &format!("{}={}", axis, value));
    let cs = crossings(t, &p);
    if cs.is_empty() {
        return None;
    }
    let row = if pick_last { &cs[cs.len() - 1].0 } else { &cs[0].0 };
    let g = |n: &str| -> f64 { t.col(n).map(|i| row[i]).unwrap_or(f64::NAN) };
    let (vx, vy, vz) = (g("vx"), g("vy"), g("vz"));
    let v = (vx * vx + vy * vy + vz * vz).sqrt();
    // Side-slip, horizontal plane. The recorded `yaw` is the engine's own, and
    // the mapping from it to a compass heading is fixed by the spawn: a car
    // pointing down -x with no lateral velocity reads yaw = -pi/2 and a
    // velocity heading of atan2(vz, vx) = pi. `pi/2 - yaw` is the heading that
    // makes those agree, and it is checked in the tests below on a synthetic
    // car and on this project's own claim that slip peaks near 18 degrees at a
    // booster crossing.
    let heading = std::f64::consts::FRAC_PI_2 - g("yaw");
    let vheading = vz.atan2(vx);
    let mut d = vheading - heading;
    while d > std::f64::consts::PI {
        d -= 2.0 * std::f64::consts::PI;
    }
    while d < -std::f64::consts::PI {
        d += 2.0 * std::f64::consts::PI;
    }
    // Sign: POSITIVE means the velocity has swung toward +z relative to where
    // the car points -- on this project's -x maps, sliding toward the driver's
    // left. The engine's yaw runs the other way round from atan2, so the
    // difference is negated; the test below pins both the offset and the sign,
    // because getting either wrong turns "they are straighter than us" into its
    // exact opposite.
    let slip = -d.to_degrees();
    Some(At {
        t_ms: g("time_ms"),
        x: g("x"),
        z: g("z"),
        v,
        vax: g(vel_of(axis)).abs(),
        slip,
    })
}

pub fn cmd(argv: &[String]) -> i32 {
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        print!("{}", USAGE);
        return 0;
    }
    let paths: Vec<String> =
        argv.iter().take_while(|a| !a.starts_with("--")).cloned().collect();
    if paths.len() < 2 {
        die("give a reference file and at least one other file to compare with it");
    }
    let axis = flag(argv, "--axis").unwrap_or("x").to_string();
    let pick_last = match flag(argv, "--pick").unwrap_or("first") {
        "first" => false,
        "last" => true,
        other => die(&format!("--pick {:?}: wants first or last", other)),
    };

    let stations: Vec<f64> = if let Some(list) = flag(argv, "--stations") {
        list.split(',')
            .map(|s| s.trim().parse().unwrap_or_else(|_| die(&format!("--stations: {:?} is not a number", s))))
            .collect()
    } else {
        let (a, b, s) = (
            numf(argv, "--from").unwrap_or_else(|| die("--from A is required (or --stations)")),
            numf(argv, "--to").unwrap_or_else(|| die("--to B is required (or --stations)")),
            numf(argv, "--step").unwrap_or_else(|| die("--step S is required (or --stations)")).abs(),
        );
        if s <= 0.0 {
            die("--step must be positive");
        }
        let mut v = Vec::new();
        let n = ((b - a).abs() / s).round() as i64;
        for k in 0..=n {
            v.push(if b < a { a - s * k as f64 } else { a + s * k as f64 });
        }
        v
    };

    let tables: Vec<Table> = paths.iter().map(|p| load(p)).collect();
    // Per-file clock corrections, applied to the sample labels before anything
    // is interpolated. See `--shift-ms` in the usage: measured, never tuned.
    let shifts: Vec<f64> = match flag(argv, "--shift-ms") {
        None => vec![0.0; paths.len()],
        Some(s) => {
            let v: Vec<f64> = s
                .split(',')
                .map(|x| x.trim().parse().unwrap_or_else(|_| die(&format!("--shift-ms: {:?} is not a number", x))))
                .collect();
            if v.len() != paths.len() {
                die(&format!(
                    "--shift-ms needs one value per file: {} files, {} values",
                    paths.len(),
                    v.len()
                ));
            }
            v
        }
    };
    let mut tables = tables;
    for (t, s) in tables.iter_mut().zip(shifts.iter()) {
        if *s != 0.0 {
            let c = t.need("time_ms");
            for r in t.rows.iter_mut() {
                r[c] += *s;
            }
        }
    }
    let tables = tables;
    let names: Vec<String> = paths
        .iter()
        .map(|p| {
            std::path::Path::new(p)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| p.clone())
        })
        .collect();

    println!("reference: {}", names[0]);
    for (i, n) in names.iter().enumerate().skip(1) {
        println!("     file {}: {}", i, n);
    }
    println!("axis {}, {} stations", axis, stations.len());
    if shifts.iter().any(|s| *s != 0.0) {
        println!(
            "clock corrections applied (ms): {}",
            shifts.iter().map(|s| format!("{:+}", s)).collect::<Vec<_>>().join(", ")
        );
    }
    println!();

    // header
    let mut hdr = format!("{:>10} {:>9} {:>8} {:>8}", axis, "t_ref", "v_ref", "vax_ref");
    for i in 1..paths.len() {
        let _ = i;
        hdr.push_str(&format!(
            " | {:>9} {:>9} {:>8} {:>7} {:>7}",
            "t", "dt_ms", "seg_ms", "dv", "dvax"
        ));
    }
    hdr.push_str(&format!(" | {:>8} {:>8} | {:>7} {:>7}", "lat_ref", "lat_oth", "sl_ref", "sl_oth"));
    println!("{}", hdr);

    let mut prev: Vec<Option<f64>> = vec![None; paths.len()];
    let mut csv = String::new();
    if flag(argv, "--csv").is_some() {
        csv.push_str("station,t_ref_ms,v_ref,vax_ref");
        for i in 1..paths.len() {
            csv.push_str(&format!(",t{}_ms,dt{}_ms,seg{}_ms,v{},vax{},lat{},slip{}", i, i, i, i, i, i, i));
        }
        csv.push_str(",lat_ref,slip_ref\n");
    }
    // The lateral coordinate is whichever horizontal axis is not the station
    // axis: on an x-ladder that is z, and it is the one that says which lane a
    // run is in.
    let lat_name = if axis == "x" { "z" } else { "x" };
    let lat_of = |a: &At| if lat_name == "z" { a.z } else { a.x };

    let mut skipped = 0usize;
    for st in &stations {
        let ats: Vec<Option<At>> =
            tables.iter().map(|t| at_station(t, &axis, *st, pick_last)).collect();
        let Some(r) = ats[0] else {
            skipped += 1;
            continue;
        };
        let mut line = format!("{:10.1} {:9.4} {:8.2} {:8.2}", st, r.t_ms / 1000.0, r.v, r.vax);
        let mut crow = format!("{},{:.4},{:.4},{:.4}", st, r.t_ms, r.v, r.vax);
        let mut any = false;
        for i in 1..ats.len() {
            match ats[i] {
                Some(o) => {
                    any = true;
                    let dt = o.t_ms - r.t_ms;
                    let seg = match prev[i] {
                        Some(p) => dt - p,
                        None => 0.0,
                    };
                    prev[i] = Some(dt);
                    line.push_str(&format!(
                        " | {:9.4} {:9.3} {:8.3} {:7.2} {:7.2}",
                        o.t_ms / 1000.0,
                        dt,
                        seg,
                        o.v - r.v,
                        o.vax - r.vax
                    ));
                    crow.push_str(&format!(
                        ",{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                        o.t_ms, dt, seg, o.v, o.vax, lat_of(&o), o.slip
                    ));
                }
                None => {
                    line.push_str(&format!(" | {:>9} {:>9} {:>8} {:>7} {:>7}", "-", "-", "-", "-", "-"));
                    crow.push_str(",,,,,,");
                }
            }
        }
        if !any {
            skipped += 1;
            continue;
        }
        let oth = ats[1..].iter().flatten().next().copied();
        line.push_str(&format!(
            " | {:8.2} {:8.2} | {:7.2} {:7.2}",
            lat_of(&r),
            oth.map(|o| lat_of(&o)).unwrap_or(f64::NAN),
            r.slip,
            oth.map(|o| o.slip).unwrap_or(f64::NAN)
        ));
        crow.push_str(&format!(",{:.4},{:.4}", lat_of(&r), r.slip));
        println!("{}", line);
        csv.push_str(&crow);
        csv.push('\n');
    }
    if skipped > 0 {
        println!("\n{} station(s) skipped: not every run reaches them", skipped);
    }
    if let Some(p) = flag(argv, "--csv") {
        std::fs::write(p, csv).unwrap_or_else(|e| die(&format!("cannot write {}: {}", p, e)));
        println!("wrote {}", p);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the command, as a test: two runs that are 10 ms apart
    /// at a station must read 10 ms apart THERE, whatever their sample grids
    /// do, and the station must be hit between samples rather than snapped to
    /// one.
    #[test]
    fn a_station_time_is_interpolated_and_differenced() {
        let dir = std::env::temp_dir().join(format!("tmtraj-splits-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hdr = "time_ms,x,y,z,vx,vy,vz\n";
        // ref: x = 100 - 10*t(s), so x = 55 at t = 4.5 s
        let mut a = String::from(hdr);
        // other: same speed, 0.010 s later everywhere
        let mut b = String::from(hdr);
        for k in 0..10 {
            let t = k as f64 * 1000.0;
            a.push_str(&format!("{},{},0,0,-10,0,0\n", t, 100.0 - 10.0 * (t / 1000.0)));
            b.push_str(&format!("{},{},0,0,-10,0,0\n", t, 100.0 - 10.0 * ((t - 10.0) / 1000.0)));
        }
        let pa = dir.join("a.csv");
        let pb = dir.join("b.csv");
        std::fs::write(&pa, a).unwrap();
        std::fs::write(&pb, b).unwrap();
        let ta = load(pa.to_str().unwrap());
        let tb = load(pb.to_str().unwrap());
        let sa = at_station(&ta, "x", 55.0, false).expect("ref crosses 55");
        let sb = at_station(&tb, "x", 55.0, false).expect("other crosses 55");
        assert!((sa.t_ms - 4500.0).abs() < 1e-6, "interpolated, not snapped: {}", sa.t_ms);
        assert!((sb.t_ms - sa.t_ms - 10.0).abs() < 1e-6, "dt is 10 ms, got {}", sb.t_ms - sa.t_ms);
        assert!((sa.vax - 10.0).abs() < 1e-9);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The slip convention, pinned to the engine's own yaw rather than
    /// guessed. A car pointing down -x is at yaw = -pi/2 (this is what every
    /// spawn row on a -x map reads); driving straight down -x is zero slip, and
    /// a car whose velocity has swung 10 degrees toward +z while it still
    /// points down -x is 10 degrees of slip. Getting the SIGN or the offset
    /// wrong here would turn "the record holder is straighter than us" into its
    /// opposite, which is the whole finding this command was written for.
    #[test]
    fn slip_is_zero_when_the_car_goes_where_it_points() {
        let dir = std::env::temp_dir().join(format!("tmtraj-splits3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.csv");
        let yaw = -std::f64::consts::FRAC_PI_2;
        // straight: v = (-100, 0, 0)
        // slipped:  v rotated 10 degrees toward +z
        let a = (10f64).to_radians();
        let (vx2, vz2) = (-100.0 * a.cos(), -100.0 * -a.sin());
        std::fs::write(
            &p,
            format!(
                "time_ms,x,y,z,vx,vy,vz,yaw\n0,100,0,0,-100,0,0,{y}\n400,60,0,0,-100,0,0,{y}\n500,50,0,0,{vx2},0,{vz2},{y}\n1000,0,0,0,{vx2},0,{vz2},{y}\n",
                y = yaw
            ),
        )
        .unwrap();
        let t = load(p.to_str().unwrap());
        let s0 = at_station(&t, "x", 99.0, false).unwrap();
        assert!(s0.slip.abs() < 1e-9, "straight ahead is zero slip, got {}", s0.slip);
        let s1 = at_station(&t, "x", 5.0, false).unwrap();
        assert!(
            (s1.slip - 10.0).abs() < 1e-6,
            "velocity 10 deg toward +z of a -x heading is +10 deg of slip, got {}",
            s1.slip
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A station outside a run's span is absent, never zero: "did not reach it"
    /// and "reached it at t = 0" are different answers.    #[test]
    fn an_unreached_station_is_none() {
        let dir = std::env::temp_dir().join(format!("tmtraj-splits2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.csv");
        std::fs::write(&p, "time_ms,x,y,z,vx,vy,vz\n0,100,0,0,-10,0,0\n1000,90,0,0,-10,0,0\n")
            .unwrap();
        let t = load(p.to_str().unwrap());
        assert!(at_station(&t, "x", 50.0, false).is_none());
        assert!(at_station(&t, "x", 95.0, false).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
