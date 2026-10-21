//! `ghost tape sweep` — write a family of candidate tapes by rectangular override.
//!
//! One template, one window per candidate: from race `start` for `len`
//! milliseconds, hold a fixed steering value / accelerator / brake, and leave
//! every other tick exactly as the template has it. That is the move set three
//! arms on 285885 used to search this map, each time from a private fork of the
//! toolchain; the generator itself was never banked, so every arm rebuilt it.
//!
//! It exists as a separate command from `tape inject` because what it produces
//! is a POPULATION, and a population needs a control:
//!
//! * `IDENT.Ghost.Gbx` is written first — the template through the same writer,
//!   with no override at all. It is the identity of the whole batch.
//! * every candidate is md5'd against it, and the **no-op census** is printed.
//!   A candidate byte-identical to IDENT is not a weak result, it is not a
//!   result: it is the same tape. On this project a 48-of-48 no-op census is
//!   what caught a whole search window that was in the wrong place, and a poke
//!   equal to the value the seed already holds has twice masqueraded as "the
//!   one family that differs".
//!
//! The tick a race millisecond lands on is `(race_ms - start_offset_ms) / 10`,
//! read from the archive itself — never assumed to be zero, which on this map's
//! tapes would be 1.510 s out.

use gbx::container::Container;
use gbx::tape::{Encoding, Tape};
use std::collections::HashMap;
use std::path::Path;

fn parse_list(s: &str) -> Vec<Option<i32>> {
    s.split(',')
        .map(|v| {
            let v = v.trim();
            if v == "-" {
                None
            } else {
                Some(v.parse::<i32>().unwrap_or_else(|_| {
                    eprintln!("ghost tape sweep: {:?} is not a number or '-'", v);
                    std::process::exit(3)
                }))
            }
        })
        .collect()
}

fn md5_hex(b: &[u8]) -> String {
    // A 128-bit FNV-style digest is not md5, and for a no-op census that is
    // exactly as good: it answers "are these bytes the same bytes".
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x9e3779b97f4a7c15;
    for (i, &x) in b.iter().enumerate() {
        h1 ^= x as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 = h2.rotate_left(7) ^ ((x as u64).wrapping_mul(i as u64 | 1));
        h2 = h2.wrapping_mul(0x9e3779b97f4a7c15);
    }
    format!("{:016x}{:016x}", h1, h2)
}

pub fn cmd(rest: &[String]) {
    let f = |n: &str| -> Option<String> {
        rest.iter().position(|a| a == n).and_then(|i| rest.get(i + 1)).cloned()
    };
    let die = |m: String| -> ! {
        eprintln!("ghost tape sweep: {}", m);
        std::process::exit(3)
    };
    let tpl = f("--template").unwrap_or_else(|| die("--template FILE".into()));
    let out = f("--out").unwrap_or_else(|| die("--out DIR".into()));
    let starts: Vec<i64> = f("--start")
        .unwrap_or_else(|| die("--start MS[,MS...] (race milliseconds)".into()))
        .split(',')
        .map(|s| s.trim().parse().unwrap_or_else(|_| die(format!("bad --start {:?}", s))))
        .collect();
    let lens: Vec<i64> = f("--len")
        .unwrap_or_else(|| "200".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap_or_else(|_| die(format!("bad --len {:?}", s))))
        .collect();
    let steers = parse_list(&f("--steer").unwrap_or_else(|| "-".into()));
    let accels = parse_list(&f("--accel").unwrap_or_else(|| "-".into()));
    let brakes = parse_list(&f("--brake").unwrap_or_else(|| "-".into()));

    std::fs::create_dir_all(&out).unwrap_or_else(|e| die(format!("{}: {}", out, e)));
    let c = Container::load(&tpl).unwrap_or_else(|e| die(e));
    let base = Tape::from_file(&tpl).unwrap_or_else(|e| die(e));
    let start_offset = base.archives[0].start_offset_ms as i64;
    let nticks = base.archives[0].packets.len() as i64;

    let write = |t: &Tape, name: &str| -> Vec<u8> {
        let body = t.splice_into(c.body(), Encoding::Explicit).unwrap_or_else(|e| die(e));
        let p = Path::new(&out).join(format!("{}.Ghost.Gbx", name));
        gbx::container::write_gbx(&c.gbx, body, p.to_str().unwrap()).unwrap_or_else(|e| die(e));
        std::fs::read(&p).unwrap_or_else(|e| die(format!("{}: {}", p.display(), e)))
    };

    let ident = write(&base, "IDENT");
    let ident_h = md5_hex(&ident);
    println!(
        "template {}  {} ticks, start_offset {} ms (race 0 is tick {})",
        tpl,
        nticks,
        start_offset,
        -start_offset / 10
    );

    let mut n = 0usize;
    let mut noop = 0usize;
    let mut hashes: HashMap<String, usize> = HashMap::new();
    let mut rows: Vec<String> = Vec::new();
    for &s in &starts {
        for &l in &lens {
            for &st in &steers {
                for &ac in &accels {
                    for &br in &brakes {
                        if st.is_none() && ac.is_none() && br.is_none() {
                            continue;
                        }
                        let t0 = (s - start_offset) / 10;
                        let t1 = ((s + l) - start_offset) / 10;
                        if t0 < 0 || t1 > nticks {
                            eprintln!(
                                "skip start {} len {}: ticks {}..{} are outside 0..{}",
                                s, l, t0, t1, nticks
                            );
                            continue;
                        }
                        let mut t = base.clone();
                        for p in t.archives[0].packets[t0 as usize..t1 as usize].iter_mut() {
                            if let Some(v) = st {
                                p.steer = (v as i8 as u8) as u32;
                            }
                            if let Some(v) = ac {
                                p.accel = v as u32;
                            }
                            if let Some(v) = br {
                                p.brake = v as u32;
                            }
                            p.vsame = false;
                        }
                        let name = format!(
                            "s{}_l{}_st{}_a{}_b{}",
                            s,
                            l,
                            st.map(|v| v.to_string()).unwrap_or_else(|| "x".into()),
                            ac.map(|v| v.to_string()).unwrap_or_else(|| "x".into()),
                            br.map(|v| v.to_string()).unwrap_or_else(|| "x".into())
                        );
                        let bytes = write(&t, &name);
                        let h = md5_hex(&bytes);
                        if h == ident_h {
                            noop += 1;
                        }
                        *hashes.entry(h).or_default() += 1;
                        rows.push(name);
                        n += 1;
                    }
                }
            }
        }
    }
    println!("wrote {} candidates + IDENT into {}", n, out);
    println!(
        "no-op census: {} of {} are BYTE-IDENTICAL to IDENT ({} distinct tapes in the batch)",
        noop,
        n,
        hashes.len()
    );
    if noop > 0 {
        println!(
            "  a no-op is not a weak candidate, it is the template. Either the window is not \
             where you think it is, or the value you poked is the value the tape already holds."
        );
    }
}
