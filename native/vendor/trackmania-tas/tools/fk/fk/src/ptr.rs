//! THE STRUCTURE THAT OWNS THE VEHICLE STATE — finding it, and dereferencing it.
//!
//! Everything else in this crate finds the car by SEARCHING: a window of engine
//! memory is streamed to disk at every instant of a run and then swept for a
//! float triple that matches a position the file already knows. That is a
//! search for something with one right answer, and it costs 1.36 GB of disk and
//! eleven minutes per regeneration.
//!
//! The engine holds a POINTER to the car. This module finds it, and once found
//! a regeneration reads a pointer, dereferences it and gathers 864 bytes.
//!
//! # How it is found, and why the method cannot fool itself
//!
//! One engine run, stopped at the shim's handover, gives two things at once:
//! a **snapshot** of every writable mapping in the server ([`Snapshot::take`],
//! taken while the process is halted, so nothing is torn), and — after the run
//! finishes — **the address of the car**, identified the way `fk carrier` and
//! `gather_fields` already identify it: the copy whose position triple
//! reproduces the recording's own path to a micron AND whose four wheel slots
//! hold live floats. The snapshot is then walked BACKWARDS from that address
//! ([`Snapshot::chains_to`]): every 8-byte slot holding a pointer into the
//! vehicle struct, then every slot pointing at one of those slots, and so on,
//! until a slot lands in the game binary's own writable data — a **static**
//! address, `module + X`, which is the thing a future run can dereference.
//!
//! The scan is an equality test over a snapshot, so it has no threshold to
//! tune and no ranking to trust. What it does have is a coverage question — "no
//! pointer" could mean "no pointer" or "not scanned" — and that is what
//! [`Snapshot::recall_control`] answers: it draws slots that DO hold pointers,
//! hides their values, and asks the same scan to find them again. A negative is
//! only a negative next to that number.

use forkoracle::procmem::{maps, Region};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// THE CHAIN, as measured on this binary.
///
/// `TrackmaniaServer` md5 `0f0f4b25f31f80c60c81404366c95e68`
/// (`date=2026-05-15_18_00 git=128182-0de74ece09e`), found by `fk ptr find` on
/// 2026-08-23 and produced by an INDEPENDENT `find` on four files across three
/// maps, each in its own server process.
///
/// Read it as
///
/// ```text
///     vehicles = *( *(module + 0x1e45148) + 0x148 )   the engine's vehicle array
///     car[k]   = *(vehicles + 8k)         k = 0..3    four vehicle objects
///     state    = car[k] + 0x46c                       the simulated car's vis state
/// ```
///
/// and the position triple `CARRIER.md` anchors on is `state + 0x50`.
///
/// **The root is a singleton.** `module + 0x1e45148` is written by exactly one
/// instruction in the whole binary (`19a96b7 mov %rsi,0x49ba8a(%rip)`) and read
/// nowhere else — a global set once at start-up, which is what a chain wants
/// under it. The array holds four vehicle objects at a stride of 0x1e08, and
/// the same four appear in a second array elsewhere on the heap.
///
/// **TWO VIS STATES PER OBJECT, AND THIS IS THE PART THAT WAS NEARLY MISSED.**
/// Each vehicle object carries two `CSceneVehicleVisState`s, at `+0x46c` and
/// `+0x848`. On a downloaded recording they hold the same state to a micron, so
/// the first version of this chain named `+0x848` and passed everything put to
/// it: three maps, six processes, byte-identical regenerations. On a
/// TRANSPLANTED container — a published TAS ghost whose telemetry record is a
/// stranger's — they separate. **`+0x46c` follows the tape the engine is
/// simulating; `+0x848` follows the record the file carries, 978 m away.** The
/// acceptance refused it and the run fell back to the blind window, which is
/// the system working — but a chain that is right on every file where the
/// record IS the run is not thereby a chain to the car, and only the one file
/// where the two differ could say so.
///
/// Measured with `+0x46c`, four runs of a recording and four of the transplant:
/// the pointer resolved on 7 of 8 (the eighth fell back, and the blind window
/// then failed too — that map's own 1-in-8 decoy rate) and **every file it
/// wrote is byte-identical to the blind window's**. Offering both members
/// instead makes the choice nondeterministic — one run in six picked `+0x848`,
/// which also qualified there, and wrote different bytes — so the default names
/// the simulation's state alone and the comma form stays for calibrating a new
/// build, where the two have to be compared before one is chosen.
///
/// Three other roots reach the same array on all three maps and are kept here
/// because a second witness is what makes the first one a measurement:
///
/// ```text
///     mod+0x1cb97c8:0:+0x368:+0x148#4x8+0x46c,+0x848   a second global
///     mod+0x1e14828:0:+0x1e0:+0x0#4x8+0x46c,+0x848     a third
///     mod+0x1d56e48:0:+0x188                           DO NOT USE -- see below
/// ```
///
/// The last of those was the first chain found and it is the one to avoid.
/// `mod+0x1d56e48` is not a data structure: the function at `f20700` stores
/// `lea -0x38(%rbp)` into it, so the global holds the address of a DEAD STACK
/// FRAME, and the chain works only because that frame's contents are
/// deterministic. It resolved to a vehicle on every map tried, which is exactly
/// why it is written down rather than quietly dropped — a chain can be
/// perfectly reproducible and still be built on nothing.
///
/// This is a CALIBRATION, not a law: another build moves every number in it.
/// Nothing trusts it. Every consumer resolves it and then applies the same
/// acceptance the blind search applies — the state must reproduce the run's own
/// measured path and all four wheel-rotation slots must be live — so a stale
/// chain fails and the caller falls back to searching.
/// The chain that names the car ON THE BINARY THIS PROJECT RUNS.
///
/// `fk ptr check` grades it: "state 0x…e50: 0.000000 m median from the
/// recording's own path (p90 0.000000, worst 0.000000) over 213 paired
/// instants, 4 of 4 wheel slots live -- ACCEPTED".
///
/// The previous default ended `+0x46c` and landed on a BARE POSITION COPY --
/// "0 of 4 wheel slots live … a pointer to it would be a pointer to the wrong
/// thing" -- so the field gather fell back to the blind window every time. The
/// vis state on this build sits at `+0x4e8` in the vehicle object, and every
/// chain `fk ptr find` reports ends there, at depths 2 through 4. Re-derive it
/// with `fk ptr find` after a server upgrade; `measure_anchors` falls back to
/// the search when it stops resolving, so a stale chain costs time and cannot
/// produce a wrong file.
/// The chain to the ANCHOR copy -- the one `measure_anchors` needs.
///
/// `fk ptr check` on this binary: "state 0x…c18: 0.000000 m median from the
/// recording's own path (p90 0.000000, worst 0.000000) over 533 paired
/// instants, 0 of 4 wheel slots live". Zero wheels is correct: this is the
/// bare position copy, and it carries the 3x3 rotation at -36 and WorldVel at
/// +12 that the anchor wants. DEFAULT_CHAIN below names the OTHER copy, the
/// vis state with the wheels, which is what `--carrier` gathers from.
///
/// Re-derive after a server upgrade with `fk ptr find --bare`.
/// Chains to the WHEELED vis state -- what `--carrier` gathers from.
///
/// Separate from CAR_CHAINS because the two gates want different objects. The
/// anchor's acceptance test asks for a trajectory, and a BARE POSITION COPY
/// passes it; the carrier needs the wheels, and reading a bare copy gives it
/// "0 of 4 wheel slots live" and a fall into the blind window.
///
/// Derive one with plain `fk ptr find` (NO --at): its acceptance requires
/// 4 of 4 wheel slots live, so what it reports is by construction a carrier
/// chain. On 126859 that is the entry below, and using it takes the map from
/// 29.07 s to 15.07 s.
pub const CARRIER_CHAINS: &[&str] = &[
    // 126859, "4 of 4 wheel slots live". Several, from DIFFERENT checkpoints,
    // for the same reason CAR_CHAINS carries several per map: any one of them
    // resolves in only some processes (the first was measured at 1 hit in 3),
    // and a miss costs one resolve while a hit saves a blind-window scan.
    "mod+0x1cba348:0:+0x2c0:+0x240:+0x848",
    "mod+0x1d58ef0:0:+0x360:+0x48:+0x258:+0x848",
    "mod+0x1e44510:0:+0xb0:+0x300:+0x3e8:+0x848",
    "mod+0x1e45148:0:+0x198:+0x38:+0x20:+0x848",
    // 286279, from `--ckpt 2600 --bare`. The plain (1e-5) bar refuses this
    // map's car at 0.000501 m -- half a millimetre, over 4719 instants of a
    // 235-second run, which is float drift on a lap 20x longer than the bar
    // was tuned for, not a wrong object. --bare's 1e-3 accepts it and the walk
    // then reports these.
    "mod+0x1d56e48:0:+0x158:+0x848",
    "mod+0x1d56e50:0:+0x180:+0x848",
    "mod+0x1d91b00:0:+0x30:+0x60:+0x848",
    // 126859 again, from `--ckpt 7000 --bare`. Its film escalates to 12
    // parallel attempts and each is a separate process drawing its own layout,
    // so parallelism multiplies the coin flip instead of hedging it -- more
    // chains is the only thing that helps.
    "mod+0x1cba348:0:+0x330:+0x320:+0x848",
    "mod+0x1d56e48:0:+0x260:+0x0:+0x848",
    "mod+0x1d56e50:0:+0x288:+0x0:+0x848",
];

pub const ANCHOR_CHAIN: &str = "mod+0x1d56e48:0:+0x68:+0x8:+0x4e8";

/// EVERY chain that reaches a vehicle state on this binary, shortest first.
///
/// One chain is not enough, and 287431 is why: `DEFAULT_CHAIN` resolves there
/// and its position reads fine, but the orientation at -36 is garbage
/// (`|q|-1 p99.5 is 7.46e32`), because on that map `+0xd8` reaches a different
/// object than it does on 203072. Which pointer walks to the driven car is a
/// property of the RUN, not only of the binary.
///
/// So carry the whole set `fk ptr find` reported -- they all end at `+0x4e8`,
/// the vis state -- and let the self-check that already exists say which one
/// is the car this time. That is still deterministic and still free: half a
/// dozen pointer walks, no memory scanned, and every candidate faces the same
/// structural test. Re-derive the list with `fk ptr find` after a server
/// upgrade.
pub const CAR_CHAINS: &[&str] = &[
    // 203072 and friends: the vis state at +0x4e8.
    "mod+0x1d56e48:0:+0xd8:+0x4e8",
    "mod+0x1d56e48:0:+0x68:+0x8:+0x4e8",
    "mod+0x1d56e50:0:+0x10:+0x28:+0x4e8",
    "mod+0x1cba348:0:+0x238:+0x140:+0x298:+0x4e8",
    "mod+0x1d58ef0:0:+0x360:+0x48:+0x3c8:+0x4e8",
    "mod+0x1e45148:0:+0x198:+0x38:+0x48:+0x4e8",
    "mod+0x1e59460:0:+0x180:+0x328:+0x328:+0x4e8",
    // WHAT THE TWO OUTPUTS ACTUALLY DIFFER BY, since it is not the run.
    // `ghost manifest diff` on a chain-run file against a search-run file:
    //
    //   record bytes_consumed   58655 == 58655
    //   entities                identical (415 / 0 / 416 samples, same deltas)
    //   first_vehicle_sample    raw_hex IDENTICAL, 116 bytes
    //   tape, validation block  identical
    //   0x0911F000 (compressed) 10400 vs 10443 bytes   <-- the only difference
    //
    // The TRAJECTORY IS BYTE-IDENTICAL. Both files oracle to 20.756 with
    // cps=7, and the identity strings (skin, display name) match too. The 43
    // bytes live inside the compressed node's payload, below what the manifest
    // resolves. So the two regenerations are the same run, and the md5
    // difference is metadata rather than physics -- which is why every gate
    // passes on both. Worth knowing before anyone "fixes" the chain to chase
    // a byte match that was never a correctness problem.
    //
    // 287431 -- THESE PASS ABOUT ONE RUN IN FOUR. Measured: 1 of 4 regens hit
    // "path 1693.2 m over the run" (89 s); the other 3 fell through to the
    // memory search (and all produced the same md5 b5d4f44f, so the SEARCH is
    // deterministic and the chains are not). The object they name exists in
    // some processes and not others -- this map's 646 m freefall hands the car
    // between entities, and which entity a given process ends up holding at
    // the checkpoint varies.
    //
    // They are kept because a hit is a 2x speed-up and a miss costs only the
    // resolve attempt, but this map is NOT off the memory search: the fallback
    // still runs three times in four, and cannot be deleted.
    //
    // Derived with `fk ptr find --at search --tick 400`: the memory
    // search LOCATES the car in a live process, and the walk starts from that
    // verified address rather than from a candidate chain. Note the root
    // mod+0x1d56e48 -- the same static slot 203072's working chain starts
    // from, which is what a real answer looks like here.
    "mod+0x1d56e48:0:+0x158:+0x4e8",
    "mod+0x1d56e50:0:+0x180:+0x4e8",
    "mod+0x1e45148:0:+0x148:+0x8:+0x4e8",
    "mod+0x1e59460:0:+0x180:+0x2d8:+0x4e8",
    // ...and the same search at tick 1200 finds a DIFFERENT object, at a
    // positive base offset, whose chains end +0x4c rather than +0x4e8. Both
    // sets are real cars at the tick they were found; the tick-400 ones score
    // "path length 0.0 m" over the whole run, i.e. they are frozen after the
    // handover. These are the later object.
    // 126859's WHEELED vis state, terminal +0x848.
    //
    // Plain `fk ptr find` (no --at) requires 4 of 4 wheel slots live, and on
    // this map it reports "the car is at 0x… 0.000000 m from the recording's
    // own path over 480 instants, 4 of 4 wheel slots live" for this chain.
    // Every other route to 126859 lands on the BARE position copy at +0xdd4 --
    // the search does too -- which satisfies the trajectory gate and leaves the
    // carrier with no wheels (af305d3, 3b7375b). A third terminal offset, and
    // the one the carrier actually needs here.
    "mod+0x1cba348:0:+0x2c0:+0x240:+0x848",
    // THESE CHAINS ARE NOT PER-MAP. Measured across the staged corpus: 126859
    // resolves `mod+0x1e45148:0:+0x1c8:+0xdd4` -- the chain derived on 287431 --
    // and scores "path 2698.7 m over the run" with it. So the list works as a
    // shared pool: a chain derived on one map routinely serves another, which
    // is why banking generously is right and why a map should never be
    // declared chainless until the whole pool has been tried on it.
    //
    //     126859   204 s -> 31 s      (fell back before these were banked)
    //     294446    89 s -> 42.6 s
    //     287431   190 s -> 14.8 s
    //     203072            5.6 s
    //
    // FOUR TICKS, FOUR DIFFERENT OBJECTS -- and banking them all is the fix.
    //
    // `--at search --tick N` at 400, 600, 900 and 1200 finds the car at four
    // distinct addresses with three distinct terminal offsets (+0x4e8, +0xdd4,
    // +0x4c). That is this map's 646 m freefall handing the car between
    // entities: which one a process holds at the checkpoint varies, which is
    // why a single tick's chains passed only 1 run in 4 (9a090c1).
    //
    // There is no need to work out WHICH entity a given process will hold.
    // Bank a chain for each and one of them resolves live every time: measured
    // 4 hits in 4 runs, and the regeneration went 190 s (search) -> 14.8 s.
    // A chain that misses costs only its resolve attempt.
    "mod+0x1e45148:0:+0x1c8:+0xdd4",
    "mod+0x1cb97c8:0:+0x368:+0x1c8:+0xdd4",
    "mod+0x1d56e48:0:+0x78:+0x298:+0xdd4",
    "mod+0x1cba348:0:+0x1b8:+0x80:+0x4e8",
    "mod+0x1d56e48:0:+0x1a8:+0x8:+0x4e8",
    "mod+0x1d56e50:0:+0x1d0:+0x8:+0x4e8",
    "mod+0x1cba348:0:+0x1b8:+0x80:+0x4c",
    "mod+0x1d56e48:0:+0x260:+0x8:+0x4c",
    "mod+0x1d56e50:0:+0x288:+0x8:+0x4c",
    // (Chains "derived with --at anchor" for 287431 were REMOVED. They came
    // from an anchor address of 0x3f8000003f800538 -- 0x3f800000 is the float
    // 1.0 bit pattern, twice -- so the walk was aimed at float data misread as
    // a pointer, and the chains it produced reached an object that does not
    // exist. `--at anchor` now refuses an address outside every writable
    // mapping, so this cannot be repeated silently.)
    // 287431: the SAME roots, a different walk, and the state at +0x6268.
    // This is the evidence that a chain is not a property of the binary
    // alone -- `fk ptr find` on this map reports six chains and every one of
    // them ends +0x6268, where every chain on 203072 ends +0x4e8.
    "mod+0x1d56e48:0:+0x178:+0xc0:+0x6268",
    "mod+0x1d56e50:0:+0x1a0:+0xc0:+0x6268",
    "mod+0x1e45148:0:+0x160:+0xc0:+0x6268",
    "mod+0x1cb97c8:0:+0x368:+0x160:+0xc0:+0x6268",
    "mod+0x1cba348:0:+0x2d8:+0x208:+0xc0:+0x6268",
    "mod+0x1e45178:0:+0x368:+0x160:+0xc0:+0x6268",
];

/// Chains proven for one (server binary, map), learned by `fk ptr find`.
///
/// CACHING A CHAIN IS SAFE, in a way that caching an ADDRESS never was. An
/// address is a heap location and belongs to the process that measured it --
/// that mistake is what `Anchors::pos_delta` used to be, and what made regen a
/// lottery. A chain is a walk from static data and is resolved fresh in every
/// process, so a cached one is either right or it fails loudly.
///
/// `fk ptr find` appends here; `measure_anchors` reads it before falling back
/// to the built-in list above.
pub fn chain_cache_path() -> std::path::PathBuf {
    std::env::temp_dir().join("fk-car-chains.tsv")
}

pub fn chain_cache_key(server: &str, map: &str) -> Option<String> {
    let bin = std::path::Path::new(server).join("TrackmaniaServer");
    let m = std::fs::metadata(&bin).ok()?;
    let mt = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    let map = std::path::Path::new(map).file_name()?.to_string_lossy().to_string();
    Some(format!("{}:{}:{}", mt, m.len(), map))
}

pub fn chain_cache_get(server: &str, map: &str) -> Vec<String> {
    let Some(key) = chain_cache_key(server, map) else { return Vec::new() };
    let Ok(txt) = std::fs::read_to_string(chain_cache_path()) else { return Vec::new() };
    let mut out = Vec::new();
    for l in txt.lines().rev() {
        let mut it = l.split('\t');
        if it.next() == Some(key.as_str()) {
            if let Some(c) = it.next() {
                if !out.iter().any(|x| x == c) {
                    out.push(c.to_string());
                }
            }
        }
    }
    out
}

pub fn chain_cache_put(server: &str, map: &str, chain: &str) {
    use std::io::Write;
    let Some(key) = chain_cache_key(server, map) else { return };
    if chain_cache_get(server, map).iter().any(|c| c == chain) {
        return;
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(chain_cache_path()) {
        let _ = writeln!(f, "{}\t{}", key, chain);
    }
}

pub const DEFAULT_CHAIN: &str = "mod+0x1d56e48:0:+0xd8:+0x4e8";

/// The pre-2026-08-30 chain, kept for reference: it resolves on this binary
/// and names a copy of the car with no wheel data.
pub const LEGACY_CHAIN: &str = "mod+0x1e45148:0:+0x148#4x8+0x46c";

/// A POOL, not an element — and that correction cost two rounds.
///
/// The chain was first written `:+0x8:+0x848`: one fixed array index. It
/// resolved to a live car on six runs across three maps, and then the
/// neighbouring index `+0x18` resolved to a live car too on a seventh. Both
/// cannot be a property of the index: WHICH ELEMENT IS LIVE VARIES BY PROCESS,
/// and a fixed index was a coin that had come up heads six times.
///
/// So a spec names the ARRAY and, since a vehicle object carries more than one
/// vis state, every member offset worth reading:
/// `<chain to the array>#<count>x<stride>+<member>[,+<member>...]`. Every
/// resulting state is gathered and the choice between them is made by the rule
/// that was already making it — the copy whose position reproduces the run's
/// own measured path and whose four wheel slots are live.
///
/// This is the shape of the answer to "find the structure that owns the
/// copies": the structure is an array of four vehicle objects with two vis
/// states each, and what it buys is a search space of eight candidates instead
/// of 1.25 MB.
pub fn parse_pool(spec: &str) -> Option<(String, usize, u64, Vec<i64>)> {
    let (chain, rest) = spec.split_once('#')?;
    let (count, rest) = rest.split_once('x')?;
    let (stride, members) = match rest.find(['+', '-']) {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "+0"),
    };
    // MEMBERS, PLURAL. A vehicle object carries TWO `CSceneVehicleVisState`s
    // -- at +0x46c and +0x848 on this build -- and on a file whose record is
    // its own run they hold the same state to a micron, so which one a search
    // picks is arbitrary. On a TRANSPLANTED container they separate: +0x46c
    // follows the tape the engine is simulating and +0x848 follows the record
    // the file carries, 978 m away. Both are offered and the copy rule chooses,
    // because "which member is the simulation" is a fact about the file, not
    // about the build.
    let members: Vec<i64> =
        members.split(',').map(|m| parse_i64(m)).collect::<Result<Vec<_>, _>>().ok()?;
    Some((chain.to_string(), count.parse().ok()?, parse_i64(stride).ok()? as u64, members))
}

/// Resolve a pool spec to every member state it names, in array order.
///
/// A null or unreadable element is dropped rather than fatal: an array of four
/// slots with two vehicles in it is a normal thing for an engine to have, and
/// the caller grades what comes back.
pub fn resolve_pool(pid: i32, module: u64, spec: &str) -> Result<Vec<u64>, String> {
    let Some((chain, n, stride, members)) = parse_pool(spec) else {
        return resolve(pid, module, spec).map(|a| vec![a]);
    };
    let base = resolve(pid, module, &format!("{}:+0x0", chain))?;
    let mut f = File::open(format!("/proc/{}/mem", pid)).map_err(|e| format!("open mem: {}", e))?;
    let mut out = Vec::new();
    for k in 0..n {
        let at = base + k as u64 * stride;
        let mut b = [0u8; 8];
        if f.seek(SeekFrom::Start(at)).is_err() || f.read_exact(&mut b).is_err() {
            continue;
        }
        let v = u64::from_le_bytes(b);
        if v < 0x1000 {
            continue;
        }
        // AND IT MUST BE READABLE. Two of this array's four slots hold
        // something that is not a vehicle object, and a segment centred on one
        // of them is clipped away to nothing by the mapping bound -- which
        // reads downstream as "the gather came back 4 bytes wide" rather than
        // as a bad pointer. Prove the struct is there before naming it.
        for member in &members {
            let st = (v as i64 + member) as u64;
            let mut probe = [0u8; 8];
            if f.seek(SeekFrom::Start(st)).is_err() || f.read_exact(&mut probe).is_err() {
                continue;
            }
            if f.seek(SeekFrom::Start(st + 0x358)).is_err() || f.read_exact(&mut probe).is_err() {
                continue;
            }
            if !out.contains(&st) {
                out.push(st);
            }
        }
    }
    if out.is_empty() {
        return Err(format!("the pool at {:#x} holds no usable pointer", base));
    }
    Ok(out)
}

/// What a mapping is, for the purpose of "can a future run name this address".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// The game binary's own writable data (`.data`, `.bss`). THE ANSWER: an
    /// address here is `module + X` and X is the same in every process.
    Static,
    /// Writable data of some other mapped file (a shared library).
    Library,
    Heap,
    Stack,
    /// Anonymous memory that is neither the heap nor a thread stack.
    Anon,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Static => "static",
            Kind::Library => "library",
            Kind::Heap => "heap",
            Kind::Stack => "stack",
            Kind::Anon => "anon",
        }
    }
}

/// One mapping, its bytes, and what kind of thing it is.
impl Snapshot {
    /// Is this address inside any writable mapping we captured?
    ///
    /// The cheap "is this an object at all?" test. A chain that misreads float
    /// data as a pointer produces an address in no mapping whatsoever, and
    /// walking to one yields chains to a thing that does not exist.
    pub fn contains(&self, a: u64) -> bool {
        self.chunks
            .iter()
            .any(|c| a >= c.start && a < c.start + c.bytes.len() as u64)
    }
}

pub struct Chunk {
    pub start: u64,
    pub kind: Kind,
    pub path: String,
    pub bytes: Vec<u8>,
    /// Search this chunk for pointers.
    ///
    /// False for the executable's read-only mappings. They are snapshotted
    /// because vtables, RTTI records and class-name strings live there and
    /// naming the structure is half the point — but a pointer to a heap object
    /// cannot be in memory the process cannot write, so scanning them would
    /// only add candidates that cannot exist.
    pub scan: bool,
}

/// Every writable mapping of a stopped server, plus where its module is.
pub struct Snapshot {
    pub pid: i32,
    pub chunks: Vec<Chunk>,
    /// Load address of the main executable, so a static slot can be reported as
    /// `module + X`. The offset X is what transfers between runs; the address
    /// does not (the binary is PIE and the loader moves it).
    pub module: u64,
    pub module_path: String,
    /// Mappings that could not be read, as `(start, len)`. A negative result is
    /// only as good as this list is short, so it is carried rather than logged.
    pub unread: Vec<(u64, u64)>,
    pub bytes: u64,
}

/// A step of a resolved chain: read a pointer at `slot`, which points `delta`
/// bytes below the thing we wanted.
#[derive(Clone, Debug)]
pub struct Step {
    pub slot: u64,
    pub value: u64,
    /// `target - value`: how far into the pointee the thing we were looking for
    /// sits. Zero means the pointer names it exactly.
    pub delta: i64,
}

/// A whole path from a static (or otherwise rooted) slot down to the car.
#[derive(Clone, Debug)]
pub struct Chain {
    /// Outermost first: `steps[0].slot` is the root.
    pub steps: Vec<Step>,
    pub root_kind: Kind,
    /// `root - module` when the root is static.
    pub root_rel: i64,
}

impl Chain {
    /// The chain in the form [`resolve`] takes:
    /// `mod+0xXXXX:o1:o2:...:ok:+f`, where `a = root`, then `a = read64(a + oi)`
    /// for each offset in order, and the answer is `a + f`.
    pub fn spec(&self) -> String {
        let mut s = if self.root_kind == Kind::Static {
            format!("mod{}", hexoff(self.root_rel))
        } else {
            format!("abs{:#x}", self.steps[0].slot)
        };
        // The first read is at the root itself; every later read is at the
        // previous pointee plus however far short of the next slot it landed.
        s.push_str(":0");
        for w in self.steps.windows(2) {
            let (up, down) = (&w[0], &w[1]);
            s.push_str(&format!(":{}", hexoff(down.slot as i64 - up.value as i64)));
        }
        s.push_str(&format!(":{}", hexoff(self.steps.last().unwrap().delta)));
        s
    }

    pub fn depth(&self) -> usize {
        self.steps.len()
    }
}

/// A signed offset as the parser reads it back. `{:+#x}` cannot be used: it
/// formats a negative `i64` as its unsigned two's-complement value, so a chain
/// printed with it resolves to a different address than the one it was found
/// at. Caught by `a_chain_spec_round_trips_through_the_parser`.
pub fn hexoff(v: i64) -> String {
    if v < 0 {
        format!("-{:#x}", v.unsigned_abs())
    } else {
        format!("+{:#x}", v)
    }
}

/// Resolve a chain spec in a LIVE process. This is the whole point of the
/// exercise: no gather, no sweep, one `pread` per step.
///
/// `module` is the live process's load address for the main executable, which
/// the caller reads from `/proc/<pid>/maps` ([`module_base`]) — a spec is
/// module-RELATIVE precisely because that address moves.
pub fn resolve(pid: i32, module: u64, spec: &str) -> Result<u64, String> {
    let mut it = spec.split(':');
    let root = it.next().ok_or("empty chain spec")?;
    let mut a: u64 = if let Some(h) = root.strip_prefix("mod") {
        let v = parse_i64(h)?;
        (module as i64 + v) as u64
    } else if let Some(h) = root.strip_prefix("abs") {
        parse_i64(h)? as u64
    } else {
        return Err(format!("chain root must be mod+0x… or abs0x…, not {:?}", root));
    };
    let parts: Vec<&str> = it.collect();
    if parts.len() < 2 {
        return Err("a chain needs at least one deref and a final adjust".into());
    }
    let mut f = File::open(format!("/proc/{}/mem", pid)).map_err(|e| format!("open mem: {}", e))?;
    for (i, p) in parts.iter().enumerate() {
        let o = parse_i64(p)?;
        if i + 1 == parts.len() {
            // the final adjust is arithmetic, not a read
            return Ok((a as i64 + o) as u64);
        }
        let at = (a as i64 + o) as u64;
        let mut b = [0u8; 8];
        f.seek(SeekFrom::Start(at)).map_err(|e| format!("seek {:#x}: {}", at, e))?;
        f.read_exact(&mut b).map_err(|e| format!("read {:#x}: {}", at, e))?;
        a = u64::from_le_bytes(b);
        if a < 0x1000 {
            return Err(format!("step {} at {:#x} is a null pointer -- the chain is stale", i, at));
        }
        // TRIED AND REVERTED: rejecting a stack address here (a >= 0x7f00...).
        //
        // On 294446 a chain resolves to 0x7ffec82f2a08, which IS in the stack
        // range, and the obvious reading -- "the stack is not a vehicle state"
        // -- is wrong. That address is the carrier window the gather goes on to
        // use; refusing it made the map take 114.98 s and produce NO coverage
        // line at all, against 42.88 s working. The engine really does hand
        // back a stack-resident view of the state on this map.
        //
        // The 30 s blind-window retry that motivated the guard is real, but it
        // is not caused by the address being on the stack.
    }
    unreachable!()
}

fn parse_i64(s: &str) -> Result<i64, String> {
    let s = s.trim();
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let v = if let Some(h) = s.strip_prefix("0x") {
        i64::from_str_radix(h, 16)
    } else {
        s.parse::<i64>()
    }
    .map_err(|e| format!("{:?}: {}", s, e))?;
    Ok(if neg { -v } else { v })
}

/// The load address of the main executable in a live process.
pub fn module_base(pid: i32) -> Option<(u64, String)> {
    let rs = maps(pid);
    let exe = std::fs::read_link(format!("/proc/{}/exe", pid)).ok()?;
    let exe = exe.to_string_lossy().into_owned();
    rs.iter().filter(|r| r.path == exe).map(|r| r.start).min().map(|b| (b, exe))
}

fn classify(r: &Region, exe: &str, exe_last_end: u64) -> Kind {
    if r.path == exe {
        Kind::Static
    } else if r.path == "[heap]" {
        Kind::Heap
    } else if r.path == "[stack]" || r.path.starts_with("[stack:") {
        Kind::Stack
    } else if r.path.is_empty() && r.start == exe_last_end {
        // `.bss` is an anonymous mapping the loader places immediately after
        // the binary's last file-backed page. It is as static as `.data` and a
        // global pointer is at least as likely to live there.
        Kind::Static
    } else if r.path.starts_with('/') {
        Kind::Library
    } else {
        Kind::Anon
    }
}

impl Snapshot {
    /// Read every writable mapping of a STOPPED process.
    ///
    /// Stopped matters: the fork server halts the engine at the shim's handover
    /// and this is taken there, so a pointer and the object it points at are
    /// read from the same instant. Taken from a running process the two can
    /// disagree, and the failure would look like a missing pointer rather than
    /// like a race.
    pub fn take(pid: i32) -> Result<Snapshot, String> {
        let rs = maps(pid);
        if rs.is_empty() {
            return Err(format!("no mappings for pid {} -- is it still alive?", pid));
        }
        let exe = std::fs::read_link(format!("/proc/{}/exe", pid))
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let module = rs.iter().filter(|r| r.path == exe).map(|r| r.start).min().unwrap_or(0);
        let exe_last_end = rs.iter().filter(|r| r.path == exe).map(|r| r.end).max().unwrap_or(0);
        let mut f =
            File::open(format!("/proc/{}/mem", pid)).map_err(|e| format!("open mem: {}", e))?;
        let mut chunks = Vec::new();
        let mut unread = Vec::new();
        let mut bytes = 0u64;
        for r in &rs {
            if !r.perms.starts_with('r') {
                continue;
            }
            // The executable's own read-only pages come along for the RTTI
            // walk; everything else that is not writable is skipped.
            let scan = r.perms.contains('w');
            if !scan && r.path != exe {
                continue;
            }
            if r.path.starts_with("/dev") || r.path.starts_with('[') && r.path != "[heap]"
                && r.path != "[stack]"
            {
                // [vvar] and friends are not memory anybody keeps a car in, and
                // reading them can fail in ways that abort a whole scan.
                if r.path != "[heap]" && r.path != "[stack]" {
                    continue;
                }
            }
            let len = (r.end - r.start) as usize;
            let mut buf = vec![0u8; len];
            if f.seek(SeekFrom::Start(r.start)).is_err() || f.read_exact(&mut buf).is_err() {
                unread.push((r.start, r.end - r.start));
                continue;
            }
            bytes += len as u64;
            chunks.push(Chunk {
                start: r.start,
                kind: classify(r, &exe, exe_last_end),
                path: r.path.clone(),
                bytes: buf,
                scan,
            });
        }
        Ok(Snapshot { pid, chunks, module, module_path: exe, unread, bytes })
    }

    /// Which chunk holds an address.
    pub fn chunk_of(&self, addr: u64) -> Option<&Chunk> {
        self.chunks
            .iter()
            .find(|c| addr >= c.start && addr < c.start + c.bytes.len() as u64)
    }

    pub fn kind_of(&self, addr: u64) -> Option<Kind> {
        self.chunk_of(addr).map(|c| c.kind)
    }

    /// Eight bytes of the snapshot, if they are in it.
    pub fn u64_at(&self, addr: u64) -> Option<u64> {
        let c = self.chunk_of(addr)?;
        let o = (addr - c.start) as usize;
        c.bytes.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    /// A NUL-terminated string in the snapshot.
    pub fn cstr_at(&self, addr: u64) -> Option<String> {
        let c = self.chunk_of(addr)?;
        let o = (addr - c.start) as usize;
        let b = c.bytes.get(o..(o + 256).min(c.bytes.len()))?;
        let n = b.iter().position(|&x| x == 0)?;
        if n == 0 || !b[..n].iter().all(|&x| (0x20..0x7f).contains(&x)) {
            return None;
        }
        Some(String::from_utf8_lossy(&b[..n]).into_owned())
    }

    /// WHAT IS THIS OBJECT? A C++ object with virtual functions starts with a
    /// vtable pointer, the word before the vtable is its `type_info`, and the
    /// word after that `type_info`'s own vtable is its mangled name — so an
    /// address in the heap can name its own class, if the binary was built with
    /// RTTI. Returns `(vtable module-relative, the name)`.
    ///
    /// A `None` is not "this is not an object": it is "the first word does not
    /// lead to a readable name", which is equally what a plain struct, a
    /// devirtualised class or an RTTI-less build looks like.
    pub fn class_of(&self, obj: u64) -> Option<(i64, String)> {
        let vt = self.u64_at(obj)?;
        let ti = self.u64_at(vt.checked_sub(8)?)?;
        let name = self.cstr_at(self.u64_at(ti + 8)?)?;
        Some((vt as i64 - self.module as i64, name))
    }

    /// Every 8-byte aligned slot whose value lands in one of `targets`.
    ///
    /// A target is a half-open range: `[lo, hi)`. The car is a range because a
    /// pointer to a struct is a pointer to its FIRST byte and the address we
    /// know is 0x50 into it; a slot is a range because a pointer to an object
    /// names the object, not the field inside it that happens to hold the next
    /// pointer.
    pub fn find_pointers(&self, targets: &[(u64, u64)]) -> Vec<(u64, u64)> {
        let mut t: Vec<(u64, u64)> = targets.to_vec();
        t.sort_unstable();
        // Merge, so the binary search below is over disjoint ranges.
        let mut m: Vec<(u64, u64)> = Vec::with_capacity(t.len());
        for (lo, hi) in t {
            match m.last_mut() {
                Some(l) if lo <= l.1 => l.1 = l.1.max(hi),
                _ => m.push((lo, hi)),
            }
        }
        let (min, max) = match (m.first(), m.last()) {
            (Some(a), Some(b)) => (a.0, b.1),
            _ => return Vec::new(),
        };
        let hit = |v: u64| -> bool {
            if v < min || v >= max {
                return false;
            }
            match m.binary_search_by(|r| {
                if v < r.0 {
                    std::cmp::Ordering::Greater
                } else if v >= r.1 {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            }) {
                Ok(_) => true,
                Err(_) => false,
            }
        };
        let nthr = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(64);
        let mut out: Vec<(u64, u64)> = Vec::new();
        std::thread::scope(|s| {
            let mut hs = Vec::new();
            for c in self.chunks.iter().filter(|c| c.scan) {
                let words = c.bytes.len() / 8;
                let per = words.div_ceil(nthr.max(1));
                if per == 0 {
                    continue;
                }
                for w0 in (0..words).step_by(per) {
                    let w1 = (w0 + per).min(words);
                    let hitr = &hit;
                    hs.push(s.spawn(move || {
                        let mut v: Vec<(u64, u64)> = Vec::new();
                        for w in w0..w1 {
                            let val = u64::from_le_bytes(
                                c.bytes[w * 8..w * 8 + 8].try_into().unwrap(),
                            );
                            if hitr(val) {
                                v.push((c.start + w as u64 * 8, val));
                            }
                        }
                        v
                    }));
                }
            }
            for h in hs {
                out.extend(h.join().unwrap_or_default());
            }
        });
        out.sort_unstable();
        out
    }

    /// THE POSITIVE CONTROL for a negative result.
    ///
    /// "Nothing points at the car" and "the scan cannot see the slot that
    /// does" are the same output. So: draw `n` slots that DO hold a pointer
    /// into snapshotted memory, ask [`find_pointers`] for exactly those values,
    /// and report how many come back. A recall below 100 % is a bug in the
    /// scan; a recall of 100 % is what makes an empty result mean something.
    pub fn recall_control(&self, n: usize) -> (usize, usize) {
        let mut planted: Vec<(u64, u64)> = Vec::new();
        // A cheap deterministic spread: walk the chunks and take every
        // (len/n)-th word that happens to hold a pointer into the snapshot.
        'outer: for c in self.chunks.iter().filter(|c| c.scan) {
            let words = c.bytes.len() / 8;
            let step = (words / (n.max(1) / self.chunks.len().max(1) + 1)).max(1);
            for w in (0..words).step_by(step) {
                let v = u64::from_le_bytes(c.bytes[w * 8..w * 8 + 8].try_into().unwrap());
                if self.chunk_of(v).is_some() {
                    planted.push((c.start + w as u64 * 8, v));
                    if planted.len() >= n {
                        break 'outer;
                    }
                }
            }
        }
        if planted.is_empty() {
            return (0, 0);
        }
        let targets: Vec<(u64, u64)> = planted.iter().map(|(_, v)| (*v, *v + 1)).collect();
        let found = self.find_pointers(&targets);
        let set: std::collections::HashSet<u64> = found.iter().map(|(s, _)| *s).collect();
        let ok = planted.iter().filter(|(s, _)| set.contains(s)).count();
        (ok, planted.len())
    }

    /// Walk backwards from `target` (a range: the vehicle struct) to every
    /// chain of at most `depth` pointers whose root is a STATIC address.
    ///
    /// `maxoff` is how far into an object a pointer to it may land — the
    /// classic pointer-path bound. It is a bound on the SEARCH, not a
    /// threshold on a measurement: every chain it returns is then resolved and
    /// checked against the engine, and one that does not land on the car is
    /// discarded rather than ranked.
    pub fn chains_to(
        &self,
        target: (u64, u64),
        depth: usize,
        maxoff: u64,
        cap: usize,
    ) -> Vec<Chain> {
        // BFS level by level. `parent[slot]` = (the step that got here, the
        // slot one level down), so a chain can be reconstructed by walking in.
        let mut parent: HashMap<u64, (Step, Option<u64>)> = HashMap::new();
        let mut frontier: Vec<u64> = Vec::new();
        let mut out: Vec<Chain> = Vec::new();
        let mut level_targets: Vec<(u64, u64)> = vec![target];
        for _ in 0..depth {
            let hits = self.find_pointers(&level_targets);
            if hits.is_empty() {
                break;
            }
            let mut next: Vec<u64> = Vec::new();
            for (slot, value) in hits.iter().copied() {
                if parent.contains_key(&slot) {
                    continue;
                }
                // Which target did it hit? The nearest one at or above the
                // value; for level 0 that is the car, for later levels the slot
                // one level down.
                let down = if frontier.is_empty() {
                    None
                } else {
                    frontier
                        .iter()
                        .copied()
                        .filter(|d| *d >= value && *d - value <= maxoff)
                        .min()
                };
                let base = down.unwrap_or(target.0);
                let step = Step { slot, value, delta: base as i64 - value as i64 };
                parent.insert(slot, (step, down));
                next.push(slot);
                if self.kind_of(slot) == Some(Kind::Static) {
                    out.push(self.chain_from(slot, &parent));
                    if out.len() >= cap {
                        return out;
                    }
                }
            }
            frontier = next;
            level_targets = frontier
                .iter()
                .map(|s| (s.saturating_sub(maxoff), *s + 1))
                .collect();
            if level_targets.is_empty() {
                break;
            }
        }
        out
    }

    /// Same walk, but reporting EVERY slot found at level 1 whatever its kind —
    /// the raw evidence that a pointer to the car exists at all.
    pub fn direct_pointers(&self, target: (u64, u64)) -> Vec<(u64, u64, Kind)> {
        self.find_pointers(&[target])
            .into_iter()
            .map(|(s, v)| (s, v, self.kind_of(s).unwrap_or(Kind::Anon)))
            .collect()
    }

    fn chain_from(&self, root: u64, parent: &HashMap<u64, (Step, Option<u64>)>) -> Chain {
        let mut steps: Vec<Step> = Vec::new();
        let mut cur = Some(root);
        while let Some(s) = cur {
            let (step, down) = parent.get(&s).unwrap().clone();
            steps.push(step);
            cur = down;
        }
        let kind = self.kind_of(root).unwrap_or(Kind::Anon);
        Chain {
            root_rel: root as i64 - self.module as i64,
            root_kind: kind,
            steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_spec_round_trips_through_the_parser() {
        // A two-deep chain: root at module+0x1000 holding a pointer 0x20 short
        // of the next slot, which holds a pointer 0x50 into the car.
        let ch = Chain {
            steps: vec![
                Step { slot: 0x1000, value: 0x9000, delta: 0 },
                Step { slot: 0x9020, value: 0x40000, delta: -0x50 },
            ],
            root_kind: Kind::Static,
            root_rel: 0x1000,
        };
        assert_eq!(ch.spec(), "mod+0x1000:0:+0x20:-0x50");
        // and every offset in it reads back as the number it was printed for
        for (s, want) in [("+0x20", 0x20i64), ("-0x50", -0x50)] {
            assert_eq!(parse_i64(s).unwrap(), want);
        }
    }

    #[test]
    fn the_parser_reads_every_form_an_offset_can_take() {
        assert_eq!(parse_i64("0x10").unwrap(), 16);
        assert_eq!(parse_i64("-0x10").unwrap(), -16);
        assert_eq!(parse_i64("+0x10").unwrap(), 16);
        assert_eq!(parse_i64("10").unwrap(), 10);
        assert!(parse_i64("nonsense").is_err());
    }
}

/// The clock bias for one (server binary, map).
///
/// CACHING THIS WAS MEASURED SLOWER ONCE, and that measurement is no longer
/// about this code. It was taken when the bias scan shared its forked engine
/// with the anchor measurement that followed it, so skipping the scan cost
/// more sharing than it saved work (cold 58.97 s against warm 81.94 s). The
/// anchor server is gone -- the chain list is static and the clean run
/// resolves it in its own process -- so the bias measurement is now a server
/// started for nothing else, ~2 s of a ~6 s run.
///
/// The value is a property of the pair and does not drift: every attempt on
/// 203072 measures +2200. It is NOT a memory address, so unlike the old
/// calibration cache there is nothing here that can be valid in one process
/// and wrong in the next. A stale entry shows up immediately as a coverage
/// hole ("N recorded instants INSIDE THE RACE have no engine instant"), which
/// aborts the run rather than writing a bad file.
pub fn bias_cache_get(server: &str, map: &str) -> Option<i64> {
    let key = chain_cache_key(server, map)?;
    let txt = std::fs::read_to_string(std::env::temp_dir().join("fk-clock-bias.tsv")).ok()?;
    txt.lines().rev().find_map(|l| {
        let mut it = l.split('\t');
        (it.next()? == key).then(|| it.next()?.parse().ok())?
    })
}

pub fn bias_cache_put(server: &str, map: &str, bias: i64) {
    use std::io::Write;
    let Some(key) = chain_cache_key(server, map) else { return };
    if bias_cache_get(server, map) == Some(bias) {
        return;
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::temp_dir().join("fk-clock-bias.tsv"))
    {
        let _ = writeln!(f, "{}\t{}", key, bias);
    }
}
