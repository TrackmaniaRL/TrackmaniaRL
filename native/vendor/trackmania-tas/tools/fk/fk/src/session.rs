//! Starting a fork server, and the four things every command needs to say.
//!
//! # The god-struct this replaces
//!
//! Every one of `fk`'s ~45 subcommands used to take the same `Cfg`: twenty-four
//! fields (`template map server work shim csv tick ckpt tol mode out every n
//! addr span len tape difftick obs obstag steerdiv diffmag nth`) parsed by one
//! function, of which any given command read four or five. `--obstag` was
//! accepted by the memory-poker; `--difftick` by the tape cutter. A flag being
//! accepted said nothing about whether it did anything, and the only way to
//! find out was to read the command's body.
//!
//! What is actually shared is small and it is all here: which engine, which
//! map, which tape, where to work, and where to stop. Everything else is an
//! argument to one command and is parsed by that command.

use crate::tape::Tape;
use forkoracle::forksrv::{write_key, ForkServer};
use std::path::{Path, PathBuf};

/// Where the engine is and where to work. Five fields, and every command needs
/// all five.
#[derive(Clone, Debug)]
pub struct Engine {
    /// The dedicated-server install (the directory holding `TrackmaniaServer`).
    pub server: PathBuf,
    /// The map to put in `UserData/Maps`.
    ///
    /// For a `.Replay.Gbx` this is **decoration**: the replay carries the whole
    /// map in chunk `0x03093002` and the server simulates that copy. Proven by
    /// validating with an empty Maps directory and getting the identical time.
    /// `ghost map show FILE` says which case a file is.
    pub map: PathBuf,
    /// The `LD_PRELOAD` shim that counts `lroundf` and services the fork
    /// protocol.
    pub shim: PathBuf,
    /// Scratch root. **Per process by default.** A shared one is the single
    /// root cause behind four separate silent-corruption defects in this
    /// project: two runs pointed at one directory swap replays, and the loser
    /// measures another tape's prefix with its own tail patched in — a real,
    /// self-consistent trajectory of a car that drove somewhere else, which
    /// every internal check passes. The directory lock refuses sharing rather
    /// than racing.
    pub work: PathBuf,
    /// Delete `work` on exit. Off when the caller asked for a specific `--work`,
    /// because then they want to look at it.
    pub work_is_temporary: bool,
}

impl Engine {
    pub fn default_work() -> PathBuf {
        PathBuf::from(format!("/tmp/fk/work-{}", std::process::id()))
    }

    /// Fail now, with the reason, rather than inside a forked child where the
    /// only symptom is a bare `ERR`.
    pub fn check(&self) -> Result<(), String> {
        for (what, p) in [
            ("dedicated server", self.server.join("TrackmaniaServer")),
            ("shim", self.shim.clone()),
            ("map", self.map.clone()),
        ] {
            if !p.exists() {
                return Err(format!("no {} at {}", what, p.display()));
            }
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.work_is_temporary {
            let _ = std::fs::remove_dir_all(&self.work);
        }
    }
}

/// Where to stop the simulation.
///
/// The engine has no notion of "tick 2323"; it has an `lroundf` call count. The
/// two are related by a line fitted on three segment maps, and the fit is only
/// ever used to CHOOSE a checkpoint — never to label anything, because the
/// count is not a fixed simulation point (see [`Session::probe_tick`]).
#[derive(Clone, Copy, Debug)]
pub enum Checkpoint {
    /// A raw `lroundf` call count.
    Clock(u64),
    /// A tape tick, converted through the fitted line.
    Tick(i64),
    /// A fraction of the tape's own length.
    Fraction(f64),
}

/// `clock = 36141 + 25.483 * race_ms`, fitted on three segment maps.
///
/// Linear in SIMULATED time, so it transfers between maps. **It is an
/// estimate**: asking for tick 600 on map 2 lands on tick 689. That is fine for
/// choosing a checkpoint and useless for anything else — use
/// [`Session::probe_tick`] to learn where the server actually stopped, and
/// never label a sample from this.
pub fn clock_for_race_ms(ms: i64) -> u64 {
    (36141.0 + 25.483 * ms as f64).max(1000.0) as u64
}

/// The TOTAL `lroundf` count for one full validation of `ghost` on `map`.
///
/// Measured, not fitted, because `--at frac:F` should mean F of the run on any
/// map rather than F of a line fitted on three segment maps of one ghost. Costs
/// one validation with the shim preloaded (~0.5 s); the shim prints the total on
/// stderr and exits.
///
/// A run under load will not give the same total twice — `lroundf` moves in
/// whole chunks of ~62 calls under contention — so this is a placement aid, not
/// a measurement to quote.
pub fn total_clock(
    work: &Path,
    server: &Path,
    map: &Path,
    ghost: &Path,
    shim: &Path,
) -> Result<u64, String> {
    let dir = work.join("clock");
    let replays = dir.join("UserData/Replays");
    let maps = dir.join("UserData/Maps");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&replays).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&maps).map_err(|e| e.to_string())?;
    let link = |t: PathBuf, a: PathBuf| {
        if std::fs::symlink_metadata(&a).is_err() {
            let _ = std::os::unix::fs::symlink(t, a);
        }
    };
    link(server.join("Packs"), dir.join("Packs"));
    link(server.join("TrackmaniaServer"), dir.join("TrackmaniaServer"));
    let m = map.canonicalize().map_err(|e| format!("{}: {}", map.display(), e))?;
    link(m.clone(), maps.join(m.file_name().unwrap()));
    // A COPY, not a symlink, and under a name the server will read. The server
    // ignores anything not named *.Ghost.Gbx / *.Replay.Gbx and reports a bare
    // DNF instead.
    std::fs::copy(ghost, replays.join("clock.Ghost.Gbx")).map_err(|e| e.to_string())?;
    let out = std::process::Command::new("./TrackmaniaServer")
        .args(["/nodaemon", "/validatepath=."])
        .current_dir(&dir)
        .env("LD_PRELOAD", shim.canonicalize().map_err(|e| e.to_string())?)
        .output()
        .map_err(|e| format!("launching the server: {}", e))?;
    let _ = std::fs::remove_dir_all(&dir);
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .find_map(|l| l.strip_prefix("FKSHIM lroundf_total ")?.trim().parse().ok())
        .ok_or_else(|| {
            "the shim did not report an lroundf total -- is LD_PRELOAD reaching the server?"
                .to_string()
        })
}

impl Checkpoint {
    /// Turn a checkpoint into the `lroundf` count the shim stops on.
    ///
    /// `Fraction` MEASURES: it runs one full validation with the shim preloaded
    /// and takes `f` of the real total, so `--at frac:0.95` means 95 % of the
    /// run on any map. The alternative — `f` of the fitted line — is `f` of a
    /// line fitted on three segment maps of one ghost, which is a different
    /// thing that happens to agree on that ghost.
    pub fn to_clock(self, engine: &Engine, tape: &Tape) -> Result<u64, String> {
        Ok(match self {
            Checkpoint::Clock(c) => c,
            Checkpoint::Tick(t) => clock_for_race_ms(t * 10 + tape.start_offset_ms as i64),
            Checkpoint::Fraction(f) => {
                if !(0.0..1.0).contains(&f) {
                    return Err(format!("--at frac:{} is not in [0, 1)", f));
                }
                std::fs::create_dir_all(&engine.work).map_err(|e| e.to_string())?;
                let g = engine.work.join("clockref.Ghost.Gbx");
                tape.write_reference(&g)?;
                let total =
                    total_clock(&engine.work, &engine.server, &engine.map, &g, &engine.shim)?;
                (total as f64 * f) as u64
            }
        })
    }
}

/// A started fork server, plus the tape it is running.
///
/// Holding the two together is not tidiness. The one question no measurement of
/// a simulated trajectory should be trusted without is *is the simulator running
/// the tape I asked about*, and answering it needs both.
pub struct Session {
    pub srv: ForkServer,
    pub tape: Tape,
    pub checkpoint_clock: u64,
}

impl Session {
    /// Start a server, stop it at `at`, and prove it is running `tape`.
    ///
    /// The identity check is not optional and there is no flag to skip it. It
    /// costs one 70 KB read of `/proc/<pid>/mem` and it is the only thing that
    /// can see a swapped replay: two runs sharing a work directory produce a
    /// genuine, self-consistent trajectory of the wrong car, and no internal
    /// consistency test can tell.
    pub fn start(engine: &Engine, tape: Tape, at: Checkpoint) -> Result<Session, String> {
        engine.check()?;
        std::fs::create_dir_all(&engine.work).map_err(|e| e.to_string())?;
        let clock = at.to_clock(engine, &tape)?;
        let refp = engine.work.join("reference.Ghost.Gbx");
        tape.write_reference(&refp)?;
        let key = engine.work.join("key.bin");
        write_key(&key, &tape.steer);
        let srv = ForkServer::start(
            &engine.work.join("srv"),
            &engine.server,
            &engine.map,
            &refp,
            &key,
            &engine.shim,
            clock,
        )?;
        let s = Session { srv, tape, checkpoint_clock: clock };
        s.assert_running_our_tape()?;
        Ok(s)
    }

    /// THE IDENTITY CONTROL. See [`forkoracle::layout::verify_tape`].
    pub fn assert_running_our_tape(&self) -> Result<(), String> {
        forkoracle::layout::verify_tape(
            self.srv.pid(),
            self.srv.base,
            &self.tape.steer,
            &self.tape.accel,
            &self.tape.brake,
        )
    }

    /// The first tick the simulation has NOT yet consumed, asked of the engine
    /// itself.
    ///
    /// Everything downstream depends on this being right, and every cheaper way
    /// of getting it is wrong. Clock- or behaviour-derived estimates were wrong
    /// on 2 of 30. The engine is asked instead: `mprotect` every page the input
    /// array touches to `PROT_NONE` in a throwaway child, catch `SIGSEGV`, and
    /// read `si_addr` — the record it was about to read.
    ///
    /// **A failed probe is a hard abort, never a fallback.** A resume that
    /// rewrites an already-consumed record is a silent no-op that scores exactly
    /// the incumbent's score, so `delta == 0` is accepted and that lineage is
    /// contaminated for free.
    pub fn probe_tick(&mut self) -> Result<usize, String> {
        self.srv.probe_tick()
    }
}

// ---------------------------------------------------------------------------
// The recorder's view of the same five things.
//
// `fk regen` runs the ORIGINAL FILE, never a re-encode of it: its whole job is
// to rewrite a ghost's telemetry while leaving the tape exactly as stored, and
// staging a re-encoded reference would put a second encoder in the path of the
// one operation that must not touch the inputs. So it needs a server started on
// a file rather than on a `Tape`, and `Ctx` is the flat form of `Engine` that
// the recorder threads through its ladder of checkpoints.

#[derive(Clone, Debug)]
pub struct Ctx {
    pub template: String,
    pub map: String,
    pub server: String,
    pub work: String,
    pub shim: String,
    /// 0 = "use the ladder", any other value pins the checkpoint.
    pub ckpt: u64,
}

impl Ctx {
    pub fn from_engine(e: &Engine, template: &str, ckpt: u64) -> Ctx {
        Ctx {
            template: template.to_string(),
            map: e.map.to_string_lossy().into(),
            server: e.server.to_string_lossy().into(),
            work: e.work.to_string_lossy().into(),
            shim: e.shim.to_string_lossy().into(),
            ckpt,
        }
    }
}

/// `clock = 36141 + 25.483 * race_ms`, addressed by tape tick.
pub fn clock_for_tick(tick: i64, start_offset_ms: i32) -> u64 {
    clock_for_race_ms(tick * 10 + start_offset_ms as i64)
}

/// Start a fork server that runs `ghost` verbatim.
///
/// The shim is keyed on the tape's steer sequence so it can find the decoded
/// input array in the engine's heap; the key comes from the same tape the file
/// carries, so a mismatch here means the file's inputs are not what was
/// decoded.
pub fn start_server_on_file(
    c: &Ctx,
    tape: &crate::tape::Tape,
    work: &Path,
    ckpt: u64,
    ghost: &Path,
) -> Result<ForkServer, String> {
    std::fs::create_dir_all(work).map_err(|e| e.to_string())?;
    crate::tape::check_oracle_readable_name(ghost)?;
    let key = work.join("key.bin");
    write_key(&key, &tape.steer);
    ForkServer::start(
        &work.join("srv"),
        Path::new(&c.server),
        Path::new(&c.map),
        ghost,
        &key,
        Path::new(&c.shim),
        ckpt,
    )
}

/// One input record per tape tick from `from` to the end.
pub fn tail_recs(
    steer: &[u8],
    accel: &[u8],
    brake: &[u8],
    from: usize,
) -> Vec<forkoracle::forksrv::Rec> {
    (from..steer.len())
        .map(|t| forkoracle::forksrv::rec_of(steer[t], accel[t], brake[t]))
        .collect()
}

/// Where `libforkshim.so` is, if the caller did not say.
///
/// Beside the `fk` binary first (where a deployment would put it), then
/// `tools/search/target/release/`, which is where `cargo build` leaves it: the
/// shim belongs to the SEARCH workspace, because it and the driver
/// `#[path]`-include one `pred_core.rs` and two copies of that file would be
/// two judges. So it is never in `fk`'s own target directory, and a lookup that
/// only checked there would never find it.
///
/// It was `libfkshim.so` until the crate was renamed `forkshim`; both names are
/// accepted, so a stale build reads as a stale build rather than as a mystery.
pub fn default_shim() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let here = exe.parent()?;
    for dir in [here.to_path_buf(), here.join("../../../search/target/release")] {
        for name in ["libforkshim.so", "libfkshim.so"] {
            let p = dir.join(name);
            if p.exists() {
                return p.canonicalize().ok();
            }
        }
    }
    None
}

/// A PERSISTENT SERVER IS NOT POSSIBLE WITH THIS ENGINE. Settled by reading
/// the shim, after two failed attempts and one wrong explanation.
///
/// The chain of facts, each checkable in the source:
///
///  1. `ForkServer::go` sends 'G' and then calls `self.child.wait()`.
///  2. In the shim, 'G' sets the sampling flags and RETURNS from the command
///     loop. It does not fork -- deliberately: "a forked child is NOT the
///     clean run", measured at 0 of 522 ticks agreeing and up to 2.9 m of
///     divergence. The parent's own /validatepath run is the clean run.
///  3. The parent then finishes its race, sampling as it goes, and exits.
///     The process ending is the RACE ending, not a teardown we chose.
///  4. The command loop is entered from the validator's simulation-binding
///     callback "before the first simulation tick" -- once per process, per
///     validation run.
///
/// So the clean run is single-use by construction. Keeping the process alive
/// would mean re-entering /validatepath for a second race, which is the 1.99 s
/// of engine start-up all over again. There is no non-terminal 'G' to write.
///
/// The consequence for speed: 2.24 s of engine CPU per boot is a hard floor
/// for one regeneration, and the only remaining saving is one boot instead of
/// two (~3.3 s). Under a second is not reachable with this binary.
///
/// TRIED AND REVERTED -- A REUSED SERVER IS NOT A BOOTED SERVER.
///
/// Wiring this into `run_clean_anch` (take at the top, put back at the end)
/// made a 5.9 s regen take 41.8 s AND changed the output: md5 ee875d65 where
/// every correct run gives eb1b8a7c. A fork server's parent sits at its
/// checkpoint holding state the first caller's gather consumes -- handing it
/// to a second caller does not hand over that state, so the second gather
/// re-derives it the slow way and reads something subtly different.
///
/// Pooling the boot needs the parent RESET to the checkpoint between callers,
/// which is a forkoracle change, not a caller-side one. Left here, unused, so
/// the next attempt starts from the measurement.
///
/// A BOOTED ENGINE, KEPT FOR THE LIFE OF THE PROCESS.
///
/// `start_server_on_file` costs ~2 s: it lays out a scratch root, launches
/// `TrackmaniaServer` under the shim, and runs the simulation to the
/// checkpoint. A regen does that TWICE -- once for the clean run and once
/// inside `gather_fields` -- at the same checkpoint, with the same tape,
/// resolving the same address ("the anchor would have said 0x…, +0 bytes").
/// Four of the remaining 5.9 s are those two boots; the gathers themselves are
/// about a second each.
///
/// The parent process sits AT the checkpoint and forks a child per request --
/// that is the whole design of the fork server -- so handing the same parent
/// to a second caller is exactly what it is for. The key is everything that
/// determines the booted state: the server build, the map, the ghost, the
/// checkpoint, and the tape's own input array.
///
/// The pool holds the server until the process exits. `fk regen` is one
/// regeneration per process, so nothing accumulates; a caller that wants the
/// old behaviour passes FK_NO_SERVER_POOL.
pub struct ServerPool {
    entries: Vec<(String, ForkServer)>,
}

impl ServerPool {
    pub const fn new() -> ServerPool {
        ServerPool { entries: Vec::new() }
    }

    fn key(c: &Ctx, tape: &crate::tape::Tape, ckpt: u64, ghost: &Path) -> String {
        let mut h: u64 = 1469598103934665603;
        for b in tape.steer.iter() {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        format!(
            "{}|{}|{}|{}|{:x}",
            c.server,
            c.map,
            ghost.display(),
            ckpt,
            h
        )
    }

    /// Take the booted server for these parameters, booting one if the pool
    /// has none. The caller OWNS it and must `put` it back rather than calling
    /// `.quit()`, or the next caller pays the boot again.
    pub fn take(
        &mut self,
        c: &Ctx,
        tape: &crate::tape::Tape,
        work: &Path,
        ckpt: u64,
        ghost: &Path,
    ) -> Result<ForkServer, String> {
        let k = Self::key(c, tape, ckpt, ghost);
        if let Some(i) = self.entries.iter().position(|(kk, _)| *kk == k) {
            let (_, s) = self.entries.swap_remove(i);
            if std::env::var("FK_POOL_DEBUG").is_ok() {
                eprintln!("POOL HIT  {}", k);
            }
            return Ok(s);
        }
        if std::env::var("FK_POOL_DEBUG").is_ok() {
            eprintln!("POOL MISS {}", k);
        }
        start_server_on_file(c, tape, work, ckpt, ghost)
    }

    pub fn put(&mut self, c: &Ctx, tape: &crate::tape::Tape, ckpt: u64, ghost: &Path, s: ForkServer) {
        if std::env::var("FK_NO_SERVER_POOL").is_ok() {
            s.quit();
            return;
        }
        let k = Self::key(c, tape, ckpt, ghost);
        self.entries.push((k, s));
    }
}

thread_local! {
    pub static SERVER_POOL: std::cell::RefCell<ServerPool> =
        std::cell::RefCell::new(ServerPool::new());
}

/// Borrow a booted server for these parameters, run `f`, and put it back.
pub fn with_server<T>(
    c: &Ctx,
    tape: &crate::tape::Tape,
    work: &Path,
    ckpt: u64,
    ghost: &Path,
    f: impl FnOnce(&mut ForkServer) -> T,
) -> Result<T, String> {
    let mut srv = SERVER_POOL.with(|p| p.borrow_mut().take(c, tape, work, ckpt, ghost))?;
    let out = f(&mut srv);
    SERVER_POOL.with(|p| p.borrow_mut().put(c, tape, ckpt, ghost, srv));
    Ok(out)
}
