//! Driver for the LD_PRELOAD fork server (`fkshim`).
//!
//! Owns the server process, the checkpoint handshake and the per-candidate
//! request/response, and can check every answer against a full from-tick-0
//! validation of the same inputs.

use std::io::{Read, Write};
use std::os::raw::c_int;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

extern "C" {
    fn pipe2(fds: *mut c_int, flags: c_int) -> c_int;
    fn dup2(old: c_int, new: c_int) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn fcntl(fd: c_int, cmd: c_int, arg: c_int) -> c_int;
    fn kill(pid: c_int, sig: c_int) -> c_int;
}
const F_SETFD: c_int = 2;
const O_CLOEXEC: c_int = 0o2000000;
const SIGKILL: c_int = 9;

pub const STRIDE: usize = 32;

// ---------------------------------------------------------------------------
// THE WIRE, BUILT IN ONE PLACE
//
// Every command the shim understands is encoded by exactly one function here.
// The root fork server and a savestate-tree node are two TRANSPORTS (two
// inherited pipes vs one unix socket) and they must never become two protocol
// implementations -- `tmtraj corpus dup` in this project shelled out to a
// command that did not exist and reported the whole corpus clean, and the
// standing rule from it is that one format gets one encoder.

/// `'R'`: fork, rewrite ticks `from..` with `recs`, run to the finish.
pub fn payload_run(from: usize, recs: &[Rec]) -> Vec<u8> {
    let mut p = Vec::with_capacity(5 + recs.len() * 16);
    p.push(b'R');
    p.extend_from_slice(&(recs.len() as u32).to_le_bytes());
    for (i, r) in recs.iter().enumerate() {
        p.extend_from_slice(&((from + i) as u32).to_le_bytes());
        p.extend_from_slice(&r.steer.to_le_bytes());
        p.extend_from_slice(&r.gas.to_le_bytes());
        p.extend_from_slice(&r.brake.to_le_bytes());
    }
    p
}

/// `'P'`: ask the engine which input record it is about to consume.
pub fn payload_probe() -> Vec<u8> {
    b"P".to_vec()
}

/// `'N'`: fork a child that does nothing. The floor under every other cost.
pub fn payload_null_fork() -> Vec<u8> {
    b"N".to_vec()
}

/// How a branch child is told to stop and become a node.
#[derive(Clone, Debug)]
pub struct BranchReq<'a> {
    /// Ticks to rewrite before resuming. **Forward-only**: the caller is
    /// responsible for every index being strictly past the parent's probed
    /// boundary, and `tree::Node::branch` refuses otherwise.
    pub from: usize,
    pub recs: &'a [Rec],
    /// How many more `lroundf` calls the child simulates before it re-enters
    /// the fork server. ~255 to the tick.
    pub stop_after_lroundf: u64,
    /// The driver's listening unix socket. The node connects to it and serves
    /// the same protocol down it.
    pub sock: &'a str,
    /// Where the child writes its per-tick state trace, or `""` for none.
    pub trace_path: &'a str,
    /// Memory to gather per sample: the vehicle state and the race clock.
    pub segs: &'a [(u64, u32)],
    pub sample_stride: u64,
    pub sample_max: u32,
    /// Dedup key `(offset, length)` inside the gathered record.
    pub key: (u32, u32),
}

/// `'B'`: the branch. See the shim's handler for the field order.
pub fn payload_branch(b: &BranchReq) -> Vec<u8> {
    let mut p = Vec::with_capacity(64 + b.sock.len() + b.recs.len() * 16);
    p.push(b'B');
    p.extend_from_slice(&(b.recs.len() as u32).to_le_bytes());
    p.extend_from_slice(&b.stop_after_lroundf.to_le_bytes());
    p.extend_from_slice(&(b.sock.len() as u32).to_le_bytes());
    p.extend_from_slice(b.sock.as_bytes());
    p.extend_from_slice(&(b.trace_path.len() as u32).to_le_bytes());
    p.extend_from_slice(b.trace_path.as_bytes());
    p.extend_from_slice(&(b.segs.len() as u32).to_le_bytes());
    for (a, l) in b.segs {
        p.extend_from_slice(&a.to_le_bytes());
        p.extend_from_slice(&l.to_le_bytes());
    }
    p.extend_from_slice(&b.sample_stride.to_le_bytes());
    p.extend_from_slice(&b.sample_max.to_le_bytes());
    p.extend_from_slice(&b.key.1.to_le_bytes());
    p.extend_from_slice(&b.key.0.to_le_bytes());
    for (i, r) in b.recs.iter().enumerate() {
        p.extend_from_slice(&((b.from + i) as u32).to_le_bytes());
        p.extend_from_slice(&r.steer.to_le_bytes());
        p.extend_from_slice(&r.gas.to_le_bytes());
        p.extend_from_slice(&r.brake.to_le_bytes());
    }
    p
}

/// Write one length-prefixed frame.
pub fn write_frame<W: Write>(w: &mut W, p: &[u8]) -> std::io::Result<()> {
    w.write_all(&(p.len() as u32).to_le_bytes())?;
    w.write_all(p)?;
    w.flush()
}

/// The reply to `'P'`, parsed. `Err` is a HARD ABORT for every caller: a resume
/// that rewrites an already-consumed record is a silent no-op that scores
/// exactly the incumbent's score.
pub fn parse_probe(s: &str) -> Result<usize, String> {
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("TICK ") {
            return rest.trim().parse::<usize>().map_err(|e| e.to_string());
        }
    }
    Err(format!("probe failed: {:?}", s.trim()))
}

/// One tick of engine input, in the engine's own representation.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Rec {
    pub steer: f32,
    pub gas: f32,
    pub brake: f32,
}

pub fn rec_of(steer: u8, accel: u8, brake: u8) -> Rec {
    Rec {
        steer: ((steer as i8) as f32) / 127.0,
        gas: if accel != 0 { 1.0 } else { 0.0 },
        brake: if brake != 0 { 1.0 } else { 0.0 },
    }
}

/// The reference ghost's steer axis, which is all `write_key` needs of a tape.
struct SteerTape<'a> {
    steer: &'a [u8],
}

/// The key file the shim uses to find the decoded input array in its own
/// address space: the reference ghost's steer axis, plus the most distinctive
/// window in it.
pub fn write_key(path: &Path, steer: &[u8]) {
    let f = SteerTape { steer };
    let n = f.steer.len();
    const M: usize = 24;
    let mut t0 = 0usize;
    let mut best = 0usize;
    for t in 0..n.saturating_sub(M) {
        let mut seen = [false; 256];
        let mut d = 0;
        for k in 0..M {
            if !seen[f.steer[t + k] as usize] {
                seen[f.steer[t + k] as usize] = true;
                d += 1;
            }
        }
        if d > best {
            best = d;
            t0 = t;
        }
    }
    let mut out = Vec::with_capacity(12 + 4 * n);
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(t0 as u32).to_le_bytes());
    out.extend_from_slice(&(M as u32).to_le_bytes());
    for t in 0..n {
        out.extend_from_slice(&(((f.steer[t] as i8) as f32) / 127.0).to_le_bytes());
    }
    std::fs::write(path, out).unwrap();
}

pub struct ForkServer {
    child: Child,
    cmd_w: std::fs::File,
    res_r: std::fs::File,
    pub base: u64,
    pub clock: u64,
    /// `this` captured at the validator's simulation-binding callback.
    pub validator_controller: u64,
    /// The simulation object passed in rcx at that same callback.
    pub validation_sim: u64,
    pub dir: PathBuf,
}

/// A pipe whose ends are CLOSE-ON-EXEC.
///
/// THIS MATTERS MORE THAN IT LOOKS. With a plain `pipe()` both ends leak into
/// every process spawned afterwards -- including `fk btraj`, which the search
/// runs as a subprocess, and the TrackmaniaServer that btraj then spawns. A
/// single server that outlives its driver therefore holds the write end of
/// EVERY worker's response pipe, so a worker whose own server has died never
/// sees EOF and blocks forever, and it holds btraj's stderr, so the search's
/// `Command::output()` never returns. Measured: one orphan (cwd
/// `/tmp/fk/stw/srv (deleted)`) holding six of a search's worker pipes plus the
/// stderr of a btraj that had already exited, with the search's main thread
/// parked in `read()` on it for 20 minutes.
///
/// The two ends the server really needs are handed over by `dup2` in
/// `pre_exec`, which clears CLOEXEC on the new descriptor, so nothing is lost.
fn mk_pipe() -> (c_int, c_int) {
    let mut fds = [0i32; 2];
    let r = unsafe { pipe2(fds.as_mut_ptr(), O_CLOEXEC) };
    assert_eq!(r, 0, "pipe2");
    (fds[0], fds[1])
}

/// Claim `dir` for this process, or refuse.
///
/// The lock file holds the owning pid. A stale lock (the owner is gone) is
/// taken over; a live one is an error, because the alternative -- two servers
/// rebuilding one directory -- is a silent wrong answer, not a crash.
fn take_dir_lock(dir: &Path) -> Result<(), String> {
    let _ = std::fs::create_dir_all(dir);
    let lock = dir.join(".fkowner");
    if let Ok(s) = std::fs::read_to_string(&lock) {
        if let Ok(pid) = s.trim().parse::<i32>() {
            if pid != std::process::id() as i32 && unsafe { kill(pid, 0) } == 0 {
                if std::env::var("FK_ALLOW_SHARED_WORK").is_ok() {
                    // Escape hatch used by the reliability tests, which
                    // deliberately collide two runs to prove the SECOND line of
                    // defence (the tape identity check) catches what the lock
                    // would have prevented.
                    eprintln!(
                        "FKDRV: work directory {} is in use by pid {}, continuing because \
                         FK_ALLOW_SHARED_WORK is set",
                        dir.display(),
                        pid
                    );
                } else {
                    return Err(format!(
                        "work directory {} is in use by live pid {} -- give this run its own \
                         --work directory (two servers in one directory silently swap tapes)",
                        dir.display(),
                        pid
                    ));
                }
            }
        }
    }
    std::fs::write(&lock, format!("{}\n", std::process::id())).map_err(|e| e.to_string())
}

impl Drop for ForkServer {
    /// Never leave a server behind. An orphaned TrackmaniaServer holds every
    /// descriptor it inherited -- including, before the CLOEXEC fix, other
    /// workers' pipes -- and idles at a few percent of a core for ever.
    fn drop(&mut self) {
        unsafe {
            kill(self.child.id() as c_int, SIGKILL);
        }
        let _ = self.child.wait();
        let _ = std::fs::remove_file(self.dir.join(".fkowner"));
    }
}

impl ForkServer {
    /// Lay out a worker directory holding exactly the reference ghost, start
    /// the server under the shim, and wait for it to reach the checkpoint.
    pub fn start(
        dir: &Path,
        server_dir: &Path,
        map: &Path,
        ref_ghost: &Path,
        key: &Path,
        shim: &Path,
        ckpt: u64,
    ) -> Result<ForkServer, String> {
        let replays = dir.join("UserData/Replays");
        let maps = dir.join("UserData/Maps");
        // EXCLUSIVE OWNERSHIP OF THE DIRECTORY.
        //
        // `start` rebuilds this tree from scratch, so two processes pointed at
        // one directory destroy each other's server: the loser's replay is
        // replaced by the winner's, and it goes on to simulate SOMEONE ELSE'S
        // tape with its own tail patched in -- a real, self-consistent
        // trajectory of a car that drove somewhere else, which every internal
        // check passes. That is how a hidden default (`/tmp/fk/stw`, used by
        // every `fk btraj` the search runs) corrupted 17-35% of profile
        // refreshes. Refuse to share instead of racing.
        take_dir_lock(dir)?;
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(&replays).map_err(|e| format!("{}: {}", replays.display(), e))?;
        std::fs::create_dir_all(&maps).map_err(|e| e.to_string())?;
        take_dir_lock(dir)?;
        let link = |t: PathBuf, a: PathBuf| -> Result<(), String> {
            match std::os::unix::fs::symlink(t, &a) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
                Err(e) => Err(format!("symlink {}: {}", a.display(), e)),
            }
        };
        link(server_dir.join("Packs"), dir.join("Packs"))?;
        link(
            server_dir.join("TrackmaniaServer"),
            dir.join("TrackmaniaServer"),
        )?;
        link(
            map.canonicalize().map_err(|e| e.to_string())?,
            maps.join(map.file_name().unwrap()),
        )?;
        std::fs::copy(ref_ghost, replays.join("g.Ghost.Gbx")).map_err(|e| e.to_string())?;

        // The validator's own stdout (`ValidatedResult`, the finish time) goes
        // to a file rather than /dev/null: the clean-run sampler ('G') lets the
        // PARENT run to completion, and its printed time is the re-verification
        // of the very run whose telemetry was just recorded.
        let outf = std::fs::File::create(dir.join("stdout.log")).map_err(|e| e.to_string())?;
        let mut c = Command::new("./TrackmaniaServer");
        c.args(["/nodaemon", "/validatepath=."])
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(outf))
            // Enables the one-shot, build-checked hook at the validator's
            // simulation-binding callback. `start_raw` deliberately does not
            // set this: its shimhost tests do not contain that server code.
            .env("FKSHIM_VALIDATOR_CAR", "1");
        let srv = ForkServer::start_raw(dir, c, key, shim, ckpt)?;
        if srv.validator_controller == 0 || srv.validation_sim == 0 {
            return Err(
                "validator simulation callback was not captured; refusing to locate a car by scan"
                    .into(),
            );
        }
        Ok(srv)
    }

    /// Launch an ARBITRARY command under the shim and complete the handshake.
    ///
    /// [`ForkServer::start`] is this plus the dedicated server's own directory
    /// layout. It is split out so the savestate tree can be exercised against a
    /// stand-in host (`shimhost`) with no game, no map and no `.Ghost.Gbx` —
    /// the fork/probe/branch machinery is process mechanics and one array in
    /// memory, and testing it should not require either an engine or a human
    /// recording. One launcher, so the two cannot drift.
    ///
    /// `dir` must already exist and be claimed by the caller. `c` must already
    /// carry its program, arguments, working directory and stdout.
    pub fn start_raw(
        dir: &Path,
        mut c: Command,
        key: &Path,
        shim: &Path,
        ckpt: u64,
    ) -> Result<ForkServer, String> {
        let (cmd_r, cmd_w) = mk_pipe();
        let (res_r, res_w) = mk_pipe();

        // The child's stderr goes to a FILE in its own directory, never to an
        // inherited pipe. Inheriting means a server that outlives its driver
        // holds that pipe open for ever, and a parent doing `Command::output()`
        // on the driver then never sees EOF -- the search's aggregation loop
        // parked in `read()` for the rest of the run, which is what "the search
        // is alive but makes no progress" actually was.
        let logf = std::fs::File::create(dir.join("server.log")).map_err(|e| e.to_string())?;
        c.env(
            "LD_PRELOAD",
            shim.canonicalize().map_err(|e| e.to_string())?,
        )
        .env("FKSHIM_CKPT", ckpt.to_string())
        .env("FKSHIM_KEY", key.canonicalize().map_err(|e| e.to_string())?)
        .env("FKSHIM_CMD_FD", "3")
        .env("FKSHIM_RES_FD", "4")
        .stderr(Stdio::from(logf));
        unsafe {
            c.pre_exec(move || {
                dup2(cmd_r, 3);
                dup2(res_w, 4);
                fcntl(3, F_SETFD, 0);
                fcntl(4, F_SETFD, 0);
                Ok(())
            });
        }
        let child = c.spawn().map_err(|e| e.to_string())?;
        unsafe {
            close(cmd_r);
            close(res_w);
        }
        use std::os::unix::io::FromRawFd;
        let cmd_w = unsafe { std::fs::File::from_raw_fd(cmd_w) };
        let res_r = unsafe { std::fs::File::from_raw_fd(res_r) };

        // From here on the child is OURS: every exit must take it with us, or
        // it becomes an orphan holding inherited descriptors open. `Drop` does
        // that, so an early `return Err` is now safe.
        let mut srv = ForkServer {
            child,
            cmd_w,
            res_r,
            base: 0,
            clock: 0,
            validator_controller: 0,
            validation_sim: 0,
            dir: dir.to_path_buf(),
        };

        let hello = match read_frame(&mut srv.res_r) {
            Some(h) => h,
            None => {
                return Err(format!(
                    "server never reached the checkpoint (pid {})",
                    srv.child.id()
                ))
            }
        };
        let s = String::from_utf8_lossy(&hello).into_owned();
        let ready = parse_ready_full(&s)?;
        srv.base = ready.base;
        srv.clock = ready.clock;
        srv.validator_controller = ready.validator_controller.unwrap_or(0);
        srv.validation_sim = ready.validation_sim.unwrap_or(0);
        Ok(srv)
    }

    /// Fork a child from the checkpoint, rewrite ticks `from..` with `recs`,
    /// run it to the finish, and return the validator's JSON block.
    pub fn run(&mut self, from: usize, recs: &[Rec]) -> String {
        let p = payload_run(from, recs);
        write_frame(&mut self.cmd_w, &p).unwrap();
        match read_frame(&mut self.res_r) {
            Some(v) => String::from_utf8_lossy(&v).into_owned(),
            None => String::new(),
        }
    }

    /// THE BRANCH: fork a child that appends `req.recs`, consumes
    /// `req.stop_after_lroundf` more calls, and then re-enters the fork server
    /// on `req.sock` as a node of its own. Returns the node's pid.
    ///
    /// This call does NOT wait for the node. The server answers `BRANCHED
    /// <pid>` as soon as the fork returns and goes back to serving; the node
    /// announces itself down the socket when it gets there, which is what
    /// `tree::Tree::accept` picks up. Keeping the two apart is what lets a
    /// driver have many branches in flight at once on one server.
    pub fn branch_clock(&mut self, req: &BranchReq) -> Result<i32, String> {
        let mut p = payload_branch(req); p[0] = b'K';
        write_frame(&mut self.cmd_w, &p).map_err(|e| e.to_string())?;
        let v = read_frame(&mut self.res_r).ok_or("branch: no reply")?;
        parse_branched(&String::from_utf8_lossy(&v))
    }
    pub fn branch(&mut self, req: &BranchReq) -> Result<i32, String> {
        let p = payload_branch(req);
        write_frame(&mut self.cmd_w, &p).map_err(|e| e.to_string())?;
        let v = read_frame(&mut self.res_r).ok_or("branch: no reply")?;
        parse_branched(&String::from_utf8_lossy(&v))
    }

    /// Like `run`, but the child also gathers `segs` -- a list of
    /// `(addr, len)` -- into one record every `stride` clock ticks. `key` is
    /// `(offset, length)` within the gathered record: a sample is emitted only
    /// when those bytes change.
    pub fn run_sampled_segs(
        &mut self,
        from: usize,
        recs: &[Rec],
        segs: &[(u64, u32)],
        stride: u64,
        max: u32,
        key: (u32, u32),
    ) -> (String, Vec<u8>) {
        self.run_sampled_segs_ex(from, recs, segs, stride, max, key, 0)
    }

    /// As `run_sampled_segs`, plus a simulated-time budget for the child.
    ///
    /// `budget_lroundf` (0 = unlimited) caps how far the child simulates:
    /// ~255 `lroundf` calls to the tick. A locate probe needs a handful of
    /// ticks, and without a cap the child runs the WHOLE remaining tape --
    /// 43 000 ticks on this project's 440 s record, for six ticks of data.
    /// Set bit 31 of `max` to make the child exit as soon as the sample budget
    /// is spent instead of simulating on in silence.
    pub fn run_sampled_segs_ex(
        &mut self,
        from: usize,
        recs: &[Rec],
        segs: &[(u64, u32)],
        stride: u64,
        max: u32,
        key: (u32, u32),
        budget_lroundf: u32,
    ) -> (String, Vec<u8>) {
        let mut p = Vec::with_capacity(33 + segs.len() * 12 + recs.len() * 16);
        p.push(b'S');
        p.extend_from_slice(&(recs.len() as u32).to_le_bytes());
        p.extend_from_slice(&(segs.len() as u32).to_le_bytes());
        p.extend_from_slice(&stride.to_le_bytes());
        p.extend_from_slice(&max.to_le_bytes());
        p.extend_from_slice(&key.1.to_le_bytes());
        p.extend_from_slice(&key.0.to_le_bytes());
        for (a, l) in segs {
            p.extend_from_slice(&a.to_le_bytes());
            p.extend_from_slice(&l.to_le_bytes());
        }
        for (i, r) in recs.iter().enumerate() {
            p.extend_from_slice(&((from + i) as u32).to_le_bytes());
            p.extend_from_slice(&r.steer.to_le_bytes());
            p.extend_from_slice(&r.gas.to_le_bytes());
            p.extend_from_slice(&r.brake.to_le_bytes());
        }
        p.extend_from_slice(&budget_lroundf.to_le_bytes());
        self.cmd_w
            .write_all(&(p.len() as u32).to_le_bytes())
            .unwrap();
        self.cmd_w.write_all(&p).unwrap();
        self.cmd_w.flush().unwrap();
        let json = match read_frame(&mut self.res_r) {
            Some(v) => String::from_utf8_lossy(&v).into_owned(),
            None => String::new(),
        };
        let blob = read_frame(&mut self.res_r).unwrap_or_default();
        (json, blob)
    }

    /// `run_sampled_segs_ex` plus the 50 ms GRID GATE: sample only at instants
    /// where `(clock_at(gate.0) - gate.2) % gate.1 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_sampled_segs_gated(
        &mut self,
        from: usize,
        recs: &[Rec],
        segs: &[(u64, u32)],
        stride: u64,
        max: u32,
        key: (u32, u32),
        budget_lroundf: u32,
        gate: (u64, u32, u32),
    ) -> (String, Vec<u8>) {
        let mut p = Vec::with_capacity(49 + segs.len() * 12 + recs.len() * 16);
        p.push(b'S');
        p.extend_from_slice(&(recs.len() as u32).to_le_bytes());
        p.extend_from_slice(&(segs.len() as u32).to_le_bytes());
        p.extend_from_slice(&stride.to_le_bytes());
        p.extend_from_slice(&max.to_le_bytes());
        p.extend_from_slice(&key.1.to_le_bytes());
        p.extend_from_slice(&key.0.to_le_bytes());
        for (a, l) in segs {
            p.extend_from_slice(&a.to_le_bytes());
            p.extend_from_slice(&l.to_le_bytes());
        }
        for (i, r) in recs.iter().enumerate() {
            p.extend_from_slice(&((from + i) as u32).to_le_bytes());
            p.extend_from_slice(&r.steer.to_le_bytes());
            p.extend_from_slice(&r.gas.to_le_bytes());
            p.extend_from_slice(&r.brake.to_le_bytes());
        }
        p.extend_from_slice(&budget_lroundf.to_le_bytes());
        p.extend_from_slice(&gate.0.to_le_bytes());
        p.extend_from_slice(&gate.1.to_le_bytes());
        p.extend_from_slice(&gate.2.to_le_bytes());
        self.cmd_w
            .write_all(&(p.len() as u32).to_le_bytes())
            .unwrap();
        self.cmd_w.write_all(&p).unwrap();
        self.cmd_w.flush().unwrap();
        let json = match read_frame(&mut self.res_r) {
            Some(v) => String::from_utf8_lossy(&v).into_owned(),
            None => String::new(),
        };
        let blob = read_frame(&mut self.res_r).unwrap_or_default();
        (json, blob)
    }

    /// THE CLEAN-RUN SAMPLER. Arm sampling in the PARENT and let it run to the
    /// end of the tape with no fork, no resume and no input patch.
    ///
    /// Returns once the server process has exited. The samples are in `path`,
    /// the validator's own output in `<dir>/stdout.log`.
    #[allow(clippy::too_many_arguments)]
    pub fn go(
        &mut self,
        segs: &[(u64, u32)],
        stride: u64,
        max: u32,
        key: (u32, u32),
        gate: (u64, u32, u32),
        path: &str,
    ) -> Result<String, String> {
        let pb = path.as_bytes();
        let mut p = Vec::with_capacity(45 + pb.len() + segs.len() * 12);
        p.push(b'G');
        p.extend_from_slice(&(segs.len() as u32).to_le_bytes());
        p.extend_from_slice(&stride.to_le_bytes());
        p.extend_from_slice(&max.to_le_bytes());
        p.extend_from_slice(&key.1.to_le_bytes());
        p.extend_from_slice(&key.0.to_le_bytes());
        p.extend_from_slice(&gate.0.to_le_bytes());
        p.extend_from_slice(&gate.1.to_le_bytes());
        p.extend_from_slice(&gate.2.to_le_bytes());
        p.extend_from_slice(&(pb.len() as u32).to_le_bytes());
        p.extend_from_slice(pb);
        for (a, l) in segs {
            p.extend_from_slice(&a.to_le_bytes());
            p.extend_from_slice(&l.to_le_bytes());
        }
        self.cmd_w
            .write_all(&(p.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        self.cmd_w.write_all(&p).map_err(|e| e.to_string())?;
        self.cmd_w.flush().map_err(|e| e.to_string())?;
        match read_frame(&mut self.res_r) {
            Some(v) if v == b"GO" => {}
            Some(v) => return Err(String::from_utf8_lossy(&v).into_owned()),
            None => return Err("server closed before acknowledging G".into()),
        }
        let st = self.child.wait().map_err(|e| e.to_string())?;
        Ok(
            std::fs::read_to_string(self.dir.join("stdout.log")).unwrap_or_default()
                + &format!("\nEXIT {:?}", st.code()),
        )
    }

    /// One contiguous window: the common case.
    pub fn run_sampled(
        &mut self,
        from: usize,
        recs: &[Rec],
        addr: u64,
        len: u32,
        stride: u64,
        max: u32,
        key: (u32, u32),
    ) -> (String, Vec<u8>) {
        self.run_sampled_segs(from, recs, &[(addr, len)], stride, max, key)
    }

    /// Arm the watchdog: predicates, the reference line and the memory
    /// segments to watch, sent ONCE. Every later fork inherits them.
    pub fn arm(&mut self, payload: &[u8]) -> String {
        self.cmd_w
            .write_all(&(payload.len() as u32).to_le_bytes())
            .unwrap();
        self.cmd_w.write_all(payload).unwrap();
        self.cmd_w.flush().unwrap();
        match read_frame(&mut self.res_r) {
            Some(v) => String::from_utf8_lossy(&v).into_owned(),
            None => String::new(),
        }
    }

    /// Like `run`, but with the armed predicates evaluated in the child every
    /// tick. Returns the validator's JSON (empty when the child was aborted)
    /// and the raw summary block.
    pub fn run_watched(&mut self, from: usize, recs: &[Rec]) -> (String, Vec<u8>) {
        let mut p = Vec::with_capacity(5 + recs.len() * 16);
        p.push(b'W');
        p.extend_from_slice(&(recs.len() as u32).to_le_bytes());
        for (i, r) in recs.iter().enumerate() {
            p.extend_from_slice(&((from + i) as u32).to_le_bytes());
            p.extend_from_slice(&r.steer.to_le_bytes());
            p.extend_from_slice(&r.gas.to_le_bytes());
            p.extend_from_slice(&r.brake.to_le_bytes());
        }
        self.cmd_w
            .write_all(&(p.len() as u32).to_le_bytes())
            .unwrap();
        self.cmd_w.write_all(&p).unwrap();
        self.cmd_w.flush().unwrap();
        let json = match read_frame(&mut self.res_r) {
            Some(v) => String::from_utf8_lossy(&v).into_owned(),
            None => String::new(),
        };
        let blob = read_frame(&mut self.res_r).unwrap_or_default();
        (json, blob)
    }

    /// Fork a child that does nothing: isolates fork + child-startup cost.
    pub fn null_fork(&mut self) -> String {
        let p = b"N";
        self.cmd_w
            .write_all(&(p.len() as u32).to_le_bytes())
            .unwrap();
        self.cmd_w.write_all(p).unwrap();
        self.cmd_w.flush().unwrap();
        match read_frame(&mut self.res_r) {
            Some(v) => String::from_utf8_lossy(&v).into_owned(),
            None => String::new(),
        }
    }

    /// Ask the engine which tick it is about to consume: the first tick that is
    /// safe to rewrite at this checkpoint.
    /// Arm the sampler's pointer chain (shim command 'C').
    ///
    /// Segment 0's address is recomputed at EVERY sampled instant as
    /// `walk(root, offs) - back`, instead of being fixed when the gather
    /// starts. That is the only way to follow a car the engine reallocates
    /// mid-race: on 287431 the object a one-shot resolve names is left frozen
    /// (y stuck at 20.875, vy holding -277.794 m/s) once the 646 m freefall
    /// entity hands over to the driving one.
    ///
    /// `offs` empty disarms it, which is the default and costs one relaxed
    /// load per sample.
    pub fn arm_chain(&mut self, root: u64, back: u64, offs: &[u64]) -> Result<(), String> {
        let mut p = Vec::with_capacity(21 + offs.len() * 8);
        p.push(b'C');
        p.extend_from_slice(&root.to_le_bytes());
        p.extend_from_slice(&back.to_le_bytes());
        p.extend_from_slice(&(offs.len() as u32).to_le_bytes());
        for o in offs {
            p.extend_from_slice(&o.to_le_bytes());
        }
        write_frame(&mut self.cmd_w, &p).map_err(|e| e.to_string())?;
        let v = read_frame(&mut self.res_r).ok_or("arm_chain: no reply")?;
        let r = String::from_utf8_lossy(&v);
        if r.starts_with("CHAIN at") {
            // The shim walked it once and says where it landed. Surfacing this
            // is the difference between "the chain is wrong" and "the chain is
            // right and the sampling is wrong", which otherwise look identical
            // three layers up ("0 instants sampled").
            println!("  shim: {}", r.trim());
            Ok(())
        } else if r.starts_with("CHAIN") {
            Err(format!("arm_chain: the shim could not walk it: {}", r.trim()))
        } else {
            Err(format!("arm_chain: {}", r))
        }
    }

    pub fn probe_tick(&mut self) -> Result<usize, String> {
        write_frame(&mut self.cmd_w, &payload_probe()).map_err(|e| e.to_string())?;
        let v = read_frame(&mut self.res_r).ok_or("probe: no reply")?;
        parse_probe(&String::from_utf8_lossy(&v))
    }

    pub fn pid(&self) -> i32 {
        self.child.id() as i32
    }

    pub fn quit(mut self) {
        let q = b"Q";
        let _ = self.cmd_w.write_all(&(q.len() as u32).to_le_bytes());
        let _ = self.cmd_w.write_all(q);
        let _ = self.cmd_w.flush();
        // Give it a moment to exit on its own, then make sure. `Drop` runs
        // straight after this and SIGKILLs whatever is left, so a server that
        // ignores 'Q' (or never got to the command loop) cannot survive us.
        for _ in 0..50 {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
                Err(_) => return,
            }
        }
    }
}

/// Read one length-prefixed frame, with a WATCHDOG.
///
/// A blocking read here is a single point of failure for a whole search: if a
/// fork server never answers, the worker thread waiting on it is parked in
/// `anon_pipe_read` forever, and a search whose workers are all parked makes no
/// further progress while still looking alive.
///
/// OBSERVED, twice: a fork server that sails past its `lroundf` checkpoint and
/// settles into the dedicated server's ordinary 1x-realtime loop
/// (`do_sys_poll` + `hrtimer_nanosleep`, ~4% of a core, no children). It never
/// sends READY, and every worker on it blocks. One stalled run out of five cost
/// an entire session's compute before this was diagnosed.
///
/// So: poll with a deadline. A timeout returns None, which every caller already
/// treats as a dead server -- the worker then restarts it rather than waiting
/// on a corpse. The timeout is generous (a legitimate deep-checkpoint start
/// takes a few seconds) because a false timeout costs a restart, while too
/// short a one would thrash.
///
/// GENERIC OVER THE TRANSPORT on purpose. The root fork server talks over two
/// inherited pipes; a **tree node** (`tree::Node`) talks over one unix socket,
/// because a branch child that served its parent's pipe would race the parent
/// for every command byte. Those are two transports and they must not become
/// two protocols: this project's own history has a rule about two
/// implementations of one wire format, so both go through this function and the
/// `payload_*` builders below.
pub fn read_frame<R: Read + std::os::unix::io::AsRawFd>(f: &mut R) -> Option<Vec<u8>> {
    let ms = frame_timeout_ms();
    let t0 = std::time::Instant::now();
    if !wait_readable(f, ms) {
        eprintln!(
            "FKDRV STALL: no frame within {} ms on fd {} (waited {:.1}s) -- treating the fork server as dead",
            ms,
            f.as_raw_fd(),
            t0.elapsed().as_secs_f64()
        );
        return None;
    }
    let mut hdr = [0u8; 4];
    if f.read_exact(&mut hdr).is_err() {
        return None;
    }
    let n = u32::from_le_bytes(hdr) as usize;
    let mut v = vec![0u8; n];
    if n > 0 {
        f.read_exact(&mut v).ok()?;
    }
    Some(v)
}

/// The frame deadline, overridable with `FK_FRAME_TIMEOUT_MS` so reliability
/// studies can make a stall show up in seconds instead of minutes.
pub fn frame_timeout_ms() -> i32 {
    std::env::var("FK_FRAME_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(FRAME_TIMEOUT_MS)
}

pub const FRAME_TIMEOUT_MS: i32 = 120_000;

fn wait_readable<R: std::os::unix::io::AsRawFd>(f: &R, ms: i32) -> bool {
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    extern "C" {
        fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
    }
    let mut p = PollFd {
        fd: f.as_raw_fd(),
        events: 1, /* POLLIN */
        revents: 0,
    };
    let r = unsafe { poll(&mut p, 1, ms) };
    r > 0
}

/// `BRANCHED <pid>`, parsed.
pub fn parse_branched(s: &str) -> Result<i32, String> {
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("BRANCHED ") {
            return rest.trim().parse::<i32>().map_err(|e| e.to_string());
        }
    }
    Err(format!("branch refused: {:?}", s.trim()))
}

/// `READY <base> <clock> [<pid> [<validator-controller> <validation-sim>]]`.
///
/// The final pair is present for real Trackmania servers launched with the
/// validator hook enabled. `shimhost` deliberately omits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ready {
    pub base: u64,
    pub clock: u64,
    pub pid: Option<i32>,
    pub validator_controller: Option<u64>,
    pub validation_sim: Option<u64>,
}

pub fn parse_ready_full(s: &str) -> Result<Ready, String> {
    let mut it = s.split_whitespace();
    if it.next() != Some("READY") {
        return Err(format!("bad handshake: {}", s.trim()));
    }
    let base = it.next().unwrap_or("0").parse().unwrap_or(0);
    let clock = it.next().unwrap_or("0").parse().unwrap_or(0);
    let pid = it.next().and_then(|v| v.parse().ok());
    let validator_controller = it.next().and_then(|v| v.parse().ok()).filter(|v| *v != 0);
    let validation_sim = it.next().and_then(|v| v.parse().ok()).filter(|v| *v != 0);
    if validator_controller.is_some() != validation_sim.is_some() {
        return Err(format!(
            "incomplete validator capture in handshake: {}",
            s.trim()
        ));
    }
    Ok(Ready {
        base,
        clock,
        pid,
        validator_controller,
        validation_sim,
    })
}

/// Compatibility parser for branch/shimhost protocol users that need only the
/// original three fields. Extra validator provenance is parsed and ignored.
pub fn parse_ready(s: &str) -> Result<(u64, u64, Option<i32>), String> {
    let r = parse_ready_full(s)?;
    Ok((r.base, r.clock, r.pid))
}

/// `(time_ms, checkpoints_reached)` from a validator JSON block.
/// The server's "never crossed the line" sentinel, as it appears in a time
/// field: a huge u32 read as an i64.
///
/// **There are two copies of this number and that is deliberate.** The other is
/// `ghost::oracle::BAD_TIME_MS`, in the crate that owns the full server
/// transcript; this crate has no dependencies at all, on purpose, because the
/// LD_PRELOAD shim compiles part of it into a game server. So the copies are
/// pinned against each other by a test instead:
/// `the_two_copies_of_the_never_crossed_sentinel_agree` in `tests/protocol.rs`,
/// where `ghost` is a dev-dependency and the merge that cannot happen in the
/// build happens in the suite.
///
/// It matters here because the fork's reply is what the STATE OBJECTIVE reads
/// to decide whether a candidate finished: a sentinel taken as a value is a
/// finish at 4 294 967.295 seconds, which the gate's top band would accept as
/// "it did the thing and finished" and the search would adopt a wreck as its
/// incumbent. The guard catches it at the bank -- the plain oracle disagrees
/// and the run stops -- but only after the ranking has been wrong.
pub const BAD_TIME_MS: i64 = 4_294_967_000;

/// Read the fork server's validator reply.
///
/// The stream is truncated and has no `FileName`, which is why this is not
/// `ghost::oracle`'s parser. What it shares is the two rules that parser was
/// built for: the time comes from `ValidatedResult` and never from the file's
/// own `DeclaredResult`, and a sentinel is not a time.
pub fn parse_result(text: &str) -> (Option<i64>, Option<u32>) {
    let mut time = None;
    let mut cps = None;
    let mut in_validated = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("\"ValidatedResult\"") {
            in_validated = !t.contains("null");
        } else if in_validated && t.starts_with("\"Time\"") {
            time = t
                .split(':')
                .nth(1)
                .and_then(|s| s.trim().trim_end_matches(',').parse::<i64>().ok())
                .filter(|&ms| (0..=BAD_TIME_MS).contains(&ms));
            in_validated = false;
        } else if t.starts_with("\"Desc\"") {
            if let Some(p) = t.find("reached some checkpoints (") {
                let rest = &t[p + "reached some checkpoints (".len()..];
                cps = rest.split(' ').next().and_then(|s| s.trim().parse().ok());
            } else if t.contains("wrong simu") {
                cps = Some(0);
            }
        }
    }
    (time, cps)
}
