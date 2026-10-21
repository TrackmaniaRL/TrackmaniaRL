//! THE SAVESTATE TREE: driver side.
//!
//! # What a tree node is
//!
//! The classic fork server has one fork point. Every candidate is a child that
//! rewrites the tail of the input array, runs to the finish, prints a time and
//! dies. Anything the child learned on the way is lost, so reaching a state
//! 2000 ticks in costs a full re-simulation of those 2000 ticks, every time.
//!
//! A **node** is a fork child that stops after a few ticks and re-enters the
//! fork server itself, on a socket of its own. It is a savestate: a paused
//! simulation that can be forked again. A search over macros then pays the
//! macro, not the prefix.
//!
//! # The three things that make this safe rather than merely fast
//!
//! 1. **Every node probes its own consumed boundary.** The `lroundf` checkpoint
//!    is not a fixed simulation point -- under load it moves in whole chunks of
//!    ~62 calls -- so where a node stopped is a property of that node and of
//!    nothing else. A node whose probe fails is destroyed, never used with an
//!    assumed boundary. (`tm2020-forkserver.md`: a failed probe is a hard
//!    abort, never a fallback.)
//! 2. **`branch` refuses to write at or below that boundary.** A record the
//!    engine has already consumed cannot be un-consumed; rewriting it is a
//!    silent no-op that scores exactly the parent's score, so the mutation
//!    disappears and the lineage is contaminated for free. That is defect 1 of
//!    `tm2020-phantoms.md` and it is the reason this module exists. The refusal
//!    is here, at the lowest level that knows the boundary, and it is not
//!    optional.
//! 3. **Calibration may only push the boundary LATER, never earlier.** A caller
//!    that has its own estimate combines it as `max(estimate, probe + 1)`, never
//!    `min`. `Node::floor` does that arithmetic so no caller has to remember it.
//!
//! # Why a unix socket
//!
//! A branch child inherits its parent's command pipe. If it served that pipe,
//! two processes would race for every command byte and the loser would
//! eventually execute a command assembled from the middle of somebody else's
//! patch payload. So the node gets fresh descriptors: it connects to a listener
//! the driver owns, and the driver matches the connection to the branch it
//! asked for **by the pid the node names in its own handshake**.

use crate::forksrv::{
    parse_probe, parse_ready, payload_branch, payload_probe, payload_run, read_frame, write_frame,
    BranchReq, Rec,
};
use std::os::raw::c_int;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

extern "C" {
    fn kill(pid: c_int, sig: c_int) -> c_int;
}
const SIGKILL: c_int = 9;

/// The driver's listening end. One per fork server.
pub struct Tree {
    listener: UnixListener,
    path: PathBuf,
    /// Every node this tree has handed out and not yet reaped. Dropping the
    /// tree kills all of them: a node is a paused 150 MB engine, and a beam of
    /// orphans is how a box dies.
    live: Vec<i32>,
}

impl Tree {
    /// Bind the listener inside the fork server's own work directory.
    ///
    /// The work directory is already per-process and already locked, so the
    /// socket inherits that isolation rather than inventing its own naming
    /// scheme. Two runs cannot collide on it for the same reason two runs
    /// cannot collide on their replays -- which is the failure mode behind four
    /// separate silent corruptions in this project's history.
    pub fn new(dir: &Path) -> Result<Tree, String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let path = dir.join("branch.sock");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).map_err(|e| {
            format!("binding the branch socket at {}: {}", path.display(), e)
        })?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        Ok(Tree { listener, path, live: Vec::new() })
    }

    /// The path a `BranchReq` must carry. A `sockaddr_un` path is 107 bytes at
    /// most and the shim refuses a longer one rather than truncating it into a
    /// path that names something else.
    pub fn sock_path(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }

    /// Wait for one node to arrive and complete its handshake.
    ///
    /// Returns the node with its own `base`, `clock` and pid. The caller checks
    /// the pid against what `branch` returned: a node is only the node you
    /// asked for if it says so itself.
    pub fn accept(&mut self, timeout_ms: i32) -> Result<Node, String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms.max(0) as u64);
        let stream = loop {
            match self.listener.accept() {
                Ok((s, _)) => break s,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() >= deadline {
                        return Err(format!(
                            "no branch node connected within {} ms -- the child either never \
                             reached its stop point or could not reach {}",
                            timeout_ms,
                            self.path.display()
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                Err(e) => return Err(format!("accept: {}", e)),
            }
        };
        stream.set_nonblocking(false).map_err(|e| e.to_string())?;
        let mut n = Node { sock: stream, base: 0, clock: 0, pid: -1, boundary: None, dead: false };
        let hello = read_frame(&mut n.sock).ok_or("a branch node connected and said nothing")?;
        let s = String::from_utf8_lossy(&hello).into_owned();
        let (base, clock, pid) = parse_ready(&s)?;
        let pid = pid.ok_or_else(|| {
            format!(
                "a branch node handshook without naming its pid ({:?}) -- a node the driver \
                 cannot kill is an orphan holding a 150 MB address space",
                s.trim()
            )
        })?;
        n.base = base;
        n.clock = clock;
        n.pid = pid;
        self.live.push(pid);
        Ok(n)
    }

    /// Forget a node the caller has already destroyed.
    pub fn reaped(&mut self, pid: i32) {
        self.live.retain(|p| *p != pid);
    }

    pub fn live_count(&self) -> usize {
        self.live.len()
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        for p in std::mem::take(&mut self.live) {
            unsafe { kill(p, SIGKILL) };
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A paused simulation that can be forked again.
pub struct Node {
    sock: UnixStream,
    /// The decoded input array, at the address every descendant shares.
    pub base: u64,
    /// The `lroundf` count at which this node stopped. **Not a simulation
    /// point**: it is where THIS process happened to stop, and it differs
    /// between nodes of one tree. Never label anything from it.
    pub clock: u64,
    pub pid: i32,
    boundary: Option<usize>,
    dead: bool,
}

impl Node {
    /// THE PROBE. The first tick this node has not consumed, asked of the
    /// engine by taking read access to the input array away from a throwaway
    /// child and catching the fault.
    ///
    /// Every node runs this once, for itself. Inheriting a parent's boundary is
    /// precisely the defect that produced 312 false finishes.
    pub fn probe(&mut self) -> Result<usize, String> {
        let t = self.request(&payload_probe()).and_then(|s| parse_probe(&s))?;
        self.boundary = Some(t);
        Ok(t)
    }

    /// The first tick it is safe to write, combining the probe with an optional
    /// outside estimate.
    ///
    /// `max(estimate, probe + 1)` and never `min`: tick `p` is the record the
    /// engine is about to read and is already partly consumed, and calibration
    /// may only push the boundary LATER. Getting this backwards is what made
    /// 23 of 100 candidates silently wrong on a re-verification.
    pub fn floor(&self, calibrated: Option<usize>) -> Result<usize, String> {
        let p = self.boundary.ok_or_else(|| {
            "this node has not probed its own boundary -- a node without a probe is not a node"
                .to_string()
        })?;
        Ok(match calibrated {
            Some(c) => c.max(p + 1),
            None => p + 1,
        })
    }

    pub fn probed_boundary(&self) -> Option<usize> {
        self.boundary
    }

    /// Fork a child that runs the tape to the finish and returns the
    /// validator's JSON. The node itself is untouched and can be forked again.
    pub fn run(&mut self, from: usize, recs: &[Rec]) -> Result<String, String> {
        self.check_forward(from)?;
        self.request(&payload_run(from, recs))
    }

    /// Fork a child that appends and becomes a node of its own.
    ///
    /// **Refuses any write at or below this node's own probed boundary.** That
    /// refusal is the whole safety story of the forward-only regime and there
    /// is no flag to turn it off: a caller who needs to rewrite history has to
    /// re-simulate from a node that has not consumed it yet.
    pub fn branch_clock(&mut self, req: &BranchReq) -> Result<i32, String> {
        self.check_forward(req.from)?;
        let mut payload = payload_branch(req); payload[0] = b'K';
        let s = self.request(&payload)?;
        crate::forksrv::parse_branched(&s)
    }
    pub fn branch(&mut self, req: &BranchReq) -> Result<i32, String> {
        self.check_forward(req.from)?;
        let s = self.request(&payload_branch(req))?;
        crate::forksrv::parse_branched(&s)
    }

    /// **THE NEGATIVE CONTROL'S ONLY DOOR.** Run a candidate whose writes go
    /// BELOW this node's probed boundary.
    ///
    /// It exists because rung 0.5 needs both halves: the positive half (a
    /// forward-only candidate agrees with the plain oracle) is decoration on
    /// its own, since a fork server that happened to be fine for unrelated
    /// reasons passes it. The negative half has to reproduce the known wrong
    /// answer, and it cannot do that through an API that refuses it.
    ///
    /// The name is this long on purpose. There is no flag on [`Node::run`] or
    /// [`Node::branch`] that reaches this behaviour, nothing defaults to it, and
    /// one `grep` over the tree finds every caller. A search that calls it is a
    /// search that typed the words.
    pub fn run_below_boundary_for_the_negative_control(
        &mut self,
        from: usize,
        recs: &[Rec],
    ) -> Result<String, String> {
        self.request(&payload_run(from, recs))
    }

    /// The forward-only rule, enforced.
    fn check_forward(&self, from: usize) -> Result<(), String> {
        let p = self.boundary.ok_or_else(|| {
            "refusing to write into a node that has not probed its own consumed boundary"
                .to_string()
        })?;
        if from <= p {
            return Err(format!(
                "FORWARD-ONLY VIOLATION: this node has already consumed tick {}, and a write at \
                 tick {} would be a silent no-op that scores exactly the parent's score. Branch \
                 from an ancestor that has not consumed it.",
                p, from
            ));
        }
        Ok(())
    }

    fn request(&mut self, p: &[u8]) -> Result<String, String> {
        if self.dead {
            return Err("this node has been destroyed".into());
        }
        write_frame(&mut self.sock, p).map_err(|e| format!("node {}: {}", self.pid, e))?;
        match read_frame(&mut self.sock) {
            Some(v) => Ok(String::from_utf8_lossy(&v).into_owned()),
            None => {
                self.dead = true;
                Err(format!("node {} stopped answering", self.pid))
            }
        }
    }

    /// End this node. Idempotent.
    pub fn destroy(&mut self) {
        if !self.dead && self.pid > 0 {
            unsafe { kill(self.pid, SIGKILL) };
        }
        self.dead = true;
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A node with no probe refuses every write. The failure mode this guards
    /// is not a crash, it is a plausible number: a write below the boundary is
    /// a no-op and the candidate scores exactly its parent.
    #[test]
    fn a_node_without_a_probe_refuses_to_be_written_to() {
        let n = Node {
            sock: UnixStream::pair().unwrap().0,
            base: 0,
            clock: 0,
            pid: 1,
            boundary: None,
            dead: false,
        };
        let e = n.check_forward(500).unwrap_err();
        assert!(e.contains("has not probed"), "{}", e);
        assert!(n.floor(None).is_err(), "a node with no probe has no floor either");
    }

    /// The forward-only refusal fires exactly at the boundary, and the boundary
    /// tick ITSELF is refused -- the engine is about to read it, so it is
    /// already partly consumed.
    #[test]
    fn the_forward_only_refusal_fires_at_and_below_the_boundary() {
        let n = Node {
            sock: UnixStream::pair().unwrap().0,
            base: 0,
            clock: 0,
            pid: 1,
            boundary: Some(171),
            dead: false,
        };
        for t in [0usize, 1, 170, 171] {
            let e = n.check_forward(t).unwrap_err();
            assert!(e.contains("FORWARD-ONLY VIOLATION"), "tick {} was allowed: {}", t, e);
        }
        for t in [172usize, 173, 4000] {
            assert!(n.check_forward(t).is_ok(), "tick {} was refused and should not be", t);
        }
    }

    /// Calibration may only push the boundary LATER. An earlier estimate is
    /// ignored, not honoured -- the probe is authoritative about what is
    /// already consumed.
    #[test]
    fn calibration_can_only_move_the_floor_later() {
        let n = Node {
            sock: UnixStream::pair().unwrap().0,
            base: 0,
            clock: 0,
            pid: 1,
            boundary: Some(171),
            dead: false,
        };
        assert_eq!(n.floor(None).unwrap(), 172);
        assert_eq!(n.floor(Some(100)).unwrap(), 172, "an EARLIER estimate must be ignored");
        assert_eq!(n.floor(Some(172)).unwrap(), 172);
        assert_eq!(n.floor(Some(400)).unwrap(), 400, "a LATER estimate must win");
    }
}
