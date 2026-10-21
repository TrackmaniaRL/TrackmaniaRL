//! **`Branch`** — the savestate-tree interface of DESIGN §3.1, for the explorer.
//!
//! ```text
//! fork(node)            -> handle
//! advance(handle, k)    -> (state_trace, handle')
//! finish(handle)        -> the facts a Verdict is made of
//! ```
//!
//! # What a handle is
//!
//! A **live paused simulation**: a process holding the whole engine, stopped
//! inside the physics loop, able to fork again. Not a serialized state, not a
//! replay position — an actual dedicated server sitting on a socket waiting to
//! be told what to do next. Advancing one costs a fork plus the ticks you asked
//! for; it does **not** cost the prefix, which is the entire point.
//!
//! # THE ONE RULE, AND IT IS IN THE TYPE
//!
//! **`advance` refuses any tick index at or below the handle's own probed
//! consumed boundary.**
//!
//! Not a convention, not a lint, not something the caller remembers: a method
//! that returns `Err`. The reason is the most expensive defect in this
//! project's history. The engine decodes the input tape up front into one
//! 32-byte record per tick and consumes them in order. A resume rewrites those
//! records — and **a record already consumed cannot be un-consumed, so
//! rewriting it is a silent no-op.** The candidate then scores *exactly* its
//! parent's score, `delta == 0` is accepted as "no improvement", and that
//! lineage is quietly contaminated: 312 false finishes, none of which survived
//! a full re-validation, and the clustering (always one arm-seed) looked like
//! anything but a boundary bug for four investigations.
//!
//! Three consequences are baked in here and none of them are negotiable:
//!
//! * **Every node probes its own boundary.** The `lroundf` checkpoint is not a
//!   fixed simulation point — under load it moves in whole ~62-call chunks, and
//!   a real run had 135 of 150 workers stop past the master's single
//!   calibration. A boundary inherited from a parent, a sibling or a master is
//!   a boundary for a different process.
//! * **A failed probe is a hard abort.** [`Forest::advance`] destroys a node
//!   whose probe fails rather than continuing with an estimate. A fallback here
//!   is how a plausible number 2–3 ms off gets banked.
//! * **Calibration may only move the floor LATER.** `max(calibrated, probe + 1)`
//!   and never `min`; `probe + 1` because tick `p` is the record the engine is
//!   about to read and is already partly consumed. Getting this backwards made
//!   23 of 100 candidates silently wrong on a re-verification that had passed
//!   4700/4700 the first time.
//!
//! # What this crate does NOT do
//!
//! It does not decide whether a fork answer is *true*. A fork answer is never a
//! result: the plain oracle re-simulating the written tape is. [`ForkAnswer`]
//! therefore carries, beside the time, **how far the tape was from the
//! reference the node's server checkpointed on** — the number that says whether
//! the answer is inside the regime where the fork was ever measured exact.
//! DESIGN §3.1 wants that inside `Verdict`; `Verdict` belongs to agent A and
//! this crate deliberately does not define one.

use forkoracle::forksrv::{parse_result, BranchReq, ForkServer, Rec};
use forkoracle::layout::{decode_rows, Layout, Row};
use forkoracle::tree::{Node, Tree};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// ~`lroundf` calls per simulated 10 ms tick.
///
/// Measured at ~255 on this engine and used ONLY to choose how far a branch
/// child runs before it stops. Nothing is ever labelled from it: where a child
/// actually stopped is what its own probe says, and the two differ by up to a
/// tick. This is the same discipline `session::clock_for_race_ms` states for
/// the fitted clock line — an estimate is allowed to place a checkpoint and is
/// never allowed to name one.
pub const LROUNDF_PER_TICK: u64 = 255;

/// A live paused simulation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Handle(pub u64);

/// The root: the fork server's own checkpoint, which is not a branch node and
/// has no pid of its own to kill.
pub const ROOT: Handle = Handle(0);

/// One tick of car state, as the fork-state readout produces it: position to
/// 3.4 mm RMS, orientation to ~2e-5 in the quaternion.
pub type StateTrace = Vec<Row>;

/// What a fork child reported about a candidate — the raw facts, not a verdict.
///
/// `Verdict` is agent A's type and this crate does not define one. What it does
/// is carry everything A's type needs, **including the distance from the
/// reference**, so nobody downstream has to reconstruct it: a fork answer is
/// only inside its validated regime when the tape is a small late perturbation
/// of the reference the server checkpointed on, and 0 of 312 answers survived
/// when it was not.
#[derive(Clone, Debug)]
pub struct ForkAnswer {
    /// `ValidatedResult.Time`, never the file's own declared time, and never a
    /// sentinel read as a value.
    pub time_ms: Option<i64>,
    /// Checkpoints reached, when the run did not finish.
    pub dnf_cps: Option<u32>,
    /// The node's own probed consumed boundary.
    pub boundary: usize,
    /// First tick at which the evaluated tape differs from the reference the
    /// server is running, and how many ticks differ in total. **This is the
    /// number that says whether to believe the answer.**
    pub first_diff_tick: Option<usize>,
    pub ticks_differing: usize,
    /// The validator's raw block, for anything that wants to re-read it.
    pub raw: String,
}

/// How a node writes its state trace, and where the car lives in its memory.
#[derive(Clone, Debug)]
pub struct TraceCfg {
    /// The located vehicle state and race clock. `layout::segments` builds the
    /// three segments; a `Layout` is per-server because the heap is bimodal
    /// run to run and PIE moves everything.
    pub layout: Layout,
    /// Where trace files go. One file per branch, named by pid, deleted when
    /// the node is dropped.
    pub dir: PathBuf,
    /// `lroundf` calls between sample attempts. 1 catches every tick.
    pub stride: u64,
    /// Cap on samples per branch, so a runaway child cannot fill a disk.
    pub max: u32,
}

struct Held {
    node: Node,
    trace: PathBuf,
    /// The ticks this node's lineage has written, so an answer can say how far
    /// it is from the reference without re-deriving it.
    written: Vec<usize>,
}

/// The tree, and everything alive in it.
pub struct Forest {
    root: ForkServer,
    tree: Tree,
    cfg: Option<TraceCfg>,
    nodes: HashMap<Handle, Held>,
    next: u64,
    /// The root's own probed boundary. Probed once, for the root, like any
    /// other node.
    root_boundary: Option<usize>,
    /// The reference tape the server is running, for the distance-from-
    /// reference field of every answer.
    reference: Vec<Rec>,
}

impl Forest {
    /// Take ownership of a started fork server and open a tree over it.
    ///
    /// `reference` is the tape the server is simulating, tick for tick. It is
    /// held so every answer can report how far the evaluated tape was from it.
    pub fn new(
        root: ForkServer,
        work: &Path,
        reference: Vec<Rec>,
        cfg: Option<TraceCfg>,
    ) -> Result<Forest, String> {
        let tree = Tree::new(work)?;
        Ok(Forest {
            root,
            tree,
            cfg,
            nodes: HashMap::new(),
            next: 1,
            root_boundary: None,
            reference,
        })
    }

    /// Probe the root's own consumed boundary. Must be called before the first
    /// `advance` from `ROOT`.
    pub fn probe_root(&mut self) -> Result<usize, String> {
        let t = self.root.probe_tick()?;
        self.root_boundary = Some(t);
        Ok(t)
    }

    /// The first tick it is safe to write at `h`.
    ///
    /// `max(calibrated, probe + 1)` — calibration may only push it LATER.
    pub fn floor(&self, h: Handle, calibrated: Option<usize>) -> Result<usize, String> {
        if h == ROOT {
            let p = self.root_boundary.ok_or(
                "the root has not probed its own boundary -- call probe_root before writing",
            )?;
            return Ok(match calibrated {
                Some(c) => c.max(p + 1),
                None => p + 1,
            });
        }
        self.held(h)?.node.floor(calibrated)
    }

    pub fn probed_boundary(&self, h: Handle) -> Option<usize> {
        if h == ROOT {
            return self.root_boundary;
        }
        self.nodes.get(&h).and_then(|x| x.node.probed_boundary())
    }

    /// **`fork(node) -> handle`** — a second handle at (essentially) the same
    /// point, with no inputs appended.
    ///
    /// It is `advance` with an empty macro, and it is honest about what that
    /// means: the new node stops one `lroundf` call later than its parent, not
    /// at the identical instant, and it probes its own boundary like any other
    /// node. There is no cheaper way to duplicate a paused process, and
    /// pretending the copy is at the same tick is exactly the assumption that
    /// makes an inherited boundary wrong.
    pub fn fork(&mut self, h: Handle) -> Result<Handle, String> {
        let (_, child) = self.advance(h, &[], 0, 1)?;
        Ok(child)
    }

    /// **`advance(handle, inputs[k]) -> (state_trace, handle')`**
    ///
    /// Append `inputs` starting at `from`, let the child consume `k_ticks` more
    /// ticks, and hand back the child as a new fork point together with the
    /// per-tick car state it produced on the way.
    ///
    /// Refuses if `from` is at or below this handle's own probed consumed
    /// boundary. Destroys the new node and returns `Err` if the node cannot
    /// probe its own boundary — a node without a probe is not a node.
    pub fn advance(
        &mut self,
        h: Handle,
        inputs: &[Rec],
        from: usize,
        k_ticks: u64,
    ) -> Result<(StateTrace, Handle), String> {
        // THE FORWARD-ONLY REFUSAL. Checked here against the parent's own
        // probe, and again inside `tree::Node::branch` for a node, so a caller
        // reaching past this API still cannot get underneath it.
        if !inputs.is_empty() {
            let floor = self.floor(h, None)?;
            if from < floor {
                return Err(format!(
                    "FORWARD-ONLY VIOLATION: handle {:?} has consumed through tick {}, so the \
                     first tick it may be given is {}; you asked to write from {}. A write below \
                     the boundary is a SILENT NO-OP that scores exactly this node's own score.",
                    h,
                    floor - 1,
                    floor,
                    from
                ));
            }
        }

        let id = Handle(self.next);
        self.next += 1;
        let sock = self.tree.sock_path();
        let (trace_path, segs, stride, max) = match &self.cfg {
            Some(c) => (
                c.dir.join(format!("trace-{}.bin", id.0)),
                forkoracle::layout::segments(&c.layout),
                c.stride,
                c.max,
            ),
            None => (PathBuf::new(), Vec::new(), 0, 0),
        };
        let tp = trace_path.to_string_lossy().into_owned();
        let req = BranchReq {
            from,
            recs: inputs,
            // ~255 calls to the tick. Where the child ACTUALLY stops is what
            // its probe says; this only decides roughly how far it goes.
            stop_after_lroundf: (k_ticks * LROUNDF_PER_TICK).max(1),
            sock: &sock,
            trace_path: &tp,
            segs: &segs,
            sample_stride: stride.max(1),
            sample_max: max,
            // Dedup on the whole gathered record, which contains the race
            // clock. Keying on the POSITION instead silently loses a tick
            // whenever the car does not move -- a countdown, a crash, a respawn
            // -- and shifts everything after it.
            key: (0, (segs.iter().map(|s| s.1).sum::<u32>()).max(1)),
        };

        let pid = match h {
            ROOT => self.root.branch(&req)?,
            _ => self.held_mut(h)?.node.branch(&req)?,
        };

        let mut node = self.tree.accept(forkoracle::forksrv::frame_timeout_ms())?;
        if node.pid != pid {
            // A node is only the node you asked for if it says so itself. Two
            // branches in flight on one server would otherwise be told apart by
            // arrival order, which is the shape of every swapped-replay defect
            // in this project.
            node.destroy();
            return Err(format!(
                "branch mismatch: asked for pid {} and pid {} arrived on the socket",
                pid, node.pid
            ));
        }
        // EVERY NODE PROBES ITS OWN BOUNDARY, and a failed probe is a hard
        // abort. Not "fall back to the parent's" -- that is the defect.
        if let Err(e) = node.probe() {
            let pid = node.pid;
            node.destroy();
            self.tree.reaped(pid);
            return Err(format!("node {} could not probe its own boundary: {}", pid, e));
        }

        let mut written = match h {
            ROOT => Vec::new(),
            _ => self.held(h)?.written.clone(),
        };
        written.extend((from..from + inputs.len()).filter(|t| {
            inputs
                .get(t - from)
                .zip(self.reference.get(*t))
                .map(|(a, b)| a != b)
                .unwrap_or(true)
        }));

        let trace = if trace_path.as_os_str().is_empty() {
            Vec::new()
        } else {
            self.read_trace(&trace_path)?
        };

        self.nodes.insert(id, Held { node, trace: trace_path, written });
        Ok((trace, id))
    }

    /// **`finish(handle) -> the facts a Verdict is made of`** — fork a child
    /// that runs `tail` to the end of the tape and report what the validator
    /// said.
    ///
    /// The handle survives: finishing is a question asked of a node, not a
    /// consumption of it.
    pub fn finish(&mut self, h: Handle, tail: &[Rec], from: usize) -> Result<ForkAnswer, String> {
        let boundary = self
            .probed_boundary(h)
            .ok_or("refusing to finish from a node that has not probed its own boundary")?;
        let raw = match h {
            ROOT => {
                let floor = self.floor(h, None)?;
                if from < floor {
                    return Err(format!(
                        "FORWARD-ONLY VIOLATION: the root has consumed through tick {}; \
                         you asked to write from {}",
                        floor - 1,
                        from
                    ));
                }
                self.root.run(from, tail)
            }
            _ => self.held_mut(h)?.node.run(from, tail)?,
        };
        let (time_ms, dnf_cps) = parse_result(&raw);

        // Distance from the reference: everything this lineage wrote, plus this
        // tail, that differs from the tape the server is simulating.
        let mut diff: Vec<usize> = match h {
            ROOT => Vec::new(),
            _ => self.held(h)?.written.clone(),
        };
        for (i, r) in tail.iter().enumerate() {
            let t = from + i;
            if self.reference.get(t) != Some(r) {
                diff.push(t);
            }
        }
        diff.sort_unstable();
        diff.dedup();
        Ok(ForkAnswer {
            time_ms,
            dnf_cps,
            boundary,
            first_diff_tick: diff.first().copied(),
            ticks_differing: diff.len(),
            raw,
        })
    }

    /// Destroy a node and forget it.
    pub fn release(&mut self, h: Handle) {
        if let Some(mut x) = self.nodes.remove(&h) {
            let pid = x.node.pid;
            x.node.destroy();
            self.tree.reaped(pid);
            if !x.trace.as_os_str().is_empty() {
                let _ = std::fs::remove_file(&x.trace);
            }
        }
    }

    pub fn live_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The pid of a live node, for the memory arm and for anything that has to
    /// end it from outside.
    pub fn node_pid(&self, h: Handle) -> Option<i32> {
        self.nodes.get(&h).map(|x| x.node.pid)
    }

    /// The root server, for the arms that measure it directly.
    pub fn root_mut(&mut self) -> &mut ForkServer {
        &mut self.root
    }

    fn read_trace(&self, p: &Path) -> Result<StateTrace, String> {
        let cfg = self.cfg.as_ref().ok_or("no trace configuration")?;
        let blob = std::fs::read(p).map_err(|e| format!("{}: {}", p.display(), e))?;
        let (rows, warn) = decode_rows(&blob, &cfg.layout, 0);
        if !warn.is_empty() {
            // A GAP IS NOT COSMETIC. The clock must step by exactly 10 between
            // rows; anything else means a tick was lost or duplicated and every
            // row after it is mislabelled. Refuse rather than return a trace
            // that reads fine and is off by a tick.
            return Err(format!(
                "the state trace is not tick-continuous ({}); refusing it rather than \
                 returning rows whose labels are wrong after the gap",
                warn.join("; ")
            ));
        }
        Ok(rows)
    }

    fn held(&self, h: Handle) -> Result<&Held, String> {
        self.nodes.get(&h).ok_or_else(|| format!("no such handle: {:?}", h))
    }

    fn held_mut(&mut self, h: Handle) -> Result<&mut Held, String> {
        self.nodes.get_mut(&h).ok_or_else(|| format!("no such handle: {:?}", h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LROUNDF_PER_TICK` is a PLACEMENT aid. This test exists to pin the
    /// comment: if someone ever uses it to label a tick, the arithmetic below
    /// is what they will be relying on, and it is only good to about a tick.
    #[test]
    fn the_tick_estimate_is_only_ever_used_to_place_a_stop() {
        assert_eq!(LROUNDF_PER_TICK * 10, 2550);
        // one tick of slop at ten ticks is 10%: too coarse to label with,
        // fine to stop with.
        assert!(LROUNDF_PER_TICK >= 200 && LROUNDF_PER_TICK <= 300);
    }
}
