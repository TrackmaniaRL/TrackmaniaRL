//! The fork server's DRIVER side: everything the SEARCH and `fk` both do to a
//! live dedicated server.
//!
//! The split is by CALLER, not by topic. What is here is what `tmsearch` runs
//! on every candidate — start a server, arm the watchdog, resume a fork, locate
//! the car without a reference, score it — and `fk` drives the same code, so
//! there is one definition of what a resume is.
//!
//! What is NOT here is the CLOCK-FIRST locator (`fk::locate`): nothing in the
//! search calls it. It finds the engine's race clock first and keys every
//! sample on it, which is what makes it correct when the car does not move (a
//! respawn) or the engine writes the state twice inside one tick. `blind` keys
//! on the 24-byte position+velocity window instead and is what a search needs,
//! because an evolved candidate has no recorded telemetry to match against.
//!
//! `pred_core` is the same source the LD_PRELOAD shim compiles into the child,
//! so a predicate means exactly one thing on both sides of the fork.

pub mod pred_core;
pub mod forksrv;
pub mod tree;
pub mod pred;
pub mod layout;
pub mod blind;
pub mod inputs;
pub mod procmem;

/// What an armed event clause saw, as the driver reports it.
///
/// `Unarmed` and `Silent` look identical in the child's summary -- a
/// `fire_tick` of -1 either way -- and they mean opposite things to the
/// ranking, so the distinction is the driver's own knowledge and is carried in
/// the type rather than inferred from the wire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventSeen {
    Unarmed,
    Silent,
    Fired {
        tick: i32,
        /// The condition's value when it fired -- on a launch detector, the
        /// size of the one-tick speed rise, in m/s.
        value: f32,
        pos: [f32; 3],
        /// The after-key. 0 when no after-key was given (a flat band, which is
    /// correct); NEGATIVE INFINITY when one WAS given and the after-window was
    /// empty, so an empty window is the worst measured value rather than -- as
    /// it was until 267460 measured it -- the best one.
        after: f32,
        /// -1 when the run ended on the firing tick, or no after-key was given.
        after_tick: i32,
    },
}

impl std::fmt::Display for EventSeen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventSeen::Unarmed => write!(f, "no event clause"),
            EventSeen::Silent => write!(f, "the event never fired"),
            EventSeen::Fired { tick, value, pos, after, after_tick } => write!(
                f,
                "FIRED at tick {} ({:+.2}) at ({:.2}, {:.2}, {:.2}); after {:+.4}{}",
                tick,
                value,
                pos[0],
                pos[1],
                pos[2],
                after,
                if *after_tick >= 0 {
                    format!(" at tick {}", after_tick)
                } else {
                    String::new()
                }
            ),
        }
    }
}
