//! The subcommands, in the LIBRARY rather than beside `main`, so the test suite
//! calls them the way the CLI does instead of re-implementing them.

pub mod carrier;
pub mod events;
pub mod liveness;
pub mod probe;
pub mod ptr;
pub mod regen;
pub mod resync;
pub mod server;
pub mod trace;
pub mod tree;
pub mod watch;
