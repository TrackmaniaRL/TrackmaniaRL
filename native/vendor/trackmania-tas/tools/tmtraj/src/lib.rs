//! `tmtraj` — read-only analysis of a Trackmania 2020 run.
//!
//! It never writes a ghost. The file format lives in `gbx`; every mutation
//! lives in `ghost`. What is here is the layer above the bytes: what a
//! trajectory says the car did, whether two recordings are the same run, and
//! whether a file is a physically coherent run of a car.

mod csvdiff;
mod run;
pub use run::run;

pub(crate) mod adjudicate;
pub(crate) mod adjudicate_batch;
pub(crate) mod blockdiffcmd;
pub(crate) mod checkcmd;
pub(crate) mod claimscmd;
pub(crate) mod cli;
pub(crate) mod corpuscmd;
pub(crate) mod diffcmd;
pub(crate) mod fmt;
pub(crate) mod facingcmd;
pub(crate) mod impactcmd;
pub(crate) mod intgcmd;
pub mod aircmd;
pub(crate) mod envelope;
pub(crate) mod geomcmd;
pub mod json;
pub mod lines;
pub(crate) mod manifest;
pub(crate) mod provcmd;
pub(crate) mod scancmd;
pub(crate) mod routecmd;
pub(crate) mod selftest;
pub(crate) mod serial;
pub(crate) mod shippedcmd;
pub(crate) mod splitscmd;
pub mod stats;
pub(crate) mod whlcmd;

/// Internals this crate's own integration tests reach into. **Not an API.**
///
/// Every other module is `pub(crate)` on purpose: `pub` switches off the
/// dead-code warning, and that is how a crate accretes sixty commands and
/// nobody notices forty of them stopped being called. Making the dispatcher a
/// library entry point (`run`) and everything else crate-private turned the
/// compiler back into the dead-code detector, which found eighteen unreachable
/// functions and structs on the first build — including three that had been
/// documented in the README as if they were live.
///
/// An integration test is a separate crate, so it cannot see `pub(crate)`.
/// Named items only, never a whole module, except `lines` and `stats`, whose
/// golden suites walk most of their surface.
#[doc(hidden)]
pub mod testonly {
    pub use crate::intgcmd::md5_hex;
    pub use crate::selftest::selftest;
    pub use crate::serial::{
        csv_string, full_json_string, path_json_string, SampleFields, Val, CSV_COLUMNS,
    };
}
