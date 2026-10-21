# Native driver and simulation shim

This workspace contains the source for the experimental Linux TrackmaniaRL
validator backend. `driver/` controls the session and `shim/` provides the
simulation hooks. The selected upstream libraries live under
`vendor/trackmania-tas/`; they are kept separate from TrackmaniaRL additions.

The workspace contains no game executable, Packs, map, ghost, reference
trajectory, captured state or compiled library. Obtain the required inputs
separately. Building this code does not download or start Trackmania.

## Build on Linux or WSL2

Use an x86_64 Linux Rust toolchain. The source was prepared using Rust 1.99.1.
From the repository root:

```bash
cargo check --locked --manifest-path native/Cargo.toml --workspace
cargo test --locked --manifest-path native/Cargo.toml -p forkshim --lib
cargo build --locked --release --manifest-path native/Cargo.toml
```

The outputs are `native/target/release/tmre-reset-driver` and
`native/target/release/libforkshim.so`. Keep them outside Git and release
archives. The Python wheel contains the adapter and license notices; the
source distribution also contains this workspace.

The driver requires an explicit canonical `TMRE_RUNTIME_ROOT`. It uses the
reviewed directory layout inside that root; it does not discover arbitrary
installations, maps or server versions. See the
[backend guide](../readme/native-headless.md) for the runtime contract.

## Source builds and runtime qualification

The adapter accepts a specific reviewed binary profile and a hash-bound
deployment manifest. Relocating sources, formatting code or changing the
compiler can change binary hashes. A successful build alone does not qualify
the new output or authorize replacing an expected hash in the adapter.

Use existing qualified deployment inputs for the pinned profile. A newly built
driver/shim needs its own build identity and controlled trajectory, input,
reset and terminal checks before it is registered as a supported profile.
Do not reuse a historical approval for a new binary or claim that source tests
establish graphical-client parity, a learned finish or a record time.

The bounded 32-episode/two-actor research runner and its frozen audit archives
are separate from the public SDK example. This workspace does not turn that
research runner into an unlimited training configuration.

## Licenses and provenance

Christopher Chedeau's original code is MIT. The license was added immediately
after the exact source base used here. Preserve
[the full upstream license](licenses/trackmania-tas-MIT.txt), the root `NOTICE`
and [the source provenance](SOURCE-PROVENANCE.json). TrackmaniaRL additions
are MIT; [DEPENDENCIES.md](DEPENDENCIES.md) records registry license choices.

Upstream comments describe that project's own historical measurements. They
are not additional qualifications of this SDK port. The vendored files are
retained for build provenance rather than rewritten to imply new ownership.

These code licenses do not grant rights to Ubisoft software or to maps,
ghosts and recordings. The shim patches executable instructions in a supplied
server; it is not only a telemetry reader. Applicable server terms and
statutory interoperability exceptions must be assessed for the intended use.
No approval from Ubisoft or Nadeo is claimed.
