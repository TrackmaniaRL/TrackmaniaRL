# Native headless source publication review

Prepared on 2026-10-21 against TrackmaniaRL main
`c6f4dfefebe84fbf931376fabd7d85f4de89c996` (1.2.13).

## Included source and notices

The branch includes the Python adapter and bounded CPU example, explicit
runtime-root/deployment checks, and a buildable Rust workspace for the driver,
shim and selected upstream libraries. No Trackmania executable, Packs, maps,
ghosts, captured state or compiled native library is included in these additions.
The Python wheel carries the adapter and MIT notices. The source archive also
contains the native workspace and lockfile.

Christopher Chedeau added MIT immediately after source base
`235f4943188732f491ab9243954ede759e8a7f16` in
[license commit ad6bf1c](https://github.com/vjeux/trackmania-tas/commit/ad6bf1cb4085d554ce981bf2a29f659211a0b6a9).
The full license is preserved in the source tree and wheel license metadata.
`native/SOURCE-PROVENANCE.json`, `native/DEPENDENCIES.md` and root `NOTICE`
record provenance and dependency obligations. Code licenses do not license
Ubisoft software, third-party maps/ghosts or the intended use of the shim.

## Validation

- Full CPU suite on Linux/ext4: **1298 passed, 1 skipped** (154.25 seconds).
  One Torch/oneDNN warning remains. The earlier run from the Windows mount was interrupted
  after 21 passing tests because of filesystem overhead; it is not counted.
- Ruff lint and formatting: passed; mypy: **299 source files**, passed.
- Locked uv dependency resolution: passed without dependency changes.
- Locked Rust workspace check and release build: passed on x86_64 Linux,
  Rust 1.99.1. Upstream unused/dead-code warnings remain.
- Shim protocol tests: **2 passed**. A stale truncation fixture was corrected
  to account for the existing trailing plane-axis word; no production parser
  behavior was changed by that fix.
- Wheel and source archive built; content, license and privacy checks passed.
  Archive checks reject executable/game file suffixes and disguised ELF/PE/GBX
  payloads. These bounded checks supplement manual review.
- Installed-wheel import and portable-profile resource checks passed from
  outside the checkout. No game process was started.
- Diagram sources, SVG/HTML and PNG were regenerated; PNG was visually checked.

## Remaining runtime scope

This source preparation does not requalify newly compiled native binaries.
Historical engine results apply to their frozen binaries; expected hashes and
approvals have not been relaxed. Newly built driver/shim outputs require a
separate controlled reset/input/trajectory/terminal qualification before use
as a supported profile. The public asset bootstrap, arbitrary map support,
unlimited distributed runner and client-parity validation are not supplied
by this branch. See [the backend guide](../../readme/native-headless.md) and
[historical integration evidence](native-headless-integration.md).
