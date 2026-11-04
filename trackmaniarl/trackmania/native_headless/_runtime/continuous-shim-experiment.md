# Experimental continuous-pedal shim

This describes an isolated experiment, not a qualified training deployment.
The original private source is `terminal-boundary-build583/source`, and its
`tools/search/forkshim/src/post_observe/native_ipc.rs` had SHA-256
`5674d59eb16ec9481ddcf2d95b76159c9ebc322454736dae20efdba2358dd492`.
The unmodified `libforkshim.so` had SHA-256
`ec861211a43ae675229ad954574d185ac1c1898e5a104ccd82a70169fad86bc0`.

In a new copy, replace exactly two occurrences of:

```rust
!(gas==0||gas==0x3f800000)||!(brake==0||brake==0x3f800000)
```

with:

```rust
!f32::from_bits(gas).is_finite()||!(0.0..=1.0).contains(&f32::from_bits(gas))||!f32::from_bits(brake).is_finite()||!(0.0..=1.0).contains(&f32::from_bits(brake))
```

These checks are in the future-record and immediate-current-record request
handlers. The resulting source file had SHA-256
`f9fd6b1123edf685af2384f8a3a90b99d66d2d346b36c20bc8dbac65dbc09871`.
Build in the copied `tools/search/forkshim` with `cargo build --release --offline`,
an isolated `CARGO_TARGET_DIR`, and the existing private offline toolchain
`/home/kuba/tmrl-rea-tools-20261021-001` (`cargo` and `rustup` directories).
The experimental binary built on this machine had SHA-256
`4f177a018b064f63bf4854b3724237a86fdbc265ebe8ffc0b9f2acee73ec2cea`;
builds on other toolchains need not be bit-identical. Never overwrite a pinned
runtime: put the binary in a new copy and create a separate manifest.

On 2026-10-30 a fresh isolated runtime processed four 10 ms actions with gas
0.25/0.75, brake 0.25/0.75, and steering 0/±0.25. The receipt, audit and POST
records matched the requested float32 bytes, and the observed physical state
reported the same controls. The process closed with exit 0 and no cleanup
errors or remaining owned processes. Data are in
`/home/kuba/tmrl-full-qr-20261022-011/codex-continuous-probe-20261030-04`;
the modified source and runtime are sibling `codex-analog-shim-20261030-01`
and `codex-analog-runtime-20261030-01`. This probe used an experimental V8
decoder permitting finite pedal values in [0,1]. The original shim rejected
gas 0.25 before acknowledgement; that failed probe was retained as
`codex-continuous-probe-20261030-01`.
The same four commands were repeated successfully in
`codex-continuous-probe-20261030-05` using a fresh V8 source deployment of the
adapter and decoder in this repository. That run also exited 0 with no forced
cleanup or remaining owned processes.

Four ticks do not establish actuator parity, reward correctness, finish or
motor/event-source qualification. Existing motor and paired-Finish evidence
is pinned to the original shim and must not be relabelled for the new binary.
The motor validator requires an actual 150-second source capture, 15,001
physical frames, 3,001 finish boundaries, at least 1,000 contacts and the
client comparison evidence. The finish validator requires actual paired client
and native Goal/accepted replay records. Generate and validate those for the
new shim before enabling a campaign; the factory retains all existing pin checks.

## Follow-up source and Finish qualification

The follow-up qualification on 2026-10-30 used the same experimental binary,
without training. All outputs and adapted executable scripts are retained under
`/home/kuba/tmrl-full-qr-20261022-011/codex-analog-qualification-20261030-01`.
The separate source deployment was `codex-continuous-source-20261030-02` and
the runtime `codex-analog-runtime-20261030-01`. The validator replayed the
actual accepted Windows ghost with the new shim: valid result 34,268 ms, one
checkpoint and no respawns. The fresh actual Goal observation matched the
historical client UI time. The paired Finish reference and SHA-256 are:

```
paired-finish/reference.json
e426ede7972693f1eeec492e01e12240e910e02b65dd3fe537f9892ea8496142
```

One new source capture covered 150 simulated seconds, 15,001 physical frames,
3,001 Finish boundaries and 87,945 contacts. Three additional 12-second native
captures repeated the historical right, left and straight controls. Their
motor predictions were recalculated against the original immutable Windows
callback data: 26, 27 and 28 comparisons respectively, all 81 agreeing on the
original strict collision threshold. The new motor reference and SHA-256 are:

```
motor-reference-replayed/reference.json
8d64ff8d8852282590199270a93c8490b8949f4ca237ada34d6ddcf9ff53a703
```

Both references passed their existing independent loaders. Every control
capture reported no remaining owned processes or cleanup errors. The earlier
`motor-reference` in the same qualification directory used old native comparison
results and is superseded, not the final evidence. The reference includes
`native_replays` metadata with SHA-256 for the three fresh traces; the current
motor loader does not independently check this extra metadata, so an auditor must
check those hashes separately along with the comparison code. The new 150-second
and paired-control trials used discrete controls; they do not establish motor or
reward equivalence under every analog action.

The retained scripts are adapted from the earlier `run-paired-ui585.py`,
`run-live-rumble587.py` and `compare-motor*.py` in the F: report directory. Their
outputs are immutable evidence: reruns require fresh output, trace, work and
cohort directories and the exact pinned private inputs. `run-paired.py` creates
the new Goal observation; `run-live.py` records the 150-second source;
`run-right.py`, `run-left.py`, `run-straight.py` capture paired native traces;
the corresponding `compare-*.py` recalculate against stored Windows callbacks.
Check pinned assets and available disk space before running. These scripts do
not start training. Their SHA-256 values, in the order listed above, are:

```
a29380c6527c40a3c7b1534b1905e78b917f468277718b1fb264ed09c7d0dea0  run-paired.py
e85ddf1c93b58bc4ce75478a53ea7ca41c57564b3d1675824c50f4d8c9d9b237  run-live.py
a005d490d943a00730eb163bd2be8364942ee29dfba42284683fd1963d74aa80  run-right.py
3a25daff7d7548ccb9277a0cdaa6950f0cb12c46b21a6ec123d9fdfacfd2862d  run-left.py
f00bef4ed33991e3bd781efb47a527484d343408da90e125708dac2edf00c7b9  run-straight.py
5b378c41dc67fca03e57e17c591116b81a3c940cb7f001cd2f87ac92129abc4d  compare-right.py
a99ec57ff1da0621984e62e9c1397af415b162a204e5b74698db824052251014  compare-left.py
55d727f7a3ef81b7307fbe90184338320c28acc2930c4bb63086656e8c0b1ddb  compare-straight.py
```
