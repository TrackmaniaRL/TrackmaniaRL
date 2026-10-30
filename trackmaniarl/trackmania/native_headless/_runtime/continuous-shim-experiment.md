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
