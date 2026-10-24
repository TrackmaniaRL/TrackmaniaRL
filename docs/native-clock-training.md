> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Source-bound clock for native-only training

Native-only training uses the actual V7 `POST.nativeT` counter and a separately
versioned nominal simulation-time contract. It does not require, reuse, or extend
the Windows/native race-clock correspondence. A delta of ten native counter units
is one nominal 0.01-second engine interval. This is an engine simulation unit; it
does not measure wall time or identify the race-clock origin.

The canonical local artifact is
`<USER_HOME>/tmrl-headless-training-20261022-007/clock-proof001/final001/clock.json`,
SHA256 `5fffae5f54ad88b20e757b5ef7e5da5c2c1360d4331b8d782e809d1bef5e9a33`.
Its independent `unit-proof.json` has SHA256
`2a9db19308926aa96aa0ada8148b2dcbf0155889de4f7c3550b0eba5d8f374a6`.
The earlier top-level `clock-proof001/clock.json` is a preserved historical draft;
the final artifact includes an immutable `generator-source.py` snapshot.

## Evidence for the unit

The pinned Linux server has SHA256
`3b6107824feea402bd32c888e994dcd6df4a10afc8676a2647ac9560ccf0d779`.
`scripts/headless_native_clock.py` checks exact instruction bytes in this ELF,
maps virtual addresses through its file-backed `PT_LOAD` headers, and checks the
actual call and RIP-relative constant operands. The generator never executes the
server, shim, driver, or any game process.

| Source address | Checked operation | Consequence |
| --- | --- | --- |
| `0x1219750` / `0x1219753` | Load completed counter into EBX; store it at owned simulation `+0x48` | The observed simulation clock is the completed-loop counter. |
| `0x1219774` | Existing POST boundary copies completed counter before adding raw frame dt | POST identifies a completed interval, rather than a future requested target. |
| `0x121a050` through `0x121a071` | Load actual raw frame dt into EDX and call `0x119f1b0` | The conversion receives the loop's integration duration. |
| `0x119f1e1` | `CVTSI2SS XMM0,EDX` | Convert actual integer frame dt to scalar float. |
| `0x119f1e5` | `MULSS XMM0,[0x399e90]` | Scale frame dt with the engine's stored constant. |
| `0x119f1ed` | Save the converted duration at stack `-0x100` | Preserve the scaled duration for component updates. |
| `0x11a1511` / `0x11a1519` | Load that saved duration into XMM0; call `0x937060` | Pass scaled duration into a simulation component update. |

The constant at virtual/file address `0x399e90` is the exact four bytes
`6f12833a`, binary32 word `0x3a83126f`: `0.0010000000474974513`.
The declared `seconds_per_tick = 0.001` is its nominal rational scale, 1/1000.
The proof also records the actual binary32 constant and a separately labelled
round-to-nearest illustration of its product with ten. It does not claim that
floating-point integration computes exact rational arithmetic or alter the
engine's floating-point state.

The original shim source, `search/forkshim/src/post_observe.rs`, SHA256
`7192f57bb4e2052236c248616f3f35c771e99197a01dfef846aafe5ce6d6527a`,
requires actual `dt == frame_dt == 10` and
`simclock == completed == current` before publishing POST. It is bound to the
unchanged original source manifest. Production `qualify_post` separately checks
the POST header, owner PID/TID, source mask and role, counter equality, ten-unit
dt, and grid alignment. Every accepted transition still needs this actual
evidence; a requested N or elapsed host time cannot replace it.

The retained R5 public production-environment reference provides an independent
observed ledger: 1,780 consecutive actual POST intervals, from native counter
2,200 through 20,000, with ten-unit increments and dt/frame-dt equality. Its real
terminal is joined to the original validator and TMRESULT. Computed result time
17,715 and `validationAcceptance=false` are preserved. Neither that result time
nor the terminal supplies a native race-clock origin or an interpolated physical
finish timestamp. The unit is established by the binary conversion above; the
long ledger corroborates the counter and integration invariants.

## Pinned scope and duration bounds

The artifact schema is `tmrl.native-simulation-clock/v1`; its independent unit
evidence schema is `tmrl.native-simulation-unit-source/v1`. Both declare:

```text
nominal-engine-simulation-seconds; no wall/race origin mapping
```

It binds profile `native-immediate-kinematics6/v7`, reset
`neutral-race-start/v7`, schedule `current-record-before-load/v1`, and the exact
server, map, reference and input tape. All referenced source files are hashed
when loading the artifact. The manifest and generator preserve the source/build
identities and full source paths. No original asset or production pin changes.

Reset yields actual counter 2,200 and current input-reader index 152. The pinned
tape has 4,152 records. V7 requires `p + n < input_count` before any input write.
The last supported controlled boundary is therefore:

```text
2200 + (4152 - 152 - 1) * 10 = 42190
```

This is a source-derived bound for this tape and reader protocol, rather than an
extension of the observed Windows range. The source-bound unit conversion applies
within the full pinned execution scope; individual steps must still pass runtime
POST and reader validation. It does not assert measured physical trajectories at
every counter up to 42,190.

A configured episode duration of 20,000 raw units ends no later than native
22,200 from this reset and corresponds to 20 nominal simulation seconds. The
existing 39,950-unit default ends at 42,150, also within the supported bound.
Durations are positive multiples of ten. N1/N5 requests use actual returned
intervals; terminal ragged intervals retain their actual duration. An unsupported
final-reader request fails before writes. Reader EOF alone is never a finish, and
autonomous release/drain intervals are never credited to a learner.

## Training contract

`NativeOnlyClockProof.from_artifact(path, expected_sha256=...)` loads the explicit
artifact and checks all source hashes. `native_only_contract(proof, reference)`
creates purpose `native-only-training`; `NativeOnlyAdapter` keeps the existing V7
applied-input, raw-state, terminal and reward validation. The contract records its
native clock evidence hash, units and semantics, so a checkpoint from a different
clock contract is rejected. A reset must also present the actual pinned engine
and tape identities.

This artifact grants native nominal clock qualification only.
`paired_training_qualified` remains false and `windows_clock_mapping` remains
null. Windows command acknowledgement and Windows/native comparison gates remain
separate.

## Reproduce the proof without running an engine

Use a fresh output path; the generator rejects an existing directory. The
following command reads retained assets and the R5 actual ledger only:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/mnt/c<USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania \
<USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python \
  -m scripts.headless_native_clock \
  --config <USER_HOME>/tmrl-headless-parity-20261021-005/production-profile001/config.json \
  --reference <USER_HOME>/tmrl-headless-parity-20261021-001/runtime/experiments/headless-native16-training-001/reference.json \
  --shim-post-source <USER_HOME>/tmrl-headless-parity-20261021-002/native-input-v7/source/tools/search/forkshim/src/post_observe.rs \
  --finish-run <USER_HOME>/tmrl-headless-parity-20261021-005/public-source-reference001 \
  --output <USER_HOME>/tmrl-headless-training-20261022-007/clock-proof001/reproduction001
```

The original final artifact can be checked with its SHA above. A reproduction has
new absolute paths and therefore a new artifact hash; it must be explicitly
selected and pinned before use.

The 15 focused tests cover exact constant/call linkage, changed or missing source
windows, ELF file mapping and ambiguity, actual counter/dt disagreement, missing
ledger rows, and source-derived tape bounds. Final tests: 15 PASS; strict local
mypy and Ruff checks PASS. Reports and verbatim proof copies are retained under
`F:/AITrackmania/reports/headless-training-20261022-r7/clock-proof001/`.
