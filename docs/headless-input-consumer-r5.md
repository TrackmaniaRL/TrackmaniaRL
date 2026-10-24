> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# r5 literal input consumer and API analysis

This offline analysis uses the retained `input-consumer007-live002` and
`input-consumer007-live003` files under
`F:/AITrackmania/reports/headless-training-parity-20261021-r5`.
It does not run the game, install a plugin, issue controls, change production
pins, or grant a training gate. The raw callback semantics and hook provenance
remain defined by `HeadlessInputConsumer007/README.md`.

`scripts/input_consumer_analysis.py` validates the completed capture, exact
restoration, executable/Openplanet pins and binding to the diagnostic report.
It retains literal callback words and compares the actual gas/brake/steer,
filtered steer, vehicle observed fields, and participant cache. The observed
brake is stored as an integer boolean; only its explicit comparison encoding
uses float32 zero/one, while the original word remains in the report.

## Actual results

| Retained evidence | live002 | live003 long mild left |
| --- | ---: | ---: |
| Callback records | 199 | 228 |
| Original API packets, including initial | 299 | 344 |
| Fresh006 boundary packets | 6 | 6 |
| Actual versus vehicle/cache SGB equal | 197/199 | 226/228 |
| Vehicle versus participant cache equal | 199/199 | 228/228 |
| Actual steer versus filtered steer equal | 199/199 | 228/228 |
| Unique physics6 content matches to original API | 197 | 227 |
| Distinct matched states with API steer differing by 1 ULP | 0 | 21 |

In live002 the actual gas changes at callback sequence50 and the vehicle/cache
change at51. Actual steer changes at126 and vehicle/cache at127. In live003
the corresponding sequences are14/15 for gas and29/30 for steer. These are
same-callback field comparisons: each transition has one callback with the
new actual control and the previous observed/cache value. Brake has no
differences in either capture; no nonzero braking phase was exercised.

Both captures' actual and filtered steer words are only zero and `0xbe2952a5`.
The vehicle/cache words also use only those two values. Live003's original API
stream additionally contains21 packets with `0xbe2952a6`, exactly one binary32
representable value away from `0xbe2952a5`. All21 packets have separate unique
physics6 content matches to callbacks whose actual, filtered, vehicle and
cached steer are `0xbe2952a5`. Their API race times are:

```text
1140 1170 1190 1210 1230 1270 1290 1310 1330 1350 1370
1390 1410 1430 1470 1490 1510 1530 1550 1590 1610 ms
```

The corresponding callback sequences are107,110,112,114,116,120,122,124,126,
128,130,132,134,136,140,142,144,146,148,152,154. Each matched physics6 signature
occurs once in the callback capture and at exactly one retained API race time.
No matched API physical state has ambiguous retained SGB patterns here. This
locates a difference between the retained API sample and the matching physical
content; it does not establish an atomic cross-thread snapshot or the mechanism
that produced the API value. The six fresh006 boundary samples did not sample
these21 intermediate states and cannot independently exclude the difference.
The issued mild-left command is separately retained and is not substituted for
the measured float32 controls.

## R13D and API observation relation

The analyzer fits alternating unique moving physical-content matches, then
checks the held-out matches using exact rational arithmetic. Repeated physical
signatures and signatures appearing at multiple API race times are rejected as
anchors. No nearest-time search, interpolation, missing-state fill or ULP
tolerance is used.

| Observation-only relation | live002 | live003 |
| --- | ---: | ---: |
| Exact content matches checked | 197 | 227 |
| Fit / held-out | 99 / 98 | 114 / 113 |
| Observed API race span | 1960 ms | 2260 ms |
| Literal R13D minus matched API race | 214842100 | 214998900 |
| Consecutive callback R13D differences | 198 times10 | 227 times10 |
| Fresh006 matching boundary states | 5 | 5 |

For each capture, every retained original-stream match satisfies
`API race = R13D - offset` with its listed offset. Five fresh006 boundary states
per capture independently have unique matching physical content and the same
offset, but do not meet the analyzer's minimum of three fit plus three held-out
states for a separate fit. The offsets differ between captures. The report
therefore calls this an observed content relation, without treating R13D as a
generally qualified simulation clock or granting a new time range.

Original API content is absent for callback sequences0 and198 in live002 and
227 in live003. Those exclusions remain explicit. Fresh006 has a final
live003 boundary match at callback227. The channels are analyzed separately;
they are not merged to manufacture a complete original stream.

Fresh006's first pre-neutral boundary race is50 ms ahead of the saved original
stream pre-phase race in live002 and60 ms ahead in live003. The later pre-gas
and pre-steer boundaries are10 ms ahead in both. These are differences between
two actual retained observations. The reports retain producer begin/end,
host request/receive timestamps, and controller-call start/end separately.
They do not turn producer time or controller return into a physical input ACK.

## Relation to the separate native replay

The independent reports at
`<USER_HOME>/tmrl-headless-parity-20261021-005/consumer-native-replay002/report.json`
and `consumer-native-replay003/report.json` test the explicitly listed phase
hypotheses using the actual007 controls. The current-control-to-next-state
hypothesis reproduces198/198 and227/227 physical10 states respectively,
including quaternion, position, velocity and the Y/VY components. Previous
control and same-row hypotheses differ. Those are separate bounded physical
replay results for these captures. This analyzer neither selects a phase
winner automatically nor turns them into equivalence of issued commands.

All of this analyzer's clock qualification, issued-command ACK, whole-step
phase, atomic cross-thread snapshot and paired-training flags remain false.
The observed physical replay scope does not close those gates.

## Reproduce offline

Run from the worktree with its current source on the import path. The output
must be a fresh file; the analyzer refuses to overwrite an existing report.

```powershell
python -m scripts.input_consumer_analysis `
  F:/AITrackmania/reports/headless-training-parity-20261021-r5/input-consumer007-live002 `
  F:/AITrackmania/reports/headless-training-parity-20261021-r5/input-consumer007-analysis002.json
python -m scripts.input_consumer_analysis `
  F:/AITrackmania/reports/headless-training-parity-20261021-r5/input-consumer007-live003 `
  F:/AITrackmania/reports/headless-training-parity-20261021-r5/input-consumer007-analysis003.json
```

Both reports bind all three source files and analyzer source by full SHA256.
The retained report SHA256 values are:

```text
analysis002 0e4c647763ed36b709a548f787af0fbb9351c7531d05d3870b763b6b575e3b02
analysis003 cb204c3e50195ebc9772b001036371bea9d310fd7cdb9cf7107930ad8d1d20a6
```

Six synthetic regression tests cover one-ULP retention, explicit brake encoding,
ambiguous API input retention, repeated-physics anchor rejection, held-out
relation failure and negative/nonfinite source rejection. The tests launch no
game, controller, plugin, or native engine.
