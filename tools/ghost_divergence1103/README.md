# Offline ghost divergence diagnostics

This isolated tool reads saved files. It does not import TrackmaniaRL, start a
simulator, load a model, alter packages, or attach to a process. It writes only
explicitly requested new JSON files, using exclusive creation. Existing outputs
and evidence cannot be overwritten.

Run from the repository root using the existing development Python:

```text
python -m tools.ghost_divergence1103.diagnose_ghost_divergence --ghost decoded-ghost.json --headless decoded-headless.json --input-start-ms 10 --output new-report.json
```

Both input documents use `tmrl.ghost-divergence-series/v1`, a nonempty `map_uid`,
`coordinate_frame: world_xyz_y_up`, and strictly increasing `frames`. Each frame
contains an integer `time_ms` divisible by ten and optional `position_m`,
`velocity_m_s`, `control_sgb` (steer, gas, brake), `contacts`, and `phase`.
Vectors have three finite numbers. Units are metres and metres per second.
Missing observations must be absent or null, never interpolated or held forward.
Contacts can contain independently decoded wheel/contact context; unknown fields
must remain unidentified. `provenance` records the actual source and clock mapping.

The default thresholds are strictly greater than 0.001m and 0.001m/s. The report
contains the first observed position/velocity threshold crossings, first control
difference, counts, maximum errors, source cadences, countdown context, and ten
ticks before/after each event. A first state difference after a gap is reported
with its possible onset interval, not an invented exact tick. No clock shift is
searched or fitted. Nonoverlapping or incomplete channels cannot produce a
successful no-difference conclusion.

For the archived V7 traces, `race_ms = NativeSimClock - resetNativeT`. Controls
in POST records describe the interval ending at that timestamp. The boundary at
race0 closes an administrative neutral reset; the first policy interval ends at
race10. `--input-start-ms 10` therefore reports driving differences separately
from raw countdown/reset differences, which remain visible in the same report.
This convention is explicit and does not establish other clients' clock parity.

## Adapter for the saved oZeus/Jimmy inputs

`prepare_archived_inputs` accepts the existing Rust decoder's full `dump`,
`manifest`, and full countdown-inclusive `tape csv` exports, plus the closed
native trace archive, its receipt, and its saved factory configuration:

```text
python -m tools.ghost_divergence1103.prepare_archived_inputs --dump body.txt --manifest manifest.json --controls controls-full.csv --map-uid MAP_UID --archive session.tar.gz --receipt receipt.json --factory-config factory-config.json --ghost-output new-ghost.json --headless-output new-headless.json
```

Generate exports only with the existing read-only Rust commands (`dump` with
`--len` equal to manifest `header.body_bytes`, and `tape csv --from -2000`).
This adapter supports record-v10 and digital steering only. It consumes the full
record grammar and checks entity counts, timestamps, first raw car sample and
first decoded transform against the independent Rust manifest. The Python record
decoder port is named honestly in provenance; it is not presented as a compiled
Rust full-state export. Velocity comes from the stored log-speed and signed8-bit
direction, never position differences. The full tape's count/start are checked.

Archives are read in place without extraction. Their SHA must match the saved
receipt. Actual POST controls are joined to the corresponding pre-load write
audits, rather than reconstructed from a desired action. Native wheel getter
fields and cumulative delivered-hook contact counters are included when present.
These summaries are diagnostic context, not a new physical/contact qualification
or proof of coverage of every collision path. Ghost wheel rotations and raw,
uninterpreted ground-mode bytes are distinguished from native wheel getters.

Map identity is checked against the literal in the Rust-decoded body and the
saved factory's expected UID; this is not a fresh engine map/build readback.
Packed ghost velocity can cross a 1mm/s threshold without a physical fault.
Ghost state samples50ms apart cannot identify an exact first10ms physical drift.
The archive, original ghost, sparse raw states, and failed probes remain intact.

Verification: eight focused diagnostic tests, strict mypy, Ruff, and offline
comparisons of both the oZeus attempt and the successful Jimmy control. No new
physics or training is needed for these checks.
