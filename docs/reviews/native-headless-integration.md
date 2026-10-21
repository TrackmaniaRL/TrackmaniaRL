# Experimental native headless integration review

Review date: 2026-10-19. This records bounded validation of the optional native
environment and its separate CPU QR-DQN example on TrackmaniaRL 1.2.13.
It is not a general production-backend or portable-native-build qualification.

## Source publication preparation, 2026-10-21

The branch now includes the MIT-licensed native driver, shim and selected
upstream source libraries under `native/`, with upstream attribution and source
provenance. It also ports explicit runtime-root/deployment checks from the
portable SDK source. These publication changes are source/build validation;
the historical engine runs below do not qualify freshly compiled binaries.
A complete public asset bootstrap remains outside this branch.

## Transport follow-up, 2026-10-20

The corrected terminal contract `tmrl.native-headless.terminal/v2` was exercised
from a normal factory reset with the frozen discrete Jimmy demonstration:
1777 continuing N1 decisions and one qualified terminal N1 decision. The last
boundary ACK arrived before zero-byte EOF during the next reader request. The
API preserved that final transition, reported `completedRequestedBoundary=True`
and left terminal features, reader, pending input and bootstrap absent.

All 1798 own native76 states matched the original pinned control, including the
1792-state domain 2100 through 20010 checked by the caller. The complete original
validator and collector agree on computed time 17715 and one checkpoint;
`IsValid=false` remains separate. Natural reap and driver close succeeded, the
parent buffers were unchanged and no owned processes survived. An independent
review checked the raw requests, responses, audits, source snapshots and capture.
This was a scripted demonstration, not a learned finish or client-parity test.

An earlier caller failed because it required availability at native clock 20070;
the fresh run made the result available at 20010. Both failed captures remain
preserved. Neither availability value establishes the physical finish timestamp.
The typed revision changes only an `Any` annotation after that transport fix.

Evidence: actual033b report
`4665553bb562477632057571779111a3d678baea4663aea301c52716f01cc179`,
independent-review manifest
`b601bc05ab4ff9e545cba16df6c62a34a26833478255a7004c862d826fb0a20a`.
The original source-port audit metadata is historical, not the current test report.

The CPU-only ext4 CI copy passed 1281 tests with one skip, all 292 source files
under mypy, Ruff and format checks. All 76 headless unit tests passed separately.
Five earlier distributed deadline failures passed with their original limits.
This environment also corrected four dependency-version drifts and Git newline
handling, so this is not evidence of a filesystem-only performance cause.

## Historical probes before the terminal-v2 correction

## Checked runtime cases

| Case | Recorded result | Independent checks |
| --- | --- | --- |
| Packaged control replay | One normal reset and 799 N5 decisions, ending by the explicit time limit | All 4015 native76 states, feature44 values, rewards and complete 192-byte requests/160-byte receipts matched the pinned reference. All receipts and applied-input audits were joined to owned native states. |
| Fresh CPU training example | 799 training decisions +1598 heldout decisions; 199 updates | All 2397 transitions and 12045 native76 states qualified. Initial evaluation and the first training episode matched the earlier pilot. All 199 non-timing metrics and the final online model tensors/Torch RNG matched that pilot at the same update count. |

The state counts include the administrative start prefix. That prefix is
excluded from policy experiences and learning credit. Native clock values are
raw engine units: normal reset 2230 and time-limit endpoint 42180 do not mean race
time or a physical finish timestamp.

Every checked native session reaped its child, preserved parent tape/guard
buffers and closed the driver with code 0 and no cleanup signals. Full driver
birth identity was saved. Parent/child live birth-identity snapshots were not
persisted, so historical process lookups were not used to invent that evidence.
Raw RX and its persistent staged copy matched exactly. Retention of temporary
Linux files is not a durability guarantee.

## Training and checkpoint scope

The example has explicit training/evaluation branches and separate phase IDs.
Only accepted training decisions enter replay or earn credit. One update is due
per four accepted training transitions; the checked 799-row episode produced 199
updates with remainder 3. Updates run after closing native resources, and each
episode takes a fresh policy snapshot.

The saved fresh-example checkpoint has the native ragged contract, update count
199, 12 online parameter tensors, 12 target parameter tensors and 12 finite Adam
slots at step 199. Target weights remain initial because the 1000-update
synchronization interval was not reached. Checkpoint inspection constructed no
learner/model and ran no optimizer, backward pass or native process.

This minimal example does not persist replay contents, actual sampled IDs,
sampler RNG or per-update model hashes. Ingestion/evaluation isolation is bound
to the executed source and saved decision records; it is not a complete
per-update computation replay. The checkpoint has no demonstrated resume or
automatic migration path from historical research checkpoints.

The final greedy evaluation did not establish improvement over the initial
policy. All three example episodes truncated at the configured duration limit.
No learned finish or generalization result is claimed.

## Version and lifecycle limits

The native runtime uses installed package modules without importing Python
scripts from a research directory. Original executables, shim, driver, map,
input tape and validation inputs remain external, hash-checked assets.
The 272 upstream Python files match the pinned main source after accounting
only for checkout CRLF/LF differences. Historical byte-hash gates were preserved.

The initial control probe's executed public facade was preserved separately.
Later direct-facade code differs only by a runtime-identity `typing.cast` wrapper.
The additional factory UUID directory leases passed four inert creation checks;
the engine probes used direct environment instances and do not qualify repeated
engine resets through that factory.

Cleaned-package natural EOF and N1–N4 stepping were not exercised by these
runtime probes. Historical assisted EOF evidence remains a separate layer.
The source keeps missing terminal features absent, distinguishes inconclusive
evidence from a terminal and uses nested cleanup paths, but this does not replace
a fresh natural-EOF probe. The known Popen/interruption-before-assignment gap
also remains outside normal-close validation.

The fixed prebuilt native layout, partial observations and reference assistance
remain explicit. There is no arbitrary snapshot restore, arbitrary map/build
support, shared-memory transport or established graphical-client trajectory
parity. Different timing windows do not establish a training speedup.

See the [backend guide](../../readme/native-headless.md) for configuration,
external assets, the terminal-aware integration boundary and code-sharing
limits. Driver/shim source is included under `native/`; a reproducible public
bootstrap for all qualified external inputs is still not included.
