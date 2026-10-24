# Full QR headless profile

The full profile keeps the historical V5 feature pipeline and incident-gated V6
graph encoder: 192 features, 78 actions, 64 fixed quantiles, a dueling head,
batch 256, replay capacity 1,000,000, learning rate 0.0001, gamma 0.995,
10,000 warmup transitions and 0.25 updates per accepted transition. Actors and
the learner use the ordinary SDK replay, WAL and checkpoint contracts.

Native execution requires Linux. Imports and configuration checks do not
authorize a real process. Game assets, the shim, geometry, primary captures and
pinned reference manifests are caller-supplied and excluded from the Python
distribution. There is no fallback to a Windows client or common6 observation.

Actor/learner campaigns require the `distributed` extra, for example
`uv add "trackmaniarl[distributed]"`. Native actors do not require Openplanet,
screen capture or a Windows control backend. W&B logging remains optional.

## Package boundaries

The environment entry is
`trackmaniarl.trackmania.native_headless.full_qr.factory:FullQrEnvironmentFactory`.
Its explicit `config` binds runtime paths, the whole asset-manifest SHA, map
identity, geometry, historical rewards, archive location and a qualified event
source. `native_events:QualifiedNativeEvents`, in the same package, requires the
paired Finish and physical/motor reference manifests and their SHA pins.

Campaign components live under `trackmaniarl.experiments.full_qr`:

- `components:BatchedIncidentEncoder` preserves V6 computation and parameters
  while accepting leading batch/sequence axes;
- `training_performance:CompiledEncoderQrLearner` provides explicit, qualified
  execution choices without changing the objective;
- `campaign_launcher:LaunchSettings` and `launch` supervise independent actors
  and learner processes with owned identities and pidfds;
- `campaign_budget:BudgetSettings` and `campaign_clock` keep the budget clock
  separate from heartbeat and native-session liveness.

These components do not globally change the default learner, runtime, CLI,
reward or action table.

## Preparing a deployment

The original V6/V7 runtime retains its original protocols and time domains.
The full 150-second V8 profile uses a separate source deployment:

```python
from pathlib import Path

import trackmaniarl

from trackmaniarl.experiments.full_qr.runtime_profile import build_source_deployment

library_parent = Path(trackmaniarl.__file__).resolve().parent.parent
receipt = build_source_deployment(
    library_parent,
    Path("/srv/trackmania/full-qr-source"),
)
```

The destination must be fresh, canonical and on Linux storage outside `/mnt`.
The builder checks each exact source prerequisite before copying the package;
it neither starts native processes nor qualifies the result. Dependencies must
already be installed in the selected worker interpreters. Run workers from the
separate deployment, not a process that already imported the original V7 modules.

Supply a reviewed asset manifest and independently qualify the complete
reference/source binding. Changing a hash, accepting an old diagnostic manifest,
or copying a source model does not establish parity. The original raw V7 build
receipt, when needed to provision its predecessor, remains a private input to
`scripts/headless_provision_immediate.py`. Its SHA requirement is retained even
though that private receipt is excluded from the distribution.

## Data and termination contract

### Inputs and preprocessing

The provider accepts an integer action ID in `0..77`. Its decomposition is
`steer_bin, variant = divmod(action_id, 6)` and
`gas, brake_mode = divmod(variant, 3)`: 13 steering bins, two gas settings,
and off/full/tap brake modes. SDK control vectors use `[gas, brake, steer]`;
native input records use `[steer, gas, brake]`. The SDK tap sentinel `-1`
becomes a five-record brake plan `[1, 0, 0, 0, 0]`, never a negative native
brake value. Action 36 is neutral; action 39 applies gas with neutral steering.

The separate V8 deployment applies the measured signed-byte steering plateaus
`(-127, -106, -85, -63, -42, -21, 0, 21, 42, 63, 85, 106, 127) / 127`,
rounded to float32. For example, requested steering `0.5` becomes approximately
`0.496063`. The generic SDK action table remains unchanged. Issued commands,
observed control channels and historical reward inputs have distinct roles;
they are not required to be numerically identical.

`NativeGraphFrame` carries the 23 channels actually consumed by the historical
V5 calculations: race time, position, velocity, direction, speed, RPM, gear,
four wheel-slip channels, skidding count, flying duration, adherence and three
observed control channels. It is not a complete 33-field Openplanet packet.
Unused positions in its private indexing carrier remain NaN so accidental
dependencies fail validation. Production joins use measured adherence at the
same vehicle boundary; a getter-default diagnostic is not a measured substitute.

`NativeGraphFeaturePipeline.transform_observation` produces finite float32
tensors, with causal state cleared by `reset_episode`:

| Tensor | Shape | Processing |
| --- | --- | --- |
| `physics` | `(60,)` | Motion, orientation, observed controls, curvature and skidding; includes normalized speed, finite-difference rates and bounded controls |
| `track` | `(3, 88)` | Geometry in the track representation; the encoder divides these values by 50 |
| `context` | `(29,)` | Road clearance, vehicle context, progress frequencies and speed-dependent previews; clipped to `[-4, 4]` |
| `recovery` | `(8,)` | Causal changes and decaying incident memory; clipped to `[-4, 4]` |

The encoder's 192 output features are a learned representation, not 192 raw
telemetry channels. These preprocessing methods belong to TrackmaniaRL, not
Openplanet. Public API declarations alone do not establish current native field
offsets, freshness, units or write access. Older `native44` pending-reference
documentation describes a different profile and must not replace this contract.

Ordinary accepted decisions span 50 native race milliseconds, with a maximum
150-second episode. Motor, material, contact and adherence observations bind to
the current owned process and physical clock. Finish requires the paired
positive Goal source. A final collector may supply real boundary observations
when the engine ends between ordinary IPC publications; it cannot invent a
missing physical sample or treat EOF as Finish.

Committed score time and observed physical-boundary time are reported
separately. Terminal replay discounts remain zero. Causal feature history,
historical rewards and coverage checks apply before a transition enters replay.
Invalid or incomplete source data abort the environment.

Motor rendering explicitly uses virtual 150 FPS. This is not bitwise Windows
render jitter: a documented right-turn notification can occur at 1800 ms rather
than 1750 ms. Preserve this difference in comparisons.

## Storage, ownership and stopping

Each actor receives private trace/work state and a fresh archive subdirectory.
Archives must be outside runtime/session roots. Retention prunes only that
actor's closed, verified archives; it never deletes live traces, old campaigns,
another process's files or WSL backups.

The campaign retains 10 GiB on the host system volume and 4 GiB on private work
and explicitly configured report volumes. WSL retains its separate Windows
system-mount reserve. Reports do not assume a particular secondary drive letter.
A trace quota remains required: periodic disk checks are not a kernel quota.

The qualified four-hour and seven-hour profiles require explicit budget
metadata. A transition count or early checkpoint is not completion. A successful
parent result requires the complete elapsed budget, `wall_clock_limit`, final
checkpoint identity, all worker exits zero and empty forced/cleanup/survivor
lists. Independent readback and exact identity checks complete the audit.

`monotonic_raw` budgets use Linux `CLOCK_MONOTONIC_RAW`; legacy `monotonic` is
explicitly separate. Heartbeat age and stall detection use ordinary monotonic
time. Never subtract timestamps from different domains. Compiler warmup occurs
before readiness and does not advance the optimizer.

## Execution and measurement

Choose actors by measurement. CUDA Graphs apply only when explicitly selected
for the qualified encoder; CPU actors remain eager. Pinned transfer/lookahead
remains optional because its isolated benefit did not establish a production
advantage. The default ordinary learner is unchanged.

Checkpoint writing already runs asynchronously. Numeric NumPy replay arrays
now use separate Torch storage records, avoiding large protocol-2 Python byte
strings while retaining NumPy arrays on load and the restricted weights-only
reader. Old checkpoints load. New arrays require this reader; older releases
are not assumed to understand the new restore helper.

See [measured results and their limits](../docs/benchmarks/full-qr-headless.md).
Published package names differ from the frozen laboratory source paths. Preserve
old deployments and checkpoints; do not silently resume a renamed configuration.
