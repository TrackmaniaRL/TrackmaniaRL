# Native headless environment and the full QR profile

The native environment is independent of QR-DQN. Its current qualified action
contract is 78 discrete actions; models and learners are selected separately.
The historical QR preset below is one measured configuration, not a requirement
of the environment. The V8 factory must run inside the separately prepared V8
source deployment described below. The original installed V6/V7 package retains
its own protocol and rejects a V8 runtime configuration.

Use the neutral factory path in a new configuration inside that deployment:

```yaml
class_path: trackmaniarl.trackmania.native_headless:NativeTelemetryEnvironmentFactory
kwargs:
  config: # supply the qualified deployment configuration described below
    # runtime, assets, geometry, reward and event_source bindings
```

`NativeTelemetryEnvironment` and `NativeTelemetryEnvironmentFactory` export the
existing implementation. Historical `full_qr` import paths remain valid, so a
renamed factory path is an explicit new run configuration, not a checkpoint
migration. The V8 wire profile and native source pins keep their original names.

For an illustrative client-side replay, see the [TICK CPU-search demonstration](../docs/headless-tick-demonstration.md).
It is not a measure of synchronized execution or autonomous model performance.

## Install and prerequisites

Use Python 3.12 on Linux or WSL. Install the release containing this feature,
not an older PyPI release that predates it. For a wheel built from this PR, run:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install /path/to/trackmaniarl-1.2.17-py3-none-any.whl
```

Replace the wheel path with the actual built filename. For distributed training,
install the `distributed` extra from the same wheel, for example
`python -m pip install '/path/to/trackmaniarl-1.2.17-py3-none-any.whl[distributed]'`.
Choose a Torch installation appropriate for your host separately if the default does not match
its CPU or CUDA setup. See the [Python quickstart](python-quickstart.md) for
normal installation and [engineering tools](../experiments/full_qr/README.md)
for the distinction between packaged components and repository-only diagnostics.

The package does not include game assets, native binaries, private build receipts
or qualified event references. The caller supplies and qualifies these inputs.
V7 provisioning and V8 source deployment are separate; installing the wheel alone
does not prepare either runtime. The V7 example below assumes an already
provisioned and validated configuration. The V8 deployment requirements are
explained under [Preparing a deployment](#preparing-a-deployment).

## A bounded Python session

The installed library also exposes the separate V7 immediate-state API. It
requires Linux, a previously provisioned V7 configuration and the caller's
verified game assets. Set `TMRL_HEADLESS_CONFIG` to the resulting `config.json`;
this example starts a native process and always closes it:

```python
import json
import os
from pathlib import Path

from trackmaniarl.trackmania.native_headless import (
    NativeImmediateEnvironment,
    NativeRuntimeConfig,
)

config_path = Path(os.environ["TMRL_HEADLESS_CONFIG"])
settings = NativeRuntimeConfig.from_mapping(json.loads(config_path.read_text(encoding="utf-8")))
environment = NativeImmediateEnvironment(settings)
try:
    observation, info = environment.reset()
    result = environment.step_native(39, native_iterations=1)
    print(result["status"], result["actualIterations"])
finally:
    environment.close()
```

V7 reports physical observations and applied controls; it does not supply a
qualified reward or automatically insert training transitions. This API uses a
separate, bounded V7 profile, not the 150-second V8 factory described below.

## Model and observation choices

The default observations retain V5 preprocessing. Choose their encoder, temporal
core, head and algorithm independently. `StructuredMlpSensorEncoder` accepts
explicit named sensor shapes instead of requiring the V6 graph encoder:

```yaml
encoder:
  class_path: trackmaniarl.models.encoders:StructuredMlpSensorEncoder
  kwargs:
    input_shapes:
      physics: [60]
      track: [3, 88]
      context: [29]
      recovery: [8]
    output_dim: 192
    hidden_dim: 256
```

It flattens each sensor's event dimensions, uses sorted sensor names and preserves
batch/sequence axes. It does not rescale or replace measured input values.

For project-specific preprocessing, set `config.observation_pipeline` to a
`ComponentSpec` (`class_path` and `kwargs`). Each environment constructs a fresh
instance implementing `NativeObservationPipeline`: `reset_episode`,
`transform_observation`, `transform_terminal` and `transform_committed_terminal`.
Outputs must be a nonempty mapping of finite numeric sensors. Invalid features
close the provider before reward-event consumption. Returned observations own
their tensor and NumPy buffers, so a pipeline may reuse working buffers on later
steps without changing earlier observations; each snapshot has a copy cost.
Changing preprocessing does not change native action execution, source coverage
or reward semantics; the project must qualify the new representation for its task.

[`examples/native_observation_pipeline.py`](../examples/native_observation_pipeline.py)
is a small, tested example that selects position and velocity as a six-element
`float32` sensor. Copy the file into your own project, for example as
`my_project/native_pipeline.py`, and select it in the V8 environment config:

```yaml
observation_pipeline:
  class_path: my_project.native_pipeline:PositionVelocityPipeline
  kwargs: {}
```

The wheel does not install the repository's `examples` directory. The three
transform methods receive different regular, terminal and committed-terminal
frame types, respectively. Each provides `channels`: 23 finite values in native
graph order; indices 1–3 are position and 4–6 are velocity. The example uses
only those common fields. These native frame types are part of the experimental
V8 interface; do not treat the sequence as a complete Windows telemetry packet.
The same file defines `PositionVelocityFeatures`, which converts each observation
to a float32 Torch tensor before policy inference and inherits replay transition
collation. Configure it separately in the run specification, with a model encoder
that accepts the same named shape:

```yaml
components:
  feature_pipeline:
    class_path: my_project.native_pipeline:PositionVelocityFeatures
    kwargs: {}
  model_factory:
    class_path: trackmaniarl.models.factory:CompositeValueModelFactory
    kwargs:
      encoder:
        class_path: trackmaniarl.models.encoders:StructuredMlpSensorEncoder
        kwargs: {input_shapes: {kinematics: [6]}, output_dim: 192}
      # Also supply compatible temporal, head and strategy components.
```

This excerpt omits the other required run components. It is an untrained example,
not a drop-in replacement for qualified V5
preprocessing. Test and qualify the full representation for your task.

The run's separate `components.feature_pipeline` must accept that prepared mapping
and collate it for the selected learner. The default V5 component expects the four
sensors above. A custom representation needs a matching SDK pipeline; Torch
policies require tensor leaves at the policy boundary. The environment factory
does not silently replace this run component or its checkpoint fingerprint.
For Torch-based batching, choose NumPy dtypes supported by PyTorch. For example,
where `longdouble` has extended precision, it needs an explicit conversion or a
compatible custom pipeline. The environment does not silently reduce precision.
Use the ordinary `DiscreteValueLearner` for DQN/QR/IQN and
`StableDiscreteSoftActorCritic` for discrete SAC. The optional compiled QR learner
remains specific to its qualified QR profile.

| Choice | Current support and evidence |
| --- | --- |
| DQN, QR-DQN, IQN, discrete SAC | SDK update/action tests with both V6 graph and structured MLP encoders; these are synthetic compatibility tests, not four native training benchmarks |
| Recurrent models | The MLP preserves sequence axes; select a compatible SDK temporal core and replay schedule separately |
| Continuous SAC, REDQ, TQC, current Gaussian PPO | Their continuous actions do not satisfy this native 78-ID contract; an independently qualified actuator adapter is required |
| CNN/vision models | This profile supplies telemetry, not rendered images; a real image source and its timing contract are required |

## Historical full QR training profile

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

For a bounded diagnostic campaign, set both metadata fields explicitly:

```yaml
metadata:
  campaign_mode: diagnostic
  requested_wall_time_seconds: 1800
```

The watchdog, storage reserves, final checkpoint verification and process
ownership checks still apply. The result is labelled `diagnostic` and is not a
four- or seven-hour qualification. Without this explicit mode, the supported
qualification durations remain 14,400 and 25,200 seconds.

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
- `campaign_policy` defines the campaign clock, storage reserves and learning
  schedule checks; `campaign_budget:BudgetSettings` configures the watchdog;
- `campaign_worker` owns worker startup, shutdown and counter reporting, keeping
  the budget clock separate from heartbeat and native-session liveness.

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
or copying a source model does not establish parity. The original raw V7 source manifest and build
receipt, when needed to provision its predecessor, remain private inputs to
`python -m trackmaniarl.trackmania.native_headless.provision_immediate`. Its SHA requirement is retained even
though those private files are excluded from the distribution.

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
