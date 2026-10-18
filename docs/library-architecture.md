# Library architecture and data contracts

This guide describes TrackmaniaRL 1.2.13. RunSpec and checkpoint schemas remain 2.0.

The supported entry points are `RunSpec`, `resolve_run`, `Trainer` and
`__version__` from `trackmaniarl`, the contracts in `trackmaniarl.core` and the
component paths documented in the [SDK](../readme/sdk.md). The CLI entry point
is `trackmaniarl.cli:entrypoint`. Underscore-prefixed helpers are internal.
The CLI does not re-export them as a public facade.

## A decision and an update

Openplanet emits a 33-field frame. The environment checks session identity,
applies a control action and returns a transition. The feature pipeline turns
telemetry and map geometry into model inputs. The actor journals transitions.
the coordinator ingests them into replay and the learner samples training
batches. Updated policy snapshots return to the actor asynchronously.

This is the off-policy execution path. PPO uses the local cycle described below.

![Versioned runtime diagram](diagrams/runtime-architecture-preview.svg)

The [editable source](diagrams/runtime-architecture.excalidraw) and
[detailed runtime guide](../readme/architecture.md) cover process ownership,
authenticated remote transport, WAL durability and recovery after interruption.

## Default observations and model

The starter Trackmania configuration uses `LidarFeaturePipeline` with schema 5:
20 normalized telemetry features, a `[4, 60]` boundary lookahead tensor and a
60-element validity mask. Boundary channels contain left/right lateral and
forward coordinates in the car's frame. This is geometry-derived lidar, not
a rendered camera image or physical ray-cast sensor. Geometry must be supplied
for each new map and follow its route in order.

`LidarSensorEncoder` produces a 256-dimensional embedding. `IdentityTemporalCore`
adds no recurrent state. A dueling `ImplicitQuantileHead` and
`RandomQuantileStrategy` produce IQN action values. The actor uses the configured
exploration schedule during collection. Evaluation uses evaluation mode. The
composite encoder/core/head/strategy contract also supports Standard Q, QR-DQN
and FQF. Temporal and actor-critic alternatives have explicit requirements in
the [algorithm matrix](../readme/algorithms.md).

The default 78 actions combine 13 steering levels, two throttle levels and three
brake modes (none, full, timed tap). Steering is normalized to `[-1, 1]`. Throttle
and brake to `[0, 1]`. Timed taps have controller semantics, so action cadence
and backend affect the trajectory. Keyboard control discretizes steering and
lacks gamepad rumble collision detection. A backend change requires a benchmark.

## Replay and demonstrations

### PPO and vision data flow

PPO keeps collection and optimization in one process. A fixed policy snapshot
collects `sequence_length` transitions. Each transition records the sampled
pre-squash action, its log probability and the state value, in addition to the
usual reward and episode flags. `OnPolicySequenceSampler` returns that fresh
rollout once. The learner computes GAE and runs shuffled optimization minibatches
for `update_epochs`, then the collector receives a new policy snapshot.

An episode can end inside a rollout. The collector resets the game and image
history and continues collecting until the rollout is full. GAE does not propagate
advantage across that reset. A true terminal state has zero bootstrap discount,
while a time-limit truncation can use the value of its final pre-reset observation.
See [the PPO update explanation](../readme/vision.md#how-a-ppo-update-works).

The camera path uses `VisionEnvironmentFactory` to sample RGB after each telemetry
reset/step. Telemetry still determines rewards and race termination.
`VisionFeaturePipeline` resizes and normalizes RGB, optionally converts to grayscale,
and concatenates recent images along the channel axis. With the default settings,
the policy sees `[4,84,84]` and a rollout batch has `[1,T,4,84,84]`.
Independent CNNs supply features to PPO's action and value heads. Minibatches
flatten only batch/time axes, preserving the image axes.

The same `VisionSensorEncoder` fits the composite value model's sensor interface,
so an image-based IQN run uses the existing off-policy machinery. Camera frame
stacks are finite observation history, not a recurrent hidden state. Replay stores
the prepared image stacks and sampling does not capture or preprocess new images.
See [vision configuration](../readme/vision.md) for source contracts and memory use.

### Off-policy replay and recorded demonstrations

Replay stores episode identity, steps, raw/prepared observation contracts,
actions, reward, termination/truncation and demonstration metadata. Sequence
sampling stays within episode histories. N-step targets stop at episode
boundaries. True termination disables bootstrapping. Truncation preserves the
configured bootstrap contract. PER, expert fractions and elite sampling alter
sampling, never benchmark inclusion.

Demonstration archives bind map UID, geometry, timing and control alignment.
Do not relabel an archive from another map or change its metadata to bypass a
check. Record it again when the contracts differ. `offline-pretrain`, `bc-train`,
`train --demo`, exact `bc-train --resume`, DAgger and human recovery are described
in the [imitation-learning guide](../readme/imitation-learning.md).

## Checkpoints and evaluation

Training checkpoint schema 2.0 carries learner/optimizer state and runtime,
replay and sampler state appropriate to its training path. Architecture and
run fingerprints guard exact resume. Policy-only artifacts can benchmark or
initialize a compatible model, but do not contain a resumable optimizer.
The codec uses Zstandard-compressed Torch serialization. Use the library codec,
not a direct `torch.load` call and load only trusted artifacts.

The normal `benchmark` command evaluates the configured learner policy. It
does not attach `neighbors`, choose a demonstration or alter actions. Its
artifact records every attempt, errors, map identity, finish time, progress,
control latency and producer-frame skips. Conditional mean/median finish time
must be read alongside finish rate. Failed gates keep the results. See
[recording](recording.md) and the [full historical report](benchmarks/2026-09-08-v108-live.md).

### Dropped telemetry frames and evaluation validity

`telemetry_skipped_frames_total` counts skipped producer frames, not lost video
frames, failed physics updates, or necessarily missed policy decisions. The
environment may read several telemetry frames while holding one action to reach
its configured decision interval. Inspect skips together with
`step_race_time_ms_p99`, `step_race_time_ms_max`, measurement validity/count,
inference latency, controller-apply time and telemetry-wait time. A raw skip count
alone cannot identify whether the bottleneck is the game, CPU/GPU scheduling,
transport, recording, or policy inference.

Missing observations can delay feedback: the car continues under the currently
applied controls and a correction can arrive after the useful braking or steering
point. A recurrent or finite-difference feature also sees a less regular sequence.
The effect is trajectory-dependent. Do not assume every skip slows a lap by a
fixed amount, or that every collision was caused by a skip. Compare complete,
fresh series with the same policy, map, control cadence and recording settings
before attributing a difference to hardware or to the model.

Finish time normally comes from the terminal telemetry race clock, not from
`number_of_received_frames × nominal_frame_duration`. Skips therefore do not
directly subtract time from the reported lap. Nevertheless, delayed or missing
terminal observations and the telemetry clock's precision limit the measurement.
the evaluator has an elapsed-time fallback if a positive terminal race clock is
unavailable. Such a fallback is not evidence of a precise in-game finish time.
Keep clock-validity checks and video evidence and label the clock source. A
healthy same-clock finish notification is valid, unlike a regressing clock or a
nonterminal zero-duration step. In V108, the best telemetry time was 36.560 s,
while the recorded game overlay read 36.568 s.

Declare telemetry acceptance limits **before** running a benchmark. Preserve
every attempted trial, including slow laps, DNFs and telemetry errors. The strict
`--reject-telemetry-skips` option rejects the **whole series** when skips occur.
it does not calculate a better mean from the surviving laps. A rejected series
remains diagnostic evidence. Fix the setup and run a new complete series rather
than replacing individual attempts until a target passes. Report finish rate,
finish-only time statistics, all-attempt outcomes and telemetry quality together.

The [V108 report](benchmarks/2026-09-08-v108-live.md) recorded 458 skipped producer
frames and a maximum race-clock step of 80 ms over all 30 attempts. Its mean is
an observed result under those conditions, **not a zero-drop reference** or proof
that the same controller has an equivalent mean on another machine.

## Optional models and research provenance

`trackmaniarl.experiments.graph_iqn*` remains installable because its graph,
context and recovery components have independent tests, work with configurable
geometry and are dependencies of existing checkpoints and recovery fine-tuning.
They are opt-in and have no generalization guarantee. Removing older graph
classes would break the V6 inheritance chain, not merely remove an alias.

`experiments/sub37/` is repository-only and excluded from wheels. It contains
map-specific masks, nearest-neighbor action support and the historical candidate
runner. It imports the library. The public training path does not import it.
See [support status](support-status.md) for the rest of the surface.
