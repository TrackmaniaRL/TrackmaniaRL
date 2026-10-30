# Camera vision with every RL algorithm

This guide applies to TrackmaniaRL 1.2.16 with RunSpec 2.0.

Camera observations are independent of the RL algorithm. All public RL learners
support image models: Q/DQN, QR-DQN, IQN, FQF, SAC, REDQ, TQC, stable discrete SAC
and PPO. The same `VisionFeaturePipeline` and `VisionSensorEncoder` supply images
and CNN features across these families.

The Trackmania template generates these complete camera configurations:

| Algorithm | Configuration | Model |
| --- | --- | --- |
| Q/DQN | `run-q-vision.yaml` | Composite scalar Q |
| QR-DQN | `run-qr-vision.yaml` | Composite fixed quantiles |
| IQN | `run-iqn-vision.yaml` | Composite implicit quantiles |
| FQF | `run-fqf-vision.yaml` | Composite learned fractions |
| SAC | `run-sac-vision.yaml` | CNN actor and twin Q critics |
| REDQ | `run-redq-vision.yaml` | CNN actor and Q ensemble |
| TQC | `run-tqc-vision.yaml` | CNN actor and quantile ensemble |
| Stable discrete SAC | `run-discrete-sac-vision.yaml` | Categorical CNN actor and twin Q critics |
| PPO | `run-ppo-vision.yaml` | CNN actor and state-value critic |

Use any filename from this table with `validate`, `train`, `resume` and `benchmark`.
Off-policy camera models also use the distributed actor/learner runtime. Capture
runs on each actor's desktop. PPO uses local on-policy collection.

PPO is a supported local on-policy algorithm. The Trackmania project template
generates `run-ppo.yaml` (telemetry) and `run-ppo-vision.yaml` (camera), alongside
the default IQN/lidar `run.yaml`. Configure the map UID, map file and geometry in
the selected file using the usual Trackmania setup instructions.

```powershell
uv run trackmaniarl init my-agent --template trackmania
cd my-agent
uv add "trackmaniarl[trackmania,distributed,vision]"
uv run trackmaniarl validate run-ppo-vision.yaml
uv run trackmaniarl train run-ppo-vision.yaml
uv run trackmaniarl resume run-ppo-vision.yaml artifacts/RUN/checkpoints/CHECKPOINT.pt
uv run trackmaniarl benchmark run-ppo-vision.yaml artifacts/RUN/checkpoints/CHECKPOINT.pt
```

`validate` uses synthetic images and performs a learner update plus a checkpoint
round trip without opening the game or capturing the desktop. For telemetry PPO,
use `run-ppo.yaml`. It does not need the vision extra.

## Image observations

`trackmaniarl.trackmania.vision:VisionFeaturePipeline` accepts RGB `uint8` arrays
with shape `(height, width, 3)`. It resizes with bilinear antialiasing, optionally
converts to grayscale, scales to `[0, 1]`, and concatenates consecutive frames on
the channel axis. Defaults are four grayscale 84x84 frames, giving `(4, 84, 84)`
float32 observations. On reset it repeats the first frame. Replay stores completed
stacks. Sampling a batch does not advance the frame history.

Configure the feature pipeline through `kwargs.config`:

```yaml
config: {width: 84, height: 84, grayscale: true, frame_stack: 4}
```

Match the model factory's `kwargs.channels` to `frame_stack` for grayscale or
`3 * frame_stack` for RGB. The bundled PPO config disables running observation
normalization because image features already have a fixed range.

## Native headless schematic vision

Treat this representation as experimental. For the established native starting
profile, retain [V5 telemetry/geometry](headless-full-qr.md#model-and-observation-choices).
Schematic vision has its own observation contract and must not be advertised as
training on Trackmania screenshots.

Native execution can feed the same CNN without a game window, screen capture,
Openplanet, EGL or a graphics driver. This opt-in source is a **schematic
perspective road ribbon**, not Trackmania RGB. It projects the recorded left/right
boundary coordinates with depth testing. White edge bands and a solid background
are visualization colors, not observed game materials or terrain. There are no
textures, HUD, car models, obstacles or inferred scene meshes. Virtual lidar
extensions beyond `recorded_count` are excluded.

Start from a qualified native configuration described in the [headless
guide](headless-full-qr.md), then add its observation component:

```yaml
components:
  environment:
    class_path: trackmaniarl.trackmania.native_headless:NativeTelemetryEnvironmentFactory
    kwargs:
      config:
        # Retain the qualified runtime/assets/event_source/geometry/reward bindings.
        observation_pipeline:
          class_path: trackmaniarl.trackmania.native_headless.vision:NativeSchematicVisionPipeline
          kwargs:
            geometry_path: /srv/trackmania/assets/track.geometry.npz
            expected_map_uid: YOUR_MAP_UID
            camera: {width: 84, height: 84, horizontal_fov_degrees: 90,
                     camera_height_m: 3, downward_pitch_degrees: 10, far_m: 120}
            vision: {width: 84, height: 84, grayscale: true, frame_stack: 4}
  feature_pipeline:
    class_path: trackmaniarl.trackmania.vision:PreparedVisionFeaturePipeline
    kwargs:
      config: {width: 84, height: 84, grayscale: true, frame_stack: 4}
```

This is a component override, not a complete deployment preset. Use exactly the
same map-bound geometry for the environment and camera. Replace the whole desktop
environment component; `capture` and `window_capture` are not native options.
For a composite DQN/QR/IQN model, select this encoder under
`components.model_factory.kwargs.encoder` and match the temporal/head feature
dimension to `output_dim`:

```yaml
class_path: trackmaniarl.trackmania.vision_models:VisionSensorEncoder
kwargs: {channels: 4, output_dim: 256, hidden_dim: 64}
```

The native factory still accepts only the qualified 78 discrete action IDs.
Continuous SAC/REDQ/TQC and the current Gaussian PPO require a separate qualified
action adapter even though their models support images. The graph-specific
compiled QR learner is not an image learner; use the ordinary value learner or
discrete SAC with a compatible CNN.

`NativeSchematicVisionPipeline` renders synchronously from the actual frame's
position and forward direction, once per accepted 50 ms observation. The camera
is explicitly world-upright: the recorded channels do not determine the game's
camera roll. Partial Goal frames use their actual sampled physics time. A
committed score does not invent an image at a later score timestamp, and a score
with an unchanged physical sample does not append another history frame.
`last_frame_time_ms` and read-only `last_rgb` expose the latest camera sample for
diagnostics; the environment's `observed_physics_time_ms` is the corresponding
physics clock. The separate virtual 150 FPS motor clock does not render images.

Each native environment owns its four-frame history, cleared on reset. It returns
`{"images": float32_CHW_stack}`; `PreparedVisionFeaturePipeline` validates and
unwraps this tensor without rendering, normalization or stacking again. Camera
and preprocessing dimensions must agree. Native inputs, rewards, source coverage
and EOF/Goal guards are unchanged; a render error closes the provider through the
existing environment failure path.

Train fresh weights for this representation. Existing game-RGB vision weights
are not qualified for schematic images, and transfer back to game screenshots
is unproven. This is a different observation contract from V5/V6 telemetry, so
use a new run configuration and matching checkpoint fingerprint.

### Choosing a native observation profile

Compare useful policies, not just generated frames or accepted transitions per
second. V5/V6 and the ribbon CNN expose different information and have different
architectures. A comparison therefore selects a complete training profile; it
does not isolate the causal effect of images versus lidar.

Use the same pinned map, reset, calibrated actions, rewards, termination guards,
actor resources and QR training recipe. Train fresh weights for each profile
using at least three paired training seeds. Predeclare an equal wall-clock budget
from learner readiness, including replay warmup and checkpoint pauses; report
compiler/startup and shutdown time separately. Run profiles sequentially and
alternate their order across seeds to avoid competing for the same GPU.

Evaluate each final checkpoint without learning or exploration, on that same
map and reset. Report completion per training seed, actual Goal time conditional
on completion, progress and failure reasons; an unfinished run has no finish
time. Repeated deterministic evaluations check repeatability, not independent
success probabilities. Report accepted transitions, updates and throughput
separately from policy quality. Keep every failed or incomplete run visible.

A short screening study may remain inconclusive. Retain the established profile
if completion and training-seed stability do not show a clear improvement; do
not promote schematic vision because its renderer is cheaper. This comparison
does not establish generalization to other maps, game-RGB transfer, or performance
with a full million-transition replay.

Budget replay memory explicitly: one float32 `(4,84,84)` stack is 112,896 bytes.
Two independent stacks per transition already cost about 441 MiB at capacity
2,048, before replay/model overhead. Replay links consecutive next observations
where possible, but retains whole observation stacks. Do not copy the telemetry
preset's million-transition capacity into this representation.

### Lossless compact ribbon observations

For this schematic renderer, set `observation_pipeline.kwargs.compact: true`,
then select these matching SDK components:

```yaml
feature_pipeline:
  class_path: trackmaniarl.trackmania.native_headless.vision:RibbonVisionFeaturePipeline
  kwargs:
    config: {width: 84, height: 84, grayscale: true, frame_stack: 4}
# Under components.model_factory.kwargs:
encoder:
  class_path: trackmaniarl.trackmania.native_headless.vision:RibbonVisionSensorEncoder
  kwargs:
    config: {width: 84, height: 84, grayscale: true, frame_stack: 4}
    output_dim: 256
    hidden_dim: 64
```

Both components must match the native pipeline's `vision` configuration. Replay
and transport keep `(frame_stack,height,width)` uint8 palette IDs. The encoder
expands them on its device to the same normalized RGB or grayscale values before
the unchanged CNN. This is lossless for the three ribbon colors; it is not a
compression format for desktop screenshots. One default stack uses 28,224 bytes,
four times smaller than float32 grayscale or twelve times smaller than float32
RGB. Frame stacks are still stored whole: one million observation stacks alone
would require about 26.3 GiB, before terminal next observations and other state.

Optional `storage: base3` packs five palette IDs into one byte. Set it on all
three matching components: the native observation pipeline, feature pipeline
and sensor encoder. A default stack then uses 5,645 bytes instead of 28,224;
one million observation stacks use about 5.26 GiB. The decoder expands an exact
palette lookup on the model's device, retaining every pixel, frame and CNN
parameter. Padding in the final byte must be zero. This option has its own model
representation fingerprint; start a fresh run when changing storage. These
figures describe observation storage, not total replay or checkpoint memory,
and do not establish a training-speed improvement.

Checkpoint capacity needs its own RAM and disk budget. An asynchronous replay
snapshot remains resident while training continues. The codec streams compressed
output, but loading expands a temporary archive beside the checkpoint before
restoring its contents. Compressed size is not the required free-space bound.
Generic runs can set top-level `checkpoint_root` independently of `artifacts_dir`
to place both saved checkpoints and that loading workspace on another volume.
The default `TorchCheckpointCodec.max_decompressed_bytes` is 8 GiB; larger replay
archives need an explicit, measured limit in that component's `kwargs` and enough
space on the artifact volume. Small-replay throughput measurements do not qualify
a million-entry vision replay or its peak checkpoint memory.

The renderer conservatively culls spatial blocks before projection, preserving
the depth-tested raster result. Diagnostic `last_rgb` expands colors only when
requested. These optimizations do not skip physics, contacts, rewards or images.

For the optional compiled CPU rasterizer, install
`trackmaniarl[native-vision]` and set
`observation_pipeline.kwargs.camera.raster_backend: numba`. The default remains
`numpy` and does not import Numba. The compiled path retains float64 projection,
the same clipping, inverse-depth test, pixel-center convention and palette. It
uses no fast-math or parallel depth reductions. Its first image compiles the
kernel once per process; compilation is not persisted into installed packages.
Include this startup cost when measuring short runs. Ordinary RGB diagnostic
buffers retain the NumPy implementation.

CPU replay tensors are stacked once into owned batch storage before checking
every shape, dtype and value bound. Byte-image checks share one NumPy view;
floating-point tensors retain the ordinary Torch validation and gradients.
PER draws retain the scalar search's comparison order and conditional
RNG calls, including uniform/expert mixtures. Unchanged tree weights skip only
exactly zero prefix-sum updates. One-step replay gathers each column once for the selected
batch. Rows own their data independently of replay, including repeated sample
IDs, and terminal overrides retain their original checks. Multi-step returns
keep their existing traversal. Uniform byte-image bounds use minimum/maximum
reductions; heterogeneous bounds still validate every coordinate.

The native host also keeps a sorted index of actual POST
clocks, avoiding a full history scan per tick. Original evidence is retained;
duplicate, future-event, receipt, causal-clock and EOF checks remain in force.

For QR on CUDA float32, `FixedQuantileHead` accepts
`kwargs.config.dense_cuda_selected: true`. It uses the existing dense selected
action calculation for training batches of at least 64 frames. It preserves the
quantile objective and parameter layout; floating-point accumulation can differ
slightly from the sparse formula. Other devices and smaller batches retain the
existing calculation. This option does not enable the graph-specific compiler.

For stateless QR frame batches, the opt-in
`trackmaniarl.experiments.full_qr.compiled_frame_qr:CompiledFrameQrLearner`
compiles the combined forward and quantile loss on Linux CUDA float32. It works
with a compatible CNN or graph encoder. CPU actor replicas retain eager
execution. Recurrent histories and auxiliary objectives require the ordinary
learner. Replay sampling, priority validation, Adam, clipping, target updates and
diagnostics retain their existing order.
Run setup and updates on the same learner thread. Moving a warmed CUDA graph
to another thread is rejected before CUDA execution; actors remain separate
processes and checkpoint writes may still run in the background.

Set `compiler_warmup` to a canonical `path` and `sha256` for an actual
qualification batch made by the configured feature pipeline and sampler. Save
the batch as a bounded, weights-only archive:

```python
from dataclasses import fields
from hashlib import sha256
from trackmaniarl.core.builtins import TorchCheckpointCodec

TorchCheckpointCodec().save(
    {"schema": "tmrl.compiler-frame-batch/v1",
     "batch": {field.name: getattr(batch, field.name) for field in fields(batch)}},
    path,
)
compiler_warmup = {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
```

Both compressed and expanded warmup inputs are bounded to 256 MiB. Warmup
restores RNG state, verifies unchanged online/target state and performs zero
optimizer updates. Training checks the same frame shape, dtype, gamma and
importance-weight layout before entering the compiled function. A PER batch
therefore needs the PER weight layout in its warmup archive. Parameters and
checkpoint names remain unchanged; compiled float32 arithmetic can differ in
rounding. Include initial compilation and checkpoint time in whole-run measures.

Local off-policy training can set `training.reuse_environment: true` to retain
construction and reference checks between episodes. Each episode still calls
`reset(seed=run_seed + episode_index)` and clears image history; session shutdown,
evaluation and failure paths close the environment. Leave this disabled for
factories that require reconstruction. Distributed actors already retain an
environment. PPO already owns its rollout environment and rejects this option.

Native warm resets retain the original driver's watchdog and disk quota. Before
reusing a driver, the provider requires at least 180 seconds of its original
deadline remaining, plus the existing next-episode disk reserve; otherwise it
rolls over to a fresh owned driver. This headroom check is not a guarantee that a
slow host completes an episode: the original watchdog still fails closed.

Budget distributed chunks against the uncompressed message limit. At the default
16 MiB limit, 128 transitions containing two independent float32 grayscale stacks
already exceed the limit. Use smaller chunks (for example 32) or the compact
representation, and include metadata overhead when sizing batches. A small
compressed file does not bypass the decompressed-size guard. Start a fresh run
when changing representation or execution configuration; fingerprints remain
strict.

Render at the requested model resolution (default 84x84); the CPU rasterizer
uses bounded scratch batches and leaves the GPU available to the learner. Measure
the complete workload before choosing actors. The historical **4.5004x** telemetry
pilot predates the brake-tap correction and is not a vision benchmark. Renderer
FPS, an encoder-only probe or a short CUDA update cannot establish whole-training
speed or time to a useful model.

`trackmaniarl.trackmania.vision_models:VisionPpoModelFactory` supplies independent
CNN encoders for a bounded Gaussian actor and state-value critic. Actions are
`[accelerator, brake, steer]` with bounds `[0, 0, -1]` to `[1, 1, 1]`.
`VisionSensorEncoder` also implements the existing composite value encoder
contract: use it with `CompositeValueModelFactory` and scalar Q, QR, IQN or FQF
heads. Match `output_dim` to the temporal core's `input_dim`.

For SAC, REDQ, TQC and stable discrete SAC, use
`trackmaniarl.trackmania.vision_models:VisionActorCriticModelFactory` with
`kwargs.algorithm` set to `sac`, `redq`, `tqc` or `discrete-sac`. Fields in `kwargs.config`
are `channels` (4), `hidden_dim` (256), `critic_count` (10), `quantile_count` (25)
and `action_count` (78). Ensemble size applies to REDQ/TQC, quantile count to TQC
and action count to discrete SAC. Keep the standard 78 actions for Trackmania.
Every actor and critic owns an independent CNN and optimizer gradients reach it.

Custom models can reuse `VisionSensorEncoder` as an observation encoder in the
existing actor, critic or composite model contracts. It preserves batch and time
axes, so value models can combine it with the available temporal cores. A model
that explicitly expects lidar vectors or graph nodes must use an image encoder
or explicitly combine both modalities. Raw images are not interchangeable with
telemetry inputs to an already trained checkpoint. Camera behavior cloning uses
`run-bc-vision.yaml` and aligned RGB/action archives. See the
[camera BC guide](vision-bc.md) for training, resume, evaluation and dataset import.
Telemetry-only DAgger and graph-recovery archives cannot supply missing camera frames.

## Lidar and paired lidar + vision

The Trackmania scaffold also generates `run-ALGORITHM-lidar.yaml` and
`run-ALGORITHM-lidar-vision.yaml` for `q`, `qr`, `iqn`, `fqf`, `sac`, `redq`,
`tqc`, `discrete-sac` and `ppo`. For example:

```powershell
uv run trackmaniarl validate run-sac-lidar-vision.yaml
uv run trackmaniarl train run-sac-lidar-vision.yaml
```

Configure map geometry and the viewport crop before live training. The camera
extra is needed only for live image capture. Lidar here means the project's
geometry-derived boundary lookahead plus telemetry, not a physical laser scanner.

Paired configurations set `VisionEnvironmentFactory.kwargs.include_telemetry: true`.
Reset and step then return `{"telemetry": raw_telemetry, "images": rgb}`.
`LidarVisionFeaturePipeline` applies the existing lidar and image pipelines to
their respective inputs, producing `{"lidar": lidar_mapping, "images": image_stack}`.
Its `kwargs.lidar` contains the ordinary lidar configuration and `kwargs.vision`
contains the image configuration. It resets both pipelines between episodes.
Batch collation uses prepared observations and never advances either history.

`LidarVisionSensorEncoder` encodes each branch independently, concatenates the
features and applies a learned linear projection with SiLU. Configure its
`kwargs.lidar` using `LidarSensorConfig`, its `kwargs.channels` to match the image
stack and its `kwargs.output_dim` to match the downstream model. It preserves batch
and sequence axes. Paired pipelines require lidar `history_length: 1`. Use a
temporal core in a composite value model for sequence learning. Image frame
stacking remains configurable independently. The default templates are feedforward.

For SAC, REDQ, TQC, discrete SAC and PPO,
`trackmaniarl.models.sensor_actor_critic:SensorActorCriticModelFactory` accepts:

```yaml
algorithm: sac
encoder:
  class_path: trackmaniarl.trackmania.multimodal:LidarVisionSensorEncoder
  kwargs:
    lidar: {output_dim: 256}
    channels: 4
    output_dim: 256
config:
  feature_dim: 256
  action_low: [0, 0, -1]
  action_high: [1, 1, 1]
```

These fields belong under the factory's `kwargs`. Each actor and critic gets a
separate encoder instance. `feature_dim` must equal encoder `output_dim`.
For lidar alone, select `BatchedLidarSensorEncoder` with `kwargs.config` containing
the lidar sensor configuration. For images alone, select `VisionSensorEncoder`.
The composite Q/QR/IQN/FQF factory accepts the same encoder components.

Every actor and critic calls `encoder(observation)` with one complete observation
argument. Mapping and tuple unpacking belongs inside a custom encoder, not in the
actor. Models with several sensors therefore share one observation contract.
Action bounds must be finite, ordered and representable in float32.

Integration tests exercise updates, both sensor gradients, training and checkpoint
resume for all nine RL families with lidar and with paired inputs. This does not
establish live driving performance or atomic camera/telemetry synchronization.
For paired behavior cloning use `run-bc-lidar-vision.yaml` and archives containing
aligned RGB, telemetry and actions. See the [paired BC instructions](vision-bc.md#paired-lidar--vision).

Off-policy camera templates use a 2048-transition replay buffer, batch size 32
and 512 warmup transitions. Increase capacity only after checking memory usage.
Their replay stores image stacks as float32, just like the PPO rollout below.
Synthetic integration tests cover an update, training and checkpoint resume for
every RL learner family. This verifies software compatibility, not lap times.

## Live frame source

`trackmaniarl.trackmania.vision_environment:VisionEnvironmentFactory` wraps the
Openplanet environment. Its `kwargs.config` is the standard Trackmania environment
configuration. `kwargs.capture` selects a desktop rectangle in pixels:

```yaml
capture: {left: 0, top: 0, width: 1280, height: 720}
```

Set this rectangle to the game viewport and keep it visible and unobstructed.
Capture uses the optional [MSS backend](https://python-mss.readthedocs.io/v10.2.0/examples.html),
including conversion from BGRA to RGB. Windows and Linux/X11 desktop capture
depend on the host display. Headless/Wayland capture is not provided by this wrapper.
The feature pipeline and models can also consume RGB frames from a custom environment
without installing MSS. `VisionEnvironment` accepts a `FrameSource` implementation
with `capture()` and `close()` for alternative camera sources.

Each reset/step obtains telemetry and then captures a fresh image. The policy
observes images. Telemetry still drives rewards, progress, finish detection and
controls. Camera and telemetry are not synchronized atomically. The environment
reports `vision/capture_ms` per observation. Keep camera, crop and display settings
consistent between training and evaluation. Telemetry-only demonstration files
cannot train this vision policy because they do not contain image observations.

## PPO lifecycle and resource limits

Use `OnPolicySequenceSampler`, `n_step: 1`, and a replay capacity at least as large
as `training.sequence_length`. PPO collects a fixed number of fresh transitions,
updates for the configured epochs, and starts another rollout. Rollouts may cross
episode boundaries. GAE stops at episode ends and bootstraps time-limit truncations.
`total_transitions` must be divisible by `sequence_length`. Distributed actor/learner
commands remain off-policy only.

The vision template keeps 2048 transitions. Float32 image stacks consume much more
memory than telemetry: two `(4,84,84)` observations per transition require roughly
441 MiB at this capacity, before learner tensors and overhead. Adjust rollout size,
capacity, minibatch size and image size together for the host's RAM/VRAM.

Checkpoints restore model, optimizer, normalizers, learning-rate progress, scaler
and RNG. Resume restarts the live environment and clears image history at an episode
boundary. It cannot restore the running game's exact visual state. Automated tests
exercise synthetic image rollouts and checkpoint continuation, not racing performance.

## How a PPO update works

The actor CNN turns the image stack into a feature vector. The Gaussian head
predicts an action mean and learns a state-independent standard deviation. During
training it samples a latent action, applies `tanh`, and rescales each coordinate
to the accelerator, brake and steering bounds. Evaluation uses the mean instead
of sampling. A separate CNN and value head estimate the future discounted return.

Collection saves the latent action, its behavior-policy log probability and the
value estimate before any optimization. PPO uses those saved values to compare
the updated policy against the policy that actually drove the rollout.

The one-step value error is `reward + bootstrap_discount * next_value - value`.
GAE accumulates these errors backward, decaying them by the discount and
`gae_lambda`. An episode end stops that recursion. A true termination also removes
the next-value bootstrap, while truncation keeps it. Returns are advantages plus
the saved values. Advantages are normalized over the rollout before optimization.

For each minibatch, the probability ratio is
`exp(new_log_probability - behavior_log_probability)`. The policy objective clips
that ratio to `1 ± clip_epsilon`, limiting the incentive for a large change from
the behavior policy. The value objective trains predicted returns and clips the
change from saved values. An entropy bonus encourages exploration, and gradient
clipping limits update size. `target_kl` can stop further epochs when average KL
exceeds the configured threshold. This implementation checks it after each epoch.

Observation-normalizer statistics stay fixed during the rollout's update and are
updated afterward for the next policy snapshot. Reward normalization estimates a
scale from discounted returns and resets its running return at episode boundaries.
The optimizer's learning rate decays linearly with processed transitions. All these
states are saved so continuation uses the same learning schedule.

Monitor `loss/policy`, `loss/value`, `state/entropy`, `state/approx_kl`,
`state/clip_fraction`, `state/early_stop` and `state/learning_rate` alongside finish
rate and progress. Low training loss alone does not establish successful driving.
