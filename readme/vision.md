# Camera vision with every RL algorithm

This guide applies to TrackmaniaRL 1.2.12 with RunSpec 2.0.

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
