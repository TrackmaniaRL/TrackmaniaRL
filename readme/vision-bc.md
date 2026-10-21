# Behavior cloning from camera images

TrackmaniaRL 1.2.14 supports camera BC through the same `bc-train` and
`bc-benchmark` lifecycle as lidar BC. The project template generates
`run-bc-vision.yaml`, including a CNN model, image preprocessing, desktop capture
and the standard 78-action table. Set your map UID, geometry and capture rectangle
as described in [camera setup](vision.md).

```powershell
uv run trackmaniarl validate run-bc-vision.yaml
uv run trackmaniarl bc-train run-bc-vision.yaml --demo demonstrations/vision
uv run trackmaniarl bc-train run-bc-vision.yaml --demo demonstrations/vision --resume artifacts/RUN/checkpoints/bc-latest.pt
uv run trackmaniarl bc-benchmark run-bc-vision.yaml artifacts/RUN/checkpoints/bc-best-validation.pt --trials 20 --report-only
```

`validate` uses synthetic images without capturing the screen. `bc-train` loads
RGB demonstrations from disk and does not open the game. `bc-benchmark` captures
the visible viewport and sends the policy's compact action indices to the game.
Configure the evaluation suite's finish-rate and median-time targets to use its
release gate. `--report-only` prints the result without failing that gate.

## Demonstration format and alignment

Camera BC requires at least three distinct complete episode archives. Each
`.npz` contains `frames`, `actions`, `timestamps_ms` and JSON `metadata`. Use the
public functions in `trackmaniarl.trackmania.imitation_learning.vision_data` to
write and read them. Loading never enables pickle.

- `frames` is a uint8 RGB array of shape `(steps, height, width, 3)`.
- `actions` is a one-dimensional integer array of the same length. Values are
  canonical Trackmania action IDs from 0 to 77, not compact model indices.
- `timestamps_ms` contains increasing, nonnegative episode-relative capture
  times. Each frame must precede the episode finish.
- `contract` records the map UID, geometry SHA-256, action repeat, decision
  interval and `frame_start` control alignment. It must match the configured game.
- `finish_time_s` is the positive finite finish time for this complete episode.

Capture each image **before** applying its labelled action. Do not pair a
post-action screenshot with the action that caused it. Recording/import code
must establish that alignment and preserve decision cadence. Archive validation
checks shapes, ordering and declared timing contracts, but cannot prove that
independently recorded videos and controls are synchronized.
Keep `demonstration_action_lead_ms: 0` and `demonstration_control_aggregation: false`.
These archives already contain aligned discrete labels and are not resampled.

The existing telemetry recorder and its `.npz` files do not contain screenshots.
They cannot be converted into camera demonstrations without separately recorded,
aligned RGB data. A custom collector or importer can write the following archive
using its own `rgb_frames`, `canonical_actions`, `capture_times_ms` and
`recorded_finish_time_s` arrays and scalar:

```python
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.imitation_learning import RecoveryContract
from trackmaniarl.trackmania.imitation_learning.vision_data import (
    VisionDemonstration,
    save_vision_demonstration,
)

geometry = BoundaryGeometry("assets/my-map.geometry.npz")
contract = RecoveryContract(
    map_uid=geometry.map_uid,
    geometry_sha256=geometry.sha256,
    action_repeat_frames=1,
    decision_interval_ms=50.0,
    control_alignment="frame_start",
)
episode = VisionDemonstration(
    frames=rgb_frames,
    actions=canonical_actions,
    timestamps_ms=capture_times_ms,
    contract=contract,
    finish_time_s=recorded_finish_time_s,
)
save_vision_demonstration("demonstrations/vision/lap-001.npz", episode)
```

Create the destination directory first. Saving refuses to overwrite a file.
Keep the camera position and crop identical in collection and evaluation. Crop
out control-input overlays that would reveal the label being predicted.

## Model and dataset behavior

The public components are
`imitation_learning.vision:VisionBehaviorCloningPipeline` and
`imitation_learning.vision:VisionBehaviorCloningModelFactory`, both under
`trackmaniarl.trackmania`. The pipeline returns an `images` tensor in a mapping.
It shares resizing, grayscale conversion and episode-local channel stacking with
the RL vision pipeline. Default observations contain four grayscale 84x84 frames.

The model's `kwargs.action_ids` must exactly match the environment's
`kwargs.config.compact_action_ids`, including order. The loader maps canonical
labels to compact indices. In `kwargs.config`, set `channels` to the pipeline's
channel count and `hidden_dim` to the CNN feature dimension. Optional
`previous_action_conditioning` adds an embedding of the prior compact action.
It uses a dedicated start token at each episode boundary and never sees the
current label. `switch_logit_margin` and top-level `minimum_action_hold_steps`
control switching during inference.

This CNN uses stacked frames rather than a recurrent hidden state. BC shares
weighted classification, class balancing, validation, optimizer, scheduler,
gradient-scaler and RNG checkpoint handling with lidar models. The common
`BehaviorCloningModel` base in `imitation_learning.model_contract` defines the
interface for other BC encoders without forcing lidar-shaped observations.

Splitting happens by complete episode before augmentation. Identical visual/action
episodes are rejected even if copied to different files or retimestamped. Frame
history resets between episodes. The immutable dataset manifest fingerprints the
archives, split, preprocessing and model configuration for resume.

`--horizontal-flip-augmentation` adds reflected training images and mirrors
steering labels and previous-action metadata. The action subset must contain
left/right pairs. Use this only when reflection is appropriate for your camera
and visual scene. Validation episodes remain unchanged.

BC currently materializes the preprocessed dataset in RAM. One `(4,84,84)`
float32 stack takes about 110 KiB. A 40-second episode at 20 decisions per second
uses about 86 MiB for its stacks alone, with additional temporary batching memory.
Reduce resolution, frame stack or dataset size to fit available RAM. Live capture
requires the vision extra, while offline image training does not require MSS.

The existing telemetry-based DAgger and recovery archives remain separate data
workflows. Their geometry and state labels do not supply RGB images to camera BC.

## Transfer to an RL image model

The selected `bc-best-validation.pt` can initialize compatible image encoders in
composite Q, QR, IQN or FQF models through `--model-initialization-checkpoint`.
Match CNN channel and feature dimensions in both configurations. BC transfer
selects encoder and temporal tensors only. The categorical BC head is never
copied into an RL value head. Inspect the generated `warm-start.json` for matched
and mismatched tensors. This is initialization of a new RL run, not BC resume.

## Paired lidar + vision

Use the generated `run-bc-lidar-vision.yaml` configuration for supervised fusion.
The environment preserves telemetry with `include_telemetry: true`. The
`LidarVisionBehaviorCloningPipeline` transforms both sensors and exposes the flat
BC fields `lidar`, `lidar_mask`, `telemetry` and `images`. The model factory is
`VisionBehaviorCloningModelFactory` with `kwargs.config.lidar` set to a
`LidarSensorConfig` mapping. Match its dimensions to the lidar pipeline, and match
`channels` to the image stack. Both sensor branches and the fusion projection train.

Extend each `VisionDemonstration` with `telemetry=telemetry_rows`, a finite numeric
array of shape `(steps, 33)` using the standard Openplanet field order. Every row
must accompany the image preceding the same action. Column 3 must exactly match
`timestamps_ms`. The writer stores these episodes as `trackmaniarl-paired-demo-v1`
without pickle. It retains the same map, geometry, timing, action-range and duplicate
episode checks as image-only archives. Neither missing telemetry nor missing RGB
is reconstructed or silently substituted. Sensor modes must match the model,
pipeline, environment and every archive.

```powershell
uv run trackmaniarl validate run-bc-lidar-vision.yaml
uv run trackmaniarl bc-train run-bc-lidar-vision.yaml --demo demonstrations/paired
uv run trackmaniarl bc-train run-bc-lidar-vision.yaml --demo demonstrations/paired --resume artifacts/RUN/checkpoints/bc-latest.pt
uv run trackmaniarl bc-benchmark run-bc-lidar-vision.yaml artifacts/RUN/checkpoints/bc-best-validation.pt --report-only
```

The template masks current telemetry controls to prevent action-label leakage.
The pipeline refuses unmasked controls. Previous-action conditioning remains
optional and shifts labels by one step, restarting at each episode boundary.
Both sensor histories reset between episodes and evaluation maps. Lidar history
length is one, while image frame stacking remains configurable.

Default paired templates do not enable horizontal reflection. For reflection,
use the existing supported local-velocity 8-channel lidar and 46/49-feature
telemetry layout. The augmentation then mirrors images, lidar, telemetry and
action labels together, after splitting training from validation. Unsupported
layouts fail explicitly. Telemetry-only recovery and DAgger files still cannot
supply missing RGB frames.
