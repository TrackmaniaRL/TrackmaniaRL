# Trackmania training and release workflow for 1.2.12

The default configuration below trains IQN from lidar and telemetry. The generated
`run-ppo.yaml` trains PPO from telemetry, and `run-ppo-vision.yaml` trains PPO from
camera frames. See [vision and PPO](vision.md) for those complete alternatives.

For demonstration recording, behavior cloning, DAgger recovery, exact BC
resume and the required closed-loop gate, see the
[imitation-learning workflow](imitation-learning.md).

Install the released CLI, then create the game project only on a machine that
has Trackmania, Openplanet and a virtual gamepad driver:

```powershell
uv tool install --index https://download.pytorch.org/whl/cpu --with "torch==2.11.0+cpu" "trackmaniarl==1.2.12"
trackmaniarl init my-agent --template trackmania
cd my-agent
uv sync
uv run trackmaniarl validate run.yaml
```

> **Windows driver boundary:** the generated TrackMania project installs
> `vgamepad`, whose normal installer may install or repair the system-wide
> ViGEmBus driver. Review the pinned `vgamepad` source before provisioning a
> host. `VGAMEPAD_SKIP_VIGEMBUS_INSTALL=true` skips that driver installer for
> CI or an already provisioned machine. It does not provide a driver, so the
> gamepad backend still requires a compatible ViGEmBus installation. Use the
> keyboard backend when no virtual gamepad driver should be installed.

### Linux game host (experimental)

Trackmania has no native Linux build and Ubisoft Nadeo does not officially support
it on Linux. TrackmaniaRL supports the complete local pipeline on a desktop Linux
session by running the game and Openplanet through Steam/Proton while the Python
process and virtual controller run natively. WSL is not a supported game host.

1. Install Trackmania from [Steam](https://store.steampowered.com/app/2225070/Trackmania/).
   If Steam does not offer the game, enable Steam Play for all titles in
   **Settings > Compatibility** and select Proton Experimental.
2. Install Protontricks using your distribution's package manager or Flatpak.
   Download the current Openplanet installer, launch it with
   `protontricks-launch Openplanet*.exe`, select Trackmania when prompted and
   install into Trackmania's directory.
3. In Trackmania's Steam properties, set this launch option so Proton loads
   Openplanet's `dinput8.dll`:

   ```text
   WINEDLLOVERRIDES="dinput8=n,b" %command%
   ```

4. Install the native libevdev runtime (`libevdev2` on Debian/Ubuntu,
   `libevdev` on Fedora/Arch), load `uinput` and grant the logged-in desktop user
   access with a udev rule:

   ```bash
   sudo modprobe uinput
   printf 'KERNEL=="uinput", TAG+="uaccess"\n' | \
     sudo tee /etc/udev/rules.d/50-uinput.rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger /dev/uinput
   ```

   Log out and back in if `/dev/uinput` is still inaccessible. Do not run the
   trainer as root.
5. Follow the project installation above, then verify the native controller
   before opening a training session:

   ```bash
   uv run python -c "import vgamepad as vg; g = vg.VX360Gamepad(); g.reset(); g.update(); print('virtual gamepad ready')"
   uv run trackmaniarl track check --config run.yaml
   ```

Editor-validation resets on Linux require a local X11 session and `xdotool`,
with Trackmania already focused. This reset path rejects Wayland, SSH and
headless sessions and does not activate background windows. Software tests do
not replace checking reset and controller readiness in the game before training.

Install **TrackmaniaRL Connect** from Openplanet's in-game Plugin Manager just as
on Windows. The native Python process and the Proton game share localhost, so no
port forwarding is required. If Openplanet stops loading after a Trackmania
update, rerun the latest Openplanet installer in the same Proton prefix and check
the DLL override. Linux game integration is supported by the project but remains
experimental because the game and compatibility layer can change independently.

In Openplanet's **Plugin Manager**, install the signed
[**TrackmaniaRL Connect**](https://openplanet.dev/plugin/sac_getdata) plugin
(identifier `SAC_GetData`) and verify version **2.4.0**. Enable
[School Mode](https://openplanet.dev/docs/school-mode) on Openplanet 1.26.0 or
newer, which blocks online play and official leaderboard submissions while the
plugin is active. Do not copy the bundled developer-reference `.as` file into
`Scripts` or enable an unmanaged loose TrackmaniaRL script alongside the managed
plugin.

<p align="center">
  <img src="../docs/diagrams/trackmania-integration-preview.svg" alt="Trackmania and signed Openplanet plugin integration boundary" width="900">
</p>

[Editable integration diagram](../docs/diagrams/trackmania-integration.excalidraw) ·
[local preview](../docs/diagrams/trackmania-integration-preview.html)

The generated `run.yaml` selects the control device explicitly:

```yaml
components:
  environment:
    kwargs:
      config:
        control_backend: gamepad
```

Use `gamepad` for analog steering and rumble-based collision detection. Select
`keyboard` when a virtual gamepad is unavailable. The keyboard backend converts
analog model output to digital gas/brake and left/right input, with a steering
dead zone and cannot provide rumble collision signals. The choice belongs to
the environment, not the model, so the same policy can drive either backend.
expect different driving dynamics after analog-to-digital conversion.

### Check the gamepad's steering curve

Trackmania bends the virtual stick through its own **Analog Sensitivity** and
**Analog Dead Zone** (Settings > Controls, per car) before the car steers. At a
low sensitivity the smaller steering levels all drive straight: on one measured
install at sensitivity 0.1 and dead zone 0.05, 7 of the 13 default levels steered
identically, so the policy could not tell them apart. With the car sitting on a
loaded map and no gas, measure it:

```powershell
uv run trackmaniarl track check-steering --output assets/steering-curve.json
```

The command reports how many steering levels steer differently. Either set the
sensitivity to 1.0 and the dead zone to its minimum and measure again, or keep the
measured curve and point the environment at it. The gamepad backend then sends
the stick position that produces each requested steer:

```yaml
        steering_curve_path: assets/steering-curve.json
```

Before starting separate `learner` or `actor` commands, generate one random
distributed token and store the same value as
`TRACKMANIARL_DISTRIBUTED_TOKEN` in each project's ignored `.env` file:

```powershell
uv run python -c "print(__import__('secrets').token_urlsafe(32))"
```

Local `train` and `smoke` still authenticate their actor/learner processes, but
the launcher generates an ephemeral token internally. They do not require this
environment variable.

Generated TrackMania agents select the project's tested PyTorch CUDA runtime by
default on Windows and Linux. A newer NVIDIA driver stays compatible. macOS
falls back to its normal PyPI/MPS Torch wheel. ROCm hosts require the matching
AMD Torch build.

The generated `.npz` is a structural placeholder only. Before training, record
the two map boundaries by hand and build a UID-bound asset. Do not reuse an
asset from another map:

```powershell
uv run trackmaniarl track record-boundary left assets/my-map-left.npy
uv run trackmaniarl track record-boundary right assets/my-map-right.npy
uv run trackmaniarl track build-geometry assets/my-map.geometry.npz --left assets/my-map-left.npy --right assets/my-map-right.npy --map-uid YOUR_MAP_UID --map-path maps/my-map.Map.Gbx
```

### Build a map from a ghost instead

A replay or ghost of one finished lap can replace both hand-driven boundaries.
Convert it to JSON with a [GBX.NET](https://github.com/BigBang1112/gbx-net)
based exporter, in this shape: a `map_uid`, `checkpoints` with `time_ms` (the last
one is the finish), `samples` every 50 ms with `time_ms`, `position`, `velocity`,
`rotation` (`[x, y, z, w]`), `speed` and optional `extras` holding `steer`, `gas`,
`brake` and `wheel_slip`, and, to use the lap as a demonstration, `inputs`: every
input change with `time_ms`, `steer`, `throttle` and `brake`. Map UIDs used for
asset filenames may contain ASCII letters, digits, underscores and hyphens;
reserved Windows filenames and paths escaping the output directory are rejected
before any files are written. Then run:

```powershell
uv run trackmaniarl track from-ghost record.json --map-path maps/my-map.Map.Gbx
```

The lap's line becomes the reward centre line inside a corridor of virtual walls
(`--half-width`, default 8 m either side; the inner wall narrows on turns tighter
than that), and its timing becomes a pace reference. The command prints the
`run.yaml` values to set, including `time_attack_target_s` at the lap's time.
With a record ghost the lidar then measures distance from the record line and
the pace features measure time lost against the record at every point. The
corridor is not the real track: the agent cannot find a route outside it, so
widen it when the record line is not the one you want to beat.

Replace `YOUR_MAP_UID` with the UID reported by `track check` and
set that same UID in `environment.kwargs.config.expected_map_uid`,
`feature_pipeline.kwargs.config.expected_map_uid` and
`evaluation.maps[].expected_map_uid`. Load that local `.Map.Gbx` manually in
Trackmania, enter it with a visible vehicle and run:

```powershell
uv run trackmaniarl track check --config run.yaml
uv run trackmaniarl smoke run.yaml --transitions 100
uv run trackmaniarl train run.yaml
```

Openplanet does not expose a documented safe API for loading an arbitrary local
map. The plugin's second localhost port (default `9001`) therefore verifies the
already loaded UID and protocol before every training, smoke and evaluation
episode, then confirms a ready local player after controller reset. A timeout,
disconnect, UID mismatch or readiness rejection terminates the actor with a
failing process status. The protocol does not expose the plugin package's
signature or version. Verify those properties in Plugin Manager.

Comparison launchers use a separate bounded preflight that verifies the map and
player before issuing one reset, then polls for an active unfinished race near
the start. It honors stop files during socket reads, records the last clock,
finish flag and failure phase, and closes its temporary controller and telemetry
connection. It does not press a finish-confirmation button blindly. Keyboard
restart/confirmation also verifies that Trackmania is the foreground window;
failed activation raises before any key is sent.

The reference baseline is a 78-action dueling IQN (`13` steering levels × `2`
gas levels × no brake, full brake, or timed brake tap), not TQC. With default
feature settings, each observation contains 20 normalized telemetry values, a
`[4, 60]` car-local boundary tensor and a 60-element validity mask, all derived
from the documented 33-field `SAC_GetData` packet and the geometry
asset. The four lidar channels are the lateral/forward coordinates of the left
and right boundaries. The local frame comes from `api.Position` and `vis.Dir`.
it does not require aim-yaw telemetry. TQC remains an optional example only.

Model factories publish their train-time contract and learners publish the
contracts they accept. Composed Q/QR-DQN/IQN/FQF models expose `discrete_value`, the telemetry
TQC baseline exposes `continuous_quantile_actor_critic` and behavior cloning
exposes `categorical_policy`. `trackmaniarl validate` rejects a mismatched pair
before model setup instead of failing later on a missing head. Models remain
interchangeable between algorithms that consume the same contract. Algorithms
with different objectives require a matching model head.

## Value algorithm selection

All discrete value experiments use `DiscreteValueLearner`. The YAML composition
selects the algorithm:

| Experiment | Head | Strategy |
| --- | --- | --- |
| scalar baseline | `ScalarQHead` | `ScalarValueStrategy` |
| QR-DQN | `FixedQuantileHead` | `FixedQuantileStrategy` |
| IQN | `ImplicitQuantileHead` | `RandomQuantileStrategy` |
| FQF | `ImplicitQuantileHead` | `LearnedFractionStrategy` |

For recurrent experiments select `GruTemporalCore` or `MambaTemporalCore`, set
`training.sequence_length` and configure learner `burn_in`. The lidar encoder
is frame-only: the model vectorizes `[B,T]` into `[B*T]` before encoding and
restores `[B,T,D]` before the temporal core.

FQF creates its own fraction optimizer from
`LearnedFractionStrategy.auxiliary_parameters()`. Monitor fraction entropy and
boundary spacing as well as TD/quantile metrics. The target network contains
its own fraction proposal network and target quantiles are evaluated only for
the action selected by online Double-DQN.

To initialize FQF from a proven IQN run, use the warm-start loader for named
`encoder`, `temporal` and compatible `head` tensors from a current compressed
checkpoint. Pre-2.0 checkpoints are rejected and tensors with changed names,
shapes or dtypes are reported without being copied. Preserve the generated
match report with the experiment artifacts.

## Mamba temporal core

`MambaTemporalCore` is an opt-in temporal component. Use it as a named experiment
after recording an identical GRU baseline with the same seed, replay, update
budget and evaluation suite.

The `torch` backend is portable across Windows, Linux, CPU and CUDA. The
`native` backend requires a working `mamba-ssm` selective-scan kernel. `auto`
probes native forward/backward and records the Pure PyTorch fallback reason.
Both backends use the same model parameters and checkpoint fingerprint.

Install the `mamba` extra only when testing the native kernel. The `torch`
backend is implemented locally with standard Torch operations and works without
`mamba-ssm`:

```bash
uv sync --extra mamba
```

Select the model explicitly in `run.yaml` and set
`training.sequence_length > 1`. In the composed replay path, keep
`LidarFeaturePipeline` config `history_length: 1`: the sampler creates `[B,T,...]`
sequences, `FrameBatchAdapter` flattens their frames for the sensor encoder and
the temporal core receives the restored sequence. Do not also stack history in
the feature pipeline.

```yaml
components:
  model_factory:
    class_path: trackmaniarl.models.factory:CompositeValueModelFactory
    kwargs:
      encoder:
        class_path: trackmaniarl.trackmania.encoders:LidarSensorEncoder
        kwargs:
          config: {telemetry_dim: 20, spatial_bins: 12, output_dim: 256}
      temporal:
        class_path: trackmaniarl.models.temporal:MambaTemporalCore
        kwargs: {input_dim: 256, backend: auto, d_state: 16, d_conv: 4, expand: 2}
      head:
        class_path: trackmaniarl.models.heads:ImplicitQuantileHead
        kwargs:
          config: {feature_dim: 256, action_count: 78, cosine_count: 64, dueling: true}
      strategy:
        class_path: trackmaniarl.models.strategies:LearnedFractionStrategy
        kwargs: {feature_dim: 256, fraction_count: 32}
  learner:
    class_path: trackmaniarl.algorithms.value_based:DiscreteValueLearner
    kwargs: {burn_in: 4}
training:
  sequence_length: 16
```

The sensor encoder processes the 16 replay frames as one vectorized `B*T`
batch. The Mamba core consumes `[B,T,D]`. `burn_in: 4` builds its initial
recurrent state without gradients and excludes those positions from losses and
priorities.

The optional dependency is imported only when the native backend is probed, so
normal imports, GRU runs and Pure PyTorch Mamba remain independent of
`mamba-ssm`. On Windows, CPU or an unsupported CUDA build, use `backend: torch`
or let `auto` record its fallback. Treat every new deployment platform as
unsupported until it passes the bounded live smoke test on that exact host.
offline contract tests alone are not evidence of game compatibility.

Every run writes `manifest.json`, versioned `events.jsonl`, compressed episode
artifacts, checkpoints and study records. Resume a stopped run with:

```bash
uv run trackmaniarl resume run.yaml artifacts/trackmania-iqn/checkpoints/distributed-update-XXXXXXXX.pt
```

Replace `trackmania-iqn` if your `run_id` differs.

W&B is optional. The generated project logs locally until you run
`uv add "trackmaniarl[trackmania,distributed,wandb]"` and add an
explicit `WandbTracker` under `components.additional_loggers`. Supply
`WANDB_API_KEY` only through a private environment or ignored `.env` file.
The generated project retains its vetted `vgamepad` source during this update.
an existing project must retain the same direct source pin documented in the
[installation guide](quickstart.md#installation).

`trackmaniarl smoke` is the required Windows preflight. It collects a bounded number of
real actions, completes at least one update, verifies a live policy refresh,
and restores the produced checkpoint. It will operate the virtual gamepad:

```bash
uv run trackmaniarl smoke run.yaml --transitions 100
```

The release benchmark is deterministic only in the sense that it repeats the
same local map and assets. It does not claim game-engine seed control. It uses
the `trials_per_map`, `min_finish_rate`, `target_median_s` and optional
`target_mean_s` and `max_step_race_time_ms` thresholds in `run.yaml`, writes
`evaluation.json` with
per-trial status, latency/FPS and map UID and fails when any configured
acceptance threshold is missed:

```bash
uv run trackmaniarl benchmark run.yaml artifacts/trackmania-iqn-lidar/checkpoints/distributed-update-XXXXXXXX.pt
```

With the first-party evaluator, add `--stop-file PATH` to cancel a benchmark
without a console interrupt. The path must not already exist. Creating it stops
the evaluation between control steps or during a window-availability pause, then
closes the environment. The resulting `evaluation.json` has `status: cancelled`,
retains completed trials and any active trial as an `operator_interruption` DNF,
and leaves unstarted trials absent. It cannot pass as a completed benchmark.
Native environment calls must return before cancellation can be observed.
Successful artifacts use `status: complete`; both include the expected trial
count and the evaluated checkpoint's SHA-256 digest. Checkpoint replacement
during evaluation is rejected.

For a diagnostic run that rejects every lap containing one or more skipped
telemetry frames, add `--reject-telemetry-skips`. This is intentionally stricter
than the normal runtime-health gate and is useful for measuring the effect of
telemetry delivery independently of driving quality.

For a competition-oriented ten-trial gate, configure both time targets and
require every trial to finish:

```yaml
evaluation:
  trials_per_map: 10
  target_median_s: 37.0
  target_mean_s: 37.0
  max_step_race_time_ms: 100.0
  min_finish_rate: 1.0
```

The reliable checkpoint leader then ranks qualifying candidates by finish
rate, mean and median. The separate fastest leader remains based on the best
single lap, so an exploratory sub-36 result is retained without allowing one
fast lap to hide a slow tail. When the race-clock runtime guard is configured,
every evaluated control step must report a finite, positive measurement. A
valid maximum cannot hide a missing or invalid step.

## Human recovery fine-tuning

This workflow requires a graph V6 source model, not the default lidar IQN or
a PPO checkpoint. To configure a graph source, copy the generated `run.yaml`
and replace the following component blocks. Keep the complete environment,
replay, logging and evaluation configuration for your own map.

```yaml
feature_pipeline:
  class_path: trackmaniarl.experiments.graph_iqn_v5:BoundaryGraphFeaturePipelineV5
  kwargs:
    geometry_path: assets/my-map.geometry.npz
    expected_map_uid: REPLACE_WITH_YOUR_MAP_UID
model_factory:
  class_path: trackmaniarl.models.factory:CompositeValueModelFactory
  kwargs:
    encoder:
      class_path: trackmaniarl.experiments.graph_iqn_v6:IncidentGatedTrackGnnSimbaEncoderV6
    temporal:
      class_path: trackmaniarl.models.temporal:IdentityTemporalCore
      kwargs: {input_dim: 192}
    head:
      class_path: trackmaniarl.experiments.graph_iqn:DuelingImplicitQuantileHead
    strategy:
      class_path: trackmaniarl.models.strategies:RandomQuantileStrategy
```

These are keys under `components`. First train and benchmark the source with
`DiscreteValueLearner`. For recovery collection and fine-tuning, keep the same
architecture and change `components.learner.class_path` to
`trackmaniarl.experiments.graph_iqn_v6:IncidentRecoveryOnlyDiscreteValueLearner`.
Use the matching source checkpoint for both commands. The `experiments` module
name is retained for import and checkpoint compatibility, not an unsupported
feature designation. The pipeline reconstructs causal graph/recovery features
from telemetry. Only the incident-gated adapter is trained by `recovery-finetune`.

`track record-recovery` creates targeted recovery supervision without changing
the proven nominal policy. The checkpoint drives deterministically to a seeded
progress point, the collector applies one short full-steer perturbation, then
waits until the virtual input is observably neutral before sounding an alert.
The human takes over, recovers and finishes the lap. The injected action and
human reaction delay are retained only as causal context. The first clear
physical input starts the expert labels. Unfinished laps, restarts and
kinematically implausible respawn/teleport jumps are discarded. The requested
impulse is not trusted blindly: its start, duration and control are reconstructed
from the controls echoed by telemetry and a missing, wrong-direction or too-short
impulse rejects the attempt.

Collect multiple completed left- and right-side recoveries from the exact
checkpoint that will be adapted:

```powershell
$Config = (Resolve-Path 'run.yaml').Path
$Checkpoint = (Resolve-Path 'artifacts/source/checkpoints/fastest-eval-policy.pt').Path
$RecoveryDir = Join-Path (Get-Location) 'demos-recovery'

uv run trackmaniarl track check --config "$Config"
uv run trackmaniarl track record-recovery "$RecoveryDir" `
  --config "$Config" `
  --checkpoint "$Checkpoint" `
  --count 36
```

Keep hands off the controls until the audible `TAKE OVER NOW` notification.
The default 36-lap session balances three progress windows from 55% to 83%,
three perturbation windows from 50 to 200 ms and both planned directions, with
two attempts per cell. On a bend, the applied steer is chosen opposite to the
policy's current steer to create a meaningful deviation. The actual control is
stored. A rejected attempt is resampled inside the same coverage cell.
Only completed laps are saved. Each archive embeds the source checkpoint's
SHA-256. Fine-tuning rejects mixed or stale policy data. It reconstructs the
full causal feature history, selects only post-takeover samples for which the
incident gate is active, removes episodes with too little gated supervision or
implausibly slow normalized recovery and derives progress, severity and direction
strata from what actually happened rather than from the requested plan. The data
gate requires all nine progress-by-severity cells and the deterministic held-out
split preserves balanced progress, severity and direction coverage. Episodes are
sampled uniformly so a long lap cannot dominate. It optimizes only the bounded
recovery adapter:

```powershell
$Candidate = Join-Path (Get-Location) 'artifacts/recovery-policy.pt'
uv run trackmaniarl recovery-finetune "$Config" "$Checkpoint" `
  --recovery "$RecoveryDir" `
  --updates 300 `
  --batch-size 128 `
  --validation-fraction 0.25 `
  --minimum-gate 0.01 `
  --minimum-usable-episodes 24 `
  --minimum-validation-episodes 6 `
  --minimum-gated-samples 500 `
  --minimum-samples-per-episode 15 `
  --maximum-normalized-recovery-time 80 `
  --minimum-source-disagreement 0.05 `
  --maximum-source-disagreement 0.20 `
  --output "$Candidate"

uv run trackmaniarl benchmark "$Config" "$Candidate" `
  --trials 10 `
  --target-median 37 `
  --target-mean 37 `
  --max-step-race-time-ms 100 `
  --min-finish-rate 1
```

The candidate is deliberately a `policy_only` checkpoint: it is valid for
benchmarking and later warm-start initialization, but not for `trackmaniarl resume`.
This prevents weights from the selected validation update being paired with stale
optimizer momentum from a later update. The output path must be new. The command
will not overwrite an existing candidate.

Do not follow this phase automatically with unrestricted online TD updates.
First compare the candidate with its unchanged source checkpoint. Promote only
after 10/10 finishes and a mean below 37 seconds. Confirm a passing screen with
a fresh 30-trial benchmark and screen recording.

For a TrackMania-facing release, run the Windows smoke preflight and the
configured benchmark on the real game. Treat telemetry or controller errors as
a release blocker. A multi-hour Windows soak with a resumed checkpoint is
recommended when changing runtime, transport or checkpointing behavior. Use
`scripts/verify_soak.py` to produce `soak-report.json` for that extended
evidence.

## Connection troubleshooting

- **Port 9000 refuses the connection:** keep only signed TrackmaniaRL Connect
  2.4.0 enabled in Plugin Manager, enable School Mode and enter the local map.
- **Port 9000 connects but sends no complete frame:** make sure a local vehicle
  is visible. The supported plugin waits for real player and vehicle-visual
  state instead of zero-filling missing fields.
- **Port 9001, protocol or readiness fails:** reload the managed plugin and
  return to the local map. Protocol 2 and a ready local player are required.
- **The active UID differs:** use the UID printed by `track check`, replace all
  three UID settings and rebuild geometry from that exact `.Map.Gbx`.
- **Geometry checksum is missing or different:** re-run `track build-geometry`
  with `--map-path`. The generated `.npz` is intentionally only a placeholder.
- **Reset times out:** confirm that the configured `gamepad` or `keyboard`
  backend actually restarts the race timer before increasing any timeout.

### Window-aware vision capture (Windows)

For `VisionEnvironmentFactory`, replace the fixed `capture` argument with
`window_capture: {title: Trackmania, client_width: 1280, client_height: 720}`.
The two capture modes are mutually exclusive. Window capture measures the client
area in physical pixels, restores the configured client size at each episode,
and follows its screen position. The feature pipeline still determines the
model's image dimensions. Telemetry-only environments do not resize windows.

Keep the game visible and in the foreground. Losing focus or minimizing it
releases controls and pauses collection; returning to the game starts a fresh
episode and observation history. An interrupted evaluation trial remains a DNF.
Frames captured while the window moves are discarded. The runtime does not steal
focus. This mode requires Windows; fixed-region capture remains available.

Training and resume also accept `--stop-file PATH`. Use a path that does not
already exist, then create the file to request orderly shutdown. Blocking native
calls must return before the request can be observed.
