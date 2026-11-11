# Train an agent on your own map

[Documentation index](README.md) | [Python quickstart](python-quickstart.md)

## Installation

Install [uv](https://docs.astral.sh/uv/), then create a project. On Windows,
the generated project's `vgamepad` dependency may install or repair the
system-wide ViGEmBus driver during `uv sync`. Read the
[driver warning](trackmania.md) before proceeding. Generated projects select
CUDA PyTorch wheels on Windows and Linux. For a CPU-only project generated from
the released CLI, after `init` and before `uv sync`, change
`index = "pytorch-cuda"` to `index = "pytorch-cpu"` in the generated
`pyproject.toml` under `[tool.uv.sources]`. A project generated from a local
checkout can also inherit the checkout's Torch source and constraints, so that
change alone may not resolve the build. See the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) for other
builds.

```powershell
uv tool install trackmaniarl
trackmaniarl init my-agent --template trackmania
cd my-agent
uv sync
```

To install a locally built wheel instead:

```powershell
uv tool install path\to\trackmaniarl-VERSION-py3-none-any.whl
```

To work from this repository, run `uv sync --group dev` and prefix commands with
`uv run`.

For a game-free installation check, run the small SDK starter in a separate
terminal from the directory that contains `my-agent`:

```powershell
trackmaniarl init sdk-check --template starter
cd sdk-check
uv sync
uv run trackmaniarl inspect-config run.yaml
uv run trackmaniarl validate run.yaml
```

Validation executes trusted component code and a synthetic update. It does not
drive the game or establish driving performance. CPU execution is sufficient for
this check. Live neural inference and training benefit from an NVIDIA GPU with
a compatible PyTorch build. See [platform and performance guidance](performance.md).

## Game setup

You need:

- Trackmania 2020 on Windows, or on desktop Linux through Steam and Proton
- Openplanet in School Mode
- the **TrackmaniaRL Connect / SAC_GetData** plugin
- your map open in the editor's validation mode, with the car at the start.

The plugin uses TCP ports 9000 (telemetry) and 9001 (session and map status) on
localhost. The setup guide explains plugin installation and controller setup:
[Trackmania setup](trackmania.md).

On Linux, install Trackmania with Steam/Proton, run the Openplanet installer in
Trackmania's Proton prefix and grant your user access to `/dev/uinput`. The setup
guide includes the complete Linux procedure and a virtual-gamepad smoke test.

Check the connection before recording data or training:

```powershell
uv run trackmaniarl track check
```

## Configure a map

Copy your `.Map.Gbx` file to `maps/my-map.Map.Gbx` before building geometry.
Use the UID printed by `track check` in place of `YOUR_MAP_UID` below. In the
game, start at the beginning of the map and drive along the left boundary to
the finish while the first command records. Restart the map, then use the
second command and drive along the right boundary to the finish:

```powershell
uv run trackmaniarl track record-boundary left assets/my-map-left.npy
uv run trackmaniarl track record-boundary right assets/my-map-right.npy
```

Build the geometry after both recordings finish:

```bash
uv run trackmaniarl track build-geometry assets/my-map.geometry.npz --left assets/my-map-left.npy --right assets/my-map-right.npy --map-uid YOUR_MAP_UID --map-path maps/my-map.Map.Gbx
```

Replace every `REPLACE_WITH_YOUR_MAP_UID` in `run.yaml`: the environment,
feature pipeline and evaluation map each need the same UID. If you use different
filenames, update their paths in the configuration too. See
[examples/own-map.yaml](../examples/own-map.yaml) for a complete configuration.
TrackmaniaRL verifies the open map but does not select it for you.

## Train and resume

Validate the configuration and run a short smoke test before a full training run.
The generated default is two million transitions; adjust `training.total_transitions`
in `run.yaml` if you want a different budget. The smoke test controls the game.

```powershell
uv run trackmaniarl validate run.yaml
uv run trackmaniarl track check --config run.yaml
uv run trackmaniarl smoke run.yaml --transitions 100
uv run trackmaniarl train run.yaml
```

The starter project uses a lidar encoder, a dueling IQN policy and 78 discrete
control actions. An actor drives while the learner trains from replay.

For continuous off-policy control, use the generated `run-sac.yaml`,
`run-redq.yaml` or `run-tqc.yaml`. `run-sd-sac.yaml` provides a categorical
actor-critic with the 78-action table. All four include first-party telemetry models.

Every RL family supports camera observations. The template also generates
`run-q-vision.yaml`, `run-qr-vision.yaml`, `run-iqn-vision.yaml`,
`run-fqf-vision.yaml`, `run-sac-vision.yaml`, `run-redq-vision.yaml`,
`run-tqc-vision.yaml` and `run-sd-sac-vision.yaml`.
For supervised learning from aligned RGB demonstrations, use `run-bc-vision.yaml`
and the [camera BC guide](vision-bc.md).

For local on-policy PPO, use the generated `run-ppo.yaml` (telemetry) or
`run-ppo-vision.yaml` (camera images). Configure the map in the chosen file.
The [vision and PPO guide](vision.md) explains camera setup, CNN input,
fresh rollouts and the PPO update cycle.

Demonstrations are optional:

```powershell
uv run trackmaniarl track record-demo demonstrations --config run.yaml --count 5
uv run trackmaniarl train run.yaml --demo demonstrations
```

Resume the full training state from a checkpoint:

```powershell
uv run trackmaniarl resume run.yaml artifacts/RUN/checkpoints/CHECKPOINT.pt
```

Resume checks the run fingerprint and refuses mismatches. New checkpoints include
a diagnostic manifest with hashed configuration sections and relative library
source paths; an error identifies changed sections or files without printing
configuration values. Older checkpoints lack this detail and remain subject to
the same strict compatibility check. Restore the original code and configuration
to resume; the diagnostic manifest does not bypass fingerprint validation.
For journal migration, safe shutdown and actor spool recovery, see
[migration and recovery](recovery.md).

To start a new run with compatible weights, use
`train --model-initialization-checkpoint CHECKPOINT.pt`. A policy-only checkpoint
does not contain optimizer or replay state and cannot be used for an exact resume.

## Evaluate a checkpoint

Run every trial and keep the complete result:

```bash
uv run trackmaniarl benchmark run.yaml artifacts/RUN/checkpoints/CHECKPOINT.pt --trials 30 --min-finish-rate 1
```

On Windows, add `--record recordings/benchmark.mkv` to record the Trackmania
window. The built-in recorder requires FFmpeg and is not supported on Linux.
Use `--ffmpeg PATH` or `--window-title TITLE` when the defaults do not match your
system.

Each benchmark writes `evaluation.json` and an append-only trial timeline. A DNF
still counts as an attempt. Mean and median times cover finished attempts, so report
them together with the finish rate. Telemetry drops are recorded because they can
delay observations and control decisions. `--reject-telemetry-skips` rejects the
whole evaluation. It never removes individual slow attempts.

More detail: [evaluation architecture](../docs/library-architecture.md#checkpoints-and-evaluation),
[recording](../docs/recording.md) and [troubleshooting](../docs/troubleshooting.md).

