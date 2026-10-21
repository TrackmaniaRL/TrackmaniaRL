[![Latest release](https://img.shields.io/github/v/release/TrackmaniaRL/TrackmaniaRL)](https://github.com/TrackmaniaRL/TrackmaniaRL/releases/latest)
[![PyPI version](https://img.shields.io/pypi/v/trackmaniarl)](https://pypi.org/project/TrackmaniaRL/)

![TrackmaniaRL logo](https://raw.githubusercontent.com/TrackmaniaRL/TrackmaniaRL/main/docs/assets/trackmaniarl-logo.png)

TrackmaniaRL is a Python library for building and training reinforcement-learning
agents in Trackmania 2020. Compose your own models, feature pipelines, replay
samplers and learners, then use the same configuration from Python or the CLI.

- **Composable models:** encoders, temporal cores, value heads and actor-critic models.
- **Training workflows:** online RL, demonstrations, behavior cloning and checkpoint resume.
- **Game integration:** telemetry, camera observations, map geometry and control.
- **Repeatable evaluation:** complete trial records, finish rates and timing metrics.

The Python package and game integration support Windows and desktop Linux. On Linux,
Trackmania runs through Proton and is not officially supported by Ubisoft Nadeo, so
the game setup is experimental. WSL is not a supported game host.
TrackmaniaRL 1.2.x requires Python 3.12 and uses RunSpec 2.0 and checkpoint schema 2.0.

## Installation

Add the library to your Python 3.12 project:

```bash
pip install trackmaniarl
```

Optional integrations are available as extras, for example
`pip install "trackmaniarl[vision,wandb,distributed]"`.
See [platform and PyTorch setup](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/performance.md)
for CPU and GPU installation details.

## Train across computers

For off-policy algorithms such as IQN and SAC, multiple Trackmania actors on
separate computers can send rollouts asynchronously to one learner that trains
the model and publishes policy updates. Follow the
[multi-computer setup guide](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/distributed-setup.md)
for installation, matching assets, authentication, encrypted tunneling and
commands; PPO training remains local.

## Use from Python

Given a `run.yaml` with an environment and learner configured:

```python
from pathlib import Path

from trackmaniarl import RunSpec, Trainer, resolve_run

config = Path("run.yaml").resolve()
run = resolve_run(RunSpec.from_yaml(config), base_dir=config.parent)
try:
    result = Trainer(run).train()
    print(f"Collected {result.transitions} transitions; ran {result.updates} updates")
finally:
    run.logger.close()
```

`Trainer` runs local training. The CLI also provides the distributed actor/learner
workflow. Start with the [Python quickstart](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/python-quickstart.md)
for a complete example that needs no game, GPU, map assets or tracking account.

| Example | What it demonstrates |
| --- | --- |
| [Game-free training](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/examples/game_free.py) | Custom environment and reward, replay, updates and checkpoints |
| [Train a configuration](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/examples/train.py) | Python API, resource cleanup and checkpoint resume |
| [Custom model](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/examples/custom_model.py) | Your own encoder in a bundled value model |
| [Own-map configuration](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/examples/own-map.yaml) | Connect the components to Trackmania |

## Train in Trackmania

Create a separate project for your maps, models, configurations and training data:

```bash
trackmaniarl init my-agent --template trackmania
cd my-agent
uv sync
```

Follow the [game quickstart](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/quickstart.md)
to install Openplanet, connect the game, build map geometry and run training and
evaluation. The generated project includes configurations for IQN, SAC, REDQ,
TQC, discrete SAC, PPO and camera-based variants.

## Neural inference in motion

![Full drive alongside road, car and context branches, residual blocks, action values and changing steering and pedal controls](https://raw.githubusercontent.com/TrackmaniaRL/TrackmaniaRL/main/docs/assets/trackmaniarl-neural-flow.gif)

This full drive shows a selected evaluation lap from the contributor's strongest
checkpoint for this map. The contributor reports approximately 12 wall-clock hours
of training before the recording; capturing and rendering the film did not update
its weights. The panel is an architecture showcase: it follows map-relative road
geometry, a 60-value car/physics vector and 29 engineered driving-context values
through the model to action scores and selected controls. The changing activation
colours show real tensor values, but they are not feature attribution and individual
cells are not intended to explain a decision by themselves. This selected lap uses
the map-specific `neighbors` action filter. It is neither an unassisted-policy
benchmark nor evidence of generalization. See [the model and recording explanation](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/docs/activation-film.md)
for provenance, the five-attempt film series and [GIF reproduction](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/docs/neural-flow-media.md).

## Documentation

Start at the [documentation index](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/README.md).

| Task | Guide |
| --- | --- |
| Learn the Python API | [Python quickstart](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/python-quickstart.md) · [SDK reference](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/sdk.md) |
| Extend the library | [Public API](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/public-api.md) · [Examples](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/examples/README.md) |
| Configure training | [Configuration](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/configuration.md) · [Algorithms](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/algorithms.md) |
| Use images or demonstrations | [Vision and PPO](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/vision.md) · [Imitation learning](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/imitation-learning.md) |
| Understand the runtime | [Architecture](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/docs/library-architecture.md) · [Distributed training](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/architecture.md) |
| Diagnose a run | [Observability](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/observability.md) · [Troubleshooting](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/docs/troubleshooting.md) |

## Development and research

```powershell
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict trackmaniarl
uv run pytest
uv build
uv run python scripts/check_distribution.py
```

See [contributing](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/CONTRIBUTING.md)
and the [development guide](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/readme/development.md)
for the full checks. The [repository layout](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/docs/repository-layout.md)
separates the published package, examples, tests and local experiment data.

TrackmaniaRL is released under the [MIT license](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/LICENSE). It originated from TMRL.
Attribution is in [NOTICE](https://github.com/TrackmaniaRL/TrackmaniaRL/blob/main/NOTICE). It is not affiliated with Ubisoft or Nadeo.
