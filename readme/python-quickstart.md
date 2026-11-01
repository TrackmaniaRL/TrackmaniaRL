# Python quickstart

[Documentation index](README.md) · [Public API](public-api.md) · [SDK reference](sdk.md)

TrackmaniaRL can be imported by your application. A `RunSpec` describes the
components; `resolve_run` constructs them and `Trainer` runs local training.
The command line uses the same configuration format.

## Run without the game

With Python 3.12, install the library in your environment:

```bash
pip install trackmaniarl
```

Download [game_free.py](../examples/game_free.py) into your project and run:

```bash
python game_free.py --output artifacts/example
```

This collects 32 transitions across four eight-step episodes, samples replay,
updates a synthetic learner and writes checkpoints and event logs below
`artifacts/example/python-example/`. It needs no game, GPU, map or online service.
`SmokeLearner` counts updates; it does not optimize a neural network or learn to drive.
Use a fresh output directory for an independent repeat.

The example implements `ToyEnvironmentFactory.create(seed=...)`, an environment
with `reset` and `step`, and a reward for matching a target action. Replace the
reward expression in `step` to experiment with your own objective. The same
factory contract can construct a real environment.

## Train from a configuration

Install [uv](https://docs.astral.sh/uv/) if you do not have it, then create a
starter project in a directory outside the library checkout. The generated project
selects CUDA PyTorch wheels on Windows and Linux. For a CPU-only project generated
from the released CLI, after `init` and before `uv sync`, change
`index = "pytorch-cuda"` to `index = "pytorch-cpu"` in the generated
`pyproject.toml` under `[tool.uv.sources]`. A project generated from a local
checkout can also inherit the checkout's Torch source and constraints, so that
change alone may not resolve the build. Consult the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) for other
builds.

```bash
trackmaniarl init my-agent --template starter
cd my-agent
uv sync
```

Save the following as `train.py` in that project, then run `uv run python train.py`:

```python
from pathlib import Path

from trackmaniarl import RunSpec, Trainer, resolve_run

config = Path("run.yaml").resolve()
spec = RunSpec.from_yaml(config)
run = resolve_run(spec, base_dir=config.parent)
try:
    result = Trainer(run).train()
    print(result)
finally:
    run.logger.close()
```

Resolving paths against the configuration directory makes the run independent of
the process's working directory. Your custom component package must be installed
in the active environment. Close the logger even when training raises an error.

The [complete command-line wrapper](../examples/train.py) also accepts
`--resume path/to/checkpoint.pt`. Resume requires a full training checkpoint and
compatible configuration; a policy-only export is not a training checkpoint.
`Trainer` is the local runtime; use the [distributed workflow](architecture.md)
for separate actor and learner processes.

## Plug in a model

Download and run [custom_model.py](../examples/custom_model.py):

```bash
python custom_model.py
```

It combines a custom PyTorch encoder with the bundled identity temporal core,
scalar Q head and scalar value strategy. Two observations produce a `(2, 3)`
tensor of action values. The weights are untrained.

The encoder consumes individual frames and declares its `output_dim`; sequence
history belongs to the temporal core. In an installed project, use an importable
path such as `my_agent.models:MyEncoder` instead of the standalone example's
`__main__:MyEncoder`. See [model composition](sdk.md#value-model-composition)
for quantile heads, recurrence and full training configuration.

For live driving, continue with the [game quickstart](quickstart.md).
