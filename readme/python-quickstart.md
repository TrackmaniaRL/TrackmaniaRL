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

For local training, Ctrl+C during collection or a learner update does not save
the partially completed cycle. Resume from a previously completed checkpoint;
work since that checkpoint is lost. If no checkpoint has been saved yet, restart
the run. Periodic checkpoints limit the amount of work that can be lost.
New local checkpoints record whether a periodic evaluation remains due. Resume
finishes outstanding updates and then runs the evaluation before collecting
more data. A small receipt beside the checkpoint records successful completion;
preserve the receipt, `evaluation-checkpoints/` and evaluation artifacts when
moving recovery files. For checkpoint-aware evaluators, the receipt checks the
checkpoint and snapshot hashes; for the built-in evaluator it also checks the
required `evaluation.json` artifact.
An absent, invalid, moved or outdated receipt or artifact causes the evaluation
to run again. Custom evaluators without `set_checkpoint` do not receive a
snapshot, and are responsible for preserving their own result artifacts.
If a process stops after evaluation completes but before the receipt is saved,
the evaluation may be repeated. This does not guarantee durability across a
power failure.

Older checkpoints do not record this evaluation phase; their pending evaluation
cannot be reconstructed reliably. New checkpoints contain an additional field
that older library versions do not accept. Checkpoints from a different library
version can also be rejected when the run's source fingerprint differs.

Local evaluations with checkpoint-aware evaluators use a separate full snapshot
in `evaluation-checkpoints/`, so later recovery saves cannot change its bytes.
These snapshots can be as large as a full training checkpoint. When
`checkpoint_keep_last` is set, the same limit applies separately to completed
evaluation snapshots; a snapshot referenced by `evaluation.json` and an
in-progress snapshot are protected even when that exceeds the limit. Interrupted
evaluations can temporarily leave additional snapshots. Custom evaluators that
retain historical results must archive the referenced snapshots before they are
pruned. With the default unlimited retention, these snapshots accumulate until
the user removes them.

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
