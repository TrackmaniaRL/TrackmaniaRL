# Python examples

Install TrackmaniaRL into a Python 3.12 environment, then download the desired
script and run it from your own project. From a development checkout, use
`uv run python examples/SCRIPT.py` instead.

| Example | Run | Purpose |
| --- | --- | --- |
| [game_free.py](game_free.py) | `python game_free.py --output artifacts/example` | Custom environment and reward; 32-transition lifecycle with a synthetic learner |
| [train.py](train.py) | `python train.py path/to/run.yaml` | Train using the public API; optional `--resume` for full checkpoints |
| [custom_model.py](custom_model.py) | `python custom_model.py` | Custom encoder composed with bundled model parts; CPU inference |
| [own-map.yaml](own-map.yaml) | Follow the game quickstart | Trackmania configuration requiring your map and geometry |

The first and third examples require no game, GPU or private assets.
The training wrapper requires the components named in its configuration to be
installed and any game/data prerequisites to be met.

Start with the [Python quickstart](../readme/python-quickstart.md), then read the
[SDK reference](../readme/sdk.md) or [game quickstart](../readme/quickstart.md).
Keep experimental configurations and generated outputs in your application
project; these examples are portable library usage, not historical benchmarks.
