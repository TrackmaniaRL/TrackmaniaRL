"""Check configs and synthetic learner updates without connecting to the game."""

from __future__ import annotations

import json
import tempfile

import torch

from experiments.tmrl_test_comparison.generate import HERE, ROOT
from trackmaniarl import RunSpec, resolve_run
from trackmaniarl.core.runtime_validation import validate_resolved_run


def main() -> None:
    torch.set_num_threads(2)
    paths = json.loads((HERE / "manifest.json").read_text())
    for item in paths:
        path = ROOT / item
        spec = RunSpec.from_yaml(path)
        for entry in spec.evaluation.maps:
            for value in (entry.map_path, entry.geometry_path):
                assert (path.parent / value).is_file(), value
    print(f"Schema and map files OK: {len(paths)} configs", flush=True)
    with tempfile.TemporaryDirectory(prefix="tmrl-comparison-check-") as temporary:
        for item in paths:
            if "/pilot/" not in item:
                continue
            path = ROOT / item
            data = RunSpec.from_yaml(path).model_dump(mode="json")
            data["artifacts_dir"] = temporary
            data["components"]["additional_loggers"] = []
            data["components"]["learner"]["kwargs"]["execution"] = {
                "device": "cpu",
                "precision": "float32",
                "torch_threads": 2,
            }
            data["training"].update(batch_size=2, sequence_length=1)
            if "ppo-" in path.name:
                data["training"].update(batch_size=1, sequence_length=8)
                data["components"]["learner"]["kwargs"].update(update_epochs=1, minibatch_size=8)
            run = resolve_run(RunSpec.model_validate(data), base_dir=path.parent)
            try:
                metrics = validate_resolved_run(run)
                assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), metrics
                print(f"PASS {path.name}: synthetic update and checkpoint round-trip", flush=True)
            finally:
                run.logger.close()


if __name__ == "__main__":
    main()
