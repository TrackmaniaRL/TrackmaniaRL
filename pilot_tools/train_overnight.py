"""PPO trend evaluation with timing reported independently of measurement integrity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pilot_tools import train_trend
from pilot_tools.ppo_overnight import measurement
from pilot_tools.ppo_pilot import validate_traces


def validate_midpoint(evaluator: Any) -> None:
    checkpoint = Path(evaluator.checkpoint)
    payload = json.loads((evaluator.run_dir / "evaluation.json").read_text())
    result = measurement(checkpoint, payload)
    validate_traces(evaluator.root, (checkpoint,))
    print(json.dumps({"midpoint_measurement": result,
                      "automatic_full_training": False}), flush=True)


def main() -> None:
    train_trend.validate_midpoint = validate_midpoint
    train_trend.main()


if __name__ == "__main__":
    main()
