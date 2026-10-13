"""Standard trainer with one exact-transition diagnostic evaluation hook."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl import cli
from trackmaniarl.core import training_loop

ORIGINAL_PERIODIC = training_loop._run_periodic_evaluation


def validate_midpoint(evaluator: Any) -> None:
    from pilot_tools.ppo_pilot import pilot_measurement, validate_traces

    checkpoint = Path(evaluator.checkpoint)
    payload = json.loads((evaluator.run_dir / "evaluation.json").read_text())
    # Runtime/identity/trace failures stop the run; a low finish count does not.
    pilot_measurement(checkpoint, payload)
    validate_traces(evaluator.root, (checkpoint,))


def trend_evaluation(trainer: Any, session: Any, previous_episodes: int) -> None:
    if session.counters.transitions != 102400:
        ORIGINAL_PERIODIC(trainer, session, previous_episodes)
        return
    trainer._checkpoint_for_evaluation(session.checkpoints, session.counters)
    training_loop._close_rollout_environment(session)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            session.evaluation = trainer.run.evaluator.evaluate(trainer.run.learner.policy())
            validate_midpoint(trainer.run.evaluator)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    trainer._log("eval/suite", session.evaluation, session.counters)


def main() -> None:
    training_loop._run_periodic_evaluation = trend_evaluation
    try:
        cli.entrypoint()
    finally:
        training_loop._run_periodic_evaluation = ORIGINAL_PERIODIC


if __name__ == "__main__":
    main()
