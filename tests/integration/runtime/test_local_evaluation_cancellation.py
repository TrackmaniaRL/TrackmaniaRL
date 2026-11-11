"""Cancellation during local evaluation must preserve resumable state."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.runtime.test_local_periodic_evaluation_resume import (
    _Evaluator,
    _Factory,
    _run,
    _spec,
    _train,
)
from trackmaniarl.core.environment_errors import EvaluationCancelledError


class _Cancel(_Evaluator):
    def __init__(self, events: list[str], stop: Path) -> None:
        super().__init__(events)
        self.stop_path = stop
        self.error: Exception = EvaluationCancelledError("cancelled")
        self.request_stop = True

    def evaluate(self, policy: object) -> dict[str, float]:
        del policy
        self.events.append("evaluation")
        if self.request_stop:
            self.stop_path.touch()
        raise self.error


@pytest.mark.parametrize("interval", [None, 1], ids=["final", "periodic"])
def test_cancelled_evaluation_with_stop_is_resumable(tmp_path: Path, interval: int | None) -> None:
    spec = _spec(tmp_path)
    spec = spec.model_copy(
        update={
            "training": spec.training.model_copy(
                update={
                    "total_transitions": 4,
                    "evaluate_every_episodes": interval,
                    "save_final_checkpoint": True,
                }
            )
        }
    )
    stop = tmp_path / "stop"
    events: list[str] = []
    run = _run(
        spec,
        _Cancel(events, stop),
        _Factory(events),
    )
    result = _train(run, stop=stop)
    assert result.evaluation is None
    checkpoint = result.checkpoints[-1]
    state = run.checkpoint_codec.load(checkpoint)
    assert state["pending_periodic_evaluation"] is (interval is not None)
    assert not checkpoint.with_name(checkpoint.name + ".evaluation.json").exists()
    stop.unlink()
    resumed_events: list[str] = []
    resumed = _run(
        spec, _Evaluator(resumed_events, stop=stop), _Factory(resumed_events, fail_on_create=True)
    )
    resumed_result = _train(resumed, checkpoint=checkpoint, stop=stop)
    assert resumed_events == ["evaluation"]
    assert resumed_result.evaluation is not None


@pytest.mark.parametrize(
    ("error", "request_stop"),
    [(EvaluationCancelledError("unsolicited"), False), (ValueError("real failure"), True)],
)
def test_unrelated_errors_are_not_suppressed(
    tmp_path: Path,
    error: Exception,
    request_stop: bool,  # noqa: FBT001
) -> None:
    spec = _spec(tmp_path)
    stop = tmp_path / "stop"
    events: list[str] = []
    evaluator = _Cancel(events, stop)
    evaluator.error = error
    evaluator.request_stop = request_stop
    run = _run(spec, evaluator, _Factory(events))
    with pytest.raises(type(error), match=str(error)):
        _train(run, stop=stop)
