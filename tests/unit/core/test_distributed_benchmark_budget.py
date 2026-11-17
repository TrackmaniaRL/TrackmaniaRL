"""The synthetic benchmark stops at an episode boundary without overshoot."""

from __future__ import annotations

import threading

import pytest

from scripts import benchmark_distributed_replay as benchmark
from trackmaniarl.core.environment_errors import EnvironmentPausedError


def test_budget_is_signalled_on_last_step_before_the_next_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached, release = threading.Event(), threading.Event()
    monkeypatch.setattr(benchmark, "_BUDGET_REACHED", reached)
    monkeypatch.setattr(benchmark, "_RELEASE_BUDGET", release)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _: None)
    env = benchmark.ToyEnvironment(8)
    env.reset()
    for index in range(8):
        _, _, terminated, _, _ = env.step(0)
        assert reached.is_set() is (index == 7)
        assert terminated is (index == 7)
    release.set()
    with pytest.raises(EnvironmentPausedError):
        env.reset()
    assert env.total_steps == 8
