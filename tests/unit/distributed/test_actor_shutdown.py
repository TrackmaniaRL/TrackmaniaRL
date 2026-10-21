"""Actor cleanup also runs when startup or environment teardown fails."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from trackmaniarl.distributed.actor import ActorRuntime


@pytest.mark.parametrize("failure_stage", ["register", "create", "collect", "close", "join"])
def test_actor_failure_always_stops_workers_and_closes_transport(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    events: list[str] = []
    actor = object.__new__(ActorRuntime)
    actor.actor_id, actor.stop_reason = "test", "running"
    actor.stop = threading.Event()

    def action(stage: str) -> None:
        events.append(stage)
        if stage == failure_stage:
            raise RuntimeError(f"failed {stage}")

    def create(*, seed: int) -> SimpleNamespace:
        action("create")
        return SimpleNamespace(close=lambda: action("close"))

    def join(senders: list[threading.Thread]) -> None:
        assert actor.stop.is_set()
        action("join")

    actor._components = lambda: (object(), SimpleNamespace(create=create))
    actor._register = lambda: action("register")
    actor._initialize_policy = lambda initial: None
    actor._actor_seed = lambda: 0
    actor._collect = lambda environment, pipeline: action("collect")
    actor._join_senders = join
    actor._raise_background_failure = lambda: None
    actor.client = SimpleNamespace(close=lambda: events.append("transport-close"))
    monkeypatch.setattr(
        "trackmaniarl.distributed.actor.actor_background.start_background_workers", lambda _: []
    )

    with pytest.raises(RuntimeError, match=f"failed {failure_stage}"):
        actor.run_forever()

    assert actor.stop.is_set()
    assert events[-2:] == ["join", "transport-close"]
    assert ("close" in events) == (failure_stage not in {"register", "create"})
