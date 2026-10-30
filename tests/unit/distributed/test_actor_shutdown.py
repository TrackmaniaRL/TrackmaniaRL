"""Actor cleanup also runs when startup or environment teardown fails."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_background import (
    external_stop_loop,
    start_background_workers,
)


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


def test_started_workers_include_policy_and_shutdown_services() -> None:
    stop = threading.Event()
    runtime = SimpleNamespace(
        stop=stop,
        spec=SimpleNamespace(distributed=SimpleNamespace(max_inflight_chunks=2)),
        _sender_loop=lambda: stop.wait(1),
        _policy_loop=lambda: stop.wait(1),
        _heartbeat_loop=lambda: stop.wait(1),
        _external_stop_loop=lambda: stop.wait(1),
    )
    workers = start_background_workers(runtime)
    try:
        assert len(workers) == 5
        assert {worker.name for worker in workers} >= {
            "trackmaniarl-policy-refresh",
            "trackmaniarl-actor-heartbeat",
            "trackmaniarl-actor-shutdown",
        }
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=1)
    assert all(not worker.is_alive() for worker in workers)


def test_actor_joins_policy_worker_after_rpc_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel_closed = threading.Event()
    retired = threading.Event()
    actor = object.__new__(ActorRuntime)
    actor.actor_id, actor.stop_reason = "test", "learner requested stop"
    actor.stop = threading.Event()
    environment = SimpleNamespace(close=lambda: None)
    actor._components = lambda: (object(), SimpleNamespace(create=lambda **_: environment))
    actor._register = lambda: {}
    actor._initialize_policy = lambda _: None
    actor._actor_seed = lambda: 0
    actor._collect = lambda *_: None
    actor._raise_background_failure = lambda: None
    # Skip only the first spool-drain grace, so cancellation is deterministic;
    # the real final worker join must still wait for the policy service.
    actor._join_senders = lambda _: None
    actor.client = SimpleNamespace(close=channel_closed.set)

    def policy_refresh() -> None:
        assert channel_closed.wait(1)
        retired.set()

    worker = threading.Thread(target=policy_refresh, name="trackmaniarl-policy-refresh")

    def start(_: object) -> list[threading.Thread]:
        worker.start()
        return [worker]

    monkeypatch.setattr(
        "trackmaniarl.distributed.actor.actor_background.start_background_workers", start
    )
    actor.run_forever()
    assert retired.is_set()
    assert not worker.is_alive()


def test_external_stop_waiter_exits_on_actor_local_stop() -> None:
    runtime = SimpleNamespace(
        external_stop=threading.Event(), stop=threading.Event(), stop_reason="learner stop"
    )
    worker = threading.Thread(target=external_stop_loop, args=(runtime,))
    worker.start()
    runtime.stop.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert not runtime.external_stop.is_set()
    assert runtime.stop_reason == "learner stop"


def test_actor_refuses_success_with_unjoined_background_worker() -> None:
    stuck = SimpleNamespace(
        name="stuck-policy-refresh", join=lambda **_: None, is_alive=lambda: True
    )
    with pytest.raises(RuntimeError, match="stuck-policy-refresh"):
        ActorRuntime._join_workers([stuck])


def test_worker_join_failure_preserves_original_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = object.__new__(ActorRuntime)
    actor.actor_id, actor.stop_reason = "test", "running"
    actor.stop = threading.Event()
    actor._components = lambda: (_ for _ in ()).throw(ValueError("original failure"))
    actor._join_senders = lambda _: None
    actor.client = SimpleNamespace(close=lambda: None)
    actor._join_workers = lambda _: (_ for _ in ()).throw(RuntimeError("stuck worker"))

    with pytest.raises(ValueError, match="original failure") as failure:
        actor.run_forever()
    assert any("stuck worker" in note for note in failure.value.__notes__)
