from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from trackmaniarl.core.spec import DistributedSpec
from trackmaniarl.distributed.actor_collection import CollectionContext, _wait_for_update_budget
from trackmaniarl.distributed.coordinator_checkpoint import _restore_distributed
from trackmaniarl.distributed.coordinator_ingest import _credit_updates
from trackmaniarl.distributed.coordinator_rpc import _policy_response
from trackmaniarl.distributed.coordinator_support import _Counters


@pytest.mark.parametrize(("strict", "expected"), [(True, 540.0), (False, 512.0)])
def test_credit_crossing_threshold_is_preserved_in_strict_mode(
    *, strict: bool, expected: float
) -> None:
    config = DistributedSpec(strict_update_budget=strict)
    coordinator = SimpleNamespace(
        run=SimpleNamespace(
            spec=SimpleNamespace(
                distributed=config,
                training=SimpleNamespace(warmup_transitions=10000, updates_per_transition=0.25),
            )
        ),
        counters=_Counters(transitions=10160, update_credit=500.0),
    )
    _credit_updates(coordinator, 10000)
    assert coordinator.counters.update_credit == expected


def test_strict_resume_preserves_credit_above_pause_threshold() -> None:
    coordinator = SimpleNamespace(
        run=SimpleNamespace(
            spec=SimpleNamespace(distributed=DistributedSpec(strict_update_budget=True)),
            learner=Mock(),
            replay_store=Mock(),
            sampler=Mock(),
        ),
        _recover_journal=Mock(),
    )
    with patch("trackmaniarl.distributed.coordinator_checkpoint.load_state_dict"):
        _restore_distributed(
            SimpleNamespace(coordinator=coordinator),
            {"learner": {}, "replay_store": {}, "sampler": {}},
            {"update_credit": 900.0},
        )
    assert coordinator.counters.update_credit == 900.0


def test_policy_response_releases_actor_after_learner_catches_up() -> None:
    coordinator = SimpleNamespace(
        run=SimpleNamespace(
            spec=SimpleNamespace(distributed=DistributedSpec(strict_update_budget=True))
        ),
        counters=_Counters(update_credit=512.0),
        _policy_payload=b"policy",
        _epsilon=lambda profile: 0.1,
        _should_stop=lambda: False,
    )
    assert not _policy_response(coordinator, 0, -1)["collect_allowed"]
    coordinator.counters.update_credit = 511.0
    assert _policy_response(coordinator, 0, -1)["collect_allowed"]


@pytest.mark.parametrize("stop_instead", [False, True])
def test_episode_wait_releases_controls_and_is_interruptible(*, stop_instead: bool) -> None:
    released, returned = Event(), Event()
    runtime = SimpleNamespace(
        actor_id="test",
        stop=Event(),
        collect_allowed=Event(),
        spec=SimpleNamespace(distributed=DistributedSpec(strict_update_budget=True)),
    )
    context = CollectionContext(runtime, SimpleNamespace(release_controls=released.set), None)
    result = []

    def wait() -> None:
        result.append(_wait_for_update_budget(context))
        returned.set()

    worker = Thread(target=wait, daemon=True)
    worker.start()
    try:
        assert released.wait(2)
        assert not returned.is_set()
        (runtime.stop if stop_instead else runtime.collect_allowed).set()
        assert returned.wait(2)
        assert result == [not stop_instead]
    finally:
        runtime.stop.set()
        worker.join(2)
