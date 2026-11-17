"""Exercise real actor/factory/controller stop boundaries without game sockets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tests.unit.distributed.test_actor_multimap_suite import _Policy, _Runtime
from tests.unit.trackmania.test_map_switch_controller import FakePlugin
from tests.unit.trackmania.test_multimap_environment import _factory
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.spec import EvaluationSuiteSpec
from trackmaniarl.distributed import actor_evaluation
from trackmaniarl.distributed.evaluation_assets import evaluation_assets_digest
from trackmaniarl.trackmania import map_switch_controller, multimap_environment
from trackmaniarl.trackmania.map_switch_journal import MapSwitchJournal


@pytest.mark.skipif(os.name != "nt", reason="read-locked staging requires Windows")
def test_stop_during_switch_finishes_operation_restores_and_never_spools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    factory, item = _factory(tmp_path)
    suite = EvaluationSuiteSpec(
        name="stop-test",
        version="1",
        maps=(item, item.model_copy(update={"id": "other-copy"})),
    )
    runtime.spec.evaluation = suite
    runtime.base_dir = tmp_path
    runtime._multi_map_factory = factory  # type: ignore[assignment]
    runtime._active_evaluation_assets_sha256 = evaluation_assets_digest(suite, tmp_path)

    def unexpected_action(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("reset, inference or episode must not occur after stop")

    runtime._reset_environment = unexpected_action  # type: ignore[method-assign]
    runtime._evaluate_episode = unexpected_action  # type: ignore[method-assign]
    monkeypatch.setattr(_Policy, "act", unexpected_action)

    class StopDuringSwitch(FakePlugin):
        def __init__(self) -> None:
            super().__init__()
            self.history: list[tuple[str, str, str]] = []

        def __call__(self, request: dict[str, str]) -> dict[str, Any]:
            is_evaluation_switch = (
                request["command"] == "switch_map" and request["expected_map_uid"] == "other"
            )
            if is_evaluation_switch:
                self.state = "loading"
            response = super().__call__(request)
            self.history.append((request["command"], self.uid, str(response.get("state", ""))))
            if is_evaluation_switch:
                runtime.stop.set()
            elif request["command"] == "switch_status" and self.uid == "other":
                self.state = "ready"
            return response

    class FakeClient:
        def close(self) -> None:
            pass

    class FakeEnvironment:
        def __init__(self, *args: Any, controller: Any = None, **kwargs: Any) -> None:
            self.controller = controller if controller is not None else object()
            self.client = FakeClient()

        def release_controls(self) -> None:
            events.append("released")

        def close(self) -> None:
            events.append("closed")

        def reset(self, **kwargs: Any) -> None:
            raise AssertionError("reset must not occur after stop")

        def step(self, action: Any) -> None:
            raise AssertionError("step must not occur after stop")

    plugin = StopDuringSwitch()
    monkeypatch.setattr(map_switch_controller, "_registry_root", lambda: tmp_path / "registry")
    monkeypatch.setattr(multimap_environment, "OpenPlanetSwitchTransport", lambda *_: plugin)
    monkeypatch.setattr(
        multimap_environment.OpenPlanetEnvironmentFactory,
        "create",
        lambda *a, **kw: FakeEnvironment(),
    )
    monkeypatch.setattr(multimap_environment, "OpenPlanetEnvironment", FakeEnvironment)
    monkeypatch.setattr(actor_evaluation, "_supports_multimap_policy", lambda _: True)
    factory.validate_scheduled_evaluation_suite(suite, IdentityFeaturePipeline(), tmp_path)
    MapSwitchJournal(factory.journal_path, "instance", create=True)
    training = factory.create(seed=0)
    marker = training.switch._telemetry_quarantine_path
    try:
        assert marker.exists()
        actor_evaluation.evaluate(runtime, training, IdentityFeaturePipeline())  # type: ignore[arg-type]
        assert runtime.stop.is_set()
        assert plugin.switch_count == 3  # training, the completed evaluation switch, restore
        assert plugin.uid == "training"
        evaluation_switch = plugin.history.index(("switch_map", "other", "loading"))
        restore = next(
            i
            for i, (command, uid, _) in enumerate(plugin.history)
            if command == "switch_map" and uid == "training" and i > evaluation_switch
        )
        assert ("switch_status", "other", "loading") in plugin.history[:restore]
        assert plugin.history[:restore].count(("switch_status", "other", "ready")) >= 2
        assert not training._evaluation_active
        assert marker.exists()  # held until the whole session closes cleanly
        assert not runtime.requests
        assert not any(event.startswith("trial:") for event in events)
    finally:
        training.close()
    assert not marker.exists()
