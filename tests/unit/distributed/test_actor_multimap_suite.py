from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from trackmaniarl.distributed import actor_evaluation

_supports_multimap_policy = actor_evaluation._supports_multimap_policy
_verify_assigned_assets = actor_evaluation._verify_assigned_assets


class _Policy:
    def __init__(self) -> None:
        self.state = 0
        self.action_selector = _Selector()
        self.model = torch.nn.Linear(1, 1)

    def reset_episode(self) -> None:
        pass

    def act(self, observation: Any, mode: Any) -> int:
        self.state += 1
        self.action_selector.state += 1
        return 0

    def export_state(self) -> dict[str, Any]:
        return {"state": self.state}

    def load_state(self, state: dict[str, Any]) -> None:
        self.state = state["state"]


class _Selector:
    def __init__(self) -> None:
        self.state = 0


class _Codec:
    def decode(self, snapshot: bytes) -> dict[str, Any]:
        assert snapshot == b"assigned-policy"
        return {"state": 0}

    def encode(self, state: dict[str, Any]) -> bytes:
        assert state == {"state": 0}
        return b"assigned-policy"


class _Pipeline:
    def reset_episode(self) -> None:
        pass

    def transform_observation(self, observation: Any) -> Any:
        return observation


class _Factory:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.fail_entry = False
        self.fail_restore = False
        self.fail_verify = False
        self.reuse_training = False

    @contextmanager
    def staged_suite(self, suite: Any, base: Path, assets: str) -> Iterator[Any]:
        self.events.append("stage")
        try:
            yield suite
        finally:
            self.events.append("unstage")

    def evaluation_pipeline(self, pipeline: Any, map_spec: Any) -> Any:
        self.events.append(f"pipeline:{map_spec.id}")
        return pipeline if self.reuse_training else _Pipeline()

    @contextmanager
    def evaluation_map(self, training: Any, map_spec: Any) -> Iterator[str]:
        self.events.append(f"switch:{map_spec.id}")
        if self.fail_entry:
            raise RuntimeError("ambiguous switch")
        try:
            yield map_spec.id
        finally:
            self.events.append(f"close:{map_spec.id}")

    def restore_training(self, training: Any) -> None:
        self.events.append("restore")
        if self.fail_restore:
            raise RuntimeError("training map unresolved")

    def verify_evaluation_map(self, training: Any, map_spec: Any) -> None:
        self.events.append(f"verify:{map_spec.id}")
        if self.fail_verify:
            raise RuntimeError("evaluation map changed")


class _Runtime:
    def __init__(self, events: list[str]) -> None:
        maps = [SimpleNamespace(id=name, expected_map_uid=f"uid-{name}") for name in ("a", "b")]
        self.spec = SimpleNamespace(evaluation=SimpleNamespace(maps=maps, trials_per_map=2))
        self.base_dir = Path(".")
        self.actor_id = "actor"
        self.stop = Event()
        self._evaluation_index = 0
        self._active_evaluation_assignment_id = "assigned"
        self._active_evaluation_initial_snapshot = b"assigned-policy"
        self._active_evaluation_assets_sha256 = "assigned-assets"
        self._multi_map_factory = _Factory(events)
        self.codec = _Codec()
        self.events = events
        self.requests: list[Any] = []
        self.stop_on_first = False
        self.fail_first = False
        self.trial_starts: list[int] = []
        self.selector_starts: list[int] = []
        self.template = _Policy()

    def _evaluation_policy(self) -> tuple[_Policy, int]:
        return self.template, 7

    def _new_policy(self) -> _Policy:
        return _Policy()

    def _reset_environment(self, request: Any) -> int:
        return 0

    def _evaluate_episode(self, request: Any) -> dict[str, Any]:
        self.events.append(f"trial:{request.environment}")
        self.trial_starts.append(request.policy.state)
        self.selector_starts.append(request.policy.action_selector.state)
        request.policy.act(0, None)
        if self.stop_on_first:
            self.stop.set()
        if self.fail_first:
            self.fail_first = False
            return {"termination": "telemetry_error", "policy_version": request.version}
        return {"termination": "finished", "policy_version": request.version}

    def _evaluation_telemetry_failure(self, failure: Any) -> dict[str, Any]:
        return {"termination": "telemetry_error", "policy_version": failure.version}

    def _spool(self, request: Any) -> None:
        self.events.append("spool")
        self.requests.append(request)


@pytest.fixture(autouse=True)
def _resolve_without_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(actor_evaluation, "_verify_assigned_assets", lambda runtime: None)
    monkeypatch.setattr(actor_evaluation, "_supports_multimap_policy", lambda policy: True)


def test_suite_restores_before_spool_and_labels_all_trials() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    actor_evaluation.evaluate(runtime, object(), _Pipeline())  # type: ignore[arg-type]
    assert events == [
        "stage",
        "pipeline:a",
        "switch:a",
        "trial:a",
        "trial:a",
        "verify:a",
        "close:a",
        "pipeline:b",
        "switch:b",
        "trial:b",
        "trial:b",
        "verify:b",
        "close:b",
        "restore",
        "unstage",
        "spool",
    ]
    request = runtime.requests[0]
    assert runtime.trial_starts == [0, 0, 0, 0]
    assert runtime.selector_starts == [0, 0, 0, 0]
    assert runtime.template.action_selector.state == 0
    assert request.evaluation_snapshot == b"assigned-policy"
    assert [
        (item["map_id"], item["map_uid"], item["trial_index"]) for item in request.evaluations
    ] == [("a", "uid-a", 0), ("a", "uid-a", 1), ("b", "uid-b", 0), ("b", "uid-b", 1)]
    assert all(item["evaluation_assignment_id"] == "assigned" for item in request.evaluations)
    assert all(
        item["evaluation_assets_sha256"] == "assigned-assets" for item in request.evaluations
    )


def test_unsupported_policy_is_rejected_before_staging_or_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    monkeypatch.setattr(actor_evaluation, "_supports_multimap_policy", lambda _: False)
    with pytest.raises(RuntimeError, match="policy or selector is not supported"):
        actor_evaluation.evaluate(runtime, object(), _Pipeline())  # type: ignore[arg-type]
    assert events == []
    assert not runtime.requests


def test_stopped_suite_restores_and_never_spools_partial_results() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runtime.stop_on_first = True
    actor_evaluation.evaluate(runtime, object(), _Pipeline())  # type: ignore[arg-type]
    assert events[-3:] == ["close:a", "restore", "unstage"]
    assert not runtime.requests


def test_telemetry_failure_still_accounts_for_every_assigned_trial() -> None:
    runtime = _Runtime([])
    runtime.fail_first = True
    actor_evaluation.evaluate(runtime, object(), _Pipeline())  # type: ignore[arg-type]
    results = runtime.requests[0].evaluations
    assert [(item["map_id"], item["trial_index"]) for item in results] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    assert [item["termination"] for item in results] == [
        "telemetry_error",
        "telemetry_error",
        "finished",
        "finished",
    ]


def test_failed_context_entry_still_attempts_restore_and_propagates_failure() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runtime._multi_map_factory.fail_entry = True
    runtime._multi_map_factory.fail_restore = True
    with pytest.raises(RuntimeError, match="training map unresolved"):
        actor_evaluation.evaluate(runtime, object(), _Pipeline())  # type: ignore[arg-type]
    assert events == ["stage", "pipeline:a", "switch:a", "restore", "unstage"]
    assert not runtime.requests


def test_shared_pipeline_is_rejected_before_switch() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runtime._multi_map_factory.reuse_training = True
    with pytest.raises(RuntimeError, match="isolated pipeline"):
        actor_evaluation.evaluate(runtime, object(), _Pipeline())  # type: ignore[arg-type]
    assert events == ["stage", "pipeline:a", "unstage"]
    assert not runtime.requests


def test_map_mismatch_after_trials_restores_without_spooling() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runtime._multi_map_factory.fail_verify = True
    with pytest.raises(RuntimeError, match="evaluation map changed"):
        actor_evaluation.evaluate(runtime, object(), _Pipeline())  # type: ignore[arg-type]
    assert events[-3:] == ["close:a", "restore", "unstage"]
    assert not runtime.requests


def test_assigned_snapshot_does_not_mutate_a_shared_source_policy() -> None:
    runtime = _Runtime([])
    runtime.template.state = 9
    runtime._new_policy = lambda: runtime.template  # type: ignore[method-assign]
    policy, version = actor_evaluation._policy_from_request(
        runtime,
        (b"assigned-policy", 7),  # type: ignore[arg-type]
    )
    assert version == 7
    assert policy is not runtime.template
    assert runtime.template.state == 9
    assert policy.state == 0
    assert policy.action_selector is not runtime.template.action_selector


@pytest.mark.parametrize("part", ["policy", "selector"])
def test_assigned_evaluation_rejects_unisolatable_policy(part: str) -> None:
    class Uncopyable:
        def __deepcopy__(self, memo: Any) -> Any:
            return self

    runtime = _Runtime([])
    if part == "policy":
        runtime._new_policy = lambda: Uncopyable()  # type: ignore[method-assign]
    else:
        runtime.template.action_selector = Uncopyable()  # type: ignore[assignment]
        runtime._new_policy = lambda: runtime.template  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="cannot be isolated"):
        actor_evaluation._policy_from_request(
            runtime,
            (b"assigned-policy", 7),  # type: ignore[arg-type]
        )


def test_multi_map_rejects_model_with_shared_deepcopy() -> None:
    class SharedModel(torch.nn.Module):
        def __deepcopy__(self, memo: Any) -> Any:
            return self

    runtime = _Runtime([])
    runtime.template.model = SharedModel()  # type: ignore[assignment]
    runtime._new_policy = lambda: runtime.template  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="cannot be isolated"):
        actor_evaluation._policy_from_request(
            runtime,
            (b"assigned-policy", 7),  # type: ignore[arg-type]
        )


def test_multi_map_rejects_distinct_modules_with_shared_tensor_storage() -> None:
    class SharedStorage(torch.nn.Module):
        def __init__(self, value: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(value)

        def __deepcopy__(self, memo: Any) -> Any:
            return SharedStorage(self.weight.detach())

    runtime = _Runtime([])
    runtime.template.model = SharedStorage(torch.ones(1))  # type: ignore[assignment]
    runtime._new_policy = lambda: runtime.template  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="cannot be isolated"):
        actor_evaluation._policy_from_request(
            runtime,
            (b"assigned-policy", 7),  # type: ignore[arg-type]
        )


def test_single_map_keeps_noncopyable_legacy_policy() -> None:
    runtime = _Runtime([])
    runtime.spec.evaluation.maps = runtime.spec.evaluation.maps[:1]
    runtime.template.lock = Lock()  # type: ignore[attr-defined]
    runtime._new_policy = lambda: runtime.template  # type: ignore[method-assign]
    policy, _ = actor_evaluation._policy_from_request(
        runtime,
        (b"assigned-policy", 7),  # type: ignore[arg-type]
    )
    assert policy is runtime.template


def test_unknown_custom_selector_is_not_supported() -> None:
    from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy

    policy = object.__new__(DiscreteValuePolicy)
    policy.model = torch.nn.Linear(1, 1)  # type: ignore[assignment]
    policy.action_selector = _Selector()
    assert not _supports_multimap_policy(policy)


def test_multi_map_requires_assignment_even_when_assets_are_present() -> None:
    runtime = _Runtime([])
    runtime._active_evaluation_assignment_id = ""
    with pytest.raises(RuntimeError, match="assignment ID"):
        _verify_assigned_assets(runtime)  # type: ignore[arg-type]


@pytest.mark.parametrize("model_kind", ["scalar", "iqn_gru"])
def test_builtin_value_policy_and_selector_have_independent_storage(
    monkeypatch: pytest.MonkeyPatch,
    model_kind: str,
) -> None:
    from tests.unit._composite_value_fixtures import _scalar_model
    from trackmaniarl.algorithms.value_based.policy import (
        DiscreteValuePolicy,
        DiscreteValuePolicyConfig,
    )
    from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
    from trackmaniarl.models.contracts import RiskSpec
    from trackmaniarl.models.encoders import MlpSensorEncoder
    from trackmaniarl.models.heads import ImplicitQuantileHead, ImplicitQuantileHeadConfig
    from trackmaniarl.models.strategies import RandomQuantileStrategy
    from trackmaniarl.models.temporal import GruTemporalCore
    from trackmaniarl.trackmania.actions import TrackmaniaActionSelector

    selector = TrackmaniaActionSelector()
    model = _scalar_model()
    if model_kind == "iqn_gru":
        model = CompositeValueModel(
            CompositeModules(
                MlpSensorEncoder(4, 6, 8),
                GruTemporalCore(6, hidden_dim=6),
                ImplicitQuantileHead(
                    ImplicitQuantileHeadConfig(6, 3, cosine_count=8, dueling=True)
                ),
                RandomQuantileStrategy(
                    train_quantile_count=4, target_quantile_count=5, evaluation_quantile_count=6
                ),
            )
        )
    policy = DiscreteValuePolicy(
        model,
        torch.device("cpu"),
        DiscreteValuePolicyConfig(0.0, None, RiskSpec(), RiskSpec(), selector),
    )
    monkeypatch.setattr(actor_evaluation, "_supports_multimap_policy", _supports_multimap_policy)
    assert _supports_multimap_policy(policy)
    copied = actor_evaluation._isolated_policy(policy)
    assert copied.action_selector is not policy.action_selector
    assert copied.action_selector.weights.data_ptr() != policy.action_selector.weights.data_ptr()
    for before, after in zip(policy.model.parameters(), copied.model.parameters(), strict=True):
        assert before.untyped_storage().data_ptr() != after.untyped_storage().data_ptr()
