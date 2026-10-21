from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from trackmaniarl.commands import recovery_finetune as command
from trackmaniarl.commands import recovery_finetune_checkpoint as checkpointing
from trackmaniarl.commands import recovery_finetune_training as training
from trackmaniarl.commands.recovery_finetune_types import _FineTuneSettings
from trackmaniarl.core.checkpoints import validate_checkpoint_v2, validate_policy_checkpoint_v2


def _selection_loop() -> tuple[Any, torch.nn.Linear]:
    adapter = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        adapter.weight.fill_(2.0)
    learner = SimpleNamespace(
        model=SimpleNamespace(encoder=SimpleNamespace(recovery_adapter=adapter))
    )
    context = SimpleNamespace(learner=learner, settings=SimpleNamespace(batch_size=1, updates=3))
    data = SimpleNamespace(train_by_episode=((object(),),), validation_by_episode=())
    return SimpleNamespace(context=context, data=data), adapter


@pytest.mark.parametrize("best_update", [0, 2])
def test_selected_candidate_restores_before_policy_checkpoint(
    monkeypatch: pytest.MonkeyPatch, best_update: int
) -> None:
    loop, adapter = _selection_loop()
    candidate = training._BestCandidate(
        best_update, None, 0.8, None, {"weight": torch.ones((1, 1))}
    )

    def evaluate_restored(*_args: object) -> dict[str, float]:
        torch.testing.assert_close(adapter.weight, torch.ones((1, 1)))
        return {"restored": 1.0}

    monkeypatch.setattr(training, "_evaluate_episodes", evaluate_restored)

    result = training._finish_fine_tune(loop, candidate)

    assert result.best_update == best_update
    assert result.train_metrics == {"restored": 1.0}
    torch.testing.assert_close(adapter.weight, torch.ones((1, 1)))


def _full_learner_state() -> dict[str, object]:
    modules = {name: {} for name in ("encoder", "temporal", "head", "strategy")}
    return {
        "schema_version": "2.0",
        "architecture_fingerprint": "fingerprint",
        "online": modules,
        "target": modules,
        "optimizers": {"main": {"state": {}}, "strategy": None},
        "objectives": [],
        "training": {},
        "runtime": {},
    }


def test_recovery_output_learner_state_is_explicitly_policy_only() -> None:
    learner = SimpleNamespace(state_dict=_full_learner_state)

    policy = checkpointing._policy_only_learner_state(learner)

    assert set(policy) == {"schema_version", "architecture_fingerprint", "online"}
    validate_policy_checkpoint_v2(policy)
    with pytest.raises(ValueError, match=r"checkpoint 2\.0 keys differ"):
        validate_checkpoint_v2(policy)


def _source_architecture_fixture() -> tuple[dict[str, Any], Any, Any, list[object]]:
    state: dict[str, Any] = {
        "schema_version": "2.0",
        "learner": {"architecture_fingerprint": "source"},
    }
    run = SimpleNamespace(checkpoint_codec=SimpleNamespace(load=lambda _path: state))
    loaded: list[object] = []
    model = SimpleNamespace(
        architecture_fingerprint=lambda: "configured",
        legacy_architecture_fingerprint=lambda: "legacy",
    )
    learner = SimpleNamespace(model=model, load_policy_state_dict=loaded.append)
    return state, run, learner, loaded


def test_recovery_rejects_a_different_source_architecture(tmp_path: Path) -> None:
    _state, run, learner, loaded = _source_architecture_fixture()

    with pytest.raises(ValueError, match="architecture does not match"):
        command._require_source_architecture(run, cast(Any, learner), tmp_path / "source.pt")

    assert loaded == []


@pytest.mark.parametrize("fingerprint", ["configured", "legacy"])
def test_recovery_loads_a_compatible_source_policy(tmp_path: Path, fingerprint: str) -> None:
    state, run, learner, loaded = _source_architecture_fixture()
    state["learner"]["architecture_fingerprint"] = fingerprint

    command._require_source_architecture(run, cast(Any, learner), tmp_path / "source.pt")

    assert loaded == [state["learner"]]


def test_recovery_output_never_overwrites_source_or_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    existing = tmp_path / "existing.pt"
    source.touch()
    existing.touch()

    with pytest.raises(ValueError, match="differ from the source"):
        command._validated_output(source, source)
    with pytest.raises(FileExistsError, match="already exists"):
        command._validated_output(existing, source)


def _fine_tune_settings() -> _FineTuneSettings:
    return _FineTuneSettings(300, 128, 0.25, 0.01, 25, 24, 6, 500, 15, 80.0, 0.05, 0.20)


def test_recovery_provenance_captures_every_data_and_selection_setting() -> None:
    payload = checkpointing._provenance_settings(_fine_tune_settings())

    assert payload["validation_fraction"] == 0.25
    assert payload["log_interval"] == 25
    assert payload["minimum_usable_episodes"] == 24
    assert payload["minimum_validation_episodes"] == 6
    assert payload["minimum_gated_samples"] == 500
    assert payload["minimum_samples_per_episode"] == 15
    assert payload["maximum_normalized_recovery_time_s"] == 80.0
    assert payload["minimum_source_disagreement"] == 0.05
    assert payload["maximum_source_disagreement"] == 0.20
