"""Real timer/SDK STOP/checkpoint-codec integration, without game or learner."""

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.training_support import TrainingResult
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.experiments.full_qr.campaign_budget import (
    BudgetSettings,
    CampaignBudget,
    StorageReserve,
)
from trackmaniarl.experiments.full_qr.campaign_coordinator import (
    BudgetedCoordinator,
    PilotCoordinatorSettings,
    verify_final_checkpoint,
)


def test_production_settings_require_four_hours_and_a_fresh_shared_signal(tmp_path: Path) -> None:
    signal = threading.Event()
    coordinator = CoordinatorConfig(
        "127.0.0.1:0", "test-token", "fingerprint", external_stop=signal
    )
    budget = BudgetSettings(tmp_path / "budget", (StorageReserve(tmp_path, 1),))
    settings = PilotCoordinatorSettings(coordinator, budget, lambda _: None)
    assert settings.budget.duration_seconds == 14400
    with pytest.raises(ValueError, match="14400"):
        replace(settings, budget=replace(budget, duration_seconds=240))
    with pytest.raises(ValueError, match="fresh"):
        replace(settings, coordinator=replace(coordinator, resume_checkpoint=tmp_path / "old.pt"))
    signal.set()
    with pytest.raises(ValueError, match="already stopped"):
        replace(settings)


def test_real_deadline_wakes_the_same_signal_consumed_by_the_sdk(tmp_path: Path) -> None:
    signal = threading.Event()
    budget = CampaignBudget(
        BudgetSettings(tmp_path / "budget", (StorageReserve(tmp_path, 1),), duration_seconds=0.05),
        stop_signal=signal,
    )
    # Exercise the actual SDK stop-boundary method, without constructing a
    # learner or claiming a 50ms diagnostic timer is the production pilot.
    coordinator = object.__new__(BudgetedCoordinator)
    coordinator.budget, coordinator.external_stop = budget, signal
    assert not coordinator._external_stop_requested()
    budget.start()
    assert signal.wait(2)
    assert coordinator._external_stop_requested()
    state = budget.close()
    assert state["stop_reason"] == "wall_clock_limit"
    assert state["watchdog_closed"]


def test_final_checkpoint_is_actually_decoded_and_counters_are_checked(tmp_path: Path) -> None:
    codec = TorchCheckpointCodec()
    path = tmp_path / "checkpoints/distributed-update-00000042.pt"
    state = {
        "schema_version": "2.0",
        "run_fingerprint": "expected",
        "distributed": {"transitions": 12000, "updates": 42, "episodes": 3},
        "learner": {
            "schema_version": "2.0",
            "architecture_fingerprint": "synthetic-architecture",
            "online": {
                name: {"weight": torch.tensor([1.0])}
                for name in ("encoder", "temporal", "head", "strategy")
            },
            "target": {
                name: {"weight": torch.tensor([1.0])}
                for name in ("encoder", "temporal", "head", "strategy")
            },
            "optimizers": {"main": {"state": {}, "param_groups": []}},
            "objectives": {},
            "training": {"update_count": 42, "scaler": None, "rng": {}, "schedules": {}},
            "runtime": {},
        },
    }
    codec.save(state, path)
    model = SimpleNamespace(architecture_fingerprint=lambda: "synthetic-architecture")
    run = SimpleNamespace(
        run_dir=tmp_path, checkpoint_codec=codec, learner=SimpleNamespace(model=model)
    )
    result = TrainingResult(3, 12000, 42, (path,), None)
    proof = verify_final_checkpoint(run, result, "expected")
    assert proof["updates"] == 42
    assert proof["bytes"] > 0
    assert len(proof["sha256"]) == 64
    with pytest.raises(ValueError, match="counters differ"):
        verify_final_checkpoint(run, replace(result, updates=43), "expected")
    with pytest.raises(ValueError, match="identity"):
        verify_final_checkpoint(run, result, "other-run")
    state["learner"]["online"]["head"]["weight"].fill_(float("nan"))
    codec.save(state, path)
    with pytest.raises(ValueError, match="nonfinite"):
        verify_final_checkpoint(run, result, "expected")
