from __future__ import annotations

from pathlib import Path

import pytest
import torch

_MUST_HAVE_TESTS = frozenset(
    {
        "tests/unit/core/test_run_spec_serialization.py::test_run_spec_is_frozen_and_round_trips_through_yaml",
        "tests/unit/core/test_n_step_replay.py::test_n_step_return_stops_on_termination_and_does_not_bootstrap",
        "tests/unit/learning/test_algorithms.py::test_tqc_updates_across_target_quantile_shapes",
        "tests/unit/models/test_value_learner.py::test_iqn_builds_resumable_exact_evaluated_policy_checkpoint",
        "tests/unit/trackmania/test_lidar_geometry.py::test_lidar_pipeline_validates_schema_and_builds_masked_local_observation",
        "tests/integration/runtime/test_core_runtime.py::test_resolved_run_writes_manifest_and_smoke_checkpoint",
        "tests/integration/runtime/test_cli_training.py::test_smoke_training_reserves_transitions_for_a_learner_update",
        "tests/integration/runtime/test_distributed_end_to_end.py::test_two_fake_actors_feed_slow_learner_without_data_loss",
        "tests/integration/trackmania/test_environment_runtime.py::test_environment_step_reports_applied_control_and_race_time_delta",
        "tests/unit/test_release_distribution.py::test_distribution_contains_current_packaging_sources",
    }
)


def pytest_configure() -> None:
    (Path(__file__).resolve().parents[1] / ".pytest-cache").mkdir(exist_ok=True)


@pytest.fixture
def symlink_support(tmp_path: Path) -> None:
    """Skip link-dependent cases only when this Windows account lacks the capability."""
    target = tmp_path / "symlink-capability-target"
    target.mkdir()
    try:
        (tmp_path / "symlink-capability-link").symlink_to(target, target_is_directory=True)
    except NotImplementedError:
        pytest.skip("this filesystem cannot create symbolic links")
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symbolic links require Developer Mode or an elevated test account")
        raise


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Label the small, release-blocking regression suite kept in the default run."""
    for item in items:
        test_id = item.nodeid.split("[", maxsplit=1)[0]
        if test_id in _MUST_HAVE_TESTS:
            item.add_marker(pytest.mark.must_have)


@pytest.fixture(autouse=True)
def _isolate_test_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(
        "trackmaniarl.algorithms.execution.visible_accelerators",
        lambda: set(),
    )
