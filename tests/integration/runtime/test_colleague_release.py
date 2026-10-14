"""Release invariants: equal budgets, portable assets and explicit timing decision."""
import copy

import pytest
import yaml

from experiments.tmrl_test_comparison import campaign as base
from experiments.tmrl_test_comparison.generate import configuration
from experiments.tmrl_test_comparison.release_20261014 import campaign as release
from trackmaniarl.core.spec import RunSpec


@pytest.fixture
def activated(monkeypatch: pytest.MonkeyPatch) -> dict:
    for name in ("HERE", "PLAN", "CAMPAIGN_ID", "source_pins", "configuration"):
        monkeypatch.setattr(base, name, getattr(base, name))
    release.configure()
    return base.verify()


def test_release_preserves_shared_protocol_and_assets(activated: dict) -> None:
    assert len(activated["jobs"]) == 15
    assert not {"sd-sac", "dsac", "discrete-sac"} & {
        job["algorithm"] for job in activated["jobs"]}
    assert len({job["run_id"] for job in activated["jobs"]}) == 15
    for job in activated["jobs"]:
        path = base.ROOT / job["config"]
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        original = configuration(job["algorithm"], job["seed"], "full")
        spec = RunSpec.model_validate(data)
        assert spec.training.total_transitions == 8_192_000
        assert spec.evaluation.trials_per_map == 30
        assert spec.evaluation.max_step_race_time_ms is None
        expected = copy.deepcopy(original["components"])
        actual = copy.deepcopy(data["components"])
        for components in (expected, actual):
            env = components["environment"]["kwargs"]["config"]
            env.pop("geometry_path")
            env.pop("steering_curve_path")
            components["feature_pipeline"]["kwargs"].pop("geometry_path")
            if job["algorithm"] == "ppo":
                components["feature_pipeline"].pop("class_path")
        assert expected == actual
        for asset in (data["evaluation"]["maps"][0]["map_path"],
                      data["evaluation"]["maps"][0]["geometry_path"]):
            assert (path.parent / asset).resolve().is_file()
        assert (path.parent / data["artifacts_dir"]).resolve() == (
            base.ROOT / "artifacts/tmrl-test-comparison").resolve()
        if job["algorithm"] == "ppo":
            assert spec.components.feature_pipeline.class_path == (
                "pilot_tools.ppo_recovery:PersistentPpoRecoveryPipeline")


def test_pin_change_blocks_before_any_launch(
    activated: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins = dict(activated["source_pins"])
    assert "pilot_tools/ppo_recovery.py" in pins
    pins["pilot_tools/ppo_recovery.py"] = "changed"
    monkeypatch.setattr(base, "source_pins", lambda: pins)
    with pytest.raises(RuntimeError, match="Frozen campaign"):
        base.verify()


def test_stop_paths_belong_to_new_campaign(activated: dict) -> None:
    for worker in activated["assignments"]:
        stops = base.stop_paths(worker)
        assert base.ROOT / ("artifacts/tmrl-test-comparison/weeks-20261014-v2/"
                            f"STOP-{worker}") in stops
        assert not any("weeks-20261008-v1" in str(path) for path in stops)
