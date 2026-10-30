"""Offline qualification preparation checks; never start training."""

import json
from pathlib import Path

import pytest
import yaml

from experiments.headless_comparison_prototype import prepare_qualification as module


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    base = Path(__file__).resolve().parent
    reference = yaml.safe_load((base / "reference_configs/qr-s17.yaml").read_text())
    reference["components"]["environment"]["kwargs"]["config"] = {
        "runtime": {"runtime_root": str(tmp_path / "runtime")}
    }
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "source.py").write_text("# pinned\n")
    jobs = []
    for name in ("qr", "iqn", "sac", "tqc", "ppo", "q", "fqf", "discrete-sac", "redq"):
        blocked = name in ("sac", "tqc", "ppo", "redq")
        config = deployment / (name + (".yaml.blocked" if blocked else ".yaml"))
        config.write_text(yaml.safe_dump(reference))
        jobs.append(
            {
                "algorithm": name,
                "config": str(config),
                "config_sha256": module.sha(config),
                "status": "blocked_continuous_reference_qualification"
                if blocked
                else "candidate_not_qualified",
            }
        )
    (deployment / "campaign-preparation.json").write_text(
        json.dumps(
            {
                "status": "not_runnable",
                "training_started": False,
                "source_file_sha256": {"source.py": module.sha(deployment / "source.py")},
                "jobs": jobs,
            }
        )
    )
    return deployment


def test_prepares_short_budgets_and_keeps_continuous_blocked(
    candidate: Path, tmp_path: Path
) -> None:
    result = module.prepare(candidate, tmp_path / "out")
    assert result["training_started_at_preparation"] is False
    assert result["transitions"] == 2048
    assert result["trials_per_map"] == 2
    assert len(result["jobs"]) == 9
    for job in result["jobs"]:
        config = yaml.safe_load(Path(job["config"]).read_text())
        assert config["training"]["total_transitions"] == 2048
        assert config["training"]["checkpoint_milestones"] == [2048]
        assert config["evaluation"]["trials_per_map"] == 2
        if job["algorithm"] in ("sac", "tqc", "ppo", "redq"):
            assert job["config"].endswith(".blocked")
            assert job["status"] == "blocked_continuous_reference_qualification"


def test_changed_source_is_rejected_before_output_creation(candidate: Path, tmp_path: Path) -> None:
    (candidate / "source.py").write_text("# modified\n")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="source changed"):
        module.prepare(candidate, output)
    assert not output.exists()
