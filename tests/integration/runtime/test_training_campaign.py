"""Campaign membership and experimental learner exclusion."""

from pathlib import Path

import pytest
import yaml

from experiments.tmrl_test_comparison.generate import ALGORITHMS, configuration
from experiments.tmrl_test_comparison.policy import (
    CAMPAIGN_ASSIGNMENTS,
    require_standard_algorithm,
    require_standard_config,
)


def test_campaign_excludes_sd_sac_and_preserves_action_groups() -> None:
    assert "sd-sac" not in ALGORITHMS
    assert "sd-sac" not in {value for group in CAMPAIGN_ASSIGNMENTS.values() for value in group}
    assert configuration("sac", 17, "full")["metadata"]["group"] == "continuous-3"
    assert configuration("qr", 17, "full")["metadata"]["group"] == "discrete-78"
    assert configuration("sd-sac", 17, "full")["metadata"]["support_tier"] == "experimental"
    with pytest.raises(ValueError, match="Unknown"):
        require_standard_algorithm("bogus")


def test_metadata_cannot_smuggle_experimental_learner_into_queue(tmp_path: Path) -> None:
    data = configuration("sd-sac", 17, "full")
    data["metadata"]["algorithm"] = "qr"
    path = tmp_path / "spoofed.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="EXPERIMENTAL"):
        require_standard_config(path)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_single_config_has_valid_relocated_assets(algorithm: str, tmp_path: Path) -> None:
    from experiments.tmrl_test_comparison.generate import ROOT, write_configuration
    from trackmaniarl.core.spec import RunSpec

    path = tmp_path / "nested" / f"{algorithm}.yaml"
    write_configuration(algorithm, 29, "pilot", path)
    spec = RunSpec.from_yaml(path)
    assert spec.seed == 29
    assert spec.metadata["algorithm"] == algorithm
    assert spec.components.sampler.kwargs["seed"] == 29
    for asset in (spec.evaluation.maps[0].map_path, spec.evaluation.maps[0].geometry_path):
        assert (path.parent / asset).is_file()
    assert (path.parent / spec.artifacts_dir).resolve() == ROOT / "artifacts/tmrl-test-comparison"
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_configuration(algorithm, 43, "full", path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("alias", ["sd-sac", "dsac", "discrete-sac"])
def test_generator_rejects_experimental_before_writing(alias: str, tmp_path: Path) -> None:
    from experiments.tmrl_test_comparison.generate import write_configuration

    path = tmp_path / "absent" / "config.yaml"
    with pytest.raises(ValueError, match="EXPERIMENTAL"):
        write_configuration(alias, 17, "pilot", path)
    assert not path.parent.exists()
