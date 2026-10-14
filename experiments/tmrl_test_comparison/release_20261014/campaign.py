"""New, immutable colleague campaign; reuse existing supervision without editing old plans."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from experiments.tmrl_test_comparison import campaign as base
from experiments.tmrl_test_comparison.generate import ROOT
from experiments.tmrl_test_comparison.generate import configuration as original_configuration

HERE = Path(__file__).resolve().parent
ORIGINAL_HERE = HERE.parent
CAMPAIGN_ID = "weeks-20261014-v2"


def source_pins() -> dict[str, str]:
    paths = list((ROOT / "trackmaniarl").rglob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    paths += list(ORIGINAL_HERE.glob("*.py")) + list(ORIGINAL_HERE.glob("*.ps1"))
    paths += list(HERE.glob("*.py")) + list(HERE.glob("*.ps1"))
    paths += [ROOT / "pilot_tools/ppo_recovery.py",
              ROOT / "docs/benchmarks/2026-09-08-v108-config.json",
              ROOT / "my-trackmania-agent/maps/trackmaniarl-test.Map.Gbx",
              ROOT / "my-trackmania-agent/assets/trackmaniarl-test.geometry.npz"]
    return {path.relative_to(ROOT).as_posix(): base.digest(path) for path in sorted(paths)}


def configuration(algorithm: str, seed: int, stage: str) -> dict[str, Any]:
    data = original_configuration(algorithm, seed, stage)
    old_parent = ORIGINAL_HERE / "configs" / stage
    new_parent = HERE / "configs/weeks"

    def relocate(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: relocate(child) for key, child in value.items()}
        if isinstance(value, list):
            return [relocate(child) for child in value]
        if isinstance(value, str) and value.startswith("../../../../"):
            return Path(os.path.relpath((old_parent / value).resolve(), new_parent)).as_posix()
        return value

    data = cast(dict[str, Any], relocate(data))
    # Human explicitly removed latency as a launch/result gate on 14 October.
    # Keep valid measurements, runtime errors and checkpoint identity mandatory.
    data["evaluation"]["max_step_race_time_ms"] = None
    if algorithm == "ppo":
        data["components"]["feature_pipeline"]["class_path"] = (
            "pilot_tools.ppo_recovery:PersistentPpoRecoveryPipeline")
    data["metadata"].update(timing_policy="report_only_human_authorized_20261014",
                            ppo_recovery="persistent" if algorithm == "ppo" else "not_applicable")
    return data


def configure() -> None:
    base.HERE = HERE  # type: ignore[attr-defined]
    base.PLAN = HERE / "campaign.json"
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.source_pins = source_pins
    base.configuration = configuration  # type: ignore[attr-defined]


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
