"""Bounded scenario plans and materialized per-map configurations."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from trackmaniarl.research.manifest import load_manifest, preflight, write_json

SCENARIOS = ("per-map", "multi-map", "zero-shot", "fine-tune")


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    config: str


class Protocol(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"]
    scenario: Literal["per-map", "multi-map", "zero-shot", "fine-tune"]
    training_seeds: list[StrictInt] = Field(min_length=1)
    evaluation_trials_per_checkpoint: int = Field(gt=0)
    steps_per_training_run: int = Field(ge=0)
    timeout_s: float = Field(gt=0, allow_inf_nan=False)
    checkpoint_rule: str = Field(min_length=1)
    variants: list[Variant] = Field(min_length=1)
    required_splits: dict[Literal["development", "train", "validation", "test"], int]
    runtime_gate: str


def plan_runs(manifest_path: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    """Materialize configs only after asset validation; do not execute any command."""
    preflight(manifest_path)
    manifest = load_manifest(manifest_path)
    protocol = Protocol.model_validate_json(protocol_path.read_text(encoding="utf-8")).model_dump()
    scenario = protocol["scenario"]
    if scenario not in SCENARIOS or protocol.get("schema_version") != "1":
        raise ValueError("unsupported research protocol")
    seeds = protocol["training_seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("training_seeds must be nonnegative integer identifiers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be independent identifiers, not duplicate trials")
    blockers = []
    required = protocol["required_splits"]
    if not required or any(count < 1 for count in required.values()):
        raise ValueError("required_splits must declare positive map counts")
    if scenario == "per-map" and set(required) - {"development", "train"}:
        raise ValueError("per-map training cannot use validation or test splits")
    if scenario == "per-map" and protocol["steps_per_training_run"] < 1:
        raise ValueError("per-map training requires a positive interaction budget")
    for split, count in required.items():
        if sum(entry.split == split for entry in manifest.maps) < count:
            blockers.append(f"requires {count} actual map(s) in split {split}")
    if scenario != "per-map":
        blockers.append(protocol["runtime_gate"])
    payload = {
        "schema_version": "1",
        "scenario": scenario,
        "status": "blocked" if blockers else "prepared",
        "blockers": blockers,
        "runs": [],
        "executed": False,
    }
    if blockers:
        write_json(output / "plan.json", payload)
        return payload
    for variant in protocol["variants"]:
        base = yaml.safe_load((protocol_path.parent / variant["config"]).read_text("utf-8"))
        for entry in manifest.maps:
            if entry.split not in required:
                continue
            for seed in seeds:
                config = copy.deepcopy(base)
                run_id = f"research-{entry.id}-{variant['id']}-s{seed}"
                config["run_id"], config["seed"] = run_id, seed
                config["artifacts_dir"] = str((output / "artifacts").resolve())
                config["training"]["total_transitions"] = protocol["steps_per_training_run"]
                config["evaluation"]["trials_per_map"] = protocol[
                    "evaluation_trials_per_checkpoint"
                ]
                env = config["components"]["environment"]["kwargs"]["config"]
                for key in ("geometry_path", "reward_points_path"):
                    env.pop(key, None)
                key = "geometry_path" if entry.reward_kind == "geometry" else "reward_points_path"
                reward_path = str((manifest_path.parent / entry.reward_path).resolve())
                env.update({key: reward_path, "expected_map_uid": entry.uid})
                env["maximum_race_time_s"] = protocol["timeout_s"]
                config["evaluation"]["maps"] = [
                    {
                        "id": entry.id,
                        key: reward_path,
                        "expected_map_uid": entry.uid,
                        "map_path": str((manifest_path.parent / entry.map_path).resolve())
                        if entry.map_path
                        else None,
                    }
                ]
                config_path = (output / f"{run_id}.yaml").resolve()
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with config_path.open("x", encoding="utf-8") as stream:
                    yaml.safe_dump(config, stream, sort_keys=False)
                payload["runs"].append(
                    {
                        "run_id": run_id,
                        "map_uid": entry.uid,
                        "training_seed": seed,
                        "config": str(config_path),
                        "command": [
                            "trackmaniarl",
                            "train",
                            str(config_path),
                            "--stop-file",
                            str((output / f"{run_id}.stop").resolve()),
                        ],
                    }
                )
    write_json(output / "plan.json", payload)
    return payload
