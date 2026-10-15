"""Portable frozen campaign protocol; historical release plans are never rewritten."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from experiments.tmrl_test_comparison.campaign import digest
from experiments.tmrl_test_comparison.generate import ROOT as ROOT
from experiments.tmrl_test_comparison.generate import SEEDS, configuration
from experiments.tmrl_test_comparison.policy import CAMPAIGN_ASSIGNMENTS, require_standard_config

HERE = Path(__file__).resolve().parent
PLAN = HERE / "campaign.json"
CAMPAIGN_ID = "weeks-pr-20261015"
BUDGET = 8_192_000
BASE = ROOT / "artifacts/tmrl-test-comparison"


def source_pins() -> dict[str, str]:
    paths = list((ROOT / "trackmaniarl").rglob("*.py"))
    paths += list(HERE.glob("*.py")) + list(HERE.parent.glob("*.py"))
    paths += [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "pilot_tools/ppo_recovery.py",
        ROOT / "docs/benchmarks/2026-09-08-v108-config.json",
        ROOT / "my-trackmania-agent/maps/trackmaniarl-test.Map.Gbx",
        ROOT / "my-trackmania-agent/assets/trackmaniarl-test.geometry.npz",
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def job_config(algorithm: str, seed: int) -> dict[str, Any]:
    def relocate(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith("../../../../"):
            return "../" + value
        return value

    data: dict[str, Any] = relocate(configuration(algorithm, seed, "full"))
    data["run_id"] = f"tmrl-test-{CAMPAIGN_ID}-{algorithm}-s{seed}"
    data["training"]["total_transitions"] = BUDGET
    data["evaluation"]["max_step_race_time_ms"] = None
    if algorithm == "ppo":
        data["components"]["feature_pipeline"]["class_path"] = (
            "pilot_tools.ppo_recovery:PersistentPpoRecoveryPipeline"
        )
    data["metadata"].update(
        stage="weeks",
        campaign=CAMPAIGN_ID,
        timing_policy="report_only_human_authorized_20261014",
        ppo_recovery="persistent" if algorithm == "ppo" else "not_applicable",
    )
    return data


def prepare() -> dict[str, Any]:
    directory = HERE / "configs/weeks"
    if PLAN.exists() or directory.exists():
        raise RuntimeError("Release already prepared; never overwrite published or used plans")
    jobs = []
    for worker, algorithms in CAMPAIGN_ASSIGNMENTS.items():
        for algorithm in algorithms:
            for seed in SEEDS:
                data = job_config(algorithm, seed)
                if (BASE / data["run_id"]).exists():
                    raise RuntimeError("Run already exists; no replacement or resume")
                jobs.append((worker, algorithm, seed, data))
    directory.mkdir(parents=True)
    records = []
    for worker, algorithm, seed, data in jobs:
        path = directory / f"{algorithm}-s{seed}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        records.append(
            {
                "worker": worker,
                "algorithm": algorithm,
                "seed": seed,
                "run_id": data["run_id"],
                "config": path.relative_to(ROOT).as_posix(),
                "config_sha256": digest(path),
            }
        )
    plan = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "PREPARED_NOT_LAUNCHED",
        "total_transitions_per_run": BUDGET,
        "seeds": list(SEEDS),
        "evaluation_trials": 30,
        "assignments": {k: list(v) for k, v in CAMPAIGN_ASSIGNMENTS.items()},
        "excluded_algorithms": ["sd-sac", "dsac", "discrete-sac"],
        "minimum_free_disk_gib": 100,
        "maximum_run_hours": 336,
        "shutdown_grace_seconds": 600,
        "automatic_restart": False,
        "automatic_resume": False,
        "timing_policy": "report_only_human_authorized_20261014",
        "jobs": records,
        "source_pins": source_pins(),
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def verify() -> dict[str, Any]:
    plan: dict[str, Any] = json.loads(PLAN.read_text(encoding="utf-8"))
    expected = {(w, a, s) for w, algs in CAMPAIGN_ASSIGNMENTS.items() for a in algs for s in SEEDS}
    if (
        plan["campaign_id"] != CAMPAIGN_ID
        or plan["total_transitions_per_run"] != BUDGET
        or plan["minimum_free_disk_gib"] != 100
        or plan["maximum_run_hours"] != 336
        or plan["shutdown_grace_seconds"] != 600
        or plan["automatic_restart"] is not False
        or plan["automatic_resume"] is not False
        or plan["evaluation_trials"] != 30
        or plan["status"] != "PREPARED_NOT_LAUNCHED"
        or plan["seeds"] != list(SEEDS)
        or plan["assignments"] != {k: list(v) for k, v in CAMPAIGN_ASSIGNMENTS.items()}
        or plan["excluded_algorithms"] != ["sd-sac", "dsac", "discrete-sac"]
        or plan["timing_policy"] != "report_only_human_authorized_20261014"
    ):
        raise RuntimeError("Frozen campaign protocol changed")
    if plan["source_pins"] != source_pins():
        raise RuntimeError("Frozen source/assets/dependencies changed; do not launch")
    found = set()
    for job in plan["jobs"]:
        key = (job["worker"], job["algorithm"], job["seed"])
        path = ROOT / job["config"]
        if (
            key not in expected
            or key in found
            or path.resolve()
            != (HERE / "configs/weeks" / f"{job['algorithm']}-s{job['seed']}.yaml").resolve()
        ):
            raise RuntimeError("Invalid, duplicate or unassigned campaign job")
        found.add(key)
        if digest(path) != job["config_sha256"]:
            raise RuntimeError("Frozen configuration changed")
        spec = require_standard_config(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data != job_config(job["algorithm"], job["seed"]) or spec.run_id != job["run_id"]:
            raise RuntimeError("Job differs from the frozen protocol")
    if found != expected:
        raise RuntimeError("Missing campaign jobs")
    return plan


def stop_paths(worker: str) -> list[Path]:
    return [
        ROOT / "artifacts/STOP",
        BASE / "STOP",
        BASE / CAMPAIGN_ID / "STOP",
        BASE / CAMPAIGN_ID / f"STOP-{worker}",
    ] + [ROOT / "artifacts" / f"STOP-{a}" for a in CAMPAIGN_ASSIGNMENTS[worker]]
