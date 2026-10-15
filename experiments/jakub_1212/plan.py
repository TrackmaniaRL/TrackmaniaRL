"""Immutable protocol for Jakub Szulc's first full QR run on library 1.2.12."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/tmrl-test-comparison"
CAMPAIGN_ID = "paper-1212-20261015"
PLAN = HERE / "plan.json"
CONFIG = HERE / "qr.yaml"
ASSIGNMENTS = {"Jakub Szulc": ["qr", "iqn"]}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_pins() -> dict[str, str]:
    paths = list((ROOT / "trackmaniarl").rglob("*.py"))
    paths += list(HERE.glob("*.py")) + list((HERE / "assets").iterdir())
    paths += [CONFIG, HERE / "PROTOCOL.md", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def prepare() -> dict[str, Any]:
    if PLAN.exists():
        raise RuntimeError("Never overwrite a frozen protocol")
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "library_release": "1.2.12",
        "library_commit": "d4aa669e390692a6fc255afbafb18da6e3f3d867",
        "operator": "Jakub Szulc",
        "assignments": ASSIGNMENTS,
        "planned_seeds": [17, 29, 43],
        "total_transitions_per_run": 8_192_000,
        "evaluation_trials": 30,
        "minimum_free_disk_gib": 100,
        "maximum_run_hours": 336,
        "shutdown_grace_seconds": 600,
        "automatic_restart": False,
        "automatic_resume": False,
        "automatic_next_job": False,
        "jobs": [
            {
                "worker": "jakub",
                "operator": "Jakub Szulc",
                "algorithm": "qr",
                "seed": 17,
                "run_id": data["run_id"],
                "config": CONFIG.relative_to(ROOT).as_posix(),
            }
        ],
        "source_pins": source_pins(),
    }
    PLAN.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(PLAN.read_text(encoding="utf-8"))
    required = {
        "library_release": "1.2.12",
        "assignments": ASSIGNMENTS,
        "planned_seeds": [17, 29, 43],
        "total_transitions_per_run": 8_192_000,
        "evaluation_trials": 30,
        "minimum_free_disk_gib": 100,
        "maximum_run_hours": 336,
        "shutdown_grace_seconds": 600,
        "automatic_restart": False,
        "automatic_resume": False,
        "automatic_next_job": False,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise RuntimeError("Frozen protocol settings changed")
    if result["source_pins"] != source_pins():
        raise RuntimeError("Frozen sources, configuration or assets changed")
    if result["operator"] != "Jakub Szulc" or len(result["jobs"]) != 1:
        raise RuntimeError("Only Jakub Szulc's first job is authorized here")
    if result["jobs"][0]["algorithm"] != "qr" or result["jobs"][0]["seed"] != 17:
        raise RuntimeError("Unexpected job identity")
    return result


def stop_paths(worker: str) -> list[Path]:
    if worker != "jakub":
        raise ValueError("This launcher only runs Jakub Szulc's training")
    return [
        ROOT / "artifacts/STOP",
        BASE / "STOP",
        BASE / CAMPAIGN_ID / "STOP",
        BASE / CAMPAIGN_ID / "STOP-jakub",
        ROOT / "artifacts/STOP-qr",
    ]
