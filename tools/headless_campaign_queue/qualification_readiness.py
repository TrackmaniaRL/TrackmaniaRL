"""Assess evidence for a *first*, explicitly experimental campaign.

This command is read-only except for its new output file. It never launches,
creates a runnable manifest, or claims full-budget or desktop-client parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.headless_campaign_queue.qualification_audit import audit, read_object, sha


def physical_sources(config: Path, expected_uid: str) -> dict[str, Any]:
    """Revalidate pinned source assets and primary motor/Finish evidence offline."""
    from trackmaniarl.core.spec import RunSpec
    from trackmaniarl.trackmania.geometry import BoundaryGeometry
    from trackmaniarl.trackmania.native_headless.config import NativeRuntimeConfig
    from trackmaniarl.trackmania.native_headless.full_qr.factory import verify_bound_assets
    from trackmaniarl.trackmania.native_headless.full_qr.finish_reference import load_reference
    from trackmaniarl.trackmania.native_headless.full_qr.motor_qualification import (
        load_motor_reference,
    )

    spec = RunSpec.from_yaml(config)
    component = spec.components.environment
    if not component.class_path.startswith("trackmaniarl.trackmania.native_headless"):
        raise ValueError("native headless environment required")
    value = component.kwargs["config"]
    if value["expected_map_uid"] != expected_uid:
        raise ValueError("native map identity mismatch")
    runtime = NativeRuntimeConfig.from_mapping(value["runtime"])
    assets = verify_bound_assets(runtime, value["asset_manifest_sha256"])
    geometry = BoundaryGeometry(Path(value["geometry_path"]), expected_map_uid=expected_uid)
    geometry.validate_map(assets.paths["map"])
    event = value["event_source"]
    if (
        event["class_path"]
        != "trackmaniarl.trackmania.native_headless.full_qr.native_events:QualifiedNativeEvents"
    ):
        raise ValueError("qualified source binding required")
    kwargs = event["kwargs"]
    finish_path, motor_path = (
        Path(kwargs["finish_reference_path"]),
        Path(kwargs["motor_reference_path"]),
    )
    finish = load_reference(finish_path, kwargs["finish_reference_sha256"])
    motor = load_motor_reference(motor_path, kwargs["motor_reference_sha256"])
    pins = {role: sha(path) for role, path in assets.paths.items()}
    finish_roles = {"server", "map", "shim"}
    if (
        finish.map_uid != expected_uid
        or set(finish.asset_pins) != finish_roles
        or any(finish.asset_pins[role] != pins[role] for role in finish_roles)
        or dict(motor.asset_pins) != pins
    ):
        raise ValueError("reference sources do not bind to configured assets and map")
    raw_motor = read_object(motor_path)
    native_replays = raw_motor.get("native_replays")
    replay_pins = {}
    if native_replays is not None:
        if set(native_replays) != {"right", "left", "straight"}:
            raise ValueError("native replay evidence incomplete")
        for name, entry in native_replays.items():
            path = (motor_path.parent / entry["name"]).resolve(strict=True)
            if path.parent != motor_path.parent.resolve() or sha(path) != entry["sha256"]:
                raise ValueError("native motor replay evidence changed")
            replay_pins[name] = entry["sha256"]
    continuous = component.class_path.endswith(":ExperimentalContinuousEnvironmentFactory")
    if continuous and (
        value.get("allow_experimental_continuous") is not True
        or value.get("continuous_shim_sha256") != pins["shim"]
    ):
        raise ValueError("experimental continuous shim pin missing or mismatched")
    return {
        "asset_manifest_sha256": value["asset_manifest_sha256"],
        "asset_pins": pins,
        "finish_reference_sha256": kwargs["finish_reference_sha256"],
        "motor_reference_sha256": kwargs["motor_reference_sha256"],
        "native_replay_sha256": replay_pins,
        "continuous_experimental": continuous,
    }


def assess(inventory: dict, physical: dict[str, dict], draft: dict) -> dict:
    rows = inventory["results"]
    expected = [job["id"] for job in draft["jobs"]]
    if (
        len(expected) < 5
        or len(set(expected)) != len(expected)
        or {"qr", "iqn", "sac", "tqc", "ppo"} - set(expected)
    ):
        raise ValueError("draft lacks the complete team comparison")
    if [row.get("id") for row in rows] != expected:
        raise ValueError("evidence inventory does not match draft jobs")
    details, reasons = [], []
    for row in rows:
        name = row["id"]
        missing = []
        if row.get("status") == "invalid_or_incomplete":
            missing.append("invalid short/full provenance")
        if not row.get("short_receipt"):
            missing.append("short training and evaluation")
        if not row.get("same_checkpoint_restore"):
            missing.append("offline restore of the same checkpoint")
        source = physical.get(name, {})
        if source.get("status") != "PASS":
            missing.append("configured physical and Finish sources")
        if missing:
            reasons.append({"id": name, "missing": missing})
        details.append(
            {
                "id": name,
                "short_receipt": row.get("short_receipt"),
                "same_checkpoint_restore": row.get("same_checkpoint_restore"),
                "physical": source,
                "short_full_config_differences": row.get("config_differences", []),
            }
        )
    return {
        "kind": "headless-first-campaign-readiness-v1",
        "status": "NOT_READY" if reasons else "READY_FOR_FIRST_CAMPAIGN",
        "campaign_not_started": True,
        "automatic_launch_authorized": False,
        "missing": reasons,
        "jobs": details,
        "limitations": [
            (
                "Readiness is for a first experimental campaign, not pro"
                "of of full-budget completion or recovery."
            ),
            (
                "Short runs differ in budget, warmup, trial count and ou"
                "tput paths; their differences are retained per job."
            ),
            (
                "Discrete client comparisons and sampled continuous acti"
                "on observations do not prove arbitrary analog physics/r"
                "eward equivalence."
            ),
            (
                "The actual final transition count may exceed the target"
                " and is subject to the declared acceptance bound."
            ),
            "The report does not authorize a launch and does not produce a runnable manifest.",
        ],
    }


def build(draft_path: Path, short_manifest: Path, restore_reports: list[Path]) -> dict:
    import importlib

    import tools.headless_campaign_queue.qualification_audit as inventory_module
    from trackmaniarl.core.spec import RunSpec

    draft = read_object(draft_path)
    deployment = Path(read_object(short_manifest)["candidate_deployment"]).resolve(strict=True)
    namespace = importlib.import_module("experiments")
    namespace_roots = {Path(path).resolve() for path in getattr(namespace, "__path__", ())}
    if not namespace_roots or any(not root.is_relative_to(deployment) for root in namespace_roots):
        raise ValueError(
            "experiments namespace is contaminated by paths outside the frozen deployment"
        )
    inventory = audit(draft_path, short_manifest, restore_reports)
    physical: dict[str, dict] = {}
    cache: dict[str, dict] = {}
    for job in draft["jobs"]:
        name = job["id"]
        try:
            # This re-reads the specified config after the inventory's hash check.
            config = Path(job["config"])
            if sha(config) != job["config_sha256"]:
                raise ValueError("config changed after evidence inventory")
            component = RunSpec.from_yaml(config).components.environment
            bound = component.kwargs["config"]
            runtime = {
                k: v for k, v in bound["runtime"].items() if k not in {"trace_dir", "work_dir"}
            }
            key = json.dumps(
                {
                    "class": component.class_path,
                    "runtime": runtime,
                    "assets": bound["asset_manifest_sha256"],
                    "geometry": bound["geometry_path"],
                    "map": bound["expected_map_uid"],
                    "source": bound["event_source"],
                    "analog": bound.get("allow_experimental_continuous"),
                    "shim": bound.get("continuous_shim_sha256"),
                },
                sort_keys=True,
                default=str,
            )
            if key not in cache:
                cache[key] = physical_sources(config, draft["map_uid"])
            result = cache[key]
            physical[name] = {"status": "PASS", **result}
        except Exception as exc:
            physical[name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    result = assess(inventory, physical, draft)
    result.update(
        auditor_sha256=sha(Path(__file__)),
        inventory_code_sha256=sha(Path(inventory_module.__file__)),
        draft_sha256=sha(draft_path),
        short_manifest_sha256=sha(short_manifest),
        restore_reports=[
            {"path": str(path.resolve()), "sha256": sha(path)} for path in restore_reports
        ],
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--short-manifest", required=True, type=Path)
    parser.add_argument("--restore-report", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        parser.error("output must be a fresh absolute file")
    report = build(args.draft, args.short_manifest, args.restore_report)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "sha256": sha(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
