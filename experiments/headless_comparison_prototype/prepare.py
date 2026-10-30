"""Prepare fresh V8 source and discrete comparison configs; never start a run.

The input native profile must come from a reviewed, pinned deployment. Continuous
jobs remain explicitly blocked until their actual actuator is implemented and
qualified. The generated QR/IQN configs also require pre-launch validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from trackmaniarl.core.spec import RunSpec
from trackmaniarl.experiments.full_qr.runtime_profile import build_source_deployment

TEAM_ALGORITHMS = ("qr", "iqn", "sac", "tqc", "ppo")
ADDITIONAL_ALGORITHMS = ("q", "fqf", "discrete-sac", "redq")
ALGORITHMS = TEAM_ALGORITHMS + ADDITIONAL_ALGORITHMS
UID = "oqIJ5rQDRrNwLPTh9H2p_W4tLof"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Preparation:
    source: Path
    template: Path
    destination: Path
    artifacts: Path
    analog_manifest: Path | None = None
    analog_manifest_sha256: str | None = None
    analog_finish: Path | None = None
    analog_finish_sha256: str | None = None
    analog_motor: Path | None = None
    analog_motor_sha256: str | None = None
    shared_analog_for_discrete: bool = False


def prepare(settings: Preparation) -> dict:
    source, template, destination, artifacts = (
        settings.source,
        settings.template,
        settings.destination,
        settings.artifacts,
    )
    source, template = source.resolve(strict=True), template.resolve(strict=True)
    if not destination.is_absolute() or destination.exists() or not artifacts.is_absolute():
        raise ValueError("fresh absolute destination and absolute artifact root required")
    profile = yaml.safe_load(template.read_text(encoding="utf-8"))
    native = copy.deepcopy(profile["components"]["environment"]["kwargs"]["config"])
    runtime = native["runtime"]
    manifest = Path(runtime["asset_manifest"])
    if _sha(manifest) != native["asset_manifest_sha256"]:
        raise ValueError("native asset manifest hash mismatch")
    event = native["event_source"]
    for name in ("finish", "motor"):
        if (
            _sha(Path(event["kwargs"][f"{name}_reference_path"]))
            != event["kwargs"][f"{name}_reference_sha256"]
        ):
            raise ValueError(f"{name} reference hash mismatch")
    if native["expected_map_uid"] != UID:
        raise ValueError("template map UID mismatch")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    runtime_root = Path(runtime["runtime_root"])
    roles = manifest_data["roles"]
    if set(roles) != {"server", "driver", "shim", "map", "tape"}:
        raise ValueError("unexpected native asset roles")
    for relative in roles.values():
        asset = (runtime_root / relative).resolve(strict=True)
        if not asset.is_relative_to(runtime_root.resolve(strict=True)) or not asset.is_file():
            raise ValueError("native asset outside runtime")
        record = manifest_data["assets"][relative]
        if asset.stat().st_size != record["bytes"] or _sha(asset) != record["sha256"]:
            raise ValueError("native asset hash or size mismatch")
    map_path = runtime_root / roles["map"]
    if any((artifacts / f"headless-comparison-{name}-s17").exists() for name in ALGORITHMS):
        raise ValueError("campaign artifacts already exist; choose a fresh artifact root")
    from trackmaniarl.trackmania.geometry import BoundaryGeometry

    geometry = Path(native["geometry_path"])
    boundary = BoundaryGeometry(geometry, expected_map_uid=UID)
    boundary.validate_map(map_path)
    analog_root = analog_map = None
    analog_refs = None
    if settings.analog_manifest is not None:
        analog_manifest = settings.analog_manifest.resolve(strict=True)
        if (
            not settings.analog_manifest_sha256
            or _sha(analog_manifest) != settings.analog_manifest_sha256
        ):
            raise ValueError("experimental analog manifest hash mismatch")
        analog_data = json.loads(analog_manifest.read_text(encoding="utf-8"))
        analog_root = Path(analog_data["runtime_root"])
        if set(analog_data["roles"]) != {"server", "driver", "shim", "map", "tape"}:
            raise ValueError("unexpected analog asset roles")
        for relative in analog_data["roles"].values():
            asset = (analog_root / relative).resolve(strict=True)
            if not asset.is_relative_to(analog_root.resolve(strict=True)) or not asset.is_file():
                raise ValueError("analog asset outside runtime")
            record = analog_data["assets"][relative]
            if asset.stat().st_size != record["bytes"] or _sha(asset) != record["sha256"]:
                raise ValueError("analog asset hash or size mismatch")
        analog_map = analog_root / analog_data["roles"]["map"]
        boundary.validate_map(analog_map)
        supplied = (
            settings.analog_finish,
            settings.analog_finish_sha256,
            settings.analog_motor,
            settings.analog_motor_sha256,
        )
        if any(item is not None for item in supplied):
            if any(item is None for item in supplied):
                raise ValueError("both analog references and their SHA256 pins are required")
            from trackmaniarl.trackmania.native_headless.full_qr.finish_reference import (
                load_reference,
            )
            from trackmaniarl.trackmania.native_headless.full_qr.motor_qualification import (
                load_motor_reference,
            )

            finish_path = settings.analog_finish.resolve(strict=True)
            motor_path = settings.analog_motor.resolve(strict=True)
            finish = load_reference(finish_path, settings.analog_finish_sha256)
            motor = load_motor_reference(motor_path, settings.analog_motor_sha256)
            pins = {
                role: analog_data["assets"][relative]["sha256"]
                for role, relative in analog_data["roles"].items()
            }
            if finish.map_uid != UID or any(pins.get(k) != v for k, v in finish.asset_pins.items()):
                raise ValueError("analog finish reference does not bind this runtime")
            if any(pins.get(k) != v for k, v in motor.asset_pins.items()):
                raise ValueError("analog motor reference does not bind this runtime")
            analog_refs = {
                "finish_reference_path": str(finish_path),
                "finish_reference_sha256": settings.analog_finish_sha256,
                "motor_reference_path": str(motor_path),
                "motor_reference_sha256": settings.analog_motor_sha256,
            }
    elif settings.analog_manifest_sha256 is not None:
        raise ValueError("analog manifest required alongside its hash")
    elif any(
        (
            settings.analog_finish,
            settings.analog_finish_sha256,
            settings.analog_motor,
            settings.analog_motor_sha256,
        )
    ):
        raise ValueError("analog references require the analog manifest")
    if settings.shared_analog_for_discrete and analog_refs is None:
        raise ValueError("shared analog runtime requires both qualified analog references")
    event["class_path"] = (
        "trackmaniarl.trackmania.native_headless.full_qr.native_events:QualifiedNativeEvents"
    )
    # Never copy a previous experiment's custom reward reference or curriculum.
    allowed = {
        "event_source",
        "runtime",
        "driver_episode_limit",
        "next_episode_reserve_bytes",
        "asset_manifest_sha256",
        "geometry_path",
        "expected_map_uid",
        "reward",
        "trace_archive_root",
        "keep_trace_archives",
    }
    native = {key: value for key, value in native.items() if key in allowed}
    receipt = build_source_deployment(source, destination)
    component_dir = destination / "experiments" / "headless_comparison_prototype"
    component_dir.mkdir(parents=True)
    for name in ("ppo_native.py", "campaign_draft.py", "prepare_qualification.py"):
        shutil.copy2(source / "experiments/headless_comparison_prototype" / name, component_dir)
    shutil.copytree(
        source / "tools/headless_campaign_queue",
        destination / "tools/headless_campaign_queue",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    source_files = sorted(
        path
        for root in (
            destination / "trackmaniarl",
            destination / "experiments",
            destination / "tools",
        )
        for path in root.rglob("*")
        if path.is_file()
    )
    source_hashes = {str(path.relative_to(destination)): _sha(path) for path in source_files}
    configs = destination / "campaign-configs"
    configs.mkdir()
    jobs = []
    team_qr_path = (
        source / "experiments/headless_comparison_prototype/reference_configs/qr-s17.yaml"
    )
    team_qr = yaml.safe_load(team_qr_path.read_text(encoding="utf-8"))
    for name in ALGORITHMS:
        reference_dir = (
            "reference_configs" if name in TEAM_ALGORITHMS else "additional_reference_configs"
        )
        reference = (
            source
            / "experiments/headless_comparison_prototype"
            / reference_dir
            / f"{name}-s17.yaml"
        )
        config = yaml.safe_load(reference.read_text(encoding="utf-8"))
        config["run_id"] = f"headless-comparison-{name}-s17"
        config["artifacts_dir"] = str(artifacts)
        model = config["components"]["model_factory"]["kwargs"]
        model["encoder"]["class_path"] = (
            "trackmaniarl.experiments.full_qr.components:BatchedIncidentEncoder"
        )
        # The historical reward is carried by the reference config, not a diagnostic run.
        reference_env = config["components"]["environment"]["kwargs"]["config"]
        if name in ADDITIONAL_ALGORITHMS:
            reference_env = team_qr["components"]["environment"]["kwargs"]["config"]
            config["training"]["total_transitions"] = 7_168_000
            config["distributed"]["strict_update_budget"] = True
            config["training"]["checkpoint_milestones"] = [
                2_048_000,
                4_096_000,
                6_144_000,
                7_168_000,
            ]
            config["metadata"]["scope"] = (
                "additional exploratory; original budget 2048000; team reward and 7168000 budget"
            )
        n = copy.deepcopy(native)
        n["reward"] = {k: reference_env[k] for k in native["reward"]}
        n["runtime"]["trace_dir"] = str(destination / "native-traces" / name)
        n["runtime"]["work_dir"] = str(
            Path(runtime["runtime_root"]) / "build/server-probe" / destination.name / name
        )
        n["trace_archive_root"] = str(artifacts / "native-archives" / name)
        n["geometry_path"] = str(geometry)
        if settings.shared_analog_for_discrete and name not in ("sac", "tqc", "ppo", "redq"):
            n["runtime"]["runtime_root"] = str(analog_root)
            n["runtime"]["asset_manifest"] = str(settings.analog_manifest)
            n["runtime"]["work_dir"] = str(
                analog_root / "build/server-probe" / destination.name / name
            )
            n["asset_manifest_sha256"] = settings.analog_manifest_sha256
            n["event_source"]["kwargs"] = analog_refs.copy()
            config["metadata"]["shared_analog_shim"] = "discrete contract; qualification pending"
        if name == "ppo":
            n["observation_pipeline"] = {
                "class_path": (
                    "experiments.headless_comparison_prototype.ppo_native:"
                    "NativePersistentPpoRecoveryPipeline"
                ),
                "kwargs": {"geometry_path": str(geometry), "expected_map_uid": UID},
            }
            config["components"]["feature_pipeline"]["class_path"] = (
                "trackmaniarl.experiments.graph_iqn_v5:BoundaryGraphFeaturePipelineV5"
            )
        config["components"]["feature_pipeline"]["kwargs"]["geometry_path"] = str(geometry)
        config["components"]["environment"] = {
            "class_path": (
                "trackmaniarl.trackmania.native_headless:NativeTelemetryEnvironmentFactory"
            ),
            "kwargs": {"config": n},
        }
        config["evaluation"]["maps"][0].update(map_path=str(map_path), geometry_path=str(geometry))
        if settings.shared_analog_for_discrete and name not in ("sac", "tqc", "ppo", "redq"):
            config["evaluation"]["maps"][0]["map_path"] = str(analog_map)
        config["metadata"]["backend"] = "native-headless-experimental"
        if name in ("sac", "tqc", "ppo", "redq"):
            job = {
                "algorithm": name,
                "status": "blocked_continuous_reference_qualification",
                "reference_sha256": _sha(reference),
            }
            if analog_root is not None and analog_map is not None:
                n["allow_experimental_continuous"] = True
                n["continuous_shim_sha256"] = analog_data["assets"][analog_data["roles"]["shim"]][
                    "sha256"
                ]
                n["runtime"]["runtime_root"] = str(analog_root)
                n["runtime"]["asset_manifest"] = str(settings.analog_manifest)
                n["runtime"]["work_dir"] = str(
                    analog_root / "build/server-probe" / destination.name / name
                )
                n["asset_manifest_sha256"] = settings.analog_manifest_sha256
                config["components"]["environment"]["class_path"] = (
                    "trackmaniarl.trackmania.native_headless.full_qr.continuous:"
                    "ExperimentalContinuousEnvironmentFactory"
                )
                config["evaluation"]["maps"][0]["map_path"] = str(analog_map)
                if analog_refs is not None:
                    n["event_source"]["kwargs"] = analog_refs.copy()
                    job["status"] = "candidate_not_qualified"
                    config["metadata"]["qualification"] = (
                        "reference pins verified; full run qualification pending"
                    )
                else:
                    config["metadata"]["qualification"] = (
                        "BLOCKED: finish/motor references bind the old shim"
                    )
                suffix = "" if analog_refs is not None else ".blocked"
                file = configs / f"{name}-s17.yaml{suffix}"
                file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                RunSpec.from_yaml(file)
                job.update(config=str(file), config_sha256=_sha(file))
            jobs.append(job)
            continue
        file = configs / f"{name}-s17.yaml"
        file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        RunSpec.from_yaml(file)
        jobs.append(
            {
                "algorithm": name,
                "status": "candidate_not_qualified",
                "config": str(file),
                "config_sha256": _sha(file),
                "reference_sha256": _sha(reference),
            }
        )
    result = {
        "status": "not_runnable",
        "training_started": False,
        "source_deployment": receipt,
        "source_file_sha256": source_hashes,
        "template_sha256": _sha(template),
        "geometry_sha256": _sha(geometry),
        "map_sha256": _sha(map_path),
        "jobs": jobs,
    }
    (destination / "campaign-preparation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--native-template", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--analog-manifest", type=Path)
    parser.add_argument("--analog-manifest-sha256")
    parser.add_argument("--analog-finish", type=Path)
    parser.add_argument("--analog-finish-sha256")
    parser.add_argument("--analog-motor", type=Path)
    parser.add_argument("--analog-motor-sha256")
    parser.add_argument("--shared-analog-for-discrete", action="store_true")
    args = parser.parse_args()
    settings = Preparation(
        args.source,
        args.native_template,
        args.destination,
        args.artifacts,
        args.analog_manifest,
        args.analog_manifest_sha256,
        args.analog_finish,
        args.analog_finish_sha256,
        args.analog_motor,
        args.analog_motor_sha256,
        args.shared_analog_for_discrete,
    )
    result = prepare(settings)
    print(json.dumps({"status": result["status"], "jobs": result["jobs"]}, indent=2))


if __name__ == "__main__":
    main()
