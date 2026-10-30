"""Offline checks; these fixtures do not qualify native assets or control."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.headless_comparison_prototype import prepare as module
from trackmaniarl.trackmania import geometry


def test_generate_discrete_configs_and_explicitly_block_continuous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).resolve().parents[2]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    roles = {key: key for key in ("server", "driver", "shim", "map", "tape")}
    assets = {}
    for relative in roles.values():
        content = relative.encode()
        (runtime / relative).write_bytes(content)
        assets[relative] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    manifest = runtime / "assets.json"
    manifest.write_text(json.dumps({"roles": roles, "assets": assets}), encoding="utf-8")
    refs = {}
    for name in ("finish", "motor"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        refs[f"{name}_reference_path"] = str(path)
        refs[f"{name}_reference_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    geometry_path = tmp_path / "geometry.npz"
    geometry_path.write_bytes(b"geometry")
    config = {
        "event_source": {"class_path": "legacy", "kwargs": refs},
        "runtime": {"runtime_root": str(runtime), "asset_manifest": str(manifest)},
        "asset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "expected_map_uid": module.UID,
        "geometry_path": str(geometry_path),
        "reward": {"maximum_race_time_s": 150},
        "trace_archive_root": str(tmp_path / "unused"),
    }
    template = tmp_path / "old.yaml"
    template.write_text(
        yaml.safe_dump({"components": {"environment": {"kwargs": {"config": config}}}})
    )

    class Boundary:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def validate_map(self, _: Path) -> None:
            pass

    monkeypatch.setattr(geometry, "BoundaryGeometry", Boundary)

    def fake_deployment(_: Path, destination: Path) -> dict:
        destination.mkdir()
        return {"training_started": False}

    monkeypatch.setattr(module, "build_source_deployment", fake_deployment)
    monkeypatch.setattr(module.RunSpec, "from_yaml", lambda _: SimpleNamespace())
    result = module.prepare(
        module.Preparation(source, template, tmp_path / "out", tmp_path / "artifacts")
    )
    assert result["status"] == "not_runnable"
    assert not result["training_started"]
    assert [job["status"] for job in result["jobs"]] == [
        "candidate_not_qualified",
        "candidate_not_qualified",
        *["blocked_continuous_reference_qualification"] * 3,
        *["candidate_not_qualified"] * 3,
        "blocked_continuous_reference_qualification",
    ]
    qr = yaml.safe_load((tmp_path / "out/campaign-configs/qr-s17.yaml").read_text())
    assert qr["training"]["total_transitions"] == 7_168_000
    assert (
        qr["components"]["environment"]["kwargs"]["config"]["reward"]["maximum_race_time_s"] == 150
    )
    assert not (tmp_path / "out/campaign-configs/sac-s17.yaml").exists()
    q = yaml.safe_load((tmp_path / "out/campaign-configs/q-s17.yaml").read_text())
    assert q["training"]["total_transitions"] == 7_168_000
    assert "additional exploratory" in q["metadata"]["scope"]
    with pytest.raises(ValueError, match="requires both qualified analog references"):
        module.prepare(
            module.Preparation(
                source,
                template,
                tmp_path / "unqualified",
                tmp_path / "artifacts",
                shared_analog_for_discrete=True,
            )
        )
    assert not (tmp_path / "unqualified").exists()
    manifest_data = json.loads(manifest.read_text())
    manifest_data["runtime_root"] = str(runtime)
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    config["asset_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    template.write_text(
        yaml.safe_dump({"components": {"environment": {"kwargs": {"config": config}}}})
    )
    result_analog = module.prepare(
        module.Preparation(
            source,
            template,
            tmp_path / "analog",
            tmp_path / "artifacts",
            manifest,
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )
    )
    ppo_job = next(job for job in result_analog["jobs"] if job["algorithm"] == "ppo")
    assert ppo_job["status"] == "blocked_continuous_reference_qualification"
    ppo = yaml.safe_load(Path(ppo_job["config"]).read_text())
    assert ppo_job["config"].endswith(".yaml.blocked")
    assert ppo["components"]["environment"]["kwargs"]["config"]["allow_experimental_continuous"]
    assert "BLOCKED" in ppo["metadata"]["qualification"]
