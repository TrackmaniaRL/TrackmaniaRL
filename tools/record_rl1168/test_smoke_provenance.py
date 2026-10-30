"""The functional smoke path must not bypass the backend qualification guards."""

import hashlib
import json
import sys
from pathlib import Path

from tools.record_rl1168 import smoke


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path, *, covered):
    source = (tmp_path / "backend").resolve()
    source.mkdir()
    entries = []
    for name in ("orientation", "multi_goal", "environment"):
        path = source / f"{name}.py"
        data = b"raise AssertionError('unqualified backend executed')\n"
        path.write_bytes(data)
        entries.append({"path": str(path), "sha256": _sha(data)})
    manifest = (tmp_path / "manifest.json").resolve()
    _write_json(manifest, entries if covered else [])
    preparation = tmp_path / "preparation"
    preparation.mkdir()
    for filename in (
        "initialization.pt",
        "route-geometry.npy",
        "demonstrations-TRAINING-ONLY.npz",
    ):
        (preparation / filename).write_bytes(filename.encode())
    plan = {
        "backend_config": {"source": str(source)},
        "backend_source_manifest": str(manifest),
        "backend_source_manifest_sha256": _sha(manifest.read_bytes()),
    }
    _write_json(preparation / "plan.json", plan)
    code = Path(smoke.__file__).parent
    _write_json(
        preparation / "source-pins.json",
        {path.name: _sha(path.read_bytes()) for path in code.glob("*.py")},
    )
    receipt = {
        "plan_sha256": _sha((preparation / "plan.json").read_bytes()),
        "source_pins_sha256": _sha((preparation / "source-pins.json").read_bytes()),
        "checkpoint_sha256": _sha((preparation / "initialization.pt").read_bytes()),
        "route_sha256": _sha((preparation / "route-geometry.npy").read_bytes()),
        "dataset_sha256": _sha((preparation / "demonstrations-TRAINING-ONLY.npz").read_bytes()),
        "deployment_manifest_sha256": "qualified-sdk",
    }
    _write_json(preparation / "preparation.json", receipt)
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    _write_json(
        qualification / "result.json",
        {
            "status": "PASS_FULL_START_BC",
            "cleanupErrors": [],
            "trials": [{}, {}, {}],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "plan_sha256": receipt["plan_sha256"],
            "source_pins_sha256": receipt["source_pins_sha256"],
            "preparation_sha256": _sha((preparation / "preparation.json").read_bytes()),
            "deployment_manifest_sha256": "qualified-sdk",
        },
    )
    return source, preparation, qualification


def _run(monkeypatch, tmp_path, preparation, qualification):
    output = tmp_path / "smoke-output"
    monkeypatch.setattr(sys, "argv", ["smoke", str(preparation), str(qualification), str(output)])
    smoke.main()
    return json.loads((output / "result.json").read_text(encoding="utf-8"))


def test_smoke_rejects_backend_not_covered_by_manifest(tmp_path, monkeypatch):
    source, preparation, qualification = _fixture(tmp_path, covered=False)
    initial_path = list(sys.path)
    result = _run(monkeypatch, tmp_path, preparation, qualification)
    assert result["status"] == "FAILED"
    assert "does not cover" in result["error"]
    assert str(source) not in sys.path
    assert sys.path == initial_path


def test_smoke_rejects_different_deployment_before_backend_import(tmp_path, monkeypatch):
    from tools.record_rl1168 import deployment

    _, preparation, qualification = _fixture(tmp_path, covered=True)
    monkeypatch.setattr(
        deployment,
        "activate_from_environment",
        lambda: {"deployment_manifest_sha256": "different-sdk"},
    )
    result = _run(monkeypatch, tmp_path, preparation, qualification)
    assert result["status"] == "FAILED"
    assert "preparation does not bind" in result["error"]
    assert result["training_episodes"] == []


def test_smoke_rejects_learner_changed_after_qualification(tmp_path, monkeypatch):
    _, preparation, qualification = _fixture(tmp_path, covered=True)
    pins_path = preparation / "source-pins.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    pins["learner.py"] = _sha(b"earlier learner version")
    _write_json(pins_path, pins)
    receipt_path = preparation / "preparation.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_pins_sha256"] = _sha(pins_path.read_bytes())
    _write_json(receipt_path, receipt)
    proof_path = qualification / "result.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["source_pins_sha256"] = receipt["source_pins_sha256"]
    proof["preparation_sha256"] = _sha(receipt_path.read_bytes())
    _write_json(proof_path, proof)
    result = _run(monkeypatch, tmp_path, preparation, qualification)
    assert result["status"] == "FAILED"
    assert "prepared source changed: learner.py" in result["error"]
    assert result["training_episodes"] == []
