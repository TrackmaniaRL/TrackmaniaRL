"""Regression tests for backend source and qualification bindings."""

import hashlib
import json
import sys

import pytest

from tools.record_rl1168.provenance import (
    _REQUIRED_SOURCES,
    load_verified_backend,
    verify_backend,
    verify_preparation,
    verify_qualification_binding,
    verify_source_pins,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def manifest_context(tmp_path):
    source = (tmp_path / "backend").resolve()
    source.mkdir()
    contents = {
        "orientation": b"value = 7\n",
        "multi_goal": b"value = 8\n",
        "environment": b"from orientation import value\nclass FallEnvironment: pass\n",
    }
    entries = []
    for name, data in contents.items():
        path = source / (name + ".py")
        path.write_bytes(data)
        entries.append({"path": str(path), "sha256": digest(data)})
    manifest = (tmp_path / "manifest.json").resolve()
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    plan = {
        "backend_config": {"source": str(source)},
        "backend_source_manifest": str(manifest),
        "backend_source_manifest_sha256": digest(manifest.read_bytes()),
    }
    return source, manifest, entries, plan


def rewrite_manifest(manifest, entries, plan):
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    plan["backend_source_manifest_sha256"] = digest(manifest.read_bytes())


def test_zero_matches_and_missing_imports_are_not_accepted(manifest_context, tmp_path):
    source, manifest, entries, plan = manifest_context
    other = (tmp_path / "other").resolve()
    other.mkdir()
    plan["backend_config"]["source"] = str(other)
    with pytest.raises(ValueError, match="does not cover"):
        verify_backend(plan)
    plan["backend_config"]["source"] = str(source)
    rewrite_manifest(manifest, entries[:-1], plan)
    with pytest.raises(ValueError, match="does not cover environment"):
        verify_backend(plan)


def test_duplicate_traversal_and_changed_module_are_rejected(manifest_context):
    source, manifest, entries, plan = manifest_context
    rewrite_manifest(manifest, [*entries, entries[0]], plan)
    with pytest.raises(ValueError, match="duplicate"):
        verify_backend(plan)
    traversal = {**entries[0], "path": str(source / ".." / "backend" / "orientation.py")}
    rewrite_manifest(manifest, [*entries, traversal], plan)
    with pytest.raises(ValueError, match="noncanonical"):
        verify_backend(plan)
    rewrite_manifest(manifest, entries, plan)
    (source / "multi_goal.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_backend(plan)


def test_backend_executes_verified_bytes_without_adding_source_to_path(
    manifest_context, monkeypatch
):
    source, _, _, plan = manifest_context
    verified = verify_backend(plan)
    (source / "orientation.py").write_text("raise AssertionError('changed')", encoding="utf-8")
    for name in ("orientation", "multi_goal", "environment"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    old_path = list(sys.path)
    try:
        cls = load_verified_backend(verified)
        assert cls.__name__ == "FallEnvironment"
        assert sys.modules["environment"].value == 7
        assert sys.path == old_path
    finally:
        for name in ("orientation", "multi_goal", "environment"):
            sys.modules.pop(name, None)


def test_backend_refuses_symlinked_module(manifest_context, tmp_path):
    source, _, _, plan = manifest_context
    original = source / "orientation.py"
    target = tmp_path / "replacement.py"
    target.write_bytes(original.read_bytes())
    original.unlink()
    try:
        original.symlink_to(target)
    except OSError:
        pytest.skip("creating symbolic links is not available")
    with pytest.raises(ValueError, match="canonical regular file"):
        verify_backend(plan)


def test_plan_and_qualification_are_bound(tmp_path):
    plan = b'{"backend_config":{"source":"original"}}'
    pins = b'{"model.py":"hash"}'
    (tmp_path / "plan.json").write_bytes(plan)
    (tmp_path / "source-pins.json").write_bytes(pins)
    (tmp_path / "preparation.json").write_text(
        json.dumps({"plan_sha256": digest(plan), "source_pins_sha256": digest(pins)}),
        encoding="utf-8",
    )
    _, _, _, plan_sha, pins_sha, receipt_sha = verify_preparation(tmp_path)
    proof = {
        "plan_sha256": plan_sha,
        "source_pins_sha256": pins_sha,
        "preparation_sha256": receipt_sha,
        "deployment_manifest_sha256": "deployment",
    }
    verify_qualification_binding(proof, plan_sha, pins_sha, receipt_sha, "deployment")
    with pytest.raises(ValueError, match="qualification does not bind"):
        verify_qualification_binding(proof, plan_sha, pins_sha, receipt_sha, "different deployment")
    with pytest.raises(ValueError, match="qualification does not bind"):
        verify_qualification_binding({}, plan_sha, pins_sha, receipt_sha, "deployment")
    (tmp_path / "plan.json").write_bytes(b'{"backend_config":{"source":"changed"}}')
    with pytest.raises(ValueError, match="preparation plan"):
        verify_preparation(tmp_path)
    (tmp_path / "plan.json").write_bytes(plan)
    (tmp_path / "source-pins.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="preparation source pins"):
        verify_preparation(tmp_path)


def test_legacy_preparation_fails_without_launching_native(tmp_path, monkeypatch):
    from tools.record_rl1168 import qualify

    preparation = tmp_path / "preparation"
    preparation.mkdir()
    (preparation / "plan.json").write_text("{}", encoding="utf-8")
    (preparation / "source-pins.json").write_text("{}", encoding="utf-8")
    (preparation / "preparation.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "qualification"
    monkeypatch.setattr(sys, "argv", ["qualify", str(preparation), str(output)])
    qualify.main()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert "preparation plan" in result["error"]
    assert result["trials"] == []


def test_all_prepared_sources_are_rechecked_after_qualification(tmp_path):
    for name in _REQUIRED_SOURCES:
        (tmp_path / name).write_text(f"# prepared {name}\n", encoding="utf-8")
    pins = {path.name: digest(path.read_bytes()) for path in tmp_path.glob("*.py")}
    verify_source_pins(pins, tmp_path)
    learner = tmp_path / "learner.py"
    original = learner.read_bytes()
    learner.write_bytes(b"# altered after qualification\n")
    with pytest.raises(ValueError, match=r"prepared source changed: learner\.py"):
        verify_source_pins(pins, tmp_path)
    learner.write_bytes(original)
    package = tmp_path / "injected"
    package.mkdir()
    (package / "__init__.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executable source set changed"):
        verify_source_pins(pins, tmp_path)
    with pytest.raises(ValueError, match="source pins are incomplete"):
        verify_source_pins({"model.py": pins["model.py"]}, tmp_path)
