"""V8 deployment binding checks without starting a native process."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.record_rl1168 import deployment
from trackmaniarl.experiments.full_qr.runtime_profile import _REPLACEMENTS, transform_source


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    good = (tmp_path / "verified").resolve()
    package = good / "trackmaniarl"
    source = Path(__file__).resolve().parents[2]
    shutil.copytree(
        source / "trackmaniarl",
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    for relative in _REPLACEMENTS:
        target = good / relative
        target.write_text(
            transform_source(relative, target.read_bytes().decode("utf-8")),
            encoding="utf-8",
            newline="\n",
        )
    receipt = {
        "schema": "tmrl.full-qr-source-deployment/v1",
        "protocol": "full-qr-current150/v8",
        "destination": str(good),
    }
    (good / "source-deployment.json").write_text(json.dumps(receipt), encoding="utf-8")
    data = {"schema": deployment._SCHEMA, "deployment": str(good), "files": deployment._files(good)}
    manifest = (tmp_path / "manifest.json").resolve()
    manifest.write_text(json.dumps(data), encoding="utf-8")
    bad = (tmp_path / "old-backend").resolve()
    (bad / "trackmaniarl").mkdir(parents=True)
    (bad / "trackmaniarl" / "__init__.py").write_text('IDENTITY = "old"\n', encoding="utf-8")
    return good, manifest, bad, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _run(good: Path, manifest: Path, bad: Path, digest: str, *, preload: bool = False):
    script = """
import sys
sys.path.insert(0, sys.argv[1])
if sys.argv[2] == 'preload':
    import trackmaniarl
from tools.record_rl1168.deployment import activate_from_environment
binding = activate_from_environment()
import trackmaniarl
print(binding['imported_package'])
"""
    env = dict(
        os.environ,
        TMRL_V8_DEPLOYMENT=str(good),
        TMRL_V8_MANIFEST=str(manifest),
        TMRL_V8_MANIFEST_SHA256=digest,
    )
    return subprocess.run(
        [sys.executable, "-B", "-c", script, str(bad), "preload" if preload else "clean"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def test_verified_deployment_precedes_competing_old_backend(tmp_path: Path) -> None:
    good, manifest, bad, digest = _fixture(tmp_path)
    result = _run(good, manifest, bad, digest)
    assert result.returncode == 0, result.stderr
    assert str(good / "trackmaniarl" / "__init__.py") in result.stdout


def test_already_imported_old_package_is_rejected(tmp_path: Path) -> None:
    good, manifest, bad, digest = _fixture(tmp_path)
    result = _run(good, manifest, bad, digest, preload=True)
    assert result.returncode != 0
    assert "already imported outside" in result.stderr


@pytest.mark.parametrize("mutation", ["modify", "add", "delete", "manifest"])
def test_pinned_deployment_changes_fail_closed(tmp_path: Path, mutation: str) -> None:
    good, manifest, bad, digest = _fixture(tmp_path)
    target = good / "trackmaniarl" / "__init__.py"
    if mutation == "modify":
        target.write_text('IDENTITY = "tampered"\n', encoding="utf-8")
    elif mutation == "add":
        (good / "trackmaniarl" / "injected.py").write_text("pass\n", encoding="utf-8")
    elif mutation == "delete":
        target.unlink()
    else:
        manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = _run(good, manifest, bad, digest)
    assert result.returncode != 0
    assert "V8 deployment" in result.stderr or "V8 package initializer missing" in result.stderr


def test_old_sdk_with_self_consistent_manifest_is_rejected(tmp_path: Path) -> None:
    good, manifest, bad, _ = _fixture(tmp_path)
    with (good / "trackmaniarl" / "__init__.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# An older SDK, even if the manifest matches it.\n")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"] = deployment._files(good)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result = _run(good, manifest, bad, digest)
    assert result.returncode != 0
    assert "does not match the current checkout" in result.stderr
