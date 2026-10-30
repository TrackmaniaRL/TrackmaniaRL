"""Preparation must reject an unverified SDK before importing model modules."""

import builtins
import json
import os
import subprocess
import sys

import pytest

from tools.record_rl1168 import deployment, prepare


def test_programmatic_prepare_fails_before_imports(tmp_path, monkeypatch):
    def reject():
        raise ValueError("unverified deployment")

    monkeypatch.setattr(deployment, "activate_from_environment", reject)
    original_import = builtins.__import__
    forbidden = []

    def track_import(name, *args, **kwargs):
        if name.startswith(("trackmaniarl", "tools.record_rl1168.model")):
            forbidden.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["prepare", str(tmp_path / "previous"), str(output)])
    with pytest.raises(ValueError, match="unverified deployment"):
        prepare.main()
    report = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED_PREPARATION"
    assert "unverified deployment" in report["error"]
    assert forbidden == []


def test_cli_prepare_rejects_missing_deployment_before_work(tmp_path):
    output = tmp_path / "output"
    environment = os.environ.copy()
    for name in ("TMRL_V8_DEPLOYMENT", "TMRL_V8_MANIFEST", "TMRL_V8_MANIFEST_SHA256"):
        environment.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.record_rl1168.prepare",
            str(tmp_path / "previous"),
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    report = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAILED_PREPARATION"
    assert "deployment manifest" in report["error"]
    assert not (output / "initialization.pt").exists()
