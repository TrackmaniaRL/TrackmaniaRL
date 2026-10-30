"""Startup errors must leave a failure receipt without launching a child."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module", "receipt"),
    [("supervise", "supervision.json"), ("compare", "comparison-completion.json")],
)
def test_invalid_deployment_writes_failure_receipt(tmp_path: Path, module: str, receipt: str):
    (tmp_path / "launch.json").write_text(
        json.dumps({"preparation": str(tmp_path / "prep"), "deployment_manifest_sha256": "a" * 64}),
        encoding="utf-8",
    )
    env = dict(os.environ)
    for key in ("TMRL_V8_DEPLOYMENT", "TMRL_V8_MANIFEST", "TMRL_V8_MANIFEST_SHA256"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "tools.record_rl1168." + module, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode != 0
    result = json.loads((tmp_path / receipt).read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "qualified V8 deployment manifest SHA256 required" in result["error"]
    assert result["cohort_started"] is False
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "bc-control").exists()
    if module == "supervise":
        assert result["closure"] is None
    else:
        assert result["stages"] == []
        assert "emergency_closure" not in result
