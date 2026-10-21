"""Fresh-process CLI guard tests; no model/runtime import or engine execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

CHECKOUT = Path(__file__).resolve().parents[4]
MODULE = "examples.native_headless.train"


def _probe(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    program = (
        "import json,runpy,sys\n"
        f"sys.argv={[MODULE, *arguments]!r}\n"
        "status=0\n"
        "try:runpy.run_module('examples.native_headless.train',run_name='__main__')\n"
        "except SystemExit as error:status=error.code\n"
        "print('CLI_GUARD='+json.dumps({'status':status,'torch':'torch' in sys.modules,"
        "'runtime':any(name.startswith('trackmaniarl.trackmania.native_headless._runtime') "
        "for name in sys.modules)}))\n"
    )
    return subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=CHECKOUT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _metadata(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("CLI_GUARD=")]
    assert len(lines) == 1, result.stdout
    value: dict[str, Any] = json.loads(lines[0].removeprefix("CLI_GUARD="))
    assert value["torch"] is False
    assert value["runtime"] is False
    return value


def _paths(root: Path) -> list[str]:
    return [
        "--runtime-root",
        str(root / "not-read-runtime"),
        "--asset-manifest",
        str(root / "not-read-manifest.json"),
        "--out",
        str(root / "output"),
        "--work",
        str(root / "work"),
    ]


def test_help_never_imports_model_or_native_runtime() -> None:
    result = _probe(["--help"])
    assert _metadata(result)["status"] == 0
    assert "--execute-native-model" in result.stdout


def test_explicit_opt_in_is_required_before_optional_imports(tmp_path: Path) -> None:
    result = _probe(_paths(tmp_path))
    assert _metadata(result)["status"] == 2
    assert "execution requires --execute-native-model" in result.stderr
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "work").exists()


def test_episode_cap_rejects_before_model_or_asset_reads(tmp_path: Path) -> None:
    result = _probe(["--execute-native-model", *_paths(tmp_path), "--train-episodes", "4"])
    assert _metadata(result)["status"] == 2
    assert "one to three" in result.stderr
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "work").exists()


def test_portable_profile_requires_explicit_deployment_before_model_import(tmp_path: Path) -> None:
    result = _probe(["--execute-native-model", *_paths(tmp_path)])
    assert _metadata(result)["status"] == 2
    assert "--deployment-manifest is required" in result.stderr
    assert not (tmp_path / "output").exists()


def test_existing_output_is_never_overwritten() -> None:
    # Outside checkout: isolate the exists guard from the separate checkout guard.
    with TemporaryDirectory() as directory:
        root = Path(directory)
        out = root / "output"
        out.mkdir()
        sentinel = out / "sentinel.txt"
        sentinel.write_text("preserve fixture bytes", encoding="utf-8")
        result = _probe(["--execute-native-model", *_paths(root)])
        assert _metadata(result)["status"] == 2
        assert "fresh output/work" in result.stderr
        assert sentinel.read_text(encoding="utf-8") == "preserve fixture bytes"
        assert not (root / "work").exists()


def test_plain_import_is_inert() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import importlib,sys;importlib.import_module('examples.native_headless.train');"
            "assert 'torch' not in sys.modules;"
            "assert not any(n.startswith('trackmaniarl.trackmania.native_headless._runtime') "
            "for n in sys.modules);print('inert')",
        ],
        cwd=CHECKOUT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "inert"
