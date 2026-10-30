"""Owned storage fault tests; no game or native engine is launched."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless import training_storage as storage

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux native no-follow directory I/O")


def test_directory_bytes_deduplicates_roots_and_never_follows_links(tmp_path: Path) -> None:
    root, foreign = tmp_path / "run", tmp_path / "foreign"
    root.mkdir()
    foreign.mkdir()
    child = root / "nested"
    child.mkdir()
    (child / "row.bin").write_bytes(b"abc")
    (foreign / "large.bin").write_bytes(b"x" * 100_000)
    link = root / "foreign-link"
    link.symlink_to(foreign, target_is_directory=True)
    expected = 3 + link.lstat().st_size
    assert storage.directory_bytes([root, child, root, tmp_path / "absent"]) == expected
    assert storage.directory_bytes([link]) == link.lstat().st_size


def test_explicit_root_cannot_traverse_a_parent_symlink(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (tmp_path / "alias").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(storage.TrainingStorageError, match="traverse a symlink"):
        storage.directory_bytes([tmp_path / "alias" / "child"])


def test_accounting_has_a_finite_entry_limit_and_handles_owned_socket_name(tmp_path: Path) -> None:
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(f"/proc/self/fd/{directory}/control.sock")
    finally:
        os.close(directory)
    assert storage.directory_bytes([tmp_path]) == 0
    (tmp_path / "row.bin").write_bytes(b"evidence")
    with pytest.raises(storage.TrainingStorageError, match="entry limit"):
        storage.directory_bytes([tmp_path], max_entries=2)
    assert storage.directory_bytes([tmp_path]) == len(b"evidence")


def test_atomic_json_replaces_only_after_complete_finite_serialization(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(b"previous report")
    receipt = storage.atomic_json(path, {"complete": True, "rows": 5})
    assert json.loads(path.read_bytes()) == {"complete": True, "rows": 5}
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Out of range float"):
        storage.atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_bytes())["complete"] is True


def test_budget_failure_preserves_previous_json_and_failed_stage(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    original = b"previous report"
    path.write_bytes(original)
    with pytest.raises(storage.TrainingStorageError, match="byte limit"):
        storage.atomic_json(path, {"value": "larger than remaining space"}, max_bytes=2)
    assert path.read_bytes() == original
    stages = list(tmp_path.glob(".report.json.stage-*"))
    assert len(stages) == 1
    assert stages[0].stat().st_size <= 2
    assert storage.directory_bytes([tmp_path]) == len(original) + stages[0].stat().st_size


def test_actual_torch_checkpoint_roundtrip_and_receipt(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "latest.pt"
    tensor = torch.tensor([1.0, -0.5, 3.0])
    receipt = storage.atomic_checkpoint(path, {"tensor": tensor, "episode": 2}, max_bytes=8192)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["tensor"], tensor)
    assert loaded["episode"] == 2
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_torch_zip_budget_failure_keeps_old_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    path.write_bytes(b"known prior checkpoint")
    with pytest.raises(storage.TrainingStorageError, match="byte limit"):
        storage.atomic_checkpoint(path, {"rows": list(range(100))}, max_bytes=1)
    assert path.read_bytes() == b"known prior checkpoint"
    assert all(stage.stat().st_size <= 1 for stage in tmp_path.glob(".latest.pt.stage-*"))


def test_replace_failure_retains_old_report_and_complete_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(b"prior")

    def fail_replace(*args: Any, **kwargs: Any) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        storage.atomic_json(path, {"new": True})
    assert path.read_bytes() == b"prior"
    stages = list(tmp_path.glob(".report.json.stage-*"))
    assert len(stages) == 1
    assert json.loads(stages[0].read_bytes()) == {"new": True}


def test_atomic_target_cannot_follow_a_symlink(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.json"
    foreign.write_bytes(b"foreign evidence")
    path = tmp_path / "report.json"
    path.symlink_to(foreign)
    with pytest.raises(storage.TrainingStorageError, match="symlink"):
        storage.atomic_json(path, {"must": "not publish"})
    assert foreign.read_bytes() == b"foreign evidence"
    assert list(tmp_path.glob(".report.json.stage-*")) == []


class OwnedEnvironment:
    def __init__(self, root: Path, *, close_error: bool = False) -> None:
        self.config = SimpleNamespace(trace_dir=root)
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise OSError("retained raw publication failed")


def cleanup_receipt(path: Path, **changes: Any) -> None:
    value = {
        "failures": [],
        "supervisor": {"remainingOwned": [], "cleanupErrors": []},
        "retainedLocalDirectory": "/tmp/tmre-immediate-v7-storage-unit",
        **changes,
    }
    (path / "close-v7.json").write_text(json.dumps(value), encoding="utf-8")


def test_cleanup_requires_original_proof_and_returns_explicit_extra_root(tmp_path: Path) -> None:
    cleanup_receipt(tmp_path)
    environment = OwnedEnvironment(tmp_path)
    report = storage.verify_native_cleanup(environment)
    assert report["verified"] is True
    assert report["owned_processes_gone"] is True
    assert report["retained_local_directory"] == "/tmp/tmre-immediate-v7-storage-unit"
    assert environment.close_calls == 1


def test_close_failure_can_still_prove_zero_processes_without_claiming_clean_success(
    tmp_path: Path,
) -> None:
    cleanup_receipt(tmp_path)
    report = storage.verify_native_cleanup(OwnedEnvironment(tmp_path, close_error=True))
    assert report["verified"] is False
    assert report["owned_processes_gone"] is True
    assert "publication failed" in report["close_errors"][0]


@pytest.mark.parametrize("mode", ["inotify", "full-scan"])
def test_cleanup_preserves_literal_trace_accounting_metadata(tmp_path: Path, mode: str) -> None:
    accounting = {
        "mode": mode,
        "fallback_reason": "fixture fallback" if mode == "full-scan" else None,
        "full_scans": 3,
    }
    cleanup_receipt(tmp_path, traceAccounting=accounting)
    report = storage.verify_native_cleanup(OwnedEnvironment(tmp_path))
    assert report["verified"] is True
    assert report["trace_accounting"] == accounting


def test_fifo_cleanup_receipt_without_writer_cannot_block_verification(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "close-v7.json")
    source = (
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "from trackmaniarl.trackmania.native_headless import training_storage as storage\n"
        "environment = SimpleNamespace("
        "config=SimpleNamespace(trace_dir=Path(sys.argv[1])), close=lambda: None)\n"
        "started = time.monotonic()\n"
        "report = storage.verify_native_cleanup(environment)\n"
        "elapsed = time.monotonic() - started\n"
        "print(json.dumps({'report': report, 'verification_seconds': elapsed}), flush=True)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        # Cold Torch imports and interpreter teardown are outside the bounded
        # FIFO verification itself. A blocked operation still fails this
        # independent outer watchdog; successful verification must be prompt.
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert result["verification_seconds"] < 1
    report = result["report"]
    assert report["verified"] is False
    assert report["owned_processes_gone"] is False
    assert "regular file" in report["close_errors"][0]


@pytest.mark.parametrize("kind", ["directory", "symlink", "oversize"])
def test_nonregular_or_oversize_cleanup_receipt_cannot_qualify(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "close-v7.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        foreign = tmp_path / "foreign.json"
        foreign.write_text("{}")
        path.symlink_to(foreign)
    else:
        with path.open("wb") as stream:
            stream.truncate(storage.MAX_CLEANUP_RECEIPT_BYTES + 1)
    report = storage.verify_native_cleanup(OwnedEnvironment(tmp_path))
    assert report["verified"] is False
    assert report["close_errors"]


@pytest.mark.parametrize("kind", ["missing", "invalid-extra-root", "owned-survivor"])
def test_incomplete_cleanup_is_never_promoted_to_success(tmp_path: Path, kind: str) -> None:
    if kind == "invalid-extra-root":
        cleanup_receipt(tmp_path, retainedLocalDirectory="/foreign/other-run")
    elif kind == "owned-survivor":
        cleanup_receipt(tmp_path, supervisor={"remainingOwned": [123], "cleanupErrors": []})
    report = storage.verify_native_cleanup(OwnedEnvironment(tmp_path))
    assert report["verified"] is False
