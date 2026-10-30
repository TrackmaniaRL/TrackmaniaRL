from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime import immediate
from trackmaniarl.trackmania.native_headless.assets import VerifiedAssets
from trackmaniarl.trackmania.native_headless.config import IMMEDIATE_PROFILE, IMMEDIATE_PROTOCOL


def test_launch_excludes_inherited_runtime_secrets_from_process_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    paths = {name: tmp_path / name for name in ("driver", "tape", "shim")}
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    manifest = {
        "assets": {
            name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()
        },
        "roles": {name: name for name in paths},
    }
    manifest_path = tmp_path / "assets.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verified = VerifiedAssets(
        profile=IMMEDIATE_PROFILE,
        runtime_root=runtime,
        paths=paths,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    settings = SimpleNamespace(
        trace_dir=tmp_path / "trace",
        work_dir=tmp_path / "work",
        max_trace_bytes=1024 * 1024,
        protocol_profile=IMMEDIATE_PROTOCOL,
        runtime_root=runtime,
        asset_manifest=manifest_path,
        deployment_manifest_sha256="synthetic",
        timeout_s=10,
    )

    monkeypatch.setenv("TMRE_API_KEY", "synthetic-private-tmre")
    monkeypatch.setenv("FKSECRET", "synthetic-private-fk")
    monkeypatch.setenv("FKSHIM_READER", "wrong-inherited-value")
    monkeypatch.setenv("LD_PRELOAD", "synthetic-preload")
    monkeypatch.setenv("NORMAL_ENV", "retained")
    monkeypatch.setattr(immediate, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(immediate, "linux_socket", SimpleNamespace(AF_UNIX=1))
    monkeypatch.setattr(immediate.tempfile, "mkdtemp", lambda **_: str(private))
    monkeypatch.setattr(
        immediate,
        "socket",
        SimpleNamespace(
            SOCK_STREAM=1,
            socket=lambda *_: SimpleNamespace(bind=lambda *_: None, listen=lambda *_: None),
        ),
    )
    monkeypatch.setattr(immediate.ImmediateSession, "check_budget", lambda *_: None)
    monkeypatch.setattr(immediate.ImmediateSession, "close", lambda *_, **__: None)

    class StopBeforeNativeLaunchError(Exception):
        pass

    captured: dict[str, Any] = {}

    def stop_before_native_launch(*_: Any, **kwargs: Any) -> None:
        captured.update(kwargs["env"])
        kwargs["stderr"].close()
        raise StopBeforeNativeLaunchError

    monkeypatch.setattr(immediate, "spawn_supervised", stop_before_native_launch)
    with pytest.raises(StopBeforeNativeLaunchError):
        immediate.ImmediateSession(settings, verified)

    record_text = (settings.trace_dir / "launch-v7.json").read_text(encoding="utf-8")
    saved_env = json.loads(record_text)["environment"]
    assert "synthetic-private-tmre" not in record_text
    assert "synthetic-private-fk" not in record_text
    assert "TMRE_API_KEY" not in captured
    assert "FKSECRET" not in captured
    assert "LD_PRELOAD" not in captured
    assert captured["NORMAL_ENV"] == "retained"
    assert captured["FKSHIM_READER"] == "1"
    assert saved_env == {
        key: value for key, value in captured.items() if key.startswith(("FK", "TMRE"))
    }
    assert set(saved_env) == {
        "TMRE_RUNTIME_ROOT",
        "TMRE_TREE_DIR",
        "FKSHIM_POST_DIR",
        "FKSHIM_INNER_SOCKET",
        "FKSHIM_INNER_START",
        "FKSHIM_READER",
        "FKSHIM_RESULT",
        "FK_FRAME_TIMEOUT_MS",
    }
