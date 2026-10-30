"""Archive mechanics only; these fixtures do not qualify native ownership."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trackmaniarl.trackmania.native_headless.full_qr import trace_retention as module


def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, quota: int = 4096
) -> tuple[module.ClosedTraceRetention, dict[str, Path]]:
    settings = module.TraceRetentionSettings(
        tmp_path / "traces",
        tmp_path / "work",
        tmp_path / "archives",
        keep_archives=1,
        max_source_bytes=quota,
    )
    retention = module.ClosedTraceRetention(settings)
    roots = {}
    for prefix in ("trace", "work", "temporary"):
        root = tmp_path / prefix
        root.mkdir()
        (root / "literal.bin").write_bytes(prefix.encode())
        roots[prefix] = root
    # Ownership/closure has separate real-driver tests. Isolate only archive IO.
    monkeypatch.setattr(retention, "_closed_roots", lambda *_: roots)
    return retention, roots


def test_source_quota_refuses_before_archive_or_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retention, roots = fixture(tmp_path, monkeypatch, quota=1)
    with pytest.raises(ValueError, match="input quota"):
        retention.archive(None, [], reason="fixture")
    assert all(root.exists() for root in roots.values())
    assert not retention.settings.archive_root.exists()


def test_archive_readback_failure_keeps_all_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retention, roots = fixture(tmp_path, monkeypatch)
    original = module._digest
    calls = {}

    def changed(path: Path) -> str:
        calls[path] = calls.get(path, 0) + 1
        return original(path) if calls[path] == 1 else "0" * 64

    monkeypatch.setattr(module, "_digest", changed)
    with pytest.raises(ValueError, match="changed during verification"):
        retention.archive(None, [], reason="fixture")
    assert all((root / "literal.bin").exists() for root in roots.values())
    assert not list(retention.settings.archive_root.glob("*.tar.gz"))


def test_verified_archive_prunes_only_owned_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retention, roots = fixture(tmp_path, monkeypatch)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_bytes(b"preserve")
    result = retention.archive(None, [], reason="fixture")
    assert result["sources_pruned"]
    assert all(not root.exists() for root in roots.values())
    assert (unrelated / "keep").read_bytes() == b"preserve"
    assert module._digest(Path(result["archive"])) == result["archive_sha256"]
    receipt = retention.settings.archive_root / "session-00000001.json"
    assert json.loads(receipt.read_text())["sources_pruned"]


def test_live_session_refused_without_archive_creation(tmp_path: Path) -> None:
    retention = module.ClosedTraceRetention(
        module.TraceRetentionSettings(tmp_path / "traces", tmp_path / "work", tmp_path / "archives")
    )
    session = SimpleNamespace(
        closed=False,
        out=tmp_path / "trace",
        work=tmp_path / "work",
        tmp=tmp_path / "temporary",
    )
    with pytest.raises(ValueError, match="fully closed"):
        retention.archive(session, [], reason="must_refuse")
    assert not retention.settings.archive_root.exists()
