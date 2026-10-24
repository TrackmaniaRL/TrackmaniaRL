"""Real Linux file events, alias freshness, bounded rebuilds and exact fallback."""

from __future__ import annotations

import errno
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime import quota_monitor as qm
from trackmaniarl.trackmania.native_headless._runtime.capture import RxPersistenceError

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux inode notifications")


@pytest.fixture
def create_monitor(tmp_path: Path) -> Iterator[Callable[..., qm.TraceQuotaMonitor]]:
    roots = tuple(tmp_path / name for name in ("trace", "work", "private"))
    for root in roots:
        root.mkdir()
    monitors: list[qm.TraceQuotaMonitor] = []

    def create(**overrides: Any) -> qm.TraceQuotaMonitor:
        options = {"work_dir": roots[1], "allowed_work_targets": (), **overrides}
        monitor = qm.TraceQuotaMonitor(roots, **options)
        monitors.append(monitor)
        return monitor

    yield create
    for monitor in monitors:
        monitor.close()


def test_append_through_outside_hardlink_updates_every_owned_name(
    tmp_path: Path, create_monitor: Callable[..., qm.TraceQuotaMonitor]
) -> None:
    source = tmp_path / "trace" / "rx.bin"
    source.write_bytes(b"abc")
    os.link(source, tmp_path / "work" / "alias.bin")
    outside = tmp_path / "outside.bin"
    os.link(source, outside)
    monitor = create_monitor()
    assert monitor.usage().used_bytes == 6
    assert monitor.mode == "inotify"
    scans = monitor.full_scans
    with outside.open("ab") as stream:
        stream.write(b"def")
    assert monitor.usage().used_bytes == 12
    assert monitor.full_scans == scans
    outside.write_bytes(b"x")
    assert monitor.usage().used_bytes == 2
    assert monitor.full_scans == scans


def test_preexisting_open_descriptor_stays_fresh_after_rename(
    tmp_path: Path, create_monitor: Callable[..., qm.TraceQuotaMonitor]
) -> None:
    source, renamed = tmp_path / "trace" / "old.bin", tmp_path / "work" / "new.bin"
    source.write_bytes(b"abc")
    descriptor = os.open(source, os.O_WRONLY | os.O_APPEND)
    monitor = create_monitor()
    try:
        assert monitor.usage().used_bytes == 3
        source.rename(renamed)
        os.write(descriptor, b"de")
        assert monitor.usage().used_bytes == 5
        scans = monitor.full_scans
        os.write(descriptor, b"fg")
        assert monitor.usage().used_bytes == 7
        assert monitor.full_scans == scans
    finally:
        os.close(descriptor)


def test_unlinked_name_is_free_but_held_descriptor_still_updates_remaining_alias(
    tmp_path: Path, create_monitor: Callable[..., qm.TraceQuotaMonitor]
) -> None:
    source, alias = tmp_path / "trace" / "old.bin", tmp_path / "work" / "live.bin"
    source.write_bytes(b"abc")
    os.link(source, alias)
    descriptor = os.open(source, os.O_WRONLY | os.O_APPEND)
    monitor = create_monitor()
    try:
        assert monitor.usage().used_bytes == 6
        source.unlink()
        os.write(descriptor, b"de")
        assert monitor.usage().used_bytes == 5
        os.write(descriptor, b"fg")
        assert monitor.usage().used_bytes == 7
    finally:
        os.close(descriptor)


def test_append_between_initial_stat_and_watch_is_counted_after_watch(
    tmp_path: Path,
    create_monitor: Callable[..., qm.TraceQuotaMonitor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trace" / "rx.bin"
    source.write_bytes(b"abc")
    original = qm._Inotify.watch
    appended = False

    def watch(self: Any, descriptor: int, *, is_directory: bool) -> int:
        nonlocal appended
        if not is_directory and not appended:
            with source.open("ab") as stream:
                stream.write(b"before-watch")
            appended = True
        return original(self, descriptor, is_directory=is_directory)

    monkeypatch.setattr(qm._Inotify, "watch", watch)
    monitor = create_monitor()
    assert monitor.usage().used_bytes == len(b"abcbefore-watch")
    assert appended


@pytest.mark.parametrize("event", [qm.IN_Q_OVERFLOW, qm.IN_IGNORED, qm.IN_UNMOUNT])
def test_lost_or_invalid_watch_events_force_complete_rebuild(  # noqa: PLR0913
    tmp_path: Path,
    create_monitor: Callable[..., qm.TraceQuotaMonitor],
    monkeypatch: pytest.MonkeyPatch,
    event: int,
) -> None:
    source = tmp_path / "trace" / "rx.bin"
    source.write_bytes(b"abc")
    monitor = create_monitor()
    assert monitor.usage().used_bytes == 3
    snapshot = monitor._snapshot
    assert snapshot is not None
    old = snapshot.backend
    source.write_bytes(b"changed")
    monkeypatch.setattr(old, "events", lambda: [(-1, event, "")])
    assert monitor.usage().used_bytes == 7
    assert monitor.full_scans == 2
    assert old.fd == -1


def test_forced_reconciliation_replaces_and_closes_old_watch_descriptor(
    tmp_path: Path, create_monitor: Callable[..., qm.TraceQuotaMonitor]
) -> None:
    (tmp_path / "trace" / "rx.bin").write_bytes(b"abc")
    monitor = create_monitor()
    assert monitor.usage().used_bytes == 3
    old = monitor._snapshot
    assert old is not None
    assert monitor.usage(force=True).used_bytes == 3
    assert old.backend.fd == -1
    assert monitor.full_scans == 2
    current = monitor._snapshot
    assert current is not None
    monitor.close()
    assert current.backend.fd == -1
    assert monitor.usage().used_bytes == 3
    assert monitor.mode == "full-scan"


def test_watch_limit_failure_closes_partial_backend_and_falls_back_to_fresh_scan(
    tmp_path: Path,
    create_monitor: Callable[..., qm.TraceQuotaMonitor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trace" / "rx.bin"
    source.write_bytes(b"abc")
    original = qm._Inotify.watch
    backends: list[Any] = []

    def limit(self: Any, descriptor: int, *, is_directory: bool) -> int:
        if not is_directory:
            backends.append(self)
            raise OSError(errno.ENOSPC, "injected watch limit")
        return original(self, descriptor, is_directory=is_directory)

    monkeypatch.setattr(qm._Inotify, "watch", limit)
    monitor = create_monitor()
    assert monitor.usage().used_bytes == 3
    assert monitor.mode == "full-scan"
    assert "watch limit" in str(monitor.fallback_reason)
    assert backends
    assert all(backend.fd == -1 for backend in backends)
    source.write_bytes(b"external-grow")
    assert monitor.usage().used_bytes == len(b"external-grow")


def test_unavailable_inotify_still_accounts_external_growth_exactly(
    tmp_path: Path,
    create_monitor: Callable[..., qm.TraceQuotaMonitor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "trace" / "rx.bin"
    source.write_bytes(b"abc")

    def unavailable() -> None:
        raise qm._UnavailableError("injected unsupported notification API")

    monkeypatch.setattr(qm, "_Inotify", unavailable)
    monitor = create_monitor()
    assert monitor.usage().used_bytes == 3
    assert monitor.mode == "full-scan"
    source.write_bytes(b"changed")
    assert monitor.usage().used_bytes == 7


def test_event_limit_fails_closed_instead_of_spinning(
    tmp_path: Path,
    create_monitor: Callable[..., qm.TraceQuotaMonitor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "trace" / "rx.bin").write_bytes(b"abc")
    monitor = create_monitor()
    monitor.usage()
    snapshot = monitor._snapshot
    assert snapshot is not None
    wd = next(iter(snapshot.file_watches))
    monkeypatch.setattr(qm, "MAX_EVENTS", 3)
    monkeypatch.setattr(snapshot.backend, "events", lambda: [(wd, qm.IN_MODIFY, "")] * 4)
    with pytest.raises(RxPersistenceError, match="event limit"):
        monitor.usage()


def test_structural_churn_has_a_finite_rebuild_limit(
    create_monitor: Callable[..., qm.TraceQuotaMonitor], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qm._Inotify, "events", lambda self: [(-1, qm.IN_Q_OVERFLOW, "")])
    monitor = create_monitor()
    with pytest.raises(RxPersistenceError, match="rebuild limit"):
        monitor.usage()
    assert monitor.full_scans == qm.MAX_REBUILDS
    snapshot = monitor._snapshot
    assert snapshot is not None
    monitor.close()
    assert snapshot.backend.fd == -1


def test_allowed_link_parent_change_is_detected_without_a_root_event(
    tmp_path: Path, create_monitor: Callable[..., qm.TraceQuotaMonitor]
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    target = assets / "server"
    target.write_bytes(b"original")
    link = tmp_path / "work" / "server"
    link.symlink_to(target)
    monitor = create_monitor(allowed_work_targets=(target,))
    assert monitor.usage().used_bytes == link.lstat().st_size
    assets.rename(tmp_path / "original-assets")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "server").write_bytes(b"foreign")
    assets.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(RxPersistenceError, match="symlink"):
        monitor.usage()


def test_new_unknown_symlink_rebuild_fails_closed_and_allows_explicit_close(
    tmp_path: Path, create_monitor: Callable[..., qm.TraceQuotaMonitor]
) -> None:
    monitor = create_monitor()
    assert monitor.usage().used_bytes == 0
    old = monitor._snapshot
    assert old is not None
    (tmp_path / "trace" / "foreign").symlink_to(tmp_path)
    with pytest.raises(RxPersistenceError, match="symlink"):
        monitor.usage()
    monitor.close()
    assert old.backend.fd == -1
