"""Fresh quota scans preserve externally written and retained session evidence."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime.capture import Capture, RxPersistenceError
from trackmaniarl.trackmania.native_headless._runtime.immediate import ImmediateSession
from trackmaniarl.trackmania.native_headless._runtime.quota import trace_usage

LINUX = pytest.mark.skipif(sys.platform != "linux", reason="native no-follow filesystem I/O")


def session_tree(tmp_path: Path) -> ImmediateSession:
    session = ImmediateSession.__new__(ImmediateSession)
    session.out, session.work, session.tmp = (tmp_path / part for part in ("trace", "work", "tmp"))
    for directory in (session.out, session.work, session.tmp):
        directory.mkdir()
    session.local_posts = session.tmp / "posts"
    session.local_posts.mkdir()
    session.address = session.tmp / "control.sock"
    session.paths, session.captures = {}, []
    session.max_trace_bytes = 100
    return session


@LINUX
def test_each_scan_counts_external_append_new_file_truncate_and_delete(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    post = session.local_posts / "post-123.bin"
    post.write_bytes(b"abc")
    (session.work / "stdout.bin").write_bytes(b"12")
    assert session.quota_status()["used_bytes"] == 5
    with post.open("ab") as stream:
        stream.write(b"12345")
    journal = session.out / "journal.jsonl"
    journal.write_bytes(b"1234")
    assert session.quota_status()["used_bytes"] == 14
    session.max_trace_bytes = 14
    session.check_budget()
    with pytest.raises(RxPersistenceError, match="trace budget"):
        session.check_budget(1)
    post.write_bytes(b"a")
    journal.unlink()
    assert session.quota_status()["used_bytes"] == 3
    session.check_budget(11)


@LINUX
def test_status_reserves_post_copies_and_only_unpublished_rx(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    (session.local_posts / "post-123.bin").write_bytes(b"post")
    (session.local_posts / "result-123.bin").write_bytes(b"result")
    (session.local_posts / "other.txt").write_bytes(b"aux")
    nested = session.local_posts / "nested"
    nested.mkdir()
    (nested / "not-copied.bin").write_bytes(b"nested")
    pending, published = Capture.__new__(Capture), Capture.__new__(Capture)
    pending.persisted, pending.published, pending.provenance = 7, None, {}
    pending.local, pending.ep = session.tmp / "rx.bin", session.out
    published.persisted, published.published = 100, {"bytes": 100}
    session.captures = [pending, published]
    status = session.quota_status()
    assert status["used_bytes"] == 19
    assert status["max_bytes"] == 100
    assert status["close_duplicate_reserve_bytes"] == 24
    assert status["close_metadata_reserve_bytes"] > 64 * 1024 + 4096
    assert status["close_reserve_bytes"] == 24 + status["close_metadata_reserve_bytes"]


@LINUX
def test_only_pinned_work_symlinks_are_counted_without_traversal(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    session.max_trace_bytes = 10_000
    assets = tmp_path / "assets"
    assets.mkdir()
    server, map_path = assets / "server", assets / "map"
    server.write_bytes(b"private binary is not charged through symlink")
    map_path.write_bytes(b"map")
    packs = assets / "Packs"
    packs.mkdir()
    (packs / "large.pak").write_bytes(b"large asset is not traversed")
    session.paths = {"server": server, "map": map_path}
    links = [session.work / name for name in ("server", "map", "Packs")]
    for link, target in zip(links, (server, map_path, packs), strict=True):
        link.symlink_to(target, target_is_directory=target.is_dir())
    assert session.quota_status()["used_bytes"] == sum(link.lstat().st_size for link in links)
    foreign_link = session.out / "foreign-server"
    foreign_link.symlink_to(server)
    with pytest.raises(RxPersistenceError, match="symlink"):
        session.check_budget()
    foreign_link.unlink()
    session.check_budget()
    links[0].unlink()
    links[0].symlink_to(tmp_path / "absent")
    with pytest.raises(RxPersistenceError, match="accounting failed"):
        session.check_budget()


@LINUX
def test_allowed_target_parent_alias_is_rejected_on_the_next_scan(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    session.max_trace_bytes = 10_000
    assets = tmp_path / "assets"
    assets.mkdir()
    server = assets / "server"
    server.write_bytes(b"original")
    session.paths = {"server": server, "map": assets / "map"}
    (session.work / "server").symlink_to(server)
    session.check_budget()
    assets.rename(tmp_path / "original-assets")
    foreign = tmp_path / "foreign-assets"
    foreign.mkdir()
    (foreign / "server").write_bytes(b"foreign")
    assets.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(RxPersistenceError, match="symlink"):
        session.check_budget()
    assert (foreign / "server").read_bytes() == b"foreign"


@LINUX
def test_symlink_swapped_for_classified_directory_never_traverses_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = session_tree(tmp_path)
    nested = session.out / "nested"
    nested.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "unowned.bin").write_bytes(b"not owned")
    original_open = os.open
    swapped = False

    def replace_before_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        if path == "nested" and kwargs.get("dir_fd") is not None:
            swapped = True
            nested.rmdir()
            nested.symlink_to(foreign, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    # Keep capability detection tied to the original supported function.
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, replace_before_open})
    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises(RxPersistenceError, match="accounting failed"):
        session.check_budget()
    assert swapped
    assert (foreign / "unowned.bin").read_bytes() == b"not owned"


@LINUX
@pytest.mark.parametrize("in_socket_path", [False, True])
def test_root_symlink_and_unknown_fifo_are_rejected_without_opening(
    tmp_path: Path, *, in_socket_path: bool
) -> None:
    session = session_tree(tmp_path)
    fifo = session.tmp / "branch.sock" if in_socket_path else session.work / "unexpected.fifo"
    os.mkfifo(fifo)
    with pytest.raises(RxPersistenceError, match="non-file") as caught:
        session.check_budget()
    assert str(fifo) in str(caught.value)
    fifo.unlink()
    session.out.rmdir()
    session.out.symlink_to(session.work, target_is_directory=True)
    with pytest.raises(RxPersistenceError, match="root must remain"):
        session.check_budget()


@LINUX
def test_owned_socket_is_ignored_but_unexpected_socket_is_rejected(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    # Use directory descriptors to keep the bound address below sun_path's limit.
    directory = os.open(session.tmp, os.O_RDONLY | os.O_DIRECTORY)
    owned, branch, foreign = (socket.socket(socket.AF_UNIX) for _ in range(3))
    try:
        owned.bind(f"/proc/self/fd/{directory}/control.sock")
        branch.bind(f"/proc/self/fd/{directory}/branch.sock")
        session.check_budget()
        foreign.bind(f"/proc/self/fd/{directory}/foreign.sock")
        with pytest.raises(RxPersistenceError, match="non-file") as caught:
            session.check_budget()
        assert str(session.tmp / "foreign.sock") in str(caught.value)
    finally:
        owned.close()
        branch.close()
        foreign.close()
        os.close(directory)


@LINUX
@pytest.mark.parametrize("outside", ["trace", "work"])
def test_branch_socket_name_is_not_allowed_outside_the_private_session(
    tmp_path: Path, outside: str
) -> None:
    session = session_tree(tmp_path)
    root = session.out if outside == "trace" else session.work
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    foreign = socket.socket(socket.AF_UNIX)
    try:
        foreign.bind(f"/proc/self/fd/{directory}/branch.sock")
        with pytest.raises(RxPersistenceError, match="non-file") as caught:
            session.check_budget()
        assert str(root / "branch.sock") in str(caught.value)
    finally:
        foreign.close()
        os.close(directory)


@LINUX
def test_hard_link_names_and_retained_tmp_are_charged_individually(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    local = session.tmp / "rx.bin"
    local.write_bytes(b"12345")
    os.link(local, session.out / "rx.bin")
    assert session.quota_status()["used_bytes"] == 10
    with local.open("ab") as stream:
        stream.write(b"678")
    assert session.quota_status()["used_bytes"] == 16
    (session.out / "rx.bin").unlink()
    assert session.quota_status()["used_bytes"] == 8


def test_bounded_scan_rejects_entry_explosion(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    for index in range(4):
        (session.out / str(index)).touch()
    with pytest.raises(RxPersistenceError, match="entry limit"):
        trace_usage((session.out,), work_dir=session.work, allowed_work_targets=(), max_entries=4)


@pytest.mark.parametrize("additional", [-1, True, 0.5])
def test_invalid_reservation_cannot_reduce_measured_total(tmp_path: Path, additional: Any) -> None:
    session = session_tree(tmp_path)
    with pytest.raises(RxPersistenceError, match="nonnegative integer"):
        session.check_budget(additional)


@LINUX
def test_refused_publication_leaves_no_quota_debt_or_new_file(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    retained = session.tmp / "retained.bin"
    retained.write_bytes(b"12345")
    session.max_trace_bytes = 7
    destination = session.out / "new.bin"
    with pytest.raises(RxPersistenceError, match="trace budget"):
        session._publish_bytes(destination, b"123")
    assert not destination.exists()
    assert session.quota_status()["used_bytes"] == 5
    session._publish_bytes(destination, b"12")
    assert session.quota_status()["used_bytes"] == 7
    assert retained.read_bytes() == b"12345"


@LINUX
def test_close_without_captures_still_reports_externally_grown_overrun(tmp_path: Path) -> None:
    session = session_tree(tmp_path)
    session.max_trace_bytes = 5
    stderr = session.out / "driver.stderr.bin"
    stderr.write_bytes(b"abc")
    assert session.quota_status()["used_bytes"] == 3
    monitor = session._quota_monitor
    assert monitor is not None
    stderr.write_bytes(b"external-overrun")
    session.closed = False
    session.connection = session.listener = session.stderr = None
    session.commands, session.messages = [], []
    session.proc = SimpleNamespace(
        stop=lambda **_: {"remainingOwned": [], "cleanupErrors": [], "returncode": 1},
        close_control=lambda: None,
    )
    session.close(error=True)
    report = json.loads((session.out / "close-v7.json").read_bytes())
    assert report["supervisor"]["remainingOwned"] == []
    assert any("trace budget exceeded" in failure for failure in report["failures"])
    assert report["traceAccounting"]["mode"] == "inotify"
    assert report["traceAccounting"]["fallback_reason"] is None
    assert report["traceAccounting"]["full_scans"] >= 2
    assert monitor._snapshot is None
    assert monitor.mode == "full-scan"


@LINUX
def test_close_publication_failure_cannot_leak_notification_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania.native_headless._runtime import immediate

    session = session_tree(tmp_path)
    session.quota_status()
    monitor = session._quota_monitor
    assert monitor is not None
    session.closed = False
    session.connection = session.listener = session.stderr = None
    session.commands, session.messages = [], []
    session.proc = SimpleNamespace(
        stop=lambda **_: {"remainingOwned": [], "cleanupErrors": [], "returncode": 1},
        close_control=lambda: None,
    )

    def fail_publication(*args: Any) -> None:
        raise OSError("injected close receipt failure")

    monkeypatch.setattr(immediate, "json_exclusive", fail_publication)
    with pytest.raises(OSError, match="close receipt failure"):
        session.close(error=True)
    assert monitor._snapshot is None
    assert monitor.mode == "full-scan"
