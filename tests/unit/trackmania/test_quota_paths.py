"""Kernel path fast acceptance retains the strict work-link decision."""

from __future__ import annotations

import errno
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime import quota_paths as qp

LINUX = pytest.mark.skipif(sys.platform != "linux", reason="Linux O_PATH/procfd fast path")
pytestmark = pytest.mark.usefixtures("symlink_support")


def decision(call: Callable[[], bool]) -> tuple[str, object]:
    try:
        return "value", call()
    except (OSError, RuntimeError) as error:
        return "error", type(error)


@pytest.mark.parametrize("kind", ["absolute", "relative", "chain", "long-chain", "directory"])
def test_absolute_relative_and_long_chains_retain_strict_decision(
    tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "asset"
    if kind == "directory":
        target.mkdir()
    else:
        target.write_bytes(b"asset")
    work = tmp_path / "work"
    work.mkdir()
    link = work / "asset"
    destination = target
    if kind in ("chain", "long-chain"):
        for index in range(3 if kind == "chain" else 45):
            following = tmp_path / f"chain-{index}"
            following.symlink_to(destination, target_is_directory=target.is_dir())
            destination = following
    link.symlink_to(
        os.path.relpath(destination, work) if kind == "relative" else destination,
        target_is_directory=target.is_dir(),
    )
    assert qp.is_allowed_work_link(link, (target,)) == (link.resolve(strict=True) == target)
    assert qp.is_allowed_work_link(link, (target,)) is True


@LINUX
def test_hardlink_alias_does_not_become_an_allowed_canonical_name(tmp_path: Path) -> None:
    target, alias, link = (tmp_path / name for name in ("asset", "alias", "work-link"))
    target.write_bytes(b"asset")
    os.link(target, alias)
    link.symlink_to(alias)
    assert qp.is_allowed_work_link(link, (target,)) is False
    assert link.resolve(strict=True) not in (target,)


@LINUX
def test_parent_alias_and_parent_rename_match_strict_resolution(tmp_path: Path) -> None:
    assets, parent_alias = tmp_path / "assets", tmp_path / "parent-alias"
    assets.mkdir()
    target = assets / "asset"
    target.write_bytes(b"asset")
    parent_alias.symlink_to(assets, target_is_directory=True)
    link = tmp_path / "work-link"
    link.symlink_to(parent_alias / "asset")
    assert qp.is_allowed_work_link(link, (target,)) is True
    assets.rename(tmp_path / "original-assets")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "asset").write_bytes(b"foreign")
    assets.symlink_to(foreign, target_is_directory=True)
    assert qp.is_allowed_work_link(link, (target,)) is False
    assert link.resolve(strict=True) not in (target,)


@pytest.mark.parametrize("kind", ["missing", "loop"])
def test_missing_and_looping_links_preserve_original_errors(tmp_path: Path, kind: str) -> None:
    target, link = tmp_path / "asset", tmp_path / "work-link"
    link.symlink_to(target if kind == "missing" else link)
    original = decision(lambda: link.resolve(strict=True) in (target,))
    assert decision(lambda: qp.is_allowed_work_link(link, (target,))) == original


@LINUX
def test_fifo_target_uses_nonblocking_metadata_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, link = tmp_path / "fifo", tmp_path / "work-link"
    os.mkfifo(target)
    link.symlink_to(target)
    original_open = os.open

    def metadata_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        assert flags & os.O_PATH
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", metadata_open)
    assert qp.is_allowed_work_link(link, (target,)) is True
    assert link.resolve(strict=True) == target


@LINUX
@pytest.mark.parametrize("representation", ["permission-error", "unfamiliar", "deleted-name"])
def test_fallback_resolves_strictly_and_always_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, representation: str
) -> None:
    target = tmp_path / ("asset (deleted)" if representation == "deleted-name" else "asset")
    target.write_bytes(b"asset")
    link = tmp_path / "work-link"
    link.symlink_to(target)
    original_readlink = os.readlink
    descriptors: list[int] = []

    def proc_readlink(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path).startswith("/proc/self/fd/"):
            descriptors.append(int(str(path).rsplit("/", 1)[-1]))
            if representation == "permission-error":
                raise PermissionError("injected unavailable procfs")
            if representation == "unfamiliar":
                return "(unreachable)/unfamiliar"
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", proc_readlink)
    assert qp.is_allowed_work_link(link, (target,)) is True
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError, match="Errno") as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


@LINUX
@pytest.mark.parametrize("mutation", ["delete-target", "replace-link"])
def test_deleted_target_or_swapped_link_cannot_fast_accept_stale_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    target, foreign, link = (tmp_path / name for name in ("asset", "foreign", "work-link"))
    target.write_bytes(b"asset")
    foreign.write_bytes(b"foreign")
    link.symlink_to(target)
    original_readlink = os.readlink
    descriptors: list[int] = []

    def mutate_then_read(path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(path).startswith("/proc/self/fd/"):
            descriptors.append(int(str(path).rsplit("/", 1)[-1]))
            if mutation == "delete-target":
                target.unlink()
            else:
                link.unlink()
                link.symlink_to(foreign)
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", mutate_then_read)
    actual = decision(lambda: qp.is_allowed_work_link(link, (target,)))
    assert actual == decision(lambda: link.resolve(strict=True) in (target,))
    assert actual != ("value", True)
    for descriptor in descriptors:
        with pytest.raises(OSError, match="Errno") as caught:
            os.fstat(descriptor)
        assert caught.value.errno == errno.EBADF


def test_non_linux_path_uses_original_strict_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, link = tmp_path / "asset", tmp_path / "work-link"
    target.write_bytes(b"asset")
    link.symlink_to(target)
    monkeypatch.setattr(qp, "sys", SimpleNamespace(platform="other"))

    def no_open(*args: Any, **kwargs: Any) -> int:
        pytest.fail("non-Linux fallback attempted a Linux metadata descriptor")

    monkeypatch.setattr(os, "open", no_open)
    assert qp.is_allowed_work_link(link, (target,)) is True
