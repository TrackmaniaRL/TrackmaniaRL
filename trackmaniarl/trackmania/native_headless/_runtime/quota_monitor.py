"""Linux event accounting with exact no-follow scans as the fallback.

Directory watches detect structural changes. File-inode watches also observe
writes through outside hardlinks or an already-open renamed descriptor. Every
changed inode is reconciled using fresh metadata for all its owned names.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import stat
import struct
import sys
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from trackmaniarl.trackmania.native_headless._runtime.capture import RxPersistenceError
from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os
from trackmaniarl.trackmania.native_headless._runtime.quota import TraceUsage, trace_usage
from trackmaniarl.trackmania.native_headless._runtime.quota_paths import is_allowed_work_link

IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_UNMOUNT = 0x00002000
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_ONLYDIR = 0x01000000
IN_MASK_ADD = 0x20000000

STRUCTURAL = (
    IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_CREATE
    | IN_DELETE
    | IN_DELETE_SELF
    | IN_MOVE_SELF
    | IN_UNMOUNT
    | IN_Q_OVERFLOW
    | IN_IGNORED
)
CHANGE = IN_MODIFY | IN_ATTRIB | IN_CLOSE_WRITE
MAX_EVENTS = 200_000
MAX_REBUILDS = 4
Inode = tuple[int, int]


def _inode(metadata: os.stat_result) -> Inode:
    return metadata.st_dev, metadata.st_ino


class _UnavailableError(Exception):
    pass


class _RebuildError(Exception):
    pass


class _Inotify:
    def __init__(self) -> None:
        self.fd = -1
        self._stream: BinaryIO | None = None
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            initialize = libc.inotify_init1
            initialize.argtypes, initialize.restype = [ctypes.c_int], ctypes.c_int
            self._add = libc.inotify_add_watch
            self._add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            self._add.restype = ctypes.c_int
            self.fd = initialize(linux_os.O_NONBLOCK | linux_os.O_CLOEXEC)
            if self.fd < 0:
                raise OSError(ctypes.get_errno(), "inotify_init1")
            # FileIO also closes an abandoned monitor's descriptor on object
            # destruction; normal/failure/session paths still close explicitly.
            self._stream = os.fdopen(self.fd, "rb", buffering=0)
        except (OSError, AttributeError) as error:
            self.close()
            raise _UnavailableError(str(error)) from error

    def watch(self, descriptor: int, *, is_directory: bool) -> int:
        mask = CHANGE | STRUCTURAL | IN_MASK_ADD
        if is_directory:
            mask |= IN_ONLYDIR
        result = self._add(self.fd, os.fsencode(f"/proc/self/fd/{descriptor}"), mask)
        if result < 0:
            raise _UnavailableError(str(OSError(ctypes.get_errno(), "inotify_add_watch")))
        return int(result)

    def events(self) -> list[tuple[int, int, str]]:
        try:
            raw = os.read(self.fd, 65536)
        except BlockingIOError:
            return []
        except OSError as error:
            raise _UnavailableError(str(error)) from error
        if not raw:
            raise _UnavailableError("inotify EOF")
        events: list[tuple[int, int, str]] = []
        offset = 0
        while offset < len(raw):
            if len(raw) - offset < 16:
                raise _UnavailableError("truncated inotify header")
            wd, mask, _cookie, size = struct.unpack_from("iIII", raw, offset)
            offset += 16
            if size > len(raw) - offset:
                raise _UnavailableError("truncated inotify name")
            name = os.fsdecode(raw[offset : offset + size].split(b"\0", 1)[0])
            events.append((wd, mask, name))
            offset += size
        return events

    def close(self) -> None:
        descriptor, self.fd = self.fd, -1
        if descriptor >= 0:
            with contextlib.suppress(OSError):
                if self._stream is None:
                    os.close(descriptor)
                else:
                    self._stream.close()


@dataclass(frozen=True, slots=True)
class _File:
    inode: Inode
    size: int


class _Snapshot:
    def __init__(self, backend: _Inotify) -> None:
        self.backend = backend
        self.directories: dict[Path, Inode] = {}
        self.directory_watches: dict[int, Path] = {}
        self.files: dict[Path, _File] = {}
        self.aliases: dict[Inode, set[Path]] = {}
        self.file_watches: dict[int, Inode] = {}
        self.links: dict[Path, _File] = {}
        self.usage = TraceUsage(0, 0)

    def _watch(self, descriptor: int, *, is_directory: bool) -> int:
        try:
            return self.backend.watch(descriptor, is_directory=is_directory)
        except OSError as error:
            raise _UnavailableError(str(error)) from error

    def directory(self, path: Path, descriptor: int | None) -> None:
        if descriptor is None:
            raise _UnavailableError("no no-follow directory descriptor support")
        identity = _inode(os.fstat(descriptor))
        wd = self._watch(descriptor, is_directory=True)
        if wd in self.directory_watches and self.directory_watches[wd] != path:
            raise _UnavailableError("directory inode has multiple root names")
        self.directories[path], self.directory_watches[wd] = identity, path

    def regular(
        self, path: Path, descriptor: int | None, observed: os.stat_result
    ) -> os.stat_result:
        if descriptor is None:
            raise _UnavailableError("no no-follow file descriptor support")
        flags = linux_os.O_PATH | linux_os.O_NOFOLLOW
        try:
            opened = os.open(path.name, flags, dir_fd=descriptor)
        except FileNotFoundError as error:
            raise _RebuildError from error
        try:
            actual = os.fstat(opened)
            if not stat.S_ISREG(actual.st_mode) or _inode(actual) != _inode(observed):
                raise _RebuildError
            identity = _inode(actual)
            if identity not in self.aliases:
                wd = self._watch(opened, is_directory=False)
                self.file_watches[wd] = identity
                self.aliases[identity] = set()
            # Re-read AFTER installing the inode watch: an append between the
            # scanner's initial lstat and watch registration cannot be lost.
            actual = os.fstat(opened)
            self.aliases[identity].add(path)
            self.files[path] = _File(identity, actual.st_size)
            return actual
        finally:
            os.close(opened)

    def symlink(self, path: Path, observed: os.stat_result) -> None:
        self.links[path] = _File(_inode(observed), observed.st_size)


class TraceQuotaMonitor:
    def __init__(  # noqa: PLR0913
        self,
        roots: Iterable[Path | None],
        *,
        work_dir: Path,
        allowed_work_targets: Collection[Path],
        local_posts: Path | None = None,
        control_socket: Path | None = None,
        branch_socket: Path | None = None,
        max_entries: int = 100_000,
    ) -> None:
        self.roots = tuple(root for root in roots if root is not None)
        self.options: dict[str, Any] = {
            "work_dir": work_dir,
            "allowed_work_targets": frozenset(allowed_work_targets),
            "local_posts": local_posts,
            "control_socket": control_socket,
            "branch_socket": branch_socket,
            "max_entries": max_entries,
        }
        self.local_posts = local_posts
        self.mode = "inotify"
        self.fallback_reason: str | None = None
        self.full_scans = 0
        self._closed = False
        self._snapshot: _Snapshot | None = None
        self._root_identities: dict[Path, Inode | None] = {}
        if sys.platform != "linux" or not hasattr(os, "O_PATH"):
            self._fallback("Linux inode watches unavailable")
        elif any(
            a == b or a in b.parents or b in a.parents
            for i, a in enumerate(self.roots)
            for b in self.roots[i + 1 :]
        ):
            self._fallback("overlapping roots require exact scan accounting")

    def _fallback(self, reason: str) -> None:
        if self._snapshot is not None:
            self._snapshot.backend.close()
            self._snapshot = None
        self.mode, self.fallback_reason = "full-scan", reason

    def _scan(self) -> TraceUsage:
        self.full_scans += 1
        return trace_usage(self.roots, **self.options)

    def _rebuild(self) -> None:
        backend = _Inotify()
        snapshot = _Snapshot(backend)
        try:
            self.full_scans += 1
            snapshot.usage = trace_usage(self.roots, observer=snapshot, **self.options)
            identities = {root: snapshot.directories.get(root) for root in self.roots}
        except BaseException:
            backend.close()
            raise
        old, self._snapshot = self._snapshot, snapshot
        self._root_identities = identities
        if old is not None:
            old.backend.close()

    def _roots_changed(self) -> bool:
        for root, expected in self._root_identities.items():
            try:
                actual = root.lstat()
            except FileNotFoundError:
                if expected is not None:
                    return True
            else:
                if not stat.S_ISDIR(actual.st_mode) or _inode(actual) != expected:
                    return True
        return False

    def _links(self, snapshot: _Snapshot) -> None:
        for path, expected in snapshot.links.items():
            observed = path.lstat()
            if not stat.S_ISLNK(observed.st_mode) or _inode(observed) != expected.inode:
                raise _RebuildError
            if not is_allowed_work_link(
                path, self.options["allowed_work_targets"], observed=observed
            ):
                raise RxPersistenceError(f"V7 unexpected trace/work symlink rejected: {path}")

    def _refresh(self, snapshot: _Snapshot, changed: set[Inode]) -> None:
        total, posts = snapshot.usage.used_bytes, snapshot.usage.post_copy_bytes
        flags = os.O_RDONLY | linux_os.O_DIRECTORY | linux_os.O_NOFOLLOW
        for identity in changed:
            aliases = snapshot.aliases[identity]
            size = 0
            for path in aliases:
                try:
                    directory = os.open(path.parent, flags)
                except FileNotFoundError as error:
                    raise _RebuildError from error
                try:
                    if _inode(os.fstat(directory)) != snapshot.directories[path.parent]:
                        raise _RebuildError
                    try:
                        observed = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
                    except FileNotFoundError as error:
                        raise _RebuildError from error
                    if not stat.S_ISREG(observed.st_mode) or _inode(observed) != identity:
                        raise _RebuildError
                    size = max(size, observed.st_size)
                finally:
                    os.close(directory)
            for path in aliases:
                previous = snapshot.files[path]
                total += size - previous.size
                if path.parent == self.local_posts and path.name.endswith(".bin"):
                    posts += size - previous.size
                snapshot.files[path] = _File(identity, size)
        snapshot.usage = TraceUsage(total, posts)

    def _drain(self, snapshot: _Snapshot) -> None:
        count = 0
        while True:
            events = snapshot.backend.events()
            if not events:
                return
            changed: set[Inode] = set()
            for wd, mask, name in events:
                count += 1
                if count > MAX_EVENTS:
                    raise RxPersistenceError("V7 quota event limit exceeded")
                if mask & STRUCTURAL:
                    raise _RebuildError
                if not mask & CHANGE:
                    raise _RebuildError
                if wd in snapshot.file_watches:
                    changed.add(snapshot.file_watches[wd])
                elif wd in snapshot.directory_watches and name:
                    path = snapshot.directory_watches[wd] / name
                    if path not in snapshot.files:
                        raise _RebuildError
                    changed.add(snapshot.files[path].inode)
                else:
                    raise _RebuildError
            self._refresh(snapshot, changed)

    def usage(self, *, force: bool = False) -> TraceUsage:
        if self.mode == "full-scan" or self._closed:
            return self._scan()
        rebuild = force or self._snapshot is None or self._roots_changed()
        try:
            for _attempt in range(MAX_REBUILDS):
                try:
                    if rebuild:
                        self._rebuild()
                    snapshot = self._snapshot
                    assert snapshot is not None
                    self._drain(snapshot)
                    self._links(snapshot)
                    return snapshot.usage
                except _RebuildError:
                    rebuild = True
            raise RxPersistenceError("V7 quota structural rebuild limit exceeded")
        except _UnavailableError as error:
            self._fallback(str(error))
            return self._scan()
        except (OSError, RuntimeError) as error:
            if isinstance(error, RxPersistenceError):
                raise
            raise RxPersistenceError("V7 quota event accounting failed: " + repr(error)) from error

    def close(self) -> None:
        self._closed = True
        self._fallback("monitor closed")
