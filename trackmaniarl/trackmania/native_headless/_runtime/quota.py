"""Fresh, bounded session accounting without repeated pathlib metadata calls.

Every scan observes external native append growth. No bytes or directory entries
are cached between calls, and hard-linked names are charged independently.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from trackmaniarl.trackmania.native_headless._runtime.capture import RxPersistenceError
from trackmaniarl.trackmania.native_headless._runtime.quota_paths import is_allowed_work_link


@dataclass(frozen=True, slots=True)
class TraceUsage:
    used_bytes: int
    post_copy_bytes: int


class TraceObserver(Protocol):
    def directory(self, path: Path, descriptor: int | None) -> None: ...

    def regular(
        self, path: Path, descriptor: int | None, observed: os.stat_result
    ) -> os.stat_result: ...

    def symlink(self, path: Path, observed: os.stat_result) -> None: ...


def trace_usage(  # noqa: PLR0913
    roots: Iterable[Path | None],
    *,
    work_dir: Path,
    allowed_work_targets: Collection[Path],
    local_posts: Path | None = None,
    control_socket: Path | None = None,
    branch_socket: Path | None = None,
    max_entries: int = 100_000,
    observer: TraceObserver | None = None,
) -> TraceUsage:
    """Read each entry's metadata once and never traverse a nested symlink.

    Linux directory descriptors bind traversal to the classified inode. The
    portable path branch supports offline tests; native execution requires Linux.
    Explicit roots may be absent during partial initialization or failed close.
    """
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    use_fds = (
        os.scandir in os.supports_fd
        and os.open in os.supports_dir_fd
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
    )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    total = posts = entries = 0

    def visit(  # noqa: PLR0913
        directory: Path, descriptor: int | None, root: Path, depth: int
    ) -> None:
        nonlocal total, posts, entries
        if depth > 128:
            raise RxPersistenceError("V7 trace directory depth limit exceeded")
        if observer is not None:
            observer.directory(directory, descriptor)
        with os.scandir(directory if descriptor is None else descriptor) as children:
            for child in children:
                entries += 1
                if entries > max_entries:
                    raise RxPersistenceError("V7 trace entry limit exceeded")
                observed = child.stat(follow_symlinks=False)
                mode = observed.st_mode
                if stat.S_ISREG(mode):
                    if observer is not None:
                        observed = observer.regular(directory / child.name, descriptor, observed)
                    total += observed.st_size
                    if directory == local_posts and child.name.endswith(".bin"):
                        posts += observed.st_size
                elif stat.S_ISLNK(mode):
                    if root != work_dir or not is_allowed_work_link(
                        directory / child.name, allowed_work_targets, observed=observed
                    ):
                        raise RxPersistenceError(
                            f"V7 unexpected trace/work symlink rejected: {directory / child.name}"
                        )
                    if observer is not None:
                        observer.symlink(directory / child.name, observed)
                    total += observed.st_size
                elif stat.S_ISDIR(mode):
                    nested_path = directory / child.name
                    if descriptor is None:
                        visit(nested_path, None, root, depth + 1)
                        actual = nested_path.lstat()
                    else:
                        nested = os.open(child.name, flags, dir_fd=descriptor)
                        try:
                            actual = os.fstat(nested)
                            if (actual.st_dev, actual.st_ino) != (observed.st_dev, observed.st_ino):
                                raise RxPersistenceError("V7 trace directory changed during scan")
                            visit(nested_path, nested, root, depth + 1)
                        finally:
                            os.close(nested)
                    if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
                        observed.st_dev,
                        observed.st_ino,
                    ):
                        raise RxPersistenceError("V7 trace directory changed during scan")
                elif stat.S_ISSOCK(mode) and directory / child.name in (
                    control_socket,
                    branch_socket,
                ):
                    continue
                else:
                    raise RxPersistenceError(
                        "V7 unexpected non-file trace/work entry rejected: "
                        f"{directory / child.name} mode={mode:#o}"
                    )

    try:
        for root in roots:
            if root is None:
                continue
            entries += 1
            if entries > max_entries:
                raise RxPersistenceError("V7 trace entry limit exceeded")
            try:
                observed = root.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(observed.st_mode):
                raise RxPersistenceError("V7 trace/work root must remain a directory")
            if use_fds:
                descriptor = os.open(root, flags)
                try:
                    actual = os.fstat(descriptor)
                    if (actual.st_dev, actual.st_ino) != (observed.st_dev, observed.st_ino):
                        raise RxPersistenceError("V7 trace root changed during scan")
                    visit(root, descriptor, root, 0)
                finally:
                    os.close(descriptor)
            else:
                visit(root, None, root, 0)
                actual = root.lstat()
                if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
                    observed.st_dev,
                    observed.st_ino,
                ):
                    raise RxPersistenceError("V7 trace root changed during scan")
    except OSError as error:
        if isinstance(error, RxPersistenceError):
            raise
        raise RxPersistenceError("V7 trace accounting failed: " + repr(error)) from error
    return TraceUsage(total, posts)
