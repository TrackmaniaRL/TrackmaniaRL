"""Lock-free, one-way stop shared by independently supervised SDK workers."""

from __future__ import annotations

import mmap
import os
import stat
import sys
import threading
import time
from pathlib import Path

from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os


class MappedStopSignal:
    """One preallocated shared byte; no per-step filesystem lookup or IPC lock.

    Create only in the fresh campaign directory before spawning workers. Attach
    uses O_NOFOLLOW and checks the actual open file, never truncating/resetting
    an old flag. The mapping remains shared across independent process trees.
    """

    _mapping: mmap.mmap
    _closed: bool
    _closed_stop: bool
    _wait_condition: threading.Condition

    def __init__(self, path: Path, *, create: bool = False) -> None:
        if sys.platform != "linux":
            raise RuntimeError("the shared campaign mapping requires Linux")
        if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent:
            raise ValueError("canonical existing campaign parent directory required")
        flags = os.O_RDWR | linux_os.O_NOFOLLOW
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                os.write(fd, b"\0")
                os.fsync(fd)
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != 1
                or metadata.st_uid != linux_os.getuid()
                or metadata.st_nlink != 1
            ):
                raise ValueError("one owned regular single-link stop byte required")
            self._mapping = mmap.mmap(fd, 1, access=mmap.ACCESS_WRITE)
            self._closed = False
            self._closed_stop = False
            self._wait_condition = threading.Condition()
            try:
                self.is_set()
            except BaseException:
                self._mapping.close()
                raise
        finally:
            os.close(fd)

    def is_set(self) -> bool:
        if self._closed:
            raise RuntimeError("campaign stop mapping is closed")
        value = self._mapping[0]
        if value not in (0, 1):
            raise ValueError("campaign stop byte was corrupted")
        return value == 1

    def set(self) -> None:
        if self._closed:
            raise RuntimeError("campaign stop mapping is closed")
        # Visibility in other mappings does not wait for disk IO or flushing.
        self.is_set()
        self._mapping[0] = 1

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        # Synchronize local waiter/unmap lifetime only. Per-step is_set() and
        # cross-process visibility still use the original shared byte.
        with self._wait_condition:
            if self._closed:
                raise RuntimeError("campaign stop mapping is closed")
            while True:
                if self._closed:
                    if self._closed_stop:
                        return True
                    raise RuntimeError("campaign stop mapping closed before stop was set")
                if self.is_set():
                    return True
                remaining = 0.05 if deadline is None else min(0.05, deadline - time.monotonic())
                if remaining <= 0:
                    return False
                self._wait_condition.wait(remaining)

    def close(self) -> None:
        with self._wait_condition:
            if not self._closed:
                # An already waiting shutdown thread may observe this local
                # receipt after unmapping. Closing an unset handle is never
                # converted into a global stop or a successful wait.
                self._closed_stop = self._mapping[0] == 1
                self._mapping.close()
                self._closed = True
                self._wait_condition.notify_all()
