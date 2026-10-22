"""One game controller across checkouts: Windows named mutex or Linux user flock."""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def controller_lease() -> Iterator[None]:
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.CreateMutexW(None, False, "Global\\TrackmaniaRL.ComparisonController")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = kernel.WaitForSingleObject(handle, 0) in {0, 0x80}
        try:
            if not acquired:
                raise RuntimeError("Another comparison controller is active")
            yield
        finally:
            if acquired:
                kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
    elif sys.platform == "linux":
        import fcntl

        path = Path(tempfile.gettempdir()) / f"trackmaniarl-comparison-{os.getuid()}.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another comparison controller is active") from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        # Do not unlink the inode: another process could already be waiting on it.
    else:
        raise RuntimeError("This campaign supports Windows or Linux X11 only")
