"""Windows read-locked copies for experimental map switching.

No lock prevents a game engine from loading unrelated bytes; live UID and engine
behavior require independent attestation. Non-Windows platforms fail closed.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
from ctypes import wintypes
from pathlib import Path
from typing import Any, BinaryIO

from trackmaniarl.distributed.evaluation_assets import evaluation_assets_digest
from trackmaniarl.trackmania.map_switch_controller import _InstanceLock


class StagedAssets:
    """Own a private directory and deny writes/deletes while its files are in use."""

    def __init__(
        self, root: Path, *, max_directories: int = 16, max_bytes: int = 2 * 1024**3
    ) -> None:
        if os.name != "nt":
            raise RuntimeError("read-locked staging currently requires Windows")
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("staging root must be an existing directory")
        if type(max_directories) is not int or max_directories < 1:
            raise ValueError("staging directory limit must be positive")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("staging byte limit must be positive")
        self.max_bytes, self._bytes_written = max_bytes, 0
        self._directory_handles: list[int] = []
        try:
            for ancestor in reversed((root, *root.parents)):
                self._directory_handles.append(_lock_directory(ancestor))
            with _InstanceLock(root / ".trackmaniarl-staging.lock"):
                count = sum(
                    1 for item in root.iterdir() if item.name.startswith("trackmaniarl-map-")
                )
                if count >= max_directories:
                    raise RuntimeError("staging directory limit reached; explicit cleanup required")
                self.directory = Path(tempfile.mkdtemp(prefix="trackmaniarl-map-", dir=root))
            self._directory_handles.append(_lock_directory(self.directory))
        except BaseException as original:
            for handle in reversed(self._directory_handles):
                try:
                    _close_handle(handle)
                except BaseException as error:
                    original.add_note(f"Staging initialization cleanup failed: {error!r}")
            raise
        self._files: list[tuple[Path, BinaryIO]] = []
        self._closed = False

    def stage(self, source: Path, name: str) -> Path:
        if self._closed:
            raise RuntimeError("staging is closed")
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("staged asset name must be a simple filename")
        target = self.directory / name
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_BINARY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as original:
                while chunk := original.read(1024 * 1024):
                    self._bytes_written += len(chunk)
                    if self._bytes_written > self.max_bytes:
                        raise RuntimeError("staging byte limit reached; explicit cleanup required")
                    if output.write(chunk) != len(chunk):
                        raise OSError("short write while staging an asset")
                output.flush()
                os.fsync(output.fileno())
                identity = os.fstat(output.fileno())
            locked = _open_read_only_locked(target)
            try:
                current = os.fstat(locked.fileno())
                if (identity.st_dev, identity.st_ino) != (current.st_dev, current.st_ino):
                    raise RuntimeError("staged asset identity changed before locking")
                if (target.stat().st_dev, target.stat().st_ino) != (current.st_dev, current.st_ino):
                    raise RuntimeError("staged asset path changed before locking")
                self._files.append((target, locked))
            except BaseException:
                locked.close()
                raise
            return target
        except BaseException:
            # Target was created by this stage call, but another process may
            # hold it. Preserve it rather than suppressing the original error.
            raise

    def close(self) -> None:
        """Release locks without deleting paths that another process may still use.

        An interrupted game load can outlive the actor. Retaining these files
        also avoids deleting a substituted path after the lock is released.
        """
        if self._closed:
            return
        errors: list[BaseException] = []
        pending_files: list[tuple[Path, BinaryIO]] = []
        for path, handle in reversed(self._files):
            try:
                handle.close()
            except BaseException as error:
                errors.append(error)
                if not handle.closed:
                    pending_files.append((path, handle))
        pending_directories: list[int] = []
        for directory_handle in reversed(self._directory_handles):
            try:
                _close_handle(directory_handle)
            except BaseException as error:
                errors.append(error)
                pending_directories.append(directory_handle)
        self._files = list(reversed(pending_files))
        self._directory_handles = list(reversed(pending_directories))
        self._closed = not self._files and not self._directory_handles
        if errors:
            first, *remaining = errors
            for additional in remaining:
                first.add_note(f"Additional staging cleanup failure: {additional!r}")
            raise first

    def __enter__(self) -> StagedAssets:
        return self

    def __exit__(self, exc_type: object, error: BaseException | None, traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Staging cleanup also failed: {cleanup_error!r}")

    def stage_suite(self, suite: Any, base_dir: Path, expected_digest: str) -> Any:
        """Lock all suite files, then compare the copies to the assigned digest."""
        maps = []
        for index, item in enumerate(suite.maps):
            paths: dict[str, Path | None] = {}
            for field in ("map_path", "geometry_path", "reward_points_path"):
                original = getattr(item, field)
                if original is None:
                    paths[field] = None
                    continue
                suffix = ".Map.Gbx" if field == "map_path" else ".npz"
                paths[field] = self.stage(base_dir / original, f"{index}-{field}{suffix}")
            maps.append(item.model_copy(update=paths))
        staged = suite.model_copy(update={"maps": tuple(maps)})
        if evaluation_assets_digest(staged, Path(".")) != expected_digest:
            raise RuntimeError("staged evaluation assets differ from the assigned suite")
        return staged


def _open_read_only_locked(path: Path) -> BinaryIO:
    if os.name != "nt":
        raise RuntimeError("read-locked staging currently requires Windows")
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 0x00000001, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "rb")


def _close_handle(handle: int) -> None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    if not kernel.CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _lock_directory(path: Path) -> int:
    """Keep every path component stable and refuse a reparse-point directory."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    # Read attributes; allow readers and writers but deny directory deletion or
    # rename. Open the final component itself instead of following a junction.
    handle = create(str(path), 0x80, 0x3, None, 3, 0x02200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())

    class AttributeTagInfo(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]

    info = AttributeTagInfo()
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        _close_handle(handle)
        raise ctypes.WinError(error)
    if info.attributes & 0x400:
        _close_handle(handle)
        raise RuntimeError("staging path contains a reparse-point directory")
    return int(handle)
