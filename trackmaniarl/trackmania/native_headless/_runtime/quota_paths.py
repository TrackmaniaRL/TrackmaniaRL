"""Fresh kernel resolution for ordinary Linux work links, with strict fallback."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Collection
from pathlib import Path


def is_allowed_work_link(
    path: Path, allowed_targets: Collection[Path], *, observed: os.stat_result | None = None
) -> bool:
    """Fast accept a stable symlink whose followed kernel path is exactly allowed.

    An O_PATH descriptor follows all target/parent symlinks without opening file
    contents or blocking on FIFOs. Unfamiliar representations, races and procfs
    failures always retain Path.resolve(strict=True)'s original decision.
    """
    if sys.platform == "linux" and hasattr(os, "O_PATH"):
        descriptor = None
        try:
            before = path.lstat() if observed is None else observed
            if stat.S_ISLNK(before.st_mode):
                descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
                actual = os.readlink(f"/proc/self/fd/{descriptor}")
                if (
                    os.path.isabs(actual)
                    and " (deleted)" not in actual
                    and any(actual == os.fspath(target) for target in allowed_targets)
                ):
                    after = path.lstat()
                    if stat.S_ISLNK(after.st_mode) and (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                    ) == (after.st_dev, after.st_ino, after.st_size):
                        return True
        except OSError:
            # Includes missing/inaccessible procfs, excessive kernel link depth,
            # target deletion and link replacement during this observation.
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return path.resolve(strict=True) in allowed_targets
