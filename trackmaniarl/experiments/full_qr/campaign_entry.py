"""Process entry for the independently supervised full QR campaign workers.

The parent supplies a pinned private request; existing worker qualification
guards run before an SDK worker creates outputs or resolves a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

from trackmaniarl.experiments.full_qr.campaign_worker import (
    WorkerSettings,
    run_actor_worker,
    run_learner_worker,
)
from trackmaniarl.trackmania.native_headless._runtime.posix_api import linux_os


def load_worker_request(path: Path, sha256: str) -> WorkerSettings:
    if sys.platform != "linux":
        raise RuntimeError("full headless campaign workers require Linux")
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or str(path).startswith("/mnt/")
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise ValueError("canonical pinned private ext4 worker request required")
    descriptor = os.open(path, os.O_RDONLY | linux_os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != linux_os.getuid()
            or before.st_mode & 0o077
            or not 0 < before.st_size <= 16384
        ):
            raise ValueError("small owned private worker request required")
        raw = linux_os.pread(descriptor, before.st_size, 0)
        after = os.fstat(descriptor)
        if (
            len(raw) != before.st_size
            or (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns)
            or hashlib.sha256(raw).hexdigest() != sha256
        ):
            raise ValueError("immutable worker request changed")
    finally:
        os.close(descriptor)
    request = json.loads(raw)
    expected = {field.name for field in fields(WorkerSettings)}
    if (
        type(request) is not dict
        or set(request) != expected
        or any(type(value) is not str or not value for value in request.values())
    ):
        raise ValueError("exact explicitly typed worker settings fields required")
    settings = WorkerSettings(
        directory=Path(request["directory"]),
        config_path=Path(request["config_path"]),
        config_sha256=request["config_sha256"],
        token_path=Path(request["token_path"]),
        stop_path=Path(request["stop_path"]),
        endpoint=request["endpoint"],
        worker_name=request["worker_name"],
        artifact_root=Path(request["artifact_root"]),
    )
    if settings.directory != path.parent:
        raise ValueError("worker request must belong to its campaign directory")
    return settings


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("learner", "actor"))
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    arguments = parser.parse_args(argv)
    settings = load_worker_request(arguments.settings, arguments.sha256)
    if arguments.role == "learner":
        run_learner_worker(settings)
    else:
        run_actor_worker(settings)


if __name__ == "__main__":
    main()
