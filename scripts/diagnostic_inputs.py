"""Explicit caller-owned evidence paths for historical reproduction utilities."""

from __future__ import annotations

import os
from pathlib import Path


def configured_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"configure an explicit existing evidence path: {name}")
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError(f"evidence path must be canonical and absolute: {name}")
    return path
