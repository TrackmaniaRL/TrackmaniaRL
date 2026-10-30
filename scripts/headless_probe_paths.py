"""Explicit local inputs for opt-in Windows diagnostics; no import-time host lookup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DiagnosticProbePaths:
    comparison_root: Path
    config: Path

    @classmethod
    def from_environment(cls) -> DiagnosticProbePaths:
        """Resolve only deliberately configured probe inputs before any controller I/O."""
        values = []
        for key in ("TRACKMANIARL_COMPARISON_ROOT", "TRACKMANIARL_PROBE_CONFIG"):
            configured = os.environ.get(key)
            if not configured:
                raise ValueError(f"explicit diagnostic input required: {key}")
            path = Path(configured)
            if not path.is_absolute() or path.resolve(strict=True) != path:
                raise ValueError(f"canonical absolute diagnostic input required: {key}")
            values.append(path)
        if not values[0].is_dir() or not values[1].is_file():
            raise ValueError("the explicit comparison root and RunSpec file must exist")
        return cls(values[0], values[1])
