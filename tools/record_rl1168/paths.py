"""Explicit campaign storage configuration and disk reserves."""

import os
import shutil
from pathlib import Path


def private_base():
    raw = os.environ.get("TMRL_CAMPAIGN_PRIVATE_BASE")
    if not raw:
        raise ValueError("TMRL_CAMPAIGN_PRIVATE_BASE must name an existing absolute directory")
    path = Path(raw)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("TMRL_CAMPAIGN_PRIVATE_BASE must name an existing absolute directory")
    return path.resolve(strict=True)


def check_disk_reserves(report_path, private_path):
    for path in (Path(report_path), Path(private_path)):
        if shutil.disk_usage(path).free < 4 * 2**30:
            raise RuntimeError("disk reserve " + str(path))
