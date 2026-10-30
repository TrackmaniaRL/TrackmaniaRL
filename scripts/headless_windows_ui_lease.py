"""Hold the existing comparison mutex during bounded Openplanet diagnostics.

No game input is issued here. The UI client must stop acting before the lease
expires or when its STOP/release file appears.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from scripts.headless_input_profile import existing_controller_lease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--controller-lease", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise ValueError("the Windows comparison mutex is required")
    import psutil

    args.output.mkdir(exist_ok=False)
    own = psutil.Process()
    allowed = {own.pid}
    parent = own.parent()
    if parent is not None and Path(parent.exe()) == Path(sys.executable):
        allowed.add(parent.pid)
    foreign = [
        p.pid
        for p in psutil.process_iter(["name"])
        if p.pid not in allowed and (p.info["name"] or "").lower() in ("python.exe", "pythonw.exe")
    ]
    if foreign:
        raise RuntimeError("another Python actor is present: " + str(foreign))
    receipt = {
        "pid": os.getpid(),
        "birth": own.create_time(),
        "active": False,
        "expires_unix_ms": int((time.time() + 240) * 1000),
        "mutex": "Global\\TrackmaniaRL.ComparisonController",
        "game_input_issued": False,
    }
    with existing_controller_lease(args.controller_lease):
        receipt["active"] = True
        path = args.output / "lease.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        print(json.dumps(receipt), flush=True)
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if (args.output / "STOP").exists() or (args.output / "RELEASE").exists():
                    break
                time.sleep(0.1)
        finally:
            receipt["active"] = False
            path.write_text(json.dumps(receipt), encoding="utf-8")


if __name__ == "__main__":
    main()
