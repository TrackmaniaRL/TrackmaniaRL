"""Fresh preflight attempt after the user restored foreground readiness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import psutil


def main() -> None:
    base = Path("H:/Studia/inzynierskie/inzynierkav2/AITrackmania/artifacts/tmrl-test-comparison")
    previous = base / "ppo-trend-continuation-20261013"
    proof = json.loads((previous / "post-closure-verification.json").read_text())
    authorization = json.loads((previous / "authorization.json").read_text())
    if not proof["all_owned_closed"] or not proof["mutex_free"]:
        raise RuntimeError("Previous preflight attempt is not closed")
    for identity in proof["owned_identities"]:
        try:
            process = psutil.Process(identity["pid"])
            if abs(process.create_time() - identity["create_time"]) < .001:
                raise RuntimeError("Previous owned process remains alive")
        except psutil.NoSuchProcess:
            pass
    stop = previous / "STOP"
    if (stop.read_text() != "supervisor_failure\n"
            or stop.stat().st_mtime > datetime.fromisoformat(proof["checked_at"]).timestamp()):
        raise RuntimeError("Previous STOP changed after closure")
    helper = Path(__file__).with_name("continue_ppo_trend.py")
    if hashlib.sha256(helper.read_bytes()).hexdigest() != authorization["supervisor_helper_sha256"]:
        raise RuntimeError("Original continuation helper changed")
    spec = importlib.util.spec_from_file_location("_authorized_ppo_continuation", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError("Continuation supervisor unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.QUEUE = base / "ppo-trend-continuation-20261013-r2"
    module.main()


if __name__ == "__main__":
    main()
