"""Export every real neutral POST for a bounded observation-clock diagnostic."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from scripts.diagnostic_inputs import configured_path
from scripts.headless_immediate_probe import ImmediateSession
from trackmaniarl.trackmania.native_headless._runtime import env, features


def main() -> None:
    output = Path(sys.argv[1])
    if sys.platform != "linux" or not output.is_absolute() or str(output).startswith("/mnt/"):
        raise ValueError("fresh Linux/ext4 output required")
    output.mkdir(parents=True, exist_ok=False)
    config_path = Path(str(configured_path("TRACKMANIARL_NATIVE_CONFIG")))
    receipt_path = Path(str(configured_path("TRACKMANIARL_NATIVE_RECEIPT")))
    config = json.loads(config_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    deployment = Path(config["deployment_manifest"]).read_bytes()
    if hashlib.sha256(deployment).hexdigest() != config["deployment_manifest_sha256"]:
        raise ValueError("original deployment manifest differs")
    driver = json.loads(deployment)["driverBuildApproval"]["driverSha256"]
    report: dict[str, Any] = {
        "schema": "tmrl.neutral-native-clock-posts/v1",
        "completed": False,
        "training": False,
        "clock_alignment_claimed": False,
        "input_acknowledgement_claimed": False,
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (config_path, receipt_path)
        },
        "rows": [],
        "steps": [],
    }
    begin = time.monotonic()
    session = None

    def retain(raw: bytes) -> None:
        assert session is not None
        state, clock = env.qualify_post(raw, session.pid, session.meta)
        rows = report["rows"]
        if rows and clock != rows[-1]["native_t"] + 10:
            raise ValueError("actual POST sequence has a missing or repeated native interval")
        rows.append({"native_t": clock, "state": features.native16(state), "postHex": raw.hex()})

    try:
        session = ImmediateSession(
            Path(config["runtime_root"]), receipt, output / output.name, 2190, driver
        )
        session.reset()
        retain(session.boundary)
        for _ in range(161):  # All actual boundaries2190..10240, not host timing.
            if (output / "STOP").exists() or time.monotonic() - begin > 120:
                raise InterruptedError("neutral clock capture STOP/deadline")
            evidence = session.step(36, 5)
            report["steps"].append(evidence)
            for raw_hex in evidence["appliedPostHex"]:
                retain(bytes.fromhex(raw_hex))
        report["completed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if session is not None:
                session.close(error=not report["completed"])
        except BaseException as error:
            report["cleanup_error"] = repr(error)
            raise
        finally:
            if session is not None:
                report["cleanup"] = getattr(session, "cleanup", None)
            report["whole_main_seconds"] = time.monotonic() - begin
            with (output / "report.json").open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "completed": report["completed"],
                "actual_posts": len(report["rows"]),
                "remainingOwned": report["cleanup"]["remainingOwned"],
            }
        )
    )


if __name__ == "__main__":
    main()
