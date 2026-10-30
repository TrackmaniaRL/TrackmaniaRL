"""Outer pidfd cohort owns trainer descendants and the absolute deadline."""

import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from tools.record_rl1168.paths import check_disk_reserves, private_base
from tools.record_rl1168.train import write


def main():
    root = Path(sys.argv[1])
    started = time.monotonic()
    report = {"status": "starting", "parent_pid": os.getpid(), "cohort_started": False}
    cohort = None
    try:
        launch = json.loads((root / "launch.json").read_text())
        prep = Path(launch["preparation"])
        from tools.record_rl1168.deployment import activate_from_environment

        verified = activate_from_environment()
        if verified["deployment_manifest_sha256"] != launch["deployment_manifest_sha256"]:
            raise ValueError("launcher V8 deployment differs from qualified deployment")
        from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
        from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity

        until = datetime.fromisoformat(launch["deadline_utc"].replace("Z", "+00:00"))
        budget = (until - datetime.now(UTC)).total_seconds()
        private = private_base() / (root.name + "-supervision")

        def stop():
            run = root / "run"
            if run.exists():
                (run / "STOP").touch(exist_ok=True)

        cohort = CampaignProcessCohort(private, stop)
        report.update(parent=process_identity(os.getpid()), deadline_utc=launch["deadline_utc"])
        if budget < 1800:
            raise ValueError("insufficient new budget")
        for name, digest in json.loads((root / "source-sha256.json").read_text()).items():
            if hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() != digest:
                raise ValueError("frozen source changed")
        report["trainer"] = cohort.start(
            "trainer",
            [
                sys.executable,
                "-B",
                "-m",
                "tools.record_rl1168.train",
                str(prep),
                launch["qualification"],
                str(root / "run"),
                "--until-utc",
                launch["deadline_utc"],
            ],
            cwd=root / "source",
        )
        report["cohort_started"] = True
        report["status"] = "running"
        while not cohort.exited():
            if time.monotonic() - started >= budget - 30 or (root / "STOP").exists():
                stop()
                report.update(status="stopped", stop_reason="absolute_deadline_or_operator")
                break
            check_disk_reserves(root, private)
            write(
                root / "supervisor-state.json",
                dict(report, elapsed_seconds=time.monotonic() - started),
            )
            time.sleep(3)
        report["exits"] = cohort.exited()
        if report["exits"] == {"trainer": 0}:
            completion = json.loads((root / "run/completion.json").read_text())
            report["status"] = "completed" if completion["status"] == "completed" else "failed"
            report["trainer_status"] = completion["status"]
        elif report["exits"]:
            report.update(status="failed", error="trainer exited unsuccessfully")
    except BaseException as error:
        report.update(status="failed", error=repr(error))
    finally:
        if cohort is None:
            report["closure"] = None
        else:
            report["closure"] = cohort.close(grace_seconds=20)
        if report["closure"] and any(
            report["closure"][k] for k in ("remainingOwned", "cleanupErrors", "forced")
        ):
            report["status"] = "failed"
        report["elapsed_seconds"] = time.monotonic() - started
        write(root / "supervision.json", report)
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
