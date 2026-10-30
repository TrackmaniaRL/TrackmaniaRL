"""Sequential, pidfd-supervised retention ablation under one absolute deadline."""

import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tools.record_rl1168.paths import check_disk_reserves, private_base
from tools.record_rl1168.train import write


def main():
    root = Path(sys.argv[1])
    started = time.monotonic()
    report = {
        "status": "starting",
        "parent_pid": os.getpid(),
        "cohort_started": False,
        "stages": [],
        "comparison": "single matched seed, same update cap; transition counts disclosed",
        "public_WR_claim": False,
    }
    active = None
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
        deadline = started + (until - datetime.now(UTC)).total_seconds()
        report.update(
            status="running",
            parent=process_identity(os.getpid()),
            boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            deadline_utc=launch["deadline_utc"],
        )
        for name, digest in json.loads((root / "source-sha256.json").read_text()).items():
            if hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() != digest:
                raise ValueError("frozen source changed")
        for name, weight, seconds in [
            ("bc-control", 2.0, 840),
            ("retention-strong", 2.0, 1860),
            ("retention-weak", 0.2, 1860),
            ("retention-none", 0.0, 1860),
        ]:
            if (root / "STOP").exists() or time.monotonic() + seconds > deadline - 40:
                raise RuntimeError("insufficient remaining comparison budget or STOP")
            folder = root / name
            folder.mkdir()
            output = folder / "run"
            stage_end = min(
                until - timedelta(seconds=40),
                datetime.now(UTC) + timedelta(seconds=seconds),
            )

            def stop(output=output):
                if output.exists():
                    (output / "STOP").touch(exist_ok=True)

            private = private_base() / (root.name + "-" + name + "-supervision")
            active = CampaignProcessCohort(private, stop)
            command = [
                sys.executable,
                "-B",
                "-m",
                "tools.record_rl1168.train",
                str(prep),
                launch["qualification"],
                str(output),
                "--until-utc",
                stage_end.isoformat(),
                "--retention-weight",
                str(weight),
                "--max-updates",
                "2000",
                "--evaluation-reserve",
                "600",
            ]
            if name == "bc-control":
                command += ["--evaluation-only"]
            stage = {
                "name": name,
                "retention_weight": weight,
                "deadline_utc": stage_end.isoformat(),
                "identity": active.start("trainer", command, cwd=root / "source"),
            }
            report["cohort_started"] = True
            report["active"] = stage
            write(root / "comparison-state.json", report)
            stage_deadline = time.monotonic() + seconds
            while not active.exited():
                if (
                    time.monotonic() >= min(stage_deadline, deadline) - 30
                    or (root / "STOP").exists()
                ):
                    stop()
                    break
                check_disk_reserves(root, private)
                write(
                    root / "comparison-state.json",
                    dict(report, elapsed_seconds=time.monotonic() - started),
                )
                time.sleep(3)
            stage["closure"] = active.close(grace_seconds=20)
            active = None
            write(folder / "supervision.json", stage)
            if any(stage["closure"][k] for k in ["remainingOwned", "cleanupErrors", "forced"]):
                raise RuntimeError("unclean stage closure")
            if stage["closure"]["children"]["trainer"]["returncode"] != 0:
                raise RuntimeError("stage nonzero exit")
            completion = json.loads((output / "completion.json").read_text())
            stage["result"] = {k: v for k, v in completion.items() if k != "evaluation"}
            policy = output / "evaluated-policy.pt"
            if (
                hashlib.sha256(policy.read_bytes()).hexdigest()
                != completion["evaluated_policy_sha256"]
            ):
                raise RuntimeError("evaluation policy SHA mismatch")
            report["stages"].append(stage)
            report["active"] = None
            write(root / "comparison-state.json", report)
            if completion["status"] != "completed" or completion["evaluation_trials"] != 100:
                raise RuntimeError("incomplete stage evaluation")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
    finally:
        if active is not None:
            report["emergency_closure"] = active.close(grace_seconds=20)
        report["elapsed_seconds"] = time.monotonic() - started
        write(root / "comparison-completion.json", report)
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
