"""One native Python entry point for Windows and Linux X11; jobs is the safe default."""

from __future__ import annotations

import argparse
import json
import signal

from experiments.tmrl_test_comparison.policy import CAMPAIGN_ASSIGNMENTS
from experiments.tmrl_test_comparison.release_20261014_v3 import plan, runtime
from experiments.tmrl_test_comparison.release_20261014_v3.lease import controller_lease


def _interrupt(signum: int, frame: object) -> None:
    del signum, frame
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="jobs",
        choices=("prepare", "verify", "jobs", "host-check", "check-game", "run"),
    )
    parser.add_argument("--worker", choices=tuple(CAMPAIGN_ASSIGNMENTS))
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.command in {"prepare", "verify"}:
        result = plan.prepare() if args.command == "prepare" else plan.verify()
        print(f"{result['status']}: {len(result['jobs'])} jobs; no game input")
        return
    if args.worker is None:
        parser.error("--worker is required")
    protocol = plan.verify()
    jobs = [j for j in protocol["jobs"] if j["worker"] == args.worker]
    if args.command == "jobs" or (args.command == "run" and not args.launch):
        print(json.dumps(jobs, indent=2))
        print("No game input. run --worker YOUR_NAME --launch starts the queue.")
        return
    if args.command == "host-check":
        print(runtime.host_check(protocol, args.worker))
        return
    signal.signal(signal.SIGTERM, _interrupt)
    with controller_lease():
        runtime.assert_idle()
        if args.command == "check-game":
            runtime.host_check(protocol, args.worker)
            from uuid import uuid4

            from experiments.tmrl_test_comparison.preflight import run_preflight

            receipt = (
                plan.BASE / plan.CAMPAIGN_ID / "preflight" / f"{args.worker}-{uuid4().hex}.json"
            )
            run_preflight(
                plan.ROOT / jobs[0]["config"],
                receipt_path=receipt,
                stop_files=tuple(plan.stop_paths(args.worker)),
                require_new_run=True,
            )
            print(receipt)
        else:
            for job in jobs:
                plan.verify()
                runtime.run_job(protocol, job)


if __name__ == "__main__":
    main()
