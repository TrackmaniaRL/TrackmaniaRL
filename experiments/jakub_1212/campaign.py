"""Prepare, verify or explicitly launch only Jakub Szulc's QR seed 17."""

from __future__ import annotations

import argparse
import json

from experiments.jakub_1212 import plan, runtime
from experiments.jakub_1212.lease import controller_lease


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "host-check", "check-game", "run"))
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    protocol = plan.prepare() if args.command == "prepare" else plan.verify()
    if args.command in {"prepare", "verify"}:
        print(
            json.dumps(
                {"status": "verified", "operator": protocol["operator"], "jobs": protocol["jobs"]},
                indent=2,
            )
        )
        return
    if args.command == "host-check":
        print(runtime.host_check(protocol, "jakub"))
        return
    if args.command == "run" and not args.launch:
        print("No launch: add --launch to start only QR seed 17")
        return
    with controller_lease():
        runtime.assert_idle()
        if args.command == "check-game":
            from uuid import uuid4

            from experiments.jakub_1212.preflight import run_preflight

            receipt = plan.BASE / plan.CAMPAIGN_ID / "preflight" / f"{uuid4().hex}.json"
            run_preflight(
                plan.CONFIG,
                receipt_path=receipt,
                stop_files=tuple(plan.stop_paths("jakub")),
                require_new_run=True,
            )
            print(receipt)
        else:
            runtime.run_job(protocol, protocol["jobs"][0])


if __name__ == "__main__":
    main()
