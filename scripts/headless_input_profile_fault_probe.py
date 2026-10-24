"""Bounded live rollback checks for plugin004, without driving or UI activation."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

from scripts.headless_input_profile import InputProfileConfig, NoArtificialLagLease


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-map-uid", required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("explicit --live required")
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "tmrl.live-profile-rollback/v1",
        "driving_controls_issued": False,
        "cases": [],
    }
    try:
        for name in ("client-eof", "lease-timeout", "invalid-operation", "lost-release-ack"):
            lease = NoArtificialLagLease(
                InputProfileConfig(
                    expected_map_uid=args.expected_map_uid,
                    ttl_seconds=1 if name == "lease-timeout" else 5,
                    journal_path=args.output / f"{name}.json",
                )
            )
            with lease:
                lease.check()
                if name == "client-eof":
                    lease._close_socket()  # Deliberate transport fault; plugin must restore on EOF.
                elif name == "lease-timeout":
                    until = time.monotonic() + 1.2
                    while time.monotonic() < until:
                        lease._verify_focus()
                        time.sleep(0.02)
                elif name == "invalid-operation":
                    response = lease._request("diagnostic-unknown-operation")
                    if response.get("op") == "active":
                        raise ValueError("invalid operation was silently accepted")
                else:
                    assert lease._socket is not None
                    lease._socket.shutdown(socket.SHUT_RD)
            report["cases"].append(
                {
                    "name": name,
                    "applied": lease.applied,
                    "restored_verified": lease.restored_verified,
                }
            )
        report["passed"] = all(row["restored_verified"] for row in report["cases"])
    except BaseException as error:
        report.update(passed=False, error=repr(error))
    finally:
        with (args.output / "report.json").open("x", encoding="utf-8") as target:
            json.dump(report, target, indent=2, allow_nan=False)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
