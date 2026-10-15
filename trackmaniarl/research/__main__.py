"""Offline research preparation: ``python -m trackmaniarl.research --help``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trackmaniarl.research.manifest import preflight
from trackmaniarl.research.planning import plan_runs
from trackmaniarl.research.reporting import write_report
from trackmaniarl.research.results import aggregate_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="verify local map assets without contacting game")
    check.add_argument("manifest", type=Path)
    check.add_argument("--map-id")
    check.add_argument(
        "--observed-uid", help="optional operator observation, not live verification"
    )
    plan = commands.add_parser("plan", help="materialize a bounded plan without executing it")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("protocol", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    report = commands.add_parser(
        "aggregate", help="validate real result artifacts and build tables"
    )
    report.add_argument("registry", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--plot", action="store_true", help="requires optional matplotlib")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.manifest, args.map_id, args.observed_uid)
        elif args.command == "plan":
            result = plan_runs(args.manifest, args.protocol, args.output)
        else:
            result = aggregate_registry(args.registry)
            write_report(result, args.output, plot=args.plot)
            result = {
                "status": "written",
                "output": str(args.output),
                "evidence": result["evidence"],
                "runs": len(result["runs"]),
            }
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"research: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
