"""Audit observed Connect controls separately from issued commands and physics."""

import json
import sys
from pathlib import Path


def main() -> None:
    source, output = map(Path, sys.argv[1:3])
    trace = json.loads(source.read_bytes())
    report = {
        "schema": "tmrl.windows-input-timing/v1",
        "cases": {},
        "observed_controls_prove_physical_application": False,
        "host_timestamp_source": trace.get("host_timestamp_source", "legacy-monotonic_ns"),
        "host_timing_limitation": (
            "Host apply durations are not physics timing. "
            + (
                "This trace records perf_counter_ns."
                if trace.get("host_timestamp_source") == "perf_counter_ns"
                else "Legacy monotonic_ns durations on Windows may have coarse resolution."
            )
        ),
    }
    for name, case in trace["cases"].items():
        if not case["rows"]:
            continue
        rows = []
        for row in case["rows"]:
            samples = [row["before_raw33"], *[p["raw33"] for p in row["packets"]]]
            changes = []
            by_tick = {}
            previous = None
            for raw in samples:
                t = raw[3]
                control = tuple(raw[30:33])
                by_tick.setdefault(t, set()).add(control)
                if control != previous:
                    changes.append({"race_ms": t, "observed_sgb": list(control)})
                    previous = control
            rows.append(
                {
                    "action": row["action"],
                    "start_race_ms": row["before_raw33"][3],
                    "end_race_ms": row["after_raw33"][3],
                    "issued_gbs": row["issued_gbs"],
                    "changes": changes,
                    "host_apply_seconds": (
                        row["apply_finished_host_ns"] - row["apply_start_host_ns"]
                    )
                    / 1e9,
                    "ticks_with_multiple_controls": [t for t, v in by_tick.items() if len(v) > 1],
                }
            )
        report["cases"][name] = rows
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                name: {
                    "decisions": len(rows),
                    "distinct_step_durations": sorted(
                        {r["end_race_ms"] - r["start_race_ms"] for r in rows}
                    ),
                    "ticks_with_multiple_controls": sum(
                        len(r["ticks_with_multiple_controls"]) for r in rows
                    ),
                    "first_changed_row": next((r for r in rows if len(r["changes"]) > 1), None),
                }
                for name, rows in report["cases"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
