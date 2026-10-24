"""Two explicitly approved fresh-parent repetitions of the mild-left waveform."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from headless_replay_windows_tapes import ObservedSession


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physics6(post: bytes) -> bytes:
    return post[168:192]


def main() -> None:
    config_path, receipt_path, baseline_path, output = map(Path, sys.argv[1:5])
    if output.exists():
        raise FileExistsError("fresh approved vertical experiment required")
    config = json.loads(config_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    baseline_report = json.loads(baseline_path.read_bytes())
    baseline = baseline_report["cases"]["mild_left"]
    deployment_path = Path(config["deployment_manifest"])
    if sha(deployment_path) != config["deployment_manifest_sha256"]:
        raise ValueError("pinned original deployment manifest differs")
    driver_sha = json.loads(deployment_path.read_bytes())["driverBuildApproval"]["driverSha256"]
    rows = baseline["rows"]
    if len(rows) != 200 or rows[0]["beforeRaceMs"] != 10 or rows[-1]["afterRaceMs"] != 2010:
        raise ValueError("only the approved finite 200-interval mild-left tape is allowed")
    if any(row["afterRaceMs"] - row["beforeRaceMs"] != 10 for row in rows):
        raise ValueError("complete exact observed tick tape required")
    output.mkdir(parents=True)
    report = {
        "schema": "tmrl.vertical-replay-determinism/v1",
        "passed": False,
        "native_repetitions": 2,
        "windows_control": False,
        "floating_point_control_modified": False,
        "sources": {
            "baseline": {"path": str(baseline_path), "sha256": sha(baseline_path)},
            "config": {"path": str(config_path), "sha256": sha(config_path)},
            "v7_build": {"path": str(receipt_path), "sha256": sha(receipt_path)},
        },
        "runs": [],
    }
    try:
        for repetition in range(2):
            run = {"rows": [], "completed": False}
            report["runs"].append(run)
            session = ObservedSession(
                Path(config["runtime_root"]),
                receipt,
                output / f"{output.name}-parent{repetition}",
                2190,
                driver_sha,
            )
            try:
                session.reset()
                run["neutral_seek"] = session.step(36, 1)
                for baseline_row in rows:
                    if (output / "STOP").exists():
                        raise InterruptedError("vertical replay cooperative STOP")
                    evidence = session.step_observed(baseline_row["evidence"]["waveformSource"])
                    post = bytes.fromhex(evidence["postHex"])
                    baseline_post = bytes.fromhex(baseline_row["evidence"]["postHex"])
                    run["rows"].append(
                        {
                            "race_ms": baseline_row["afterRaceMs"],
                            "same_physics6_as_baseline": physics6(post) == physics6(baseline_post),
                            "same_physics40_as_baseline": post[152:192] == baseline_post[152:192],
                            "same_native76_as_baseline": post[152:212] + post[88:104]
                            == baseline_post[152:212] + baseline_post[88:104],
                            "physical6_values": struct.unpack("<6f", physics6(post)),
                            "physics40_hex": post[152:192].hex(),
                            "native76_hex": (post[152:212] + post[88:104]).hex(),
                            "evidence": evidence,
                        }
                    )
                run["first_physics6_difference_from_baseline"] = next(
                    (row["race_ms"] for row in run["rows"] if not row["same_physics6_as_baseline"]),
                    None,
                )
                run["first_physics40_difference_from_baseline"] = next(
                    (
                        row["race_ms"]
                        for row in run["rows"]
                        if not row["same_physics40_as_baseline"]
                    ),
                    None,
                )
                run["first_native76_difference_from_baseline"] = next(
                    (row["race_ms"] for row in run["rows"] if not row["same_native76_as_baseline"]),
                    None,
                )
                run["completed"] = True
            finally:
                session.close(error=not run["completed"])
                run["cleanup"] = session.cleanup
                if session.cleanup["remainingOwned"]:
                    raise RuntimeError("vertical experiment left owned processes")
        left, right = report["runs"]
        report["first_physics40_difference_between_repetitions"] = next(
            (
                a["race_ms"]
                for a, b in zip(left["rows"], right["rows"], strict=True)
                if a["physics40_hex"] != b["physics40_hex"]
            ),
            None,
        )
        report["passed"] = all(run["completed"] for run in report["runs"])
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
