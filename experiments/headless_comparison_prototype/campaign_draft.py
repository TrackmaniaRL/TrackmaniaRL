"""Build an explicitly unqualified schema-3 draft; never starts a job."""

import argparse
import json
from pathlib import Path

import trackmaniarl
from experiments.headless_comparison_prototype.prepare_qualification import sha
from trackmaniarl.core.fingerprint import run_fingerprint
from trackmaniarl.core.spec import RunSpec

UID = "oqIJ5rQDRrNwLPTh9H2p_W4tLof"


def build(deployment: Path, output: Path, ownership_root: Path) -> dict:
    deployment = deployment.resolve(strict=True)
    if not Path(trackmaniarl.__file__).resolve().is_relative_to(deployment):
        raise ValueError("fingerprint must be calculated by the frozen deployment")
    if not output.is_absolute() or output.exists() or not ownership_root.is_absolute():
        raise ValueError("fresh absolute output and absolute ownership root required")
    manifest = deployment / "campaign-preparation.json"
    candidate = json.loads(manifest.read_text(encoding="utf-8"))
    if candidate.get("status") != "not_runnable" or candidate.get("training_started") is not False:
        raise ValueError("unexpected candidate deployment status")
    for relative, expected in candidate["source_file_sha256"].items():
        file = (deployment / relative).resolve(strict=True)
        if not file.is_relative_to(deployment) or sha(file) != expected:
            raise ValueError("deployment source changed")
    jobs = []
    for job in candidate["jobs"]:
        if "config" not in job:
            raise ValueError(f"{job['algorithm']}: missing candidate config")
        path = Path(job["config"]).resolve(strict=True)
        if not path.is_relative_to(deployment) or sha(path) != job["config_sha256"]:
            raise ValueError(f"{job['algorithm']}: candidate config changed")
        spec = RunSpec.from_yaml(path)
        if spec.seed != 17 or spec.training.total_transitions != 7_168_000:
            raise ValueError("campaign seed or budget mismatch")
        name = job["algorithm"]
        jobs.append(
            {
                "id": name,
                "config": str(path),
                "config_sha256": sha(path),
                "run_fingerprint": run_fingerprint(spec, path.parent),
                "receipt": str(ownership_root / "receipts" / f"{name}.json"),
                "stop_file": str(ownership_root / "stops" / f"{name}.stop"),
                "candidate_status": job["status"],
            }
        )
    if len(jobs) != 9 or len({j["id"] for j in jobs}) != 9:
        raise ValueError("all nine unique candidates are required")
    draft = {
        "schema_version": 3,
        "status": "draft_not_qualified",
        "seed": 17,
        "transitions": 7_168_000,
        "map_uid": UID,
        "qualification_report": None,
        "qualification_sha256": None,
        "candidate_manifest_sha256": sha(manifest),
        "jobs": jobs,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(draft, stream, indent=2)
        stream.write("\n")
    return draft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ownership-root", type=Path, required=True)
    args = parser.parse_args()
    draft = build(args.deployment, args.output, args.ownership_root)
    print(f"Prepared {len(draft['jobs'])} jobs; status is draft_not_qualified")


if __name__ == "__main__":
    main()
