"""Create short, isolated qualification configs; never start an environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from trackmaniarl.core.spec import RunSpec


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(deployment: Path, output: Path) -> dict:
    deployment = deployment.resolve(strict=True)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("qualification output must be a fresh absolute path")
    if str(output).startswith("/mnt/") or output.parent.resolve(strict=True) != output.parent:
        raise ValueError("qualification output must be a canonical Linux path")
    source_manifest = deployment / "campaign-preparation.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "not_runnable" or source.get("training_started") is not False:
        raise ValueError("unexpected candidate deployment status")
    for name, expected in source["source_file_sha256"].items():
        target = (deployment / name).resolve(strict=True)
        if not target.is_relative_to(deployment) or not target.is_file() or sha(target) != expected:
            raise ValueError("candidate deployment source changed")
    prepared = []
    for job in source["jobs"]:
        if "config" not in job:
            raise ValueError(f"{job['algorithm']}: missing candidate config")
        path = Path(job["config"]).resolve(strict=True)
        if not path.is_relative_to(deployment) or sha(path) != job["config_sha256"]:
            raise ValueError(f"{job['algorithm']}: candidate config changed")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = job["algorithm"]
        config["run_id"] = f"headless-qualification-{name}-s17"
        config["artifacts_dir"] = str(output / "artifacts")
        config["training"]["total_transitions"] = 2048
        config["training"]["checkpoint_milestones"] = [2048]
        if name != "ppo":
            config["training"]["warmup_transitions"] = 256
        config["evaluation"]["trials_per_map"] = 2
        config["components"]["additional_loggers"] = []
        native = config["components"]["environment"]["kwargs"]["config"]
        native["runtime"]["trace_dir"] = str(output / "native-traces" / name)
        native["runtime"]["work_dir"] = str(
            Path(native["runtime"]["runtime_root"]) / "build/server-probe" / output.name / name
        )
        native["trace_archive_root"] = str(output / "native-archives" / name)
        config["metadata"]["qualification_purpose"] = (
            "short infrastructure test; not a comparison result"
        )
        blocked = job["status"] != "candidate_not_qualified"
        # Continuous candidates preserve their original hard block in both name and manifest.
        name_out = f"{name}-s17.yaml" + (".blocked" if blocked else "")
        prepared.append((name, name_out, config, job["status"], path))
    if len(prepared) != 9 or len({item[0] for item in prepared}) != 9:
        raise ValueError("the complete nine-algorithm candidate set is required")
    output.mkdir()
    configs = output / "configs"
    configs.mkdir()
    jobs = []
    for algorithm, name, config, state, original in prepared:
        target = configs / name
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        RunSpec.from_yaml(target)
        jobs.append(
            {
                "algorithm": algorithm,
                "status": state,
                "config": str(target),
                "config_sha256": sha(target),
                "candidate_config_sha256": sha(original),
            }
        )
    receipt = {
        "status": "qualification_prepared_not_started",
        "transitions": 2048,
        "trials_per_map": 2,
        "training_started_at_preparation": False,
        "candidate_deployment": str(deployment),
        "candidate_manifest_sha256": sha(source_manifest),
        "jobs": jobs,
    }
    (output / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.deployment, args.output), indent=2))


if __name__ == "__main__":
    main()
