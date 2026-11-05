"""Read and inventory evidence; never promote a draft or start a job."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


ALLOWED_DIFFERENCES = frozenset(
    {
        "run_id",
        "artifacts_dir",
        "training.total_transitions",
        "training.checkpoint_milestones",
        "training.warmup_transitions",
        "evaluation.trials_per_map",
        "components.additional_loggers",
        "components.environment.kwargs.config.runtime.trace_dir",
        "components.environment.kwargs.config.runtime.work_dir",
        "components.environment.kwargs.config.trace_archive_root",
        "metadata.qualification_purpose",
    }
)


def differences(full: Any, short: Any, path: str = "") -> list[dict]:
    if isinstance(full, Mapping) and isinstance(short, Mapping):
        result: list[dict] = []
        for key in sorted(set(full) | set(short)):
            p = f"{path}.{key}" if path else str(key)
            if key not in full or key not in short:
                result.append({"path": p, "full": full.get(key), "short": short.get(key)})
            else:
                result.extend(differences(full[key], short[key], p))
        return result
    if full != short:
        return [{"path": path, "full": full, "short": short}]
    return []


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def matching_restore(  # noqa: PLR0913 -- explicit orchestration parameters
    restores: list[dict],
    name: str,
    short_fp: str,
    config_sha: str,
    receipt_sha: str,
    checkpoint_sha: str,
) -> dict | None:
    for restore in restores:
        value = restore["report"]
        source = value.get("source_sha256", {})
        if (
            value.get("kind") == "headless-qualification-offline-restore-v1"
            and value.get("status") == "PASS"
            and value.get("job_id") == name
            and value.get("run_fingerprint") == short_fp
            and source.get("checkpoint") == checkpoint_sha
            and source.get("receipt") == receipt_sha
            and source.get("config") == config_sha
        ):
            return {"path": restore["path"], "sha256": restore["sha256"]}
    return None


def audit(draft_path: Path, short_manifest_path: Path, restore_reports: list[Path]) -> dict:
    import trackmaniarl
    from trackmaniarl.core.builtins import TorchCheckpointCodec
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.spec import RunSpec

    draft, short_manifest = read_object(draft_path), read_object(short_manifest_path)
    if draft.get("schema_version") != 3 or draft.get("status") != "draft_not_qualified":
        raise ValueError("only an unqualified schema-3 draft can be audited")
    deployment = Path(short_manifest["candidate_deployment"]).resolve(strict=True)
    if not Path(trackmaniarl.__file__).resolve().is_relative_to(deployment):
        raise ValueError("audit must use the frozen candidate deployment")
    candidate_manifest = deployment / "campaign-preparation.json"
    if sha(candidate_manifest) != draft.get("candidate_manifest_sha256") or sha(
        candidate_manifest
    ) != short_manifest.get("candidate_manifest_sha256"):
        raise ValueError("draft and short configs use different source deployments")
    candidate = read_object(candidate_manifest)
    for name, digest in candidate["source_file_sha256"].items():
        file = (deployment / name).resolve(strict=True)
        if not file.is_relative_to(deployment) or sha(file) != digest:
            raise ValueError("frozen deployment source differs from its manifest")
    short_jobs = {j["algorithm"]: j for j in short_manifest["jobs"]}
    if len(short_jobs) != len(short_manifest["jobs"]):
        raise ValueError("duplicate qualification jobs")
    restores = []
    for path in restore_reports:
        value = read_object(path)
        restores.append({"path": str(path.resolve()), "sha256": sha(path), "report": value})
    results = []
    for job in draft["jobs"]:
        name = job["id"]
        row: dict[str, Any] = {
            "id": name,
            "status": "insufficient_evidence",
            "full_config_sha256": job["config_sha256"],
            "full_fingerprint": job["run_fingerprint"],
            "short_receipt": None,
            "same_checkpoint_restore": None,
            "native_action_parity": "not_evidenced",
        }
        try:
            short = short_jobs[name]
            full_path, short_path = Path(job["config"]), Path(short["config"])
            if not full_path.resolve(strict=True).is_relative_to(deployment):
                raise ValueError("full config outside deployment")
            if (
                sha(full_path) != job["config_sha256"]
                or short.get("candidate_config_sha256") != job["config_sha256"]
                or sha(short_path) != short["config_sha256"]
            ):
                raise ValueError("candidate/short config hash mismatch")
            full, small = RunSpec.from_yaml(full_path), RunSpec.from_yaml(short_path)
            full_fp, short_fp = (
                run_fingerprint(full, full_path.parent),
                run_fingerprint(small, short_path.parent),
            )
            if full_fp != job["run_fingerprint"]:
                raise ValueError("full fingerprint mismatch")
            diff = differences(full.model_dump(mode="json"), small.model_dump(mode="json"))
            row["config_differences"] = diff
            unexpected = [item["path"] for item in diff if item["path"] not in ALLOWED_DIFFERENCES]
            if unexpected:
                raise ValueError("unexpected config differences: " + ", ".join(unexpected))
            row.update(short_config_sha256=sha(short_path), short_fingerprint=short_fp)
            receipt_path = short_manifest_path.parent / (name + "-receipt.json")
            if not receipt_path.exists():
                failed = short_manifest_path.parent / (name + "-outcome-fail.json")
                if failed.is_file():
                    failure = read_object(failed)
                    if failure.get("algorithm") != name:
                        raise ValueError("failure artifact algorithm mismatch")
                    row["short_failure"] = {
                        "path": str(failed.resolve()),
                        "sha256": sha(failed),
                        "details": failure,
                    }
                row["missing"] = [
                    "short completed receipt",
                    "same-checkpoint restore",
                    "native action parity",
                ]
                results.append(row)
                continue
            receipt = read_object(receipt_path)
            if (
                receipt.get("job_id") != name
                or receipt.get("status") != "completed"
                or receipt.get("qualification_only") is not True
                or receipt.get("run_fingerprint") != short_fp
            ):
                raise ValueError("short receipt identity mismatch")
            requested, actual, limit = (
                receipt.get("requested_transitions"),
                receipt.get("transitions"),
                receipt.get("max_overshoot"),
            )
            if (
                type(requested) is not int
                or type(actual) is not int
                or type(limit) is not int
                or requested != small.training.total_transitions
                or limit < 0
                or not requested <= actual <= requested + limit
            ):
                raise ValueError("short receipt budget mismatch")
            cp, evaluation = Path(receipt["checkpoint"]), Path(receipt["evaluation"])
            run_root = (short_path.parent / small.artifacts_dir / small.run_id).resolve()
            eval_root = (
                short_path.parent / small.artifacts_dir / (small.run_id + "-final-evaluation")
            ).resolve()
            if (
                cp.is_symlink()
                or cp.resolve().parent != run_root / "checkpoints"
                or sha(cp) != receipt["checkpoint_sha256"]
            ):
                raise ValueError("short checkpoint identity or hash mismatch")
            if (
                evaluation.is_symlink()
                or evaluation.resolve() != eval_root / "evaluation.json"
                or sha(evaluation) != receipt["evaluation_sha256"]
            ):
                raise ValueError("short evaluation identity or hash mismatch")
            # The codec materializes a temporary file beside its input. Work on
            # a private copy so auditing never writes into the source run.
            with tempfile.TemporaryDirectory(prefix="headless-evidence-") as tmp:
                copy = Path(tmp) / "checkpoint.pt"
                shutil.copyfile(cp, copy)
                if sha(copy) != receipt["checkpoint_sha256"]:
                    raise ValueError("short checkpoint changed while copying")
                state = TorchCheckpointCodec().load(copy)
            on_policy = name == "ppo"
            counters = state.get("counters" if on_policy else "distributed")
            if (
                state.get("run_fingerprint") != short_fp
                or not isinstance(counters, Mapping)
                or counters.get("transitions") != actual
                or not isinstance(state.get("learner"), Mapping)
            ):
                raise ValueError("receipt and checkpoint transitions differ")
            if not on_policy and (
                not isinstance(state.get("replay_store"), Mapping)
                or not isinstance(state.get("sampler"), Mapping)
            ):
                raise ValueError("short checkpoint missing replay or sampler")
            eval_data = read_object(evaluation)
            map_spec = small.evaluation.maps[0]
            trials = eval_data.get("trials")
            if (
                eval_data.get("checkpoint") != str(cp)
                or not isinstance(trials, list)
                or len(trials) != small.evaluation.trials_per_map
            ):
                raise ValueError("short evaluation trial count or checkpoint mismatch")
            if any(
                not isinstance(trial, Mapping) or type(trial.get("trial_index")) is not int
                for trial in trials
            ):
                raise ValueError("short evaluation trial shape mismatch")
            expected_trials = {
                (map_spec.id, map_spec.expected_map_uid, n)
                for n in range(small.evaluation.trials_per_map)
            }
            observed = {
                (trial.get("map_id"), trial.get("map_uid"), trial.get("trial_index"))
                for trial in trials
            }
            if observed != expected_trials:
                raise ValueError("short evaluation trial identities mismatch")
            if any(
                row_.get("telemetry_error") or row_.get("controller_error")
                for row_ in eval_data["trials"]
            ):
                raise ValueError("short evaluation contains runtime errors")
            row["short_receipt"] = {
                "path": str(receipt_path.resolve()),
                "sha256": sha(receipt_path),
                "checkpoint_sha256": sha(cp),
                "evaluation_sha256": sha(evaluation),
                "accepted_transitions": actual,
            }
            row["same_checkpoint_restore"] = matching_restore(
                restores,
                name,
                short_fp,
                sha(short_path),
                sha(receipt_path),
                receipt["checkpoint_sha256"],
            )
            row["missing"] = (
                ["same-checkpoint restore"] if row["same_checkpoint_restore"] is None else []
            ) + ["native action parity", "full-budget qualification"]
        except Exception as exc:
            row["status"] = "invalid_or_incomplete"
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    return {
        "kind": "headless-campaign-evidence-inventory-v1",
        "status": "not_qualified",
        "draft_sha256": sha(draft_path),
        "short_manifest_sha256": sha(short_manifest_path),
        "deployment": str(deployment),
        "results": results,
        "limitations": [
            "short-run evidence is not full-budget qualification",
            "native actuator parity and same-checkpoint restore require separate evidence",
            "this report cannot make the draft runnable",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--short-manifest", required=True, type=Path)
    parser.add_argument("--restore-report", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        parser.error("output must be a fresh absolute file")
    report = audit(args.draft, args.short_manifest, args.restore_report)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"status": report["status"], "jobs": len(report["results"]), "sha256": sha(args.output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
