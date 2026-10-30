"""Build a first-experiment manifest from independently verified short evidence.

Never starts training. Imported code and each audit subprocess must use the
corresponding frozen deployment. The report does not claim full-budget success
or arbitrary analog physics equivalence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    from .qualification_bridge import compare, sha
except ImportError:  # Direct script execution against a frozen deployment.
    from qualification_bridge import compare, sha


def invoke(args: list[str], deployment: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(deployment), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(args, cwd=deployment, env=env, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(f"offline validation failed: {result.stderr[-3000:]}")


def build(draft_path: Path, short_manifests: list[Path], restore_reports: list[Path], output: Path, *, audit_only: bool = False) -> Path | None:
    import yaml
    import trackmaniarl
    from trackmaniarl.core.fingerprint import run_fingerprint
    from trackmaniarl.core.spec import RunSpec
    from tools.headless_campaign_queue.qualification_audit import differences
    from tools.headless_campaign_queue.qualification_readiness import physical_sources

    if not output.is_absolute() or output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("fresh absolute output directory with existing parent required")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft.get("schema_version") != 3 or draft.get("status") != "draft_not_qualified":
        raise ValueError("a draft schema-3 campaign is required")
    final_deployment = Path(trackmaniarl.__file__).resolve().parent.parent
    final_manifest = final_deployment / "campaign-preparation.json"
    if sha(final_manifest) != draft.get("candidate_manifest_sha256"):
        raise ValueError("draft is not from the currently imported frozen deployment")
    if output.parent.resolve().is_relative_to(final_deployment):
        raise ValueError("output must be outside the frozen deployment")
    names = [job["id"] for job in draft["jobs"]]
    if len(set(names)) != len(names) or {"qr", "iqn", "sac", "tqc", "ppo"} - set(names):
        raise ValueError("all team algorithms and unique IDs required")
    short_data = []
    for path in short_manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        deployment = Path(data["candidate_deployment"]).resolve(strict=True)
        source_proof = compare(deployment, final_deployment)
        if source_proof["old_manifest_sha256"] != data.get("candidate_manifest_sha256"):
            raise ValueError("short manifest deployment hash mismatch")
        short_data.append((path.resolve(), data, deployment, source_proof))
    if not short_data:
        raise ValueError("short-run manifests required")
    restore_args = [item for path in restore_reports for item in ("--restore-report", str(path.resolve(strict=True)))]
    audits = []
    with tempfile.TemporaryDirectory(prefix="headless-qualification-") as temporary:
        temp = Path(temporary).resolve()
        for index, (path, data, deployment, proof) in enumerate(short_data):
            old_draft, report = temp / f"draft-{index}.json", temp / f"audit-{index}.json"
            invoke([sys.executable, "-B", "-m", "experiments.headless_comparison_prototype.campaign_draft", "--deployment", str(deployment), "--output", str(old_draft), "--ownership-root", str(temp / "unused")], deployment)
            audit_script = Path(__file__).with_name("qualification_audit.py").resolve()
            invoke([sys.executable, "-B", str(audit_script), "--draft", str(old_draft), "--short-manifest", str(path), *restore_args, "--output", str(report)], deployment)
            audits.append((data, deployment, proof, json.loads(report.read_text(encoding="utf-8"))))
    output_jobs, evidence = [], {}
    physical_cache: dict[str, dict] = {}
    allowed_output_diffs = {"components.environment.kwargs.config.runtime.trace_dir", "components.environment.kwargs.config.runtime.work_dir"}
    for job in draft["jobs"]:
        name = job["id"]
        matches = [(data, dep, proof, row) for data, dep, proof, report in audits for row in report["results"] if row["id"] == name and row.get("short_receipt") and row.get("same_checkpoint_restore") and row.get("status") != "invalid_or_incomplete"]
        if len(matches) != 1:
            raise ValueError(f"{name}: expected exactly one matching short receipt and same-checkpoint restore")
        data, old_deployment, proof, row = matches[0]
        original = next(j for j in data["jobs"] if j["algorithm"] == name)
        old_full = old_deployment / "campaign-configs" / f"{name}-s17.yaml"
        new_full = Path(job["config"]).resolve(strict=True)
        if not new_full.is_relative_to(final_deployment) or sha(new_full) != job["config_sha256"] or sha(old_full) != original["candidate_config_sha256"]:
            raise ValueError(f"{name}: full candidate config hash mismatch")
        full_diff = differences(yaml.safe_load(old_full.read_text()), yaml.safe_load(new_full.read_text()))
        if any(item["path"] not in allowed_output_diffs for item in full_diff):
            raise ValueError(f"{name}: full configurations differ beyond isolated output paths")
        spec = RunSpec.from_yaml(new_full)
        if run_fingerprint(spec, new_full.parent) != job["run_fingerprint"]:
            raise ValueError(f"{name}: final fingerprint mismatch")
        env_config = spec.components.environment.kwargs["config"]
        runtime = {k: v for k, v in env_config["runtime"].items() if k not in {"trace_dir", "work_dir"}}
        key = json.dumps({"class": spec.components.environment.class_path, "runtime": runtime,
                          "assets": env_config["asset_manifest_sha256"], "geometry": env_config["geometry_path"],
                          "source": env_config["event_source"], "shim": env_config.get("continuous_shim_sha256")}, sort_keys=True, default=str)
        if key not in physical_cache:
            physical_cache[key] = physical_sources(new_full, draft["map_uid"])
        evidence[name] = {"config_sha256": job["config_sha256"], "run_fingerprint": job["run_fingerprint"],
                          "native_action_ack": "pass", "checkpoint_restore": "pass", "evaluation": "pass", "clean_process_shutdown": "pass",
                          "short_receipt": row["short_receipt"], "short_fingerprint": row["short_fingerprint"],
                          "same_checkpoint_restore": row["same_checkpoint_restore"], "short_full_config_differences": row["config_differences"],
                          "old_vs_new_full_config_differences": full_diff, "source_bridge": proof,
                          "physical_source": physical_cache[key], "scope": "short-run evidence only; no full-budget or arbitrary analog parity claim"}
        output_jobs.append({**job, "receipt": str(output / "receipts" / f"{name}.json"), "stop_file": str(output / "stops" / f"{name}.stop")})
    if audit_only:
        return None
    report = {"kind": "headless-five-qualification-v1", "scope": "first experimental campaign; short-run evidence only", "jobs": evidence,
              "limitations": ["Full-budget completion and interruption/restart were not established.", "Short configurations changed warmup, budget, trial count and storage paths; details are retained per job.", "Bounded actuator samples and discrete physical comparisons do not establish arbitrary analog parity.", "The declared overshoot is an acceptance bound, not a collection limit.", "Source bridge documents a shutdown-grace change, not identical runtime behavior."],
              "draft_sha256": sha(draft_path)}
    output.mkdir()
    (output / "receipts").mkdir()
    (output / "stops").mkdir()
    report_path = output / "qualification.json"
    with report_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    result = {**draft, "status": "qualified", "readiness_scope": "first_experimental_campaign", "qualification_report": str(report_path), "qualification_sha256": sha(report_path), "jobs": output_jobs}
    pending, final = output / "manifest.pending.json", output / "manifest.json"
    with pending.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    invoke([sys.executable, "-B", "-m", "tools.headless_campaign_queue.campaign", str(pending), "--output", str(output / "runs")], final_deployment)
    pending.rename(final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--short-manifest", action="append", required=True, type=Path)
    parser.add_argument("--restore-report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true", help="verify evidence without creating an output directory or runnable manifest")
    args = parser.parse_args()
    path = build(args.draft, args.short_manifest, args.restore_report, args.output, audit_only=args.audit_only)
    print("Evidence verified; no manifest created" if path is None else f"Prepared (not launched): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
