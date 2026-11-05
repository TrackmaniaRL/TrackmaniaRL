"""One-shot, fail-closed handoff audit for a legacy running queue job.

Capture must occur while the original supervisor and job are both alive. A later
attempt never deletes a pre-existing lock and verifies the receipt only after a
fresh lock is acquired. This module does not start a next training job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.headless_campaign_queue.__main__ import atomic_json
from tools.headless_campaign_queue.legacy_handoff import (
    capture_running_legacy,
    claim_released_lock,
    release_claimed_lock,
)
from tools.headless_campaign_queue.strict_supervisor import verify_completion


def _job_sha(job: dict) -> str:
    return hashlib.sha256(
        json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def capture_handoff(state_path: Path, output: Path, lock: Path, job: dict, digest: str) -> Path:  # noqa: PLR0913 -- explicit orchestration parameters
    """Capture live process identities to a fresh durable audit directory."""
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("expected SHA-256 manifest digest")
    if not state_path.is_absolute() or state_path.is_symlink() or not state_path.is_file():
        raise ValueError("unsafe legacy state path")
    state_raw = state_path.read_bytes()
    legacy_state = json.loads(state_raw)
    if legacy_state.get("job") != job.get("id"):
        raise ValueError("legacy state job differs from qualified plan")
    observation = capture_running_legacy(legacy_state, lock)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "handoff.json"
    atomic_json(
        snapshot,
        {
            "schema": "legacy-handoff-v1",
            "status": "watching",
            "observation": observation,
            "manifest_sha256": digest,
            "job_sha256": _job_sha(job),
            "legacy_state_sha256": hashlib.sha256(state_raw).hexdigest(),
        },
    )
    return snapshot


def attempt_handoff(  # noqa: PLR0913 -- explicit orchestration parameters
    snapshot: Path, lock: Path, job: dict, digest: str, *, verify: Any = verify_completion
) -> dict:
    """Acquire a new lock only after job and original supervisor are gone.

    On receipt failure the newly acquired lock remains and the failure is saved.
    Successful audit releases the lock: starting the next job requires its own
    qualified plan and acquisition, and is intentionally not performed here.
    """
    if not snapshot.is_absolute() or snapshot.is_symlink() or not snapshot.is_file():
        raise ValueError("unsafe handoff state")
    state = json.loads(snapshot.read_text(encoding="utf-8"))
    if (
        state.get("schema") != "legacy-handoff-v1"
        or state.get("status") != "watching"
        or state.get("manifest_sha256") != digest
        or state.get("job_sha256") != _job_sha(job)
    ):
        raise ValueError("handoff state does not match qualified plan")
    observation = state["observation"]
    lock_identity = claim_released_lock(observation, lock)
    state.update(status="auditing", lock_device=lock_identity[0], lock_inode=lock_identity[1])
    atomic_json(snapshot, state)
    try:
        evidence = verify(job, digest, observation["job"]["pid"])
        state.update(status="verified", evidence=evidence)
        atomic_json(snapshot, state)
        release_claimed_lock(lock, lock_identity)
    except BaseException as exc:
        state.update(status="blocked_requires_audit", error=f"{type(exc).__name__}: {exc}")
        atomic_json(snapshot, state)
        raise
    return evidence


def qualified_job(manifest_path: Path, expected_sha: str, job_id: str) -> dict:
    """Load an existing job from a pinned qualified manifest without freshness checks.

    This compiles evidence for audit only, not for launching a training process.
    Checkpoint and receipt are subsequently validated against this trusted pin.
    """
    from trackmaniarl.core.spec import RunSpec

    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 3 or manifest.get("status") != "qualified":
        raise ValueError("qualified campaign manifest required")
    matches = [j for j in manifest.get("jobs", []) if isinstance(j, dict) and j.get("id") == job_id]
    if len(matches) != 1:
        raise ValueError("qualified job missing or ambiguous")
    source = matches[0]
    config = Path(source["config"])
    report_path = Path(manifest["qualification_report"])
    for path in (config, report_path, Path(source["receipt"]), Path(source["stop_file"])):
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("qualified path is unsafe")
    if hashlib.sha256(config.read_bytes()).hexdigest() != source["config_sha256"]:
        raise ValueError("qualified config hash mismatch")
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != manifest["qualification_sha256"]:
        raise ValueError("qualification report hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = report.get("jobs", {}).get(job_id, {})
    if (
        evidence.get("config_sha256") != source["config_sha256"]
        or evidence.get("run_fingerprint") != source["run_fingerprint"]
        or any(
            evidence.get(k) != "pass"
            for k in (
                "native_action_ack",
                "checkpoint_restore",
                "evaluation",
                "clean_process_shutdown",
            )
        )
    ):
        raise ValueError("qualification evidence mismatch")
    spec = RunSpec.from_yaml(config)
    evaluation = spec.evaluation
    if (
        spec.seed != manifest["seed"]
        or spec.training.total_transitions != manifest["transitions"]
        or evaluation is None
        or len(evaluation.maps) != 1
        or evaluation.maps[0].expected_map_uid != manifest["map_uid"]
        or evaluation.trials_per_map != 30
    ):
        raise ValueError("qualified configuration mismatch")
    run_dir = (config.parent / spec.artifacts_dir / spec.run_id).resolve()
    return {
        "id": job_id,
        "receipt": source["receipt"],
        "transitions": manifest["transitions"],
        "max_overshoot": manifest["max_overshoot"],
        "run_fingerprint": source["run_fingerprint"],
        "map_id": evaluation.maps[0].id,
        "map_uid": manifest["map_uid"],
        "on_policy": job_id == "ppo",
        "run_dir": str(run_dir),
        "stop_file": source["stop_file"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "attempt"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--legacy-state", type=Path)
    args = parser.parse_args()
    try:
        job = qualified_job(args.manifest, args.manifest_sha256, args.job)
        lock = Path.home() / ".trackmaniarl-headless-campaign.lock"
        if args.action == "capture":
            if args.legacy_state is None:
                raise ValueError("capture requires --legacy-state")
            print(
                capture_handoff(args.legacy_state, args.audit_dir, lock, job, args.manifest_sha256)
            )
        else:
            print(
                json.dumps(
                    attempt_handoff(
                        args.audit_dir / "handoff.json", lock, job, args.manifest_sha256
                    )
                )
            )
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Legacy handoff blocked: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
