"""Run a fully specified queue, stopping on the first failed command.

All jobs must already be qualified. A successful process exit is a process result,
not a claim that a training budget or evaluation is scientifically valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


def preflight(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") not in (1, 2):
        if isinstance(data, dict) and data.get("status") == "prototype_not_runnable":
            raise ValueError(
                "campaign is not runnable: headless configurations and continuous actions are not qualified"
            )
        raise ValueError("manifest must be an object with schema_version 1")
    jobs = data.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("manifest must contain at least one job")
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be an object")
        name, cwd, argv = job.get("id"), job.get("cwd"), job.get("argv")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name)
            or name in seen
        ):
            raise ValueError(f"invalid or duplicate job id: {name!r}")
        seen.add(name)
        if not isinstance(cwd, str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir():
            raise ValueError(f"{name}: cwd must be an existing absolute directory")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(a, str) or not a or "\0" in a for a in argv)
        ):
            raise ValueError(f"{name}: argv must be a nonempty list of nonempty strings")
        exe = Path(argv[0])
        if not exe.is_absolute() or not exe.is_file() or not os.access(exe, os.X_OK):
            raise ValueError(f"{name}: executable must be an existing executable absolute file")
        if data["schema_version"] == 2:
            receipt = job.get("receipt")
            budget = job.get("transitions")
            if not isinstance(receipt, str) or not Path(receipt).is_absolute():
                raise ValueError(f"{name}: receipt must be an absolute path")
            if type(budget) is not int or budget < 1:
                raise ValueError(f"{name}: transitions must be a positive integer")
            if "max_overshoot" in job and (
                type(job["max_overshoot"]) is not int or job["max_overshoot"] < 0
            ):
                raise ValueError(f"{name}: max_overshoot must be a non-negative integer")
    if data["schema_version"] == 2 and len({j["receipt"] for j in jobs}) != len(jobs):
        raise ValueError("receipt paths must be unique")
    return data, hashlib.sha256(raw).hexdigest()


def verify_receipt(job: dict, digest: str, pid: int) -> dict:
    """Check an explicit producer receipt and the referenced checkpoint bytes."""
    path = Path(job["receipt"])
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if (
        not isinstance(receipt, dict)
        or receipt.get("job_id") != job["id"]
        or receipt.get("status") != "completed"
    ):
        raise ValueError(f"{job['id']}: invalid completion receipt")
    if receipt.get("qualification_only") is True:
        raise ValueError(f"{job['id']}: qualification-only receipt cannot complete a campaign")
    if (
        receipt.get("manifest_sha256") != digest
        or type(receipt.get("process_pid")) is not int
        or receipt["process_pid"] != pid
    ):
        raise ValueError(f"{job['id']}: receipt does not match launched process and manifest")
    actual = receipt.get("transitions")
    maximum = job.get("max_overshoot", 0)
    if type(actual) is not int or not job["transitions"] <= actual <= job["transitions"] + maximum:
        raise ValueError(f"{job['id']}: receipt transition budget mismatch")
    if "max_overshoot" in job and (
        receipt.get("requested_transitions") != job["transitions"]
        or receipt.get("max_overshoot") != maximum
    ):
        raise ValueError(f"{job['id']}: receipt requested budget or acceptance limit mismatch")
    checkpoint = receipt.get("checkpoint")
    expected = receipt.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise ValueError(f"{job['id']}: invalid checkpoint path")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{job['id']}: invalid checkpoint hash")
    h = hashlib.sha256()
    with Path(checkpoint).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError(f"{job['id']}: checkpoint hash mismatch")
    evaluation = receipt.get("evaluation")
    evaluation_sha = receipt.get("evaluation_sha256")
    if (
        not isinstance(evaluation, str)
        or not Path(evaluation).is_absolute()
        or not isinstance(evaluation_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", evaluation_sha)
    ):
        raise ValueError(f"{job['id']}: invalid evaluation evidence")
    h = hashlib.sha256()
    with Path(evaluation).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != evaluation_sha:
        raise ValueError(f"{job['id']}: evaluation hash mismatch")
    evidence = {
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sha256": expected,
        "evaluation_sha256": evaluation_sha,
    }
    if "max_overshoot" in job:
        evidence["accepted_transitions"] = actual
    return evidence


def atomic_json(path: Path, data: dict) -> None:
    fd, temp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        if os.name != "nt":
            d = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(d)
            finally:
                os.close(d)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def run(data: dict, digest: str, output: Path, resume: bool = False) -> int:
    output.mkdir(parents=True, exist_ok=True)
    # Shared across output directories for this OS user. Direct, unsupervised
    # training and another OS user are outside this lock's scope.
    global_lock = Path.home() / ".trackmaniarl-headless-campaign.lock"
    try:
        global_lock.mkdir()
    except FileExistsError:
        raise ValueError(
            f"global queue lock exists: {global_lock}; inspect prior processes before removing it"
        ) from None
    lock = output / ".lock"
    try:
        lock.mkdir()
    except FileExistsError:
        global_lock.rmdir()
        raise ValueError(
            f"queue lock exists: {lock}; inspect the prior process before removing it"
        ) from None
    except BaseException:
        global_lock.rmdir()
        raise
    state_path = output / "state.json"
    release_locks = True
    try:
        logs = [
            output / f"{j['id']}.{stream}.log"
            for j in data["jobs"]
            for stream in ("stdout", "stderr")
        ]
        start = 0
        if resume:
            if data["schema_version"] != 2:
                raise ValueError("resume requires schema 2 completion receipts")
            state = json.loads(state_path.read_bytes())
            if not isinstance(state, dict):
                raise ValueError("invalid saved queue state")
            if state.get("manifest_sha256") != digest or state.get("status") != "running":
                raise ValueError(
                    "resume requires the same manifest and an interrupted running queue"
                )
            entries = state.get("jobs")
            if not isinstance(entries, list) or [j.get("id") for j in entries] != [
                j["id"] for j in data["jobs"]
            ]:
                raise ValueError("saved queue does not match manifest")
            while start < len(entries) and entries[start].get("status") == "verified":
                if type(entries[start].get("pid")) is not int:
                    raise ValueError("completed job lacks process identity")
                evidence = verify_receipt(data["jobs"][start], digest, entries[start]["pid"])
                if evidence != entries[start].get("evidence"):
                    raise ValueError("saved completion evidence has changed")
                start += 1
            if any(entry.get("status") != "pending" for entry in entries[start:]):
                raise ValueError("a started or failed job cannot be automatically resumed")
            if any(p.exists() for p in logs[2 * start :]):
                raise ValueError("pending job log exists; inspect before recovery")
            if any(Path(j["receipt"]).exists() for j in data["jobs"][start:]):
                raise ValueError("pending job receipt exists; inspect before recovery")
            state["resumed_at"] = now()
        else:
            if state_path.exists():
                raise ValueError("state already exists; use --resume only after inspection")
            if any(p.exists() for p in logs):
                raise ValueError("job logs already exist; use a new output directory")
            if data["schema_version"] == 2 and any(
                Path(j["receipt"]).exists() for j in data["jobs"]
            ):
                raise ValueError("receipt already exists; use fresh receipt paths")
            state = {
                "schema_version": 1,
                "manifest_sha256": digest,
                "status": "running",
                "started_at": now(),
                "jobs": [{"id": j["id"], "status": "pending"} for j in data["jobs"]],
            }
        atomic_json(state_path, state)
        for index in range(start, len(data["jobs"])):
            job = data["jobs"][index]
            entry = state["jobs"][index]
            entry.update(status="running", started_at=now())
            atomic_json(state_path, state)
            try:
                with logs[2 * index].open("xb") as stdout, logs[2 * index + 1].open("xb") as stderr:
                    env = os.environ.copy()
                    env["TRACKMANIARL_QUEUE_MANIFEST_SHA256"] = digest
                    env["TRACKMANIARL_QUEUE_JOB_ID"] = job["id"]
                    process = subprocess.Popen(
                        job["argv"],
                        cwd=job["cwd"],
                        stdout=stdout,
                        stderr=stderr,
                        shell=False,
                        env=env,
                    )
                    entry["pid"] = process.pid
                    atomic_json(state_path, state)
                    returncode = process.wait()
                entry.update(status="exited", returncode=returncode, finished_at=now())
                if returncode:
                    state.update(status="failed", finished_at=now())
                    atomic_json(state_path, state)
                    return 1
                if data["schema_version"] == 2:
                    try:
                        entry["evidence"] = verify_receipt(job, digest, process.pid)
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        entry.update(status="invalid_receipt", error=str(exc))
                        state.update(status="failed", finished_at=now())
                        atomic_json(state_path, state)
                        return 1
                    entry["status"] = "verified"
            except BaseException as exc:
                # The child may have survived, so force an operator inspection.
                release_locks = False
                entry.update(status="interrupted", error=type(exc).__name__, finished_at=now())
                state.update(status="interrupted", finished_at=now())
                atomic_json(state_path, state)
                raise
            atomic_json(state_path, state)
        state.update(status="completed", finished_at=now())
        atomic_json(state_path, state)
        return 0
    finally:
        if release_locks:
            lock.rmdir()
            global_lock.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="actually start jobs; default is preflight only"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue verified completed jobs after manual inspection",
    )
    args = parser.parse_args(argv)
    try:
        data, digest = preflight(args.manifest)
        if args.resume and not args.execute:
            raise ValueError("--resume requires --execute")
        if not args.execute:
            print(
                f"Preflight OK: {len(data['jobs'])} jobs, manifest SHA-256 {digest}; nothing started"
            )
            return 0
        return run(data, digest, args.output, resume=args.resume)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Queue error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
