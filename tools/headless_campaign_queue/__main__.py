"""Run a fully specified queue, stopping on the first failed command.

All jobs must already be qualified. A successful process exit is a process result,
not a claim that a training budget or evaluation is scientifically valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preflight(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        if isinstance(data, dict) and data.get("status") == "prototype_not_runnable":
            raise ValueError("campaign is not runnable: headless configurations and continuous actions are not qualified")
        raise ValueError("manifest must be an object with schema_version 1")
    jobs = data.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("manifest must contain at least one job")
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be an object")
        name, cwd, argv = job.get("id"), job.get("cwd"), job.get("argv")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) or name in seen:
            raise ValueError(f"invalid or duplicate job id: {name!r}")
        seen.add(name)
        if not isinstance(cwd, str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir():
            raise ValueError(f"{name}: cwd must be an existing absolute directory")
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a or '\0' in a for a in argv):
            raise ValueError(f"{name}: argv must be a nonempty list of nonempty strings")
        exe = Path(argv[0])
        if not exe.is_absolute() or not exe.is_file() or not os.access(exe, os.X_OK):
            raise ValueError(f"{name}: executable must be an existing executable absolute file")
    return data, hashlib.sha256(raw).hexdigest()


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


def run(data: dict, digest: str, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    # Shared across output directories for this OS user. Direct, unsupervised
    # training and another OS user are outside this lock's scope.
    global_lock = Path.home() / ".trackmaniarl-headless-campaign.lock"
    try:
        global_lock.mkdir()
    except FileExistsError:
        raise ValueError(f"global queue lock exists: {global_lock}; inspect prior processes before removing it") from None
    lock = output / ".lock"
    try:
        lock.mkdir()
    except FileExistsError:
        global_lock.rmdir()
        raise ValueError(f"queue lock exists: {lock}; inspect the prior process before removing it") from None
    except BaseException:
        global_lock.rmdir()
        raise
    state_path = output / "state.json"
    release_locks = True
    try:
        if state_path.exists():
            raise ValueError("state already exists; automatic restart/resume is disabled; use a new output directory")
        # Exclusive creation prevents overwriting pre-existing logs even if state was removed.
        logs = [output / f"{j['id']}.{stream}.log" for j in data['jobs'] for stream in ('stdout', 'stderr')]
        if any(p.exists() for p in logs):
            raise ValueError("job logs already exist; use a new output directory")
        state = {"schema_version": 1, "manifest_sha256": digest, "status": "running", "started_at": now(),
                 "jobs": [{"id": j["id"], "status": "pending"} for j in data["jobs"]]}
        atomic_json(state_path, state)
        for index, job in enumerate(data["jobs"]):
            entry = state["jobs"][index]
            entry.update(status="running", started_at=now())
            atomic_json(state_path, state)
            try:
                with logs[2 * index].open("xb") as stdout, logs[2 * index + 1].open("xb") as stderr:
                    result = subprocess.run(job["argv"], cwd=job["cwd"], stdout=stdout, stderr=stderr, shell=False, check=False)
                entry.update(status="exited", returncode=result.returncode, finished_at=now())
                if result.returncode:
                    state.update(status="failed", finished_at=now())
                    atomic_json(state_path, state)
                    return 1
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
    parser.add_argument("--execute", action="store_true", help="actually start jobs; default is preflight only")
    args = parser.parse_args(argv)
    try:
        data, digest = preflight(args.manifest)
        if not args.execute:
            print(f"Preflight OK: {len(data['jobs'])} jobs, manifest SHA-256 {digest}; nothing started")
            return 0
        return run(data, digest, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Queue error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
