import errno
import json
import os
import subprocess
import sys
from typing import Any

import pytest

from tools.headless_campaign_queue import legacy_handoff as handoff

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux procfs")


def test_legacy_live_identity_handoff_without_touching_existing_lock(tmp_path: Any) -> None:
    lock = tmp_path / "lock"
    lock.mkdir()
    # This purpose-built child owns its own harmless subprocess, never SAC.
    program = """import json,os,pathlib,subprocess,sys
p=subprocess.Popen([sys.executable,"-c","import sys;sys.stdin.read()"],
                   stdin=subprocess.PIPE,start_new_session=True)
stat=pathlib.Path(f"/proc/{p.pid}/stat").read_text().rsplit(")",1)[1].split()
print(json.dumps({"status":"running","supervisor_pid":os.getpid(),"job_pid":p.pid,"birth_ticks":int(stat[19]),"boot_id":pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()}),flush=True)
sys.stdin.readline();p.stdin.close();p.wait()
"""
    owner = subprocess.Popen(
        [sys.executable, "-c", program], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        state = json.loads(owner.stdout.readline())
        observation = handoff.capture_running_legacy(state, lock)
        with pytest.raises(ValueError, match="still running"):
            handoff.claim_released_lock(observation, lock, reject_competitors=lambda: None)
        assert lock.is_dir()
        owner.stdin.write("\n")
        owner.stdin.flush()
        assert owner.wait(timeout=10) == 0
        # A stale, anonymous directory is never removed or claimed.
        with pytest.raises(FileExistsError):
            handoff.claim_released_lock(observation, lock, reject_competitors=lambda: None)
        assert lock.is_dir()
        lock.rmdir()  # test-owned directory only, simulates original owner release
        identity = handoff.claim_released_lock(observation, lock, reject_competitors=lambda: None)
        assert lock.is_dir()
        handoff.release_claimed_lock(lock, identity)
        assert not lock.exists()
    finally:
        if owner.poll() is None:
            owner.stdin.write("\n")
            owner.stdin.flush()
            owner.wait(timeout=10)
        owner.stdout.close()
        owner.stdin.close()


def test_capture_rejects_mismatched_birth_and_parent(tmp_path: Any) -> None:
    lock = tmp_path / "lock"
    lock.mkdir()
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys;sys.stdin.read()"],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        job, _, _ = handoff._proc_info(child.pid)
        state = {
            "status": "running",
            "supervisor_pid": os.getpid(),
            "job_pid": child.pid,
            "birth_ticks": job["start_ticks"] + 1,
            "boot_id": job["boot_id"],
        }
        with pytest.raises(ValueError, match="identities"):
            handoff.capture_running_legacy(state, lock)
        state["birth_ticks"] = job["start_ticks"]
        state["supervisor_pid"] = os.getppid()
        with pytest.raises(ValueError, match="identities"):
            handoff.capture_running_legacy(state, lock)
    finally:
        child.stdin.close()
        child.wait(timeout=10)


def test_claim_does_not_remove_preexisting_lock_after_failed_audit(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "lock"
    lock.mkdir()
    monkeypatch.setattr(handoff, "audit_owner_exited", lambda *args, **kwargs: None)
    with pytest.raises(FileExistsError):
        handoff.claim_released_lock({}, lock)
    assert lock.is_dir()


def test_release_refuses_changed_or_nonempty_lock(tmp_path: Any) -> None:
    lock = tmp_path / "lock"
    lock.mkdir()
    info = lock.stat()
    identity = info.st_dev, info.st_ino
    (lock / "marker").write_text("keep")
    with pytest.raises(OSError, match="Directory not empty") as caught:
        handoff.release_claimed_lock(lock, identity)
    assert caught.value.errno == errno.ENOTEMPTY
    assert (lock / "marker").read_text() == "keep"
    with pytest.raises(ValueError, match="identity changed"):
        handoff.release_claimed_lock(lock, (info.st_dev, info.st_ino + 1))
