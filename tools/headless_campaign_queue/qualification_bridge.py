"""Narrow source bridge for the observed 2026-10-30 shutdown-only change.

This is evidence about exact source bytes, not a claim of behavioral equivalence.
Unknown changes, files, or hash pairs fail closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OLD_TRAINING = "e8227fd66d196a0dca191dae4a01a195d1b0d1ca98a58b89419c2c9321c00396"
NEW_TRAINING = "98c3a44b84506fae9e935f2b2029b2eb555d08ed35f7def1386137c91ad78b1b"
TRAINING = "trackmaniarl/commands/training.py"
ALLOWED_TOOL_HASHES = {
    "tools/headless_campaign_queue/qualification_audit.py": {None, "2e6c3ebbdf0b16e8fab4a99922d22dec302035294dadb86775328c5b5fc4aba5"},
    "tools/headless_campaign_queue/qualification_readiness.py": {None, "2b0644a220346cfec576897fec20eeeb4523f0ebd43f09f654757bf538f4b493"},
    "tools/headless_campaign_queue/qualification_restore.py": {None, "0b6f7b261b18d2fe3fb6cdc174f3381275e5e0d05d59ab60f44ffb093b48951a"},
    "tools/headless_campaign_queue/test_qualification.py": {None, "73de1b52defaab0f3f52d987319351e0980dc3c7eafe14d5677bb92584195391", "0e003a378b40235cb229ef57c2d8bdb3058f71e8ba0a0d04edf5d82cb6b9dcbb"},
}
TARGET_TOOL_HASHES = {
    "tools/headless_campaign_queue/qualification_audit.py": "2e6c3ebbdf0b16e8fab4a99922d22dec302035294dadb86775328c5b5fc4aba5",
    "tools/headless_campaign_queue/qualification_readiness.py": "2b0644a220346cfec576897fec20eeeb4523f0ebd43f09f654757bf538f4b493",
    "tools/headless_campaign_queue/qualification_restore.py": "0b6f7b261b18d2fe3fb6cdc174f3381275e5e0d05d59ab60f44ffb093b48951a",
    "tools/headless_campaign_queue/test_qualification.py": "0e003a378b40235cb229ef57c2d8bdb3058f71e8ba0a0d04edf5d82cb6b9dcbb",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_deployment(root: Path) -> tuple[dict[str, str], str]:
    root = root.resolve(strict=True)
    manifest_path = root / "campaign-preparation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("source_file_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("deployment source inventory missing")
    for name, expected in files.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"deployment file changed: {name}")
    for folder in ("trackmaniarl", "experiments", "tools/headless_campaign_queue"):
        for path in (root / folder).rglob("*.py"):
            if "__pycache__" not in path.parts and path.relative_to(root).as_posix() not in files:
                raise ValueError(f"untracked executable source: {path}")
    return files, sha(manifest_path)


def verify_training_change(old: Path, new: Path) -> None:
    if sha(old) != OLD_TRAINING or sha(new) != NEW_TRAINING:
        raise ValueError("unrecognized training.py hash pair")
    original = old.read_text(encoding="utf-8").replace("\r\n", "\n")
    updated = new.read_text(encoding="utf-8").replace("\r\n", "\n")
    anchor = "from trackmaniarl.trackmania.demonstrations import resolve_demonstration_paths\n"
    prior = "_join_during_shutdown(processes.actor, timeout=10)"
    if original.count(anchor) != 1 or original.count(prior) != 1:
        raise ValueError("unexpected original shutdown implementation")
    replacement = anchor + "\n# Actor teardown includes native environment close, trace archiving, sender\n# drain and background-worker joins. Its internal sender and worker grace\n# periods are each 10 s, so the launcher's bound must exceed their sum.\n_ACTOR_SHUTDOWN_GRACE_S = 60.0\n"
    expected = original.replace(anchor, replacement).replace(prior, "_join_during_shutdown(processes.actor, timeout=_ACTOR_SHUTDOWN_GRACE_S)")
    if expected != updated:
        raise ValueError("training.py contains more than the pinned shutdown change")


def compare_maps(old: dict[str, str], new: dict[str, str]) -> list[dict]:
    changes = []
    for path in sorted(old.keys() | new.keys()):
        before, after = old.get(path), new.get(path)
        if before == after:
            continue
        if path == TRAINING:
            if (before, after) != (OLD_TRAINING, NEW_TRAINING):
                raise ValueError("unrecognized training source transition")
        elif path not in ALLOWED_TOOL_HASHES or before not in ALLOWED_TOOL_HASHES[path] or after != TARGET_TOOL_HASHES[path]:
            raise ValueError(f"unrecognized source change: {path}")
        changes.append({"path": path, "before_sha256": before, "after_sha256": after})
    return changes


def compare(old: Path, new: Path) -> dict:
    old_files, old_manifest = verify_deployment(old)
    new_files, new_manifest = verify_deployment(new)
    changes = compare_maps(old_files, new_files)
    if old_files.get(TRAINING) != new_files.get(TRAINING):
        verify_training_change(old / TRAINING, new / TRAINING)
    return {"old_deployment": str(old.resolve()), "new_deployment": str(new.resolve()),
            "old_manifest_sha256": old_manifest, "new_manifest_sha256": new_manifest,
            "changes": changes, "limitation": "The timeout change is documented, not proof of identical runtime behavior."}
