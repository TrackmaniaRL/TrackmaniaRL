"""Fail-closed binding of the prepared plan and native backend source files."""

import hashlib
import json
import re
import sys
import types
from pathlib import Path

_BACKEND_MODULES = ("orientation", "multi_goal", "environment")
_REQUIRED_SOURCES = {
    "__init__.py",
    "archive.py",
    "closure.py",
    "compare.py",
    "deployment.py",
    "episodes.py",
    "features.py",
    "learner.py",
    "model.py",
    "paths.py",
    "prepare.py",
    "provenance.py",
    "qualify.py",
    "runtime.py",
    "smoke.py",
    "supervise.py",
    "train.py",
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def verify_preparation(preparation):
    """Read the plan once and bind it and policy source pins to the receipt."""
    preparation = Path(preparation)
    receipt_bytes = (preparation / "preparation.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    plan_bytes = (preparation / "plan.json").read_bytes()
    pins_bytes = (preparation / "source-pins.json").read_bytes()
    plan_sha, pins_sha = sha_bytes(plan_bytes), sha_bytes(pins_bytes)
    if receipt.get("plan_sha256") != plan_sha:
        raise ValueError("preparation plan missing a matching SHA; requalify a fresh preparation")
    if receipt.get("source_pins_sha256") != pins_sha:
        raise ValueError("preparation source pins missing a matching SHA")
    return (
        json.loads(plan_bytes),
        receipt,
        json.loads(pins_bytes),
        plan_sha,
        pins_sha,
        sha_bytes(receipt_bytes),
    )


def verify_qualification_binding(proof, plan_sha, pins_sha, receipt_sha, deployment_sha):
    """Do not apply a qualification to another plan, policy ABI or deployment."""
    expected = {
        "plan_sha256": plan_sha,
        "source_pins_sha256": pins_sha,
        "preparation_sha256": receipt_sha,
        "deployment_manifest_sha256": deployment_sha,
    }
    if any(proof.get(key) != value or not value for key, value in expected.items()):
        raise ValueError("qualification does not bind this plan, source pins and deployment")


def verify_preparation_deployment(receipt, deployment_sha):
    if not deployment_sha or receipt.get("deployment_manifest_sha256") != deployment_sha:
        raise ValueError("preparation does not bind the qualified V8 deployment")


def verify_source_pins(pins, code_root):
    """Require the complete prepared source set and reject added executable modules."""
    if not isinstance(pins, dict) or not _REQUIRED_SOURCES.issubset(pins):
        raise ValueError("preparation source pins are incomplete")
    root = Path(code_root)
    executable = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("source symlinks are forbidden")
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in {".py", ".pyw", ".pyc", ".pyd", ".so"}:
            executable[path.relative_to(root).as_posix()] = path
    if set(pins) != set(executable):
        raise ValueError("prepared executable source set changed")
    for name, expected in pins.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid source pin: " + str(name))
        if sha_bytes(executable[name].read_bytes()) != expected:
            raise ValueError("prepared source changed: " + name)


def _linux_or_native_path(value):
    raw = str(value)
    if "\x00" in raw or "\n" in raw or "\r" in raw:
        raise ValueError("invalid path")
    path = Path(raw)
    if path.is_absolute():
        return path
    drive = re.fullmatch(r"([a-zA-Z]):[\\/](.*)", raw)
    if drive and sys.platform != "win32":
        return Path("/mnt") / drive.group(1).lower() / drive.group(2).replace("\\", "/")
    raise ValueError("absolute path required")


def verify_backend(plan):
    """Return the exact backend bytes after checking every import against the manifest.

    The source directory is never put on sys.path: it may also contain an old SDK.
    Extra manifest entries are allowed for historical receipts but cannot substitute
    for any of the three explicitly imported backend modules.
    """
    source = _linux_or_native_path(plan["backend_config"]["source"])
    if ".." in source.parts or not source.is_dir() or source.resolve(strict=True) != source:
        raise ValueError("backend source must be a canonical absolute directory")
    manifest = _linux_or_native_path(plan["backend_source_manifest"])
    if ".." in manifest.parts or manifest.resolve(strict=True) != manifest:
        raise ValueError("backend manifest must be a canonical absolute file")
    raw = manifest.read_bytes()
    if sha_bytes(raw) != plan["backend_source_manifest_sha256"]:
        raise ValueError("backend manifest changed")
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise ValueError("backend source manifest must be a list")
    by_path = {}
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("malformed backend manifest entry")
        path = _linux_or_native_path(item["path"])
        if ".." in path.parts or path in by_path:
            raise ValueError("duplicate or noncanonical backend manifest path")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid backend SHA")
        by_path[path] = digest
    verified = {}
    for name in _BACKEND_MODULES:
        path = source / (name + ".py")
        if path not in by_path:
            raise ValueError("backend manifest does not cover " + name)
        if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("backend module must be a canonical regular file: " + name)
        data = path.read_bytes()
        if sha_bytes(data) != by_path[path]:
            raise ValueError("backend module SHA mismatch: " + name)
        verified[name] = (path, data)
    return verified


def load_verified_backend(verified):
    """Load only the already verified bytes; do not expose the backend directory to imports."""
    names = set(_BACKEND_MODULES)
    if names.intersection(sys.modules):
        raise RuntimeError("backend module name already loaded; fresh interpreter required")
    installed = []
    try:
        for name in _BACKEND_MODULES:
            path, data = verified[name]
            module = types.ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            installed.append(name)
            exec(compile(data, str(path), "exec"), module.__dict__)
        return sys.modules["environment"].FallEnvironment
    except BaseException:
        for name in installed:
            sys.modules.pop(name, None)
        raise
