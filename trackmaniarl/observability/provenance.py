"""Content-addressed source snapshots, independent of checkpoint compatibility.

Snapshots contain source/configuration files, never ignored training artifacts.
They supplement the commit, not replace schema or checkpoint fingerprint checks.
External datasets, custom modules outside the source tree and map assets require
their own hashes in the experiment manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SOURCE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".lock",
        ".md",
        ".rst",
        ".txt",
        ".tex",
        ".bib",
        ".sh",
        ".ps1",
        ".as",
        ".cfg",
        ".ini",
        ".in",
        ".typed",
    }
)
_SOURCE_NAMES = frozenset({"LICENSE", "NOTICE", "Makefile", ".gitignore"})
_EXCLUDED_PARTS = frozenset(
    {".git", "__pycache__", ".venv", "node_modules", "artifacts", "build", "dist"}
)


@dataclass(frozen=True, slots=True)
class _SourceTree:
    root: Path
    revision: str | None
    dirty: bool | None
    tracked: frozenset[str]
    candidates: tuple[str, ...]
    patch: bytes


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        timeout=15,
    ).stdout


def _source_tree(root: Path) -> _SourceTree:
    try:
        git_root = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    except (OSError, subprocess.SubprocessError):
        return _unversioned_tree(root)
    tracked = frozenset(
        _git(git_root, "ls-files", "--cached", "-z").decode("utf-8").split("\0")
    ) - {""}
    untracked = set(
        _git(git_root, "ls-files", "--others", "--exclude-standard", "-z")
        .decode("utf-8")
        .split("\0")
    ) - {""}
    if root != git_root and not any(
        (git_root / name).is_relative_to(root) for name in tracked | untracked
    ):
        # An installed package in a repository's ignored .venv is not its checkout.
        return _unversioned_tree(root)
    return _SourceTree(
        git_root,
        _git(git_root, "rev-parse", "HEAD").decode().strip(),
        bool(_git(git_root, "status", "--porcelain=v1", "-z")),
        tracked,
        tuple(sorted(tracked | untracked)),
        _git(git_root, "diff", "--binary", "HEAD"),
    )


def _unversioned_tree(root: Path) -> _SourceTree:
    candidates = tuple(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not _EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
    )
    return _SourceTree(root, None, None, frozenset(), candidates, b"")


def _included(name: str) -> bool:
    path = Path(name)
    return path.suffix.lower() in _SOURCE_SUFFIXES or path.name in _SOURCE_NAMES


def _collect_sources(
    tree: _SourceTree, destination: Path
) -> tuple[dict[str, bytes], list[dict[str, Any]], list[dict[str, str]]]:
    payloads: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    omitted: list[dict[str, str]] = []
    for name in sorted(tree.candidates):
        path = tree.root / name
        if path.resolve().is_relative_to(destination):
            continue
        if not _included(name):
            omitted.append({"path": name, "reason": "outside source/configuration scope"})
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(tree.root):
            raise ValueError(f"Source snapshot cannot follow a symlink outside its tree: {name}")
        if not path.exists():
            records.append({"path": name, "deleted": True, "tracked": name in tree.tracked})
            continue
        if not path.is_file():
            omitted.append({"path": name, "reason": "not a regular file (e.g. submodule)"})
            continue
        content = path.read_bytes()
        payloads[name] = content
        records.append(
            {
                "path": name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "tracked": name in tree.tracked,
            }
        )
    return payloads, records, omitted


def capture_source_snapshot(
    destination: str | Path, *, root: str | Path | None = None
) -> dict[str, Any]:
    """Archive the actual source tree and return its verifiable identity.

    By default resolve the imported library's location, never an unrelated cwd.
    An installed wheel archives its package sources without claiming a Git HEAD.
    Supply ``root`` explicitly for a separate project's custom components.
    Call with a quiescent tree before collection; changes detected while capturing
    fail rather than yield a mixed snapshot. The destination must be local.
    """
    package = Path(__file__).resolve().parents[1]
    source_root = Path(root).resolve() if root is not None else package
    target = Path(destination).resolve()
    tree = _source_tree(source_root)
    payloads, records, omitted = _collect_sources(tree, target)
    if not payloads:
        raise ValueError(f"No source/configuration files found in {source_root}")
    identity = {
        "schema_version": 1,
        "git_revision": tree.revision,
        "dirty": tree.dirty,
        "files": records,
        "omitted": omitted,
        "patch_sha256": hashlib.sha256(tree.patch).hexdigest(),
    }
    snapshot_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    directory = target / snapshot_id
    archive = directory / "source.zip"
    manifest_path = directory / "manifest.json"
    _verify_stable(tree, payloads)
    if manifest_path.exists():
        return validate_source_snapshot(manifest_path)
    directory.mkdir(parents=True, exist_ok=True)
    _write_archive(archive, payloads)
    patch_path = directory / "working-tree.patch"
    patch_path.write_bytes(tree.patch)
    result = {
        **identity,
        "status": "captured",
        "snapshot_id": snapshot_id,
        "source_root": str(tree.root),
        "scope": "source/configuration; ignored datasets, weights and external modules excluded",
        "archive_path": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "patch_path": str(patch_path),
    }
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _verify_stable(tree: _SourceTree, payloads: dict[str, bytes]) -> None:
    for name, content in payloads.items():
        if (tree.root / name).read_bytes() != content:
            raise RuntimeError(f"Source changed during snapshot: {name}; retry after edits finish")
    current = _source_tree(tree.root)
    if current.revision != tree.revision or current.patch != tree.patch:
        raise RuntimeError("Git state changed during source snapshot; retry after edits finish")
    if set(current.candidates) != set(tree.candidates):
        raise RuntimeError("File set changed during source snapshot; retry after edits finish")


def _write_archive(path: Path, payloads: dict[str, bytes]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in payloads.items():
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_source_snapshot(manifest_path: str | Path) -> dict[str, Any]:
    """Verify the adjacent bundle's identity, archive and every source member.

    Use source.zip and working-tree.patch beside the manifest, not old absolute
    paths on the originating machine. Hashes establish byte identity, not whether
    the claimed experiment actually used those bytes.
    """
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Source snapshot manifest must be an object")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "captured":
        raise ValueError("Source snapshot requires captured schema_version 1")
    fields = ("schema_version", "git_revision", "dirty", "files", "omitted", "patch_sha256")
    if any(key not in manifest for key in fields):
        raise ValueError("Source snapshot is missing identity fields")
    identity = {key: manifest[key] for key in fields}
    expected = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("snapshot_id") != expected:
        raise ValueError("Source snapshot identity failed integrity check")
    archive_path = path.parent / "source.zip"
    patch_path = path.parent / "working-tree.patch"
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest.get("archive_sha256"):
        raise ValueError(f"Source archive failed integrity check: {archive_path}")
    if hashlib.sha256(patch_path.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Source patch failed integrity check")
    expected_files = _snapshot_members(manifest["files"])
    _validate_archive_members(archive_path, expected_files)
    return {**manifest, "archive_path": str(archive_path), "patch_path": str(patch_path)}


def _validate_archive_members(path: Path, expected: dict[str, dict[str, Any]]) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names) or set(names) != set(expected):
                raise ValueError("Source archive members differ from its file manifest")
            for name, record in expected.items():
                if archive.getinfo(name).file_size != record["bytes"]:
                    raise ValueError(f"Source member size failed integrity check: {name}")
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Source member failed integrity check: {name}")
    except zipfile.BadZipFile as error:
        raise ValueError("Source archive is not a valid ZIP") from error


def _snapshot_members(records: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("Source snapshot requires a nonempty file manifest")
    names: set[str] = set()
    members: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Source snapshot file records require paths")
        name = record["path"]
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or ":" in name
            or "\\" in name
            or name in names
        ):
            raise ValueError("Source snapshot contains unsafe or duplicate paths")
        names.add(name)
        if record.get("deleted") is True:
            continue
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise ValueError("Source file byte counts must be nonnegative integers")
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise ValueError("Source files require SHA-256 digests")
        members[name] = record
    if not members:
        raise ValueError("Source snapshot must include actual file contents")
    return members
