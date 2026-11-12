"""Fail closed on private archive members; never echo matched content.

This deliberately narrow scanner supplements human review and secret scanners.
It cannot establish that arbitrary text, images or model data are anonymous.
"""

from __future__ import annotations

import re
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

_PRIVATE_PARTS = frozenset(
    {".git", ".venv", "__pycache__", ".aws", ".ssh", "wandb", "artifacts", "recordings"}
)
_PRIVATE_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".npz",
        ".npy",
        ".mkv",
        ".mp4",
        ".sqlite3",
        ".db",
        ".log",
        ".pyc",
        ".pyo",
        ".pem",
        ".key",
    }
)
_CONTENT_PATTERNS = {
    "personal home path": re.compile(
        rb"(?i)(?:[a-z]:[\\/]+Users[\\/]+[a-z0-9_.-]+|/(?:home|Users)/[a-z0-9_.-]+)"
    ),
    "credential token": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
        rb"AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{32,}|AIza[A-Za-z0-9_-]{35}|"
        rb"wandb_v1_[A-Za-z0-9_]{20,})"
    ),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "URL credentials": re.compile(rb"https?://[^\s/:@]+:[^\s/@]+@"),
}
MAX_MEMBER_BYTES = 16 * 1024**2


def content_categories(data: bytes) -> list[str]:
    """Return categories only, including for binary metadata; never matched values."""
    return [name for name, pattern in _CONTENT_PATTERNS.items() if pattern.search(data)]


def validate_member(name: str, data: bytes) -> None:
    relative = PurePosixPath(name.replace("\\", "/"))
    parts = [part.lower() for part in relative.parts]
    if relative.is_absolute() or ".." in parts or any(":" in part for part in parts):
        raise RuntimeError("Unsafe archive member path [redacted]")
    if (
        any(
            part in _PRIVATE_PARTS
            or part.startswith((".env.", ".uv-cache", ".pytest", ".mypy", ".ruff"))
            or part == ".env"
            for part in parts
        )
        or relative.suffix.lower() in _PRIVATE_SUFFIXES
    ):
        raise RuntimeError(f"Forbidden private archive member: {relative.name}")
    categories = content_categories(data)
    if categories:
        raise RuntimeError(
            f"Archive privacy check failed: {relative.name}: {', '.join(categories)}"
        )


def validate_archive_privacy(archive: Path) -> None:
    """Inspect every member, including metadata and files outside the Python package."""
    seen: dict[str, bool] = {}
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                _validate_entry(member.filename, member.file_size, seen, directory=member.is_dir())
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise RuntimeError("Archive links are not permitted")
                validate_member(member.filename, package.read(member))
    else:
        with tarfile.open(archive) as source:
            for entry in source:
                _validate_entry(entry.name, entry.size, seen, directory=entry.isdir())
                if not (entry.isfile() or entry.isdir()):
                    raise RuntimeError("Archive links and special files are not permitted")
                file = source.extractfile(entry) if entry.isfile() else None
                validate_member(entry.name, file.read() if file is not None else b"")


def _validate_entry(  # noqa: PLR0913
    name: str, size: int, seen: dict[str, bool], *, directory: bool
) -> None:
    path = name.rstrip("/")
    if "\\" in path or name.endswith("//") or not path or PurePosixPath(path).as_posix() != path:
        raise RuntimeError("Noncanonical archive member path [redacted]")
    normalized = path.casefold()
    if normalized in seen:
        raise RuntimeError("Duplicate archive member [redacted]")
    parts = normalized.split("/")
    if any(seen.get("/".join(parts[:index])) is False for index in range(1, len(parts))):
        raise RuntimeError("Archive file and directory collide [redacted]")
    if not directory and any(name.startswith(normalized + "/") for name in seen):
        raise RuntimeError("Archive file and directory collide [redacted]")
    seen[normalized] = directory
    if size > MAX_MEMBER_BYTES:
        raise RuntimeError("Archive member exceeds the reviewed 16 MiB limit [redacted]")
