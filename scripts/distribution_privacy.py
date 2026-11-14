"""Fail closed on private archive members; never echo matched content.

This deliberately narrow scanner supplements human review and secret scanners.
It cannot establish that arbitrary text, images or model data are anonymous.
"""

from __future__ import annotations

import stat
import tarfile
import zipfile
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path, PurePosixPath
from typing import cast


def _load_privacy_categories() -> Callable[[str, bytes], list[str]]:
    """Load the dependency-free shared rules without importing the package runtime."""
    source = Path(__file__).resolve().parents[1] / "trackmaniarl" / "_privacy.py"
    specification = spec_from_file_location("_trackmaniarl_privacy", source)
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load shared archive privacy rules")
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    classifier = getattr(module, "privacy_categories", None)
    if not callable(classifier):
        raise RuntimeError("Shared archive privacy rules are unavailable")
    return cast(Callable[[str, bytes], list[str]], classifier)


privacy_categories = _load_privacy_categories()

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
MAX_MEMBER_BYTES = 16 * 1024**2


def validate_member(name: str, data: bytes) -> None:
    relative = PurePosixPath(name.replace("\\", "/"))
    parts = [part.lower() for part in relative.parts]
    if relative.is_absolute() or ".." in parts or any(":" in part for part in parts):
        raise RuntimeError("Unsafe archive member path [redacted]")
    categories = privacy_categories(relative.as_posix(), data)
    if categories:
        raise RuntimeError(f"Archive privacy check failed: {', '.join(categories)}")
    if (
        any(
            part in _PRIVATE_PARTS
            or part.startswith((".env.", ".uv-cache", ".pytest", ".mypy", ".ruff"))
            or part == ".env"
            for part in parts
        )
        or relative.suffix.lower() in _PRIVATE_SUFFIXES
    ):
        raise RuntimeError("Forbidden private archive member [redacted]")


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
