"""Validate the release archives produced from this checkout."""

from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import cast

if __package__:
    from .distribution_manifest import DIAGRAM_STEMS, DIAGRAM_SUFFIXES, SDIST_REQUIRED_PATHS
    from .distribution_privacy import validate_archive_privacy
else:
    from distribution_manifest import DIAGRAM_STEMS, DIAGRAM_SUFFIXES, SDIST_REQUIRED_PATHS
    from distribution_privacy import validate_archive_privacy

SBOM_NAME = "trackmaniarl-release.spdx.json"
CHECKSUMS_NAME = "SHA256SUMS"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CORE_DEPENDENCIES = frozenset(
    {"gymnasium", "numpy", "pydantic", "pyyaml", "tensordict", "torch", "zstandard"}
)
PUBLIC_EXTRAS = frozenset(
    {
        "all",
        "distributed",
        "mamba",
        "native-vision",
        "orchestrator",
        "trackmania",
        "vision",
        "wandb",
    }
)


def _project_version() -> str:
    with Path("pyproject.toml").open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def _require_members(names: set[str], required: set[str], archive: Path) -> None:
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"{archive} is missing: {', '.join(missing)}")


def _checkout_bytes(relative: PurePosixPath, archive: Path) -> bytes:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"{archive} has unsafe source member {relative.as_posix()!r}")
    source = Path(*relative.parts)
    if not source.is_file():
        raise RuntimeError(f"{archive} member {relative.as_posix()!r} has no checkout file")
    return source.read_bytes()


def _normalize_checkout_line_endings(content: bytes) -> bytes:
    """Compare text independent of Git's Windows checkout conversion."""
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return content.replace(b"\r\n", b"\n")


def _validate_wheel_checkout_files(package: zipfile.ZipFile, archive: Path) -> None:
    for member in package.infolist():
        relative = PurePosixPath(member.filename.replace("\\", "/"))
        if member.is_dir() or not relative.parts or relative.parts[0] != "trackmaniarl":
            continue
        packaged = _normalize_checkout_line_endings(package.read(member))
        checkout = _normalize_checkout_line_endings(_checkout_bytes(relative, archive))
        if packaged != checkout:
            raise RuntimeError(
                f"{archive} member {relative.as_posix()!r} differs from the checkout"
            )


def _sdist_checkout_path(member_name: str, root: str) -> PurePosixPath | None:
    packaged = PurePosixPath(member_name.replace("\\", "/"))
    if len(packaged.parts) < 2 or packaged.parts[0] != root:
        return None
    relative = PurePosixPath(*packaged.parts[1:])
    if relative.as_posix() in {"PKG-INFO", "setup.cfg"}:
        return None
    if relative.parts[0].lower().endswith(".egg-info"):
        return None
    return relative


def _validate_sdist_checkout_files(package: tarfile.TarFile, archive: Path, root: str) -> None:
    for member in package.getmembers():
        relative = _sdist_checkout_path(member.name, root)
        if relative is None or not member.isfile():
            continue
        packaged = package.extractfile(member)
        if packaged is None:
            raise RuntimeError(f"{archive} cannot read member {member.name!r}")
        checkout = _normalize_checkout_line_endings(_checkout_bytes(relative, archive))
        if _normalize_checkout_line_endings(packaged.read()) != checkout:
            raise RuntimeError(f"{archive} member {member.name!r} differs from the checkout")


def _wheel_required_members(version: str) -> set[str]:
    dist_info = f"trackmaniarl-{version}.dist-info"
    sources = {path.as_posix() for path in Path("trackmaniarl").rglob("*.py")}
    return sources | {
        "trackmaniarl/py.typed",
        "trackmaniarl/project/scaffold.py",
        "trackmaniarl/project/scaffold_run_templates.py",
        "trackmaniarl/project/scaffold_templates.py",
        "trackmaniarl/project/openplanet/README.md",
        "trackmaniarl/project/openplanet/TrackmaniaRL_Connect.as",
        "trackmaniarl/project/openplanet/info.toml",
        "trackmaniarl/trackmania/native_headless/_runtime/portable_profile.json",
        "trackmaniarl/trackmania/native_headless/_runtime/immediate_profile.json",
        "trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/v6-to-v7.patch",
        "trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/README.md",
        f"{dist_info}/entry_points.txt",
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/licenses/NOTICE",
        f"{dist_info}/METADATA",
    }


def _validate_wheel_metadata(metadata: Message, archive: Path, version: str) -> None:
    invalid = _wheel_identity_errors(metadata, version)
    invalid.extend(_wheel_dependency_errors(metadata))
    if invalid:
        raise RuntimeError(f"{archive} has invalid metadata: {', '.join(invalid)}")


def _wheel_identity_errors(metadata: Message, version: str) -> list[str]:
    expected = {
        "Name": "TrackmaniaRL",
        "Version": version,
    }
    invalid = [
        f"{key}={metadata[key]!r}" for key, value in expected.items() if metadata[key] != value
    ]
    requires_python = metadata["Requires-Python"]
    constraints = (
        {part.strip() for part in requires_python.split(",")} if requires_python else set()
    )
    if constraints != {">=3.12", "<3.13"}:
        invalid.append(f"Requires-Python={requires_python!r}")
    return invalid


def _wheel_dependency_errors(metadata: Message) -> list[str]:
    requirements = metadata.get_all("Requires-Dist") or []
    extras = frozenset(metadata.get_all("Provides-Extra") or [])
    invalid: list[str] = []
    if extras != PUBLIC_EXTRAS:
        invalid.append(f"Provides-Extra={sorted(extras)!r}")
    if any("@" in requirement for requirement in requirements):
        invalid.append("Requires-Dist contains a direct URL")
    if any("vgamepad" in requirement.lower() for requirement in requirements):
        invalid.append("Requires-Dist contains vgamepad")
    for extra in ("native-vision", "all"):
        matches = [
            match
            for requirement in requirements
            if (
                match := re.fullmatch(
                    r"\s*numba\s*([^;]+)\s*;\s*extra\s*==\s*['\"]([^'\"]+)['\"]\s*",
                    requirement,
                    flags=re.IGNORECASE,
                )
            )
            and match.group(2).lower() == extra
            and {specifier.strip() for specifier in match.group(1).split(",")}
            == {">=0.64", "<0.69"}
        ]
        if len(matches) != 1:
            invalid.append(f"Requires-Dist is missing numba bounds for extra {extra}")
    if not any(
        "libevdev>=0.13;" in requirement.lower()
        and "extra == 'trackmania'" in requirement.lower().replace('"', "'")
        for requirement in requirements
    ):
        invalid.append("Requires-Dist is missing the Trackmania libevdev extra")
    return invalid


def _validate_wheel(archive: Path, version: str) -> None:
    validate_archive_privacy(archive)
    dist_info = f"trackmaniarl-{version}.dist-info"
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        _require_members(names, _wheel_required_members(version), archive)
        if any(name.startswith("trackmaniarl/project/openplanet/Headless") for name in names):
            raise RuntimeError(f"{archive} includes repository-only Openplanet probes")
        private_evidence = {
            "trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/" + name
            for name in ("original-source-manifest.json", "build-receipt.json")
        }
        if names & private_evidence:
            raise RuntimeError(f"{archive} includes private V7 source/build evidence")
        _validate_wheel_checkout_files(package, archive)
        metadata = email.parser.BytesParser().parsebytes(package.read(f"{dist_info}/METADATA"))
    _validate_wheel_metadata(metadata, archive, version)


def _sdist_required_members(root: str) -> set[str]:
    diagrams = {
        f"{root}/docs/diagrams/{stem}{suffix}"
        for stem in DIAGRAM_STEMS
        for suffix in DIAGRAM_SUFFIXES
    }
    return diagrams | {f"{root}/{path}" for path in SDIST_REQUIRED_PATHS}


def _validate_sdist(archive: Path, version: str) -> None:
    validate_archive_privacy(archive)
    root = f"trackmaniarl-{version}"
    with tarfile.open(archive) as package:
        names = {member.name.replace("\\", "/") for member in package.getmembers()}
        private_evidence = {
            f"{root}/trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/" + name
            for name in ("original-source-manifest.json", "build-receipt.json")
        }
        if names & private_evidence:
            raise RuntimeError(f"{archive} includes private V7 source/build evidence")
        _validate_sdist_checkout_files(package, archive, root)
    _require_members(names, _sdist_required_members(root), archive)


def _release_tree(commit: str) -> tuple[str, dict[str, tuple[str, bool]]]:
    """Load the committed regular files for a specifically identified release."""
    if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit) is None:
        raise RuntimeError("Release commit must be a full Git object ID")
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(root).resolve() != Path.cwd().resolve() or resolved.lower() != commit.lower():
            raise RuntimeError(
                "Release validation requires the repository root and an exact commit"
            )
        tree = subprocess.run(
            ["git", "ls-tree", "-r", "-z", "--full-tree", resolved],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot read the specified Git release commit") from error
    digests: dict[str, str] = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, digest = header.split(b" ", 2)
        if mode in {b"100644", b"100755"} and kind == b"blob":
            digests[path.decode("utf-8")] = digest.decode("ascii")
    try:
        attributes = subprocess.run(
            ["git", "check-attr", f"--source={resolved}", "-z", "--stdin", "text"],
            input=b"".join(path.encode("utf-8") + b"\0" for path in digests),
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot read release commit Git attributes") from error
    text_attributes = {
        attributes[index].decode("utf-8"): attributes[index + 2] != b"unset"
        for index in range(0, len(attributes) - 1, 3)
    }
    files = {path: (digest, text_attributes[path]) for path, digest in digests.items()}
    return resolved, files


def _select_release_tree(
    requested: str | None, *, required: bool
) -> tuple[str, dict[str, tuple[str, bool]]] | None:
    if requested is not None:
        return _release_tree(requested)
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        if Path(".git").exists():
            raise RuntimeError("Cannot inspect the Git checkout for release validation") from None
        root = ""
    if not root or Path(root).resolve() != Path.cwd().resolve():
        if required:
            raise RuntimeError("Release validation requires a Git checkout and an exact commit")
        print(
            "Warning: outside a Git checkout; archive sources are not verified against a commit",
            file=sys.stderr,
        )
        return None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Cannot resolve the Git checkout's HEAD commit") from error
    return _release_tree(commit)


def _validate_committed_member(  # noqa: PLR0913
    path: str, content: bytes, files: dict[str, tuple[str, bool]], archive: Path
) -> None:
    entry = files.get(path)
    if entry is None:
        raise RuntimeError(f"{archive} member {path!r} is not a committed regular file")
    expected, may_normalize = entry
    algorithm = "sha256" if len(expected) == 64 else "sha1"
    candidates = [content]
    if may_normalize and b"\0" not in content:
        candidates.append(_normalize_checkout_line_endings(content))
    for candidate in candidates:
        header = f"blob {len(candidate)}\0".encode("ascii")
        if hashlib.new(algorithm, header + candidate).hexdigest() == expected:
            return
    raise RuntimeError(f"{archive} member {path!r} differs from the release commit")


def _validate_release_sources(  # noqa: PLR0913
    wheel: Path, sdist: Path, version: str, files: dict[str, tuple[str, bool]]
) -> None:
    """Bind source members in both archives to the named Git tree."""
    root = f"trackmaniarl-{version}"
    egg_info = "TrackmaniaRL.egg-info/"
    generated_sdist = {"PKG-INFO", "setup.cfg"} | {
        egg_info + name
        for name in (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        )
    }
    dist_info = f"trackmaniarl-{version}.dist-info/"
    generated_wheel = {
        dist_info + name
        for name in ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt")
    }
    wheel_licenses = {dist_info + "licenses/" + name: name for name in ("LICENSE", "NOTICE")}
    expected_modules = {
        path for path in files if path.startswith("trackmaniarl/") and path.endswith(".py")
    }
    sdist_sources: set[str] = set()
    with tarfile.open(sdist) as package:
        for member in package.getmembers():
            if not member.isfile():
                continue
            name = PurePosixPath(member.name.replace("\\", "/"))
            if len(name.parts) < 2 or name.parts[0] != root:
                raise RuntimeError(f"{sdist} contains a member outside its expected root")
            path = PurePosixPath(*name.parts[1:]).as_posix()
            sdist_sources.add(path)
            if path in generated_sdist and path not in files:
                continue
            stream = package.extractfile(member)
            if stream is None:
                raise RuntimeError(f"{sdist} cannot read member {member.name!r}")
            _validate_committed_member(path, stream.read(), files, sdist)
    _require_members(sdist_sources, expected_modules, sdist)
    wheel_sources: set[str] = set()
    with zipfile.ZipFile(wheel) as package:
        for member in package.infolist():
            if member.is_dir():
                continue
            path = member.filename.replace("\\", "/")
            wheel_sources.add(path)
            if path in generated_wheel:
                continue
            source = wheel_licenses.get(path, path)
            if source == path and not path.startswith("trackmaniarl/"):
                raise RuntimeError(f"{wheel} contains an unexpected member {path!r}")
            _validate_committed_member(source, package.read(member), files, wheel)
    _require_members(wheel_sources, expected_modules, wheel)


def _validate_wheel_record(wheel: Path, version: str) -> None:
    record = f"trackmaniarl-{version}.dist-info/RECORD"
    with zipfile.ZipFile(wheel) as package:
        names = {item.filename for item in package.infolist() if not item.is_dir()}
        try:
            rows = list(csv.reader(io.StringIO(package.read(record).decode("utf-8"), newline="")))
        except (KeyError, UnicodeDecodeError, csv.Error) as error:
            raise RuntimeError(f"{wheel} has an invalid RECORD") from error
        seen: set[str] = set()
        for row in rows:
            if len(row) != 3 or row[0] not in names or row[0] in seen:
                raise RuntimeError(f"{wheel} has an invalid RECORD")
            name, digest, size = row
            seen.add(name)
            if name == record:
                valid = not digest and not size
            else:
                content = package.read(name)
                sha = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
                valid = digest == "sha256=" + sha.decode("ascii") and size == str(len(content))
            if not valid:
                raise RuntimeError(f"{wheel} has an invalid RECORD")
        if seen != names:
            raise RuntimeError(f"{wheel} has an incomplete RECORD")


def _generated_metadata(archive: Path, version: str, *, wheel: bool) -> dict[str, bytes]:
    dist_info = f"trackmaniarl-{version}.dist-info/"
    names = {"METADATA", "WHEEL", "entry_points.txt", "top_level.txt"}
    if wheel:
        with zipfile.ZipFile(archive) as package:
            return {name: package.read(dist_info + name) for name in names}
    root = f"trackmaniarl-{version}/"
    egg = "TrackmaniaRL.egg-info/"
    names = {"PKG-INFO", "setup.cfg"} | {
        egg + name
        for name in (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        )
    }
    with tarfile.open(archive) as package:
        result: dict[str, bytes] = {}
        for name in names:
            stream = package.extractfile(root + name)
            if stream is None:
                raise RuntimeError(f"{archive} is missing generated metadata {name!r}")
            result[name] = stream.read()
        return result


def _validate_generated_metadata(  # noqa: PLR0913
    wheel: Path, sdist: Path, expected_wheel: Path, expected_sdist: Path, version: str
) -> None:
    for actual, expected, is_wheel in (
        (wheel, expected_wheel, True),
        (sdist, expected_sdist, False),
    ):
        try:
            actual_data = _generated_metadata(actual, version, wheel=is_wheel)
            expected_data = _generated_metadata(expected, version, wheel=is_wheel)
        except (KeyError, tarfile.TarError) as error:
            raise RuntimeError(f"{actual} is missing generated release metadata") from error
        for name, content in actual_data.items():
            if _normalize_checkout_line_endings(content) != _normalize_checkout_line_endings(
                expected_data[name]
            ):
                raise RuntimeError(f"{actual} generated metadata {name!r} differs from the commit")
    _validate_wheel_record(wheel, version)


def _verify_generated_metadata(  # noqa: PLR0913
    wheel: Path, sdist: Path, version: str, commit: str
) -> None:
    """Regenerate metadata from a clean export of the selected commit."""
    with tempfile.TemporaryDirectory(prefix="trackmaniarl-release-") as temporary:
        source = Path(temporary) / "source"
        source.mkdir()
        archive_path = Path(temporary) / "source.tar"
        try:
            with archive_path.open("wb") as output:
                subprocess.run(
                    ["git", "archive", "--format=tar", commit],
                    check=True,
                    stdout=output,
                    stderr=subprocess.PIPE,
                )
            with tarfile.open(archive_path) as archive:
                archive.extractall(source, filter="data")
            archive_path.unlink()
            output_dir = Path(temporary) / "expected-dist"
            subprocess.run(
                ["uv", "build", "--quiet", "--out-dir", str(output_dir)],
                cwd=source,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
            raise RuntimeError("Cannot regenerate metadata from the release commit") from error
        expected_wheel = output_dir / f"trackmaniarl-{version}-py3-none-any.whl"
        expected_sdist = output_dir / f"trackmaniarl-{version}.tar.gz"
        _validate_generated_metadata(wheel, sdist, expected_wheel, expected_sdist, version)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _load_sbom(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"Missing release SBOM: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{path} is not valid JSON") from error
    if not isinstance(document, dict) or document.get("spdxVersion") != "SPDX-2.3":
        raise RuntimeError(f"{path} is not an SPDX 2.3 JSON document")
    return cast(dict[str, object], document)


def _package_index(document: dict[str, object], path: Path) -> dict[str, tuple[str, str]]:
    packages = document.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError(f"{path} does not describe any packages")
    package_index: dict[str, tuple[str, str]] = {}
    for package_value in packages:
        if not isinstance(package_value, dict):
            continue
        package = cast(dict[str, object], package_value)
        identifier = package.get("SPDXID")
        name = package.get("name")
        package_version = package.get("versionInfo")
        if isinstance(identifier, str) and isinstance(name, str):
            package_index[identifier] = (
                _normalize_package_name(name),
                package_version if isinstance(package_version, str) else "",
            )
    return package_index


def _trackmania_ids(
    package_index: dict[str, tuple[str, str]], path: Path, version: str
) -> set[str]:
    trackmania_ids = {
        identifier
        for identifier, (name, package_version) in package_index.items()
        if name == "trackmaniarl" and package_version == version
    }
    if not trackmania_ids:
        raise RuntimeError(f"{path} does not describe TrackmaniaRL {version}")
    return trackmania_ids


def _relationship_dependencies(value: object, trackmania_ids: set[str]) -> set[str]:
    if not isinstance(value, dict):
        return set()
    relationship = cast(dict[str, object], value)
    source = relationship.get("spdxElementId")
    target = relationship.get("relatedSpdxElement")
    kind = relationship.get("relationshipType")
    if kind == "DEPENDENCY_OF" and target in trackmania_ids and isinstance(source, str):
        return {source}
    if kind == "DEPENDS_ON" and source in trackmania_ids and isinstance(target, str):
        return {target}
    return set()


def _dependency_ids(document: dict[str, object], trackmania_ids: set[str]) -> set[str]:
    relationships = document.get("relationships")
    dependency_ids: set[str] = set()
    if isinstance(relationships, list):
        for value in relationships:
            dependency_ids.update(_relationship_dependencies(value, trackmania_ids))
    return dependency_ids


def _validate_dependency_names(
    package_index: dict[str, tuple[str, str]], dependency_ids: set[str], path: Path
) -> None:
    dependency_names = {
        package_index[identifier][0] for identifier in dependency_ids if identifier in package_index
    }
    missing_dependencies = sorted(CORE_DEPENDENCIES - dependency_names)
    if missing_dependencies:
        raise RuntimeError(
            f"{path} has no TrackmaniaRL dependency relationships for: "
            f"{', '.join(missing_dependencies)}"
        )


def _validate_sbom(path: Path, version: str) -> None:
    document = _load_sbom(path)
    package_index = _package_index(document, path)
    trackmania_ids = _trackmania_ids(package_index, path, version)
    dependency_ids = _dependency_ids(document, trackmania_ids)
    _validate_dependency_names(package_index, dependency_ids, path)


def _release_subjects(version: str, directory: Path = Path("dist")) -> tuple[Path, Path, Path]:
    return (
        directory / f"trackmaniarl-{version}-py3-none-any.whl",
        directory / f"trackmaniarl-{version}.tar.gz",
        directory / SBOM_NAME,
    )


def _validate_archive_set(wheel: Path, sdist: Path) -> None:
    found = set(wheel.parent.glob("*.whl")) | set(wheel.parent.glob("*.tar.gz"))
    expected = {wheel, sdist}
    if found != expected:
        names = ", ".join(sorted(path.name for path in found ^ expected))
        raise RuntimeError(f"Release archive set differs from the expected pair: {names}")


def _write_checksums(subjects: tuple[Path, ...], target: Path) -> None:
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(subjects)]
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(target)


def _read_checksums(target: Path) -> dict[str, str]:
    if not target.is_file():
        raise RuntimeError(f"Missing release checksums: {target}")
    checksums: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or SHA256_PATTERN.fullmatch(digest) is None:
            raise RuntimeError(f"{target} has an invalid checksum entry")
        if Path(name).name != name or "/" in name or "\\" in name or name in checksums:
            raise RuntimeError(f"{target} has an unsafe or duplicate subject {name!r}")
        checksums[name] = digest
    return checksums


def _verify_checksums(subjects: tuple[Path, ...], target: Path) -> None:
    expected_names = {path.name for path in subjects}
    checksums = _read_checksums(target)
    if set(checksums) != expected_names:
        raise RuntimeError(f"{target} does not list exactly the release subjects")
    mismatches = [path.name for path in subjects if checksums[path.name] != _sha256(path)]
    if mismatches:
        raise RuntimeError(f"Checksum mismatch: {', '.join(sorted(mismatches))}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Release tag, expected in the form v<project-version>")
    parser.add_argument(
        "--release-commit", help="Require source members to match this full Git commit ID"
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    checksum_mode = parser.add_mutually_exclusive_group()
    checksum_mode.add_argument("--write-checksums", action="store_true")
    checksum_mode.add_argument("--verify-checksums", action="store_true")
    return parser.parse_args()


def _validate_release_tag(tag: str | None, version: str) -> None:
    if tag is not None and tag != f"v{version}":
        raise RuntimeError(f"Release tag {tag!r} does not match project version {version!r}")


def main() -> None:
    args = _parse_args()
    release_tree = _select_release_tree(
        args.release_commit,
        required=args.tag is not None or args.write_checksums or args.verify_checksums,
    )
    version = _project_version()
    _validate_release_tag(args.tag, version)
    wheel, sdist, sbom = _release_subjects(version, args.dist_dir)
    missing = [str(archive) for archive in (wheel, sdist) if not archive.is_file()]
    if missing:
        raise RuntimeError(f"Missing release archives: {', '.join(missing)}")
    _validate_archive_set(wheel, sdist)
    _validate_wheel(wheel, version)
    _validate_sdist(sdist, version)
    if release_tree is not None:
        commit, files = release_tree
        _validate_release_sources(wheel, sdist, version, files)
        _verify_generated_metadata(wheel, sdist, version, commit)
    if args.write_checksums or args.verify_checksums:
        _validate_sbom(sbom, version)
        subjects = (wheel, sdist, sbom)
        checksums = args.dist_dir / CHECKSUMS_NAME
        if args.write_checksums:
            _write_checksums(subjects, checksums)
        else:
            _verify_checksums(subjects, checksums)


if __name__ == "__main__":
    main()
