from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.distribution_privacy import validate_archive_privacy, validate_member


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.production",
        "a/.aws/config",
        "x/model.pt",
        "x/session.npz",
        "x/__pycache__/a.pyc",
        "../outside.txt",
        "native/libforkshim.so",
        "native/TrackmaniaServer.exe",
        "native/example.Map.Gbx",
        "native/dedicated.pak",
        "native/captured-state.bin",
    ],
)
def test_distribution_rejects_private_or_unsafe_members(name: str) -> None:
    with pytest.raises(RuntimeError):
        validate_member(name, b"")


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_archives_scan_metadata_and_nonpackage_files(tmp_path: Path, kind: str) -> None:
    # Construct a fake value so the fixture itself never contains a complete credential.
    value = b"ghp_" + b"a" * 36
    archive = tmp_path / kind
    name = "package-1.0.dist-info/METADATA"
    if kind == "wheel":
        with zipfile.ZipFile(archive, "w") as wheel:
            wheel.writestr(name, value)
    else:
        with tarfile.open(archive, "w:gz") as sdist:
            member = tarfile.TarInfo(name)
            member.size = len(value)
            sdist.addfile(member, io.BytesIO(value))
    with pytest.raises(RuntimeError, match="credential token") as failure:
        validate_archive_privacy(archive)
    assert value.decode() not in str(failure.value)


def test_distribution_rejects_personal_paths_without_echoing_them() -> None:
    private = b"C:" + b"/Users/" + b"example-account/private"
    with pytest.raises(RuntimeError, match="personal home path") as failure:
        validate_member("guide.md", private)
    assert "example-account" not in str(failure.value)
    validate_member(".env-example", b"WANDB_API_KEY=\n")


def test_distribution_rejects_archive_links(tmp_path: Path) -> None:
    archive = tmp_path / "linked.tar.gz"
    with tarfile.open(archive, "w:gz") as sdist:
        member = tarfile.TarInfo("package/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        sdist.addfile(member)
    with pytest.raises(RuntimeError, match="links"):
        validate_archive_privacy(archive)


@pytest.mark.parametrize("payload", [b"\x7fELF" + bytes(64), b"MZ" + bytes(64), b"GBX" + bytes(64)])
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_source_archives_reject_disguised_runtime_payloads(
    tmp_path: Path, payload: bytes, kind: str
) -> None:
    archive = tmp_path / kind
    name = "package/native/innocent.txt"
    if kind == "wheel":
        with zipfile.ZipFile(archive, "w") as wheel:
            wheel.writestr(name, payload)
    else:
        with tarfile.open(archive, "w:gz") as sdist:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            sdist.addfile(member, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="Executable or game payload"):
        validate_archive_privacy(archive)
