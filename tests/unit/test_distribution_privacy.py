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
    ],
)
def test_distribution_rejects_private_or_unsafe_members(name: str) -> None:
    with pytest.raises(RuntimeError):
        validate_member(name, b"")


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("credential", ["github", "wandb"])
def test_archives_scan_metadata_and_nonpackage_files(
    tmp_path: Path, kind: str, credential: str
) -> None:
    # Construct a fake value so the fixture itself never contains a complete credential.
    value = b"ghp_" + b"a" * 36 if credential == "github" else b"wandb_" + b"v1_" + b"a" * 77
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


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("alias", ["root/./setup.cfg", "root//setup.cfg"])
def test_archives_reject_noncanonical_paths(tmp_path: Path, kind: str, alias: str) -> None:
    archive = tmp_path / kind
    if kind == "wheel":
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("root/setup.cfg", b"")
            package.writestr(alias, b"")
    else:
        with tarfile.open(archive, "w:gz") as package:
            for name in ("root/setup.cfg", alias):
                package.addfile(tarfile.TarInfo(name), io.BytesIO())
    with pytest.raises(RuntimeError, match="Noncanonical"):
        validate_archive_privacy(archive)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_archives_reject_file_directory_collision(tmp_path: Path, kind: str) -> None:
    archive = tmp_path / kind
    if kind == "wheel":
        with zipfile.ZipFile(archive, "w") as package:
            package.writestr("root/name", b"")
            package.writestr("root/name/", b"")
    else:
        with tarfile.open(archive, "w:gz") as package:
            package.addfile(tarfile.TarInfo("root/name"), io.BytesIO())
            directory = tarfile.TarInfo("root/name")
            directory.type = tarfile.DIRTYPE
            package.addfile(directory)
    with pytest.raises(RuntimeError, match="Duplicate"):
        validate_archive_privacy(archive)


@pytest.mark.parametrize("names", [("root/a", "root/a/b"), ("root/a/b", "root/a")])
def test_archives_reject_implicit_file_directory_collision(
    tmp_path: Path, names: tuple[str, str]
) -> None:
    archive = tmp_path / "collision.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        for name in names:
            package.addfile(tarfile.TarInfo(name), io.BytesIO())
    with pytest.raises(RuntimeError, match="collide"):
        validate_archive_privacy(archive)
