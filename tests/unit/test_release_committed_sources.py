from __future__ import annotations

import base64
import csv
import hashlib
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import check_distribution


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def committed_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, dict[str, tuple[str, bool]]]:
    monkeypatch.chdir(tmp_path)
    _git("init", "-q")
    _git("config", "user.name", "Release Test")
    _git("config", "user.email", "release-test@example.invalid")
    Path("trackmaniarl").mkdir()
    Path("trackmaniarl/module.py").write_bytes(b"committed\n")
    Path("LICENSE").write_bytes(b"license\n")
    Path("NOTICE").write_bytes(b"notice\n")
    Path(".gitattributes").write_bytes(b"/trackmaniarl/frozen.txt -text\n")
    Path("trackmaniarl/frozen.txt").write_bytes(b"frozen\n")
    Path("trackmaniarl/binary.bin").write_bytes(b"binary\x00\n")
    Path(".gitignore").write_bytes(b"ignored.py\n")
    _git("add", ".")
    _git("commit", "-qm", "release test")
    commit = _git("rev-parse", "HEAD")
    resolved, files = check_distribution._release_tree(commit)
    assert resolved == commit
    return commit, files


def _archives(  # noqa: PLR0913
    tmp_path: Path,
    *,
    source: bytes = b"committed\n",
    extra_sdist: dict[str, bytes] | None = None,
    extra_wheel: dict[str, bytes] | None = None,
) -> tuple[Path, Path]:
    wheel, sdist = tmp_path / "test.whl", tmp_path / "test.tar.gz"
    sources = {"trackmaniarl/module.py": source, **(extra_sdist or {})}
    with tarfile.open(sdist, "w:gz") as archive:
        for path, value in sources.items():
            member = tarfile.TarInfo(f"trackmaniarl-2.0.0/{path}")
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("trackmaniarl/module.py", source)
        for path, value in (extra_wheel or {}).items():
            archive.writestr(path, value)
    return wheel, sdist


def test_release_accepts_committed_sources_and_exact_generated_metadata(
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]], tmp_path: Path
) -> None:
    _, files = committed_checkout
    wheel, sdist = _archives(
        tmp_path,
        extra_sdist={"PKG-INFO": b"metadata", "TrackmaniaRL.egg-info/SOURCES.txt": b"generated"},
        extra_wheel={
            "trackmaniarl-2.0.0.dist-info/METADATA": b"metadata",
            "trackmaniarl-2.0.0.dist-info/licenses/LICENSE": b"license\n",
        },
    )
    check_distribution._validate_release_sources(wheel, sdist, "2.0.0", files)


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_release_rejects_untracked_source_even_when_present_in_checkout(
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]], tmp_path: Path, archive_kind: str
) -> None:
    _, files = committed_checkout
    Path("trackmaniarl/private.py").write_bytes(b"untracked\n")
    kwargs = {"extra_" + archive_kind: {"trackmaniarl/private.py": b"untracked\n"}}
    wheel, sdist = _archives(tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match="not a committed regular file"):
        check_distribution._validate_release_sources(wheel, sdist, "2.0.0", files)


def test_release_rejects_modified_source_even_when_it_matches_checkout(
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]], tmp_path: Path
) -> None:
    _, files = committed_checkout
    Path("trackmaniarl/module.py").write_bytes(b"modified\n")
    wheel, sdist = _archives(tmp_path, source=b"modified\n")
    with pytest.raises(RuntimeError, match="differs from the release commit"):
        check_distribution._validate_release_sources(wheel, sdist, "2.0.0", files)


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("sdist", "Other.egg-info/private.txt"),
        ("sdist", "TrackmaniaRL.egg-info/private.txt"),
        ("wheel", "trackmaniarl-2.0.0.dist-info/private.txt"),
        ("wheel", "outside/private.txt"),
    ],
)
def test_release_does_not_exempt_arbitrary_metadata_or_other_wheel_members(  # noqa: PLR0913
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]],
    tmp_path: Path,
    kind: str,
    path: str,
) -> None:
    _, files = committed_checkout
    wheel, sdist = _archives(tmp_path, **{"extra_" + kind: {path: b"private"}})
    with pytest.raises(RuntimeError):
        check_distribution._validate_release_sources(wheel, sdist, "2.0.0", files)


def test_release_commit_requires_an_exact_existing_commit(
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit, _ = committed_checkout
    with pytest.raises(RuntimeError, match="full Git object ID"):
        check_distribution._release_tree(commit[:8])
    with pytest.raises(RuntimeError, match="Cannot read"):
        check_distribution._release_tree("0" * len(commit))
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    with pytest.raises(RuntimeError, match="repository root"):
        check_distribution._release_tree(commit)


def test_default_in_git_binds_sources_to_head(
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]], tmp_path: Path
) -> None:
    commit, _ = committed_checkout
    selected = check_distribution._select_release_tree(None, required=False)
    assert selected is not None
    assert selected[0] == commit
    Path("trackmaniarl/module.py").write_bytes(b"modified\n")
    wheel, sdist = _archives(tmp_path, source=b"modified\n")
    with pytest.raises(RuntimeError, match="differs from the release commit"):
        check_distribution._validate_release_sources(wheel, sdist, "2.0.0", selected[1])


def test_outside_git_is_warned_and_release_mode_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert check_distribution._select_release_tree(None, required=False) is None
    assert "not verified against a commit" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="requires a Git checkout"):
        check_distribution._select_release_tree(None, required=True)
    with pytest.raises(RuntimeError, match="Cannot read"):
        check_distribution._select_release_tree("0" * 40, required=True)


def test_release_accepts_windows_text_line_endings_but_not_binary_changes(
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]], tmp_path: Path
) -> None:
    _, files = committed_checkout
    Path("trackmaniarl/module.py").write_bytes(b"committed\r\n")
    wheel, sdist = _archives(tmp_path, source=b"committed\r\n")
    check_distribution._validate_release_sources(wheel, sdist, "2.0.0", files)
    for path, changed in (
        ("trackmaniarl/frozen.txt", b"frozen\r\n"),
        ("trackmaniarl/binary.bin", b"binary\x00\r\n"),
    ):
        with pytest.raises(RuntimeError, match="differs from the release commit"):
            check_distribution._validate_committed_member(path, changed, files, wheel)


def test_release_rejects_ignored_source(
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]], tmp_path: Path
) -> None:
    _, files = committed_checkout
    Path("ignored.py").write_bytes(b"ignored\n")
    wheel, sdist = _archives(tmp_path, extra_sdist={"ignored.py": b"ignored\n"})
    with pytest.raises(RuntimeError, match="not a committed regular file"):
        check_distribution._validate_release_sources(wheel, sdist, "2.0.0", files)


@pytest.mark.parametrize("flag", ["--tag", "--write-checksums", "--verify-checksums"])
def test_release_flags_fail_closed_outside_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.chdir(tmp_path)
    args = ["check_distribution.py", flag]
    if flag == "--tag":
        args.append("v2.0.0")
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(RuntimeError, match="requires a Git checkout"):
        check_distribution.main()


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_release_rejects_missing_committed_module(
    committed_checkout: tuple[str, dict[str, tuple[str, bool]]], tmp_path: Path, kind: str
) -> None:
    _, files = committed_checkout
    Path("trackmaniarl/module.py").unlink()
    wheel, sdist = _archives(tmp_path)
    if kind == "wheel":
        with zipfile.ZipFile(wheel, "w"):
            pass
    else:
        with tarfile.open(sdist, "w:gz"):
            pass
    with pytest.raises(RuntimeError, match=r"is missing: trackmaniarl/module\.py"):
        check_distribution._validate_release_sources(wheel, sdist, "2.0.0", files)


def _metadata_archives(tmp_path: Path, changed: str | None = None) -> tuple[Path, Path]:
    tmp_path.mkdir()
    root = "trackmaniarl-2.0.0/"
    dist = "trackmaniarl-2.0.0.dist-info/"
    sdist_data = {
        "PKG-INFO": b"metadata\n",
        "setup.cfg": b"[egg_info]\n",
        **{
            "TrackmaniaRL.egg-info/" + name: b"generated\n"
            for name in (
                "PKG-INFO",
                "SOURCES.txt",
                "dependency_links.txt",
                "entry_points.txt",
                "requires.txt",
                "top_level.txt",
            )
        },
    }
    wheel_data = {
        dist + name: b"generated\n"
        for name in ("METADATA", "WHEEL", "entry_points.txt", "top_level.txt")
    }
    if changed and changed.startswith("sdist:"):
        sdist_data[changed.removeprefix("sdist:")] = b"unapproved\n"
    if changed and changed.startswith("wheel:"):
        wheel_data[dist + changed.removeprefix("wheel:")] = b"unapproved\n"
    wheel, sdist = tmp_path / "test.whl", tmp_path / "test.tar.gz"
    with tarfile.open(sdist, "w:gz") as package:
        for name, content in sdist_data.items():
            member = tarfile.TarInfo(root + name)
            member.size = len(content)
            package.addfile(member, io.BytesIO(content))
    record = dist + "RECORD"
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for name, content in wheel_data.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
        writer.writerow([name, "sha256=" + digest.decode("ascii"), str(len(content))])
    writer.writerow([record, "", ""])
    with zipfile.ZipFile(wheel, "w") as package:
        for name, content in wheel_data.items():
            package.writestr(name, content)
        package.writestr(record, record_buffer.getvalue())
    return wheel, sdist


@pytest.mark.parametrize("changed", ["sdist:setup.cfg", "wheel:METADATA", "wheel:entry_points.txt"])
def test_release_rejects_modified_generated_metadata(tmp_path: Path, changed: str) -> None:
    expected_wheel, expected_sdist = _metadata_archives(tmp_path / "expected")
    wheel, sdist = _metadata_archives(tmp_path / "actual", changed)
    with pytest.raises(RuntimeError, match=r"generated metadata .* differs from the commit"):
        check_distribution._validate_generated_metadata(
            wheel, sdist, expected_wheel, expected_sdist, "2.0.0"
        )


def test_release_accepts_clean_generated_metadata_and_record(tmp_path: Path) -> None:
    wheel, sdist = _metadata_archives(tmp_path / "actual")
    expected_wheel, expected_sdist = _metadata_archives(tmp_path / "expected")
    check_distribution._validate_generated_metadata(
        wheel, sdist, expected_wheel, expected_sdist, "2.0.0"
    )
