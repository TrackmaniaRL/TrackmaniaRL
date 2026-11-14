from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from trackmaniarl.observability import artifacts, provenance
from trackmaniarl.observability.provenance import capture_source_snapshot, validate_source_snapshot


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_snapshot_patch_excludes_private_files_and_tracks_deletions(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    files = {
        "safe[1].py": "value = 1  # SYNTHETIC_REMOVED_SECRET\n",
        "removed.py": "removed = True\n",
        "renamed.py": "renamed = True\n",
        "private.pem": "old\n",
        ".env.local.json": "old\n",
        "credentials.json": "old\n",
    }
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    _git(root, "add", "--all")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    (root / "safe[1].py").write_text("value = 2\n", encoding="utf-8")
    for name in ("private.pem", ".env.local.json", "credentials.json"):
        (root / name).write_text("SYNTHETIC_PRIVATE_MARKER\n", encoding="utf-8")
    _git(root, "rm", "removed.py")
    _git(root, "mv", "renamed.py", "new-name.py")

    result = capture_source_snapshot(tmp_path / "snapshots", root=root)
    assert validate_source_snapshot(Path(result["archive_path"]).parent / "manifest.json")
    patch = Path(result["patch_path"]).read_bytes()
    assert patch == b""
    with zipfile.ZipFile(result["archive_path"]) as archive:
        assert set(archive.namelist()) == {"safe[1].py", "new-name.py"}
        assert archive.read("safe[1].py") == (root / "safe[1].py").read_bytes()
    assert any(item["path"] == "removed.py" and item.get("deleted") for item in result["files"])


def test_unversioned_snapshot_excludes_explicitly_private_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    credential = b"wandb_" + b"v1_" + b"a" * 20
    (root / ".env.production.json").write_bytes(credential)
    (root / "secrets.yaml").write_bytes(credential)
    (root / "credentials").mkdir()
    (root / "credentials" / "settings.json").write_bytes(credential)
    result = capture_source_snapshot(tmp_path / "snapshots", root=root)
    with zipfile.ZipFile(result["archive_path"]) as archive:
        assert archive.namelist() == ["main.py"]


@pytest.mark.parametrize(
    "name", ["run.yaml", "settings.json", "pyproject.toml", "main.py", "notes.md"]
)
@pytest.mark.parametrize(
    "case",
    [
        ("credential token", b"wandb_" + b"v1_" + b"a" * 20),
        ("private key", b"-----BEGIN PRIVATE KEY-----"),
        ("URL credentials", b"https://account:password@example.invalid/"),
    ],
)
def test_snapshot_rejects_known_private_content_without_echoing_it(
    tmp_path: Path, name: str, case: tuple[str, bytes]
) -> None:
    category, content = case
    root = tmp_path / "project"
    root.mkdir()
    path = root / name
    path.write_bytes(content)

    with pytest.raises(ValueError, match=category) as failure:
        capture_source_snapshot(tmp_path / "snapshots", root=root)

    assert content.decode() not in str(failure.value)
    assert path.name not in str(failure.value)


@pytest.mark.parametrize(
    "content",
    [
        b"WANDB_API_KEY=\nnormal configuration\n",
        b"WANDB_API_KEY=${WANDB_API_KEY}\nnormal configuration\n",
        b"GH_TOKEN: ${{ github.token }}\nGITHUB_TOKEN: ${{ secrets.RELEASE_TOKEN }}\n",
        b"id-token: write\n",
    ],
)
def test_snapshot_keeps_safe_content_byte_exact(tmp_path: Path, content: bytes) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "run.yaml").write_bytes(content)

    result = capture_source_snapshot(tmp_path / "snapshots", root=root)

    with zipfile.ZipFile(result["archive_path"]) as archive:
        assert archive.read("run.yaml") == content


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("run.yaml", b"WANDB_API_KEY=" + b"example-value" * 4 + b"\n"),
        (
            "settings.json",
            b'{"wandb_api_key": "' + b"example-value" * 4 + b'"}',
        ),
    ],
)
def test_snapshot_rejects_populated_config_credential_without_echoing_value(
    tmp_path: Path, name: str, content: bytes
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    path = root / name
    path.write_bytes(content)

    with pytest.raises(ValueError, match="credential assignment") as failure:
        capture_source_snapshot(tmp_path / "snapshots", root=root)

    assert content.decode() not in str(failure.value)
    assert path.name not in str(failure.value)


def test_snapshot_rejects_private_filename_before_omitted_metadata(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    private_directory = root / "secrets"
    private_directory.mkdir()
    token = "ghp_" + "a" * 36
    (private_directory / f"{token}.txt").write_text("safe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="credential token") as failure:
        capture_source_snapshot(tmp_path / "snapshots", root=root)

    assert token not in str(failure.value)
    assert not (tmp_path / "snapshots").exists()


def test_explicit_project_root_excludes_siblings_in_a_git_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    project = repo / "project"
    sibling = repo / "sibling"
    project.mkdir(parents=True)
    sibling.mkdir()
    _git(repo, "init", "-q")
    (project / "main.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "--all")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    (project / "main.py").write_text("value = 2\n", encoding="utf-8")
    (sibling / "notes.txt").write_text("SYNTHETIC_SIBLING_DATA", encoding="utf-8")
    result = capture_source_snapshot(tmp_path / "snapshots", root=project)
    with zipfile.ZipFile(result["archive_path"]) as archive:
        assert archive.namelist() == ["project/main.py"]
        assert archive.read("project/main.py") == (project / "main.py").read_bytes()
    assert Path(result["patch_path"]).read_bytes() == b""
    assert validate_source_snapshot(Path(result["archive_path"]).parent / "manifest.json")


def test_snapshot_omits_diff_for_file_directory_replacements(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "file.py").write_text("# SYNTHETIC_REMOVED_SECRET\n", encoding="utf-8")
    (root / "directory.py").mkdir()
    (root / "directory.py" / "old.py").write_text("old = True\n", encoding="utf-8")
    _git(root, "add", "--all")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    (root / "file.py").unlink()
    (root / "file.py").mkdir()
    (root / "file.py" / "safe.py").write_text("value = 2\n", encoding="utf-8")
    (root / "file.py" / "credentials.json").write_text("SYNTHETIC_PRIVATE_MARKER", encoding="utf-8")
    (root / "directory.py" / "old.py").unlink()
    (root / "directory.py").rmdir()
    (root / "directory.py").write_text("value = 3\n", encoding="utf-8")
    _git(root, "add", "--all")
    result = capture_source_snapshot(tmp_path / "snapshots", root=root)
    assert Path(result["patch_path"]).read_bytes() == b""
    with zipfile.ZipFile(result["archive_path"]) as archive:
        assert set(archive.namelist()) == {"file.py/safe.py", "directory.py"}
        assert all(b"SYNTHETIC_" not in archive.read(name) for name in archive.namelist())


def test_default_library_scope_and_project_snapshot_in_same_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    package = repo / "library" / "trackmaniarl"
    project = repo / "project"
    sibling = repo / "sibling"
    package.mkdir(parents=True)
    project.mkdir()
    sibling.mkdir()
    _git(repo, "init", "-q")
    (package / "core.py").write_text("value = 1\n", encoding="utf-8")
    (project / "main.py").write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "--all")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    (sibling / "notes.txt").write_text("SYNTHETIC_SIBLING_DATA", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(package / "observability" / "provenance.py"))
    monkeypatch.setattr(artifacts, "_environment_snapshot", lambda _run: {"environment": {}})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run = SimpleNamespace(base_dir=project, run_dir=run_dir)

    artifacts._append_environment_snapshot(run)
    attempt = json.loads((run_dir / "manifest-attempts.jsonl").read_text(encoding="utf-8"))
    source = attempt["source_provenance"]
    assert Path(source["source_root"]) == repo
    assert Path(source["scope_root"]) == package
    with zipfile.ZipFile(source["archive_path"]) as archive:
        assert archive.namelist() == ["library/trackmaniarl/core.py"]
    project_snapshot = attempt["project_provenance"]
    assert Path(project_snapshot["scope_root"]) == project
    with zipfile.ZipFile(project_snapshot["archive_path"]) as archive:
        assert archive.namelist() == ["project/main.py"]


def test_failed_snapshot_does_not_log_private_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "library" / "trackmaniarl"
    observability = package / "observability"
    observability.mkdir(parents=True)
    token = "ghp_" + "a" * 36
    (package / f"{token}.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "__file__", str(observability / "provenance.py"))
    monkeypatch.setattr(artifacts, "_environment_snapshot", lambda _run: {"environment": {}})
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    artifacts._append_environment_snapshot(SimpleNamespace(base_dir=package, run_dir=run_dir))

    attempt_path = run_dir / "manifest-attempts.jsonl"
    contents = attempt_path.read_text(encoding="utf-8")
    attempt = json.loads(contents)
    assert token not in contents
    assert attempt["source_provenance"] == {
        "status": "unavailable",
        "reason": "source snapshot unavailable",
    }


def test_snapshot_rejects_deleted_source_reappearing_during_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "main.py").write_text("value = 1\n", encoding="utf-8")
    (root / "removed.py").write_text("value = 2\n", encoding="utf-8")
    _git(root, "add", "--all")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    _git(root, "rm", "removed.py")
    original = provenance._collect_sources

    def restore_after_collection(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        (root / "removed.py").write_text("value = 2\n", encoding="utf-8")
        return result

    monkeypatch.setattr(provenance, "_collect_sources", restore_after_collection)
    with pytest.raises(RuntimeError, match="Deleted source reappeared"):
        capture_source_snapshot(tmp_path / "snapshots", root=root)


def test_reused_snapshot_returns_the_current_source_and_scope_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "main.py").write_text("value = 1\n", encoding="utf-8")

    initial = capture_source_snapshot(tmp_path / "snapshots", root=first)
    reused = capture_source_snapshot(tmp_path / "snapshots", root=second)
    assert reused["snapshot_id"] == initial["snapshot_id"]
    assert Path(reused["source_root"]) == second
    assert Path(reused["scope_root"]) == second
    assert validate_source_snapshot(Path(reused["archive_path"]).parent / "manifest.json")


def test_snapshot_rejects_valid_but_substituted_cache_entry(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    substituted = tmp_path / "substituted"
    for root, content in ((expected, "value = 1\n"), (substituted, "value = 2\n")):
        root.mkdir()
        (root / "main.py").write_text(content, encoding="utf-8")
    snapshots = tmp_path / "snapshots"
    expected_result = capture_source_snapshot(snapshots, root=expected)
    substituted_result = capture_source_snapshot(snapshots, root=substituted)
    expected_directory = Path(expected_result["archive_path"]).parent
    substituted_directory = Path(substituted_result["archive_path"]).parent

    for name in ("source.zip", "working-tree.patch", "manifest.json"):
        (expected_directory / name).write_bytes((substituted_directory / name).read_bytes())

    with pytest.raises(ValueError, match="cache identity conflicts"):
        capture_source_snapshot(snapshots, root=expected)
