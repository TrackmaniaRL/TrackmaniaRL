from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from trackmaniarl.trackmania.map_staging import StagedAssets


@pytest.mark.skipif(os.name != "nt", reason="Windows share modes are required")
def test_staging_blocks_other_process_writes_and_deletes_until_release(tmp_path: Path) -> None:
    source = tmp_path / "source.Map.Gbx"
    source.write_bytes(b"immutable test map")
    stage = StagedAssets(tmp_path)
    copied = stage.stage(source, "map.Map.Gbx")
    assert copied.read_bytes() == b"immutable test map"
    program = "import sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_bytes(b'changed')"
    attempted = subprocess.run(
        [sys.executable, "-c", program, str(copied)], capture_output=True, check=False
    )
    assert attempted.returncode != 0
    with pytest.raises(PermissionError):
        copied.unlink()
    with pytest.raises(PermissionError):
        stage.directory.rename(tmp_path / "renamed")
    assert copied.read_bytes() == b"immutable test map"
    stage.close()
    assert copied.read_bytes() == b"immutable test map"


def test_non_windows_is_explicitly_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trackmaniarl.trackmania import map_staging

    monkeypatch.setattr(map_staging.os, "name", "posix")
    with pytest.raises(RuntimeError, match="requires Windows"):
        StagedAssets(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows share modes are required")
def test_staged_suite_must_match_assigned_digest_even_if_source_changes(tmp_path: Path) -> None:
    from trackmaniarl.core.spec import EvaluationMapSpec, EvaluationSuiteSpec
    from trackmaniarl.distributed.evaluation_assets import evaluation_assets_digest

    map_path = tmp_path / "source.Map.Gbx"
    reward = tmp_path / "reward.npz"
    map_path.write_bytes(b"original")
    reward.write_bytes(b"asset")
    suite = EvaluationSuiteSpec(
        name="test",
        version="1",
        maps=(
            EvaluationMapSpec(
                id="a", expected_map_uid="uid", map_path=map_path, reward_points_path=reward
            ),
        ),
    )
    digest = evaluation_assets_digest(suite, tmp_path)
    map_path.write_bytes(b"substituted")
    with (
        StagedAssets(tmp_path) as stage,
        pytest.raises(RuntimeError, match="differ from the assigned suite"),
    ):
        stage.stage_suite(suite, tmp_path, digest)
    map_path.write_bytes(b"original")
    with StagedAssets(tmp_path) as stage:
        staged = stage.stage_suite(suite, tmp_path, digest)
        map_path.write_bytes(b"later modification")
        assert evaluation_assets_digest(staged, tmp_path) == digest
        with pytest.raises(PermissionError):
            staged.maps[0].map_path.write_bytes(b"changed")


def test_cleanup_releases_other_resources_and_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from trackmaniarl.trackmania import map_staging

    class Handle:
        def __init__(self, *, fails: bool) -> None:
            self.closed = False
            self.fails = fails

        def close(self) -> None:
            if self.fails:
                self.fails = False
                raise OSError("simulated close failure")
            self.closed = True

    stage = object.__new__(StagedAssets)
    failing, good = Handle(fails=True), Handle(fails=False)
    stage._closed = False
    stage._files = [(Path("one"), failing), (Path("two"), good)]
    stage._directory_handles = [1, 2]
    closed = []
    monkeypatch.setattr(map_staging, "_close_handle", closed.append)
    with pytest.raises(OSError, match="simulated close failure"):
        stage.close()
    assert good.closed
    assert closed == [2, 1]
    assert not stage._closed
    stage.close()
    assert failing.closed
    assert stage._closed


@pytest.mark.skipif(os.name != "nt", reason="Windows share modes are required")
def test_retained_directory_and_byte_limits_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"0123456789")
    with (
        StagedAssets(tmp_path, max_directories=1, max_bytes=4) as stage,
        pytest.raises(RuntimeError, match="byte limit"),
    ):
        stage.stage(source, "map.gbx")
    with pytest.raises(RuntimeError, match="directory limit"):
        StagedAssets(tmp_path, max_directories=1)


@pytest.mark.skipif(os.name != "nt", reason="Windows share modes are required")
def test_concurrent_staging_cannot_exceed_directory_limit(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    start = Barrier(2)

    def create() -> bool:
        start.wait(timeout=5)
        try:
            with StagedAssets(tmp_path, max_directories=1):
                return True
        except (RuntimeError, OSError):
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: create(), range(2)))
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 1
