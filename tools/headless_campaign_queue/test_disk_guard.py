from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.headless_campaign_queue import disk_guard


def spec_for(base: Path) -> SimpleNamespace:
    config = {
        "runtime": {"trace_dir": str(base / "trace" / "future"), "work_dir": str(base / "work")},
        "trace_archive_root": str(base / "archives"),
    }
    return SimpleNamespace(
        artifacts_dir="artifacts",
        checkpoint_root="checkpoints",
        components=SimpleNamespace(environment=SimpleNamespace(kwargs={"config": config})),
    )


def test_checks_every_location_using_real_free_space_and_no_fixed_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("artifacts", "checkpoints", "trace", "work", "archives", "system"):
        (tmp_path / name).mkdir()
    checked = []

    def usage(path: Path) -> SimpleNamespace:
        checked.append(path)
        return SimpleNamespace(free=11 * disk_guard.GIB)

    monkeypatch.setattr(disk_guard.shutil, "disk_usage", usage)
    receipt = disk_guard.check_disk_space(
        spec_for(tmp_path), tmp_path, system_path=tmp_path / "system"
    )
    assert [x["name"] for x in receipt] == [
        "system",
        "artifacts",
        "checkpoints",
        "native_traces",
        "native_work",
        "native_archives",
    ]
    assert checked == [
        tmp_path / n for n in ("system", "artifacts", "checkpoints", "trace", "work", "archives")
    ]
    assert not (tmp_path / "trace" / "future").exists()
    assert receipt[0]["reserve_bytes"] == 10 * disk_guard.GIB
    assert all(x["reserve_bytes"] == 4 * disk_guard.GIB for x in receipt[1:])


@pytest.mark.parametrize(
    "target", ["system", "artifacts", "checkpoints", "trace", "work", "archives"]
)
def test_below_reserve_stops_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    for name in ("artifacts", "checkpoints", "trace", "work", "archives", "system"):
        (tmp_path / name).mkdir()

    def usage(path: Path) -> SimpleNamespace:
        return SimpleNamespace(free=(0 if path.name == target else 11 * disk_guard.GIB))

    monkeypatch.setattr(disk_guard.shutil, "disk_usage", usage)
    with pytest.raises(OSError, match="insufficient free disk space"):
        disk_guard.check_disk_space(spec_for(tmp_path), tmp_path, system_path=tmp_path / "system")


def test_reserve_boundary_is_inclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    system = tmp_path / "system"
    system.mkdir()
    monkeypatch.setattr(
        disk_guard.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=(10 if path == system else 4) * disk_guard.GIB),
    )
    assert len(disk_guard.check_disk_space(spec_for(tmp_path), tmp_path, system_path=system)) == 6


def test_missing_native_config_is_not_treated_as_valid(tmp_path: Path) -> None:
    spec = spec_for(tmp_path)
    del spec.components.environment.kwargs["config"]["runtime"]["work_dir"]
    with pytest.raises(ValueError, match="native output paths"):
        disk_guard.check_disk_space(spec, tmp_path, system_path=tmp_path)


def test_schema_without_checkpoint_root_uses_artifact_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "artifacts").mkdir()
    spec = spec_for(tmp_path)
    del spec.checkpoint_root
    monkeypatch.setattr(
        disk_guard.shutil, "disk_usage", lambda _: SimpleNamespace(free=11 * disk_guard.GIB)
    )
    receipt = disk_guard.check_disk_space(spec, tmp_path, system_path=tmp_path)
    checkpoints = next(row for row in receipt if row["name"] == "checkpoints")
    assert Path(checkpoints["checked_at"]) == tmp_path / "artifacts"


def test_missing_windows_mount_cannot_fall_back_to_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disk_guard.os.path, "ismount", lambda _: False)
    with pytest.raises(OSError, match="not mounted"):
        disk_guard._existing_directory(Path("/mnt/f/some/output"))
