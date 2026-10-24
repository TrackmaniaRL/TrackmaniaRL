from pathlib import Path

import pytest

from trackmaniarl.experiments.full_qr.campaign_storage import campaign_storage_reserves


def test_output_reserves_follow_the_explicit_report_path_on_any_volume(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    directory.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    reserves = campaign_storage_reserves(directory, reports / "new-run/artifacts")
    assert reserves[0].minimum_free_bytes == 10 * 1024**3
    assert reserves[1].path == directory
    assert reserves[2].path == reports
    assert all(reserve.minimum_free_bytes == 4 * 1024**3 for reserve in reserves[1:])
    assert not (reports / "new-run").exists()


def test_noncanonical_output_cannot_redirect_the_report_reserve(tmp_path: Path) -> None:
    aliased = tmp_path / "unused/../reports"
    with pytest.raises(ValueError, match="canonical absolute"):
        campaign_storage_reserves(tmp_path, aliased)
