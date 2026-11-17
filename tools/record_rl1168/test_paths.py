"""Campaign storage environment compatibility."""

from tools.record_rl1168.paths import private_base


def test_private_base_prefers_new_name_and_accepts_legacy(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    legacy.mkdir()
    current.mkdir()
    monkeypatch.setenv("TMRL_CAMPAIGN_PRIVATE_BASE", str(legacy))
    monkeypatch.delenv("TRACKMANIARL_CAMPAIGN_PRIVATE_BASE", raising=False)
    assert private_base() == legacy.resolve()
    monkeypatch.setenv("TRACKMANIARL_CAMPAIGN_PRIVATE_BASE", str(current))
    assert private_base() == current.resolve()
