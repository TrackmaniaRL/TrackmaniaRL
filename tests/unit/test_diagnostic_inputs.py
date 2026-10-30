from pathlib import Path

import pytest

from scripts.diagnostic_inputs import configured_path
from scripts.headless_probe_paths import DiagnosticProbePaths


def test_missing_probe_configuration_fails_before_loading_any_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRACKMANIARL_COMPARISON_ROOT", raising=False)
    monkeypatch.delenv("TRACKMANIARL_PROBE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="explicit diagnostic input"):
        DiagnosticProbePaths.from_environment()


def test_explicit_existing_paths_are_retained_without_invented_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("caller-owned fixture, not a qualified runtime")
    monkeypatch.setenv("TRACKMANIARL_COMPARISON_ROOT", str(tmp_path))
    monkeypatch.setenv("TRACKMANIARL_PROBE_CONFIG", str(config))
    assert DiagnosticProbePaths.from_environment() == DiagnosticProbePaths(tmp_path, config)
    monkeypatch.setenv("TRACKMANIARL_NATIVE_CONFIG", str(config))
    assert configured_path("TRACKMANIARL_NATIVE_CONFIG") == config


def test_a_directory_cannot_substitute_for_the_explicit_run_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACKMANIARL_COMPARISON_ROOT", str(tmp_path))
    monkeypatch.setenv("TRACKMANIARL_PROBE_CONFIG", str(tmp_path))
    with pytest.raises(ValueError, match="RunSpec file"):
        DiagnosticProbePaths.from_environment()
