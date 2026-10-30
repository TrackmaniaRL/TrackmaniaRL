"""The committed protocol fixtures must be reproducible without game assets."""

import importlib.util
from pathlib import Path

import pytest

from tests.unit.models._synthetic_terminal_fixtures import write_fixtures

ROOT = Path(__file__).parents[2] / "fixtures/full_qr"


def _compare(generated: Path, committed: Path) -> None:
    actual = {path.relative_to(generated) for path in generated.rglob("*") if path.is_file()}
    expected = {path.relative_to(committed) for path in committed.rglob("*") if path.is_file()}
    assert actual == expected
    for path in actual:
        assert (generated / path).read_bytes() == (committed / path).read_bytes()


def test_contact_fixtures_reproduce_without_original_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "generate_synthetic_contacts.py"
    spec = importlib.util.spec_from_file_location("synthetic_contact_generator", script)
    assert spec is not None
    assert spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert not list(tmp_path.iterdir())  # Import has no filesystem side effects.
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    names = (
        "contact_delivery361",
        "prepared_contacts_v1",
        "prepared_impact_hook_v1",
        "wheel_contact_sources_v1",
        "wheel_contacts_v1",
    )
    for name in names:
        (tmp_path / name).mkdir()
    generator.main()
    for name in names:
        _compare(tmp_path / name, ROOT / name)


def test_terminal_fixtures_reproduce_without_original_captures(tmp_path: Path) -> None:
    write_fixtures(tmp_path)
    for directory in tmp_path.iterdir():
        assert directory.is_dir()
        _compare(directory, ROOT / directory.name)
