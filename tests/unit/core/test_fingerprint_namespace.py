"""Namespace identity is independent of duplicate import search paths."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trackmaniarl.core.fingerprint import _package_source_digest


def test_duplicate_namespace_roots_do_not_change_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "component.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    module = SimpleNamespace(__path__=[str(tmp_path)])
    with patch("trackmaniarl.core.fingerprint.importlib.import_module", return_value=module):
        expected = _package_source_digest("isolated_namespace")
        module.__path__ = [str(tmp_path), str(tmp_path / ".")]
        assert _package_source_digest("isolated_namespace") == expected
        source.write_text("VALUE = 2\n", encoding="utf-8")
        assert _package_source_digest("isolated_namespace") != expected
