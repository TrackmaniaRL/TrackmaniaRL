from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trackmaniarl.core.fingerprint import _asset_hash


@pytest.mark.parametrize("data", [b"", bytes(range(256)) * 4099], ids=["empty", "multi_chunk"])
def test_asset_hash_streams_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: bytes
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(data)

    def prohibit_read_bytes(_path: Path) -> bytes:
        raise AssertionError("asset hashing must not read the full file into memory")

    monkeypatch.setattr(Path, "read_bytes", prohibit_read_bytes)
    assert _asset_hash(path.name, tmp_path, "asset_path") == hashlib.sha256(data).hexdigest()


def test_asset_hash_preserves_missing_and_invalid_path_behavior(tmp_path: Path) -> None:
    assert _asset_hash(None, tmp_path, "asset_path") is None
    with pytest.raises(TypeError, match="asset_path must be a string"):
        _asset_hash(123, tmp_path, "asset_path")
    with pytest.raises(FileNotFoundError):
        _asset_hash("missing.bin", tmp_path, "asset_path")
