from __future__ import annotations

import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from trackmaniarl.core import builtins
from trackmaniarl.core.builtins import JsonCheckpointCodec, TorchCheckpointCodec


def _mark_execution(path: str) -> None:
    Path(path).touch()


class _ExecutablePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> Any:
        return _mark_execution, (str(self.marker),)


def test_torch_checkpoint_rejects_executable_pickle_and_cleans_up(tmp_path: Path) -> None:
    codec = TorchCheckpointCodec()
    target = tmp_path / "untrusted.checkpoint"
    marker = tmp_path / "executed"
    codec.save({"payload": _ExecutablePayload(marker)}, target)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        codec.load(target)
    assert not marker.exists()
    assert list(tmp_path.iterdir()) == [target]


def test_checkpoint_load_fits_the_same_path_budget_as_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.checkpoint"
    maximum = len(str(target.with_name(f".{target.name}.{'0' * 32}.tmp")))
    original = Path.open

    def bounded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if len(str(path)) > maximum:
            raise OSError("simulated Windows path length limit")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", bounded_open)
    codec = TorchCheckpointCodec()
    codec.save({"updates": 17}, target)
    assert codec.load(target) == {"updates": 17}
    assert list(tmp_path.iterdir()) == [target]


def test_json_save_failure_preserves_checkpoint_and_removes_partial_file(tmp_path: Path) -> None:
    codec = JsonCheckpointCodec()
    target = tmp_path / "state.json"
    codec.save({"updates": 7}, target)
    with pytest.raises(TypeError):
        codec.save({"updates": 8, "unsupported": object()}, target)
    assert codec.load(target) == {"updates": 7}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("codec", [JsonCheckpointCodec(), TorchCheckpointCodec()])
def test_overlapping_checkpoint_saves_publish_one_complete_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codec: Any
) -> None:
    target = tmp_path / "state.checkpoint"
    barrier = threading.Barrier(2, timeout=10)
    replace = builtins.os.replace

    def overlap(source: Path, destination: Path) -> None:
        barrier.wait()
        replace(source, destination)

    monkeypatch.setattr(builtins.os, "replace", overlap)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(codec.save, {"updates": value}, target) for value in (1, 2)]
        completed = 0
        for future in futures:
            try:
                future.result(timeout=20)
                completed += 1
            except PermissionError:
                # Windows can reject replacing the target during the other writer's fsync.
                # This must remain an explicit failure, never a corrupt successful save.
                assert sys.platform == "win32"
    assert completed >= 1
    assert codec.load(target) in ({"updates": 1}, {"updates": 2})
    assert list(tmp_path.iterdir()) == [target]
