"""Actual file/mapping refusal tests; cross-process visibility is tested on Linux."""

import sys
import threading
from pathlib import Path

import pytest

from trackmaniarl.experiments.full_qr.campaign_signal import MappedStopSignal

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux mmap/file ownership")


def test_corrupted_byte_is_an_error_not_an_unset_signal(tmp_path: Path) -> None:
    path = tmp_path / "stop"
    path.write_bytes(b"\2")
    with pytest.raises(ValueError, match="corrupted"):
        MappedStopSignal(path)
    assert path.read_bytes() == b"\2"


def test_attaching_never_follows_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / "foreign"
    target.write_bytes(b"\0")
    path = tmp_path / "stop"
    path.symlink_to(target)
    with pytest.raises(OSError, match="Too many levels"):
        MappedStopSignal(path)
    assert target.read_bytes() == b"\0"


def test_mapping_refuses_non_single_byte_files_without_truncation(tmp_path: Path) -> None:
    path = tmp_path / "stop"
    path.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="single-link"):
        MappedStopSignal(path)
    assert path.read_bytes() == b"preserve"


def test_stop_mapping_is_one_way_and_cannot_be_recreated(tmp_path: Path) -> None:
    path = tmp_path / "stop"
    parent = MappedStopSignal(path, create=True)
    peer = MappedStopSignal(path)
    try:
        assert not parent.is_set()
        assert not peer.wait(0)
        parent.set()
        assert peer.wait(0.1)
        peer.set()
        assert parent.is_set()
        with pytest.raises(FileExistsError):
            MappedStopSignal(path, create=True)
    finally:
        peer.close()
        parent.close()
    assert path.read_bytes() == b"\1"


@pytest.mark.parametrize("set_before_close", [True, False])
def test_close_during_wait_preserves_only_an_actual_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, set_before_close: bool
) -> None:
    path = tmp_path / "stop"
    signal = MappedStopSignal(path, create=True)
    peer = MappedStopSignal(path)
    sleeping = threading.Event()
    outcomes: list[bool | BaseException] = []
    original_wait = signal._wait_condition.wait

    def announce_wait(timeout: float | None = None) -> bool:
        sleeping.set()
        return original_wait(timeout)

    monkeypatch.setattr(signal._wait_condition, "wait", announce_wait)

    def wait_for_shutdown() -> None:
        try:
            outcomes.append(signal.wait(2))
        except BaseException as error:
            outcomes.append(error)

    waiter = threading.Thread(target=wait_for_shutdown)
    waiter.start()
    try:
        assert sleeping.wait(1)
        # Reproduce the worker's set/close before its shutdown waiter resumes.
        with signal._wait_condition:
            if set_before_close:
                signal.set()
            signal.close()
        waiter.join(1)
        assert not waiter.is_alive()
        if set_before_close:
            assert outcomes == [True]
        else:
            assert len(outcomes) == 1
            assert isinstance(outcomes[0], RuntimeError)
        assert peer.is_set() is set_before_close
        with pytest.raises(RuntimeError, match="closed"):
            signal.wait(0)
        with pytest.raises(RuntimeError, match="closed"):
            signal.is_set()
    finally:
        signal.close()
        peer.close()
        waiter.join(3)
