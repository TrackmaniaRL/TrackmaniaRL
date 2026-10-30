"""No game access: verify layout, selective export, snapshot ownership and cleanup."""

import ctypes as ct
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import headless_windows_fp_context as fp


def test_worker_receives_explicit_executable_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supplied = tmp_path / "synthetic-client.exe"
    seen = []
    monkeypatch.setattr(fp, "worker", lambda args: seen.append(args.executable) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--pid", str(os.getpid() + 1), "--executable", str(supplied), "--worker"],
    )
    assert fp.main() == 0
    assert seen == [supplied]


def test_relative_executable_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--pid", str(os.getpid() + 1), "--executable", "relative.exe", "--worker"],
    )
    with pytest.raises(SystemExit):
        fp.main()


def test_parent_forwards_executable_to_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supplied = tmp_path / "synthetic-client.exe"
    command = []

    def run(args: list[str], **kwargs: Any) -> SimpleNamespace:
        command.extend(args)
        return SimpleNamespace(stdout='{"ok": true}')

    monkeypatch.setattr(fp.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--pid",
            str(os.getpid() + 1),
            "--executable",
            str(supplied),
            "--inspect",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert fp.main() == 0
    assert command[command.index("--executable") + 1] == str(supplied)


def context() -> bytes:
    raw = bytearray(1232)
    struct.pack_into("<II", raw, 48, fp.CONTEXT_FLAGS, 0x1FA0)
    struct.pack_into("<Q", raw, 248, 0x140001234)
    struct.pack_into("<HH", raw, 256, 0x027F, 0x20)
    struct.pack_into("<II", raw, 280, 0x1FA0, 0xFFFF)
    return bytes(raw)


def test_context_projection_and_snapshot_thread_layout() -> None:
    result = fp.validate_layout()
    assert result["context_bytes"] == 1232
    assert result["pss_thread_bytes"] == 120
    assert result["pss_context_pointer_offset"] == 112


def test_only_fp_words_and_module_relative_instruction_pointer_are_exported() -> None:
    result = fp.decode_context(
        context(), [{"base": 0x140000000, "end": 0x140100000, "name": "Trackmania.exe"}]
    )
    assert result["instruction_location"] == {"module": "Trackmania.exe", "rva": "0x1234"}
    assert result["mxcsr_rounding"] == result["x87_rounding"] == "nearest-even"
    assert result["x87_precision_bits"] == 53
    assert result["mxcsr_flush_to_zero"] is result["mxcsr_denormals_are_zero"] is False
    assert not {"rip", "rsp", "integer", "stack", "context", "raw"}.intersection(result)


def test_partial_or_inconsistent_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="complete"):
        fp.decode_context(context()[:-1], [])
    raw = bytearray(context())
    struct.pack_into("<I", raw, 280, 0x9FA0)
    with pytest.raises(ValueError, match="MXCSR differ"):
        fp.decode_context(raw, [])
    struct.pack_into("<I", raw, 48, 0x100001)
    with pytest.raises(ValueError, match="lacks requested"):
        fp.decode_context(raw, [])


class FakePSS:
    """Real local aligned bytes, fake PSS descriptors; no Windows APIs."""

    success = staticmethod(fp.WindowsAPI.success)

    def __init__(self, failure: str = "") -> None:
        self.failure = failure
        self.calls = []
        self.walked = False
        self.owner = ct.create_string_buffer(1232 + 15)
        self.aligned = (ct.addressof(self.owner) + 15) & ~15
        ct.memmove(self.aligned, context(), 1232)

    def PssCaptureSnapshot(self, process, flags, context_flags, snapshot):  # noqa: N802, ANN001, ANN201, PLR0913
        assert process == 123
        assert flags == 0x180
        assert context_flags == 0x100009
        self.calls.append("capture")
        ct.cast(snapshot, ct.POINTER(fp.HANDLE))[0] = 55
        return 0

    def PssQuerySnapshot(self, snapshot, query, output, size):  # noqa: N802, ANN001, ANN201, PLR0913
        info = ct.cast(output, ct.POINTER(fp.ThreadInfo)).contents
        info.count, info.context_length = 1, 1232
        return 5 if self.failure == "query" else 0

    def PssWalkMarkerCreate(self, allocator, marker):  # noqa: N802, ANN001, ANN201
        if self.failure == "marker":
            return 8
        ct.cast(marker, ct.POINTER(fp.HANDLE))[0] = 66
        return 0

    def PssWalkSnapshot(self, snapshot, walk, marker, output, size):  # noqa: N802, ANN001, ANN201, PLR0913
        if self.failure == "walk":
            return 5
        if self.walked:
            return 259
        self.walked = True
        entry = ct.cast(output, ct.POINTER(fp.ThreadEntry)).contents
        entry.pid, entry.tid, entry.created.low = 999, 1001, 100
        entry.context, entry.context_size = self.aligned, 1232
        if self.failure == "identity":
            entry.pid = 998
        if self.failure == "decode":
            ct.memset(self.aligned + 48, 0, 4)
        return 0

    def PssWalkMarkerFree(self, marker):  # noqa: N802, ANN001, ANN201
        self.calls.append("free-marker")
        return 0

    def GetCurrentProcess(self):  # noqa: N802, ANN201
        return -1

    def PssFreeSnapshot(self, process, snapshot):  # noqa: N802, ANN001, ANN201
        assert process == -1
        assert snapshot.value == 55
        self.calls.append("free-local-snapshot")
        return 0


def test_local_snapshot_and_marker_are_freed_once_after_success() -> None:
    api = FakePSS()
    result = fp.capture_snapshot(api, 123, {"pid": 999, "created_filetime": 99}, [])
    assert result["cleanup"] == {"snapshot_free_code": 0, "marker_free_code": 0}
    assert len(result["rows"]) == 1
    assert api.calls == ["capture", "free-marker", "free-local-snapshot"]


@pytest.mark.parametrize("failure", ["query", "marker", "walk", "identity", "decode"])
def test_every_postcapture_failure_frees_the_local_snapshot(failure: str) -> None:
    api = FakePSS(failure)
    with pytest.raises(fp.SnapshotError) as error:
        fp.capture_snapshot(api, 123, {"pid": 999, "created_filetime": 99}, [])
    assert error.value.cleanup["snapshot_free_code"] == 0
    assert api.calls.count("free-local-snapshot") == 1
    assert api.calls.count("free-marker") == int(failure not in {"query", "marker"})


class MixedPSS(FakePSS):
    def __init__(self, *, active_missing: bool = False, only_terminated: bool = False) -> None:
        super().__init__()
        self.first_returned = False
        self.active_missing, self.only_terminated = active_missing, only_terminated

    def PssQuerySnapshot(self, snapshot, query, output, size):  # noqa: N802, ANN001, ANN201, PLR0913
        result = super().PssQuerySnapshot(snapshot, query, output, size)
        ct.cast(output, ct.POINTER(fp.ThreadInfo)).contents.count = 1 if self.only_terminated else 2
        return result

    def PssWalkSnapshot(self, snapshot, walk, marker, output, size):  # noqa: N802, ANN001, ANN201, PLR0913
        if not self.first_returned:
            self.first_returned = True
            entry = ct.cast(output, ct.POINTER(fp.ThreadEntry)).contents
            entry.pid, entry.tid, entry.created.low = 999, 1000, 100
            entry.flags = fp.THREAD_TERMINATED
            entry.context, entry.context_size = None, 0
            return 0
        if self.only_terminated:
            return 259
        result = super().PssWalkSnapshot(snapshot, walk, marker, output, size)
        if result == 0 and self.active_missing:
            entry = ct.cast(output, ct.POINTER(fp.ThreadEntry)).contents
            entry.context, entry.context_size = None, 0
        return result


def test_terminated_descriptor_is_retained_without_inventing_context() -> None:
    api = MixedPSS()
    result = fp.capture_snapshot(api, 123, {"pid": 999, "created_filetime": 99}, [])
    assert result["enumerated_thread_count"] == 2
    assert len(result["rows"]) == 1
    assert result["terminated_threads"] == [
        {
            "tid": 1000,
            "thread_flags": 1,
            "exit_status": 0,
            "context_bytes": 0,
            "context_present": False,
            "fp_context_decoded": False,
        }
    ]
    assert result["cleanup"] == {"snapshot_free_code": 0, "marker_free_code": 0}


@pytest.mark.parametrize("only_terminated", [False, True])
def test_active_missing_context_or_no_active_threads_still_fails(*, only_terminated: bool) -> None:
    api = MixedPSS(active_missing=True, only_terminated=only_terminated)
    with pytest.raises(fp.SnapshotError):
        fp.capture_snapshot(api, 123, {"pid": 999, "created_filetime": 99}, [])
    assert api.calls[-2:] == ["free-marker", "free-local-snapshot"]
