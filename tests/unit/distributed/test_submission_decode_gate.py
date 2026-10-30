from concurrent.futures import ThreadPoolExecutor
from threading import Lock, RLock
from time import sleep
from types import SimpleNamespace

import pytest
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.distributed import coordinator_submission


def test_parallel_submissions_serialize_decode_but_release_before_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, guard = RLock(), Lock()
    active = maximum = calls = 0

    def request(*_: object) -> dict[str, object]:
        nonlocal active, maximum, calls
        with guard:
            active += 1
            maximum = max(maximum, active)
            calls += 1
        sleep(0.01)
        with guard:
            active -= 1
        return {}

    def response(*_: object) -> BytesValue:
        def available() -> None:
            assert gate.acquire(timeout=1)
            gate.release()

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(available).result(timeout=2)
        return BytesValue(value=b"checked")

    coordinator = SimpleNamespace(_submission_decode_lock=gate, _request=request)
    monkeypatch.setattr(coordinator_submission, "_validate_submission", lambda *_: None)
    monkeypatch.setattr(coordinator_submission, "_submission", lambda *_: None)
    monkeypatch.setattr(coordinator_submission, "_hard_lagged", lambda *_: True)
    monkeypatch.setattr(coordinator_submission, "_reject_hard_lag", response)
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(
            executor.map(lambda _: coordinator_submission.submit(coordinator, None, None), range(6))
        )
    assert maximum == 1
    assert calls == 6
    assert all(result.value == b"checked" for result in results)
