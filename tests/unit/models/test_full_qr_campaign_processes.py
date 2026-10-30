"""Linux pidfd race handling; no historical PID authorizes a signal."""

import errno
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime import ownership


@pytest.mark.parametrize("current", [None, {"birth": 999}])
def test_reaped_or_reused_pid_einval_is_not_a_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, current: Any
) -> None:
    old = {"pid": 123, "birth": 1, "uid": 0, "sid": 123, "pgid": 123}
    observed = None if current is None else old | current

    def failed_open(_: int) -> int:
        raise OSError(errno.EINVAL, "fixture kernel response")

    monkeypatch.setattr(ownership.os, "pidfd_open", failed_open, raising=False)
    monkeypatch.setattr(ownership, "process_identity", lambda _: observed)
    assert ownership.signal_identity(old, 9) is False


@pytest.mark.parametrize("error_number", [errno.EINVAL, errno.EPERM])
def test_pidfd_error_for_same_owned_live_identity_remains_explicit(
    monkeypatch: pytest.MonkeyPatch, error_number: int
) -> None:
    old = {"pid": 123, "birth": 1, "uid": 0, "sid": 123, "pgid": 123}

    def failed_open(_: int) -> int:
        raise OSError(error_number, "fixture kernel response")

    monkeypatch.setattr(ownership.os, "pidfd_open", failed_open, raising=False)
    monkeypatch.setattr(ownership, "process_identity", lambda _: old)
    with pytest.raises(OSError, match="fixture kernel response") as result:
        ownership.signal_identity(old, 9)
    assert result.value.errno == error_number
