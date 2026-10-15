"""Process ownership regressions; these tests never open the game."""

import subprocess
import sys
from typing import Never

import psutil
import pytest

from experiments.tmrl_test_comparison.owned_processes import wait_for_owned_processes


class ObservedProcess:
    def __init__(self, *, running: bool = False, failure: Exception | None = None) -> None:
        self.running = running
        self.failure = failure

    def is_running(self) -> bool:
        if self.failure:
            raise self.failure
        return self.running

    def wait(self, **kwargs: object) -> Never:
        raise AssertionError("Never wait on a handle for a potentially reused PID")


def test_reused_or_exited_identity_does_not_open_a_wait_handle() -> None:
    assert wait_for_owned_processes([ObservedProcess()], timeout=0) == []


def test_live_original_identity_blocks_handoff() -> None:
    process = ObservedProcess(running=True)
    assert wait_for_owned_processes([process], timeout=0) == [process]


def test_missing_original_identity_is_closed() -> None:
    assert (
        wait_for_owned_processes([ObservedProcess(failure=psutil.NoSuchProcess(123))], timeout=0)
        == []
    )


def test_unknown_live_identity_fails_closed() -> None:
    with pytest.raises(psutil.AccessDenied):
        wait_for_owned_processes([ObservedProcess(failure=psutil.AccessDenied(123))], timeout=0)


@pytest.mark.parametrize("timeout", [-1, 31, float("inf"), float("nan")])
def test_timeout_cannot_extend_the_queue_cap(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        wait_for_owned_processes([], timeout=timeout)


def test_real_cpu_child_can_close_without_process_wait() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.1)"])
    observed = psutil.Process(child.pid)
    try:
        child.wait(timeout=5)
        assert wait_for_owned_processes([observed], timeout=1) == []
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
