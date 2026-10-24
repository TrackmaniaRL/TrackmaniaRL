"""Fresh timing diagnostics must bind bytes and obey cancellation after receive."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from scripts.headless_telemetry006 import Telemetry006Client, Telemetry006Config
from scripts.headless_windows_timing import _FreshProbeClient, _packet
from trackmaniarl.trackmania.telemetry import TelemetryFrame


def test_packet_rejects_metadata_from_a_different_sample() -> None:
    client = Telemetry006Client()
    raw = np.arange(33, dtype="<f4")
    client.last_metadata = {"sequence": 1}
    client.last_packet = (raw + 1).tobytes()
    with pytest.raises(ValueError, match="does not bind"):
        _packet(raw, client)


def test_packet_retains_exact_signed_zero_bytes() -> None:
    client = Telemetry006Client()
    raw = np.arange(33, dtype="<f4")
    raw[0] = -0.0
    client.last_metadata = {"sequence": 1}
    client.last_packet = raw.tobytes()
    packet = _packet(raw, client)
    assert bytes.fromhex(packet["raw33_le_hex"]) == raw.tobytes()
    assert packet["fresh_metadata"] == client.last_metadata


def test_receive_deadline_is_capped_by_overall_probe_budget() -> None:
    class Budget:
        def remaining(self) -> float:
            return 0.01

    client = _FreshProbeClient(Telemetry006Config(), Budget())
    assert client._remaining(time.monotonic() + 2) <= 0.01


def test_stop_during_receive_prevents_returning_the_sample(monkeypatch: Any) -> None:
    class Budget:
        stopped = False

        def remaining(self) -> float:
            if self.stopped:
                raise TimeoutError("STOP observed")
            return 1

    budget = Budget()
    client = _FreshProbeClient(Telemetry006Config(), budget)

    def receive(self: Telemetry006Client) -> TelemetryFrame:
        budget.stopped = True
        return TelemetryFrame(np.zeros(33, dtype=np.float32))

    monkeypatch.setattr(Telemetry006Client, "read", receive)
    with pytest.raises(TimeoutError, match="STOP observed"):
        client.read()
