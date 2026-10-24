"""Coverage guards on a real held 50 ms finish-control snapshot."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from struct import pack_into

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.contact_registry_capture import (
    RegistryContext,
    decode_registry_window,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/full_qr/registry_window_v1"


def evidence() -> tuple[bytes, RegistryContext]:
    raw = (FIXTURE / "window.bin").read_bytes()
    saved = json.loads((FIXTURE / "context.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == saved["sha256"]
    return raw, RegistryContext(**saved["context"])


def test_real_window_has_every_interval_despite_zero_waypoint_vector() -> None:
    raw, context = evidence()
    result = decode_registry_window(raw, context, (19900, 19950))
    assert result.intervals == 5
    assert result.contact_batches == 4
    assert result.waypoint_batches == 0
    assert asdict(result) == json.loads((FIXTURE / "context.json").read_text())["expected"]
    assert not hasattr(result, "finished")


@pytest.mark.parametrize("kind", ["missing", "partial", "duplicate", "swap"])
def test_missing_or_reordered_interval_rejected(kind: str) -> None:
    raw, context = evidence()
    damaged = {
        "missing": raw[:-128],
        "partial": raw[:-1],
        "duplicate": raw[:128] + raw[:128] + raw[256:],
        "swap": raw[128:256] + raw[:128] + raw[256:],
    }[kind]
    with pytest.raises(ValueError, match="registry"):
        decode_registry_window(damaged, context, (19900, 19950))


@pytest.mark.parametrize("offset", [16, 24, 32, 48, 88])
def test_redirected_window_record_rejected(offset: int) -> None:
    raw, context = evidence()
    damaged = bytearray(raw)
    pack_into("<I", damaged, 128 + offset, 0)
    with pytest.raises(ValueError, match="registry"):
        decode_registry_window(bytes(damaged), context, (19900, 19950))


@pytest.mark.parametrize("bounds", [(19910, 19960), (19900, 19900), (19901, 19951)])
def test_misbound_window_rejected(bounds: tuple[int, int]) -> None:
    raw, context = evidence()
    with pytest.raises(ValueError, match="registry"):
        decode_registry_window(raw, context, bounds)
