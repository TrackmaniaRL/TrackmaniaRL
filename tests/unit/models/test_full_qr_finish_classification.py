"""Replay actual classifier evidence; mutations must never acquire authority."""

import hashlib
import json
import struct
from pathlib import Path

import pytest

from trackmaniarl.trackmania.native_headless.full_qr.finish_classification import (
    EndpointClassificationContext,
    decode_endpoint_classification,
)


def _evidence() -> tuple[bytes, bytes, EndpointClassificationContext]:
    directory = Path(__file__).parents[2] / "fixtures/full_qr/endpoint_classifier_v1"
    context = json.loads((directory / "context.json").read_text())
    for item in context["files"]:
        assert hashlib.sha256((directory / item["name"]).read_bytes()).hexdigest() == item["sha256"]
    return (
        (directory / "classifier.bin").read_bytes(),
        (directory / "writer.bin").read_bytes(),
        EndpointClassificationContext(**{key: context[key] for key in ("pid", "birth", "meta")}),
    )


def test_actual_goal_classification_stays_distinct_from_client_ui() -> None:
    raw, writer, context = _evidence()
    endpoint = decode_endpoint_classification(raw, writer, context)
    assert endpoint.map_entry_name == "Goal"
    assert endpoint.classification == 3
    assert endpoint.waypoint_id == 0
    assert endpoint.native_time == 19900
    assert endpoint.raw_waypoint_time == 19915
    assert endpoint.client_ui_finish_proven is False


@pytest.mark.parametrize(
    ("offset", "format", "value"),
    [
        (848, "Q", 1),  # reused process ID with a different birth
        (1024 + 24, "Q", 1),  # duplicated classification entry
        (304, "Q", 0),  # unrelated classifier caller
        (1024 + 304 + 0x98, "Q", 0),  # return to a different caller
        (1024 + 72, "I", 19910),  # pair crosses a native interval
        (608, "Q", 0),  # map entry is not in the captured table
        (648 + 0x51, "B", 0),  # map entry has no endpoint bit
        (1024 + 80 + 10 * 8, "Q", 1),  # native checkpoint classification
        (48, "Q", 2),  # wrong hardware breakpoint fired
    ],
)
def test_corrupted_classifier_evidence_is_rejected(offset: int, format: str, value: int) -> None:
    raw, writer, context = _evidence()
    changed = bytearray(raw)
    struct.pack_into("<" + format, changed, offset, value)
    with pytest.raises(ValueError, match="mismatch"):
        decode_endpoint_classification(bytes(changed), writer, context)


def test_classifier_and_writer_must_belong_to_same_manager() -> None:
    raw, writer, context = _evidence()
    changed = bytearray(writer)
    struct.pack_into("<Q", changed, 64 + 8 * 8, 4096)
    with pytest.raises(ValueError, match="writer binding mismatch"):
        decode_endpoint_classification(raw, bytes(changed), context)


def test_missing_evidence_does_not_imply_a_negative_finish() -> None:
    _, writer, context = _evidence()
    with pytest.raises(ValueError, match="complete classification"):
        decode_endpoint_classification(b"", writer, context)
