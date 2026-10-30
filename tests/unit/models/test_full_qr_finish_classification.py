"""Exercise synthetic classifier records; mutations must not acquire authority."""

import struct

import pytest

from tests.unit.models.test_full_qr_waypoint_capture import fixture as synthetic_writer
from trackmaniarl.trackmania.native_headless.full_qr.finish_classification import (
    EndpointClassificationContext,
    decode_endpoint_classification,
)


def _evidence() -> tuple[bytes, bytes, EndpointClassificationContext]:
    writer, meta = synthetic_writer()
    writer = bytearray(writer)
    manager = 0x70000000
    struct.pack_into("<Q", writer, 64 + 8 * 8, manager)
    records = [bytearray(1024), bytearray(1024)]
    base = meta["module_slide"]
    for index, record in enumerate(records):
        record[:8] = b"TMFC0001"
        struct.pack_into("<4I", record, 8, 1024, 1, 123, 123)
        struct.pack_into(
            "<4Q", record, 24, index + 1, base + 0x12154A0, base + 0x12157BB, 1 << index
        )
        struct.pack_into("<2I", record, 56, 4, 5)
        struct.pack_into("<2I", record, 72, 19900, index)
        struct.pack_into("<Q", record, 80 + 16 * 8, base + (0x12154A0 if index == 0 else 0x12157BB))
        struct.pack_into("<Q", record, 80 + 19 * 8, 0x80000098 if index == 0 else 0x80000000)
        struct.pack_into("<Q", record, 848, 456)
    before, after = records
    struct.pack_into("<Q", before, 304, base + 0x11FB9F8)
    struct.pack_into("<Q", after, 304 + 0x98, base + 0x11FB9F8)
    struct.pack_into("<Q", before, 80 + 13 * 8, meta["participant"])
    struct.pack_into("<Q", before, 80 + 14 * 8, manager)
    struct.pack_into("<Q", after, 80 + 10 * 8, 3)
    for offset, value in (
        (576, manager),
        (584, 0x81000000),
        (592, 0x82000000),
        (600, 0x83000000),
        (608, 0x82000000),
        (776, 0x84000000),
    ):
        struct.pack_into("<Q", before, offset, value)
    before[648:652] = b"Goal"
    before[648 + 0x51] = 1
    return bytes(before + after), bytes(writer), EndpointClassificationContext(123, 456, meta)


def test_synthetic_goal_classification_stays_distinct_from_client_ui() -> None:
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
