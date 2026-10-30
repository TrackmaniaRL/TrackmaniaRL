"""Clock qualification must come from distinctive physical measurements."""

import struct

import pytest

from scripts.headless_clock_alignment import (
    AlignmentError,
    Sample,
    align_clocks,
    native_samples,
    reader_phase,
    windows_samples,
)


def series(count: int = 8) -> tuple[list[Sample], list[Sample]]:
    native = [
        Sample(2200 + index * 10, (496.0, 10.0, 624.0, index / 100, -index / 1000, index / 10000))
        for index in range(count)
    ]
    windows = [Sample(sample.time - 2190, sample.physics) for sample in native]
    return native, windows


def test_relative_clock_is_inferred_and_validated_on_unused_physical_states() -> None:
    native, windows = series()
    result = align_clocks(native, windows)
    assert result["relative_clock_alignment_qualified"] is True
    assert result["slope"] == {"numerator": 1, "denominator": 1}
    assert result["intercept"] == {"numerator": -2190, "denominator": 1}
    assert result["fit_matches"] == result["heldout_matches"] == 4
    assert result["input_archive_clock_qualified"] is False


def test_clock_scale_is_measured_instead_of_assumed() -> None:
    native, windows = series()
    windows = [Sample(2 * sample.time, sample.physics) for sample in windows]
    assert align_clocks(native, windows)["slope"] == {"numerator": 2, "denominator": 1}


def test_stationary_repeated_states_do_not_establish_alignment() -> None:
    native = [Sample(2200 + i * 10, (496.0, 10.0, 624.0, 0.0, 0.0, 0.0)) for i in range(20)]
    windows = [Sample(sample.time - 2190, sample.physics) for sample in native]
    with pytest.raises(AlignmentError, match="repeated signatures excluded"):
        align_clocks(native, windows)


def test_one_coincident_state_does_not_establish_alignment() -> None:
    native, windows = series(1)
    with pytest.raises(AlignmentError, match="only 1 unique"):
        align_clocks(native, windows)


def test_heldout_clock_disagreement_rejects_affine_fit() -> None:
    native, windows = series()
    windows[3] = Sample(windows[3].time + 1, windows[3].physics)
    with pytest.raises(AlignmentError, match="heldout"):
        align_clocks(native, windows)


def test_numeric_nearness_does_not_replace_exact_physical_identity() -> None:
    native, windows = series()
    windows = [
        Sample(sample.time, (*sample.physics[:5], sample.physics[5] + 0.000001))
        for sample in windows
    ]
    with pytest.raises(AlignmentError, match="only 0 unique"):
        align_clocks(native, windows)


def test_different_states_at_one_windows_race_time_are_rejected() -> None:
    first = [0.0] * 33
    second = [0.0] * 33
    first[3], second[3], second[4] = 10.0, 10.0, 1.0
    with pytest.raises(AlignmentError, match="different physical states share"):
        windows_samples({"rows": [{"raw33": first}, {"raw33": second}]})


def test_nonmonotonic_measured_clock_is_rejected() -> None:
    native, windows = series()
    native[3] = Sample(native[2].time, native[3].physics)
    with pytest.raises(AlignmentError, match="strictly increasing"):
        align_clocks(native, windows)


def test_windows_parser_preserves_and_checks_acquisition_order() -> None:
    first, second = [0.0] * 33, [0.0] * 33
    first[3], second[3] = 20.0, 10.0
    with pytest.raises(AlignmentError, match="regressed"):
        windows_samples({"rows": [{"raw33": first}, {"raw33": second}]})


def test_native_fractional_time_is_not_truncated_into_a_matching_counter() -> None:
    row = {
        "native_t": 2200.9,
        "state": {"physicalPosition": [496.0, 10.0, 624.0], "physicalVelocity": [0.0, 0.0, 0.0]},
    }
    with pytest.raises(AlignmentError, match="literal integer"):
        native_samples({"rows": [row]})


def test_source_reader_phase_keeps_input_label_separate_from_next_physical_sample() -> None:
    raw = bytearray(256)
    raw[:8] = b"TMREAD01"
    struct.pack_into("<4I", raw, 112, 151, 4152, 2200, 2190)
    struct.pack_into("<i", raw, 144, -1500)
    struct.pack_into("<iI", raw, 248, -1510, 1510)
    phase = reader_phase(bytes(raw))
    assert phase["archive_input_label_ms"] == 0
    assert phase["held_native_boundary"] == 2190
    assert phase["next_native_boundary"] == phase["source_start_getter_result"] == 2200
