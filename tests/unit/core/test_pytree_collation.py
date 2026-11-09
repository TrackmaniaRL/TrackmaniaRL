"""NumPy scalar batching retains values and does not alter mixed numeric input."""

import numpy as np
import pytest
import torch

from trackmaniarl.core.pytree import sanitize_finite, tree_collate
from trackmaniarl.trackmania.native_headless.full_qr.environment import _checked_observation


def test_collate_mappings_with_different_key_order() -> None:
    first = {"x": 1, "nested": {"a": 2, "b": 3}}
    second = {"nested": {"b": 30, "a": 20}, "x": 10}

    batch = tree_collate([first, second])

    assert list(batch) == ["x", "nested"]
    assert list(batch["nested"]) == ["a", "b"]
    assert batch["x"].tolist() == [1, 10]
    assert batch["nested"]["a"].tolist() == [2, 20]
    assert batch["nested"]["b"].tolist() == [3, 30]


@pytest.mark.parametrize("second", [{"x": 3}, {"x": 3, "z": 4}])
def test_collate_rejects_mappings_with_different_keys(second: dict[str, int]) -> None:
    with pytest.raises(TypeError, match="different keys"):
        tree_collate([{"x": 1, "y": 2}, second])


def test_numpy_uint64_maximum_remains_exact() -> None:
    values = [np.uint64(0), np.uint64(2**64 - 1)]
    batch = tree_collate(values)
    assert batch.dtype == torch.uint64
    assert batch.tolist() == [0, 2**64 - 1]


@pytest.mark.parametrize("values", [[np.uint64(2**64 - 1), 2], [2, np.uint64(2**64 - 1)]])
def test_mixed_uint64_and_python_int_remain_exact(values: list[object]) -> None:
    batch = tree_collate(values)
    assert batch.dtype == torch.uint64
    assert batch.tolist() == [int(value) for value in values]


@pytest.mark.parametrize(
    ("values", "expected_dtype", "expected_values"),
    [
        ([np.bool_(True), np.bool_(False)], torch.bool, [True, False]),
        ([np.float32(1), np.float32(2)], torch.float32, [1.0, 2.0]),
        ([np.int64(-1), np.int64(2)], torch.int64, [-1, 2]),
        ([np.float32(1), 2.0], torch.float32, [1.0, 2.0]),
        ([1.0, np.float32(2)], torch.float32, [1.0, 2.0]),
        ([np.int32(1), 2], torch.int64, [1, 2]),
        ([1, np.int32(2)], torch.int64, [1, 2]),
        ([np.complex64(1 + 2j), np.complex64(3 + 4j)], torch.complex64, [1 + 2j, 3 + 4j]),
        ([np.uint64(1), 2.0], torch.float32, [1.0, 2.0]),
        ([True, np.bool_(False)], torch.bool, [True, False]),
        ([np.bool_(True), False], torch.bool, [True, False]),
    ],
)
def test_numpy_scalar_and_mixed_numeric_collation(
    values: list[object], expected_dtype: torch.dtype, expected_values: list[object]
) -> None:
    batch = tree_collate([sanitize_finite(value) for value in values])
    assert batch.dtype == expected_dtype
    assert batch.tolist() == expected_values


def test_mixed_unsigned_and_signed_keeps_torch_promotion_failure() -> None:
    with pytest.raises(RuntimeError, match="Promotion"):
        tree_collate([np.uint64(1), np.int64(2)])


def test_mixed_complex_collation_preserves_existing_behavior() -> None:
    batch = tree_collate([np.complex64(1 + 2j), 3 + 4j])
    assert batch.dtype == torch.complex64
    assert batch.tolist() == [1 + 2j, 3 + 4j]


def test_sanitized_numpy_uint64_maximum_remains_exact() -> None:
    values = [np.uint64(0), np.uint64(2**64 - 1)]
    batch = tree_collate([sanitize_finite(value) for value in values])
    assert batch.dtype == torch.uint64
    assert batch.tolist() == [0, 2**64 - 1]


@pytest.mark.parametrize(
    ("scalar_type", "dtype"), [(np.float32, torch.float32), (np.float64, torch.float64)]
)
def test_sanitize_nonfinite_numpy_scalar_matches_array(
    scalar_type: type[np.floating], dtype: torch.dtype
) -> None:
    values = [scalar_type(float("nan")), scalar_type(float("inf")), scalar_type(-float("inf"))]
    batch = tree_collate([sanitize_finite(value) for value in values])
    expected = sanitize_finite(np.array(values))
    assert batch.dtype == dtype
    assert batch.tolist() == expected.tolist()
    assert torch.isfinite(batch).all()


@pytest.mark.parametrize(
    "value", [np.bytes_(b"text"), np.datetime64("2026-01-01"), np.timedelta64(5, "ms")]
)
def test_sanitize_rejects_nonnumeric_numpy_scalars(value: object) -> None:
    with pytest.raises(TypeError, match="unsupported PyTree leaf"):
        sanitize_finite(value)


@pytest.mark.parametrize(
    "value", [np.str_("text"), np.datetime64("2026-01-01"), np.timedelta64(5, "ms")]
)
def test_native_observation_rejects_nonnumeric_numpy_scalars(value: object) -> None:
    with pytest.raises(TypeError, match="numeric tensors or arrays"):
        _checked_observation({"invalid": value})
