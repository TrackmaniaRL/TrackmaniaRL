from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.checkpoint_pickle import restore_numpy_array


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int8, np.int64, np.uint32, np.float16, np.float64, np.complex64]
)
@pytest.mark.parametrize("layout", ["c", "f", "reverse", "readonly", "empty", "scalar"])
def test_checkpoint_preserves_numpy_values_and_types(
    tmp_path: Path, dtype: type, layout: str
) -> None:
    array = np.arange(24).astype(dtype).reshape(4, 6)
    if layout == "f":
        array = np.asfortranarray(array)
    elif layout == "reverse":
        array = array[::-1]
    elif layout == "readonly":
        array.setflags(write=False)
    elif layout == "empty":
        array = array[:0]
    elif layout == "scalar":
        array = np.array(array[0, 0]).reshape(())
    codec = TorchCheckpointCodec()
    path = tmp_path / "state.pt"
    codec.save({"array": array, "alias": array, "tensor": torch.arange(3)}, path)
    loaded = codec.load(path)
    assert type(loaded["array"]) is np.ndarray
    assert loaded["array"].dtype == array.dtype
    assert loaded["array"].shape == array.shape
    np.testing.assert_array_equal(loaded["array"], array)
    assert loaded["alias"] is loaded["array"]
    assert torch.equal(loaded["tensor"], torch.arange(3))


@pytest.mark.parametrize(
    "array",
    [np.arange(4, dtype=">i4"), np.array(["left", "right"]), np.zeros(2, dtype=[("x", "i4")])],
)
def test_checkpoint_keeps_legacy_numpy_fallback(tmp_path: Path, array: np.ndarray) -> None:
    codec = TorchCheckpointCodec()
    path = tmp_path / "state.pt"
    codec.save({"array": array}, path)
    restored = codec.load(path)["array"]
    expected = pickle.loads(pickle.dumps(array, protocol=2))
    assert restored.dtype == expected.dtype
    np.testing.assert_array_equal(restored, expected)


def test_numpy_restore_rejects_other_payload_types() -> None:
    with pytest.raises(ValueError, match="CPU tensor"):
        restore_numpy_array([1, 2])


def test_checkpoint_preserves_float_bit_patterns(tmp_path: Path) -> None:
    bits = np.array([0, 0x80000000, 0x7FC00001, 0x7F800000, 0xFF800000], dtype=np.uint32)
    array = bits.view(np.float32)
    path = tmp_path / "bits.pt"
    codec = TorchCheckpointCodec()
    codec.save({"array": array}, path)
    np.testing.assert_array_equal(codec.load(path)["array"].view(np.uint32), bits)


def test_checkpoint_preserves_different_dtype_views_of_one_storage(tmp_path: Path) -> None:
    array = np.arange(6, dtype=np.float32)
    path = tmp_path / "views.pt"
    codec = TorchCheckpointCodec()
    codec.save(
        {"floats": array, "bytes": array.view(np.uint8), "tensor": torch.from_numpy(array)}, path
    )
    loaded = codec.load(path)
    np.testing.assert_array_equal(loaded["floats"], array)
    np.testing.assert_array_equal(loaded["bytes"], array.view(np.uint8))
    assert torch.equal(loaded["tensor"], torch.from_numpy(array))


@pytest.mark.parametrize("tensor_first", [False, True])
def test_numpy_and_later_tensor_with_conflicting_storage_dtype(
    tmp_path: Path, *, tensor_first: bool
) -> None:
    array = np.arange(24, dtype=np.uint8)
    tensor = torch.from_numpy(array).view(torch.float32)
    entries = [("array", array), ("tensor", tensor)]
    if tensor_first:
        entries.reverse()
    codec = TorchCheckpointCodec()
    path = tmp_path / "mixed.pt"
    codec.save(dict(entries), path)
    restored = codec.load(path)
    np.testing.assert_array_equal(restored["array"], array)
    torch.testing.assert_close(restored["tensor"], tensor, rtol=0, atol=0)
    restored["array"].fill(0)
    torch.testing.assert_close(restored["tensor"], tensor, rtol=0, atol=0)


def test_strided_array_does_not_serialize_bytes_outside_its_values(tmp_path: Path) -> None:
    import zstandard

    marker = b"storage gap must not enter checkpoint"
    storage = np.zeros(4096, dtype=np.uint8)
    storage[1 : 1 + len(marker)] = np.frombuffer(marker, dtype=np.uint8)
    selected = storage[::64]
    codec = TorchCheckpointCodec()
    path = tmp_path / "gapped.pt"
    codec.save({"array": selected}, path)
    np.testing.assert_array_equal(codec.load(path)["array"], selected)
    with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
        assert marker not in reader.read()


def test_checkpoint_loads_original_protocol2_numpy_format(tmp_path: Path) -> None:
    import zstandard

    array = np.arange(18, dtype=np.float32).reshape(3, 6)
    path = tmp_path / "old.pt"
    with (
        path.open("wb") as stream,
        zstandard.ZstdCompressor().stream_writer(stream, closefd=False) as compressed,
    ):
        torch.save({"array": array}, compressed)
    np.testing.assert_array_equal(TorchCheckpointCodec().load(path)["array"], array)
