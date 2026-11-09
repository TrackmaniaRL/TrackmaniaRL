from __future__ import annotations

import numpy as np
import pytest
import torch

from trackmaniarl.algorithms.ppo_support import NormalizerConfig, _ObservationNormalizer
from trackmaniarl.distributed.codec import WireCodec


def _normalizer() -> _ObservationNormalizer:
    return _ObservationNormalizer(NormalizerConfig(enabled=True, clip=100.0))


def test_distinct_mapping_and_tuple_paths_have_independent_statistics() -> None:
    normalizer = _normalizer()
    tree = {
        "a.b": torch.tensor([[10.0], [10.0]]),
        "a": {"b": torch.tensor([[-10.0], [-10.0]])},
        "items.0": torch.tensor([[20.0], [20.0]]),
        "items": (torch.tensor([[-20.0], [-20.0]]),),
    }
    normalizer.update(tree, 1)
    result = normalizer.normalize(tree, 1)
    assert len(normalizer.state_dict()["moments"]) == 4
    assert all(
        torch.allclose(value, torch.zeros_like(value), atol=0.05)
        for value in (result["a.b"], result["a"]["b"], result["items.0"], result["items"][0])
    )

    codec = WireCodec(1_000_000)
    restored = _normalizer()
    restored.load_state_dict(codec.decode(codec.encode(normalizer.state_dict())))
    assert torch.equal(restored.normalize(tree, 1)["a.b"], result["a.b"])


def test_legacy_state_migrates_on_first_matching_observation() -> None:
    source = _normalizer()
    source.update({"a": {"b": torch.tensor([[3.0], [5.0]])}}, 1)
    state = dict(source.state_dict())
    state.pop("path_format")
    state["moments"] = {"root.a.b": next(iter(state["moments"].values()))}
    restored = _normalizer()
    restored.load_state_dict(state)
    # A re-export before the first observation must retain the legacy format.
    reloaded = _normalizer()
    reloaded.load_state_dict(restored.state_dict())
    tree = {"a": {"b": torch.tensor([4.0])}}
    assert torch.equal(reloaded.normalize(tree, 0)["a"]["b"], source.normalize(tree, 0)["a"]["b"])
    assert reloaded.state_dict()["path_format"] == 2


def test_legacy_colliding_or_missing_paths_fail_without_mutating_state() -> None:
    source = _normalizer()
    source.update(torch.tensor([[1.0], [2.0]]), 1)
    moment = next(iter(source.state_dict()["moments"].values()))
    restored = _normalizer()
    restored.load_state_dict({"enabled": True, "clip": 100.0, "moments": {"root.a.b": moment}})
    with pytest.raises(ValueError, match="missing or ambiguous"):
        restored.update({"a.b": torch.tensor([[1.0]]), "a": {"b": torch.tensor([[2.0]])}}, 1)
    with pytest.raises(ValueError, match="missing or ambiguous"):
        restored.normalize({"other": torch.tensor([1.0])}, 0)
    assert restored.state_dict()["path_format"] == 1


def test_tuple_and_numpy_mapping_keys_round_trip_and_restore_legacy() -> None:
    tuple_key = ("a.b", (np.int64(7), "c"))
    number_key = np.int32(12)
    tree = {tuple_key: torch.tensor([[3.0], [5.0]]), number_key: torch.tensor([[9.0], [11.0]])}
    source = _normalizer()
    source.update(tree, 1)
    expected = source.normalize(tree, 1)
    restored = _normalizer()
    codec = WireCodec(1_000_000)
    restored.load_state_dict(codec.decode(codec.encode(source.state_dict())))
    assert torch.equal(restored.normalize(tree, 1)[tuple_key], expected[tuple_key])
    assert torch.equal(restored.normalize(tree, 1)[number_key], expected[number_key])

    # Old states used str(key), including the tuple's exact Python rendering.
    legacy = _normalizer()
    legacy.load_state_dict(
        {
            "enabled": True,
            "clip": 100.0,
            "moments": {
                f"root.{key}": next(iter(_trained_moments(key, value).values()))
                for key, value in tree.items()
            },
        }
    )
    actual = legacy.normalize(tree, 1)
    assert torch.equal(actual[tuple_key], expected[tuple_key])
    assert torch.equal(actual[number_key], expected[number_key])


def _trained_moments(key: object, value: torch.Tensor) -> dict:
    normalizer = _normalizer()
    normalizer.update({key: value}, 1)
    return normalizer.state_dict()["moments"]
