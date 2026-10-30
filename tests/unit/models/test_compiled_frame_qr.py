"""Execution-only compilation must retain ordinary learner state and guards."""

import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from tests.unit._composite_value_fixtures import _batch, _fixed_quantile_model, _scalar_model
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.experiments.full_qr.compiled_frame_qr import (
    CompiledFrameQrLearner,
    _load_warmup,
)

_CUDA_AVAILABLE = torch.cuda.is_available()


def _warmup(batch: TrainingBatch, path: Path) -> dict[str, str]:
    TorchCheckpointCodec().save(
        {
            "schema": "tmrl.compiler-frame-batch/v1",
            "batch": {field.name: getattr(batch, field.name) for field in fields(batch)},
        },
        path,
    )
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_cpu_replica_needs_no_warmup_and_retains_reference_updates() -> None:
    model = _fixed_quantile_model()
    reference = DiscreteValueLearner(deepcopy(model), execution={"device": "cpu"})
    replica = CompiledFrameQrLearner(deepcopy(model), execution={"device": "cpu"})
    for learner in (reference, replica):
        learner.setup({"seed": 17})
    for _ in range(3):
        batch = _batch()
        expected, actual = reference.update(batch), replica.update(batch)
        assert actual[1] == expected[1]
        assert actual[0]["loss/total"] == expected[0]["loss/total"]
    for key, value in reference.model.state_dict().items():
        torch.testing.assert_close(value, replica.model.state_dict()[key], rtol=0, atol=0)
    assert replica.execution_manifest()["forward_loss_compilation"] == "eager"
    invalid = CompiledFrameQrLearner(_scalar_model(), execution={"device": "cpu"})
    with pytest.raises(TypeError, match="fixed-quantile stateless"):
        invalid.setup({})


def test_pinned_warmup_roundtrip_rejects_tampering_and_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "warmup.pt"
    batch = _batch()
    pin = _warmup(batch, path)
    actual = _load_warmup(pin)
    torch.testing.assert_close(actual.observations, batch.observations, rtol=0, atol=0)
    with pytest.raises(ValueError, match="hash changed"):
        _load_warmup({**pin, "sha256": "0" * 64})
    TorchCheckpointCodec().save({"schema": "tmrl.compiler-frame-batch/v1", "batch": {}}, path)
    pin["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="every TrainingBatch field"):
        _load_warmup(pin)


@pytest.mark.skipif(sys.platform != "linux" or not _CUDA_AVAILABLE, reason="Linux CUDA compiler")
def test_cuda_warmup_preserves_weights_then_matches_updates_and_rejects_layout_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    batch = _batch(64)
    pin = _warmup(batch, tmp_path / "warmup.pt")
    model = _fixed_quantile_model()
    model.head.dense_cuda_selected = True
    reference = DiscreteValueLearner(
        deepcopy(model),
        execution={"device": "cuda", "precision": "float32", "torch_threads": 2},
        target_update_interval=2,
    )
    compiled = CompiledFrameQrLearner(
        deepcopy(model),
        execution={"device": "cuda", "precision": "float32", "torch_threads": 2},
        compiler_warmup=pin,
        target_update_interval=2,
    )
    reference.setup({"seed": 17})
    compiled.setup({"seed": 17})
    assert compiled.update_count == 0
    assert not compiled.optimizer.state
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(RuntimeError, match="same thread"),
    ):
        executor.submit(compiled.update, batch).result()
    assert compiled.update_count == 0
    for key, value in reference.model.state_dict().items():
        torch.testing.assert_close(value, compiled.model.state_dict()[key], rtol=0, atol=0)
    for _ in range(3):
        expected, _ = reference.update(batch)
        actual, _ = compiled.update(batch)
        assert actual["loss/total"] == pytest.approx(expected["loss/total"], rel=1e-4, abs=1e-6)
    for key, value in reference.model.state_dict().items():
        torch.testing.assert_close(value, compiled.model.state_dict()[key], rtol=1e-4, atol=1e-5)
    count = compiled.update_count
    changed = replace(
        batch, importance_weights=None if batch.importance_weights is not None else torch.ones(64)
    )
    with pytest.raises(ValueError, match="warmup shape/dtype/weight layout"):
        compiled.update(changed)
    assert compiled.update_count == count
    restored = DiscreteValueLearner(deepcopy(model), execution={"device": "cpu"})
    restored.setup({})
    restored.load_state_dict(compiled.state_dict())
    assert restored.update_count == count
