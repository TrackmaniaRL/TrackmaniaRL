"""CPU admission guards and optional real CUDA learner with synthetic actors.

CPU tests never allocate a CUDA tensor. GPU tests are skipped unless the active
Torch build can use CUDA; they prove one-device plumbing and resume correctness,
not equality of CUDA and CPU floating-point updates or native-engine physics.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import headless_native_train as single
from scripts import headless_native_train_multi as multi
from tests.unit.trackmania.test_common_training_performance import _assert_exact
from tests.unit.trackmania.test_headless_native_train import components
from tests.unit.trackmania.test_headless_native_train_multi import SyntheticAdapter
from tests.unit.trackmania.test_headless_native_train_pipeline import Factory, opts
from trackmaniarl.algorithms import execution
from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy
from trackmaniarl.trackmania import common_training
from trackmaniarl.trackmania.common_training import native_only

_REAL_CUDA_AVAILABLE = torch.cuda.is_available
_REAL_VISIBLE_ACCELERATORS = execution.visible_accelerators
CUDA = pytest.mark.skipif(not _REAL_CUDA_AVAILABLE(), reason="a usable CUDA Torch build required")


@pytest.fixture
def real_cuda(monkeypatch: pytest.MonkeyPatch, _isolate_test_hardware: None) -> None:
    """Opt only these six GPU cases out of the suite's CPU hardware fixture."""
    if sys.platform != "linux":
        pytest.skip("native training uses Linux owned no-follow storage")
    if not _REAL_CUDA_AVAILABLE():
        pytest.skip("a usable CUDA Torch build required")
    monkeypatch.setattr(torch.cuda, "is_available", _REAL_CUDA_AVAILABLE)
    monkeypatch.setattr(execution, "visible_accelerators", _REAL_VISIBLE_ACCELERATORS)


@pytest.fixture(autouse=True)
def restore_execution_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.fixture
def parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    result = components(tmp_path, monkeypatch)
    monkeypatch.setattr(native_only, "NativeOnlyAdapter", SyntheticAdapter)
    SyntheticAdapter.validations, SyntheticAdapter.reject_actor = [], None
    return result


@pytest.mark.parametrize("device", ["gpu", "auto", "cuda:0", "CUDA", ""])
def test_invalid_learner_device_is_rejected_before_any_pool(
    tmp_path: Path, parts: Any, device: str
) -> None:
    factory = Factory()
    with pytest.raises(ValueError, match="explicit CPU or CUDA learner device"):
        multi.run_training(
            parts, opts(tmp_path, "invalid"), pool_factory=factory, learner_device=device
        )
    assert not factory.instances


def test_unavailable_cuda_is_rejected_before_any_pool_or_gpu_allocation(
    tmp_path: Path, parts: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    factory = Factory()
    with pytest.raises(ValueError, match="CUDA float32 learner requested but unavailable"):
        multi.run_training(
            parts, opts(tmp_path, "unavailable"), pool_factory=factory, learner_device="cuda"
        )
    assert not factory.instances


def test_cuda_context_with_wrong_workspace_is_rejected_before_learner_or_pool(
    tmp_path: Path, parts: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "synthetic-guard-only")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    learner_calls: list[Any] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        learner_calls.append((args, kwargs))
        pytest.fail("the unsafe initialized context must be rejected before learner setup")

    monkeypatch.setattr(common_training, "make_learner", forbidden)
    factory = Factory()
    with pytest.raises(ValueError, match="context predates"):
        multi.run_training(
            parts, opts(tmp_path, "wrong-workspace"), pool_factory=factory, learner_device="cuda"
        )
    assert not learner_calls
    assert not factory.instances


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_existing_gpu_cache_is_rejected_before_learner_or_pool(  # noqa: PLR0913
    tmp_path: Path,
    parts: Any,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    request: pytest.FixtureRequest,
) -> None:
    selected = opts(tmp_path, "existing-cache")
    cache = selected.output / "gpu-cache"
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    marker = outside / "evidence.txt"
    marker.write_bytes(b"unrelated cache evidence must remain unchanged")
    if kind == "directory":
        cache.mkdir()
        (cache / "owned-marker.txt").write_bytes(b"existing directory must remain unchanged")
    else:
        request.getfixturevalue("symlink_support")
        cache.symlink_to(outside, target_is_directory=True)
    learner_calls: list[Any] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        learner_calls.append((args, kwargs))
        pytest.fail("existing cache must be rejected before any learner construction")

    monkeypatch.setattr(common_training, "make_learner", forbidden)
    factory = Factory()
    with pytest.raises(ValueError, match="fresh owned output/checkpoint paths"):
        multi.run_training(parts, selected, pool_factory=factory, learner_device="cuda")
    assert not learner_calls
    assert not factory.instances
    assert sorted(path.name for path in outside.iterdir()) == ["evidence.txt"]
    assert marker.read_bytes() == b"unrelated cache evidence must remain unchanged"
    if kind == "directory":
        assert sorted(path.name for path in cache.iterdir()) == ["owned-marker.txt"]
        assert (
            cache / "owned-marker.txt"
        ).read_bytes() == b"existing directory must remain unchanged"
    else:
        assert cache.is_symlink()
        assert cache.resolve() == outside


@pytest.mark.parametrize(
    "unsafe", ["matmul-precision", "matmul-tf32", "cudnn-tf32", "nondeterministic"]
)
def test_unsafe_reported_cuda_numerics_are_rejected_before_any_pool_without_gpu(  # noqa: PLR0913
    tmp_path: Path, parts: Any, monkeypatch: pytest.MonkeyPatch, unsafe: str
) -> None:
    # Deliberately substitute a real CPU learner only to exercise the admission
    # guard on machines lacking CUDA. Real device execution has separate tests.
    original = common_training.make_learner
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "synthetic-guard-only")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic-guard-only")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))

    def cpu_guard_learner(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("execution")
        # The outer availability flag reaches the trainer's CUDA guard only.
        # CPU setup must see the real absence of CUDA and never queue a GPU RNG
        # or context initialization through the synthetic availability value.
        with monkeypatch.context() as cpu_setup:
            cpu_setup.setattr(torch.cuda, "is_available", lambda: False)
            learner = original(*args, **kwargs)
        if unsafe == "matmul-precision":
            monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "medium")
        elif unsafe == "matmul-tf32":
            monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
        elif unsafe == "cudnn-tf32":
            monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
        else:
            monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
        return learner

    monkeypatch.setattr(common_training, "make_learner", cpu_guard_learner)
    factory = Factory()
    selected = opts(tmp_path, "unsafe-" + unsafe)
    report = multi.run_training(parts, selected, pool_factory=factory, learner_device="cuda")
    assert not report["passed"]
    assert "deterministic IEEE float32" in report["error"]
    assert not factory.instances
    assert not (selected.output / "latest.pt").exists()
    assert report["transitions"] == 0
    assert report["optimizer_steps"] == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux owned native training storage")
def test_default_cpu_schedule_and_combined_training_stay_explicitly_cpu(
    tmp_path: Path, parts: Any
) -> None:
    selected = replace(opts(tmp_path, "cpu"), transitions=40)
    assert multi.multi_schedule(selected, 8) == multi.multi_schedule(
        selected, 8, learner_device="cpu"
    )
    assert "learner_execution" not in multi.multi_schedule(selected, 8)
    report = multi.run_training(
        parts, selected, actors=8, pool_factory=Factory(), batch_actions=True, overlap_learning=True
    )
    assert report["passed"], report
    assert report["transitions"] == 40
    assert report["optimizer_steps"] == 10
    assert report["learner_device"] == "cpu"
    assert "learner_execution" not in report["scheduler"]


@CUDA
@pytest.mark.usefixtures("real_cuda")
@pytest.mark.parametrize("modes", [(False, False), (True, False), (False, True), (True, True)])
def test_real_cuda_updates_keep_cpu_behavior_policy_and_pinned_numerics(  # noqa: PLR0913
    tmp_path: Path, parts: Any, monkeypatch: pytest.MonkeyPatch, modes: tuple[bool, bool]
) -> None:
    policies, learners = [], []
    original_init = DiscreteValuePolicy.__init__
    original_make = common_training.make_learner

    def capture_policy(self: Any, model: Any, device: Any, config: Any) -> None:  # noqa: PLR0913
        assert all(parameter.device.type == "cuda" for parameter in model.parameters())
        original_init(self, model, device, config)
        policies.append(self)

    def capture_learner(*args: Any, **kwargs: Any) -> Any:
        learner = original_make(*args, **kwargs)
        learners.append(learner)
        return learner

    monkeypatch.setattr(DiscreteValuePolicy, "__init__", capture_policy)
    monkeypatch.setattr(common_training, "make_learner", capture_learner)
    selected = replace(opts(tmp_path, "cuda"), transitions=40)
    report = multi.run_training(
        parts,
        selected,
        actors=8,
        pool_factory=Factory(),
        batch_actions=modes[0],
        overlap_learning=modes[1],
        learner_device="cuda",
    )
    assert report["passed"], report
    assert report["transitions"] == 40
    assert report["optimizer_steps"] == 10
    assert report["remaining_update_credit"] == 0
    assert len(policies) == len(learners) == 1
    assert learners[0].device.type == "cuda"
    assert policies[0].device.type == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in policies[0].model.parameters())
    execution = report["learner_execution"]
    assert execution["profile"] == "cuda-float32-ieee/deterministic/cpu-behavior/v1"
    assert execution["actor_device"] == "cpu"
    assert execution["precision"] == "float32"
    assert execution["matmul_precision"] == "highest"
    assert execution["matmul_allow_tf32"] is False
    assert execution["cudnn_allow_tf32"] is False
    assert execution["deterministic_algorithms"] is True
    assert execution["torch_build"] == str(torch.__version__)
    assert execution["cuda_runtime"] == torch.version.cuda
    assert execution["cublas_workspace"] == ":4096:8"
    saved = torch.load(selected.output / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["schedule"]["learner_execution"] == {
        key: execution[key] for key in saved["schedule"]["learner_execution"]
    }
    assert "cuda" in saved["learner"]["training"]["rng"]
    assert len(saved["learner"]["training"]["rng"]["cuda"]) == torch.cuda.device_count()


@CUDA
@pytest.mark.usefixtures("real_cuda")
def test_real_cuda_resume_at_physical_boundary_preserves_full_learner_rng_and_budget(
    tmp_path: Path, parts: Any
) -> None:
    modes = {"learner_device": "cuda", "batch_actions": True, "overlap_learning": True}
    full_options = replace(
        opts(tmp_path, "full"), transitions=160, episode_native_units=2000, checkpoint_every=80
    )
    full = multi.run_training(parts, full_options, actors=2, pool_factory=Factory(), **modes)
    first_options = replace(
        opts(tmp_path, "first"), transitions=160, episode_native_units=2000, checkpoint_every=80
    )
    first = multi.run_training(
        parts,
        first_options,
        actors=2,
        pool_factory=Factory(stop_round=40, stop_target=first_options.stop_file),
        **modes,
    )
    assert full["passed"], full
    assert first["passed"], first
    assert first["transitions"] == 80
    checkpoint = first_options.output / "latest.pt"
    resumed_options = replace(
        opts(tmp_path, "resumed"),
        transitions=80,
        episode_native_units=2000,
        checkpoint_every=80,
        resume=checkpoint,
        resume_sha256=single._file_sha(checkpoint),
    )
    resumed = multi.run_training(parts, resumed_options, actors=2, pool_factory=Factory(), **modes)
    assert resumed["passed"], resumed
    assert resumed["resume_loaded"]
    left = torch.load(full_options.output / "latest.pt", map_location="cpu", weights_only=False)
    right = torch.load(resumed_options.output / "latest.pt", map_location="cpu", weights_only=False)
    for key in ("learner", "sampler", "budget"):
        _assert_exact(left[key], right[key])


@CUDA
@pytest.mark.usefixtures("real_cuda")
def test_real_cuda_checkpoint_rejected_by_cpu_profile_before_any_pool(
    tmp_path: Path, parts: Any
) -> None:
    first_options = replace(opts(tmp_path, "cuda-source"), transitions=40)
    first = multi.run_training(
        parts, first_options, actors=8, pool_factory=Factory(), learner_device="cuda"
    )
    assert first["passed"], first
    checkpoint = first_options.output / "latest.pt"
    rejected_options = replace(
        opts(tmp_path, "cpu-rejected"),
        resume=checkpoint,
        resume_sha256=single._file_sha(checkpoint),
    )
    factory = Factory()
    report = multi.run_training(parts, rejected_options, actors=8, pool_factory=factory)
    assert not report["passed"]
    assert "checkpoint schema/actor/scheduler/runtime mismatch" in report["error"]
    assert not factory.instances
    assert not (rejected_options.output / "latest.pt").exists()
