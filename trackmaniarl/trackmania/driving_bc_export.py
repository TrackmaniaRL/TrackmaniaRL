"""Initialize IQN action preferences from the recurrent campaign1 BC classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.runtime import resolve_run
from trackmaniarl.core.spec import RunSpec


def initialize_iqn_head(head: Any, state: dict[str, torch.Tensor]) -> None:
    """Preserve classifier argmax while leaving return calibration to RL.

    The quantile embedding initially equals one (SiLU inverse of one in bias).
    Quantiles begin identical, but their IQN gradients differ and can learn a
    distribution. Centered, scaled classification logits are only an action prior.
    """
    if state["head.weight"].shape != head.advantage.weight.shape or head.value is None:
        raise ValueError("BC classifier does not match the dueling IQN action space")
    with torch.no_grad():
        head.quantile_embedding[0].weight.zero_()
        head.quantile_embedding[0].bias.fill_(1.2784645427610738)
        head.advantage.weight.copy_(0.1 * state["head.weight"])
        head.advantage.bias.copy_(0.1 * state["head.bias"])
        head.value.weight.zero_()
        head.value.bias.zero_()


def export_driving_bc(
    config_path: Path, checkpoint_path: Path, *, preserve_exploration: bool = False
) -> tuple[Path, Path]:
    config_path = config_path.resolve()
    output = checkpoint_path.resolve().with_name("iqn-bc-initialized.pt")
    if output.exists():
        raise FileExistsError(output)
    codec = TorchCheckpointCodec()
    checkpoint = codec.load(checkpoint_path)
    if checkpoint.get("schema_version") != "trackmaniarl-bc-policy-v2":
        raise ValueError("export requires a selected BC policy checkpoint")
    state = checkpoint["learner"]["model"]
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("BC checkpoint contains non-finite model tensors")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run = resolve_run(RunSpec.model_validate(config), base_dir=config_path.parent)
    try:
        learner: Any = run.learner
        learner.setup(
            {"seed": run.spec.seed, "run_dir": output.parent, "model_factory": run.model_factory}
        )
        for name in ("encoder", "temporal"):
            tensors = {
                key.removeprefix(name + "."): value
                for key, value in state.items()
                if key.startswith(name + ".")
            }
            getattr(learner.model, name).load_state_dict(tensors, strict=True)
        initialize_iqn_head(learner.model.head, state)
        learner.target_model.load_state_dict(learner.model.state_dict())
        codec.save({"learner": learner.state_dict()}, output)
    finally:
        run.logger.close()
    config["run_id"] += "-bc-start"
    kwargs = config["components"]["learner"]["kwargs"]
    kwargs["model_initialization_checkpoint"] = str(output)
    kwargs["warm_start_submodules"] = ["encoder", "temporal", "head"]
    if not preserve_exploration:
        config["distributed"].update(
            epsilon_start=0.15, epsilon_final=0.03, epsilon_decay_transitions=200000
        )
        config["components"]["learner"]["kwargs"]["exploration_epsilon"] = 0.15
    config["training"]["warmup_transitions"] = 1000
    target_config = config_path.with_stem(config_path.stem + "-warm")
    RunSpec.model_validate(config)
    target_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"IQN initialization: {output}\nRL config: {target_config}")
    return output, target_config
