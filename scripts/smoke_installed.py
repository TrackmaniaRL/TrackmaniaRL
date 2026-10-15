"""Exercise an installed release from outside the checkout, including a starter.

Run only in a disposable environment: this installs the generated extension.
Requires uv on PATH but no game, GPU or optional runtime integrations.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(arguments: list[str], cwd: Path) -> str:
    result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        # Captured output can contain local account paths or configuration values.
        raise RuntimeError(f"Installed release smoke command failed with exit {result.returncode}")
    return result.stdout


def main() -> None:
    import trackmaniarl

    checkout = Path(__file__).resolve().parents[1]
    installed = Path(trackmaniarl.__file__).resolve()
    if installed.is_relative_to(checkout / "trackmaniarl"):
        raise RuntimeError("Smoke test requires an installed wheel, not the source checkout")
    assert trackmaniarl.__version__ == importlib.metadata.version("TrackmaniaRL")
    for namespace in (
        "core",
        "algorithms",
        "models",
        "trackmania",
        "experiments",
        "observability",
        "distributed",
    ):
        importlib.import_module(f"trackmaniarl.{namespace}")
    with tempfile.TemporaryDirectory(prefix="trackmaniarl-release-") as temporary:
        work = Path(temporary)
        cli = [sys.executable, "-m", "trackmaniarl"]
        _run([*cli, "--help"], work)
        _run([*cli, "--version"], work)
        _run([*cli, "init", "starter", "--template", "starter"], work)
        _run([*cli, "init", "game", "--template", "trackmania"], work)
        starter = work / "starter"
        _run(
            [
                "uv",
                "--no-config",
                "pip",
                "install",
                "--no-deps",
                "--python",
                sys.executable,
                str(starter),
            ],
            work,
        )
        _run([*cli, "inspect-config", "run.yaml"], starter)
        _run([*cli, "validate", "run.yaml"], starter)
        _validate_game_configurations(work / "game")
        invalid = subprocess.run(
            [*cli, "unknown-command"],
            cwd=work,
            capture_output=True,
            text=True,
        )
        assert invalid.returncode == 2
        print(
            json.dumps(
                {
                    "version": trackmaniarl.__version__,
                    "installed_import": True,
                    "help": True,
                    "templates": 2,
                    "inspect": True,
                    "starter_validation": True,
                    "game_configurations_validated": 7,
                    "invalid_command_exit": 2,
                }
            )
        )


def _validate_game_configurations(directory: Path) -> None:
    from trackmaniarl.core.runtime import resolve_run
    from trackmaniarl.core.runtime_validation import validate_resolved_run
    from trackmaniarl.core.spec import RunSpec

    for name in (
        "run",
        "run-sac",
        "run-redq",
        "run-tqc",
        "run-sd-sac",
        "run-ppo",
        "run-ppo-vision",
    ):
        spec = RunSpec.from_yaml(directory / f"{name}.yaml")
        # Keep shipped models and wiring, bound only the offline smoke workload.
        training = spec.training.model_copy(
            update={"batch_size": 2, "n_step": 1, "sequence_length": 4 if "ppo" in name else 1}
        )
        learner = spec.components.learner
        execution = {"device": "cpu", "precision": "float32"}
        learner = learner.model_copy(update={"kwargs": {**learner.kwargs, "execution": execution}})
        components = spec.components.model_copy(update={"learner": learner})
        run = resolve_run(
            spec.model_copy(update={"training": training, "components": components}),
            base_dir=directory,
        )
        try:
            validate_resolved_run(run)
        finally:
            run.logger.close()


if __name__ == "__main__":
    main()
