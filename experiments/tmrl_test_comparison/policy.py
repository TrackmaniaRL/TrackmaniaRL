"""Campaign membership; experimental learners never enter standard queues."""

from pathlib import Path

from trackmaniarl.core.spec import RunSpec

STANDARD_ALGORITHMS = ("iqn", "q", "qr", "fqf", "sac", "tqc", "redq", "ppo")
EXPERIMENTAL_ALGORITHMS = ("sd-sac",)
CAMPAIGN_ASSIGNMENTS = {
    "jakub": ("qr", "iqn"),
    "borys": ("sac",),
    "kamil": ("tqc",),
    "kuba-p": ("ppo",),
}


def require_standard_algorithm(algorithm: str) -> str:
    if algorithm in {"sd-sac", "dsac", "discrete-sac"}:
        raise ValueError(
            "SD-SAC is EXPERIMENTAL and excluded from standard pilots and full campaigns. "
            "See RUN_ALGORITHMS.md and its historical evidence link; "
            "a separate repair project needs explicit authorization."
        )
    if algorithm not in STANDARD_ALGORITHMS:
        raise ValueError(f"Unknown comparison algorithm: {algorithm}")
    return algorithm


def require_standard_config(path: Path) -> RunSpec:
    spec = RunSpec.from_yaml(path)
    require_standard_algorithm(str(spec.metadata.get("algorithm", "")))
    learner = spec.components.learner.class_path.rsplit(":", 1)[-1]
    if learner in {"StableDiscreteSoftActorCritic", "DiscreteSoftActorCritic"}:
        require_standard_algorithm("sd-sac")
    return spec
