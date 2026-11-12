from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from trackmaniarl.project.scaffold_templates import _pyproject


@pytest.mark.parametrize(
    ("package", "bounds"),
    [
        ("grpcio", (("distributed", "all", "dev"), ("1.82.1", "1.84.0"), "1.84.1")),
        (
            "certifi",
            (("wandb", "orchestrator", "all", "dev"), ("2026.2.25", "2026.10.18"), "2026.10.19"),
        ),
        ("urllib3", (("wandb", "orchestrator", "all", "dev"), ("2.6.3", "2.7.0"), "2.8.0")),
    ],
)
def test_optional_dependency_bounds_and_lock_reject_vulnerable_versions(
    package: str, bounds: tuple[tuple[str, ...], tuple[str, ...], str]
) -> None:
    groups, blocked, allowed = bounds
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)
    with (root / "uv.lock").open("rb") as source:
        lock = tomllib.load(source)

    for group in groups:
        items = (
            project["dependency-groups"][group]
            if group == "dev"
            else project["project"]["optional-dependencies"][group]
        )
        requirement = next(req for item in items if (req := Requirement(item)).name == package)
        assert all(not requirement.specifier.contains(version) for version in blocked)
        assert requirement.specifier.contains(allowed)

    locked = next(entry for entry in lock["package"] if entry["name"] == package)
    assert Version(locked["version"]) >= Version(allowed)


@pytest.mark.parametrize("source", ["package", "starter", "trackmania"])
def test_build_backend_does_not_allow_affected_setuptools(source: str) -> None:
    if source == "package":
        project = Path(__file__).resolve().parents[2] / "pyproject.toml"
        metadata = tomllib.loads(project.read_text(encoding="utf-8"))
    else:
        metadata = tomllib.loads(_pyproject("agent", template=source))
    requirement = next(
        req
        for item in metadata["build-system"]["requires"]
        if (req := Requirement(item)).name == "setuptools"
    )
    assert not requirement.specifier.contains("84.0.0")
    assert requirement.specifier.contains("84.1.0")
