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


@pytest.mark.parametrize(
    ("package", "bounds"),
    [
        ("pydantic", (("core",), ("2.0.0", "2.3.0"), ("2.4.0", "2.12.5"))),
        (
            "protobuf",
            (
                ("distributed", "all", "dev"),
                ("5.27.0", "5.29.5", "6.30.0", "6.31.1", "6.32.0", "6.33.4", "6.33.4.post1"),
                ("5.29.6", "6.33.5", "7.0.0"),
            ),
        ),
        ("mamba-ssm", (("mamba", "all"), ("2.2.0", "2.2.6"), ("2.3.0", "2.3.2.post1"))),
    ],
)
def test_dependency_bounds_reject_vulnerable_releases(
    package: str, bounds: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
) -> None:
    groups, blocked, allowed = bounds
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    for group in groups:
        if group == "core":
            items = project["project"]["dependencies"]
        elif group == "dev":
            items = project["dependency-groups"][group]
        else:
            items = project["project"]["optional-dependencies"][group]
        requirement = next(req for item in items if (req := Requirement(item)).name == package)
        assert all(not requirement.specifier.contains(version) for version in blocked)
        assert all(requirement.specifier.contains(version) for version in allowed)
        locked = next(entry for entry in lock["package"] if entry["name"] == package)
        assert requirement.specifier.contains(locked["version"])


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


def test_runtime_and_lock_require_safe_setuptools() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    requirement = next(
        req
        for item in project["project"]["dependencies"]
        if (req := Requirement(item)).name == "setuptools"
    )
    assert not requirement.specifier.contains("84.0.0")
    assert requirement.specifier.contains("84.1.0")
    setuptools = next(item for item in lock["package"] if item["name"] == "setuptools")
    assert Version(setuptools["version"]) >= Version("84.1.0")
