from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from trackmaniarl.project.scaffold_templates import _pyproject


@pytest.mark.parametrize("source", ["package", "starter", "trackmania"])
@pytest.mark.parametrize(
    ("version", "allowed"),
    [
        ("2.4.0", False),
        ("2.9.1", False),
        ("2.10.0+cpu", True),
        ("2.11.0+cu128", True),
        ("2.13.1", True),
        ("2.14.0", False),
        ("2.14.1+cpu", False),
        ("2.14.1+cu128", False),
        ("2.15.0", True),
    ],
)
def test_torch_dependency_rejects_known_vulnerable_versions(
    source: str, version: str, *, allowed: bool
) -> None:
    if source == "package":
        project = Path(__file__).resolve().parents[2] / "pyproject.toml"
        metadata = tomllib.loads(project.read_text(encoding="utf-8"))
    else:
        metadata = tomllib.loads(_pyproject("agent", template=source))
    requirements = [Requirement(item) for item in metadata["project"]["dependencies"]]
    torch = next(item for item in requirements if item.name == "torch")
    assert torch.specifier.contains(version) is allowed
