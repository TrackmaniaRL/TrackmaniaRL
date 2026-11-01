"""Offline checks for runnable documentation and generated project filenames."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from examples.game_free import make_spec
from tests.docs.test_documentation_examples import ROOT, _markdown_anchors
from trackmaniarl.project.scaffold import create_project

CONFIG_NAME = re.compile(r"(?<![\w-])run(?:-[a-zA-Z0-9-]+)?\.ya?ml\b")
# Families explicitly listed in the algorithms guide for the wildcard names.
ALGORITHMS = ("q", "qr", "iqn", "fqf", "sac", "redq", "tqc", "sd-sac", "ppo")


def test_legacy_documentation_index_anchors_are_preserved() -> None:
    anchors = _markdown_anchors(ROOT / "readme/README.md")
    assert {
        "start-here",
        "from-setup-to-an-evaluated-policy",
        "extending-and-maintaining-the-library",
    } <= anchors


def test_scaffold_declaration_scope_ignores_user_configs() -> None:
    text = "Custom `run-experiment.yaml`.\nBEGIN\nUse `configs/run-missing.yaml`.\nEND\n"
    section = _declaration(text, "BEGIN", "END")
    assert _missing_generated_configs(section, {"run.yaml"}) == [(2, "run-missing.yaml")]


def test_documented_family_list_is_checked_against_real_filenames() -> None:
    generated = {"run-sac-lidar.yaml", "run-sd-sac-lidar.yaml"}
    assert _missing_family_configs("`sac`, `discrete-sac`", ("lidar",), generated) == [
        "run-discrete-sac-lidar.yaml"
    ]


def _declaration(markdown: str, start: str, end: str) -> str:
    assert markdown.count(start) == 1, f"expected one declaration marker: {start}"
    section = markdown.split(start, 1)[1]
    assert end in section, f"missing end marker: {end}"
    return section.split(end, 1)[0]


def _missing_family_configs(
    family_list: str, suffixes: tuple[str, ...], generated: set[str]
) -> list[str]:
    families = re.findall(r"`([^`]+)`", family_list)
    assert families, "algorithm family list is empty"
    return [
        name
        for family in families
        for suffix in suffixes
        if (name := f"run-{family}-{suffix}.yaml") not in generated
    ]


def _missing_generated_configs(markdown: str, generated: set[str]) -> list[tuple[int, str]]:
    failures = []
    for number, line in enumerate(markdown.splitlines(), 1):
        for match in CONFIG_NAME.finditer(line):
            pattern = match.group()
            names = (
                [pattern.replace("ALGORITHM", algorithm) for algorithm in ALGORITHMS]
                if "-ALGORITHM-" in pattern
                else [pattern]
            )
            failures.extend((number, name) for name in names if name not in generated)
    return failures


def _python_block(markdown: str, symbol: str) -> str:
    blocks = re.findall(r"^```python\s*\n(.*?)^```\s*$", markdown, re.MULTILINE | re.DOTALL)
    selected = [block for block in blocks if symbol in block]
    assert len(selected) == 1, (
        f"expected one Python block containing {symbol!r}, got {len(selected)}"
    )
    return selected[0]


def test_python_block_extracts_documented_code_and_rejects_missing() -> None:
    markdown = "Text\n```python\nvalue = 42\n```\n"
    assert _python_block(markdown, "value = 42") == "value = 42\n"
    with pytest.raises(AssertionError, match="expected one Python block"):
        _python_block(markdown, "missing")
    with pytest.raises(AssertionError, match="expected one Python block"):
        _python_block(markdown + markdown, "value = 42")


def test_generated_config_check_reports_stale_names_with_line_numbers(tmp_path: Path) -> None:
    document = tmp_path / "guide.md"
    document.write_text("Use `run.yaml`.\nThen use `run-missing-vision.yaml`.\n", encoding="utf-8")
    assert _missing_generated_configs(document.read_text(encoding="utf-8"), {"run.yaml"}) == [
        (2, "run-missing-vision.yaml")
    ]


def test_generated_config_check_detects_wrong_case() -> None:
    assert _missing_generated_configs("Use `run-PPO.yaml`", {"run-ppo.yaml"}) == [
        (1, "run-PPO.yaml")
    ]


def test_generated_config_check_expands_algorithm_filename_patterns() -> None:
    generated = {
        f"run-{algorithm}-vision.yaml"
        for algorithm in ("q", "qr", "iqn", "fqf", "sac", "redq", "tqc", "sd-sac")
    }
    assert _missing_generated_configs("Use `run-ALGORITHM-vision.yaml`", generated) == [
        (1, "run-ppo-vision.yaml")
    ]


def test_scaffold_filenames_in_generated_configuration_guides(tmp_path: Path) -> None:
    project = create_project(tmp_path / "agent", "agent", template="trackmania")
    generated = {path.name for path in project.glob("*.yaml")}
    assert "run.yaml" in generated
    declarations = (
        ("readme/quickstart.md", "For continuous off-policy control", "The [vision and PPO guide]"),
        ("readme/vision.md", "The Trackmania template generates", "Use any filename"),
        ("readme/algorithms.md", "## Generated Trackmania configurations", "## Support matrix"),
    )
    failures = []
    for path, start, end in declarations:
        markdown = (ROOT / path).read_text(encoding="utf-8")
        section = _declaration(markdown, start, end)
        failures.extend(
            f"{path} (declaration line {line}): {name} is not generated"
            for line, name in _missing_generated_configs(section, generated)
        )
    assert not failures, "\n" + "\n".join(failures)


def test_documented_algorithm_families_match_scaffold(tmp_path: Path) -> None:
    project = create_project(tmp_path / "agent", "agent", template="trackmania")
    generated = {path.name for path in project.glob("*.yaml")}
    vision = (ROOT / "readme/vision.md").read_text(encoding="utf-8")
    algorithms = (ROOT / "readme/algorithms.md").read_text(encoding="utf-8")
    lidar = _declaration(vision, "`run-ALGORITHM-lidar-vision.yaml` for ", ". For example:")
    camera = _declaration(algorithms, "with `q`, `qr`,", " as algorithm names")
    camera = "`q`, `qr`," + camera
    expected = {
        match.group(1)
        for name in generated
        if (match := re.fullmatch(r"run-(.+)-lidar\.yaml", name))
    }
    for names, suffixes in ((lidar, ("lidar", "lidar-vision")), (camera, ("vision",))):
        assert not _missing_family_configs(names, suffixes, generated)
        assert set(re.findall(r"`([^`]+)`", names)) == expected


def _run_python(code: str, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), str(cwd / "src"), environment.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("document", ["README.md", "readme/python-quickstart.md"])
def test_documented_training_snippet_runs_with_toy_environment(
    document: str, tmp_path: Path
) -> None:
    spec = make_spec(tmp_path / "artifacts").model_dump(mode="json")
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    code = _python_block((ROOT / document).read_text(encoding="utf-8"), "Trainer(run).train()")
    result = _run_python(code, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Training finished: transitions=32, updates=28, episodes=4" in result.stdout
    assert list((tmp_path / "artifacts" / "python-example").rglob("*.pt"))


def test_python_quickstart_trains_generated_starter_project(tmp_path: Path) -> None:
    project = create_project(tmp_path / "agent", "agent", template="starter")
    markdown = (ROOT / "readme/python-quickstart.md").read_text(encoding="utf-8")
    result = _run_python(_python_block(markdown, "Trainer(run).train()"), project)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Training finished: transitions=64" in result.stdout
    assert list((project / "artifacts" / "starter").rglob("*.pt"))


def test_documented_feature_pipeline_handles_synthetic_transition(tmp_path: Path) -> None:
    markdown = (ROOT / "readme/sdk.md").read_text(encoding="utf-8")
    code = (
        _python_block(markdown, "class SpeedFeaturePipeline:")
        + """
pipeline = SpeedFeaturePipeline()
transition = Transition({'speed': 3.0}, 2, 1.0, {'speed': 4.0}, False, True)
batch = pipeline.collate([transition])
np.testing.assert_array_equal(batch['observations'], [[3.0]])
np.testing.assert_array_equal(batch['next_observations'], [[4.0]])
assert batch['actions'].tolist() == [2]
assert batch['rewards'].tolist() == [1.0]
assert batch['terminated'].tolist() == [False]
assert batch['truncated'].tolist() == [True]
assert batch['_trackmaniarl_batch_collated'] is True
"""
    )
    result = _run_python(code, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["game_free.py", "custom_model.py"])
def test_documented_standalone_examples_run_without_game(name: str, tmp_path: Path) -> None:
    script = ROOT / "examples" / name
    args = ["--output", str(tmp_path / "artifacts")] if name == "game_free.py" else []
    code = (
        "import runpy, sys; sys.argv = sys.argv[1:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    result = _run_python(code, tmp_path, str(script), *args)
    assert result.returncode == 0, result.stdout + result.stderr
    if name == "game_free.py":
        assert "Training finished: transitions=32, updates=28, episodes=4" in result.stdout
        assert list((tmp_path / "artifacts" / "python-example").rglob("*.pt"))
    else:
        assert "(2, 3)" in result.stdout
