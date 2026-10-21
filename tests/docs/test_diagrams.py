from __future__ import annotations

import json
import runpy
import struct
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from html.parser import HTMLParser
from itertools import combinations
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIAGRAMS = ROOT / "docs" / "diagrams"
EXPECTED_STEMS = (
    "checkpoint-resume",
    "demonstration-timing",
    "distributed-security",
    "imitation-learning",
    "model-composition",
    "native-headless",
    "replay-sequence",
    "reward-decomposition",
    "runtime-architecture",
    "trackmania-integration",
)
with patch.object(sys, "path", [str(ROOT), *sys.path]):
    RENDERER = runpy.run_module("docs.diagrams.render", run_name="diagram_renderer")
BUILD_SCENE = cast(Callable[[dict[str, Any]], dict[str, Any]], RENDERER["build_scene"])
RENDER_SVG = cast(Callable[[dict[str, Any]], str], RENDERER["render_svg"])
PREVIEW_DOCUMENT = cast(
    Callable[[dict[str, Any], str, dict[str, Any], str], object], RENDERER["PreviewDocument"]
)
RENDER_HTML = cast(Callable[[object], str], RENDERER["render_html"])


def _spec(stem: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((DIAGRAMS / f"{stem}.spec.json").read_text(encoding="utf-8")),
    )


def test_diagram_set_is_deliberate() -> None:
    stems = tuple(
        path.name.removesuffix(".spec.json") for path in sorted(DIAGRAMS.glob("*.spec.json"))
    )

    assert stems == EXPECTED_STEMS


def test_svg_previews_are_valid_standalone_images() -> None:
    for stem in EXPECTED_STEMS:
        root = ET.parse(DIAGRAMS / f"{stem}-preview.svg").getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        ids = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
        for element in root.iter("{http://www.w3.org/2000/svg}use"):
            target = element.attrib["{http://www.w3.org/1999/xlink}href"]
            assert target.startswith("#")
            assert target[1:] in ids


def _assert_valid_spec(stem: str) -> None:
    spec = _spec(stem)
    groups = (spec["zones"], spec["nodes"], spec["edges"], spec.get("notes", []))
    semantic_ids = [item["id"] for group in groups for item in group]
    node_ids = {node["id"] for node in spec["nodes"]}

    assert len(semantic_ids) == len(set(semantic_ids))
    assert all(edge["from"] in node_ids and edge["to"] in node_ids for edge in spec["edges"])
    assert spec["width"] > 0
    assert spec["height"] > 0
    for edge in spec["edges"]:
        points = edge.get("points")
        assert points is None or (
            len(points) >= 2
            and all(
                len(point) == 2 and all(isinstance(value, int | float) for value in point)
                for point in points
            )
        )


def test_diagram_specs_have_valid_references_and_unique_ids() -> None:
    for stem in EXPECTED_STEMS:
        _assert_valid_spec(stem)


def _assert_previews_match_spec(stem: str) -> None:
    spec = _spec(stem)
    scene = BUILD_SCENE(spec)
    svg = RENDER_SVG(spec)

    assert json.loads((DIAGRAMS / f"{stem}.excalidraw").read_text(encoding="utf-8")) == scene
    assert (DIAGRAMS / f"{stem}-preview.svg").read_text(encoding="utf-8") == svg
    preview = PREVIEW_DOCUMENT(spec, svg, scene, stem)
    assert (DIAGRAMS / f"{stem}-preview.html").read_text(encoding="utf-8") == RENDER_HTML(preview)
    element_ids = [element["id"] for element in scene["elements"]]
    assert len(element_ids) == len(set(element_ids))


def test_committed_editable_and_web_previews_match_specs() -> None:
    for stem in EXPECTED_STEMS:
        _assert_previews_match_spec(stem)


def _assert_png_matches_canvas(stem: str) -> None:
    spec = _spec(stem)
    data = (DIAGRAMS / f"{stem}-preview.png").read_bytes()

    assert data[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert struct.unpack(">II", data[16:24]) == (spec["width"], spec["height"])


def test_png_previews_match_canvases() -> None:
    for stem in EXPECTED_STEMS:
        _assert_png_matches_canvas(stem)


class _SceneScriptParser(HTMLParser):
    """Read script raw text the same way the browser's textContent does."""

    def __init__(self) -> None:
        super().__init__()
        self.in_scene = False
        self.payload = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.in_scene = tag == "script" and dict(attrs).get("id") == "scene"

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_scene = False

    def handle_data(self, data: str) -> None:
        if self.in_scene:
            self.payload += data


def test_html_download_preserves_scene_json_and_special_characters() -> None:
    spec = _spec("runtime-architecture")
    spec["title"] = 'Quotes " & <tags> </script><script>alert(1)</script>'
    scene = BUILD_SCENE(spec)
    document = RENDER_HTML(PREVIEW_DOCUMENT(spec, RENDER_SVG(spec), scene, "test"))
    parser = _SceneScriptParser()
    parser.feed(document)

    assert json.loads(parser.payload) == scene
    assert "</script>" not in parser.payload


def test_committed_html_downloads_contain_valid_scene_json() -> None:
    for stem in EXPECTED_STEMS:
        parser = _SceneScriptParser()
        parser.feed((DIAGRAMS / f"{stem}-preview.html").read_text(encoding="utf-8"))
        assert json.loads(parser.payload) == BUILD_SCENE(_spec(stem))


def test_editable_arrow_bounds_include_every_bend() -> None:
    for stem in EXPECTED_STEMS:
        spec = _spec(stem)
        scene = BUILD_SCENE(spec)
        arrows = {item["id"]: item for item in scene["elements"] if item["type"] == "arrow"}
        for edge in spec["edges"]:
            arrow = arrows[edge["id"]]
            xs, ys = zip(*arrow["points"], strict=True)
            assert arrow["width"] == max(xs) - min(xs)
            assert arrow["height"] == max(ys) - min(ys)
            if "points" in edge:
                assert [[arrow["x"] + x, arrow["y"] + y] for x, y in arrow["points"]] == edge[
                    "points"
                ]


def test_text_stays_inside_nodes_and_notes() -> None:
    for stem in EXPECTED_STEMS:
        spec = _spec(stem)
        elements = {element["id"]: element for element in BUILD_SCENE(spec)["elements"]}
        for node in [*spec["nodes"], *spec.get("notes", [])]:
            for suffix in ("title", "formula", "detail", "text"):
                text = elements.get(f"{node['id']}-{suffix}")
                if text is None:
                    continue
                for x in (text["x"], text["x"] + text["width"]):
                    for y in (text["y"], text["y"] + text["height"]):
                        if node.get("shape") == "diamond":
                            assert (
                                abs(x - node["x"] - node["w"] / 2) / (node["w"] / 2)
                                + abs(y - node["y"] - node["h"] / 2) / (node["h"] / 2)
                            ) <= 0.94, (stem, node["id"])
                        else:
                            assert node["x"] + 7 <= x <= node["x"] + node["w"] - 7
                            assert node["y"] + 7 <= y <= node["y"] + node["h"] - 7


def test_renderer_rejects_overfilled_diamond() -> None:
    spec = _spec("imitation-learning")
    node = next(node for node in spec["nodes"] if node["id"] == "promote")
    node["detail"] = "named criteria, never implicit"
    with pytest.raises(ValueError, match="Text does not fit node 'promote'"):
        BUILD_SCENE(spec)


def test_nodes_and_notes_have_clear_separation() -> None:
    for stem in EXPECTED_STEMS:
        spec = _spec(stem)
        for a, b in combinations([*spec["nodes"], *spec.get("notes", [])], 2):
            assert (
                a["x"] + a["w"] + 8 <= b["x"]
                or b["x"] + b["w"] + 8 <= a["x"]
                or a["y"] + a["h"] + 8 <= b["y"]
                or b["y"] + b["h"] + 8 <= a["y"]
            ), (stem, a["id"], b["id"])


def test_latex_formulas_have_editable_fallbacks_and_svg_paths() -> None:
    formula_count = 0
    for stem in EXPECTED_STEMS:
        spec = _spec(stem)
        formulas = [node for node in spec["nodes"] if "formula" in node]
        formula_count += len(formulas)
        svg = RENDER_SVG(spec)
        for node in formulas:
            assert node["formula_plain"]
            assert f'data-formula="{node["id"]}-formula"' in svg
            assert f'id="{node["id"]}-formula-' in svg
    assert formula_count >= 7
