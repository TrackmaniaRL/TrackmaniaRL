# TrackmaniaRL diagrams

Open the [visual guide](index.html) to browse the complete set locally.

## Information design

Each figure answers one user question and is designed on a 1000 px canvas
for the guides' 900 px embedding width. Numbered stages establish reading order.
Short card headings name the component or action. Supporting notes explain
contracts, exceptions and operational consequences without crowding the flow.

Blue identifies configuration and incoming data. Teal identifies learning and
policy output. Amber identifies durability and evaluation gates. Neutral
surfaces, restrained borders and straight connectors keep attention on content.
The figures are explanatory overviews. The surrounding guides retain detailed
class names, mathematical definitions and configuration references.

SVG previews use bold headings. The editable sources use regular Helvetica
with the same text positions and wrapping. All semantic content remains editable.
Mathematical expressions are stored as LaTeX in the diagram specs. The SVG and
PNG previews render them as vector paths with Matplotlib Mathtext. The editable
Excalidraw source keeps an equivalent Unicode text representation.

Each diagram is stored as one reproducible set:

- `.spec.json`: compact source for deterministic regeneration.
- `.excalidraw`: canonical editable source that can be imported into Excalidraw.
- `-preview.png`: GitHub/PyPI-compatible raster preview.
- `-preview.svg`: scalable preview embedded in the detailed guides.
- `-preview.html`: local preview with a download of the editable scene.

The spec stores the semantic colors, zones, nodes, routed edges and notes used
by the repository renderer. After editing a spec, deterministically regenerate
the editable scene, SVG and HTML preview with:

```bash
uv run python -m docs.diagrams.render
```

The Python renderer uses stored Arial advance widths (`arial-widths.json`,
in ems with a 6% safety margin), preserving explicit line breaks and wrapping
long identifiers. SVG and Excalidraw use the same node layout and sans-serif
font family. Nodes need 16 px of vertical padding. Diamond labels must fit
inside their sloping sides. Overfilled nodes fail generation instead of
silently overflowing. Move explanatory text outside small decision diamonds.

To regenerate PNGs at the exact canvas size, use the optional browser check
(requires Node.js, the `playwright` package and Chrome):

```bash
node docs/diagrams/render_png.cjs
```

Set `DIAGRAM_BROWSER_PATH` to use another Chromium executable. The script
measures actual browser text bounds in every node and note, checks edge labels
against nodes, and rejects overlaps before saving each PNG. It runs locally
without uploading diagram contents. Regenerate PNGs whenever the SVG changes.
Manual Excalidraw adjustments must be reflected back in the spec.
Validate every scene and visually inspect both SVG and PNG at normal
documentation width before committing all rendered forms.

## Diagram set

| Subject | Editable source | Preview |
| --- | --- | --- |
| Runtime architecture | [runtime-architecture.excalidraw](runtime-architecture.excalidraw) | [PNG](runtime-architecture-preview.png) · [SVG](runtime-architecture-preview.svg) · [HTML](runtime-architecture-preview.html) |
| Checkpoint and resume | [checkpoint-resume.excalidraw](checkpoint-resume.excalidraw) | [PNG](checkpoint-resume-preview.png) · [SVG](checkpoint-resume-preview.svg) · [HTML](checkpoint-resume-preview.html) |
| Model composition and unified learner | [model-composition.excalidraw](model-composition.excalidraw) | [PNG](model-composition-preview.png) · [SVG](model-composition-preview.svg) · [HTML](model-composition-preview.html) |
| Imitation learning and RL handoff | [imitation-learning.excalidraw](imitation-learning.excalidraw) | [PNG](imitation-learning-preview.png) · [SVG](imitation-learning-preview.svg) · [HTML](imitation-learning-preview.html) |
| Demonstration timing and action labels | [demonstration-timing.excalidraw](demonstration-timing.excalidraw) | [PNG](demonstration-timing-preview.png) · [SVG](demonstration-timing-preview.svg) · [HTML](demonstration-timing-preview.html) |
| Reward decomposition and terminal PBRS | [reward-decomposition.excalidraw](reward-decomposition.excalidraw) | [PNG](reward-decomposition-preview.png) · [SVG](reward-decomposition-preview.svg) · [HTML](reward-decomposition-preview.html) |
| Recurrent replay, n-step targets and PER | [replay-sequence.excalidraw](replay-sequence.excalidraw) | [PNG](replay-sequence-preview.png) · [SVG](replay-sequence-preview.svg) · [HTML](replay-sequence-preview.html) |
| Local and remote deployment | [distributed-security.excalidraw](distributed-security.excalidraw) | [PNG](distributed-security-preview.png) · [SVG](distributed-security-preview.svg) · [HTML](distributed-security-preview.html) |
| Trackmania and Openplanet integration | [trackmania-integration.excalidraw](trackmania-integration.excalidraw) | [PNG](trackmania-integration-preview.png) · [SVG](trackmania-integration-preview.svg) · [HTML](trackmania-integration-preview.html) |
| Experimental native headless boundary | [native-headless.excalidraw](native-headless.excalidraw) | [PNG](native-headless-preview.png) · [SVG](native-headless-preview.svg) · [HTML](native-headless-preview.html) |

The committed HTML previews are local-only and expose a download of the
editable scene. They do not upload repository diagrams or their contents.
