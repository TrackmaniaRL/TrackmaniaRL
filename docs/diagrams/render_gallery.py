"""Build a local navigation page for the documentation diagram set."""

from __future__ import annotations

import html
import json
from pathlib import Path

STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#152b43;
font-family:Arial,Helvetica,sans-serif}main{max-width:1180px;margin:auto;padding:56px 32px}
.brand{color:#087d74;font-size:13px;letter-spacing:1.6px;font-weight:700}
h1{font-size:44px;letter-spacing:-1.5px;margin:18px 0 16px}
.intro{color:#536579;max-width:700px;font-size:18px;line-height:1.6;margin-bottom:40px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}
a{color:inherit;text-decoration:none}article{background:white;border:1px solid #dce3ea;
border-radius:12px;overflow:hidden;transition:box-shadow .15s,transform .15s}
article:hover{box-shadow:0 8px 24px #152b4310;transform:translateY(-3px)}
a:focus-visible{outline:3px solid #285cc4;outline-offset:4px}
.thumbnail{padding:18px;background:#edf1f5;height:270px;border-bottom:1px solid #dce3ea}
img{width:100%;height:100%;object-fit:contain;background:white}
.copy{padding:24px;min-height:180px}.number{font-size:12px;color:#087d74;font-weight:700}
h2{font-size:21px;line-height:1.25;margin:12px 0 16px}.open{font-size:14px;color:#536579}
footer{margin-top:40px;color:#536579;font-size:14px;line-height:1.7}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:600px){.grid{grid-template-columns:1fr}main{padding:32px 20px}h1{font-size:34px}}
"""


def render_gallery(root: Path) -> str:
    entries = []
    for path in root.glob("*.spec.json"):
        spec = json.loads(path.read_text(encoding="utf-8"))
        entries.append((spec["eyebrow"], path.name.removesuffix(".spec.json"), spec["title"]))
    cards = []
    for index, (_, stem, title) in enumerate(sorted(entries), start=1):
        title = html.escape(title)
        cards.append(
            f'<a href="{stem}-preview.html"><article>'
            f'<div class="thumbnail"><img src="{stem}-preview.svg" alt="{title}"></div>'
            f'<div class="copy"><span class="number">GUIDE {index:02}</span>'
            f'<h2>{title}</h2><span class="open">Open the diagram &#8594;</span>'
            "</div></article></a>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>TrackmaniaRL visual guide</title><style>{STYLE}</style></head><body><main>"
        '<div class="brand">TRACKMANIARL / VISUAL GUIDE</div>'
        "<h1>The system, explained.</h1>"
        '<p class="intro">Diagrams covering training, demonstrations, value learning '
        "and game integration. Start with the runtime, then explore the part you need.</p>"
        f'<div class="grid">{"".join(cards)}</div>'
        "<footer>Open any diagram to read it at documentation size or download its editable "
        "Excalidraw source. All previews work locally.</footer></main></body></html>\n"
    )
