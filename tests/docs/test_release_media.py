from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def test_neural_gif_is_bounded_looping_and_has_no_private_metadata() -> None:
    path = ROOT / "docs/assets/trackmaniarl-neural-flow.gif"
    assert path.stat().st_size < 10 * 1024**2
    with Image.open(path) as image:
        assert image.size == (400, 225)
        assert image.n_frames == 1263
        assert image.info["loop"] == 0
        duration = 0
        for index in range(image.n_frames):
            image.seek(index)
            assert not {"comment", "exif", "xmp", "icc_profile"} & image.info.keys()
            assert not image.getexif()
            assert image.info["duration"] in {30, 40}
            duration += image.info["duration"]
        assert duration == 42100


def test_neural_media_documentation_keeps_provenance_and_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/neural-flow-media.md").read_text(encoding="utf-8")
    film = (ROOT / "docs/activation-film.md").read_text(encoding="utf-8")
    readme_prose = " ".join(readme.split())
    film_prose = " ".join(film.split())
    assert (
        "https://raw.githubusercontent.com/TrackmaniaRL/TrackmaniaRL/"
        "main/docs/assets/trackmaniarl-neural-flow.gif"
    ) in readme
    assert (
        "https://raw.githubusercontent.com/TrackmaniaRL/TrackmaniaRL/"
        "main/docs/assets/trackmaniarl-logo.png"
    ) in readme
    assert "reports approximately 12 wall-clock hours" in readme_prose
    assert "capturing and rendering the film did not update its" in readme_prose
    assert "not feature attribution" in readme_prose
    assert "map-specific" in readme
    assert "No additional training or" in film_prose
    assert "IdentityTemporalCore" in film_prose
    assert "CONTEXT / 29" in film_prose
    assert "-map_metadata -1" in guide
    assert "No seek, trim or speed changes" in guide
    assert "fps=30,scale=400" in guide
    assert (ROOT / "docs/activation-film.md").is_file()
