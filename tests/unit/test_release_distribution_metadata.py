from __future__ import annotations

import email
from email.message import Message
from pathlib import Path

import pytest

from scripts import check_distribution

REPOSITORY = Path(__file__).resolve().parents[2]


def _wheel_metadata(*requirements: str) -> Message:
    headers = [
        "Name: TrackmaniaRL",
        "Version: 2.0.0",
        "Requires-Python: <3.13,>=3.12",
        "Provides-Extra: all",
        "Provides-Extra: distributed",
        "Provides-Extra: mamba",
        "Provides-Extra: native-vision",
        "Provides-Extra: orchestrator",
        "Provides-Extra: trackmania",
        "Provides-Extra: vision",
        "Provides-Extra: wandb",
        'Requires-Dist: libevdev>=0.13; sys_platform == "linux" and extra == "trackmania"',
    ]
    headers.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    return email.message_from_string("\n".join(headers))


def test_distribution_metadata_and_attribution_are_release_ready() -> None:
    check_distribution._validate_wheel_metadata(_wheel_metadata(), Path("package.whl"), "2.0.0")
    license_text = (REPOSITORY / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2021 Edouard Geze and Yann Bouteiller" in license_text
    assert "Copyright (c) 2026 Jakub Szulc" in license_text
    assert "This repository originated from TMRL" in (REPOSITORY / "NOTICE").read_text()


@pytest.mark.parametrize(
    ("requirement", "message"),
    [
        ("vgamepad>=0.1.0", "vgamepad"),
        ("example-package @ git+https://example.test/package", "direct URL"),
    ],
)
def test_distribution_metadata_rejects_unsafe_public_requirements(
    requirement: str, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        check_distribution._validate_wheel_metadata(
            _wheel_metadata(requirement), Path("package.whl"), "2.0.0"
        )
