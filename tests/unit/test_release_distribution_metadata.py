from __future__ import annotations

import email
import tarfile
import zipfile
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
        'Requires-Dist: numba<0.69,>=0.64; extra == "native-vision"',
        'Requires-Dist: numba>=0.64,<0.69; extra == "all"',
    ]
    headers.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    return email.message_from_string("\n".join(headers))


def test_distribution_metadata_and_attribution_are_release_ready() -> None:
    check_distribution._validate_wheel_metadata(_wheel_metadata(), Path("package.whl"), "2.0.0")
    license_text = (REPOSITORY / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2021 Edouard Geze and Yann Bouteiller" in license_text
    assert "Copyright (c) 2026 Jakub Szulc" in license_text
    assert "This repository originated from TMRL" in (REPOSITORY / "NOTICE").read_text()


@pytest.mark.parametrize("missing_extra", ["native-vision", "all"])
def test_distribution_metadata_rejects_missing_numba_extra(missing_extra: str) -> None:
    metadata = _wheel_metadata()
    requirements = metadata.get_all("Requires-Dist") or []
    del metadata["Requires-Dist"]
    for requirement in requirements:
        if f'extra == "{missing_extra}"' not in requirement:
            metadata.add_header("Requires-Dist", requirement)
    with pytest.raises(RuntimeError, match=f"numba bounds for extra {missing_extra}"):
        check_distribution._validate_wheel_metadata(metadata, Path("package.whl"), "2.0.0")


def test_distribution_metadata_rejects_wrong_numba_bounds() -> None:
    metadata = _wheel_metadata()
    requirements = metadata.get_all("Requires-Dist") or []
    del metadata["Requires-Dist"]
    for requirement in requirements:
        metadata.add_header("Requires-Dist", requirement.replace("<0.69", "<0.70"))
    with pytest.raises(RuntimeError, match="numba bounds for extra"):
        check_distribution._validate_wheel_metadata(metadata, Path("package.whl"), "2.0.0")


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


@pytest.mark.parametrize(
    "member",
    [
        "trackmaniarl/trackmania/native_headless/_runtime/portable_profile.json",
        "trackmaniarl/trackmania/native_headless/_runtime/immediate_profile.json",
        "trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/v6-to-v7.patch",
        "trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/README.md",
    ],
)
def test_wheel_rejects_missing_headless_runtime_data(
    member: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = check_distribution._wheel_required_members("2.0.0")
    assert member in required
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as package:
        for name in required - {member}:
            package.writestr(name, "placeholder")
    monkeypatch.setattr(check_distribution, "validate_archive_privacy", lambda _: None)
    with pytest.raises(RuntimeError, match="is missing") as error:
        check_distribution._validate_wheel(wheel, "2.0.0")
    assert member in str(error.value)


def test_wheel_rejects_repository_only_openplanet_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as package:
        for name in check_distribution._wheel_required_members("2.0.0"):
            package.writestr(name, "placeholder")
        package.writestr("trackmaniarl/project/openplanet/HeadlessTelemetry006/Main.as", "")
    monkeypatch.setattr(check_distribution, "validate_archive_privacy", lambda _: None)
    with pytest.raises(RuntimeError, match="repository-only Openplanet probes"):
        check_distribution._validate_wheel(wheel, "2.0.0")


@pytest.mark.parametrize("name", ["original-source-manifest.json", "build-receipt.json"])
def test_wheel_rejects_private_v7_evidence(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as package:
        for member in check_distribution._wheel_required_members("2.0.0"):
            package.writestr(member, "placeholder")
        package.writestr(
            "trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/" + name, ""
        )
    monkeypatch.setattr(check_distribution, "validate_archive_privacy", lambda _: None)
    with pytest.raises(RuntimeError, match="private V7 source/build evidence"):
        check_distribution._validate_wheel(wheel, "2.0.0")


@pytest.mark.parametrize("name", ["original-source-manifest.json", "build-receipt.json"])
def test_sdist_rejects_private_v7_evidence(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        info = tarfile.TarInfo(
            "trackmaniarl-2.0.0/trackmaniarl/trackmania/native_headless/"
            "_runtime/native_input_v7/" + name
        )
        package.addfile(info)
    monkeypatch.setattr(check_distribution, "validate_archive_privacy", lambda _: None)
    with pytest.raises(RuntimeError, match="private V7 source/build evidence"):
        check_distribution._validate_sdist(archive, "2.0.0")
