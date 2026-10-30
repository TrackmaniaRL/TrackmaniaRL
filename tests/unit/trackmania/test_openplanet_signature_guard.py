"""Static safety contracts for probes; no native code bytes are included."""

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3] / "experiments/full_qr/openplanet"


@pytest.mark.parametrize(
    ("folder", "tag", "name", "stolen"),
    [
        ("HeadlessPhysicalProbe005", "PHYSICAL005", "Physical005", 7),
        ("HeadlessInputConsumer007", "CONSUMER007", "Consumer007", 8),
    ],
)
def test_probe_fails_closed_before_hook_and_restores_verified_snapshot(  # noqa: PLR0913
    folder: str, tag: str, name: str, stolen: int
) -> None:
    source = (ROOT / folder / "Main.as").read_text(encoding="utf-8")
    assert re.search(rf'const string {tag}_HEX_SHA256 = "[0-9a-f]{{64}}";', source)
    assert not re.search(rf"\b{tag}_(?:HEX|PATTERN)\b", source)
    assert not re.search(r'"[0-9a-fA-F]{128}"', source)
    start = source.index(f"void {name}ReadSignature(")
    end = source.index("\n}", start)
    helper = source[start:end]
    assert "i < 64" in helper
    assert 'Text::Format("%02x", value)' in helper
    assert 'Text::Format("%02X", value)' in helper
    assert helper.index("Crypto::Sha256(observed)") < helper.index(
        f"g_{name}OriginalVerified = true"
    )
    arm = source[source.index(f"void {name}Arm()") :]
    first = arm.index(f"{name}ReadSignature(g_{name}Target)")
    pattern = arm.index(f"Dev::FindPattern(g_{name}VerifiedPattern)")
    recheck = arm.index(f"{name}Hex(g_{name}Target, 64) != g_{name}BeforeHex")
    hook = arm.index(f"Dev::Hook(g_{name}Target")
    assert first < pattern < recheck < hook
    assert f"g_{name}BeforeHex.SubStr(0, {stolen * 2})" in arm
    assert f"g_{name}AfterHex == g_{name}BeforeHex" in source
    assert f"Crypto::Sha256(g_{name}BeforeHex) == {tag}_HEX_SHA256" in source


def test_synthetic_window_hash_and_pattern_use_the_same_snapshot() -> None:
    data = bytes((i * 3 + 7) % 256 for i in range(64))
    lower = data.hex()
    pattern = " ".join(f"{value:02X}" for value in data)
    assert len(lower) == 128
    assert bytes.fromhex(pattern) == data
    assert (
        hashlib.sha256(lower.encode("ascii")).digest()
        != hashlib.sha256(data[:-1].hex().encode("ascii")).digest()
    )
