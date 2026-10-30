"""Pure offline parser and fail-closed checks; no PE/game execution."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import cast

import pytest

from scripts import input_command008_static as static
from scripts.headless_windows_native_inventory import PE


def test_decode_retains_synthetic_instructions_and_ignores_header() -> None:
    text = (
        "Disassembly of section .text:\n"
        "   140001000:\t90 \tnop\n"
        "   140001001:\tb8 01 00 00 00 \tmov    eax,0x1\n"
    )
    rows = static.decoded_rows(text, 0x140000000)
    assert rows == [
        {
            "rva": "0x1000",
            "hex": "90",
            "instruction": "nop",
        },
        {"rva": "0x1001", "hex": "b801000000", "instruction": "mov    eax,0x1"},
    ]


def test_empty_or_malformed_disassembly_cannot_count_as_instruction_proof() -> None:
    with pytest.raises(ValueError, match="no decoded instructions"):
        static.decoded_rows("not a decoded instruction or an empty .pdata fragment", 0x140000000)


@pytest.mark.parametrize("size", [0, 16385])
def test_window_bounds_fail_before_any_tool_invocation(size: int) -> None:
    with pytest.raises(ValueError, match="at most16KiB"):
        static.disassemble(cast(PE, None), 0x841500, 0x841500 + size)


def test_unpinned_binary_cannot_receive_decoded_instruction_evidence() -> None:
    image = cast(PE, SimpleNamespace(raw=b"unapproved private image"))
    with pytest.raises(ValueError, match="identity mismatch"):
        static.pinned_prefixes(image)


@pytest.mark.parametrize("corrupt", [None, "pre", "earlier", "short"])
def test_prefix_pins_compare_complete_synthetic_windows(
    monkeypatch: pytest.MonkeyPatch, corrupt: str | None
) -> None:
    first = bytes(range(64))
    second = bytes(range(64, 128))
    monkeypatch.setattr(static, "PRE_SHA256", hashlib.sha256(first).hexdigest())
    monkeypatch.setattr(static, "EARLIER_SHA256", hashlib.sha256(second).hexdigest())
    raw = bytearray(first + second)
    if corrupt == "pre":
        raw[0] ^= 1
    elif corrupt == "earlier":
        raw[64] ^= 1
    elif corrupt == "short":
        del raw[-1]
    monkeypatch.setattr(static, "EXPECTED_SHA256", hashlib.sha256(raw).hexdigest())
    image = cast(
        PE,
        SimpleNamespace(
            raw=bytes(raw),
            offset=lambda address: 0 if address == static.PRE_RVA else 64,
        ),
    )
    if corrupt:
        with pytest.raises(ValueError, match="prefix mismatch"):
            static.pinned_prefixes(image)
    else:
        digest, actual_first, actual_second = static.pinned_prefixes(image)
        assert digest == hashlib.sha256(raw).hexdigest()
        assert bytes.fromhex(actual_first) == first
        assert bytes.fromhex(actual_second) == second
