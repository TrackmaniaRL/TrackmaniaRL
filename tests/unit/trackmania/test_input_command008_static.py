"""Pure offline parser and fail-closed checks; no PE/game execution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from scripts import input_command008_static as static
from scripts.headless_windows_native_inventory import PE


def test_decode_retains_full_original_stack_store_and_ignores_header() -> None:
    text = (
        "Disassembly of section .text:\n"
        "   140841553:\t48 89 84 24 c8 00 00 00 \tmov    QWORD PTR [rsp+0xc8],rax\n"
        "   14084155b:\te8 d0 ce f7 ff       \tcall   0x1407be430\n"
    )
    rows = static.decoded_rows(text, 0x140000000)
    assert rows == [
        {
            "rva": "0x841553",
            "hex": "48898424c8000000",
            "instruction": "mov    QWORD PTR [rsp+0xc8],rax",
        },
        {"rva": "0x84155b", "hex": "e8d0cef7ff", "instruction": "call   0x1407be430"},
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
