from __future__ import annotations

import json
import re
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime.shapes import ParseRecordsSpec


def _unique(pairs: Any) -> Any:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def parse_records(spec: ParseRecordsSpec) -> Any:
    text, expected_files, expected_map_uid, mode, allow_inherited_array_close = (
        spec.text,
        spec.expected_files,
        spec.expected_map_uid,
        spec.mode,
        spec.allow_inherited_array_close,
    )
    decoder = json.JSONDecoder(object_pairs_hook=_unique)
    if mode == "fork":
        start = len(text) - len(text.lstrip())
        obj, end = decoder.raw_decode(text, start)
        if not isinstance(obj, dict):
            raise ValueError("fork response must be one object")
        tail = text[end:].lstrip()
        if tail.startswith("]"):
            if not allow_inherited_array_close:
                raise ValueError("unbound array close")
            tail = tail[1:].lstrip()
        pattern = (
            "FKCOMPLETE stdout_record_or_eof 1 samples_eof 1\\s+"
            "FKTIME fork_us \\d+ first_us \\d+ done_us \\d+\\s*"
        )
        if not re.fullmatch(pattern, tail):
            raise ValueError("incomplete or additional fork payload")
        records = [obj]
    elif mode in ("parent", "plain"):
        obj = json.loads(text, object_pairs_hook=_unique)
        if not isinstance(obj, list):
            raise ValueError("original server output must be complete array")
        records = obj
    else:
        raise ValueError("unknown envelope")
    if len(records) != len(expected_files):
        raise ValueError("wrong record count")
    names = []
    for r in records:
        if not isinstance(r, dict):
            raise ValueError("record must be object")
        if r["MapUid"] != expected_map_uid or type(r["IsValid"]) is not bool:
            raise ValueError("wrong record binding or flag")
        names.append(r["FileName"])
        if "ValidatedResult" not in r:
            raise ValueError("missing computed result")
        if r["ValidatedResult"] is None and (r["IsValid"] or not isinstance(r.get("Desc"), str)):
            raise ValueError("null result without explicit invalid diagnosis")
        for key in ("ValidatedResult", "DeclaredResult"):
            result = r[key]
            if key == "DeclaredResult" and result is None:
                raise ValueError("missing declaration")
            if result is not None:
                if not isinstance(result, dict) or set(result) != {
                    "NbCheckpoints",
                    "NbRespawns",
                    "Time",
                    "Score",
                }:
                    raise ValueError("wrong result shape")
                if not all(type(v) is int for v in result.values()):
                    raise ValueError("wrong result scalar type")
    if len(set(names)) != len(names) or set(names) != set(expected_files):
        raise ValueError("wrong/duplicate filenames")
    return records


def public_record(record: Any) -> Any:
    return {
        k: record[k]
        for k in ("IsValid", "ValidatedResult", "DeclaredResult", "MapUid", "Desc")
        if k in record
    }
