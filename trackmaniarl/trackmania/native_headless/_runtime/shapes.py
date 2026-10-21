from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NumericGroupSpec:
    b: bytes
    o: int
    n: int
    available: bool | int


@dataclass(frozen=True)
class QualifyReaderSpec:
    raw: bytes
    pid: int
    meta: dict[str, Any]
    anchor: int


@dataclass(frozen=True)
class QualifyPlanSpec:
    packet: dict[str, Any]
    before: dict[str, Any]
    action: int
    n: int
    core: Any
    codec: Any


@dataclass(frozen=True)
class StateSpec:
    packet: dict[str, Any]
    history: list[int]
    index: int
    terminal: bool


@dataclass(frozen=True)
class FeaturesSpec:
    state: dict[str, Any]
    pending: list[float]
    clock: int
    reference: Any
    projection: dict[str, Any]
    history: list[int]


@dataclass(frozen=True)
class TransitionRewardSpec:
    before: dict[str, Any]
    after: dict[str, Any]
    delta_native: int
    finish: bool


@dataclass(frozen=True)
class VerifyCreditSpec:
    events: list[bytes]
    anchor: int
    planned: list[tuple[float, float, float]]
    pending: list[float]


@dataclass(frozen=True)
class ResponseSpec:
    kind: int
    seq: int
    n: int
    epoch: int


@dataclass(frozen=True)
class RequestSpec:
    op: int
    n: int
    p: int
    k: int
    befores: Any
    afters: Any


@dataclass(frozen=True)
class ParseRecordsSpec:
    text: str
    expected_files: set[str]
    expected_map_uid: str
    mode: str
    allow_inherited_array_close: bool


@dataclass(frozen=True)
class CaptureSpec:
    ep: Path
    pid: int
    tmp: Path
    provenance: dict[str, Any]
