"""Pure Windows008 protocol assessment; never controls a game or qualifies live ACK.

The producer must separately prove writer/consumer phase, source identity,
thread-safe publication and cleanup. A passing synthetic ledger is only an
offline protocol result. No production gate imports this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

WIRE = "tmrl.windows-command-physics/v8"
NEUTRAL = (0, 0, 0)
GATE_NAMES = ("vehicle176c", "vehicle1370", "selector1600", "word1768", "float1374")
Words3 = tuple[int, int, int]


class ProtocolError(ValueError):
    """Retained evidence violates the separately declared experimental schedule."""


def _integer(  # noqa: PLR0913 -- named scalar and its two independent literal bounds
    value: Any, name: str, minimum: int = 0, maximum: int = 0xFFFFFFFF
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProtocolError(name + ": literal bounded integer required")
    return value


def _digest(value: Any, name: str, length: int = 64) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(name + ": literal lowercase hexadecimal digest required")


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _words(words: tuple[int, ...], count: int) -> tuple[float, ...]:
    if type(words) is not tuple or len(words) != count:
        raise ProtocolError("literal fixed-size word tuple required")
    for word in words:
        _integer(word, "float32 word")
    values = struct.unpack("<" + "f" * count, struct.pack("<" + "I" * count, *words))
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("nonfinite literal physical/control word")
    return tuple(values)


def _sgb(words: Words3) -> None:
    steer, gas, brake = _words(words, 3)
    if not -1 <= steer <= 1 or gas not in (0, 1) or brake not in (0, 1):
        raise ProtocolError("literal SGB outside the finite physical input domain")
    # The physical channels are float32; integer boolean1 is a tiny float.
    if words[1] not in (0, 0x3F800000) or words[2] not in (0, 0x3F800000):
        raise ProtocolError("actual pedals require exact float32 zero/one words")


@dataclass(frozen=True)
class Scope:
    nonce: str
    reset_generation: int
    owner_sha256: str

    def __post_init__(self) -> None:
        _digest(self.nonce, "session nonce", 32)
        _integer(self.reset_generation, "reset generation", 1)
        _digest(self.owner_sha256, "arm owner identity")


@dataclass(frozen=True)
class GateMask:
    name: str
    mask: int
    expected: int

    def __post_init__(self) -> None:
        if self.name not in GATE_NAMES:
            raise ProtocolError("explicit pinned normal-gate name required")
        _integer(self.mask, "normal gate mask", 1)
        _integer(self.expected, "normal gate expected bits")
        if self.name == "word1768" and self.mask > 0xFFFF:
            raise ProtocolError("word1768 has an explicit uint16 source width")
        if self.expected & ~self.mask:
            raise ProtocolError("expected normal-gate bits exceed declared mask")


@dataclass(frozen=True)
class Boundary:
    scope: Scope
    sequence: int
    r13_raw: int
    pose_words: tuple[int, ...]
    gate_words: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.scope) is not Scope:
            raise ProtocolError("actual typed scope required")
        _integer(self.sequence, "consumer boundary sequence", maximum=256)
        _integer(self.r13_raw, "literal R13D")
        _words(self.pose_words, 10)
        if type(self.gate_words) is not tuple or len(self.gate_words) != len(GATE_NAMES):
            raise ProtocolError("all five literal gate words required")
        for word in self.gate_words:
            _integer(word, "literal gate word")
        _integer(self.gate_words[3], "word1768 uint16 value", maximum=0xFFFF)

    @property
    def sha256(self) -> str:
        return _hash({"protocol": WIRE, "boundary": asdict(self)})


@dataclass(frozen=True)
class Command:
    scope: Scope
    command_id: int
    start_sequence: int
    n: int
    sgb_words: tuple[Words3, ...]
    source_boundary_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.scope) is not Scope:
            raise ProtocolError("actual typed scope required")
        _integer(self.command_id, "command ID", 1, 64)
        _integer(self.start_sequence, "command start boundary", maximum=255)
        if type(self.n) is not int or self.n not in (1, 5):
            raise ProtocolError("explicit N1/N5 required")
        if type(self.sgb_words) is not tuple or len(self.sgb_words) != self.n:
            raise ProtocolError("exactly N immutable SGB records required")
        for words in self.sgb_words:
            _sgb(words)
        if self.source_boundary_sha256 is not None:
            _digest(self.source_boundary_sha256, "policy source boundary")

    @property
    def sha256(self) -> str:
        return _hash({"protocol": WIRE, "command": asdict(self)})


@dataclass(frozen=True)
class Publication:
    command: Command
    queued_after_sequence: int
    payload_sha256: str

    def __post_init__(self) -> None:
        _integer(self.queued_after_sequence, "publication boundary", -1, 255)
        _digest(self.payload_sha256, "published command payload")
        if self.payload_sha256 != self.command.sha256:
            raise ProtocolError("published payload differs from immutable command")


@dataclass(frozen=True)
class Consumption:
    boundary_sha256: str
    execution_ordinal: int
    phase: Literal["bootstrap", "neutral-prefix", "action", "neutral-tail"]
    command_id: int | None
    command_offset: int
    payload_sha256: str | None
    before_sgb_words: Words3
    written_sgb_words: Words3
    readback_sgb_words: Words3
    consumer_sgb_words: Words3

    def __post_init__(self) -> None:
        _digest(self.boundary_sha256, "actual consumer boundary")
        _integer(self.execution_ordinal, "queue execution ordinal", maximum=255)
        _integer(self.command_offset, "command or administrative offset", maximum=255)
        if self.phase not in ("bootstrap", "neutral-prefix", "action", "neutral-tail"):
            raise ProtocolError("explicit execution phase required")
        if self.command_id is not None:
            _integer(self.command_id, "execution command ID", 1, 64)
        if self.payload_sha256 is not None:
            _digest(self.payload_sha256, "executed payload")
        _words(self.before_sgb_words, 3)
        for words in (
            self.written_sgb_words,
            self.readback_sgb_words,
            self.consumer_sgb_words,
        ):
            _sgb(words)


@dataclass(frozen=True)
class Profile:
    scope: Scope
    mode: Literal["immutable-preloaded-tape", "fixed-prefix-host"]
    prefix_intervals: int
    bootstrap_intervals: int
    tail_intervals: int
    normal_gate_masks: tuple[GateMask, ...]
    gate_source_sha256: str

    def __post_init__(self) -> None:
        if type(self.scope) is not Scope:
            raise ProtocolError("actual typed scope required")
        _integer(self.prefix_intervals, "fixed neutral prefix", maximum=32)
        _integer(self.bootstrap_intervals, "explicit administrative bootstrap", maximum=32)
        _integer(self.tail_intervals, "explicit administrative tail", maximum=16)
        if self.mode == "immutable-preloaded-tape":
            if self.prefix_intervals != 0:
                raise ProtocolError("preloaded profile has no implicit policy prefix")
        elif self.mode == "fixed-prefix-host":
            if self.prefix_intervals == 0:
                raise ProtocolError("external host requires an explicit positive fixed prefix")
        else:
            raise ProtocolError("separate explicit execution profile required")
        if (
            type(self.normal_gate_masks) is not tuple
            or any(type(gate) is not GateMask for gate in self.normal_gate_masks)
            or tuple(gate.name for gate in self.normal_gate_masks) != GATE_NAMES
        ):
            raise ProtocolError("all five pinned normal-gate masks in source order required")
        _digest(self.gate_source_sha256, "independent gate source proof")


@dataclass(frozen=True)
class ProtocolAck:
    command_id: int
    payload_sha256: str
    decision_boundary_sha256: str
    endpoint_boundary_sha256: str
    prefix_intervals: int
    action_intervals: int
    raw_r13_delta: int


@dataclass(frozen=True)
class ProtocolAssessment:
    acknowledgements: tuple[ProtocolAck, ...]
    bootstrap_intervals: int
    tail_intervals: int
    total_intervals: int
    total_raw_r13_delta: int
    origin: str = field(
        default="offline-protocol-assessment; producer authenticity not established", init=False
    )
    physical_execution_qualified: bool = field(default=False, init=False)
    paired_training_qualified: bool = field(default=False, init=False)


def verify_episode(  # noqa: PLR0913 -- independent contract, issue, boundary and consumer evidence
    profile: Profile,
    publications: tuple[Publication, ...],
    boundaries: tuple[Boundary, ...],
    consumptions: tuple[Consumption, ...],
) -> ProtocolAssessment:
    """Assess a complete raw ledger without generating or guessing any samples.

    Every interval has two real producer boundaries and a separate write/readback
    and consumer record. R13D differences are raw observed values, never race ms.
    The return value cannot qualify actual hardware or unblock paired training.
    """
    if any(type(items) is not tuple for items in (publications, boundaries, consumptions)):
        raise ProtocolError("immutable acquired tuples required")
    if not 1 <= len(publications) <= 64 or not 1 <= len(consumptions) <= 256:
        raise ProtocolError("bounded nonempty commands and consumer ledger required")
    if len(boundaries) != len(consumptions) + 1:
        raise ProtocolError("each consumed interval needs its actual next rawpose boundary")
    for index, boundary in enumerate(boundaries):
        if boundary.scope != profile.scope or boundary.sequence != index:
            raise ProtocolError("reset/owner identity or consecutive boundary sequence mismatch")
        if index and boundary.r13_raw - boundaries[index - 1].r13_raw != 10:
            raise ProtocolError("literal R13D gap/regression/wrap; no clock conversion inferred")
        for gate, actual in zip(profile.normal_gate_masks, boundary.gate_words, strict=True):
            if actual & gate.mask != gate.expected:
                raise ProtocolError("normal consumer gate mismatch: " + gate.name)

    schedule: list[tuple[str, int | None, int, str | None, Words3]] = [
        ("bootstrap", None, offset, None, NEUTRAL) for offset in range(profile.bootstrap_intervals)
    ]
    cursor = profile.bootstrap_intervals
    acknowledgements: list[ProtocolAck] = []
    previous_publication = -1
    for ordinal, publication in enumerate(publications, 1):
        command = publication.command
        if command.scope != profile.scope or command.command_id != ordinal:
            raise ProtocolError("immutable command ID order or reset/owner mismatch")
        expected_start = cursor + profile.prefix_intervals
        if command.start_sequence != expected_start:
            raise ProtocolError("command gap/overlap or undeclared prefix")
        if publication.queued_after_sequence < previous_publication:
            raise ProtocolError("publication acquisition order regressed")
        previous_publication = publication.queued_after_sequence
        if publication.queued_after_sequence >= expected_start:
            raise ProtocolError("late command cannot receive a shifted execution boundary")
        if profile.mode == "immutable-preloaded-tape":
            if (
                publication.queued_after_sequence != -1
                or command.source_boundary_sha256 is not None
            ):
                raise ProtocolError(
                    "immutable tape must be committed before acquisition, not policy"
                )
        elif (
            publication.queued_after_sequence < cursor
            or cursor >= len(boundaries)
            or command.source_boundary_sha256 != boundaries[cursor].sha256
        ):
            raise ProtocolError("policy command must follow and bind its actual decision boundary")
        end = expected_start + command.n
        if end >= len(boundaries):
            raise ProtocolError("command has no actual next rawpose endpoint")
        schedule.extend(
            ("neutral-prefix", command.command_id, offset, None, NEUTRAL)
            for offset in range(profile.prefix_intervals)
        )
        schedule.extend(
            ("action", command.command_id, offset, command.sha256, words)
            for offset, words in enumerate(command.sgb_words)
        )
        acknowledgements.append(
            ProtocolAck(
                command.command_id,
                command.sha256,
                boundaries[cursor].sha256,
                boundaries[end].sha256,
                profile.prefix_intervals,
                command.n,
                boundaries[end].r13_raw - boundaries[cursor].r13_raw,
            )
        )
        cursor = end
    schedule.extend(
        ("neutral-tail", None, offset, None, NEUTRAL) for offset in range(profile.tail_intervals)
    )
    if len(schedule) != len(consumptions):
        raise ProtocolError("uncredited/missing interval outside the declared complete schedule")
    for index, (expected, observed) in enumerate(zip(schedule, consumptions, strict=True)):
        phase, command_id, offset, payload, words = expected
        if (
            observed.boundary_sha256 != boundaries[index].sha256
            or observed.execution_ordinal != index
            or (
                observed.phase,
                observed.command_id,
                observed.command_offset,
                observed.payload_sha256,
            )
            != (phase, command_id, offset, payload)
        ):
            raise ProtocolError("queue provenance/phase/ordinal does not bind the actual interval")
        if (
            observed.written_sgb_words != words
            or observed.readback_sgb_words != words
            or observed.consumer_sgb_words != words
        ):
            raise ProtocolError("desired/write/readback/physical-consumer SGB differ exactly")
    return ProtocolAssessment(
        tuple(acknowledgements),
        profile.bootstrap_intervals,
        profile.tail_intervals,
        len(consumptions),
        boundaries[-1].r13_raw - boundaries[0].r13_raw,
    )


def unique_neutral_anchor(
    windows: Boundary, native: tuple[tuple[int, tuple[int, ...], Words3], ...]
) -> int:
    """Select one observed physical6 match, then require the full physical10.

    The returned native counter is an observed content anchor only. The caller
    must perform and retain a fresh uncredited neutral seek, not invent a jump.
    """
    if type(native) is not tuple or not 1 <= len(native) <= 1000:
        raise ProtocolError("bounded immutable actual neutral candidates required")
    matches = []
    previous = None
    for counter, pose, sgb in native:
        _integer(counter, "retained native anchor counter")
        if previous is not None and counter - previous != 10:
            raise ProtocolError("retained native candidates must be consecutive raw counters")
        previous = counter
        _words(pose, 10)
        _sgb(sgb)
        if sgb != NEUTRAL:
            raise ProtocolError("retained anchor candidates must be actually neutral")
        if pose[4:10] == windows.pose_words[4:10]:
            matches.append((counter, pose))
    if len(matches) != 1:
        raise ProtocolError("one unique actual neutral physical6 content anchor required")
    counter, pose = matches[0]
    if pose != windows.pose_words:
        raise ProtocolError("unique physical6 anchor has a different physical10")
    return counter
