"""Synthetic protocol vectors only; no game, engine, transport or live ACK."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from scripts.command_ack_r6_protocol import (
    GATE_NAMES,
    NEUTRAL,
    Boundary,
    Command,
    Consumption,
    GateMask,
    Profile,
    ProtocolError,
    Publication,
    Scope,
    unique_neutral_anchor,
    verify_episode,
)

GAS = (0, 0x3F800000, 0)
BRAKE = (0, 0x3F800000, 0x3F800000)
STEER = (0xBE2952A5, 0x3F800000, 0)
SCOPE = Scope("1" * 32, 1, "2" * 64)


def boundary(sequence: int) -> Boundary:
    return Boundary(
        SCOPE,
        sequence,
        10000 + sequence * 10,
        (0x3F800000, 0, 0, 0, 0x3F800000 + sequence, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )


def profile(prefix: int = 0, bootstrap: int = 0, tail: int = 0) -> Profile:
    # These test masks are synthetic fixture data, not claimed physical gate values.
    return Profile(
        SCOPE,
        "fixed-prefix-host" if prefix else "immutable-preloaded-tape",
        prefix,
        bootstrap,
        tail,
        tuple(GateMask(name, 1, 0) for name in GATE_NAMES),
        "3" * 64,
    )


def consumption(  # noqa: PLR0913 -- literal execution fixture dimensions are varied independently
    index: int,
    words: tuple[int, int, int],
    *,
    phase: str = "action",
    command: Command | None = None,
    offset: int = 0,
) -> Consumption:
    return Consumption(
        boundary(index).sha256,
        index,
        phase,  # type: ignore[arg-type] -- fixture parameter controls invalid phase tests
        None if command is None else command.command_id,
        offset,
        command.sha256 if phase == "action" and command is not None else None,
        NEUTRAL,
        words,
        words,
        words,
    )


def n5_tape() -> tuple[
    Profile, tuple[Publication, ...], tuple[Boundary, ...], tuple[Consumption, ...]
]:
    command = Command(SCOPE, 1, 0, 5, (BRAKE, GAS, GAS, GAS, STEER), None)
    return (
        profile(),
        (Publication(command, -1, command.sha256),),
        tuple(boundary(index) for index in range(6)),
        tuple(
            consumption(index, words, command=command, offset=index)
            for index, words in enumerate(command.sgb_words)
        ),
    )


def test_immutable_n5_keeps_first_tick_brake_tap_and_every_next_pose() -> None:
    spec, publications, boundaries, records = n5_tape()
    report = verify_episode(spec, publications, boundaries, records)
    assert report.total_intervals == 5
    assert report.total_raw_r13_delta == report.acknowledgements[0].raw_r13_delta == 50
    assert report.acknowledgements[0].endpoint_boundary_sha256 == boundary(5).sha256
    assert report.physical_execution_qualified is False
    assert report.paired_training_qualified is False


def test_host_prefix_is_included_in_duration_without_extending_action() -> None:
    command = Command(SCOPE, 1, 2, 1, (STEER,), boundary(0).sha256)
    report = verify_episode(
        profile(prefix=2),
        (Publication(command, 1, command.sha256),),
        tuple(boundary(index) for index in range(4)),
        (
            consumption(0, NEUTRAL, phase="neutral-prefix", command=command),
            consumption(1, NEUTRAL, phase="neutral-prefix", command=command, offset=1),
            consumption(2, STEER, command=command),
        ),
    )
    ack = report.acknowledgements[0]
    assert (ack.prefix_intervals, ack.action_intervals, ack.raw_r13_delta) == (2, 1, 30)
    assert report.total_intervals == 3


def test_identical_sgb_commands_still_require_distinct_execution_ids() -> None:
    commands = (
        Command(SCOPE, 1, 0, 1, (GAS,), None),
        Command(SCOPE, 2, 1, 1, (GAS,), None),
    )
    publications = tuple(Publication(command, -1, command.sha256) for command in commands)
    records = tuple(
        consumption(index, GAS, command=command) for index, command in enumerate(commands)
    )
    report = verify_episode(profile(), publications, tuple(boundary(i) for i in range(3)), records)
    assert [ack.command_id for ack in report.acknowledgements] == [1, 2]
    with pytest.raises(ProtocolError, match="queue provenance"):
        verify_episode(
            profile(),
            publications,
            tuple(boundary(i) for i in range(3)),
            (records[0], replace(records[1], command_id=1)),
        )


def test_each_host_decision_has_its_own_real_endpoint_and_fixed_prefix() -> None:
    first = Command(SCOPE, 1, 2, 1, (GAS,), boundary(0).sha256)
    second = Command(SCOPE, 2, 5, 5, (BRAKE, GAS, GAS, GAS, STEER), boundary(3).sha256)
    report = verify_episode(
        profile(prefix=2),
        (Publication(first, 1, first.sha256), Publication(second, 4, second.sha256)),
        tuple(boundary(index) for index in range(11)),
        (
            consumption(0, NEUTRAL, phase="neutral-prefix", command=first),
            consumption(1, NEUTRAL, phase="neutral-prefix", command=first, offset=1),
            consumption(2, GAS, command=first),
            consumption(3, NEUTRAL, phase="neutral-prefix", command=second),
            consumption(4, NEUTRAL, phase="neutral-prefix", command=second, offset=1),
            *(
                consumption(5 + index, words, command=second, offset=index)
                for index, words in enumerate(second.sgb_words)
            ),
        ),
    )
    assert [ack.raw_r13_delta for ack in report.acknowledgements] == [30, 70]
    assert report.acknowledgements[1].decision_boundary_sha256 == boundary(3).sha256
    assert report.total_intervals == 10


def test_previous_action_cannot_leak_into_declared_neutral_prefix() -> None:
    command = Command(SCOPE, 1, 1, 1, (GAS,), boundary(0).sha256)
    with pytest.raises(ProtocolError, match="differ exactly"):
        verify_episode(
            profile(prefix=1),
            (Publication(command, 0, command.sha256),),
            tuple(boundary(index) for index in range(3)),
            (
                consumption(0, GAS, phase="neutral-prefix", command=command),
                consumption(1, GAS, command=command),
            ),
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing-next",
        "r13-gap",
        "duplicate",
        "owner",
        "generation",
        "gate",
        "word1768",
        "float1374",
    ],
)
def test_literal_tick_faults_cannot_receive_ack(change: str) -> None:
    spec, publications, boundaries, records = n5_tape()
    values = list(boundaries)
    if change == "missing-next":
        values.pop()
    elif change == "r13-gap":
        values[3] = replace(values[3], r13_raw=10031)
    elif change == "duplicate":
        values[3] = values[2]
    elif change == "owner":
        values[3] = replace(values[3], scope=replace(SCOPE, owner_sha256="4" * 64))
    elif change == "generation":
        values[3] = replace(values[3], scope=replace(SCOPE, reset_generation=2))
    elif change == "gate":
        values[3] = replace(values[3], gate_words=(1, 0, 0, 0, 0))
    elif change == "word1768":
        values[3] = replace(values[3], gate_words=(0, 0, 0, 1, 0))
    elif change == "float1374":
        values[3] = replace(values[3], gate_words=(0, 0, 0, 0, 1))
    with pytest.raises(ProtocolError):
        verify_episode(spec, publications, tuple(values), records)


@pytest.mark.parametrize("field", ["written_sgb_words", "readback_sgb_words", "consumer_sgb_words"])
def test_one_ulp_or_later_writer_disagreement_is_not_a_physics_ack(field: str) -> None:
    spec, publications, boundaries, records = n5_tape()
    changed = replace(records[-1], **{field: (0xBE2952A6, 0x3F800000, 0)})
    with pytest.raises(ProtocolError, match="differ exactly"):
        verify_episode(spec, publications, boundaries, (*records[:-1], changed))


def test_queued_message_is_not_sufficient_without_consumption() -> None:
    spec, publications, boundaries, _ = n5_tape()
    with pytest.raises(ProtocolError, match="nonempty"):
        verify_episode(spec, publications, boundaries, ())


def test_late_host_command_does_not_shift_its_start_or_extend_previous_action() -> None:
    command = Command(SCOPE, 1, 2, 1, (GAS,), boundary(0).sha256)
    with pytest.raises(ProtocolError, match="late command"):
        verify_episode(
            profile(prefix=2),
            (Publication(command, 2, command.sha256),),
            tuple(boundary(i) for i in range(4)),
            (
                consumption(0, NEUTRAL, phase="neutral-prefix", command=command),
                consumption(1, NEUTRAL, phase="neutral-prefix", command=command, offset=1),
                consumption(2, GAS, command=command),
            ),
        )


def test_stale_policy_observation_cannot_be_labeled_current_decision() -> None:
    command = Command(SCOPE, 1, 2, 1, (GAS,), boundary(1).sha256)
    with pytest.raises(ProtocolError, match="decision boundary"):
        verify_episode(
            profile(prefix=2),
            (Publication(command, 1, command.sha256),),
            tuple(boundary(i) for i in range(4)),
            tuple(consumption(i, NEUTRAL, phase="bootstrap") for i in range(3)),
        )


def test_bootstrap_and_tail_are_explicit_and_outside_command_duration() -> None:
    command = Command(SCOPE, 1, 1, 1, (GAS,), None)
    report = verify_episode(
        profile(bootstrap=1, tail=1),
        (Publication(command, -1, command.sha256),),
        tuple(boundary(i) for i in range(4)),
        (
            consumption(0, NEUTRAL, phase="bootstrap"),
            consumption(1, GAS, command=command),
            consumption(2, NEUTRAL, phase="neutral-tail"),
        ),
    )
    assert (report.bootstrap_intervals, report.tail_intervals, report.total_intervals) == (1, 1, 3)
    assert report.acknowledgements[0].raw_r13_delta == 10
    assert report.paired_training_qualified is False


@pytest.mark.parametrize("n", [True, 0, 2, 6])
def test_only_literal_n1_or_n5_is_accepted(n: int) -> None:
    with pytest.raises(ProtocolError, match="N1/N5"):
        Command(SCOPE, 1, 0, n, (GAS,), None)


def test_integer_braking_boolean_cannot_be_used_as_float32_one() -> None:
    with pytest.raises(ProtocolError, match=r"input domain|float32"):
        Command(SCOPE, 1, 0, 1, ((0, 0x3F800000, 1),), None)


def test_payload_is_immutable_and_bound_to_publication_hash() -> None:
    command = Command(SCOPE, 1, 0, 1, (GAS,), None)
    with pytest.raises(FrozenInstanceError):
        command.n = 5  # type: ignore[misc]
    with pytest.raises(ProtocolError, match="immutable command"):
        Publication(command, -1, "f" * 64)
    spec, publications, boundaries, records = n5_tape()
    report = verify_episode(spec, publications, boundaries, records)
    with pytest.raises(ValueError, match="init=False"):
        replace(report, paired_training_qualified=True)


def test_initial_anchor_rejects_stationary_ambiguity_and_quaternion_disagreement() -> None:
    observed = boundary(0)
    assert unique_neutral_anchor(observed, ((2250, observed.pose_words, NEUTRAL),)) == 2250
    with pytest.raises(ProtocolError, match="unique"):
        unique_neutral_anchor(
            observed, ((2250, observed.pose_words, NEUTRAL), (2260, observed.pose_words, NEUTRAL))
        )
    changed_q = (observed.pose_words[0], 1, *observed.pose_words[2:])
    with pytest.raises(ProtocolError, match="different physical10"):
        unique_neutral_anchor(observed, ((2250, changed_q, NEUTRAL),))


def test_foreign_nonce_or_zero_delay_host_contract_is_rejected() -> None:
    spec, publications, boundaries, records = n5_tape()
    with pytest.raises(ProtocolError, match="identity"):
        verify_episode(
            spec,
            publications,
            (replace(boundaries[0], scope=replace(SCOPE, nonce="5" * 32)), *boundaries[1:]),
            records,
        )
    with pytest.raises(ProtocolError, match="positive fixed prefix"):
        replace(spec, mode="fixed-prefix-host")


def test_missing_extra_normal_gates_or_wrong_word_width_cannot_be_admitted() -> None:
    spec = profile()
    with pytest.raises(ProtocolError, match="all five"):
        replace(spec, normal_gate_masks=spec.normal_gate_masks[:3])
    with pytest.raises(ProtocolError, match="uint16"):
        GateMask("word1768", 0x1FFFF, 0)
    with pytest.raises(ProtocolError, match="uint16"):
        replace(boundary(0), gate_words=(0, 0, 0, 0x10000, 0))
