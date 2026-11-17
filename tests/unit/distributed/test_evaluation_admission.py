from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.distributed import coordinator_ingest, coordinator_submission
from trackmaniarl.distributed.coordinator_validation import _evaluation_admission_error


def _spec(*, suite: bool, cadence: int | None, trials: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        evaluation=SimpleNamespace(trials_per_map=trials) if suite else None,
        training=SimpleNamespace(evaluate_every_episodes=cadence),
    )


@pytest.mark.parametrize(
    ("cadence", "summary", "message"),
    [
        (None, {"evaluation_assignment_id": "assignment"}, "scheduled evaluation is disabled"),
        (1, {}, "require a coordinator assignment"),
    ],
)
def test_submit_rejects_unadmitted_suite_before_journal(  # noqa: PLR0913
    monkeypatch: pytest.MonkeyPatch, cadence: int | None, summary: dict, message: str
) -> None:
    value = {"evaluations": [summary]}
    coordinator = SimpleNamespace(
        _submission_decode_lock=RLock(),
        _request=Mock(return_value=value),
        run=SimpleNamespace(spec=_spec(suite=True, cadence=cadence), feature_pipeline=None),
        codec=object(),
        journal=Mock(),
    )
    monkeypatch.setattr(coordinator_submission, "_validate_submit_payload", lambda *a, **k: None)
    context = Mock()
    context.abort.side_effect = RuntimeError("aborted")
    with pytest.raises(RuntimeError, match="aborted"):
        coordinator_submission.submit(coordinator, BytesValue(), context)
    assert context.abort.call_args.args[0] == grpc.StatusCode.FAILED_PRECONDITION
    assert message in context.abort.call_args.args[1]
    coordinator.journal.append.assert_not_called()


def test_unadmitted_replayed_evaluation_fails_before_mutations() -> None:
    coordinator = SimpleNamespace(run=SimpleNamespace(spec=_spec(suite=True, cadence=None)))
    with pytest.raises(ValueError, match="journal contains unadmitted evaluation"):
        coordinator_ingest.ingest(coordinator, {"evaluations": [{}]}, 1)


def test_unadmitted_pending_promotion_fails_before_promoting() -> None:
    coordinator = SimpleNamespace(
        run=SimpleNamespace(spec=_spec(suite=True, cadence=None)),
        codec=SimpleNamespace(decode=lambda _: {"summaries": [{}], "snapshot": b"snapshot"}),
        journal=SimpleNamespace(pending_evaluation_promotions=lambda: [(1, b"evidence")]),
    )
    with pytest.raises(ValueError, match="pending promotion contains unadmitted evaluation"):
        coordinator_ingest._recover_pending_promotions(coordinator)


def test_game_free_and_no_evaluation_payloads_remain_accepted() -> None:
    assert _evaluation_admission_error(_spec(suite=False, cadence=None), [{}]) is None
    assert _evaluation_admission_error(_spec(suite=True, cadence=None), []) is None
    assert (
        _evaluation_admission_error(
            _spec(suite=True, cadence=1), [{"evaluation_assignment_id": "assignment"}]
        )
        is None
    )


@pytest.mark.parametrize("count", [1, 4])
def test_submit_rejects_wrong_trial_count_before_journal(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    summaries = [{"evaluation_assignment_id": "assignment"}] * count
    coordinator = SimpleNamespace(
        _submission_decode_lock=RLock(),
        _request=Mock(return_value={"evaluations": summaries}),
        run=SimpleNamespace(spec=_spec(suite=True, cadence=1, trials=3), feature_pipeline=None),
        codec=object(),
        journal=Mock(),
    )
    monkeypatch.setattr(coordinator_submission, "_validate_submit_payload", lambda *a, **k: None)
    context = Mock()
    context.abort.side_effect = RuntimeError("aborted")
    with pytest.raises(RuntimeError, match="aborted"):
        coordinator_submission.submit(coordinator, BytesValue(), context)
    assert context.abort.call_args.args == (
        grpc.StatusCode.FAILED_PRECONDITION,
        f"evaluation suite requires exactly 3 trials; got {count}",
    )
    coordinator.journal.append.assert_not_called()


@pytest.mark.parametrize("count", [1, 4])
def test_wrong_trial_count_blocks_wal_and_pending_promotion(count: int) -> None:
    summaries = [{"evaluation_assignment_id": "assignment"}] * count
    spec = _spec(suite=True, cadence=1, trials=3)
    coordinator = SimpleNamespace(
        run=SimpleNamespace(spec=spec),
        codec=SimpleNamespace(decode=lambda _: {"summaries": summaries, "snapshot": b"snapshot"}),
        journal=SimpleNamespace(pending_evaluation_promotions=lambda: [(1, b"evidence")]),
    )
    with pytest.raises(ValueError, match=r"journal contains unadmitted evaluation.*exactly 3"):
        coordinator_ingest.ingest(coordinator, {"evaluations": summaries}, 1)
    with pytest.raises(ValueError, match=r"pending promotion contains unadmitted.*exactly 3"):
        coordinator_ingest._recover_pending_promotions(coordinator)


def test_exact_trial_count_is_admitted_and_legacy_remains_unchanged() -> None:
    summaries = [{"evaluation_assignment_id": "assignment"}] * 3
    assert _evaluation_admission_error(_spec(suite=True, cadence=1, trials=3), summaries) is None
    assert _evaluation_admission_error(_spec(suite=False, cadence=1), summaries) is None


def _multimap_spec() -> SimpleNamespace:
    spec = _spec(suite=True, cadence=1, trials=2)
    spec.evaluation.maps = [
        SimpleNamespace(id="first", expected_map_uid="uid-a"),
        SimpleNamespace(id="second", expected_map_uid="uid-b"),
    ]
    return spec


def _multimap_results() -> list[dict]:
    return [
        {
            "evaluation_assignment_id": "assignment",
            "map_id": name,
            "map_uid": uid,
            "trial_index": index,
            "evaluation_assets_sha256": "a" * 64,
        }
        for name, uid in (("first", "uid-a"), ("second", "uid-b"))
        for index in range(2)
    ]


def test_multimap_admits_each_expected_trial_once() -> None:
    results = _multimap_results()
    assert _evaluation_admission_error(_multimap_spec(), list(reversed(results))) is None


@pytest.mark.parametrize("change", ["duplicate", "uid", "index", "missing", "boolean"])
def test_multimap_rejects_incomplete_or_misidentified_trials(change: str) -> None:
    results = _multimap_results()
    if change == "duplicate":
        results[-1] = dict(results[0])
    elif change == "uid":
        results[-1]["map_uid"] = "wrong"
    elif change == "index":
        results[-1]["trial_index"] = 2
    elif change == "missing":
        del results[-1]["map_id"]
    else:
        results[-1]["trial_index"] = True
    assert _evaluation_admission_error(_multimap_spec(), results) is not None


def test_multimap_invalid_trial_is_rejected_before_wal_and_during_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _multimap_results()
    results[-1] = dict(results[0])
    spec = _multimap_spec()
    coordinator = SimpleNamespace(
        _submission_decode_lock=RLock(),
        _request=Mock(return_value={"evaluations": results}),
        run=SimpleNamespace(spec=spec, feature_pipeline=None),
        codec=object(),
        journal=Mock(),
    )
    monkeypatch.setattr(coordinator_submission, "_validate_submit_payload", lambda *a, **k: None)
    context = Mock()
    context.abort.side_effect = RuntimeError("aborted")
    with pytest.raises(RuntimeError, match="aborted"):
        coordinator_submission.submit(coordinator, BytesValue(), context)
    coordinator.journal.append.assert_not_called()
    with pytest.raises(ValueError, match="journal contains unadmitted evaluation"):
        coordinator_ingest.ingest(coordinator, {"evaluations": results}, 1)
