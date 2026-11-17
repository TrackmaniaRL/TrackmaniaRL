from __future__ import annotations

from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.distributed import (
    coordinator_delivery,
    coordinator_ingest,
    coordinator_submission,
)
from trackmaniarl.distributed.evaluation_assets import evaluation_assets_digest
from trackmaniarl.distributed.journal import RolloutJournal


def _suite(tmp_path: Path) -> SimpleNamespace:
    maps = []
    for name in ("a", "b"):
        (tmp_path / f"{name}.gbx").write_bytes(name.encode())
        (tmp_path / f"{name}.npz").write_bytes((name + "geometry").encode())
        maps.append(
            SimpleNamespace(
                id=name,
                expected_map_uid=name,
                map_path=Path(f"{name}.gbx"),
                geometry_path=Path(f"{name}.npz"),
                reward_points_path=None,
            )
        )
    return SimpleNamespace(name="suite", version="1", maps=maps, trials_per_map=1)


def _coordinator(journal: RolloutJournal, suite: SimpleNamespace, path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        _lock=RLock(),
        journal=journal,
        run=SimpleNamespace(spec=SimpleNamespace(evaluation=suite), base_dir=path),
        _evaluation_due={"actor"},
        _evaluation_assignments={},
        _evaluation_schedule_frontier=0,
        _evaluation_policy_states={},
        _policy_payload=b"policy",
        counters=SimpleNamespace(policy_version=3),
        codec=SimpleNamespace(decode=lambda _: {}),
    )


def test_assignment_manifest_survives_restart_and_is_persisted_with_wal(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    journal = RolloutJournal(tmp_path / "rollouts.sqlite")
    try:
        journal.append("seed", 0, b"previous")
        coordinator = _coordinator(journal, suite, tmp_path)
        request = {"actor_id": "actor", "session_id": "one", "sequence": 1, "evaluations": []}
        assignment = coordinator_submission._assign_evaluation(coordinator, request, 1)
        assert assignment.assets_sha256 == evaluation_assets_digest(suite, tmp_path)
        journal.close()
        journal = RolloutJournal(tmp_path / "rollouts.sqlite")
        restarted = _coordinator(journal, suite, tmp_path)
        coordinator_delivery.load(restarted)
        (tmp_path / "a.gbx").write_bytes(b"changed after assignment")
        assert (
            restarted._evaluation_assignments["actor"]["assets_sha256"] == assignment.assets_sha256
        )
        assert evaluation_assets_digest(suite, tmp_path) != assignment.assets_sha256
        results = [
            {
                "policy_version": 3,
                "evaluation_assignment_id": assignment.assignment_id,
                "evaluation_assets_sha256": assignment.assets_sha256,
            }
        ]
        payload = {
            "actor_id": "actor",
            "session_id": "one",
            "sequence": 2,
            "evaluations": results,
            "evaluation_snapshot": b"policy",
        }
        bad_results = [dict(results[0], evaluation_assets_sha256="f" * 64)]
        bad_payload = {**payload, "evaluations": bad_results}
        context = Mock()
        context.abort.side_effect = RuntimeError("aborted")
        bad_submission = coordinator_submission._Submission(restarted, bad_payload, 0, False)
        with pytest.raises(RuntimeError, match="aborted"):
            coordinator_submission._append_journal(
                bad_submission, BytesValue(value=b"bad"), context
            )
        assert list(journal.rows_after(1)) == []
        submission = coordinator_submission._Submission(restarted, payload, 0, False)
        row, inserted = coordinator_submission._append_journal(
            submission, BytesValue(value=b"payload"), Mock()
        )
        assert inserted
        assert journal.evaluation_row_accepted(row) is True
        assert journal.evaluation_row_assets(row) == assignment.assets_sha256
        coordinator_ingest._verify_evaluation_row_assets(restarted, results, row)
        journal.close()
        journal = RolloutJournal(tmp_path / "rollouts.sqlite")
        restarted.journal = journal
        coordinator_ingest._verify_evaluation_row_assets(restarted, results, row)
        results[0]["evaluation_assets_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="asset binding"):
            coordinator_ingest._verify_evaluation_row_assets(restarted, results, row)
    finally:
        journal.close()


def test_actor_rejects_changed_or_missing_assigned_asset(tmp_path: Path) -> None:
    from trackmaniarl.distributed.actor_evaluation import _verify_assigned_assets

    suite = _suite(tmp_path)
    actor = SimpleNamespace(
        spec=SimpleNamespace(evaluation=suite),
        base_dir=tmp_path,
        _active_evaluation_assignment_id="assignment",
        _active_evaluation_assets_sha256=evaluation_assets_digest(suite, tmp_path),
    )
    _verify_assigned_assets(actor)
    (tmp_path / "a.npz").write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="assets differ"):
        _verify_assigned_assets(actor)
    (tmp_path / "b.gbx").unlink()
    with pytest.raises(FileNotFoundError):
        _verify_assigned_assets(actor)


def test_missing_or_mixed_digests_are_rejected() -> None:
    from tests.unit.distributed.test_evaluation_admission import _multimap_results, _multimap_spec
    from trackmaniarl.distributed.coordinator_validation import _evaluation_admission_error

    results = _multimap_results()
    del results[0]["evaluation_assets_sha256"]
    assert _evaluation_admission_error(_multimap_spec(), results) is not None
    results = _multimap_results()
    results[0]["evaluation_assets_sha256"] = "b" * 64
    assert _evaluation_admission_error(_multimap_spec(), results) is not None


def test_actor_submission_retry_and_recovery_bind_the_same_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from typing import Any

    from tests.integration.runtime.distributed_evaluation_support import _DISTRIBUTED_TOKEN
    from tests.integration.runtime.test_distributed_evaluation_integration import (
        _close,
        _config,
        _coordinator,
        _submit,
    )
    from trackmaniarl.distributed import actor_evaluation, evaluation_contract
    from trackmaniarl.distributed.actor import ActorRuntime
    from trackmaniarl.distributed.actor_background import _accept_evaluation_request
    from trackmaniarl.distributed.actor_requests import ActorProcessRequest

    path = _config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    (tmp_path / "one.gbx").write_bytes(b"first")
    (tmp_path / "two.gbx").write_bytes(b"second")
    (tmp_path / "second.txt").write_bytes(b"second-reward")
    config["evaluation"]["trials_per_map"] = 1
    config["evaluation"]["maps"] = [
        {
            "id": "one",
            "expected_map_uid": "uid-one",
            "map_path": "one.gbx",
            "reward_points_path": "reward.txt",
        },
        {
            "id": "two",
            "expected_map_uid": "uid-two",
            "map_path": "two.gbx",
            "reward_points_path": "second.txt",
        },
    ]
    path.write_text(json.dumps(config), encoding="utf-8")
    # A fake backend only exercises assignment/transport/recovery. Production
    # admission still rejects multi-map until an attested backend exists.
    monkeypatch.setattr(evaluation_contract, "validate_scheduled_evaluation", lambda *args: None)
    monkeypatch.setattr(actor_evaluation, "_supports_multimap_policy", lambda _: True)
    monkeypatch.setattr(actor_evaluation, "_validate_model_isolation", lambda *_: None)
    original_trials = actor_evaluation._evaluation_trials
    change_during = False

    def fake_maps(plan: Any, suite: Any) -> list[dict[str, Any]]:
        results = []
        for name in ("one", "two"):
            item = original_trials(plan, suite.trials_per_map)[0]
            results.append({**item, "map_id": name, "map_uid": f"uid-{name}", "trial_index": 0})
        if change_during:
            (tmp_path / "one.gbx").write_bytes(b"changed during evaluation")
        return results

    monkeypatch.setattr(actor_evaluation, "_multi_map_trials", fake_maps)
    actor = ActorRuntime(ActorProcessRequest(path, "127.0.0.1:1", "actor", _DISTRIBUTED_TOKEN))
    coordinator = _coordinator(path, actor.fingerprint)
    try:
        pipeline, factory = actor._components()
        coordinator._evaluation_due.add(actor.actor_id)
        coordinator.counters.policy_version = 1
        coordinator._policy_payload = coordinator.codec.encode({"value": 7})
        request = {
            **actor._request_base(),
            "sequence": 0,
            "policy_version": 1,
            "transitions": [],
            "episodes": [],
            "evaluations": [],
            "evaluation_snapshot": b"",
        }
        assignment = _submit(coordinator, coordinator.codec.encode(request))
        assert assignment["evaluation_assets_sha256"]
        _accept_evaluation_request(actor, assignment)
        (tmp_path / "one.gbx").write_bytes(b"changed")
        with pytest.raises(RuntimeError, match="assets differ"):
            actor._evaluate(factory.create(seed=0), pipeline)
        assert actor.queue.empty()
        (tmp_path / "one.gbx").write_bytes(b"first")
        _accept_evaluation_request(actor, assignment)
        change_during = True
        with pytest.raises(RuntimeError, match="assets differ"):
            actor._evaluate(factory.create(seed=0), pipeline)
        assert actor.queue.empty()
        (tmp_path / "one.gbx").write_bytes(b"first")
        change_during = False
    finally:
        actor.client.close()
        _close(coordinator)

    resumed = ActorRuntime(ActorProcessRequest(path, "127.0.0.1:1", "actor", _DISTRIBUTED_TOKEN))
    coordinator = _coordinator(path, resumed.fingerprint)
    try:
        pipeline, factory = resumed._components()
        request = {
            **resumed._request_base(),
            "sequence": 0,
            "policy_version": 1,
            "transitions": [],
            "episodes": [],
            "evaluations": [],
            "evaluation_snapshot": b"",
        }
        retried = _submit(coordinator, coordinator.codec.encode(request))
        assert retried["evaluation_assignment_id"] == assignment["evaluation_assignment_id"]
        assert retried["evaluation_assets_sha256"] == assignment["evaluation_assets_sha256"]
        _accept_evaluation_request(resumed, retried)
        resumed.sequence = 1
        resumed._evaluate(factory.create(seed=0), pipeline)
        payload = resumed.queue.get_nowait().read_bytes()
        assert _submit(coordinator, payload)["accepted"] is True
    finally:
        resumed.client.close()
        _close(coordinator)
    coordinator = _coordinator(path, resumed.fingerprint)
    try:
        coordinator._recover_journal(0)
        assert coordinator.counters.evaluations == 2
    finally:
        _close(coordinator)
