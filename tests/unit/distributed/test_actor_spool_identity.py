from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.codec import WireCodec


def _actor(tmp_path: Path, actor_id: str) -> ActorRuntime:
    actor = object.__new__(ActorRuntime)
    actor.base_dir = tmp_path
    actor.spec = SimpleNamespace(artifacts_dir="artifacts", run_id="run")
    actor.actor_id = actor_id
    actor.codec = WireCodec(1024 * 1024)
    return actor


def test_distinct_ids_have_distinct_spools_even_after_legacy_normalization(tmp_path: Path) -> None:
    actors = [_actor(tmp_path, name) for name in ("worker/a", "worker?a", "Worker", "worker")]
    for actor in actors:
        actor._initialize_spool()
    assert len({str(actor.spool_dir).lower() for actor in actors}) == len(actors)


def test_owned_legacy_spool_is_recovered(tmp_path: Path) -> None:
    actor = _actor(tmp_path, "worker/a")
    legacy = actor._spool_directory("worker_a")
    legacy.mkdir(parents=True)
    payload = actor.codec.encode({"actor_id": "worker/a", "sequence": 7})
    (legacy / "00000000000000000007.rollout").write_bytes(payload)

    actor._initialize_spool()

    assert (actor.spool_dir / "00000000000000000007.rollout").read_bytes() == payload
    assert actor.sequence == 8
    assert not legacy.exists()


def test_legacy_spool_owned_by_another_actor_is_left_for_owner(tmp_path: Path) -> None:
    actor = _actor(tmp_path, "worker?a")
    legacy = actor._spool_directory("worker_a")
    legacy.mkdir(parents=True)
    payload = actor.codec.encode({"actor_id": "worker/a", "sequence": 0})
    (legacy / "00000000000000000000.rollout").write_bytes(payload)

    actor._initialize_spool()

    assert (legacy / "00000000000000000000.rollout").read_bytes() == payload
    assert not list(actor.spool_dir.glob("*.rollout"))


def test_ambiguous_legacy_temporary_is_quarantined_outside_actor_spools(tmp_path: Path) -> None:
    actor = _actor(tmp_path, "worker/a")
    legacy = actor._spool_directory("worker_a")
    legacy.mkdir(parents=True)
    pending = legacy / "00000000000000000000.tmp"
    pending.write_bytes(b"partial")

    actor._initialize_spool()
    quarantined = list((legacy.parent / "spool-quarantine").glob("*.invalid"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"partial"
    assert not list(actor.spool_dir.iterdir())


def test_legacy_valid_rollout_migrates_without_absorbing_unknown_temporary(tmp_path: Path) -> None:
    actor = _actor(tmp_path, "worker/a")
    legacy = actor._spool_directory("worker_a")
    legacy.mkdir(parents=True)
    payload = actor.codec.encode({"actor_id": "worker/a", "sequence": 7})
    (legacy / "00000000000000000007.rollout").write_bytes(payload)
    (legacy / "00000000000000000008.tmp").write_bytes(b"partial")

    actor._initialize_spool()

    assert (actor.spool_dir / "00000000000000000007.rollout").read_bytes() == payload
    quarantined = list((legacy.parent / "spool-quarantine").glob("*.invalid"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"partial"
    assert not list(actor.spool_dir.glob("*.invalid"))


def test_mixed_legacy_owners_remain_untouched(tmp_path: Path) -> None:
    actor = _actor(tmp_path, "worker/a")
    legacy = actor._spool_directory("worker_a")
    legacy.mkdir(parents=True)
    for index, owner in enumerate(("worker/a", "worker?a")):
        (legacy / f"{index:020}.rollout").write_bytes(
            actor.codec.encode({"actor_id": owner, "sequence": index})
        )
    with pytest.raises(RuntimeError, match="mixed owners"):
        actor._initialize_spool()
    assert len(list(legacy.glob("*.rollout"))) == 2
