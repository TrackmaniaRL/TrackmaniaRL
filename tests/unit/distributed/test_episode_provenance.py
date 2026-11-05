from types import SimpleNamespace

from trackmaniarl.distributed.coordinator_ingest import _episode_provenance


def test_episode_provenance_is_an_ingest_snapshot_not_episode_end() -> None:
    coordinator = SimpleNamespace(
        _metric_segment_id="coordinator-segment",
        counters=SimpleNamespace(transitions=245),
    )
    chunk = {"session_id": "actor-session", "sequence": 4}
    # Multiple episodes in the same submitted chunk necessarily share the
    # snapshot; their individual completion positions are not inferred.
    first = _episode_provenance(coordinator, chunk, phase="training")
    second = _episode_provenance(coordinator, chunk, phase="training")
    assert first == second
    assert first == {
        "phase": "training",
        "metric_segment_id": "coordinator-segment",
        "actor_session_id": "actor-session",
        "actor_chunk_sequence": 4,
        "accepted_transitions_at_ingest": 245,
        "accepted_transitions_semantics": "after_entire_submitted_chunk",
    }
