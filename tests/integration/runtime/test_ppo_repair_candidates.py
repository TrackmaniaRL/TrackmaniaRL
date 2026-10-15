"""Persistent recovery regression checks without a live game."""

import numpy as np

from pilot_tools.ppo_recovery import StalledIncidentMemory


def test_stationary_edge_without_prior_incident_is_not_a_recovery_trigger() -> None:
    memory = StalledIncidentMemory()
    raw = np.zeros(8, dtype=np.float32)
    assert memory.apply(raw, 0, 0.05) is raw


def test_incident_persists_only_while_stalled_at_edge() -> None:
    memory = StalledIncidentMemory()
    incident = np.zeros(8, dtype=np.float32)
    incident[6:] = [1.0, 0.23]
    memory.apply(incident, 0.5, 0.06)
    faded = np.zeros(8, dtype=np.float32)
    retained = memory.apply(faded, 0.2, 0.06)
    assert np.array_equal(retained[6:], incident[6:])
    assert np.array_equal(faded, np.zeros(8))  # Never mutate stored features.
    assert memory.apply(faded, 3, 0.06) is faded
    assert memory.apply(faded, 0.2, 0.06) is faded


def test_leaving_edge_clears_memory_even_without_speed() -> None:
    memory = StalledIncidentMemory(memory=0.23, deficit=1.0)
    raw = np.zeros(8, dtype=np.float32)
    assert memory.apply(raw, 0, 0.5) is raw
    assert memory.memory == 0
