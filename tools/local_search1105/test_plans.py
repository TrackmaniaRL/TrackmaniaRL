"""Pure tests: no environment, SDK, subprocesses, CUDA or training."""

import pytest

from .plans import candidates, confirmed_improvement, select


def test_single_edit_bounds_and_padding() -> None:
    baseline = (39,) * 5 + (75,) * 5 + (36,) * 5
    pool = candidates(baseline, 10)
    assert len({c.digest for c in pool}) == len(pool)
    assert {c.kind for c in pool} == {
        "edge_earlier", "edge_later", "steer_step", "gas_tap", "brake_tap",
    }
    for c in pool:
        assert len(c.ticks) == len(baseline)
        assert c.ticks[10:] == baseline[10:]
        assert 1 <= sum(a != b for a, b in zip(c.ticks, baseline, strict=True)) <= 2
        assert c.ticks[:c.start] == baseline[:c.start]
        assert c.ticks[c.stop:] == baseline[c.stop:]
        assert all(0 <= a < 78 and a % 3 != 2 for a in c.ticks)


def test_boundary_shifts_are_explicit() -> None:
    pool = candidates((39,) * 5 + (75,) * 5, 10)
    assert any(c.ticks == (39,) * 3 + (75,) * 7 for c in pool)
    assert any(c.ticks == (39,) * 7 + (75,) * 3 for c in pool)


def test_sampling_is_reproducible_and_balanced() -> None:
    pool = candidates((39,) * 5 + (75,) * 5, 10)
    assert select(pool, 25, 1105) == select(pool, 25, 1105)
    assert select(pool, 25, 1105) != select(pool, 25, 1106)
    assert len({c.kind for c in select(pool, 5, 1105)}) == 5


def test_reject_noncanonical_pulse_alias() -> None:
    with pytest.raises(ValueError, match="canonical"):
        candidates((41,) * 5, 5)


def test_goal_gate() -> None:
    assert confirmed_improvement([17714] * 5, 17715)
    assert not confirmed_improvement([17715] * 5, 17715)
    assert not confirmed_improvement([17714] * 4 + [None], 17715)
    assert not confirmed_improvement([17714] * 4, 17715)
    assert not confirmed_improvement([17714] * 4 + [17713], 17715)
    assert not confirmed_improvement([0] * 5, 17715)


def test_runner_refuses_without_explicit_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    from .run import main

    monkeypatch.setattr('sys.argv', ['run', '--study', 'missing-study'])
    with pytest.raises(RuntimeError, match='explicit future execution'):
        main()
