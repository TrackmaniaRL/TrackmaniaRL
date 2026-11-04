import pytest

from tools.headless_campaign_queue.policy_stability import analyze


def test_changes_gaps_ties_and_paired_returns() -> None:
    result = analyze({
        "actions": ["left", "right"],
        "observations": [
            {"id": "a", "before": [1.0, 0.9], "after": [0.9, 1.1],
             "before_return": 0.5, "after_return": 0.8},
            {"id": "b", "before": [2.0, 2.0], "after": [2.0, 2.0]},
        ],
    })
    assert result["changed_greedy_action_count"] == 1
    assert result["changed_greedy_action_fraction"] == 0.5
    assert result["ties"] == {"before": 1, "after": 1}
    assert result["action_gap"]["before"]["median"] == pytest.approx(0.05)
    assert result["selected_q_minus_observed_return"]["after"]["mean"] == pytest.approx(0.3)
    assert result["selected_q_minus_observed_return"]["before"]["count"] == 1


@pytest.mark.parametrize("data", [
    {"actions": ["a", "a"], "observations": [{"id": "x"}]},
    {"actions": ["a", "b"], "observations": []},
    {"actions": ["a", "b"], "observations": [{"id": "x", "before": [1], "after": [1, 2]}]},
    {"actions": ["a", "b"], "observations": [
        {"id": "x", "before": [1, float('nan')], "after": [1, 2]},
    ]},
    {"actions": ["a", "b"], "observations": [{"id": "x", "before": [1, 2], "after": [1, 2]},
                                                {"id": "x", "before": [1, 2], "after": [1, 2]}]},
])
def test_rejects_invalid_data(data: dict) -> None:
    with pytest.raises(ValueError, match=r".+"):
        analyze(data)


def test_missing_returns_are_not_zero() -> None:
    result = analyze({"actions": ["a", "b"], "observations": [
        {"id": "x", "before": [0, 1], "after": [0, 1]},
    ]})
    assert result["selected_q_minus_observed_return"] == {
        "before": {"count": 0}, "after": {"count": 0},
    }
