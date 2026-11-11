"""Copies made when transitions leave replay's protected storage."""

from dataclasses import replace

from trackmaniarl.core.data import Transition
from trackmaniarl.core.pytree import tree_snapshot


def owned_transition(item: Transition) -> Transition:
    """Copy all mutable transition fields for use outside a replay lock."""
    return replace(
        item,
        observation=tree_snapshot(item.observation),
        action=tree_snapshot(item.action),
        next_observation=tree_snapshot(item.next_observation),
        info=tree_snapshot(item.info),
    )
