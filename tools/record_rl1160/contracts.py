"""Small independently testable admission and action-remapping contracts."""
from __future__ import annotations


def meets_goal(rows, maximum_ms):
    if type(maximum_ms) is not int or maximum_ms <= 0:
        raise ValueError('Positive integer native Goal threshold required')
    return any(type(r.get('actual_goal_ms')) is int and 0 < r['actual_goal_ms'] <= maximum_ms for r in rows)


def clean_dependency(row):
    closure = row.get('closure', {})
    return (row.get('phase') == 'completed'
        and all(closure.get(k) == [] for k in ('remainingOwned', 'cleanupErrors', 'forced'))
        and bool(row.get('exits')) and all(type(c) is int and c == 0 for c in row['exits'].values()))


def remap_head(prior, old_programs, new_programs, quantiles):
    """Preserve every old dueling Q value when adding unordered action programs."""
    import torch
    old = [tuple(map(int, p)) for p in old_programs]
    new = [tuple(map(int, p)) for p in new_programs]
    if len(set(old)) != len(old) or new != sorted(set(new)) or not set(old) <= set(new):
        raise ValueError('Unique old vocabulary and sorted superset required')
    lookup = {p: i for i, p in enumerate(old)}
    indices = torch.tensor([lookup.get(p, min(range(len(old)), key=lambda i: sum(a != b for a, b in zip(old[i], p)))) for p in new])
    old_w = prior['advantage.weight'].reshape(quantiles, len(old), -1)
    old_b = prior['advantage.bias'].reshape(quantiles, len(old))
    weights, bias = old_w[:, indices].clone(), old_b[:, indices].clone()
    for i, p in enumerate(new):
        if p not in lookup:
            bias[:, i] -= 20
    result = dict(prior)
    result['advantage.weight'] = weights.reshape(quantiles*len(new), -1)
    result['advantage.bias'] = bias.reshape(quantiles*len(new))
    result['value.weight'] = prior['value.weight'] + weights.mean(1) - old_w.mean(1)
    result['value.bias'] = prior['value.bias'] + bias.mean(1) - old_b.mean(1)
    return result
