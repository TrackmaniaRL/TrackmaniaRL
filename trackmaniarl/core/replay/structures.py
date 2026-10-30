"""Private data structures used by replay sampling."""

from __future__ import annotations

from typing import Any

import numpy as np


class _FenwickTree:
    """Fixed-capacity prefix-sum tree for O(log N) proportional replay draws."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.values = np.zeros(size + 1, dtype=np.float64)
        self.leaves = np.zeros(size, dtype=np.float32)

    def set(self, index: int, value: float) -> None:
        delta = value - self.leaves[index]
        self.leaves[index] = value
        # Uniform and inactive expert trees often receive an unchanged weight.
        # Keep the leaf assignment and its float32 rounding, but no prefix sum
        # needs updating when the original scalar delta is exactly zero.
        if delta == 0.0:
            return
        index += 1
        while index <= self.size:
            self.values[index] += delta
            index += index & -index

    @property
    def total(self) -> float:
        total = 0.0
        index = self.size
        while index:
            total += float(self.values[index])
            index -= index & -index
        return total

    def find(self, target: float) -> int:
        """Return the zero-based leaf containing a target in ``[0, total)``."""

        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate <= self.size and self.values[candidate] <= target:
                index = candidate
                target -= self.values[candidate]
            bit >>= 1
        return min(index, self.size - 1)

    def find_many(
        self, targets: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        """Search independent targets with the scalar search's exact bit order."""
        if targets.ndim != 1:
            raise ValueError("prefix-sum targets must be a vector")
        remaining = targets.copy()
        indices = np.zeros(len(targets), dtype=np.int64)
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            candidates = indices + bit
            inside = candidates <= self.size
            values = self.values[np.minimum(candidates, self.size)]
            advance = inside & (values <= remaining)
            indices[advance] = candidates[advance]
            remaining[advance] -= values[advance]
            bit >>= 1
        return np.minimum(indices, self.size - 1)
