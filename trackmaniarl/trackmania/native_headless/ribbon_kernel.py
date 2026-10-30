"""Optional strict float64 ribbon rasterizer, with no dependency on native physics."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import lru_cache
from math import ceil, floor
from typing import cast

import numpy as np

type IndexedRasterizer = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], None]


@lru_cache(maxsize=1)
def compiled_rasterizer() -> IndexedRasterizer:
    """Compile once per process; never write caches into an installed package."""
    try:
        numba = importlib.import_module("numba")
    except ImportError as error:
        raise RuntimeError(
            "The numba ribbon backend requires trackmaniarl[native-vision]"
        ) from error
    return cast(
        IndexedRasterizer,
        numba.njit(nogil=True, fastmath=False, boundscheck=True, cache=False)(_rasterize),
    )


def _rasterize(  # noqa: PLR0913 - independent raster inputs and output buffers
    triangles: np.ndarray,
    image: np.ndarray,
    depth: np.ndarray,
    colors: np.ndarray,
    focal: float,
) -> None:
    """Preserve projection arithmetic, epsilon, inverse depth and last-equal-depth wins.

    Ordered scalar loops avoid the reference implementation's pixel candidate
    arrays. No fast-math, parallel reductions, lower precision or depth shortcuts.
    """
    height, width = image.shape
    for index in range(len(triangles)):
        triangle = triangles[index]
        ax = triangle[0, 0] / triangle[0, 2] * focal + width / 2
        bx = triangle[1, 0] / triangle[1, 2] * focal + width / 2
        cx = triangle[2, 0] / triangle[2, 2] * focal + width / 2
        ay = height / 2 - triangle[0, 1] / triangle[0, 2] * focal
        by = height / 2 - triangle[1, 1] / triangle[1, 2] * focal
        cy = height / 2 - triangle[2, 1] / triangle[2, 2] * focal
        determinant = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(determinant) < 1e-10:
            continue
        low_x, high_x = (
            max(0, min(width, floor(min(ax, bx, cx)))),
            max(0, min(width, ceil(max(ax, bx, cx)))),
        )
        low_y, high_y = (
            max(0, min(height, floor(min(ay, by, cy)))),
            max(0, min(height, ceil(max(ay, by, cy)))),
        )
        for y in range(low_y, high_y):
            for x in range(low_x, high_x):
                wa = ((by - cy) * (x + 0.5 - cx) + (cx - bx) * (y + 0.5 - cy)) / determinant
                wb = ((cy - ay) * (x + 0.5 - cx) + (ax - cx) * (y + 0.5 - cy)) / determinant
                wc = 1.0 - wa - wb
                if wa >= -1e-9 and wb >= -1e-9 and wc >= -1e-9:
                    inverse = wa / triangle[0, 2] + wb / triangle[1, 2] + wc / triangle[2, 2]
                    if inverse >= depth[y, x]:
                        depth[y, x] = inverse
                        image[y, x] = colors[index]
