"""CPU perspective rendering of recorded road boundaries, not Trackmania graphics.

Only the recorded ribbon is drawn. There are no inferred textures, obstacles,
terrain, car meshes or virtual lidar extensions. The camera is world-upright;
native telemetry does not supply the roll of the game's camera.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.native_headless.ribbon_kernel import compiled_rasterizer


class RibbonCameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    width: int = Field(default=84, ge=8, le=512)
    height: int = Field(default=84, ge=8, le=512)
    horizontal_fov_degrees: float = Field(default=90.0, ge=30.0, le=140.0)
    camera_height_m: float = Field(default=3.0, gt=0.0, le=20.0)
    downward_pitch_degrees: float = Field(default=10.0, ge=0.0, le=80.0)
    near_m: float = Field(default=0.25, gt=0.0)
    far_m: float = Field(default=120.0, gt=0.0, le=1000.0)
    raster_backend: Literal["numpy", "numba"] = "numpy"

    @model_validator(mode="after")
    def ordered_planes(self) -> RibbonCameraConfig:
        if self.near_m >= self.far_m:
            raise ValueError("camera near plane must precede far plane")
        return self


class RibbonRenderer:
    """A depth-tested, double-sided road ribbon with a fixed schematic palette."""

    representation = "recorded-road-ribbon-perspective-v1"
    road_rgb = (72, 78, 88)
    edge_rgb = (230, 230, 225)
    background_rgb = (105, 140, 165)
    rgb_palette = (background_rgb, road_rgb, edge_rgb)

    def __init__(
        self,
        geometry: BoundaryGeometry,
        config: RibbonCameraConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = RibbonCameraConfig.model_validate({} if config is None else config)
        self._compiled = compiled_rasterizer() if self.config.raster_backend == "numba" else None
        count = geometry.recorded_count
        left, right = geometry.left[:count], geometry.right[:count]
        span = right - left
        # Edge bands are a visualization of recorded boundaries, not game materials.
        lines = (left, left + 0.03 * span, right - 0.03 * span, right)
        triangles: list[np.ndarray] = []
        colors: list[int] = []
        for index, color in enumerate((2, 1, 2)):
            a, b = lines[index], lines[index + 1]
            triangles.extend(
                (
                    np.stack((a[:-1], b[:-1], b[1:]), axis=1),
                    np.stack((a[:-1], b[1:], a[1:]), axis=1),
                )
            )
            colors.extend([color] * (2 * (count - 1)))
        self._triangles = np.concatenate(triangles).astype(np.float64)
        self._colors = np.asarray(colors, dtype=np.uint8)
        block_size = 64
        starts = np.arange(0, len(self._triangles), block_size)
        lower = np.minimum.reduceat(self._triangles.min(axis=1), starts)
        upper = np.maximum.reduceat(self._triangles.max(axis=1), starts)
        self._block_centers = (lower + upper) / 2
        self._block_extents = (upper - lower) / 2
        self._triangle_blocks = np.arange(len(self._triangles)) // block_size
        focal = self.config.width / (
            2.0 * np.tan(np.deg2rad(self.config.horizontal_fov_degrees) / 2)
        )
        self._focal = float(focal)

    def render(self, position: np.ndarray, forward: np.ndarray) -> np.ndarray:
        """Return the exact RGB palette expansion of the indexed camera image."""
        return cast(
            np.ndarray,
            np.asarray(self.rgb_palette, dtype=np.uint8)[self.render_indexed(position, forward)],
        )

    def render_indexed(self, position: np.ndarray, forward: np.ndarray) -> np.ndarray:
        """Render solely from this observed position/direction; no clock extrapolation."""
        position, forward = (
            np.asarray(position, dtype=np.float64),
            np.asarray(forward, dtype=np.float64),
        )
        if (
            position.shape != (3,)
            or forward.shape != (3,)
            or not np.isfinite([position, forward]).all()
        ):
            raise ValueError("camera requires finite three-dimensional position and forward")
        norm = float(np.linalg.norm(forward))
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("camera forward direction must be nonzero")
        forward = forward / norm
        world_up = np.array([0.0, 1.0, 0.0])
        # A deterministic alternate up axis handles vertical driving without NaNs.
        up_axis = world_up if abs(forward[1]) < 0.99 else np.array([0.0, 0.0, 1.0])
        right = np.cross(up_axis, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        pitch = np.deg2rad(self.config.downward_pitch_degrees)
        view = np.cos(pitch) * forward - np.sin(pitch) * up
        basis = np.stack((right, np.cross(view, right), view), axis=1)
        camera = position + world_up * self.config.camera_height_m
        scene, scene_colors = self._visible_triangles(camera, basis)
        triangles = (scene - camera) @ basis
        visible = (triangles[:, :, 2].max(axis=1) >= self.config.near_m) & (
            triangles[:, :, 2].min(axis=1) <= self.config.far_m
        )
        for axis, half_extent in ((0, self.config.width / 2), (1, self.config.height / 2)):
            limit = triangles[:, :, 2] * (half_extent / self._focal)
            visible &= (triangles[:, :, axis] + limit).max(axis=1) >= 0
            visible &= (limit - triangles[:, :, axis]).max(axis=1) >= 0
        image = np.zeros((self.config.height, self.config.width), dtype=np.uint8)
        depth = np.zeros(image.shape[:2], dtype=np.float64)
        triangles, colors = triangles[visible], scene_colors[visible]
        complete = (triangles[:, :, 2].min(axis=1) >= self.config.near_m) & (
            triangles[:, :, 2].max(axis=1) <= self.config.far_m
        )
        clipped, clipped_colors = [], []
        for triangle, color in zip(triangles[~complete], colors[~complete], strict=True):
            polygon = _clip_depth(triangle, self.config.near_m, keep_greater=True)
            polygon = _clip_depth(polygon, self.config.far_m, keep_greater=False)
            for index in range(1, len(polygon) - 1):
                clipped.append([polygon[0], polygon[index], polygon[index + 1]])
                clipped_colors.append(color)
        triangles, colors = triangles[complete], colors[complete]
        if clipped:
            triangles = np.concatenate((triangles, np.asarray(clipped)))
            colors = np.concatenate((colors, np.asarray(clipped_colors)))
        self._rasterize(triangles, image, depth, colors)
        return image

    def _visible_triangles(
        self, camera: np.ndarray, basis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conservative world-space block/frustum culling before vertex transforms.

        Every block encloses all its triangles. Reject only when its entire AABB
        lies outside one plane; crossings and overpasses retain the old raster path.
        """
        horizontal = self.config.width / (2 * self._focal)
        vertical = self.config.height / (2 * self._focal)
        planes = np.array(
            [
                [0, 0, 1],
                [0, 0, -1],
                [1, 0, horizontal],
                [-1, 0, horizontal],
                [0, 1, vertical],
                [0, -1, vertical],
            ],
            dtype=np.float64,
        ).T
        normals = basis @ planes
        thresholds = np.array([self.config.near_m, -self.config.far_m, 0, 0, 0, 0])
        maximum = (self._block_centers - camera) @ normals + self._block_extents @ np.abs(normals)
        visible = (maximum >= thresholds - 1e-8).all(axis=1)
        keep = visible[self._triangle_blocks]
        return self._triangles[keep], self._colors[keep]

    def _rasterize(  # noqa: PLR0913 - independent raster inputs and output buffers
        self, triangles: np.ndarray, image: np.ndarray, depth: np.ndarray, colors: np.ndarray
    ) -> None:
        if not len(triangles):
            return
        if self._compiled is not None and image.ndim == 2:
            self._compiled(triangles, image, depth, colors, self._focal)
            return
        xy = triangles[:, :, :2] / triangles[:, :, 2, None] * self._focal
        xy[:, :, 0] += self.config.width / 2
        xy[:, :, 1] = self.config.height / 2 - xy[:, :, 1]
        extent = [self.config.width, self.config.height]
        low = np.clip(np.floor(xy.min(axis=1)), 0, extent).astype(int)
        high = np.clip(np.ceil(xy.max(axis=1)), 0, extent).astype(int)
        a, b, c = xy[:, 0], xy[:, 1], xy[:, 2]
        determinant = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
            a[:, 1] - c[:, 1]
        )
        valid = (high > low).all(axis=1) & (np.abs(determinant) >= 1e-10)
        xy, triangles, colors = xy[valid], triangles[valid], colors[valid]
        low, high, determinant = low[valid], high[valid], determinant[valid]
        counts = np.prod(high - low, axis=1)
        cumulative = np.r_[0, np.cumsum(counts)]
        start = 0
        while start < len(counts):
            # Bounded scratch memory, even when many overlapping polygons fill
            # the screen. No per-pixel or per-triangle Python raster loops.
            stop = int(np.searchsorted(cumulative, cumulative[start] + 262144, side="right") - 1)
            stop = max(start + 1, stop)
            ids = np.repeat(np.arange(start, stop), counts[start:stop])
            local = np.arange(len(ids)) - np.repeat(
                cumulative[start:stop] - cumulative[start], counts[start:stop]
            )
            width = high[ids, 0] - low[ids, 0]
            x = low[ids, 0] + local % width
            y = low[ids, 1] + local // width
            a, b, c = xy[ids, 0], xy[ids, 1], xy[ids, 2]
            wa = (
                (b[:, 1] - c[:, 1]) * (x + 0.5 - c[:, 0])
                + (c[:, 0] - b[:, 0]) * (y + 0.5 - c[:, 1])
            ) / determinant[ids]
            wb = (
                (c[:, 1] - a[:, 1]) * (x + 0.5 - c[:, 0])
                + (a[:, 0] - c[:, 0]) * (y + 0.5 - c[:, 1])
            ) / determinant[ids]
            wc = 1.0 - wa - wb
            inverse = (
                wa / triangles[ids, 0, 2] + wb / triangles[ids, 1, 2] + wc / triangles[ids, 2, 2]
            )
            inside = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
            pixels = (y * self.config.width + x)[inside]
            inverse, ids = inverse[inside], ids[inside]
            np.maximum.at(depth.reshape(-1), pixels, inverse)
            nearest = inverse == depth.reshape(-1)[pixels]
            image.reshape((-1, *image.shape[2:]))[pixels[nearest]] = colors[ids[nearest]]
            start = stop


def _clip_depth(polygon: np.ndarray, plane: float, *, keep_greater: bool) -> np.ndarray:
    """Clip in camera space before projection, including near-plane crossings."""
    if len(polygon) == 0:
        return polygon
    inside = polygon[:, 2] >= plane if keep_greater else polygon[:, 2] <= plane
    if inside.all():
        return polygon
    points = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        if inside[index] != inside[index - 1]:
            fraction = (plane - previous[2]) / (current[2] - previous[2])
            points.append(previous + fraction * (current - previous))
        if inside[index]:
            points.append(current)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)
