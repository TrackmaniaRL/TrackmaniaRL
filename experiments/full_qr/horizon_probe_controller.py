"""Slow real-action controller for a bounded150s environment qualification.

This is diagnostic control, never a learner policy, training demonstration,
replay initializer or replacement for the fresh QR pilot. It must run against
the unchanged environment/reward and fail if the episode terminates early.
"""

from math import hypot

import numpy as np

from trackmaniarl.trackmania.native_headless.full_qr.native_features import NativeGraphFrame


class HorizonProbeController:
    """Follow the existing reward center slowly using only measured frames.

    Speed is measured from actual position differences and the native race
    clock, rather than assuming units for the vehicle's visual velocity cache.
    All returned actions belong to the historical78-action brake-tap table.
    """

    def __init__(self, trajectory: np.ndarray, target_speed_mps: float = 1.5) -> None:
        points = np.asarray(trajectory, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 3
            or not np.isfinite(points).all()
            or not np.isfinite(target_speed_mps)
            or not 1.0 <= target_speed_mps <= 3.0
        ):
            raise ValueError("finite pinned centerline and bounded slow speed required")
        self.points = points.copy()
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= 0):
            raise ValueError("nonduplicate centerline points required")
        self.distance = np.concatenate(([0.0], np.cumsum(lengths)))
        self.target_speed_mps = target_speed_mps
        self._nearest = 0
        self._previous: tuple[float, np.ndarray] | None = None

    def act(self, frame: NativeGraphFrame) -> int:
        if type(frame) is not NativeGraphFrame:
            raise TypeError("complete actual ordinary native frame required")
        time_ms = float(frame.channels[0])
        position = np.asarray(frame.channels[1:4], dtype=np.float64)
        if self._previous is None:
            # The actual neutral race0 record already has vertical gravity
            # velocity (-.225), despite identical2190/2200 planar positions.
            # Only horizontal motion invalidates the fresh planar controller.
            planar_speed = hypot(frame.channels[4], frame.channels[6])
            if time_ms != 0 or planar_speed > 1e-5:
                raise ValueError("fresh observed neutral reset required")
            speed = planar_speed
        else:
            previous_ms, previous_position = self._previous
            if time_ms - previous_ms != 50:
                raise ValueError("one actual50ms qualification step required")
            speed = float(np.linalg.norm(position[[0, 2]] - previous_position[[0, 2]])) / 0.05

        start, stop = max(0, self._nearest - 5), min(len(self.points), self._nearest + 40)
        distances = np.linalg.norm(self.points[start:stop] - position, axis=1)
        self._nearest = start + int(np.argmin(distances))
        if distances.min() > 10:
            raise ValueError("qualification controller left the pinned track")
        lookahead_m = 8.0
        target = int(np.searchsorted(self.distance, self.distance[self._nearest] + lookahead_m))
        if target >= len(self.points):
            raise ValueError("qualification horizon reached the centerline end")
        delta = (self.points[target] - position)[[0, 2]]
        forward = np.asarray((frame.channels[7], frame.channels[9]), dtype=np.float64)
        norm = hypot(*forward)
        squared = float(delta @ delta)
        if norm < 0.5 or squared < 1:
            raise ValueError("measured planar direction and lookahead required")
        forward /= norm
        curvature = 2 * float(forward[1] * delta[0] - forward[0] * delta[1]) / squared
        # Actual725 controls with positive steering bend toward negative worldX
        # when driving along positiveZ. Match that historical command handedness.
        # The observed slow725/729 arc has much less curvature per command
        # than the initial assumed .25; begin cornering earlier and use the
        # measured low-speed command scale, retaining the same offtrack guard.
        steering = float(np.clip(-curvature / 0.04, -1, 1))
        steer_index = int(np.rint(6 * steering)) + 6
        # Table order per steer: gas0[off,brake,tap],gas1[off,brake,tap].
        if speed < self.target_speed_mps - 0.15:
            pedal = 3
        elif speed > self.target_speed_mps + 0.15:
            pedal = 1
        else:
            pedal = 0
        self._previous = (time_ms, position.copy())
        return steer_index * 6 + pedal
