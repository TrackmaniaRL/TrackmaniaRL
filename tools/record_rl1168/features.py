"""Causal car sensors and a disclosed, static reference-route preview."""
import numpy as np

LOOKAHEAD = np.array([2, 5, 10, 20, 35, 55, 80, 120], dtype=np.float32)
OBS_DIM = 47


class RouteFeatures:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or len(self.xyz) < 3 or not np.isfinite(self.xyz).all():
            raise ValueError('finite measured reference geometry required')
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.xyz, axis=0), axis=1))]
        if self.arc[-1] <= 1:
            raise ValueError('reference has no length')
        self.index = None

    def observe(self, native22, *, reset=False):
        native = np.asarray(native22, dtype=np.float64)
        if native.shape != (22,) or not np.isfinite(native).all():
            raise ValueError('finite causal native22 state required')
        position = native[10:13]*1000
        if reset or self.index is None:
            indices = np.arange(len(self.xyz))
        else:
            center = self.arc[self.index]
            # One 10ms decision, up to 200m/s plus discrete projection tolerance.
            indices = np.flatnonzero((self.arc >= center-5) & (self.arc <= center+3))
        distances = np.linalg.norm(self.xyz[indices]-position, axis=1)
        self.index = int(indices[int(distances.argmin())])
        progress = float(self.arc[self.index])
        lateral = float(distances.min())
        points = np.stack([np.interp(progress+LOOKAHEAD, self.arc, self.xyz[:, axis]) for axis in range(3)], axis=1)
        rotation = native[1:10].reshape(3, 3)
        if not np.allclose(rotation.T@rotation, np.eye(3), atol=.01):
            raise ValueError('invalid measured orientation')
        relative = ((points-position)@rotation/100).reshape(-1)
        # No race time, reference time, reference actions, ghost or images.
        sensors = native[1:].copy()
        # TICK telemetry exposes requested steering but not native filteredSteer.
        # Reserve its slot as a literal constant on BOTH backends, never infer it.
        sensors[-1] = 0.
        result = np.r_[sensors, relative, progress/self.arc[-1], lateral/20].astype(np.float32)
        assert result.shape == (OBS_DIM,) and np.isfinite(result).all()
        return result, progress, lateral


def shaped_reward(previous_progress, progress, *, dt=.01, goal=False, failure=False):
    """Potential shaping: a full successful lap sums to 100 - actual lap time."""
    if not 0 < dt <= .02 or goal and failure:
        raise ValueError('invalid transition duration/terminal flags')
    if not np.isfinite([previous_progress, progress, dt]).all():
        raise ValueError('nonfinite reward input')
    if abs(progress-previous_progress) > 200*dt+1:
        raise ValueError('unphysical progress jump; discard transition')
    before = .05*previous_progress
    after = 0. if goal or failure else .05*progress
    return after-before-dt+(100. if goal else -100. if failure else 0.)
