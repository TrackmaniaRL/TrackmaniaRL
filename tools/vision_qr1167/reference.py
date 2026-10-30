"""Continuous ordered path projection; prevent jumps at crossing track segments."""
import numpy as np


class Reference:
    def __init__(self, records):
        self.records = records
        self.xyz = np.array([[r['position'][k] for k in ('x','y','z')] for r in records])
        self.clocks = np.array([r['raceTick'] for r in records])
        if len(records) < 3 or not np.isfinite(self.xyz).all() or not (np.diff(self.clocks) > 0).all():
            raise ValueError('reference must be finite and strictly ordered')
        self.distance = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(self.xyz, axis=0), axis=1))))
        self.index = 0

    def reset(self, race_ms, position):
        indices = np.flatnonzero(abs(self.clocks - race_ms) <= 250)
        if not len(indices):
            raise ValueError('no reference near measured start')
        return self._project(indices, position)

    def project(self, position, dt):
        if not 0 < dt <= .1:
            raise ValueError('invalid projection timestep')
        center = self.distance[self.index]
        indices = np.flatnonzero((self.distance >= center-15) & (self.distance <= center+200*dt+1))
        return self._project(indices, position)

    def _project(self, indices, position):
        point = np.array([position[k] for k in ('x','y','z')])
        if not np.isfinite(point).all():
            raise ValueError('non-finite vehicle position')
        distances = np.linalg.norm(self.xyz[indices]-point, axis=1)
        chosen = int(np.argmin(distances))
        self.index = int(indices[chosen])
        return float(self.distance[self.index]), float(distances[chosen])
