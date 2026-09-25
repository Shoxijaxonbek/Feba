"""Multi-object tracking (ByteTrack) and the trajectory container used by the rules.

Road users are tracked in three independent groups so that a pedestrian can
never inherit the id of a car (or of the bicycle she is riding):
vehicles (car/bus/truck, which the detector often confuses with each other),
two-wheelers, and persons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .detector import NAMES, NAME_ID, TWO_WHEELERS, VEHICLES

GROUPS = {
    "vehicle": [NAME_ID[n] for n in VEHICLES],
    "two_wheeler": [NAME_ID[n] for n in TWO_WHEELERS],
    "person": [NAME_ID["person"]],
}

TRACKER_ARGS = dict(track_high_thresh=0.35, track_low_thresh=0.1, new_track_thresh=0.4,
                    match_thresh=0.8, fuse_score=True)


class _Detections:
    """Minimal ultralytics `Results.boxes` look-alike accepted by BYTETracker.update."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    @property
    def conf(self) -> np.ndarray:
        return self.arr[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.arr[:, 5]

    @property
    def xywh(self) -> np.ndarray:
        a = self.arr
        return np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2, a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]], 1)

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx) -> "_Detections":
        return _Detections(self.arr[idx])


@dataclass
class Track:
    tid: int
    group: str
    t: np.ndarray        # (n,) seconds
    box: np.ndarray      # (n, 4) x1, y1, x2, y2 in scene px
    conf: np.ndarray     # (n,)
    cls: np.ndarray      # (n,) index into detector.NAMES
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        return NAMES[int(np.bincount(self.cls.astype(int)).argmax())]

    @property
    def foot(self) -> np.ndarray:
        """Bottom-centre of the box: the ground contact point of the road user."""
        return np.stack([(self.box[:, 0] + self.box[:, 2]) / 2, self.box[:, 3]], 1)

    @property
    def center(self) -> np.ndarray:
        return np.stack([(self.box[:, 0] + self.box[:, 2]) / 2, (self.box[:, 1] + self.box[:, 3]) / 2], 1)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def smooth_foot(self, window_s: float = 0.7) -> np.ndarray:
        """Foot point smoothed with a centred moving average (offline use only)."""
        key = ("foot", window_s)
        if key not in self._cache:
            self._cache[key] = _moving_average(self.t, self.foot, window_s)
        return self._cache[key]

    def velocity(self, window_s: float = 0.7) -> np.ndarray:
        """Velocity of the smoothed foot point in scene px / s, shape (n, 2)."""
        key = ("vel", window_s)
        if key not in self._cache:
            p = self.smooth_foot(window_s)
            if len(self.t) < 2:
                v = np.zeros_like(p)
            else:
                v = np.gradient(p, self.t, axis=0)
            self._cache[key] = v
        return self._cache[key]

    def speed(self, window_s: float = 0.7) -> np.ndarray:
        return np.linalg.norm(self.velocity(window_s), axis=1)


def _moving_average(t: np.ndarray, x: np.ndarray, window_s: float) -> np.ndarray:
    if len(t) < 3:
        return x.astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    half = window_s / 2
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    csum = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    out[:] = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    return out


class MultiTracker:
    """One ByteTrack instance per road-user group; feeds on per-frame detector output."""

    def __init__(self, fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker

        buffer = max(5, int(round(fps * 2.0)))  # keep lost tracks for ~2 s
        self.trackers = {g: BYTETracker(SimpleNamespace(track_buffer=buffer, **TRACKER_ARGS)) for g in GROUPS}
        self._obs: dict[int, list] = {}
        self._group_of: dict[int, str] = {}

    def update(self, t: float, dets: np.ndarray) -> list[tuple[int, str, np.ndarray]]:
        """Associate one frame. Returns [(track id, group, det row)] for the tracks seen in this frame."""
        seen = []
        for group, ids in GROUPS.items():
            sub = dets[np.isin(dets[:, 5], ids)]
            out = self.trackers[group].update(_Detections(sub))
            for row in out:
                tid, det_idx = int(row[4]), int(row[7])
                det = sub[det_idx]
                self._obs.setdefault(tid, []).append((t, *det))
                self._group_of[tid] = group
                seen.append((tid, group, det))
        return seen

    def tracks(self, min_obs: int = 3) -> dict[int, Track]:
        result = {}
        for tid, obs in self._obs.items():
            if len(obs) < min_obs:
                continue
            a = np.asarray(obs, dtype=np.float64)
            result[tid] = Track(tid, self._group_of[tid], a[:, 0], a[:, 1:5], a[:, 5], a[:, 6].astype(int))
        return result
