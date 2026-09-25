"""Shared context for the event rules: cleaned tracks, signal timeline, geometry helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..perception import Observation
from ..risk import replay_detections
from ..scene import Scene
from ..signals import GREEN, RED, UNKNOWN, signal_states
from ..tracking import Track

STILL_SPEED = 12.0   # scene px/s below which a vehicle counts as stationary (~1 km/h near the stop line)
MOVING_SPEED = 40.0  # scene px/s above which a road user is clearly moving


@dataclass
class Context:
    obs: Observation
    scene: Scene
    vehicles: dict[int, Track]
    pedestrians: dict[int, Track]
    two_wheelers: dict[int, Track]
    signal: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    risk: dict | None = None   # causal conflict score replayed on the detections (risk.replay_detections)

    @property
    def duration(self) -> float:
        return self.obs.info.duration

    @property
    def times(self) -> np.ndarray:
        return self.obs.times

    def signal_at(self, name: str, t: float | np.ndarray, hold: float = 2.0) -> np.ndarray:
        """Signal state at time(s) t; brief gaps (flashing green, occlusion) keep the last known state."""
        st_t, st = self.signal.get(name, (np.zeros(0), np.zeros(0, dtype=object)))
        t = np.atleast_1d(t)
        if len(st_t) == 0:
            return np.full(len(t), UNKNOWN, dtype=object)
        known = st != UNKNOWN
        kt, ks = st_t[known], st[known]
        out = np.full(len(t), UNKNOWN, dtype=object)
        if len(kt) == 0:
            return out
        i = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 1)
        ok = (kt[i] <= t + 1e-6) & (t - kt[i] <= hold)
        out[ok] = ks[i[ok]]
        return out

    def next_change(self, name: str, t: float, to: str = GREEN) -> float | None:
        """First time >= t at which signal `name` shows `to`."""
        if name not in self.signal:
            return None
        st_t, st = self.signal[name]
        idx = np.flatnonzero((st_t >= t) & (st == to))
        return float(st_t[idx[0]]) if len(idx) else None

    def red_since(self, name: str, t: float) -> float:
        """How long signal `name` has been red at time t (0 if it is not red)."""
        if name not in self.signal:
            return 0.0
        st_t, st = self.signal[name]
        i = np.searchsorted(st_t, t, side="right") - 1
        if i < 0 or st[i] != RED:
            return 0.0
        j = i
        while j > 0 and st[j - 1] in (RED, UNKNOWN):
            j -= 1
        return float(t - st_t[j])


def polygon_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance from points to a polygon boundary (positive inside)."""
    contour = poly.astype(np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), True) for x, y in np.atleast_2d(points)])


def box_overlap_frac(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Fraction of box `inner` (4,) covered by each of `outer` (n, 4)."""
    if len(outer) == 0:
        return np.zeros(0)
    x1 = np.maximum(inner[0], outer[:, 0])
    y1 = np.maximum(inner[1], outer[:, 1])
    x2 = np.minimum(inner[2], outer[:, 2])
    y2 = np.minimum(inner[3], outer[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _frame_boxes(tracks: dict[int, Track], times: np.ndarray) -> list[list[np.ndarray]]:
    per_frame: list[list[np.ndarray]] = [[] for _ in times]
    for tr in tracks.values():
        idx = np.clip(np.searchsorted(times, tr.t - 1e-6), 0, len(times) - 1)
        for i, b in zip(idx, tr.box):
            per_frame[i].append(b)
    return per_frame


def _subset(tr: Track, keep: np.ndarray) -> Track | None:
    if keep.sum() < 3:
        return None
    return Track(tr.tid, tr.group, tr.t[keep], tr.box[keep], tr.conf[keep], tr.cls[keep])


def build_context(obs: Observation, scene: Scene) -> Context:
    """Split tracks by kind and remove people who are not pedestrians.

    Persons whose box lies inside a vehicle box are passengers/drivers seen
    through windows (buses are full of them); persons sharing their box with a
    bicycle or motorcycle are riders. Both are dropped from the pedestrian set.
    """
    groups: dict[str, dict[int, Track]] = {"vehicle": {}, "person": {}, "two_wheeler": {}}
    for tid, tr in obs.tracks.items():
        groups[tr.group][tid] = tr
    times = obs.times
    veh_boxes = _frame_boxes(groups["vehicle"], times)
    two_boxes = _frame_boxes(groups["two_wheeler"], times)

    pedestrians = {}
    for tid, tr in groups["person"].items():
        idx = np.clip(np.searchsorted(times, tr.t - 1e-6), 0, len(times) - 1)
        keep = np.ones(len(tr.t), dtype=bool)
        rider = 0
        for k, (i, b) in enumerate(zip(idx, tr.box)):
            vb = np.asarray(veh_boxes[i]).reshape(-1, 4)
            if len(vb) and box_overlap_frac(b, vb).max() > 0.6:
                keep[k] = False
            tb = np.asarray(two_boxes[i]).reshape(-1, 4)
            if len(tb) and box_overlap_frac(b, tb).max() > 0.3:
                rider += 1
        if rider > 0.3 * len(tr.t):
            continue
        sub = _subset(tr, keep)
        if sub is not None:
            pedestrians[tid] = sub

    signal = {}
    reading = obs.side.get("signals")
    if reading is not None:
        for name in scene.signals:
            signal[name] = signal_states(reading, name)
    risk = replay_detections(obs.times, obs.detections, scene) if len(obs.times) > 2 else None
    return Context(obs, scene, groups["vehicle"], pedestrians, groups["two_wheeler"], signal, risk)


def runs(t: np.ndarray, mask: np.ndarray, max_gap: float = 0.5) -> list[tuple[float, float]]:
    """Maximal runs of True in a per-observation mask of one track (gaps < max_gap bridged)."""
    out: list[list[float]] = []
    start = prev = None
    for ti, m in zip(t, mask):
        if m:
            if start is not None and ti - prev > max_gap:
                out.append([start, prev])
                start = None
            if start is None:
                start = ti
            prev = ti
        elif start is not None and ti - prev > max_gap:
            out.append([start, prev])
            start = None
    if start is not None:
        out.append([start, prev])
    return [(s, e) for s, e in out]
