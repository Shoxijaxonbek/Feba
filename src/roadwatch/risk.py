"""Part B: causal accident-risk score from tracked road users.

Only the frames seen so far are used. Every `stride`-th frame (~5 fps) is
detected and tracked. For every pair of road users on the carriageway (at least
one of them a vehicle) we extrapolate their ground footprints (the lower part
of the box) linearly and find the time tau at which the footprints would start
to overlap. Pairs that would never touch within HORIZON_S (passing in the next
lane, following at a safe gap, standing in a queue) score 0; the others score

    raw = relative speed / (2 * tau)        [object sizes / s^2]

which is the deceleration they would need to avoid contact (a DRAC-style
surrogate safety measure). The most dangerous pair drives the frame's score.
The raw score is mapped through a logistic calibrated on the sample videos so
that ordinary traffic stays well below the 0.5 alarm threshold, then smoothed
with a fast-attack / slow-release filter so that one conflict is one alarm.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from . import alignment
from .detector import Detector
from .scene import Scene
from .tracking import MultiTracker
from .utils import Pace
from .video import SCENE_H, SCENE_W

HISTORY_S = 1.2       # velocity is fitted on this much track history
HORIZON_S = 3.0       # look-ahead for the footprint-overlap test
MIN_REL_SPEED = 0.3   # relative speed (sizes/s) below which a pair is ignored
PARKED_S = 5.0        # road users standing still longer than this (parked, queued) are left out
STILL_PX_S = 12.0
FOOTPRINT = {"vehicle": 0.30, "two_wheeler": 0.20, "person": 0.12}  # ground part of the box height
CALIB_MID = 7.5       # raw score giving risk 0.5 (above the maximum seen on normal traffic in the samples)
CALIB_SLOPE = 1.2
RELEASE_PER_S = 0.5   # smoothed score decays at most this much per second
MAX_STRIDE = 96       # time-budget guard never thins detection below one frame in ~3 s


class _History:
    def __init__(self, maxlen: int):
        self.t: deque = deque(maxlen=maxlen)
        self.box: deque = deque(maxlen=maxlen)
        self.moving_at = -math.inf   # last time the road user was seen moving

    def add(self, t: float, box: np.ndarray) -> None:
        self.t.append(t)
        self.box.append(np.asarray(box[:4], dtype=np.float64))
        if len(self.t) == 1:
            self.moving_at = t

    def parked(self, now: float) -> bool:
        return now - self.moving_at > PARKED_S

    def state(self, now: float):
        """(last box, velocity of the foot point px/s) fitted over HISTORY_S, or None if too short."""
        t = np.asarray(self.t)
        keep = t >= now - HISTORY_S
        if keep.sum() < 3:
            return None
        t, boxes = t[keep], np.asarray(self.box)[keep]
        foot = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
        tc = t - t.mean()
        denom = (tc ** 2).sum()
        if denom <= 1e-9:
            return None
        v = (tc[:, None] * (foot - foot.mean(0))).sum(0) / denom
        if np.linalg.norm(v) > STILL_PX_S:
            self.moving_at = now
        return boxes[-1], v


def footprint(box: np.ndarray, group: str) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([x1, y2 - FOOTPRINT[group] * (y2 - y1), x2, y2])


def _overlap_window(a1, a2, b1, b2, r):
    """Vectorised: times during which interval [b1, b2] moving at relative speed r overlaps [a1, a2]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (a1 - b2) / r, (a2 - b1) / r
    lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
    static = np.abs(r) < 1e-9
    touching = (b1 < a2) & (b2 > a1)
    lo = np.where(static, np.where(touching, -np.inf, np.inf), lo)
    hi = np.where(static, np.where(touching, np.inf, -np.inf), hi)
    return lo, hi


def pair_danger(fa: np.ndarray, va: np.ndarray, fb: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """DRAC-style danger of footprints moving linearly (0 if they will not touch soon).

    Works on single boxes (4,) or on stacked pairs (n, 4) / velocities (n, 2).
    """
    fa, fb, va, vb = (np.atleast_2d(x) for x in (fa, fb, va, vb))
    u = vb - va
    size = lambda f: np.sqrt(np.maximum((f[:, 2] - f[:, 0]) * (f[:, 3] - f[:, 1]) * 3, 1.0))  # noqa: E731
    rel = np.linalg.norm(u, axis=1) / (0.5 * (size(fa) + size(fb)))
    x_lo, x_hi = _overlap_window(fa[:, 0], fa[:, 2], fb[:, 0], fb[:, 2], u[:, 0])
    y_lo, y_hi = _overlap_window(fa[:, 1], fa[:, 3], fb[:, 1], fb[:, 3], u[:, 1])
    enter, leave = np.maximum(x_lo, y_lo), np.minimum(x_hi, y_hi)
    # touching soon; not already overlapping (occlusion / queue); moving relative to each other
    ok = (enter < leave) & (enter > 0.0) & (enter <= HORIZON_S) & (rel >= MIN_REL_SPEED)
    return np.where(ok, rel / (2.0 * np.maximum(enter, 0.1)), 0.0)


class CausalRisk:
    def __init__(self, detector: Detector | None, scene: Scene, proc_fps: float = 5.0):
        self.detector, self.scene, self.proc_fps = detector, scene, proc_fps

    def reset(self, meta: dict, budget_s: float | None = None) -> None:
        """`budget_s`: wall-clock seconds Part B may take for this video (harness decoding included)."""
        fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, round(fps / self.proc_fps))
        # the harness decodes every frame between our calls, so the pace covers its decoding too
        self.pace = Pace(budget_s, total=float(meta.get("n_frames") or 0), warmup=10.0 * fps)
        self.tracker = MultiTracker(fps=fps / self.stride)
        self.hist: dict[int, _History] = {}
        self.group: dict[int, str] = {}
        self.idx = 0
        self.score = 0.0
        self.last_t = 0.0
        self.last_raw = 0.0
        self.last_pair: tuple[int, int] | None = None
        self.prev_danger: dict[tuple[int, int], float] = {}
        self.to_reference: np.ndarray | None = None

    def step(self, frame: np.ndarray, t: float) -> float:
        i, self.idx = self.idx, self.idx + 1
        # going over the time budget would void the whole video (Part A too): thin detection instead
        if i % 100 == 0 and self.stride < MAX_STRIDE and self.pace.over_budget(i):
            self.stride = min(MAX_STRIDE, self.stride * 2)
            self.pace.reset_window()
        if i % self.stride:
            return self.score
        # same input as Part A (area-downscaled scene frame): the calibration is shared, and
        # uploading full 4K frames to the GPU cost more than the inference itself
        scene_frame = cv2.resize(frame, (SCENE_W, SCENE_H), interpolation=cv2.INTER_AREA)
        if self.to_reference is None:  # register the view once, on the first frame we see
            self.to_reference = alignment.estimate(scene_frame)
        dets = self.detector([scene_frame])[0]
        dets[:, :4] = alignment.map_boxes(self.to_reference, dets[:, :4])
        return self.update(t, dets)

    def update(self, t: float, dets: np.ndarray) -> float:
        """Advance with detections (scene px) of the frame at time t; returns the risk score."""
        seen = self.tracker.update(t, dets)
        maxlen = int(HISTORY_S * self.proc_fps * 2) + 2
        for tid, group, det in seen:
            self.hist.setdefault(tid, _History(maxlen)).add(t, det)
            self.group[tid] = group
        raw = self._raw_risk(t, {tid for tid, _, _ in seen})
        self.last_raw = raw
        risk = 1.0 / (1.0 + math.exp(-CALIB_SLOPE * (raw - CALIB_MID)))
        dt, self.last_t = t - self.last_t, t
        self.score = max(risk, self.score - RELEASE_PER_S * dt)
        return self.score

    def _raw_risk(self, now: float, active: set[int]) -> float:
        states = {}
        for tid in active:
            if self.group[tid] not in FOOTPRINT:
                continue  # animals and loose objects are not scored as colliding road users
            st = self.hist[tid].state(now)
            if st is None:
                continue
            box, v = st
            if self.hist[tid].parked(now):
                continue
            if self.scene.on_road(np.array([[(box[0] + box[2]) / 2, box[3]]]))[0]:
                states[tid] = (footprint(box, self.group[tid]), v)
        ids = sorted(states)
        self.last_pair = None
        a_idx, b_idx = np.triu_indices(len(ids), k=1)
        is_vehicle = np.array([self.group[i] == "vehicle" for i in ids], dtype=bool)
        keep = is_vehicle[a_idx] | is_vehicle[b_idx]   # pedestrian/cyclist-only contacts are not traffic accidents
        a_idx, b_idx = a_idx[keep], b_idx[keep]
        danger_now: dict[tuple[int, int], float] = {}
        best = 0.0
        if len(a_idx):
            F = np.array([states[i][0] for i in ids])
            V = np.array([states[i][1] for i in ids])
            danger = pair_danger(F[a_idx], V[a_idx], F[b_idx], V[b_idx])
            for k in np.flatnonzero(danger > 0):
                pair = (ids[a_idx[k]], ids[b_idx[k]])
                danger_now[pair] = float(danger[k])
                # a conflict must persist over two processed frames: one-frame spikes are box jitter
                sustained = min(danger_now[pair], self.prev_danger.get(pair, 0.0))
                if sustained > best:
                    best, self.last_pair = sustained, pair
        self.prev_danger = danger_now
        return best


def replay_detections(times: np.ndarray, detections: list[np.ndarray], scene: Scene,
                      every: int = 2) -> dict:
    """Run the causal model over already-computed detections (scene px), e.g. Part A's.

    Returns per processed frame: time, raw danger, smoothed risk and the most dangerous pair.
    Still causal: each value depends only on detections up to that frame.
    """
    fps = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 10.0
    model = CausalRisk(detector=None, scene=scene, proc_fps=fps / every)
    model.reset({"fps": fps / every, "width": SCENE_W})
    out = {"t": [], "raw": [], "risk": [], "pair": []}
    for k in range(0, len(times), every):
        risk = model.update(float(times[k]), detections[k].copy())
        out["t"].append(float(times[k]))
        out["raw"].append(model.last_raw)
        out["risk"].append(risk)
        out["pair"].append(model.last_pair)
    return {k: (np.asarray(v) if k != "pair" else v) for k, v in out.items()}
