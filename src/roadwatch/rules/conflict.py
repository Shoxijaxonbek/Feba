"""Interactions between road users: accident, near_miss; and congestion.

There is no accident or near miss in the sample videos, so these rules are
built from the class definitions and tuned to stay silent on normal traffic:

accident   the ground footprints of two road users touch while they were
           approaching each other, and right after contact both come to a
           stop (outside a signal queue) and stay stopped. Start = contact,
           end = when both are at rest (the annotation convention).
near_miss  the causal conflict score of Part B (required deceleration to avoid
           contact, see risk.py) stays above the alarm level for a moment and
           no contact follows. Start = onset of the conflict, end = when the
           pair is clear again.
congestion the signal-controlled approach stays jammed although it has green.
"""
from __future__ import annotations

import numpy as np

from ..risk import CALIB_MID
from ..segments import Event
from ..signals import GREEN
from .common import STILL_SPEED, Context, runs

CONTACT_OVERLAP = 0.15   # overlap (over the smaller one) of the ground parts of two boxes = contact
MIN_CLOSING = 0.6        # closing speed just before contact, in box sizes per second
STOP_WITHIN_S = 2.5      # both must be at rest this soon after contact...
STAY_STOPPED_S = 4.0     # ...and stay there
NEAR_MISS_RAW = CALIB_MID
NEAR_MISS_MIN_S = 0.4
CLEAR_AFTER_S = 1.0
CONGESTION_MIN_S = 10.0
CONGESTION_MIN_VEHICLES = 6


def _ground_boxes(boxes: np.ndarray) -> np.ndarray:
    """Lower 35% of each box: roughly the footprint of the road user in the image."""
    g = boxes.copy()
    g[:, 1] = boxes[:, 3] - 0.35 * (boxes[:, 3] - boxes[:, 1])
    return g


def _overlap_matrix(g: np.ndarray) -> np.ndarray:
    """Pairwise intersection over the smaller box."""
    x1 = np.maximum(g[:, None, 0], g[None, :, 0])
    y1 = np.maximum(g[:, None, 1], g[None, :, 1])
    x2 = np.minimum(g[:, None, 2], g[None, :, 2])
    y2 = np.minimum(g[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / np.maximum(np.minimum(area[:, None], area[None, :]), 1e-6)


def _first_contacts(ctx: Context) -> dict[tuple[int, int], int]:
    """(track a, track b) -> index into ctx.times of their first ground contact (>= 1 vehicle)."""
    tracks = {**ctx.vehicles, **ctx.pedestrians, **ctx.two_wheelers}
    at_frame: list[list[tuple[int, np.ndarray]]] = [[] for _ in ctx.times]
    for tid, tr in tracks.items():
        idx = np.clip(np.searchsorted(ctx.times, tr.t - 1e-6), 0, len(ctx.times) - 1)
        for i, box in zip(idx, tr.box):
            at_frame[i].append((tid, box))
    first: dict[tuple[int, int], int] = {}
    for i, items in enumerate(at_frame):
        if len(items) < 2:
            continue
        ids = np.array([tid for tid, _ in items])
        ov = _overlap_matrix(_ground_boxes(np.array([b for _, b in items])))
        for a, b in zip(*np.nonzero(np.triu(ov > CONTACT_OVERLAP, k=1))):
            pair = tuple(sorted((int(ids[a]), int(ids[b]))))
            if pair not in first and (pair[0] in ctx.vehicles or pair[1] in ctx.vehicles):
                first[pair] = i
    return first


def accident(ctx: Context) -> list[Event]:
    events = []
    tracks = {**ctx.vehicles, **ctx.pedestrians, **ctx.two_wheelers}
    for (ta, tb), i in _first_contacts(ctx).items():
        a, b = tracks[ta], tracks[tb]
        t0 = float(ctx.times[i])
        ia, ib = int(np.argmin(np.abs(a.t - t0))), int(np.argmin(np.abs(b.t - t0)))
        if ia < 3 or ib < 3:
            continue  # contact from the first sighting on: occlusion, not a collision
        if ctx.scene.in_queue_zone(a.foot[ia:ia + 1])[0] and ctx.scene.in_queue_zone(b.foot[ib:ib + 1])[0]:
            continue  # bumper-to-bumper queue
        # approaching each other just before contact
        pa = np.flatnonzero((a.t >= t0 - 1.0) & (a.t < t0))
        pb = np.flatnonzero((b.t >= t0 - 1.0) & (b.t < t0))
        if len(pa) < 2 or len(pb) < 2:
            continue
        d_before = np.linalg.norm(a.foot[pa[0]] - b.foot[pb[0]])
        d_contact = np.linalg.norm(a.foot[ia] - b.foot[ib])
        size = 0.5 * (np.sqrt(np.prod(a.box[ia, 2:] - a.box[ia, :2])) + np.sqrt(np.prod(b.box[ib, 2:] - b.box[ib, :2])))
        closing = (d_before - d_contact) / max(1e-3, t0 - a.t[pa[0]]) / max(size, 1.0)
        if closing < MIN_CLOSING:
            continue
        # both come to rest right after contact and stay there
        rest = []
        for tr in (a, b):
            still = (tr.speed() < STILL_SPEED) & (tr.t >= t0)
            spans = [(s, e) for s, e in runs(tr.t, still, 0.6) if s - t0 <= STOP_WITHIN_S]
            if not spans or spans[0][1] - spans[0][0] < STAY_STOPPED_S:
                break
            rest.append(spans[0][0])
        else:
            events.append(Event(t0, max(rest + [t0 + 0.5]), "accident", {ta, tb}))
    return events


def near_miss(ctx: Context, accidents: list[Event]) -> list[Event]:
    if ctx.risk is None:
        return []
    t, raw, pairs = ctx.risk["t"], ctx.risk["raw"], ctx.risk["pair"]
    events = []
    for s, e in runs(t, raw >= NEAR_MISS_RAW, max_gap=0.5):
        if e - s < NEAR_MISS_MIN_S:
            continue
        if any(ev.start <= e + 2.0 and ev.end >= s for ev in accidents):
            continue  # it turned into contact: that is an accident, not a near miss
        m = (t >= s) & (t <= e)
        involved = {tid for k in np.flatnonzero(m) if pairs[k] for tid in pairs[k]}
        events.append(Event(s, e + CLEAR_AFTER_S, "near_miss", involved, {"note": "ids from the causal tracker"}))
    return events


def congestion(ctx: Context) -> list[Event]:
    events = []
    times = ctx.times
    for name, sl in ctx.scene.stop_lines.items():
        members: list[set[int]] = [set() for _ in times]
        green = ctx.signal_at(sl["signal"], times) == GREEN
        for tr in ctx.vehicles.values():
            idx = np.clip(np.searchsorted(times, tr.t - 1e-6), 0, len(times) - 1)
            queued = ctx.scene.queue_zones[sl["queue_zone"]].contains(tr.smooth_foot()) & (tr.speed() < 2 * STILL_SPEED)
            for i in idx[queued]:
                members[i].add(tr.tid)
        jammed = green & (np.array([len(m) for m in members]) >= CONGESTION_MIN_VEHICLES)
        for s, e in runs(times, jammed, max_gap=1.0):
            if e - s >= CONGESTION_MIN_S:
                m = (times >= s) & (times <= e)
                events.append(Event(s, e, "congestion", set().union(*(members[i] for i in np.flatnonzero(m)))))
    return events
