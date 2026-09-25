"""Trajectory-shape events: stopped_vehicle, wrong_way, illegal_u_turn."""
from __future__ import annotations

import numpy as np

from ..flow import FlowField
from ..segments import Event
from ..tracking import Track
from .common import MOVING_SPEED, STILL_SPEED, Context, runs

# stopped_vehicle
MIN_STOPPED_S = 10.0
SAME_SPOT_IOU = 0.5      # still pieces of fragmented tracks at the same spot are one stop
SAME_SPOT_GAP_S = 5.0
# wrong_way
FLOW_COHERENCE = 0.85
FLOW_MIN_COUNT = 30
WRONG_COS = -0.5
WRONG_MIN_SPEED = 60.0
WRONG_MIN_S = 1.5
WRONG_MIN_PX = 80.0
# u-turn
UTURN_MIN_DEG = 150.0
UTURN_MIN_S = 1.5        # a real U-turn takes time; id switches flip the heading in a frame or two


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def still_periods(ctx: Context, min_s: float = 1.0) -> list[tuple[float, float, np.ndarray, set[int]]]:
    """(start, end, median box, track ids) of every stationary period of every vehicle,
    with pieces from fragmented tracks at the same spot joined together."""
    pieces = []
    for tr in ctx.vehicles.values():
        still = tr.speed() < STILL_SPEED
        for s, e in runs(tr.t, still, max_gap=0.6):
            m = (tr.t >= s) & (tr.t <= e)
            pieces.append([s, e, np.median(tr.box[m], axis=0), {tr.tid}])
    pieces.sort(key=lambda x: x[0])
    merged: list[list] = []
    for s, e, box, tids in pieces:
        for m in merged:
            if s - m[1] <= SAME_SPOT_GAP_S and _iou(box, m[2]) >= SAME_SPOT_IOU:
                m[1] = max(m[1], e)
                m[3] |= tids
                break
        else:
            merged.append([s, e, box, tids])
    return [(s, e, b, tids) for s, e, b, tids in merged if e - s >= min_s]


def stopped_vehicle(ctx: Context) -> list[Event]:
    scene = ctx.scene
    events = []
    for s, e, box, tids in still_periods(ctx, MIN_STOPPED_S):
        foot = np.array([[(box[0] + box[2]) / 2, box[3]]])
        if not scene.on_road(foot)[0] or scene.stop_exempt(foot)[0] or scene.in_queue_zone(foot)[0]:
            continue
        events.append(Event(s, e, "stopped_vehicle", tids, {"box": box.round(1).tolist()}))
    return events


def wrong_way(ctx: Context, flow: FlowField | None) -> list[Event]:
    if flow is None:
        return []
    events = []
    for tr in ctx.vehicles.values():
        if tr.duration < WRONG_MIN_S:
            continue
        p, v = tr.smooth_foot(), tr.velocity(1.0)
        speed = np.linalg.norm(v, axis=1)
        direction, coh, count = flow.lookup(p)
        cos = (v * direction).sum(1) / np.maximum(speed, 1e-6)
        wrong = (speed > WRONG_MIN_SPEED) & (coh > FLOW_COHERENCE) & (count >= FLOW_MIN_COUNT) & (cos < WRONG_COS)
        for s, e in runs(tr.t, wrong, max_gap=0.8):
            m = (tr.t >= s) & (tr.t <= e)
            if e - s >= WRONG_MIN_S and np.linalg.norm(p[m][-1] - p[m][0]) >= WRONG_MIN_PX:
                events.append(Event(s, e, "wrong_way", {tr.tid}))
    return events


def is_continuous(tr: Track, max_jump: float = 0.5, max_area_ratio: float = 1.8) -> bool:
    """False if the track jumps (an id switch onto another vehicle): a step longer than
    `max_jump` box diagonals or a sudden change of box area."""
    if len(tr.t) < 2:
        return True
    diag = np.hypot(tr.box[:, 2] - tr.box[:, 0], tr.box[:, 3] - tr.box[:, 1])
    step = np.linalg.norm(np.diff(tr.center, axis=0), axis=1)
    area = (tr.box[:, 2] - tr.box[:, 0]) * (tr.box[:, 3] - tr.box[:, 1])
    ratio = np.maximum(area[1:], 1) / np.maximum(area[:-1], 1)
    return bool((step < max_jump * diag[:-1]).all() and ((ratio < max_area_ratio) & (ratio > 1 / max_area_ratio)).all())


def illegal_u_turn(ctx: Context) -> list[Event]:
    """Vehicles whose heading turns gradually by >= 150 degrees (a U-turn); at this junction
    the median island carries a keep-right sign, so U-turns around it are prohibited."""
    events = []
    for tr in ctx.vehicles.values():
        if tr.duration < 3.0 or not is_continuous(tr):
            continue
        x1, y1, x2, y2 = tr.box.T
        clipped = (x1 < 5) | (y1 < 5) | (x2 > 1915) | (y2 > 1075)  # foot point is not the ground contact
        v = tr.velocity(1.0)
        moving = (np.linalg.norm(v, axis=1) > MOVING_SPEED) & ~clipped
        if moving.sum() < 10:
            continue
        t, v = tr.t[moving], v[moving]
        heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        edge = max(3, len(heading) // 8)
        h0, h1 = np.median(heading[:edge]), np.median(heading[-edge:])
        reversal = abs((np.degrees(h1 - h0) + 180.0) % 360.0 - 180.0)  # modulo 360: jitter can unwrap a full turn
        if reversal < UTURN_MIN_DEG:
            continue  # net reversal of direction, not just a sharp (perspective-exaggerated) turn
        turned = np.degrees(np.abs(heading - h0))
        start = t[np.argmax(turned > 20.0)]
        end = t[np.argmax(turned >= reversal - 10.0)]
        if end - start >= UTURN_MIN_S:
            events.append(Event(float(start), float(end), "illegal_u_turn", {tr.tid}))
    return events
