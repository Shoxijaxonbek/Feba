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
NEIGHBOUR_WIDTHS = 3.0   # traffic within this many box widths counts as the vehicle's surroundings
NEIGHBOURS_MOVING = 0.5  # ...and must be moving at least this share of the time
MIN_NEIGHBOUR_OBS = 20
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


def _neighbours_moving(ctx: Context, s: float, e: float, box: np.ndarray, tids: set[int]) -> float | None:
    """Share of other vehicles' observations near `box` during [s, e] that are moving.

    None when there is (almost) no traffic around: an isolated stationary vehicle.
    """
    foot = np.array([(box[0] + box[2]) / 2, box[3]])
    radius = NEIGHBOUR_WIDTHS * (box[2] - box[0])
    moving = total = 0
    for tr in ctx.vehicles.values():
        if tr.tid in tids or tr.t[-1] < s or tr.t[0] > e:
            continue
        m = (tr.t >= s) & (tr.t <= e)
        near = np.linalg.norm(tr.smooth_foot()[m] - foot, axis=1) < radius
        total += int(near.sum())
        moving += int((near & (tr.speed()[m] > MOVING_SPEED)).sum())
    return moving / total if total >= MIN_NEIGHBOUR_OBS else None


def stopped_vehicle(ctx: Context) -> list[Event]:
    """Stationary >= 10 s on a carriageway while the traffic around it keeps moving.

    Queues at any of the junction's signals, left-turners waiting inside the junction box and
    congestion all stand still together with their neighbours, which is what separates them
    from a stopped vehicle. Buses at the bus stop are exempt.
    """
    scene = ctx.scene
    stop_zones = [sl["zone"] for sl in scene.stop_lines.values()]
    events = []
    for s, e, box, tids in still_periods(ctx, MIN_STOPPED_S):
        foot = np.array([[(box[0] + box[2]) / 2, box[3]]])
        if not scene.on_road(foot)[0] or scene.in_intersection(foot)[0] or scene.in_queue_zone(foot)[0]:
            continue
        if any(z.contains(foot)[0] for z in stop_zones):
            continue
        if box[0] < 5 or box[2] > 1915 or box[3] > 1075:
            continue  # cut off by the frame edge: position and stillness are unreliable
        labels = {ctx.vehicles[t].label for t in tids if t in ctx.vehicles}
        if labels == {"bus"} and scene.stop_exempt(foot)[0]:
            continue
        share = _neighbours_moving(ctx, s, e, box, tids)
        if share is not None and share < NEIGHBOURS_MOVING:
            continue  # everyone around is stopped too: a queue or a jam
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
        # inside the junction box several legal flows cross (each signal phase has its own),
        # so the learned direction is only meaningful on the carriageways around it
        wrong &= ~ctx.scene.in_intersection(p)
        x1, y1, x2, y2 = tr.box.T
        wrong &= (x1 > 5) & (y1 > 5) & (x2 < 1915) & (y2 < 1075)  # clipped boxes: foot point is not the ground contact
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
        # the manoeuvre is the contiguous turning stretch around the point where the heading passes 90 deg
        turned = np.degrees(np.abs(heading - h0))
        apex = int(np.argmax(turned >= 90.0))
        i0 = apex
        while i0 > 0 and turned[i0 - 1] > 20.0:
            i0 -= 1
        i1 = apex
        while i1 < len(turned) - 1 and turned[i1] < reversal - 10.0:
            i1 += 1
        start, end = t[i0], t[i1]
        if end - start >= UTURN_MIN_S:
            events.append(Event(float(start), float(end), "illegal_u_turn", {tr.tid}))
    return events
