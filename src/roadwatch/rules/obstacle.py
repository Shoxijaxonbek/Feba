"""road_obstacle: an animal, or a loose object nobody is holding, on the carriageway.

Detected with the COCO classes the detector already knows (dog, cat, horse,
sheep, cow; backpack, handbag, suitcase). Debris of other kinds is not detected.
Start = obstacle appears on the road, end = it is gone (the annotation convention).
"""
from __future__ import annotations

import numpy as np

from ..segments import Event
from .common import STILL_SPEED, Context, box_overlap_frac, runs

ANIMAL_MIN_S = 2.0
OBJECT_MIN_S = 5.0        # a bag must lie there, stationary and unattended, for this long
MIN_CONF = 0.4


def road_obstacle(ctx: Context) -> list[Event]:
    people = list(ctx.pedestrians.values())
    events = []
    for tr in ctx.obstacles.values():
        foot = tr.smooth_foot()
        on_road = ctx.scene.on_road(foot) & (tr.conf >= MIN_CONF)
        if tr.label == "animal":
            mask, min_s = on_road, ANIMAL_MIN_S
        else:
            attended = np.zeros(len(tr.t), dtype=bool)
            for k, (t, box) in enumerate(zip(tr.t, tr.box)):
                near = [p.box[np.argmin(np.abs(p.t - t))] for p in people
                        if p.t[0] <= t <= p.t[-1] and np.min(np.abs(p.t - t)) < 0.3]
                attended[k] = bool(near) and box_overlap_frac(box, np.array(near)).max() > 0.0
            mask, min_s = on_road & ~attended & (tr.speed() < STILL_SPEED), OBJECT_MIN_S
        for s, e in runs(tr.t, mask, max_gap=1.0):
            if e - s >= min_s:
                events.append(Event(s, e, "road_obstacle", {tr.tid}, {"kind": tr.label}))
    return events

