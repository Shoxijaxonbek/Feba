"""Pedestrian-related events: jaywalking, failure_to_yield."""
from __future__ import annotations

import numpy as np

from ..segments import Event
from .common import MOVING_SPEED, Context, polygon_distance, runs

# jaywalking
ROAD_MARGIN = 25.0       # px the foot point must be inside the carriageway (curb-side waiting is not jaywalking)
CROSSWALK_MARGIN = 40.0  # px outside every crosswalk (drifting off the stripes / cutting a corner is not flagged)
MIN_JAYWALK_S = 2.0
MIN_PERSON_CONF = 0.45
# failure_to_yield
PED_NEAR_WIDTHS = 1.5    # pedestrian counts if within this many vehicle-box widths of the vehicle...
PED_NEAR_HEIGHTS = 3.5   # ...and this many of their own body heights (~6 m; big near-camera boxes reach too far)
PED_ON_ROAD_PX = 20.0    # ...and stands on the roadway part of the crossing, not on the curb / island
MIN_CROSSING_S = 0.3


def jaywalking(ctx: Context) -> list[Event]:
    scene = ctx.scene
    cw_polys = [z.poly for z in scene.crosswalks.values()]
    road_polys = [z.poly for z in scene.carriageway.values()]
    island_polys = [z.poly for z in scene.islands.values()]
    events = []
    for tr in ctx.pedestrians.values():
        if tr.duration < MIN_JAYWALK_S:
            continue
        p = tr.smooth_foot()
        on_road = scene.on_road(p)
        if not on_road.any():
            continue
        d_road = np.max([polygon_distance(p, poly) for poly in road_polys], axis=0)
        d_island = np.max([polygon_distance(p, poly) for poly in island_polys], axis=0) if island_polys else -np.inf
        d_cw = np.max([polygon_distance(p, poly) for poly in cw_polys], axis=0)
        mask = (on_road & (d_road > ROAD_MARGIN) & (d_island < -ROAD_MARGIN / 2)
                & (d_cw < -CROSSWALK_MARGIN) & (tr.conf >= MIN_PERSON_CONF)
                & ~scene.stop_exempt(p))  # people waiting at the bus stop spill into the curb lane
        # annotation convention: from stepping onto the road until leaving it, so each confirmed
        # jaywalk is widened to the whole stretch the person spends on the carriageway outside a crossing
        outside_crossing = on_road & (d_island < 0) & (d_cw < 0)
        stretches = runs(tr.t, outside_crossing, max_gap=0.6)
        for s, e in runs(tr.t, mask, max_gap=0.6):
            if e - s >= MIN_JAYWALK_S:
                s, e = next(((a, b) for a, b in stretches if a <= s and e <= b), (s, e))
                events.append(Event(s, e, "jaywalking", {tr.tid}))
    return events


def _bottom_points(box: np.ndarray) -> np.ndarray:
    """Three ground-contact samples along the bottom edge of a vehicle box."""
    x1, _, x2, y2 = box
    return np.array([[x1 + 0.15 * (x2 - x1), y2], [(x1 + x2) / 2, y2], [x2 - 0.15 * (x2 - x1), y2]])


def failure_to_yield(ctx: Context) -> list[Event]:
    scene = ctx.scene
    times = ctx.times
    # pedestrians present on each crosswalk, per processed frame: (track id, foot point)
    road_polys = [z.poly for z in scene.carriageway.values()]
    island_polys = [z.poly for z in scene.islands.values()]
    peds_on: dict[str, list[list[tuple[int, np.ndarray, float]]]] = {k: [[] for _ in times] for k in scene.crosswalks}
    for tr in ctx.pedestrians.values():
        p = tr.smooth_foot()
        on_roadway = np.max([polygon_distance(p, poly) for poly in road_polys], axis=0) > PED_ON_ROAD_PX
        if island_polys:
            on_roadway &= np.max([polygon_distance(p, poly) for poly in island_polys], axis=0) < -PED_ON_ROAD_PX
        idx = np.clip(np.searchsorted(times, tr.t - 1e-6), 0, len(times) - 1)
        for name, zone in scene.crosswalks.items():
            inside = zone.contains(p) & on_roadway
            heights = tr.box[:, 3] - tr.box[:, 1]
            for i, pt, h in zip(idx[inside], p[inside], heights[inside]):
                peds_on[name][i].append((tr.tid, pt, h))

    events = []
    for tr in ctx.vehicles.values():
        speed = tr.speed()
        idx = np.clip(np.searchsorted(times, tr.t - 1e-6), 0, len(times) - 1)
        for name, zone in scene.crosswalks.items():
            in_cw = np.array([zone.contains(_bottom_points(b)).any() for b in tr.box])
            if not in_cw.any():
                continue
            conflict_peds: dict[int, set[int]] = {}
            for k in np.flatnonzero(in_cw & (speed > MOVING_SPEED)):
                width = tr.box[k, 2] - tr.box[k, 0]
                near = {pid for pid, pt, h in peds_on[name][idx[k]]
                        if np.linalg.norm(pt - tr.foot[k]) < min(PED_NEAR_WIDTHS * width, PED_NEAR_HEIGHTS * h)}
                if near:
                    conflict_peds[k] = near
            if not conflict_peds:
                continue
            # the event spans the vehicle's whole passage through the crossing
            for s, e in runs(tr.t, in_cw, max_gap=0.5):
                ks = [k for k in conflict_peds if s <= tr.t[k] <= e]
                if e - s >= MIN_CROSSING_S and ks:
                    peds = set().union(*(conflict_peds[k] for k in ks))
                    events.append(Event(s, e, "failure_to_yield", {tr.tid} | peds, {"crosswalk": name}))
    return events
