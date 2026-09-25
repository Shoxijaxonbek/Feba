"""Signal-related events at stop lines with a visible signal head: red_light, stop_line."""
from __future__ import annotations

import numpy as np

from ..scene import side_of_line
from ..segments import Event
from ..signals import GREEN, RED
from .common import STILL_SPEED, Context, runs

RED_SETTLED_S = 1.0      # red must have been showing this long (amber/red boundary is ambiguous)
PAST_LINE_PX = 15.0      # front must be this far past the line to count as over it
MAX_EVENT_S = 10.0
MIN_STOP_S = 2.0


def _downstream_distance(foot: np.ndarray, line: np.ndarray, upstream_ref: np.ndarray) -> np.ndarray:
    """Perpendicular distance past the stop line (positive = beyond it, in the direction of travel)."""
    a, b = line
    sign = -np.sign(side_of_line(upstream_ref[None], a, b)[0])
    return sign * side_of_line(foot, a, b) / np.linalg.norm(b - a)


def _upstream_reference(ctx: Context) -> np.ndarray:
    zone = next(iter(ctx.scene.queue_zones.values()))
    return zone.poly.mean(axis=0)


def red_light(ctx: Context) -> list[Event]:
    events = []
    ref = _upstream_reference(ctx)
    for name, sl in ctx.scene.stop_lines.items():
        for tr in ctx.vehicles.values():
            d = _downstream_distance(tr.smooth_foot(), sl["line"], ref)
            crossed = np.flatnonzero((d[1:] > 0) & (d[:-1] <= 0)) + 1
            if len(crossed) == 0 or d[0] > 0:
                continue
            tc = float(tr.t[crossed[0]])
            state = ctx.signal_at(sl["signal"], tc)[0]
            if state != RED or ctx.red_since(sl["signal"], tc) < RED_SETTLED_S:
                continue
            # it must actually proceed into the junction while still red (not just creep over the line)
            inside = ctx.scene.in_intersection(tr.smooth_foot())
            after = np.flatnonzero(inside & (tr.t >= tc))
            if len(after) == 0:
                continue
            t_enter = float(tr.t[after[0]])
            if ctx.signal_at(sl["signal"], t_enter)[0] != RED:
                continue
            left = np.flatnonzero(~inside & (tr.t > t_enter))
            t_end = float(tr.t[left[0]]) if len(left) else float(tr.t[-1])
            events.append(Event(tc, min(t_end, tc + MAX_EVENT_S), "red_light", {tr.tid}, {"stop_line": name}))
    return events


def stop_line(ctx: Context) -> list[Event]:
    events = []
    ref = _upstream_reference(ctx)
    for name, sl in ctx.scene.stop_lines.items():
        for tr in ctx.vehicles.values():
            foot = tr.smooth_foot()
            d = _downstream_distance(foot, sl["line"], ref)
            still = tr.speed() < STILL_SPEED
            over = (d > PAST_LINE_PX) & ~ctx.scene.in_intersection(foot)
            for s, e in runs(tr.t, still & over, max_gap=0.6):
                if e - s < MIN_STOP_S or ctx.signal_at(sl["signal"], s)[0] != RED:
                    continue
                green = ctx.next_change(sl["signal"], s, GREEN)
                end = green if green is not None else ctx.duration
                events.append(Event(s, end, "stop_line", {tr.tid}, {"stop_line": name}))
    return events
