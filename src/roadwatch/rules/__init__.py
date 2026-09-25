"""Rule-based event detectors. Each rule maps a Context to a list of Events."""
from __future__ import annotations

from ..flow import FlowField
from ..perception import Observation
from ..scene import Scene
from ..segments import Event, merge_events
from .common import Context, build_context
from .conflict import accident, congestion, near_miss
from .motion import illegal_u_turn, stopped_vehicle, wrong_way
from .pedestrian import failure_to_yield, jaywalking
from .signal import red_light, stop_line

__all__ = ["Context", "build_context", "detect_all", "RULES"]

RULES = {
    "jaywalking": lambda ctx, flow: jaywalking(ctx),
    "failure_to_yield": lambda ctx, flow: failure_to_yield(ctx),
    "red_light": lambda ctx, flow: red_light(ctx),
    "stop_line": lambda ctx, flow: stop_line(ctx),
    "stopped_vehicle": lambda ctx, flow: stopped_vehicle(ctx),
    "wrong_way": wrong_way,
    "illegal_u_turn": lambda ctx, flow: illegal_u_turn(ctx),
    "congestion": lambda ctx, flow: congestion(ctx),
}


def detect_all(obs: Observation, scene: Scene, flow: FlowField | None,
               enabled: set[str] | None = None) -> list[Event]:
    """All rule hits, merged per class and clipped to the video."""
    ctx = build_context(obs, scene)
    events: list[Event] = []
    for label, rule in RULES.items():
        if enabled is None or label in enabled:
            events += rule(ctx, flow)
    crashes = accident(ctx)   # near_miss needs to know which conflicts ended in contact
    if enabled is None or "accident" in enabled:
        events += crashes
    if enabled is None or "near_miss" in enabled:
        events += near_miss(ctx, crashes)
    return merge_events(events, obs.info.duration)
