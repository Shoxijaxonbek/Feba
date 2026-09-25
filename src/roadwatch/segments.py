"""Events as time segments, and the post-processing that turns rule hits into the output list."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Event:
    start: float
    end: float
    label: str
    tracks: set[int] = field(default_factory=set)   # track ids that caused it (for review / rendering)
    info: dict = field(default_factory=dict)
    spans: dict[int, list[tuple[float, float]]] = field(default_factory=dict)  # when each track was involved

    def __post_init__(self) -> None:
        for tid in self.tracks:
            self.spans.setdefault(tid, [(self.start, self.end)])

    def involved(self, t: float, pad: float = 0.05) -> set[int]:
        """Tracks taking part in the event at time t."""
        return {tid for tid, spans in self.spans.items() if any(s - pad <= t <= e + pad for s, e in spans)}

    def as_list(self) -> list:
        return [round(self.start, 2), round(self.end, 2), self.label]


# Same-class hits closer than this are one event (the annotation guide merges
# simultaneous same-class events into one segment).
MERGE_GAP = {"jaywalking": 1.0, "failure_to_yield": 0.5, "wrong_way": 1.0}
DEFAULT_MERGE_GAP = 0.0


def merge_intervals(intervals, max_gap: float = 0.0) -> list[tuple[float, float]]:
    """Union of (start, end) intervals; ones closer than `max_gap` are joined."""
    merged: list[list[float]] = []
    for s, e in sorted(intervals):
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def merge_events(events: list[Event], duration: float) -> list[Event]:
    """Clip to the video and merge same-class events that overlap or nearly touch."""
    by_label: dict[str, list[Event]] = {}
    for ev in events:
        s, e = max(0.0, ev.start), min(duration, ev.end)
        if e > s:
            spans = {tid: list(v) for tid, v in ev.spans.items()}
            by_label.setdefault(ev.label, []).append(Event(s, e, ev.label, set(ev.tracks), dict(ev.info), spans))
    out = []
    for label, evs in by_label.items():
        gap = MERGE_GAP.get(label, DEFAULT_MERGE_GAP)
        evs.sort(key=lambda x: x.start)
        cur = evs[0]
        for ev in evs[1:]:
            if ev.start - cur.end <= gap:
                cur.end = max(cur.end, ev.end)
                cur.tracks |= ev.tracks
                for tid, spans in ev.spans.items():
                    cur.spans.setdefault(tid, []).extend(spans)
            else:
                out.append(cur)
                cur = ev
        out.append(cur)
    return sorted(out, key=lambda x: (x.start, x.label))
