"""
solution.py — the interface the organizers' harness (run_submission.py) imports.

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The implementation lives in src/roadwatch/:
    video.py       fast sampled decoding (PyAV, non-reference frames skipped)
    detector.py    YOLO11m (COCO) with GPU pre/post-processing
    tracking.py    ByteTrack per road-user group
    signals.py     traffic-light state from lamp colours
    scene.py       hand-annotated layout of the fixed camera (configs/scene.json)
    flow.py        lane directions learned from the sample videos (configs/flow_field.npz)
    rules/         one rule per event class, on tracks + layout + signal state
    risk.py        causal time-to-collision risk for Part B
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from roadwatch import pipeline  # noqa: E402
from roadwatch.risk import CausalRisk  # noqa: E402

# Official class ids. Every class stays listed; a rule that never fires simply adds nothing.
CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A — [[start_sec, end_sec, label], ...] for one video."""
    return pipeline.detect_events(video_path)


class RiskEstimator:
    """Part B — causal accident anticipation. Sees only the frames passed to step()."""

    def __init__(self) -> None:
        self._risk = CausalRisk(pipeline.get_detector(), pipeline.get_scene())

    def reset(self, meta: dict) -> None:
        self._risk.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._risk.step(frame, t_sec)
