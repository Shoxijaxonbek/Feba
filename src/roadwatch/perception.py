"""Video -> sampled frames -> detections -> tracks (+ per-frame side channels).

This is the only part of Part A that touches pixels; everything downstream
(rules, segments) works on the `Observation` it returns, which can be cached
to disk so rules can be tuned without re-running the detector.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .detector import Detector
from .tracking import MultiTracker, Track
from .video import VideoInfo, iter_frames, probe


class FrameObserver(Protocol):
    """Anything that wants to look at the sampled frames (signal reader, background model...)."""

    def observe(self, t: float, frame: np.ndarray) -> None: ...

    def result(self): ...


@dataclass
class Observation:
    info: VideoInfo
    times: np.ndarray                 # timestamps of the processed frames
    tracks: dict[int, Track]
    detections: list[np.ndarray]      # raw per-frame detector output (for EDA / rendering)
    side: dict = field(default_factory=dict)   # results of FrameObservers, by name
    timing: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "Observation":
        with open(path, "rb") as f:
            return pickle.load(f)


def observe_video(path: str, detector: Detector, target_fps: float = 10.0, batch: int = 8,
                  observers: dict[str, FrameObserver] | None = None,
                  progress: Callable[[float], None] | None = None) -> Observation:
    info = probe(path)
    observers = observers or {}
    tracker = MultiTracker(fps=target_fps)
    times: list[float] = []
    all_dets: list[np.ndarray] = []
    buf_t: list[float] = []
    buf_f: list[np.ndarray] = []
    t_start = time.perf_counter()
    t_detect = 0.0

    def flush() -> None:
        nonlocal t_detect
        t1 = time.perf_counter()
        dets = detector(buf_f)
        t_detect += time.perf_counter() - t1
        for t, d in zip(buf_t, dets):
            tracker.update(t, d)
            times.append(t)
            all_dets.append(d)
        buf_t.clear()
        buf_f.clear()

    for t, frame in iter_frames(path, target_fps=target_fps):
        for obs in observers.values():
            obs.observe(t, frame)
        buf_t.append(t)
        buf_f.append(frame)
        if len(buf_f) == batch:
            flush()
            if progress and info.duration:
                progress(min(1.0, t / info.duration))
    if buf_f:
        flush()

    total = time.perf_counter() - t_start
    return Observation(
        info=info,
        times=np.asarray(times),
        tracks=tracker.tracks(),
        detections=all_dets,
        side={name: obs.result() for name, obs in observers.items()},
        timing={"total_sec": round(total, 2), "detect_sec": round(t_detect, 2), "frames": len(times)},
    )
