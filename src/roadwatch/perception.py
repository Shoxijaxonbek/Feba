"""Video -> sampled frames -> detections -> tracks (+ per-frame side channels).

This is the only part of Part A that touches pixels; everything downstream
(rules, segments) works on the `Observation` it returns, which can be cached
to disk so rules can be tuned without re-running the detector.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from . import alignment
from .detector import Detector
from .tracking import MultiTracker, Track
from .video import VideoInfo, iter_frames, probe


class FrameObserver(Protocol):
    """Anything that wants to look at the sampled frames (signal reader, background model...).

    If it defines `align(m)`, it is called once with the 2x3 affine mapping reference
    scene coordinates to this video's frames (see alignment.py) before the first frame.
    """

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
                  progress: Callable[[float], None] | None = None,
                  max_seconds: float | None = None, max_wall_s: float | None = None) -> Observation:
    """Detect and track road users in `path` (only the first `max_seconds` if given).

    `max_wall_s` is a wall-clock budget: if the run is projected to exceed it (slow
    decoding on the evaluation machine), detection is thinned to every 2nd, then
    every 4th sampled frame. Signal lamps are still read on every frame.
    """
    info = probe(path)
    if max_seconds is not None and info.duration > max_seconds:
        info = replace(info, duration=max_seconds, n_frames=int(max_seconds * info.fps))
    observers = observers or {}
    tracker = MultiTracker(fps=target_fps)
    times: list[float] = []
    all_dets: list[np.ndarray] = []
    buf_t: list[float] = []
    buf_f: list[np.ndarray] = []
    t_start = time.perf_counter()
    t_detect = 0.0
    to_reference = None  # frame -> reference view, estimated on the first frame
    thin, k = 1, 0

    def flush() -> None:
        nonlocal t_detect
        t1 = time.perf_counter()
        dets = detector(buf_f)
        t_detect += time.perf_counter() - t1
        for t, d in zip(buf_t, dets):
            d[:, :4] = alignment.map_boxes(to_reference, d[:, :4])
            tracker.update(t, d)
            times.append(t)
            all_dets.append(d)
        buf_t.clear()
        buf_f.clear()

    for t, frame in iter_frames(path, target_fps=target_fps):
        if t > info.duration:
            break
        if to_reference is None:
            to_reference = alignment.estimate(frame)
            for obs in observers.values():
                if hasattr(obs, "align"):
                    obs.align(alignment.invert(to_reference))
        for obs in observers.values():
            obs.observe(t, frame)
        k += 1
        if (k - 1) % thin:
            continue
        buf_t.append(t)
        buf_f.append(frame)
        if len(buf_f) == batch:
            flush()
            if progress and info.duration:
                progress(min(1.0, t / info.duration))
            if max_wall_s and t > 5.0 and thin < 4:
                projected = (time.perf_counter() - t_start) * info.duration / t
                if projected > max_wall_s:
                    thin *= 2
    if buf_f:
        flush()

    total = time.perf_counter() - t_start
    return Observation(
        info=info,
        times=np.asarray(times),
        tracks=tracker.tracks(),
        detections=all_dets,
        side={name: obs.result() for name, obs in observers.items()}
        | {"alignment": (to_reference if to_reference is not None else alignment.IDENTITY).tolist()},
        timing={"total_sec": round(total, 2), "detect_sec": round(t_detect, 2), "frames": len(times),
                "thinned": thin},
    )
