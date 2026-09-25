"""Everything the website shows for one processed video (sample videos and live-demo uploads).

    result = build_result(video_id, obs, events, risk_curve)
    render_outputs(video_path, obs, events, risk_curve, out_dir)   # annotated MP4 + event thumbnails

The result dict follows website/DATA_CONTRACT.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detector import NAMES
from .perception import Observation
from .render import Annotator, render_video
from .segments import Event
from .signals import signal_states


def signal_changes(obs: Observation, name: str = "main") -> list[list]:
    reading = obs.side.get("signals")
    if reading is None or name not in reading["glow"]:
        return []
    t, st = signal_states(reading, name)
    out = []
    for ti, si in zip(t, st):
        if not out or out[-1][1] != si:
            out.append([round(float(ti), 2), str(si)])
    return out


def counts_per_second(obs: Observation) -> dict:
    """Detections per class, averaged over each second (conf >= 0.35)."""
    n = int(np.ceil(obs.info.duration))
    sums = np.zeros((n, len(NAMES)))
    frames = np.zeros(n)
    for t, d in zip(obs.times, obs.detections):
        s = min(int(t), n - 1)
        frames[s] += 1
        d = d[d[:, 4] >= 0.35]
        np.add.at(sums[s], d[:, 5].astype(int), 1)
    avg = sums / np.maximum(frames, 1)[:, None]
    out = {"t": list(range(n))}
    for k, name in enumerate(NAMES):
        out[name] = [round(float(v), 2) for v in avg[:, k]]
    return out


def downsample_curve(curve: list[list[float]] | np.ndarray, hz: float = 5.0) -> list[list[float]]:
    curve = np.asarray(curve, dtype=np.float64).reshape(-1, 2)
    if len(curve) == 0:
        return []
    keep, next_t = [], -1.0
    for t, s in curve:
        if t >= next_t:
            keep.append([round(float(t), 2), round(float(s), 3)])
            next_t = t + 1.0 / hz - 1e-6
    return keep


def build_result(video_id: str, obs: Observation, events: list[Event], risk_curve,
                 media_prefix: str, timing: dict | None = None) -> dict:
    return {
        "id": video_id,
        "duration": round(obs.info.duration, 2),
        "fps": round(obs.info.fps, 3),
        "width": obs.info.width,
        "height": obs.info.height,
        "annotated_video": f"{media_prefix}{video_id}_annotated.mp4",
        "poster": f"{media_prefix}{video_id}_poster.jpg",
        "events": [{"start": round(ev.start, 2), "end": round(ev.end, 2), "label": ev.label,
                    "tracks": sorted(int(t) for t in ev.tracks),
                    "thumb": f"{media_prefix}thumbs/{video_id}_{i:03d}.jpg",
                    "info": {k: v for k, v in ev.info.items() if k in ("crosswalk", "stop_line")}}
                   for i, ev in enumerate(events)],
        "risk": downsample_curve(risk_curve),
        "signal": signal_changes(obs),
        "counts": counts_per_second(obs),
        "timing": timing or {},
    }


def render_outputs(video_path: str, video_id: str, obs: Observation, events: list[Event], risk_curve,
                   out_dir: str | Path, progress: Callable[[float], None] | None = None) -> None:
    """Annotated MP4, a poster frame and one thumbnail per event (highlighting its road users)."""
    out_dir = Path(out_dir)
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    reading = obs.side.get("signals")
    sig = signal_states(reading, "main") if reading is not None and "main" in reading["glow"] else None
    annotator = Annotator(obs, events, [list(p) for p in np.asarray(risk_curve).reshape(-1, 2)], sig)
    thumb_at = {i: (ev.start + ev.end) / 2 for i, ev in enumerate(events)}
    poster_t = float(obs.times[-1]) * 0.3 if len(obs.times) else 0.0

    def on_frame(t: float, img: np.ndarray) -> None:
        nonlocal poster_t
        for i in [i for i, ts in thumb_at.items() if t >= ts]:
            cv2.imwrite(str(out_dir / "thumbs" / f"{video_id}_{i:03d}.jpg"),
                        cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
            del thumb_at[i]
        if poster_t is not None and t >= poster_t:
            cv2.imwrite(str(out_dir / f"{video_id}_poster.jpg"),
                        cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
            poster_t = None

    render_video(video_path, str(out_dir / f"{video_id}_annotated.mp4"), annotator, on_frame=on_frame,
                 progress=progress, duration=obs.info.duration)
