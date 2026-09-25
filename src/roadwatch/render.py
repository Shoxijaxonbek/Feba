"""Annotated playback: boxes, trails, active events, signal state and risk, as a browser-playable MP4."""
from __future__ import annotations

from fractions import Fraction
from typing import Callable

import av
import cv2
import numpy as np

from . import alignment
from .perception import Observation
from .segments import Event
from .video import iter_frames

GROUP_BGR = {"vehicle": (0, 215, 255), "person": (255, 80, 255), "two_wheeler": (0, 140, 255), "obstacle": (60, 255, 60)}
EVENT_BGR = (40, 40, 255)
SIGNAL_BGR = {"red": (40, 40, 255), "amber": (0, 180, 255), "green": (60, 220, 60), "unknown": (160, 160, 160)}
LABEL_TEXT = {
    "accident": "ACCIDENT", "near_miss": "NEAR MISS", "red_light": "RED-LIGHT RUNNING",
    "wrong_way": "WRONG WAY", "illegal_u_turn": "ILLEGAL U-TURN", "stopped_vehicle": "STOPPED VEHICLE",
    "jaywalking": "JAYWALKING", "failure_to_yield": "FAILURE TO YIELD", "illegal_turn": "ILLEGAL TURN",
    "solid_line_crossing": "SOLID LINE CROSSING", "stop_line": "STOP-LINE VIOLATION",
    "congestion": "CONGESTION", "road_obstacle": "ROAD OBSTACLE", "fire_smoke": "FIRE / SMOKE",
}


class Annotator:
    """Draws the state of the scene at time t onto a 1920x1080 scene frame."""

    def __init__(self, obs: Observation, events: list[Event], risk: list[list[float]] | None = None,
                 signal: tuple[np.ndarray, np.ndarray] | None = None, trail_s: float = 2.0):
        self.obs, self.events, self.trail_s = obs, events, trail_s
        self.risk = np.asarray(risk, dtype=np.float64).reshape(-1, 2) if risk else np.zeros((0, 2))
        self.signal = signal
        # tracks live in reference-view coordinates; draw them where they are in this video
        self.to_frame = alignment.invert(np.asarray(obs.side.get("alignment", alignment.IDENTITY)))

    def draw(self, frame: np.ndarray, t: float) -> np.ndarray:
        img = frame.copy()
        active = [ev for ev in self.events if ev.start - 0.05 <= t <= ev.end + 0.05]
        hot = set().union(*(ev.involved(t) for ev in active)) if active else set()
        for tid, tr in self.obs.tracks.items():
            if not (tr.t[0] - 0.06 <= t <= tr.t[-1] + 0.06):
                continue
            i = int(np.searchsorted(tr.t, t + 0.06)) - 1
            if i < 0 or abs(tr.t[i] - t) > 0.25:
                continue
            colour = EVENT_BGR if tid in hot else GROUP_BGR[tr.group]
            thick = 3 if tid in hot else 2
            m = (tr.t >= t - self.trail_s) & (tr.t <= t + 0.06)
            pts = alignment.apply(self.to_frame, tr.foot[m]).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, colour, thick, cv2.LINE_AA)
            x1, y1, x2, y2 = alignment.map_boxes(self.to_frame, tr.box[i:i + 1])[0].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick, cv2.LINE_AA)
        self._banner(img, t, active)
        return img

    def _banner(self, img: np.ndarray, t: float, active: list[Event]) -> None:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 64), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
        cv2.putText(img, f"{t:7.1f}s", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        x = 200
        if self.signal is not None and len(self.signal[0]):
            st_t, st = self.signal
            i = max(0, int(np.searchsorted(st_t, t)) - 1)
            state = str(st[i])
            cv2.circle(img, (x + 18, 32), 16, SIGNAL_BGR.get(state, SIGNAL_BGR["unknown"]), -1, cv2.LINE_AA)
            cv2.putText(img, "signal", (x + 42, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
            x += 160
        if len(self.risk):
            j = max(0, int(np.searchsorted(self.risk[:, 0], t)) - 1)
            r = float(self.risk[j, 1])
            cv2.rectangle(img, (x, 18), (x + 200, 46), (90, 90, 90), 2)
            colour = (60, 220, 60) if r < 0.3 else (0, 180, 255) if r < 0.5 else (40, 40, 255)
            cv2.rectangle(img, (x + 2, 20), (x + 2 + int(196 * r), 44), colour, -1)
            cv2.putText(img, f"risk {r:.2f}", (x + 212, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
            x += 360
        labels = sorted({LABEL_TEXT[ev.label] for ev in active})
        if labels:
            cv2.putText(img, " | ".join(labels), (x, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2, cv2.LINE_AA)


def render_video(video_path: str, out_path: str, annotator: Annotator, fps: float = 10.0,
                 size: tuple[int, int] = (1280, 720), progress: Callable[[float], None] | None = None,
                 duration: float | None = None,
                 on_frame: Callable[[float, np.ndarray], None] | None = None) -> None:
    """Decode `video_path` at ~`fps`, annotate every frame, encode H.264 (yuv420p, browser-safe).

    The output frame rate is the true sampling rate of the decoded frames (e.g. 10000/1001 for
    every third frame of 29.97 fps), so seeking in the annotated video matches source time.
    `on_frame(t, annotated_scene_frame)` is called for every frame (thumbnails, posters).
    """
    times = annotator.obs.times
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / fps
    rate = Fraction(1.0 / step).limit_denominator(1001)
    with av.open(out_path, mode="w", options={"movflags": "+faststart"}) as out:
        stream = out.add_stream("libx264", rate=rate)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "28", "preset": "veryfast"}
        for t, frame in iter_frames(video_path, target_fps=fps):
            if duration and t > duration:
                break
            annotated = annotator.draw(frame, t)
            if on_frame is not None:
                on_frame(t, annotated)
            img = cv2.resize(annotated, size, interpolation=cv2.INTER_AREA)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                out.mux(packet)
            if progress and duration:
                progress(min(1.0, t / duration))
        for packet in stream.encode():
            out.mux(packet)
