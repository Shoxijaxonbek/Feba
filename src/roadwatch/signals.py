"""Traffic-signal state from the colour of the lamps visible in the frame.

Each signal head in configs/scene.json lists small boxes around its lamps. Per
frame we measure how strongly each lamp glows in its own colour; per video, a
lamp is "on" when that glow is well above its dark level.
"""
from __future__ import annotations

import numpy as np

from .alignment import map_lamp_boxes

RED, AMBER, GREEN, UNKNOWN = "red", "amber", "green", "unknown"


def glow_map(pixels: np.ndarray, colour: str) -> np.ndarray:
    """Per-pixel glow of a lamp colour for BGR pixels (..., 3)."""
    px = pixels.astype(np.float32)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    if colour == "red":
        return r - np.maximum(g, b)
    if colour == "amber":
        return np.minimum(r, g) - b
    return g - r  # green lamps render cyan-green: strong G (and B), weak R


def _box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 neighbourhood of every pixel, for a stack (T, H, W)."""
    k = 2 * r + 1
    c = np.pad(np.pad(a, ((0, 0), (r, r), (r, r)), mode="edge").cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    return (c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]) / (k * k)


LOCATE_PX = 6   # a lamp is looked for this far around its registered position
SPOT = 2        # the glow is read as the mean of a (2*SPOT+1)^2 px spot on the lamp


class SignalReader:
    """FrameObserver producing a glow time series for every lamp of every configured signal head.

    Registration (alignment.py) places the lamp boxes to within a few pixels, but a lamp
    is only ~8 px wide and the head repeats the same shape three times. So the reader
    keeps a small crop around each head for the whole video and, at the end, locates
    every lamp as the spot whose own colour changes most over time (it switches on and
    off) within LOCATE_PX of the registered position, then reads its glow there.
    """

    def __init__(self, signals: dict):
        self.signals = signals
        self.t: list[float] = []
        self.crops: dict[str, list[np.ndarray]] = {name: [] for name in signals}
        self.windows: dict[str, tuple[int, int, int, int]] = {}

    def align(self, reference_to_frame: np.ndarray) -> None:
        self.signals = map_lamp_boxes(reference_to_frame, self.signals)

    def _window(self, name: str) -> tuple[int, int, int, int]:
        if name not in self.windows:
            boxes = np.array(list(self.signals[name]["lamps"].values()), dtype=np.float64)
            margin = LOCATE_PX + SPOT
            x0, y0 = np.floor(boxes[:, :2].min(0) - margin).astype(int)
            x1, y1 = np.ceil((boxes[:, :2] + boxes[:, 2:]).max(0) + margin).astype(int)
            self.windows[name] = (max(0, x0), max(0, y0), x1, y1)
        return self.windows[name]

    def observe(self, t: float, frame: np.ndarray) -> None:
        self.t.append(t)
        for name in self.signals:
            x0, y0, x1, y1 = self._window(name)
            self.crops[name].append(frame[y0:y1, x0:x1].copy())

    def result(self) -> dict:
        glow: dict[str, dict[str, np.ndarray]] = {}
        spots: dict[str, dict[str, list[int]]] = {}
        for name, cfg in self.signals.items():
            glow[name], spots[name] = {}, {}
            if not self.crops[name]:
                continue
            stack = np.stack(self.crops[name])
            x0, y0, _, _ = self._window(name)
            for lamp, (x, y, w, h) in cfg["lamps"].items():
                g = _box_mean(glow_map(stack, lamp), SPOT)
                swing = np.percentile(g, 95, axis=0) - np.percentile(g, 5, axis=0)
                cx, cy = int(round(x + w / 2 - x0)), int(round(y + h / 2 - y0))
                ya, xa = max(0, cy - LOCATE_PX), max(0, cx - LOCATE_PX)
                region = swing[ya:cy + LOCATE_PX + 1, xa:cx + LOCATE_PX + 1]
                by, bx = np.unravel_index(np.argmax(region), region.shape)
                glow[name][lamp] = g[:, ya + by, xa + bx]
                spots[name][lamp] = [x0 + xa + int(bx), y0 + ya + int(by)]
        return {"t": np.asarray(self.t), "glow": glow, "spots": spots}


def _lamp_on(glow: np.ndarray, min_contrast: float = 25.0) -> np.ndarray:
    """Two-level threshold: midway between the dark and lit levels of this lamp in this video."""
    if len(glow) == 0:
        return np.zeros(0, dtype=bool)
    lo, hi = np.percentile(glow, 10), np.percentile(glow, 90)
    if hi - lo < min_contrast:  # lamp never changed state: decide on absolute glow
        return glow > 40.0
    return glow > (lo + hi) / 2


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=1)


def _fill_short_gaps(t: np.ndarray, code: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Unknown (0) runs shorter than max_gap_s take the previous state (flashing green, brief occlusion)."""
    code = code.copy()
    i = 0
    while i < len(code):
        if code[i] == 0 and i > 0:
            j = i
            while j < len(code) and code[j] == 0:
                j += 1
            if j < len(code) and t[j] - t[i - 1] <= max_gap_s:
                code[i:j] = code[i - 1]
            i = j
        else:
            i += 1
    return code


def signal_states(reading: dict, name: str, smooth: int = 5, max_gap_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """(times, state array of RED/AMBER/GREEN/UNKNOWN) for one signal head."""
    t = reading["t"]
    glow = reading["glow"].get(name, {})
    state = np.full(len(t), UNKNOWN, dtype=object)
    code = np.zeros(len(t))  # 0 unknown, 1 red, 2 amber, 3 green
    if "red" in glow:
        code[_lamp_on(glow["red"])] = 1
    if "amber" in glow:
        code[_lamp_on(glow["amber"]) & (code == 0)] = 2
    if "green" in glow:
        code[_lamp_on(glow["green"])] = 3
    code = _fill_short_gaps(t, _median_filter(code, smooth), max_gap_s)
    for c, s in ((1, RED), (2, AMBER), (3, GREEN)):
        state[code == c] = s
    return t, state


def phase_changes(t: np.ndarray, state: np.ndarray, to: str) -> np.ndarray:
    """Times at which the signal switches into state `to`."""
    idx = np.flatnonzero((state[1:] == to) & (state[:-1] != to)) + 1
    return t[idx]
