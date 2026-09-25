"""Traffic-signal state from the colour of the lamps visible in the frame.

Each signal head in configs/scene.json lists small boxes around its red and
green lamps. Per frame we measure how strongly each lamp glows in its own
colour; per video, a lamp is "on" when that glow is well above its dark level.
"""
from __future__ import annotations

import numpy as np

RED, AMBER, GREEN, UNKNOWN = "red", "amber", "green", "unknown"


def lamp_glow(frame: np.ndarray, box: list[float], colour: str) -> float:
    """90th-percentile colour-specific glow inside `box` = (x, y, w, h) in scene px."""
    x, y, w, h = (int(round(v)) for v in box)
    roi = frame[max(0, y):y + h, max(0, x):x + w].astype(np.float32)
    if roi.size == 0:
        return 0.0
    b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    if colour == "red":
        glow = r - np.maximum(g, b)
    elif colour == "amber":
        glow = np.minimum(r, g) - b
    else:  # green lamps render cyan-green: strong G (and B), weak R
        glow = g - r
    return float(np.percentile(glow, 90))


class SignalReader:
    """FrameObserver collecting per-frame lamp glow for every configured signal head."""

    def __init__(self, signals: dict):
        self.signals = signals
        self.t: list[float] = []
        self.values: dict[str, dict[str, list[float]]] = {
            name: {lamp: [] for lamp in cfg["lamps"]} for name, cfg in signals.items()}

    def observe(self, t: float, frame: np.ndarray) -> None:
        self.t.append(t)
        for name, cfg in self.signals.items():
            for lamp, box in cfg["lamps"].items():
                self.values[name][lamp].append(lamp_glow(frame, box, lamp))

    def result(self) -> dict:
        return {"t": np.asarray(self.t),
                "glow": {n: {k: np.asarray(v) for k, v in d.items()} for n, d in self.values.items()}}


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
