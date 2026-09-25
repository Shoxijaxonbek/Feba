"""Render frames at given times with the cached tracks drawn on top (for reviewing events).

    python scripts/inspect_frames.py samples/C3902.MP4 outputs/cache/C3902.pkl 85 90 95 --tracks 758 778 --out o.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadwatch.perception import Observation  # noqa: E402
from roadwatch.video import SCENE_H, SCENE_W  # noqa: E402

GROUP_COLOURS = {"vehicle": (0, 220, 255), "person": (255, 0, 255), "two_wheeler": (0, 128, 255)}


def grab(video: str, times: list[float]) -> dict[float, np.ndarray]:
    out = {}
    with av.open(video) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        t0 = s.start_time or 0
        for want in sorted(times):
            c.seek(int(t0 + max(0.0, want - 1.0) / s.time_base), stream=s, backward=True)
            for fr in c.decode(s):
                t = float((fr.pts - t0) * s.time_base)
                if t >= want - 0.02:
                    out[want] = fr.to_ndarray(width=SCENE_W, height=SCENE_H, format="bgr24")
                    break
    return out


def draw(frame: np.ndarray, obs: Observation, t: float, highlight: set[int], trail: float = 3.0) -> np.ndarray:
    img = frame.copy()
    for tid, tr in obs.tracks.items():
        m = (tr.t <= t + 0.06) & (tr.t >= t - trail)
        if not m.any():
            continue
        hot = tid in highlight
        col = (0, 0, 255) if hot else GROUP_COLOURS[tr.group]
        p = tr.foot[m].astype(int)
        cv2.polylines(img, [p.reshape(-1, 1, 2)], False, col, 3 if hot else 1)
        if abs(tr.t[m][-1] - t) < 0.15:
            x1, y1, x2, y2 = tr.box[m][-1].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if hot else 1)
            cv2.putText(img, str(tid), (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1 + hot)
    cv2.putText(img, f"{t:.1f}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("cache")
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--tracks", nargs="*", type=int, default=[])
    ap.add_argument("--crop", nargs=4, type=int, help="x0 y0 x1 y1 in scene px")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--width", type=int, default=960, help="tile width")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    obs = Observation.load(args.cache)
    frames = grab(args.video, args.times)
    tiles = []
    for t in args.times:
        img = draw(frames[t], obs, t, set(args.tracks))
        if args.crop:
            x0, y0, x1, y1 = args.crop
            img = img[y0:y1, x0:x1]
        scale = args.width / img.shape[1]
        tiles.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
