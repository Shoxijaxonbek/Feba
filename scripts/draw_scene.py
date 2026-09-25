"""Overlay configs/scene.json on a frame of the camera (to check or edit the layout).

    python scripts/draw_scene.py samples/C3902.MP4 --t 160 --out outputs/scene_overlay.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadwatch.scene import Scene  # noqa: E402
from roadwatch.video import iter_frames  # noqa: E402

COLOURS = {"carriageway": (80, 80, 80), "islands": (60, 60, 200), "crosswalks": (255, 255, 0),
           "queue_zones": (0, 200, 255), "intersection": (255, 0, 255), "no_stop_exempt": (0, 255, 0)}


def render(frame: np.ndarray, scene: Scene) -> np.ndarray:
    fill, out = frame.copy(), frame.copy()
    for group, colour in COLOURS.items():
        for name, poly in scene.cfg.get(group, {}).items():
            pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
            if group in ("islands", "crosswalks", "no_stop_exempt"):
                cv2.fillPoly(fill, [pts], colour)
            cv2.polylines(out, [pts], True, colour, 2)
            c = np.asarray(poly).mean(0).astype(int)
            cv2.putText(out, f"{group[:-1] if group.endswith('s') else group}:{name}", tuple(c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
    for name, sl in scene.stop_lines.items():
        a, b = sl["line"].astype(int)
        cv2.line(out, tuple(a), tuple(b), (0, 0, 255), 3)
        cv2.putText(out, f"stop:{name}", tuple(a + [0, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for name, sig in scene.signals.items():
        for lamp, (x, y, w, h) in sig["lamps"].items():
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    frame = next(f for t, f in iter_frames(args.video, 10.0) if t >= args.t)
    cv2.imwrite(args.out, render(frame, Scene.load()), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
