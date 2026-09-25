"""Candidate events for building dev labels: loose-threshold rule hits, one image strip each.

Each strip shows 4 frames (start - 1 s, start + 0.5 s, middle, end) cropped around the road
users involved, with them highlighted. A reviewer accepts / rejects each candidate and fixes
its boundaries in labels/<video>.json.

    python scripts/review_candidates.py samples/C3902.MP4 --classes jaywalking failure_to_yield --out outputs/review
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_frames import draw, grab  # noqa: E402
from roadwatch.flow import FlowField  # noqa: E402
from roadwatch.perception import Observation  # noqa: E402
from roadwatch.rules import detect_all, pedestrian  # noqa: E402
from roadwatch.scene import Scene  # noqa: E402

LOOSE = {  # recall-oriented thresholds
    pedestrian: {"ROAD_MARGIN": 10.0, "CROSSWALK_MARGIN": 20.0, "MIN_JAYWALK_S": 1.0, "MIN_PERSON_CONF": 0.3,
                 "PED_NEAR_WIDTHS": 2.5, "PED_ON_ROAD_PX": 6.0},
}


def crop_box(obs: Observation, tids: set[int], s: float, e: float, margin: int = 120) -> tuple[int, int, int, int]:
    boxes = [tr.box[(tr.t >= s - 1) & (tr.t <= e)] for tid, tr in obs.tracks.items() if tid in tids]
    boxes = np.concatenate([b for b in boxes if len(b)] or [np.array([[0, 0, 1920, 1080]])])
    x0, y0 = boxes[:, :2].min(0) - margin
    x1, y1 = boxes[:, 2:].max(0) + margin
    w, h = max(x1 - x0, 480), max(y1 - y0, 270)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(np.clip(cx - w / 2, 0, 1920 - w)), int(np.clip(cy - h / 2, 0, 1080 - h))
    return x0, y0, int(min(1920, x0 + w)), int(min(1080, y0 + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--classes", nargs="*")
    ap.add_argument("--loose", action="store_true", help="recall-oriented thresholds")
    ap.add_argument("--out", default="outputs/review")
    args = ap.parse_args()
    video = Path(args.video)
    obs = Observation.load(Path(args.cache) / f"{video.stem}.pkl")
    if args.loose:
        for module, values in LOOSE.items():
            for k, v in values.items():
                setattr(module, k, v)
    events = detect_all(obs, Scene.load(), FlowField.load(), set(args.classes) if args.classes else None)
    out = Path(args.out) / video.stem
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for i, ev in enumerate(events):
        times = [max(0.0, ev.start - 1.0), ev.start + 0.5, (ev.start + ev.end) / 2, ev.end]
        frames = grab(str(video), times)
        x0, y0, x1, y1 = crop_box(obs, ev.tracks, ev.start, ev.end)
        tiles = []
        for t in times:
            img = draw(frames[t], obs, t, ev.tracks)[y0:y1, x0:x1]
            tiles.append(cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1]))))
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
        strip = np.hstack(tiles)
        cv2.putText(strip, f"#{i} {ev.label} {ev.start:.1f}-{ev.end:.1f}s", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"{i:03d}_{ev.label}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 80])
        listing.append({"i": i, "start": round(ev.start, 2), "end": round(ev.end, 2), "label": ev.label,
                        "tracks": sorted(ev.tracks), "info": ev.info})
    (out / "candidates.json").write_text(json.dumps(listing, indent=1))
    print(f"{len(events)} candidates -> {out}")


if __name__ == "__main__":
    main()
