"""One tile per detected event (mid-event frame, involved road users highlighted), for fast review.

    python scripts/review_grid.py samples/C3896.MP4 --classes jaywalking failure_to_yield --out outputs/review/C3896_ped.jpg
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
from review_candidates import crop_box  # noqa: E402
from roadwatch.perception import Observation  # noqa: E402
from roadwatch.pipeline import events_from_observation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--classes", nargs="*")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    video = Path(args.video)
    obs = Observation.load(Path(args.cache) / f"{video.stem}.pkl")
    events = [e for e in events_from_observation(obs) if not args.classes or e.label in args.classes]
    listing, tiles = [], []
    for i, ev in enumerate(events):
        tm = round((ev.start + ev.end) / 2, 2)
        frame = grab(str(video), [tm])[tm]
        x0, y0, x1, y1 = crop_box(obs, ev.involved(tm) or ev.tracks, ev.start, ev.end, margin=60)
        img = draw(frame, obs, tm, ev.involved(tm) or ev.tracks, trail=max(2.0, tm - ev.start))[y0:y1, x0:x1]
        tile = cv2.resize(img, (480, 270))
        cv2.putText(tile, f"#{i} {ev.label[:4]} {ev.start:.1f}-{ev.end:.1f}", (4, 262),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(tile)
        listing.append({"i": i, "start": round(ev.start, 2), "end": round(ev.end, 2), "label": ev.label,
                        "tracks": sorted(ev.tracks), "info": {k: v for k, v in ev.info.items() if k != "box"}})
    while len(tiles) % args.cols:
        tiles.append(np.zeros((270, 480, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 78])
    Path(args.out).with_suffix(".json").write_text(json.dumps(listing, indent=1))
    print(f"{len(events)} events -> {args.out}")


if __name__ == "__main__":
    main()
