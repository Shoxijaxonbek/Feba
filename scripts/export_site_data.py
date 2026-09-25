"""Produce the website's per-video results: annotated MP4, poster, event thumbnails and JSON.

The risk curve and timings are taken from the official harness output
(predictions_samples.json) when it covers the video, so the site shows exactly
what we submit; otherwise the causal model is replayed on the cached detections.

    python scripts/export_site_data.py samples/*.MP4 --pred predictions_samples.json --out website/data/samples
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadwatch.perception import Observation  # noqa: E402
from roadwatch.pipeline import events_from_observation, get_scene, observe  # noqa: E402
from roadwatch.report import build_result, render_outputs  # noqa: E402
from roadwatch.risk import replay_detections  # noqa: E402


def load_or_observe(video: Path, cache_dir: Path) -> Observation:
    cache = cache_dir / f"{video.stem}.pkl"
    if cache.exists():
        return Observation.load(cache)
    obs = observe(str(video))
    obs.save(cache)
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--pred", default="predictions_samples.json")
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--out", default="website/data/samples")
    ap.add_argument("--no-render", action="store_true", help="JSON only (reuse existing media)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = json.loads(Path(args.pred).read_text()) if Path(args.pred).exists() else {"videos": {}, "log": {}}
    index = []
    for video in map(Path, args.videos):
        obs = load_or_observe(video, Path(args.cache))
        events = events_from_observation(obs)
        entry = pred["videos"].get(video.name)
        if entry and entry.get("risk"):
            risk = entry["risk"]
        else:
            r = replay_detections(obs.times, obs.detections, get_scene())
            risk = np.stack([r["t"], r["risk"]], 1).tolist()
        log = pred.get("log", {}).get(video.name, {})
        timing = {"video_sec": round(obs.info.duration, 1), "part_a_sec": log.get("part_a_sec"),
                  "part_b_sec": log.get("part_b_sec"), "budget_sec": log.get("budget_sec")}
        result = build_result(video.stem, obs, events, risk, media_prefix=f"data/samples/", timing=timing)
        if not args.no_render:
            render_outputs(str(video), video.stem, obs, events, risk, out,
                           progress=lambda p, v=video.stem: print(f"\r{v}: rendering {p:5.1%}", end="", flush=True))
            print()
        (out / f"{video.stem}.json").write_text(json.dumps(result))
        index.append({k: result[k] for k in ("id", "duration", "fps", "width", "height", "annotated_video", "poster")}
                     | {"file": video.name, "n_events": len(events)})
        print(f"{video.name}: {len(events)} events")
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
