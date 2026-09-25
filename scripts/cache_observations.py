"""Run perception (detector + tracker + signal reader) on videos and cache the result.

Rules and thresholds can then be tuned on the cached observations in seconds
instead of re-running the detector:

    python scripts/cache_observations.py samples/*.MP4 --out outputs/cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadwatch.pipeline import observe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="outputs/cache")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for video in args.videos:
        dst = Path(args.out) / (Path(video).stem + ".pkl")
        if dst.exists() and not args.force:
            print(f"{dst} exists, skipping")
            continue
        obs = observe(video)
        obs.save(dst)
        print(f"{video}: {len(obs.tracks)} tracks, {obs.timing}")


if __name__ == "__main__":
    main()
