"""Replay Part B's causal risk model on cached detections (no video decoding, no GPU).

The cache holds detections at 10 fps; every second frame is fed to the model,
matching the ~5 fps it runs at inside RiskEstimator. Used to calibrate the
raw-score -> probability mapping on the sample videos.

    python scripts/replay_risk.py outputs/cache/C3902.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadwatch.perception import Observation  # noqa: E402
from roadwatch.risk import replay_detections  # noqa: E402
from roadwatch.scene import Scene  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    args = ap.parse_args()
    for path in args.caches:
        obs = Observation.load(path)
        r = replay_detections(obs.times, obs.detections, Scene.load())
        q = np.percentile(r["raw"], [50, 90, 99, 99.9])
        alarms = np.flatnonzero((r["risk"][1:] >= 0.5) & (r["risk"][:-1] < 0.5))
        print(f"{Path(path).stem}: raw p50/p90/p99/p99.9 = {np.round(q, 2)}, alarms {len(alarms)}")
        top = np.argsort(-r["raw"])[:15]
        for i in sorted(top):
            print(f"  t={r['t'][i]:6.1f} raw={r['raw'][i]:6.2f} pair={r['pair'][i]}")


if __name__ == "__main__":
    main()
