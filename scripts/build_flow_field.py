"""Learn the lane-direction field from cached observations of the sample videos.

    python scripts/build_flow_field.py outputs/cache/*.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadwatch.flow import DEFAULT_FLOW, FlowField  # noqa: E402
from roadwatch.perception import Observation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--out", default=str(DEFAULT_FLOW))
    args = ap.parse_args()
    tracks = []
    for path in args.caches:
        tracks += list(Observation.load(path).tracks.values())
    flow = FlowField.fit(tracks)
    flow.save(args.out)
    usable = ((flow.coherence > 0.85) & (flow.count >= 30)).sum()
    print(f"{len(tracks)} tracks -> {args.out}; {usable} coherent cells")


if __name__ == "__main__":
    main()
