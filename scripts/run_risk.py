"""Run Part B exactly as the harness does (every frame through RiskEstimator.step) and log
the raw conflict score next to the output, for calibration.

    python scripts/run_risk.py samples/C3902.MP4 --out outputs/risk_C3902.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import solution  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    est = solution.RiskEstimator()
    est.reset({"video_id": Path(args.video).name, "fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})
    model = est._risk
    rows, idx, t0 = [], 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        score = est.step(frame, t)
        if idx % model.stride == 0:
            rows.append([round(t, 3), round(model.last_raw, 3), round(score, 4), model.last_pair])
        idx += 1
    Path(args.out).write_text(json.dumps({"rows": rows, "sec": round(time.perf_counter() - t0, 1)}))
    raw = sorted(r[1] for r in rows)
    print(f"{len(rows)} processed frames in {time.perf_counter() - t0:.0f}s; raw max {raw[-1]:.2f}, "
          f"p99.9 {raw[int(0.999 * (len(raw) - 1))]:.2f}; frames >= 0.5: {sum(r[2] >= 0.5 for r in rows)}")


if __name__ == "__main__":
    main()
