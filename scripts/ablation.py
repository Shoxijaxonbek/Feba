"""Dev-set evaluation and ablations -> website/data/metrics.json.

For each variant (detector size x sampling rate) the sample videos are re-observed
(cached per variant), the rules are applied and the result is scored with the
official evaluate.py against our labels in labels/.

    python scripts/ablation.py samples/C3902.MP4 --out website/data/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
from roadwatch import pipeline  # noqa: E402
from roadwatch.detector import Detector  # noqa: E402
from roadwatch.perception import Observation, observe_video  # noqa: E402
from roadwatch.signals import SignalReader  # noqa: E402

VARIANTS = [  # (name, weights, sampling fps); the first one is the submission
    ("YOLO11m @ 10 fps (submitted)", "yolo11m.pt", 10.0),
    ("YOLO11s @ 10 fps", "yolo11s.pt", 10.0),
    ("YOLO11m @ 5 fps", "yolo11m.pt", 5.0),
    ("YOLO11s @ 5 fps", "yolo11s.pt", 5.0),
]


def load_labels(videos: list[Path]) -> dict:
    gt = {}
    for v in videos:
        path = ROOT / "labels" / f"{v.stem}.json"
        if path.exists():
            gt.update(json.loads(path.read_text()))
    return gt


def run_variant(videos: list[Path], weights: str, fps: float, cache: Path) -> tuple[dict, float, float]:
    detector = None
    preds, seconds, duration = {}, 0.0, 0.0
    for v in videos:
        path = cache / f"{v.stem}__{Path(weights).stem}_{fps:g}fps.pkl"
        if path.exists():
            obs = Observation.load(path)
        else:
            detector = detector or Detector(weights)
            obs = observe_video(str(v), detector, target_fps=fps,
                                observers={"signals": SignalReader(pipeline.get_scene().signals)})
            obs.save(path)
        preds[v.name] = {"events": [e.as_list() for e in pipeline.events_from_observation(obs)]}
        seconds += obs.timing["total_sec"]
        duration += obs.info.duration
    return preds, seconds, duration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache", default="outputs/cache/ablation")
    ap.add_argument("--pred", default="predictions_samples.json", help="official harness output, for timings")
    ap.add_argument("--out", default="website/data/metrics.json")
    args = ap.parse_args()
    videos = [Path(v) for v in args.videos]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    gt = load_labels(videos)
    labelled = [v for v in videos if v.name in gt]

    rows, main_report = [], None
    for name, weights, fps in VARIANTS:
        preds, seconds, duration = run_variant(labelled, weights, fps, cache)
        rep = evaluate.evaluate_part_a({k: gt[k] for k in preds}, preds)
        rows.append({"variant": name, "score_a": round(rep["score_a"], 3),
                     "micro_f1_05": round(rep["micro"]["0.5"]["f1"], 3),
                     "perception_x_realtime": round(seconds / duration, 2)})
        main_report = main_report or rep
        print(rows[-1])

    per_class = {c: {"f1_03": round(v["0.3"]["f1"], 3), "f1_05": round(v["0.5"]["f1"], 3),
                     "f1_07": round(v["0.7"]["f1"], 3), "tp": v["0.5"]["tp"], "fp": v["0.5"]["fp"], "fn": v["0.5"]["fn"]}
                 for c, v in main_report["per_class"].items()}
    timing = []
    if Path(args.pred).exists():
        for vid, log in json.loads(Path(args.pred).read_text()).get("log", {}).items():
            timing.append({"video": vid, "duration": log["duration"], "part_a_sec": log.get("part_a_sec"),
                           "part_b_sec": log.get("part_b_sec"), "budget_sec": log["budget_sec"]})
    metrics = {
        "dev_set": f"our own labels of {', '.join(v.stem for v in labelled)} (labels/); candidates came from our "
                   "rules at loose thresholds, so recall is optimistic",
        "per_class": per_class,
        "score_a": round(main_report["score_a"], 3),
        "score_b": None,
        "ablations": [{"name": "Detector size and sampling rate (Part A, dev labels)", "rows": rows}],
        "timing": timing,
    }
    Path(args.out).write_text(json.dumps(metrics, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
