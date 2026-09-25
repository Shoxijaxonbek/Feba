"""EDA of the sample videos for the website (website/data/eda/).

Needs the cached observations (scripts/cache_observations.py). Adds one cheap
keyframe pass per video for container metadata, lighting and a median
"empty road" background used under every overlay.

    python scripts/export_eda.py samples/*.MP4 --cache outputs/cache --out website/data/eda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadwatch.detector import NAMES  # noqa: E402
from roadwatch.flow import CELL, FlowField  # noqa: E402
from roadwatch.perception import Observation  # noqa: E402
from roadwatch.report import counts_per_second, signal_changes  # noqa: E402
from roadwatch.rules.common import STILL_SPEED, build_context  # noqa: E402
from roadwatch.scene import Scene  # noqa: E402
from roadwatch.video import SCENE_H, SCENE_W  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_scene import render as render_scene  # noqa: E402


def keyframe_pass(path: Path) -> dict:
    """Container metadata, mean luma every keyframe, and a median background frame."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        cc = s.codec_context
        meta = {"codec": f"{cc.name} {cc.profile or ''}".strip(), "pix_fmt": str(cc.pix_fmt),
                "bitrate_mbps": round((c.bit_rate or 0) / 1e6, 1), "size_gb": round(path.stat().st_size / 1e9, 2)}
        s.thread_type = "AUTO"
        s.codec_context.skip_frame = "NONKEY"
        t0 = s.start_time or 0
        ts, luma, stack = [], [], []
        for fr in c.decode(s):
            t = float((fr.pts - t0) * s.time_base)
            small = fr.to_ndarray(width=480, height=270, format="bgr24")
            ts.append(round(t, 1))
            luma.append(round(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()), 1))
            if len(stack) < 120 and int(t * 2) % 5 == 0:
                stack.append(fr.to_ndarray(width=SCENE_W, height=SCENE_H, format="bgr24"))
    background = np.median(np.stack(stack), axis=0).astype(np.uint8)
    return {"meta": meta, "brightness": {"t": ts, "mean": luma}, "background": background}


def heatmap(points: np.ndarray, background: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    dens = np.zeros((SCENE_H, SCENE_W), np.float32)
    if len(points):
        xy = np.clip(points.astype(int), [0, 0], [SCENE_W - 1, SCENE_H - 1])
        np.add.at(dens, (xy[:, 1], xy[:, 0]), 1)
    dens = cv2.GaussianBlur(dens, (0, 0), sigma)
    dens = np.log1p(dens / max(dens.max(), 1e-6) * 1000) / np.log1p(1000)
    colour = cv2.applyColorMap((dens * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(background, 0.45, colour, 0.55, 0)


def flow_image(flow: FlowField, background: np.ndarray) -> np.ndarray:
    img = (background * 0.7).astype(np.uint8)
    for gy in range(flow.coherence.shape[0]):
        for gx in range(flow.coherence.shape[1]):
            if flow.count[gy, gx] < 10:
                continue
            c = np.array([gx * CELL + CELL / 2, gy * CELL + CELL / 2])
            d = flow.direction[gy, gx] * 26
            coh = flow.coherence[gy, gx]
            colour = (80, 230, 80) if coh > 0.85 else (0, 210, 255) if coh > 0.5 else (60, 60, 255)
            cv2.arrowedLine(img, tuple(c.astype(int)), tuple((c + d).astype(int)), colour, 2, cv2.LINE_AA, tipLength=0.4)
    return img


def trajectory_image(observations: list[Observation], background: np.ndarray, per_video: int = 250) -> np.ndarray:
    img = (background * 0.75).astype(np.uint8)
    rng = np.random.default_rng(0)
    for obs in observations:
        tracks = [tr for tr in obs.tracks.values() if tr.duration > 2.0]
        for i in rng.permutation(len(tracks))[:per_video]:
            tr = tracks[i]
            colour = {"vehicle": (0, 215, 255), "person": (255, 80, 255), "two_wheeler": (0, 140, 255), "obstacle": (60, 255, 60)}[tr.group]
            cv2.polylines(img, [tr.smooth_foot().astype(np.int32).reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)
    return img


def cycle_stats(changes: list[list]) -> dict:
    durations: dict[str, list[float]] = {}
    for (t0, s0), (t1, _) in zip(changes, changes[1:]):
        durations.setdefault(s0, []).append(t1 - t0)
    greens = [t for t, s in changes if s == "green"]
    cycle = float(np.median(np.diff(greens))) if len(greens) > 1 else None
    # interior phases only (the first and last are cut by the clip boundaries)
    return {"cycle_s": cycle, **{f"{k}_s": round(float(np.median(v[1:-1] if len(v) > 2 else v)), 1)
                                 for k, v in durations.items() if k != "unknown"}}


def queue_series(obs: Observation, scene: Scene) -> dict:
    ctx = build_context(obs, scene)
    n = int(np.ceil(obs.info.duration))
    vehicles = np.zeros(n, int)
    stationary = np.zeros(n, int)
    for tr in ctx.vehicles.values():
        foot, speed = tr.smooth_foot(), tr.speed()
        inside = scene.in_queue_zone(foot)
        sec = tr.t.astype(int)
        for s in np.unique(sec[inside]):
            k = np.flatnonzero(inside & (sec == s))[0]
            vehicles[min(s, n - 1)] += 1
            stationary[min(s, n - 1)] += int(speed[k] < STILL_SPEED)
    return {"t": list(range(n)), "vehicles": vehicles.tolist(), "stationary": stationary.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--out", default="website/data/eda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene.load()

    videos, observations, counts, timelines, queues, backgrounds = [], [], {}, {}, {}, []
    speeds = []
    for path in map(Path, args.videos):
        obs = Observation.load(Path(args.cache) / f"{path.stem}.pkl")
        kf = keyframe_pass(path)
        backgrounds.append(kf["background"])
        info = obs.info
        videos.append({"id": path.stem, "duration": round(info.duration, 1), "fps": round(info.fps, 3),
                       "width": info.width, "height": info.height, "frames": info.n_frames, **kf["meta"],
                       "brightness": kf["brightness"], "tracks": len(obs.tracks)})
        observations.append(obs)
        counts[path.stem] = counts_per_second(obs)
        timelines[path.stem] = signal_changes(obs)
        queues[path.stem] = queue_series(obs, scene)
        for tr in obs.tracks.values():
            if tr.group == "vehicle" and tr.duration > 1.0:
                s = tr.speed()
                speeds += s[s > 5].tolist()
        print(f"{path.name}: done")

    background = np.median(np.stack(backgrounds), axis=0).astype(np.uint8)
    points = {g: np.concatenate([tr.smooth_foot() for o in observations for tr in o.tracks.values() if tr.group == g]
                                or [np.zeros((0, 2))]) for g in ("vehicle", "person")}
    images = {
        "background": background,
        "vehicle_heatmap": heatmap(points["vehicle"], background),
        "person_heatmap": heatmap(points["person"], background),
        "flow_field": flow_image(FlowField.fit([tr for o in observations for tr in o.tracks.values()]), background),
        "trajectories": trajectory_image(observations, background),
        "scene_layout": render_scene(background, scene),
    }
    for name, img in images.items():
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    hist, bins = np.histogram(np.clip(speeds, 0, 600), bins=40, range=(0, 600))
    all_changes = [c for tl in timelines.values() for c in tl]
    eda = {
        "videos": videos,
        "images": {name: f"data/eda/{name}.jpg" for name in images},
        "counts_over_time": counts,
        "signal_cycle": {**cycle_stats(timelines[videos[0]["id"]] if videos else all_changes), "timeline": timelines},
        "queue": queues,
        "speeds": {"vehicle_px_s": hist.tolist(), "bins": bins.round(1).tolist()},
        "classes": NAMES,
    }
    (out / "eda.json").write_text(json.dumps(eda))
    print(f"wrote {out / 'eda.json'}")


if __name__ == "__main__":
    main()
