"""Lane-direction field learned from the trajectories of the sample videos.

For every 40x40 px cell of the scene we store the mean unit velocity of the
vehicles that moved through it, and how coherent those directions were
(|mean of unit vectors|: 1 = everyone drives the same way). Cells inside the
junction, where traffic turns in several directions, come out incoherent and
are ignored by the wrong-way rule.

Built by scripts/build_flow_field.py and shipped as configs/flow_field.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .tracking import Track
from .video import SCENE_H, SCENE_W

DEFAULT_FLOW = Path(__file__).resolve().parents[2] / "configs" / "flow_field.npz"
CELL = 40
MIN_SPEED = 30.0


class FlowField:
    def __init__(self, direction: np.ndarray, coherence: np.ndarray, count: np.ndarray):
        self.direction, self.coherence, self.count = direction, coherence, count

    @staticmethod
    def fit(tracks: list[Track]) -> "FlowField":
        gh, gw = SCENE_H // CELL + 1, SCENE_W // CELL + 1
        acc = np.zeros((gh, gw, 2))
        cnt = np.zeros((gh, gw))
        for tr in tracks:
            if tr.group != "vehicle":
                continue
            p, v = tr.smooth_foot(), tr.velocity()
            s = np.linalg.norm(v, axis=1)
            ok = s > MIN_SPEED
            gx = np.clip((p[ok, 0] // CELL).astype(int), 0, gw - 1)
            gy = np.clip((p[ok, 1] // CELL).astype(int), 0, gh - 1)
            np.add.at(acc, (gy, gx), v[ok] / s[ok, None])
            np.add.at(cnt, (gy, gx), 1)
        mean = acc / np.maximum(cnt, 1)[..., None]
        coh = np.linalg.norm(mean, axis=-1)
        direction = mean / np.maximum(coh, 1e-6)[..., None]
        return FlowField(direction, coh, cnt)

    def save(self, path: str | Path = DEFAULT_FLOW) -> None:
        np.savez_compressed(path, direction=self.direction, coherence=self.coherence, count=self.count)

    @staticmethod
    def load(path: str | Path = DEFAULT_FLOW) -> "FlowField | None":
        path = Path(path)
        if not path.exists():
            return None
        d = np.load(path)
        return FlowField(d["direction"], d["coherence"], d["count"])

    def lookup(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(direction (n, 2), coherence (n,), count (n,)) at the given scene points."""
        gh, gw = self.coherence.shape
        gx = np.clip((points[:, 0] // CELL).astype(int), 0, gw - 1)
        gy = np.clip((points[:, 1] // CELL).astype(int), 0, gh - 1)
        return self.direction[gy, gx], self.coherence[gy, gx], self.count[gy, gx]
