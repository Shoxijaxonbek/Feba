"""Static layout of the (fixed) camera view, hand-annotated once in configs/scene.json.

All coordinates are scene pixels of a 1920x1080 frame (see video.SCENE_W/H).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "configs" / "scene.json"


def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting. points (n, 2), poly (m, 2) -> bool (n,)."""
    points = np.atleast_2d(points)
    x, y = points[:, 0:1], points[:, 1:2]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
    return crosses.sum(axis=1) % 2 == 1


def side_of_line(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed side of `points` w.r.t. the directed line a->b (positive = left in image coords)."""
    points = np.atleast_2d(points)
    return (b[0] - a[0]) * (points[:, 1] - a[1]) - (b[1] - a[1]) * (points[:, 0] - a[0])


@dataclass
class Zone:
    name: str
    poly: np.ndarray

    def contains(self, points: np.ndarray) -> np.ndarray:
        return points_in_polygon(points, self.poly)


class Scene:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        z = lambda d: {k: Zone(k, np.asarray(v, dtype=np.float64)) for k, v in d.items()}  # noqa: E731
        self.carriageway = z(cfg["carriageway"])
        self.islands = z(cfg.get("islands", {}))
        self.crosswalks = z(cfg["crosswalks"])
        self.no_stop_exempt = z(cfg.get("no_stop_exempt", {}))
        self.queue_zones = z(cfg.get("queue_zones", {}))
        self.intersection = z(cfg.get("intersection", {}))
        self.stop_lines = {k: {**v, "line": np.asarray(v["line"], dtype=np.float64)}
                           for k, v in cfg.get("stop_lines", {}).items()}
        self.signals = cfg.get("signals", {})

    @staticmethod
    def load(path: str | Path = DEFAULT_SCENE) -> "Scene":
        return Scene(json.loads(Path(path).read_text()))

    @staticmethod
    def _any(zones: dict[str, Zone], points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        out = np.zeros(len(points), dtype=bool)
        for zone in zones.values():
            out |= zone.contains(points)
        return out

    def on_road(self, points: np.ndarray) -> np.ndarray:
        """Point lies on the carriageway (not on an island / median)."""
        return self._any(self.carriageway, points) & ~self._any(self.islands, points)

    def on_crosswalk(self, points: np.ndarray) -> np.ndarray:
        return self._any(self.crosswalks, points)

    def crosswalk_of(self, points: np.ndarray) -> np.ndarray:
        """Name of the crosswalk each point is on ('' if none)."""
        points = np.atleast_2d(points)
        out = np.full(len(points), "", dtype=object)
        for name, zone in self.crosswalks.items():
            out[zone.contains(points)] = name
        return out

    def in_intersection(self, points: np.ndarray) -> np.ndarray:
        return self._any(self.intersection, points)

    def in_queue_zone(self, points: np.ndarray) -> np.ndarray:
        return self._any(self.queue_zones, points)

    def stop_exempt(self, points: np.ndarray) -> np.ndarray:
        """Bus stops, parking bays... where standing vehicles are normal."""
        return self._any(self.no_stop_exempt, points)
