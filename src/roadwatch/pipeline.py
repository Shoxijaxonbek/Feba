"""Part A entry point: video -> observation -> rule-based events.

Defaults are the submission settings (GPU). The CPU live demo overrides them
through environment variables: ROADWATCH_WEIGHTS, ROADWATCH_IMGSZ, ROADWATCH_FPS, ROADWATCH_DEVICE.
"""
from __future__ import annotations

import os
from functools import lru_cache

from .detector import Detector
from .flow import FlowField
from .perception import Observation, observe_video
from .rules import detect_all
from .scene import Scene
from .segments import Event
from .signals import SignalReader
from .utils import set_seed

DETECTOR_WEIGHTS = os.environ.get("ROADWATCH_WEIGHTS", "yolo11m.pt")
DETECTOR_IMGSZ = int(os.environ.get("ROADWATCH_IMGSZ", "1280"))
PART_A_FPS = float(os.environ.get("ROADWATCH_FPS", "10"))
DEVICE = os.environ.get("ROADWATCH_DEVICE")  # None = cuda if available


@lru_cache(maxsize=None)
def get_detector() -> Detector:
    set_seed(0)
    return Detector(DETECTOR_WEIGHTS, imgsz=DETECTOR_IMGSZ, device=DEVICE)


@lru_cache(maxsize=None)
def get_scene() -> Scene:
    return Scene.load()


@lru_cache(maxsize=None)
def get_flow() -> FlowField | None:
    return FlowField.load()


def observe(video_path: str, progress=None, max_seconds: float | None = None) -> Observation:
    scene = get_scene()
    return observe_video(video_path, get_detector(), target_fps=PART_A_FPS,
                         observers={"signals": SignalReader(scene.signals)}, progress=progress,
                         max_seconds=max_seconds)


def events_from_observation(obs: Observation) -> list[Event]:
    return detect_all(obs, get_scene(), get_flow())


def detect_events(video_path: str) -> list[list]:
    obs = observe(video_path)
    return [ev.as_list() for ev in events_from_observation(obs)]
