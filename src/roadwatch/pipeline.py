"""Part A entry point: video -> observation -> rule-based events."""
from __future__ import annotations

from functools import lru_cache

from .detector import Detector
from .flow import FlowField
from .perception import Observation, observe_video
from .rules import detect_all
from .scene import Scene
from .segments import Event
from .signals import SignalReader
from .utils import set_seed

DETECTOR_WEIGHTS = "yolo11m.pt"
PART_A_FPS = 10.0


@lru_cache(maxsize=None)
def get_detector(weights: str = DETECTOR_WEIGHTS) -> Detector:
    set_seed(0)
    return Detector(weights)


@lru_cache(maxsize=None)
def get_scene() -> Scene:
    return Scene.load()


@lru_cache(maxsize=None)
def get_flow() -> FlowField | None:
    return FlowField.load()


def observe(video_path: str, progress=None) -> Observation:
    scene = get_scene()
    return observe_video(video_path, get_detector(), target_fps=PART_A_FPS,
                         observers={"signals": SignalReader(scene.signals)}, progress=progress)


def events_from_observation(obs: Observation) -> list[Event]:
    return detect_all(obs, get_scene(), get_flow())


def detect_events(video_path: str) -> list[list]:
    obs = observe(video_path)
    return [ev.as_list() for ev in events_from_observation(obs)]
