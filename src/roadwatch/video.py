"""Fast frame sampling from long, high-bitrate camera files.

The sample clips are 3840x2160 H.264 High 4:2:2 10-bit at ~150 Mbit/s. Full
BGR conversion of every frame (what cv2.VideoCapture.read does) runs at ~30 fps
on a laptop CPU, so Part A would eat most of the time budget just decoding.
Instead we decode with PyAV and

* ask the decoder to skip non-reference frames (`skip_frame="NONREF"`): with
  the camera's I-B-B-P GOP this yields every third frame (~10 fps) at ~5x
  realtime, and falls back to decoding everything when a file has no B-frames;
* convert only the frames we keep, straight to the working resolution.

All downstream code works in a canonical 1920x1080 "scene pixel" space, so the
scene layout in configs/scene.json is independent of the input resolution.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

SCENE_W, SCENE_H = 1920, 1080


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    duration: float
    width: int
    height: int


def probe(path: str) -> VideoInfo:
    """Read container metadata (same fps/frame count the official harness sees)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(path, float(fps), n, n / float(fps) if fps else 0.0, w, h)


def _decoded(container, stream, max_bad: int = 200):
    """Decode frame by frame, skipping corrupt packets instead of aborting the whole video."""
    import av

    bad = 0
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            bad += 1
            if bad > max_bad:
                return
            continue
        bad = 0
        yield from frames


def _iter_pyav(path: str, target_fps: float, size: tuple[int, int]) -> Iterator[tuple[float, np.ndarray]]:
    import av
    import av.logging

    av.logging.set_level(av.logging.FATAL)  # corrupt packets are handled in _decoded
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        src_fps = float(stream.average_rate or 25)
        if src_fps > 1.5 * target_fps:
            stream.codec_context.skip_frame = "NONREF"
        tb = float(stream.time_base)
        t0 = stream.start_time
        next_t = 0.0
        step = 1.0 / target_fps
        for frame in _decoded(container, stream):
            if frame.pts is None:
                continue
            if t0 is None:
                t0 = frame.pts
            t = (frame.pts - t0) * tb
            if t + 0.5 / src_fps < next_t:
                continue
            next_t = max(next_t + step, t + 0.5 * step)
            yield t, frame.to_ndarray(width=size[0], height=size[1], format="bgr24")


def _iter_cv2(path: str, target_fps: float, size: tuple[int, int]) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / target_fps))
    idx = 0
    while cap.grab():
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            yield idx / fps, cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        idx += 1
    cap.release()


def iter_frames(path: str, target_fps: float = 10.0, size: tuple[int, int] = (SCENE_W, SCENE_H),
                prefetch: int = 16) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_sec, BGR frame at `size`) at roughly `target_fps`.

    Decoding runs in a background thread so it overlaps with GPU inference; it
    stops as soon as the consumer stops iterating.
    """
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    done = object()
    stop = threading.Event()
    error: list[BaseException] = []

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def worker() -> None:
        try:
            try:
                gen = _iter_pyav(path, target_fps, size)
                first = next(gen)
            except StopIteration:
                return
            except Exception:  # PyAV missing or cannot handle the file
                gen = _iter_cv2(path, target_fps, size)
                first = next(gen, None)
                if first is None:
                    return
            if not put(first):
                return
            for item in gen:
                if not put(item):
                    return
        except BaseException as exc:  # surfaced in the consumer thread
            error.append(exc)
        finally:
            put(done)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is done:
                break
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)
    if error:
        raise error[0]
