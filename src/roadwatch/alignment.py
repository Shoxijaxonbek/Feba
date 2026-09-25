"""Register each video to the reference view the scene layout was drawn on.

The camera is fixed, but a tripod nudged between recordings would shift every
zone and signal-lamp box by a few pixels, silently breaking the rules. Each
video's first frame is matched to configs/reference.jpg (the median background
of the sample videos) with SIFT features and a RANSAC similarity transform.
Detections are mapped into reference coordinates; lamp boxes are mapped back.
Small or unreliable estimates fall back to the identity.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REFERENCE = Path(__file__).resolve().parents[2] / "configs" / "reference.jpg"
WORK_W = 960              # matching resolution
MIN_INLIERS = 40
MIN_SHIFT_PX = 3.0        # below this the view is considered unchanged (matching noise)
MAX_SHIFT_PX = 250.0      # beyond this the match is not trusted
IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _gray(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Downscaled, contrast-normalised grey image: evens out midday sun vs dusk."""
    g = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def estimate(frame: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """2x3 affine mapping frame (scene px, 1920 wide) -> reference (scene px)."""
    if reference is None:
        if not REFERENCE.exists():
            return IDENTITY.copy()
        reference = cv2.imread(str(REFERENCE))
    s = WORK_W / frame.shape[1]
    size = (WORK_W, round(frame.shape[0] * s))
    a, b = _gray(frame, size), _gray(reference, size)
    # SIFT + Lowe's ratio test: ORB found too few consistent matches between midday and dusk footage
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
        return IDENTITY.copy()
    matches = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(matches) < MIN_INLIERS:
        return IDENTITY.copy()
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    m, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if m is None or inliers is None or inliers.sum() < MIN_INLIERS:
        return IDENTITY.copy()
    m[:, 2] /= s  # translation back to full scene px (rotation/scale are resolution-free)
    corners = np.array([[0, 0], [1920, 0], [0, 1080], [1920, 1080]], dtype=np.float64)
    shift = np.abs(apply(m, corners) - corners).max()
    if shift < MIN_SHIFT_PX or shift > MAX_SHIFT_PX:
        return IDENTITY.copy()
    return m


def apply(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points).astype(np.float64)
    return points @ m[:, :2].T + m[:, 2]


def invert(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(m)


def is_identity(m: np.ndarray) -> bool:
    return bool(np.allclose(m, IDENTITY))


def map_boxes(m: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Map (n, >=4) x1,y1,x2,y2 boxes; returns a copy with the axis-aligned hull of the mapped corners."""
    if is_identity(m) or len(boxes) == 0:
        return boxes
    out = boxes.copy()
    c1 = apply(m, boxes[:, [0, 1]])
    c2 = apply(m, boxes[:, [2, 3]])
    c3 = apply(m, boxes[:, [0, 3]])
    c4 = apply(m, boxes[:, [2, 1]])
    xs = np.stack([c1[:, 0], c2[:, 0], c3[:, 0], c4[:, 0]], 1)
    ys = np.stack([c1[:, 1], c2[:, 1], c3[:, 1], c4[:, 1]], 1)
    out[:, 0], out[:, 2] = xs.min(1), xs.max(1)
    out[:, 1], out[:, 3] = ys.min(1), ys.max(1)
    return out


def map_lamp_boxes(m: np.ndarray, signals: dict) -> dict:
    """Signal config with lamp boxes (x, y, w, h) mapped by m (reference -> frame)."""
    if is_identity(m):
        return signals
    scale = float(np.sqrt(abs(np.linalg.det(m[:, :2]))))
    out = {}
    for name, cfg in signals.items():
        lamps = {}
        for lamp, (x, y, w, h) in cfg["lamps"].items():
            cx, cy = apply(m, np.array([[x + w / 2, y + h / 2]]))[0]
            lamps[lamp] = [cx - w * scale / 2, cy - h * scale / 2, w * scale, h * scale]
        out[name] = {**cfg, "lamps": lamps}
    return out
