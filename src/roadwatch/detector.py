"""COCO-pretrained YOLO11 restricted to road users.

Pre-processing (colour swap, resize, letterbox) runs on the GPU: doing it with
the ultralytics CPU pipeline on 1080p frames competes with the video decoder
for CPU time and nearly doubled the per-frame cost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# COCO ids -> our coarse classes
COCO_TO_NAME = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
NAME_ID = {n: i for i, n in enumerate(NAMES)}
VEHICLES = ("car", "bus", "truck")
TWO_WHEELERS = ("bicycle", "motorcycle")

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


class Detector:
    """Batched YOLO inference. Returns boxes as (N, 6): x1, y1, x2, y2, conf, class id (index into NAMES)."""

    def __init__(self, weights: str = "yolo11m.pt", imgsz: int = 1280, conf: float = 0.1,
                 iou: float = 0.6, device: str | None = None):
        import torch
        from ultralytics import YOLO

        path = Path(weights)
        if not path.is_absolute():
            path = WEIGHTS_DIR / path
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = YOLO(str(path)).model.fuse(verbose=False).to(self.device, self.dtype).eval()
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self._map = np.full(100, -1, dtype=np.int64)
        for coco_id, name in COCO_TO_NAME.items():
            self._map[coco_id] = NAME_ID[name]

    def _geometry(self, h: int, w: int) -> tuple[float, int, int, int, int]:
        scale = self.imgsz / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        ph, pw = (-nh) % 32, (-nw) % 32
        return scale, nh, nw, ph // 2, pw // 2

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        import torch.nn.functional as F

        torch = self.torch
        h, w = frames[0].shape[:2]
        scale, nh, nw, top, left = self._geometry(h, w)
        ph, pw = (-nh) % 32, (-nw) % 32
        with torch.inference_mode():
            x = torch.from_numpy(np.stack(frames)).to(self.device)
            x = x.permute(0, 3, 1, 2).to(self.dtype)
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            x = x[:, [2, 1, 0]].div_(255)  # BGR -> RGB after the resize: 2x fewer pixels to shuffle
            x = F.pad(x, (left, pw - left, top, ph - top), value=114 / 255)
            y = self.model(x)
            y = y[0] if isinstance(y, (list, tuple)) else y
            preds = self._postprocess(y)
        out = []
        for p in preds:
            p[:, [0, 2]] = ((p[:, [0, 2]] - left) / scale).clip(0, w)
            p[:, [1, 3]] = ((p[:, [1, 3]] - top) / scale).clip(0, h)
            p[:, 5] = self._map[p[:, 5].astype(np.int64)]
            out.append(p.astype(np.float32))
        return out

    def _postprocess(self, y) -> list[np.ndarray]:
        """(B, 4 + 80, anchors) raw head output -> per-image (N, 6) after class-aware NMS.

        Replaces ultralytics.utils.nms.non_max_suppression, which costs as much as
        the forward pass itself on crowded frames (per-image Python loops).
        """
        import torchvision

        torch = self.torch
        keep_cls = torch.tensor(list(COCO_TO_NAME), device=y.device)
        y = y.float().transpose(1, 2)                      # B, anchors, 84
        scores, cls = y[..., 4:][..., keep_cls].max(-1)    # best road-user class per anchor
        cls = keep_cls[cls]
        out = []
        for b in range(y.shape[0]):
            m = scores[b] > self.conf
            if not m.any():
                out.append(np.zeros((0, 6), dtype=np.float32))
                continue
            xywh, s, c = y[b, m, :4], scores[b, m], cls[b, m]
            boxes = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
            k = torchvision.ops.batched_nms(boxes, s, c, self.iou)[:300]
            out.append(torch.cat([boxes[k], s[k, None], c[k, None].float()], 1).cpu().numpy())
        return out
