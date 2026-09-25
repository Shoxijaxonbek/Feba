"""Grid of timestamped keyframes from a video, for a quick look at a whole clip.

    python scripts/contact_sheet.py samples/C3902.MP4 --every 10 --out outputs/sheet.jpg
"""
from __future__ import annotations

import argparse

import av
import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=10.0, help="seconds between tiles")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=480, help="tile width in px")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tiles, next_t = [], args.start
    th = args.tile * 9 // 16
    with av.open(args.video) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.codec_context.skip_frame = "NONKEY"
        t0 = s.start_time or 0
        for fr in c.decode(s):
            t = float((fr.pts - t0) * s.time_base)
            if t > args.end:
                break
            if t + 1e-3 < next_t:
                continue
            next_t = t + args.every
            img = fr.to_ndarray(width=args.tile, height=th, format="bgr24")
            cv2.putText(img, f"{t:6.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(img, f"{t:6.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            tiles.append(img)
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(tiles)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
