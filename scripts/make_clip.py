"""Cut a short, upload-friendly clip (H.264 8-bit 4:2:0) out of a sample video.

    python scripts/make_clip.py samples/C3902.MP4 --start 80 --duration 30 --height 1080 --out clip.mp4
"""
from __future__ import annotations

import argparse
from fractions import Fraction

import av


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with av.open(args.video) as src, av.open(args.out, "w", options={"movflags": "+faststart"}) as dst:
        s = src.streams.video[0]
        s.thread_type = "AUTO"
        rate = Fraction(s.average_rate).limit_denominator(1001)
        width = round(s.width * args.height / s.height / 2) * 2
        out = dst.add_stream("libx264", rate=rate)
        out.width, out.height, out.pix_fmt = width, args.height, "yuv420p"
        out.options = {"crf": "20", "preset": "veryfast"}
        t0 = s.start_time or 0
        src.seek(int(t0 + args.start / s.time_base), stream=s, backward=True)
        for frame in src.decode(s):
            t = float((frame.pts - t0) * s.time_base)
            if t < args.start:
                continue
            if t >= args.start + args.duration:
                break
            img = frame.to_ndarray(width=width, height=args.height, format="bgr24")
            for packet in out.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                dst.mux(packet)
        for packet in out.encode():
            dst.mux(packet)


if __name__ == "__main__":
    main()
