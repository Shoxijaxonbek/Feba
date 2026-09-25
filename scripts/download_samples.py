"""Download the organizers' sample videos from Google Drive.

Google Drive refuses whole-file downloads of popular files ("quota exceeded")
but still serves HTTP range requests, so this fetches each file in chunks.
The MP4 header and the trailing `moov` atom are fetched first: a partially
downloaded file is then already decodable up to the last contiguous chunk.
Progress is kept in `<name>.chunks.json`, so the script can be re-run to resume.

    python scripts/download_samples.py --out samples
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
SAMPLES = {  # name: (drive id, size in bytes)
    "C3905.MP4": ("1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd", 2348992759),
    "C3902.MP4": ("10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ", 5838719827),
    "C3896.MP4": ("1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS", 6241380481),
    "C3897.MP4": ("1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_", 5838719827),
}
CHUNK = 8 * 1024 * 1024


def fetch_range(file_id: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(URL.format(id=file_id), headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 206 or "video" not in resp.headers.get("Content-Type", ""):
            raise IOError(f"unexpected response {resp.status} {resp.headers.get('Content-Type')}")
        data = resp.read()
    if len(data) != end - start + 1:
        raise IOError(f"short read {len(data)} of {end - start + 1}")
    return data


def download(name: str, file_id: str, size: int, out: Path) -> None:
    path, state_path = out / name, out / f"{name}.chunks.json"
    n_chunks = (size + CHUNK - 1) // CHUNK
    done = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    if path.exists() and not state_path.exists() and path.stat().st_size == size:
        print(f"{name}: already complete")
        return
    if not path.exists():
        with open(path, "wb") as f:
            f.truncate(size)
    # header, trailing moov, then the payload front to back
    order = [0, n_chunks - 1] + list(range(1, n_chunks - 1))
    t0, fetched = time.time(), 0
    with open(path, "r+b") as f:
        for i in order:
            if i in done:
                continue
            start, end = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
            for attempt in range(1000):
                try:
                    data = fetch_range(file_id, start, end)
                    break
                except (IOError, urllib.error.URLError, TimeoutError) as exc:
                    wait = min(300, 10 * (attempt + 1))
                    print(f"{name}: chunk {i} failed ({exc}); retry in {wait}s", flush=True)
                    time.sleep(wait)
            f.seek(start)
            f.write(data)
            done.add(i)
            fetched += len(data)
            state_path.write_text(json.dumps(sorted(done)))
            rate = fetched / max(1e-6, time.time() - t0) / 1e6
            print(f"{name}: {len(done)}/{n_chunks} chunks, {rate:.2f} MB/s", flush=True)
    state_path.unlink()
    print(f"{name}: complete")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--only", nargs="*", help="subset of file names")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (file_id, size) in SAMPLES.items():
        if args.only and name not in args.only:
            continue
        download(name, file_id, size, out)


if __name__ == "__main__":
    main()
