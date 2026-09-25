"""Assemble (and optionally upload) the Hugging Face Space for the website + live demo.

    python deploy/build_space.py                                  # -> deploy/space_build/
    python deploy/build_space.py --upload <user>/roadsense        # needs HF_TOKEN in the environment

The Space runs app/server.py on CPU (see deploy/space/Dockerfile).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "deploy" / "space_build"
INCLUDE = [
    ("app/server.py", "app/server.py"),
    ("src", "src"),
    ("configs", "configs"),
    ("weights/yolo11s.pt", "weights/yolo11s.pt"),
    ("website", "website"),
    ("deploy/space/Dockerfile", "Dockerfile"),
    ("deploy/space/requirements-space.txt", "requirements-space.txt"),
    ("deploy/space/README.md", "README.md"),
]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "DATA_CONTRACT.md", "node_modules")


def build() -> Path:
    shutil.rmtree(BUILD, ignore_errors=True)
    for src, dst in INCLUDE:
        s, d = ROOT / src, BUILD / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, ignore=SKIP)
        else:
            shutil.copy2(s, d)
    size = sum(f.stat().st_size for f in BUILD.rglob("*") if f.is_file())
    print(f"space assembled in {BUILD} ({size / 1e6:.0f} MB)")
    return BUILD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="REPO_ID", help="e.g. team/roadsense (Space is created if missing)")
    args = ap.parse_args()
    folder = build()
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(args.upload, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=str(folder), repo_id=args.upload, repo_type="space",
                          commit_message="Deploy RoadSense website + demo")
        print(f"https://huggingface.co/spaces/{args.upload}")


if __name__ == "__main__":
    main()
