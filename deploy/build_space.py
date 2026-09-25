"""Assemble (and optionally upload) the website as a static Hugging Face Space.

    python deploy/build_space.py                                    # -> deploy/space_build/
    python deploy/build_space.py --upload Shoxijaxonbek/Feba        # needs `hf auth login` or HF_TOKEN

The live demo's API runs elsewhere (deploy/serve_demo.py) and publishes its address
into the Space's data/config.json; uploads here never overwrite that file once it exists.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "deploy" / "space_build"
CONFIG = "data/config.json"


def build() -> Path:
    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.copytree(ROOT / "website", BUILD, ignore=shutil.ignore_patterns("DATA_CONTRACT.md"))
    shutil.copy2(ROOT / "deploy" / "space" / "README.md", BUILD / "README.md")
    size = sum(f.stat().st_size for f in BUILD.rglob("*") if f.is_file())
    print(f"space assembled in {BUILD} ({size / 1e6:.0f} MB)")
    return BUILD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="REPO_ID", help="e.g. Shoxijaxonbek/Feba (created if missing)")
    args = ap.parse_args()
    folder = build()
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()  # token from HF_TOKEN or the saved login
        api.create_repo(args.upload, repo_type="space", space_sdk="static", exist_ok=True)
        keep = [CONFIG] if api.file_exists(args.upload, CONFIG, repo_type="space") else None
        api.upload_folder(folder_path=str(folder), repo_id=args.upload, repo_type="space",
                          ignore_patterns=keep, commit_message="Deploy the RoadSense website")
        print(f"https://huggingface.co/spaces/{args.upload}")


if __name__ == "__main__":
    main()
