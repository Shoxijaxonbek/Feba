"""Live-demo API on Modal, so no team machine has to stay on.

    python -m modal deploy deploy/modal_app.py

Runs app/server.py unchanged. Modal's free tier needs a payment method for GPUs, so
the demo runs on CPU cores with lighter detector settings (see ENV); set GPU = "T4"
and drop them to run the exact submitted settings. One container serves all
requests (the job queue lives in its memory) and stays warm for 2 minutes after
the last request (pay per use); the first request after that waits about a minute while the
container starts and loads the model. The printed URL goes into
website/data/config.json as `api_base`.
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "/root/proj"
SKIP = ["**/__pycache__", "**/*.pyc"]
GPU = None                    # "T4" with a payment method on the Modal account
CPUS = 8.0
ENV = {"ROADWATCH_DEVICE": "cpu", "ROADWATCH_IMGSZ": "960", "ROADWATCH_FPS": "10",
       "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"} if GPU is None else {}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # ultralytics pulls in the non-headless opencv
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/" + ("cpu" if GPU is None else "cu124"))
    .pip_install("ultralytics==8.4.162", "lap>=0.5.12", "av>=12.0", "opencv-python-headless>=4.8",
                 "numpy>=1.24,<3", "fastapi>=0.110", "python-multipart>=0.0.9")
    .env({"YOLO_CONFIG_DIR": "/tmp/Ultralytics", **ENV})
    .add_local_dir(ROOT / "src", f"{PROJECT}/src", ignore=SKIP)
    .add_local_dir(ROOT / "configs", f"{PROJECT}/configs", ignore=SKIP)
    .add_local_file(ROOT / "weights" / "yolo11m.pt", f"{PROJECT}/weights/yolo11m.pt")
    .add_local_file(ROOT / "app" / "server.py", f"{PROJECT}/app/server.py")
)
app = modal.App("roadsense-demo", image=image)


@app.function(gpu=GPU, cpu=CPUS, memory=8192, max_containers=1, scaledown_window=120, timeout=60 * 60)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, PROJECT)
    from app.server import app as api, startup

    startup()
    return api
