"""Live-demo backend: upload a video, run the pipeline, get events + annotated playback back.

    uvicorn app.server:app --host 0.0.0.0 --port 7860

Serves the API under /api (see website/DATA_CONTRACT.md) and, for local use, the
static website at "/". The public site is a static Hugging Face Space that calls this
API through a tunnel (deploy/serve_demo.py), so cross-origin requests are allowed.
Jobs run one at a time in a background thread with the submission settings; on a
machine without a GPU set ROADWATCH_WEIGHTS=yolo11s.pt ROADWATCH_IMGSZ=960 ROADWATCH_FPS=5.
"""
from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roadwatch import pipeline  # noqa: E402
from roadwatch.report import build_result, render_outputs  # noqa: E402
from roadwatch.risk import replay_detections  # noqa: E402

JOBS_DIR = Path(__file__).resolve().parent / "jobs"
MAX_BYTES = 100 * 1024 * 1024   # Cloudflare's per-request limit on the tunnel
MAX_SECONDS = 120.0
JOB_TTL_S = 3 * 3600


@dataclass
class Job:
    id: str
    dir: Path
    status: str = "queued"
    stage: str = "waiting in queue"
    progress: float = 0.0
    error: str | None = None
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"status": self.status, "stage": self.stage, "progress": round(self.progress, 3), "error": self.error}


jobs: dict[str, Job] = {}
work: queue.Queue[Job] = queue.Queue()


def _run(job: Job) -> None:
    video = job.dir / "input.mp4"
    job.status = "running"

    def stage(name: str, lo: float, hi: float):
        job.stage = name
        return lambda p: setattr(job, "progress", lo + (hi - lo) * p)

    t0 = time.perf_counter()
    obs = pipeline.observe(str(video), progress=stage("detecting and tracking road users", 0.0, 0.6),
                           max_seconds=MAX_SECONDS)
    stage("applying event rules", 0.6, 0.65)(0.0)
    events = pipeline.events_from_observation(obs)
    r = replay_detections(obs.times, obs.detections, pipeline.get_scene())
    risk = np.stack([r["t"], r["risk"]], 1).tolist()
    t_analysis = time.perf_counter() - t0
    render_outputs(str(video), "result", obs, events, risk, job.dir / "media",
                   progress=stage("rendering annotated video", 0.65, 1.0))
    result = build_result("result", obs, events, risk, media_prefix=f"api/jobs/{job.id}/media/",
                          timing={"video_sec": round(obs.info.duration, 1), "analysis_sec": round(t_analysis, 1)})
    (job.dir / "result.json").write_text(json.dumps(result))
    job.progress, job.stage, job.status = 1.0, "done", "done"


def _worker() -> None:
    while True:
        job = work.get()
        try:
            _run(job)
        except Exception as exc:  # report to the page instead of dying
            traceback.print_exc()
            job.status, job.stage, job.error = "error", "failed", f"{type(exc).__name__}: {exc}"
        finally:
            (job.dir / "input.mp4").unlink(missing_ok=True)
            _cleanup()


def _cleanup() -> None:
    for jid, job in list(jobs.items()):
        if time.time() - job.created > JOB_TTL_S and job.status in ("done", "error"):
            shutil.rmtree(job.dir, ignore_errors=True)
            jobs.pop(jid, None)


app = FastAPI(title="RoadSense demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    shutil.rmtree(JOBS_DIR, ignore_errors=True)
    JOBS_DIR.mkdir(parents=True)
    threading.Thread(target=_worker, daemon=True).start()
    pipeline.get_detector()  # load weights once, before the first upload


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...)) -> dict:
    if not (video.filename or "").lower().endswith((".mp4", ".mov", ".m4v")):
        raise HTTPException(400, "please upload an .mp4 file")
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, JOBS_DIR / job_id)
    job.dir.mkdir(parents=True)
    size = 0
    with open(job.dir / "input.mp4", "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                shutil.rmtree(job.dir, ignore_errors=True)
                raise HTTPException(413, "file larger than 100 MB")
            f.write(chunk)
    jobs[job.id] = job
    work.put(job)
    job.stage = f"waiting in queue ({work.qsize()} ahead)" if work.qsize() > 1 else "starting"
    return {"job_id": job.id}


def _job(job_id: str) -> Job:
    if job_id not in jobs:
        raise HTTPException(404, "unknown or expired job")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _job(job_id).public()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _job(job_id)
    if job.status != "done":
        raise HTTPException(409, f"job is {job.status}")
    return JSONResponse(json.loads((job.dir / "result.json").read_text()))


@app.get("/api/jobs/{job_id}/media/{path:path}")
def job_media(job_id: str, path: str):
    base = (_job(job_id).dir / "media").resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "queue": work.qsize(), "weights": pipeline.DETECTOR_WEIGHTS, "fps": pipeline.PART_A_FPS}


app.mount("/", StaticFiles(directory=ROOT / "website", html=True), name="site")
