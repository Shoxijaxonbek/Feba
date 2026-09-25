"""Run the live-demo API on this machine and expose it through a Cloudflare quick tunnel.

    python deploy/serve_demo.py --space Shoxijaxonbek/Feba

Starts app/server.py (uvicorn), starts `cloudflared tunnel --url ...` (free, no account),
waits for its public https://*.trycloudflare.com address, checks /api/health through it
and publishes {"api_base": ...} as data/config.json of the static Space, so the website
calls this machine. Watches both processes and restarts whichever dies; a new tunnel
address is published again. Stop with Ctrl+C. cloudflared is downloaded to deploy/bin/
on first use. Keep the machine awake and online while the demo should be reachable.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = ROOT / "deploy" / "bin"
LOG_DIR = ROOT / "deploy" / "logs"
RELEASES = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
ASSET = {"win32": "cloudflared-windows-amd64.exe", "linux": "cloudflared-linux-amd64"}
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
CHECK_EVERY_S = 30
MAX_FAILED_CHECKS = 4


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def cloudflared() -> Path:
    exe = BIN_DIR / ("cloudflared.exe" if sys.platform == "win32" else "cloudflared")
    if not exe.exists():
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        log(f"downloading {ASSET[sys.platform]} ...")
        urllib.request.urlretrieve(RELEASES + ASSET[sys.platform], exe)
        exe.chmod(0o755)
    return exe


def healthy(base: str, timeout: float = 10.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=timeout) as r:
            return bool(json.load(r).get("ok"))
    except Exception:
        return False


def wait_healthy(base: str, limit_s: float) -> bool:
    deadline = time.time() + limit_s
    while time.time() < deadline:
        if healthy(base):
            return True
        time.sleep(3)
    return False


def start_server(port: int) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(LOG_DIR / "server.log", "a", encoding="utf-8")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1",
                             "--port", str(port)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)


def start_tunnel(port: int) -> tuple[subprocess.Popen, str | None]:
    proc = subprocess.Popen([str(cloudflared()), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace")
    found: list[str] = []
    ready = threading.Event()

    def drain() -> None:  # keep reading so the pipe never blocks; remember the address
        with open(LOG_DIR / "tunnel.log", "a", encoding="utf-8") as out:
            for line in proc.stderr:
                out.write(line)
                out.flush()
                m = TUNNEL_URL.search(line)
                if m and not found:
                    found.append(m.group(0))
                    ready.set()

    threading.Thread(target=drain, daemon=True).start()
    ready.wait(timeout=60)
    return proc, (found[0] if found else None)


def publish(space: str, base: str) -> None:
    from huggingface_hub import HfApi

    body = json.dumps({"api_base": base, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    HfApi().upload_file(path_or_fileobj=body.encode(), path_in_repo="data/config.json", repo_id=space,
                        repo_type="space", commit_message="Demo server address")
    log(f"published {base} to https://huggingface.co/spaces/{space}")


def stop(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="static Space that hosts the website, e.g. Shoxijaxonbek/Feba")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    local = f"http://127.0.0.1:{args.port}"
    server = tunnel = None
    public = None
    failed = 0
    try:
        while True:
            if server is None or server.poll() is not None:
                stop(server)
                log("starting the demo server (loading the model) ...")
                server = start_server(args.port)
                if not wait_healthy(local, 180):
                    log("server did not come up, see deploy/logs/server.log; retrying")
                    continue
                log(f"server ready at {local}")
            if tunnel is None or tunnel.poll() is not None or failed >= MAX_FAILED_CHECKS:
                stop(tunnel)
                tunnel, public = start_tunnel(args.port)
                failed = 0
                if public is None or not wait_healthy(public, 120):
                    log("tunnel not reachable yet, see deploy/logs/tunnel.log; retrying")
                    stop(tunnel)
                    tunnel = None
                    continue
                publish(args.space, public)
                log(f"demo online: {public}")
            time.sleep(CHECK_EVERY_S)
            failed = 0 if healthy(public) else failed + 1
            if failed:
                log(f"public health check failed ({failed}/{MAX_FAILED_CHECKS})")
    except KeyboardInterrupt:
        log("stopping")
    finally:
        stop(tunnel)
        stop(server)


if __name__ == "__main__":
    main()
