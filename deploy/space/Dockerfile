# Hugging Face Space (Docker SDK) serving the website and the live demo on CPU.
# Built from the folder assembled by deploy/build_space.py (not the submission's run path).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-space.txt

COPY --chown=user . .

ENV ROADWATCH_WEIGHTS=yolo11s.pt \
    ROADWATCH_IMGSZ=960 \
    ROADWATCH_FPS=5 \
    ROADWATCH_DEVICE=cpu \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    OMP_NUM_THREADS=2

USER user
EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
