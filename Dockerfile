# Driver Monitoring API — FastAPI + ONNX Runtime + MediaPipe (CPU)
#
# Build (from the repo root, with the exported model files in models/):
#   docker build -t driver-monitor-api .
# Run:
#   docker run --rm -p 8000:8000 driver-monitor-api
#   -> http://localhost:8000/docs

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV (pulled in by MediaPipe) needs these system libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes don't reinstall everything
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY configs ./configs
COPY src ./src
COPY app ./app
COPY models/model_meta.json models/distraction_fp32.onnx ./models/

# Face landmark model (~4 MB), downloaded once at build time
RUN python -c "from src.drowsiness.landmarks import ensure_face_model; ensure_face_model()"

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
