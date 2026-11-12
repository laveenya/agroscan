"""AgroScan API.   GET /health   POST /predict (multipart field 'file')."""
import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from serving.api.inference import Classifier, image_stats

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.keras")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "models/class_names.json")
RUN_INFO_PATH = os.environ.get("RUN_INFO_PATH", "models/run_info.json")
THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))
LOG_PATH = os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "8")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

state = {"clf": None, "version": "unknown", "error": None}
log_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["clf"] = Classifier(MODEL_PATH, CLASS_NAMES_PATH, THRESHOLD)
        if os.path.exists(RUN_INFO_PATH):                       # written by the DVC train stage
            state["version"] = json.load(open(RUN_INFO_PATH)).get("mlflow_run_id", "unknown")
    except Exception as e:                                      # the API still starts, /health explains what is wrong
        state["error"] = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="AgroScan API", version="1.0", lifespan=lifespan)


def log_prediction(record: dict):
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with log_lock, open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    clf = state["clf"]
    return {"status": "ok" if clf else "model_not_loaded", "model_loaded": clf is not None,
            "classes": clf.class_names if clf else None, "input_size": clf.size if clf else None,
            "confidence_threshold": THRESHOLD, "model_version": state["version"], "error": state["error"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    clf = state["clf"]
    if clf is None:
        raise HTTPException(503, f"Model not loaded: {state['error']}")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported file type {file.content_type!r}. Send a JPEG, PNG, WEBP or BMP photo.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The file could not be read as an image.")

    start = time.perf_counter()
    result = clf.predict(image)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    log_prediction({"time": datetime.now(timezone.utc).isoformat(), "model_version": state["version"],
                    "prediction": result["_top1"], "confidence": round(result["_confidence"], 4),
                    "entropy": result["_entropy"], "confident": result["confident"], "latency_ms": latency_ms,
                    **image_stats(image)})
    return {k: v for k, v in result.items() if not k.startswith("_")} | {"model_version": state["version"], "latency_ms": latency_ms}
