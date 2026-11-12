"""API tests. They build a tiny stand-in model, so they need no data and no GPU (CI can run them on every push)."""
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

CLASSES = ["Pepper__bell___Bacterial_spot", "Potato___Early_blight", "Potato___Late_blight", "Tomato_Bacterial_spot",
           "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Septoria_leaf_spot", "Tomato_healthy"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from tensorflow import keras
    from tensorflow.keras import layers

    d = tmp_path_factory.mktemp("model")
    inputs = keras.Input((64, 64, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(inputs)       # same layout idea as the real model
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    keras.Model(inputs, out).save(d / "model.keras")
    json.dump(CLASSES, open(d / "class_names.json", "w"))
    os.environ.update(MODEL_PATH=str(d / "model.keras"), CLASS_NAMES_PATH=str(d / "class_names.json"),
                      RUN_INFO_PATH=str(d / "none.json"), PREDICTION_LOG=str(d / "log" / "predictions.jsonl"),
                      CONFIDENCE_THRESHOLD="0.7")
    from fastapi.testclient import TestClient
    from serving.api.main import app
    with TestClient(app) as c:
        c.log_path = d / "log" / "predictions.jsonl"
        yield c


def jpeg_bytes(size=(300, 200)):
    arr = (np.random.rand(size[1], size[0], 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_reports_model_and_class_order(client):
    r = client.get("/health").json()
    assert r["model_loaded"] is True
    assert r["classes"] == CLASSES               # the order must never change silently
    assert r["input_size"] == 64


def test_predict_returns_top3_and_logs(client):
    r = client.post("/predict", files={"file": ("leaf.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["top3"]) == 3
    probs = [t["probability"] for t in body["top3"]]
    assert probs == sorted(probs, reverse=True)
    assert body["confident"] == (probs[0] >= 0.7)
    assert (body["prediction"] is None) == (not body["confident"])
    rows = [json.loads(line) for line in open(client.log_path)]
    assert {"brightness", "contrast", "sharpness", "confidence", "entropy"} <= set(rows[-1])


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_broken_image(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400
