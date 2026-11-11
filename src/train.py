"""DVC stage 2: train the model (two stages) and log everything to MLflow."""
import json
import os
import subprocess
import time

import mlflow
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers

from src.config import flatten, load_params, set_seeds
from src.data import eval_ds, load_all, train_stream
from src.metrics import predict_all, summarize
from src.model import build_model


class MlflowEpochLogger(keras.callbacks.Callback):
    """Sends the numbers of every epoch to MLflow (so you get learning curves in its dashboard)."""

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({f"{self.prefix}_{k}": float(v) for k, v in (logs or {}).items()}, step=epoch)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    params = load_params()
    t, d, m = params["train"], params["data"], params["model"]
    set_seeds(t["seed"])
    if m["mixed_precision"]:
        keras.mixed_precision.set_global_policy("mixed_float16")

    class_names, data = load_all(d["prepared_dir"])
    n = len(class_names)
    real_val = data["pw_val"] + data["pd_val"]
    real_val_ds = eval_ds(real_val, params, n)
    stream = train_stream(data, params, n)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name=m["backbone"]) as run:
        mlflow.log_params(flatten(params))
        mlflow.set_tags({"git_commit": git_commit(), "classes": ",".join(class_names), "stage": "train"})
        mlflow.log_params({f"n_{k}": len(v) for k, v in data.items()})

        model, base = build_model(params, n)
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=t["label_smoothing"])

        def callbacks(prefix):
            return [MlflowEpochLogger(prefix),
                    keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=t["patience"],
                                                  restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)]

        t0 = time.time()
        # stage 1: only the new head is trained
        model.compile(keras.optimizers.Adam(t["lr1"]), loss, metrics=["accuracy"])
        model.fit(stream, steps_per_epoch=t["steps_per_epoch"], epochs=t["stage1_epochs"],
                  validation_data=real_val_ds, callbacks=callbacks("stage1"))
        # stage 2: fine-tune the top part of the backbone (BatchNorm stays frozen)
        base.trainable = True
        count = len(base.layers)
        for layer in base.layers[:int(count * (1 - t["tune_frac"]))]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(keras.optimizers.Adam(t["lr2"]), loss, metrics=["accuracy"])
        model.fit(stream, steps_per_epoch=t["steps_per_epoch"], epochs=t["stage2_epochs"],
                  validation_data=real_val_ds, callbacks=callbacks("stage2"))
        minutes = (time.time() - t0) / 60

        # validation scores (used to compare experiments; the test sets are touched only in the evaluate stage)
        metrics = {"train_minutes": round(minutes, 2), "params_million": round(model.count_params() / 1e6, 2)}
        for name in ("pd_val", "pw_val"):
            probs, y = predict_all(model, eval_ds(data[name], params, n))
            for k, v in summarize(probs, y, n).items():
                if k in ("top1", "top3", "macro_f1"):
                    metrics[f"{name}_{k}"] = round(v, 4)

        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        model.save("models/model.keras")
        json.dump(class_names, open("models/class_names.json", "w"), indent=2)
        json.dump(metrics, open("reports/train_metrics.json", "w"), indent=2)
        json.dump({"mlflow_run_id": run.info.run_id, "backbone": m["backbone"]}, open("models/run_info.json", "w"))

        mlflow.log_metrics(metrics)
        mlflow.log_artifact("models/class_names.json")
        mlflow.keras.log_model(model, name="model")       # the MLflow copy of the model (used for the registry)
        print("train metrics:", metrics)


if __name__ == "__main__":
    main()
