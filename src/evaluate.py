"""DVC stage 3: test the trained model on the held-out TEST sets, log to MLflow, register the model."""
import json
import os

import mlflow
from tensorflow import keras

from src.config import load_params
from src.data import eval_ds, load_all
from src.metrics import predict_all, save_confusion_matrix, selective_table, summarize


def main():
    params = load_params()
    if params["model"]["mixed_precision"]:
        keras.mixed_precision.set_global_policy("mixed_float16")
    class_names, data = load_all(params["data"]["prepared_dir"])
    n = len(class_names)
    model = keras.models.load_model("models/model.keras")
    os.makedirs("reports/figures", exist_ok=True)

    results, tables = {}, {}
    for name in ("pd_test", "pw_test"):
        probs, y = predict_all(model, eval_ds(data[name], params, n, cache=True))
        for k, v in summarize(probs, y, n).items():
            results[f"{name}_{k}"] = round(v, 4) if isinstance(v, float) else v
        save_confusion_matrix(probs, y, class_names, f"reports/figures/confusion_matrix_{name}.png")
        tables[name] = selective_table(probs, y)
        tables[name].to_csv(f"reports/figures/confidence_table_{name}.csv", index=False)
    json.dump(results, open("reports/test_metrics.json", "w"), indent=2)
    print("test metrics:", results)

    # add the test numbers to the SAME MLflow run that trained the model, then register that model version
    info = json.load(open("models/run_info.json"))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_id=info["mlflow_run_id"]):
        mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, float)})
        mlflow.log_artifacts("reports/figures", artifact_path="figures")
        mlflow.log_artifact("reports/test_metrics.json")
    version = mlflow.register_model(f"runs:/{info['mlflow_run_id']}/model", params["mlflow"]["model_name"])
    print("registered model version:", version.version)


if __name__ == "__main__":
    main()
