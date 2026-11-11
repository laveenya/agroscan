# AgroScan - crop disease detection (MLOps project)

Classifies a leaf photo into 8 disease classes (tomato, potato, bell pepper). Real-world accuracy of the final model is documented in `metrics_results.md`.

## Project layout

| Path | What it is |
|---|---|
| `params.yaml` | every setting that changes results (image size, model, epochs, batch mix) |
| `dvc.yaml` | the pipeline: `unzip` -> `train` -> `evaluate` |
| `src/` | the code of the pipeline stages |
| `data/` | the prepared dataset zip (tracked by DVC, not Git) |
| `models/` | trained model (tracked by DVC) |
| `reports/` | metrics JSON files (tracked by Git) and figures (tracked by DVC) |
| `scripts/mlflow_relocate.py` | lets you open a Colab-made MLflow store on your laptop |

## What each tool does here

* **Git / GitHub** - versions the code, `params.yaml`, `dvc.yaml`, `dvc.lock` and the small metrics files.
* **DVC** - versions the big files (dataset zip, model, figures) and runs the pipeline; `dvc repro` re-runs only what changed.
* **MLflow** - records every training run (settings, learning curves, validation/test metrics, model) and keeps a model registry.

## Run it (Colab GPU)

Use the notebook `AgroScan_Step1_DVC_MLflow_Colab.ipynb`. In short:

```bash
dvc pull            # get the dataset zip from the DVC remote (first time: dvc add the zip, dvc push)
dvc repro           # unzip -> train -> evaluate
dvc metrics show    # table of validation and test metrics
dvc push            # store the model and figures in the DVC remote
```

## Change something and compare

Edit `params.yaml` (for example `model.backbone: efficientnetv2s`), then `dvc repro`. Compare with:

```bash
dvc params diff
dvc metrics diff
```

## View MLflow on your laptop

```bash
pip install mlflow
python scripts/mlflow_relocate.py /content/agroscan/mlartifacts "$(pwd)/mlartifacts"
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 (Experiments tab = runs, Models tab = registry).

## Rules we follow

* The test sets are used only in the `evaluate` stage; models are compared on validation data.
* Preprocessing and augmentation are inside the model; the API sends raw 0-255 pixels.
* PlantWild is licensed CC BY-NC-ND 4.0 (non-commercial): never commit or share its images.
