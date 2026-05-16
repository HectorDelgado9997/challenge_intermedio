# MLOps Setup

## Overview

This project uses **MLflow** as the MLOps tool for experiment tracking, model
logging, and artifact management. Every training run is automatically registered
with its parameters, metrics, and output artifacts.

## MLflow Integration

The tracking logic lives in `src/mlops/mlflow_tracking.py` and is called
at the end of each model's training cycle:

```python
from src.mlops.mlflow_tracking import log_model_to_mlflow

mlflow_run_id = log_model_to_mlflow(
    model_name=model_name,
    model=trained_model,
    X_train=X_train,
    metrics=test_metrics,
    cv_results=cv_results,
    confusion_matrix_path=confusion_matrix_path,
    roc_curve_path=roc_curve_path,
)
```

## What Gets Logged Per Run

| Category    | Details                                                         |
|-------------|-----------------------------------------------------------------|
| Parameters  | Model name, hyperparameters                                     |
| CV Metrics  | `cv_mean_f1`, `cv_std_f1`                                       |
| Test Metrics| `precision`, `recall`, `f1_score`, `roc_auc`                   |
| Artifacts   | Confusion matrix plot, ROC curve plot                           |
| Model       | Serialized model via `mlflow.sklearn`                           |

## Starting the MLflow UI

To visualize all experiments locally, run:

```bash
mlflow ui
```

Then open your browser at: http://127.0.0.1:5000 > MLflow stores all run data in a local `mlruns/` directory created
> automatically at the project root on the first run.

## Directory Structure After Runs

```text
mlruns/
└── 0/
    └── <run_id>/
        ├── artifacts/
        │   ├── confusion_matrix_<model>.png
        │   ├── roc_curve_<model>.png
        │   └── model/
        ├── metrics/
        └── params/
```

## Run Identification

Each model training produces a unique `mlflow_run_id` that is stored
in `outputs/metrics_summary.csv` alongside all other metrics, allowing
you to trace any result back to its exact MLflow run.

| Column            | Description                          |
|-------------------|--------------------------------------|
| `model_name`      | Name of the trained model            |
| `mlflow_run_id`   | Unique ID linking to the MLflow run  |
| `cv_mean_f1`      | Mean F1 from cross-validation        |
| `cv_std_f1`       | Std deviation of CV F1               |
| `test_f1_score`   | F1 on the held-out test set          |
| `test_roc_auc`    | ROC AUC on the held-out test set     |

## Environment Variables

Project configuration is managed with `python-dotenv`. Create a `.env`
file at the project root if needed:

```bash
# .env (example)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

> The `.env` file is listed in `.gitignore` and should never be committed.
