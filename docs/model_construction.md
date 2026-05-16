# Model Construction

## Overview

This project trains and evaluates three binary classification models to predict
whether a tumor is malignant or benign. All models are built using `scikit-learn`
and follow a unified interface through the model factory pattern.

## Models Used

| Model               | Key Characteristics                                      |
|---------------------|----------------------------------------------------------|
| Logistic Regression | Linear, interpretable, fast, good baseline               |
| K-Nearest Neighbors | Instance-based, non-parametric, sensitive to scaling     |
| Decision Tree       | Non-linear, interpretable, prone to overfitting          |

## Model Factory

All models are instantiated through `src/models/model_factory.py`:

```python
from src.models.model_factory import build_model

model = build_model("logistic_regression")
model = build_model("knn")
model = build_model("decision_tree")
```

The factory pattern ensures a consistent interface regardless of the model type,
and all valid names are registered in `src/config/settings.py`:

```python
VALID_MODEL_NAMES = ["logistic_regression", "knn", "decision_tree"]
```

## Training

Each model is trained on the training split via `src/models/train.py`:

```python
from src.models.train import train_model, run_cross_validation

# Cross-validation (5-fold, F1 scoring)
cv_results = run_cross_validation(
    model=model,
    X_train=X_train,
    y_train=y_train,
    cv_folds=5,
    scoring="f1",
)

# Final training on full training set
trained_model = train_model(model=model, X_train=X_train, y_train=y_train)
```

## Cross-Validation Strategy

| Parameter   | Value  |
|-------------|--------|
| Method      | K-Fold |
| Folds       | 5      |
| Metric      | F1     |
| Results     | `cv_mean`, `cv_std` |

## Evaluation

After training, each model is evaluated on the held-out test set
via `src/models/evaluate.py`:

```python
from src.models.evaluate import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_roc_curve_plot,
)

test_metrics = evaluate_model(model=trained_model, X_test=X_test, y_test=y_test)
```

## Metrics Collected

| Metric           | Description                                        |
|------------------|----------------------------------------------------|
| Precision        | Ratio of true positives over predicted positives   |
| Recall           | Ratio of true positives over actual positives      |
| F1 Score         | Harmonic mean of precision and recall              |
| ROC AUC          | Area under the ROC curve                           |
| Confusion Matrix | TP / FP / TN / FN breakdown                        |

## Output Artifacts

For each model, the following files are saved under `outputs/`:

```text
outputs/
├── metrics_summary.csv           ← all models compared
├── confusion_matrix_<model>.png  ← confusion matrix plot
└── roc_curve_<model>.png         ← ROC curve plot
```

## Model Comparison Summary

All results are consolidated into a single CSV at the end of the pipeline:

```python
pd.DataFrame(metrics_summary).to_csv(OUTPUTS_DIR / "metrics_summary.csv", index=False)
```

This file contains one row per model with all CV and test metrics for easy comparison.
