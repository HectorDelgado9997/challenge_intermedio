from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config.settings import FIGURES_DIR
from src.utils.exceptions import TrainingExecutionError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_evaluation_inputs(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Validate inputs required for model evaluation.
    """
    if not isinstance(model, Pipeline):
        raise TrainingExecutionError("model must be a scikit-learn Pipeline.")

    if not isinstance(X_test, pd.DataFrame):
        raise TrainingExecutionError("X_test must be a pandas DataFrame.")

    if not isinstance(y_test, pd.Series):
        raise TrainingExecutionError("y_test must be a pandas Series.")

    if X_test.empty:
        raise TrainingExecutionError("X_test cannot be empty.")

    if y_test.empty:
        raise TrainingExecutionError("y_test cannot be empty.")

    if len(X_test) != len(y_test):
        raise TrainingExecutionError(
            "X_test and y_test must have the same number of rows."
        )


def get_positive_class_scores(
    model: Pipeline,
    X_test: pd.DataFrame,
) -> Any:
    """
    Return scores for the positive class.

    For classifiers with predict_proba, the probability of class 1 is used.
    For classifiers without predict_proba, decision_function is used.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    raise TrainingExecutionError(
        "Model does not support predict_proba or decision_function."
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a trained binary classification model.

    Parameters
    ----------
    model : Pipeline
        Trained model pipeline.

    X_test : pd.DataFrame
        Test feature matrix.

    y_test : pd.Series
        Test target vector.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics.
    """
    try:
        validate_evaluation_inputs(model, X_test, y_test)

        logger.info("Starting model evaluation.")

        y_pred = model.predict(X_test)
        y_score = get_positive_class_scores(model, X_test)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_score)
        matrix = confusion_matrix(y_test, y_pred)

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": matrix.tolist(),
        }

        logger.info("Model evaluation completed.")
        logger.info("Precision: %.4f", metrics["precision"])
        logger.info("Recall: %.4f", metrics["recall"])
        logger.info("F1-score: %.4f", metrics["f1_score"])
        logger.info("ROC-AUC: %.4f", metrics["roc_auc"])
        logger.info("Confusion matrix: %s", metrics["confusion_matrix"])

        return metrics

    except TrainingExecutionError:
        raise

    except Exception as exc:
        raise TrainingExecutionError(
            f"Unexpected error during model evaluation: {exc}"
        ) from exc


def save_confusion_matrix_plot(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Save a confusion matrix plot as a PNG artifact.
    """
    try:
        validate_evaluation_inputs(model, X_test, y_test)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"confusion_matrix_{model_name}.png"

        display = ConfusionMatrixDisplay.from_estimator(
            estimator=model,
            X=X_test,
            y=y_test,
            display_labels=["Benign", "Malignant"],
        )

        display.ax_.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info("Confusion matrix plot saved to %s", output_path)

        return output_path

    except Exception as exc:
        raise TrainingExecutionError(
            f"Unexpected error while saving confusion matrix plot: {exc}"
        ) from exc


def save_roc_curve_plot(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Save a ROC curve plot as a PNG artifact.
    """
    try:
        validate_evaluation_inputs(model, X_test, y_test)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"roc_curve_{model_name}.png"

        display = RocCurveDisplay.from_estimator(
            estimator=model,
            X=X_test,
            y=y_test,
        )

        display.ax_.set_title(f"ROC Curve - {model_name}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info("ROC curve plot saved to %s", output_path)

        return output_path

    except Exception as exc:
        raise TrainingExecutionError(
            f"Unexpected error while saving ROC curve plot: {exc}"
        ) from exc
