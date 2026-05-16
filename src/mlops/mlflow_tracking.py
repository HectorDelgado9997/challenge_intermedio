from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline

from src.config.settings import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.utils.exceptions import MLflowLoggingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def configure_mlflow() -> None:
    """
    Configure MLflow to use a local tracking server.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        logger.info("MLflow tracking URI configured: %s", MLFLOW_TRACKING_URI)
        logger.info("MLflow experiment configured: %s", MLFLOW_EXPERIMENT_NAME)

    except Exception as exc:
        raise MLflowLoggingError(
            f"Unexpected error while configuring MLflow: {exc}"
        ) from exc


def validate_mlflow_inputs(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    metrics: dict[str, Any],
    cv_results: dict[str, Any],
    confusion_matrix_path: Path,
    roc_curve_path: Path,
) -> None:
    """
    Validate inputs required for MLflow logging.
    """
    if not isinstance(model_name, str) or not model_name:
        raise MLflowLoggingError("model_name must be a non-empty string.")

    if not isinstance(model, Pipeline):
        raise MLflowLoggingError("model must be a scikit-learn Pipeline.")

    if not isinstance(X_train, pd.DataFrame):
        raise MLflowLoggingError("X_train must be a pandas DataFrame.")

    if X_train.empty:
        raise MLflowLoggingError("X_train cannot be empty.")

    if not isinstance(metrics, dict):
        raise MLflowLoggingError("metrics must be a dictionary.")

    if not isinstance(cv_results, dict):
        raise MLflowLoggingError("cv_results must be a dictionary.")

    required_metrics = [
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    missing_metrics = [
        metric for metric in required_metrics if metric not in metrics
    ]

    if missing_metrics:
        raise MLflowLoggingError(f"Missing required metrics: {missing_metrics}")

    if not Path(confusion_matrix_path).exists():
        raise MLflowLoggingError(
            f"Confusion matrix artifact not found: {confusion_matrix_path}"
        )

    if not Path(roc_curve_path).exists():
        raise MLflowLoggingError(
            f"ROC curve artifact not found: {roc_curve_path}"
        )


def get_model_hyperparameters(model: Pipeline) -> dict[str, Any]:
    """
    Extract classifier hyperparameters from a scikit-learn Pipeline.
    """
    try:
        classifier = model.named_steps["classifier"]
        return classifier.get_params()

    except Exception as exc:
        raise MLflowLoggingError(
            f"Could not extract model hyperparameters: {exc}"
        ) from exc


def log_model_to_mlflow(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    metrics: dict[str, Any],
    cv_results: dict[str, Any],
    confusion_matrix_path: Path,
    roc_curve_path: Path,
) -> str:
    """
    Log a trained model, parameters, metrics and artifacts to MLflow.

    Parameters
    ----------
    model_name : str
        Name of the trained model.

    model : Pipeline
        Trained scikit-learn pipeline.

    X_train : pd.DataFrame
        Training feature matrix used to infer the model signature.

    metrics : dict[str, Any]
        Test evaluation metrics.

    cv_results : dict[str, Any]
        Cross-validation results.

    confusion_matrix_path : Path
        Path to saved confusion matrix plot.

    roc_curve_path : Path
        Path to saved ROC curve plot.

    Returns
    -------
    str
        MLflow run ID.
    """
    try:
        validate_mlflow_inputs(
            model_name=model_name,
            model=model,
            X_train=X_train,
            metrics=metrics,
            cv_results=cv_results,
            confusion_matrix_path=confusion_matrix_path,
            roc_curve_path=roc_curve_path,
        )

        configure_mlflow()

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            logger.info("Started MLflow run for model: %s", model_name)
            logger.info("MLflow run ID: %s", run_id)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("cv_folds", cv_results["cv_folds"])
            mlflow.log_param("cv_scoring", cv_results["scoring"])

            hyperparameters = get_model_hyperparameters(model)

            for param_name, param_value in hyperparameters.items():
                mlflow.log_param(param_name, param_value)

            mlflow.log_metric("cv_mean_f1", cv_results["cv_mean"])
            mlflow.log_metric("cv_std_f1", cv_results["cv_std"])
            mlflow.log_metric("test_precision", metrics["precision"])
            mlflow.log_metric("test_recall", metrics["recall"])
            mlflow.log_metric("test_f1_score", metrics["f1_score"])
            mlflow.log_metric("test_roc_auc", metrics["roc_auc"])

            mlflow.log_artifact(str(confusion_matrix_path))
            mlflow.log_artifact(str(roc_curve_path))

            input_example = X_train.head(5)
            model_prediction = model.predict(input_example)
            signature = infer_signature(input_example, model_prediction)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            logger.info("Model logged successfully to MLflow: %s", model_name)

            return run_id

    except MLflowLoggingError:
        raise

    except Exception as exc:
        raise MLflowLoggingError(
            f"Unexpected error while logging model to MLflow: {exc}"
        ) from exc
