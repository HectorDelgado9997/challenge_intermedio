from typing import Any

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.config.settings import CV_FOLDS
from src.utils.exceptions import TrainingExecutionError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_training_inputs(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
) -> None:
    """
    Validate inputs required for model training and cross-validation.
    """
    if not isinstance(model, Pipeline):
        raise TrainingExecutionError("model must be a scikit-learn Pipeline.")

    if not isinstance(X_train, pd.DataFrame):
        raise TrainingExecutionError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise TrainingExecutionError("y_train must be a pandas Series.")

    if X_train.empty:
        raise TrainingExecutionError("X_train cannot be empty.")

    if y_train.empty:
        raise TrainingExecutionError("y_train cannot be empty.")

    if len(X_train) != len(y_train):
        raise TrainingExecutionError(
            "X_train and y_train must have the same number of rows."
        )

    if not isinstance(cv_folds, int):
        raise TrainingExecutionError("cv_folds must be an integer.")

    if cv_folds < 2:
        raise TrainingExecutionError("cv_folds must be at least 2.")

    if cv_folds > len(y_train):
        raise TrainingExecutionError(
            "cv_folds cannot be greater than the number of training samples."
        )


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train a scikit-learn Pipeline.

    Parameters
    ----------
    model : Pipeline
        Model pipeline.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    Pipeline
        Trained model pipeline.
    """
    try:
        validate_training_inputs(
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv_folds=CV_FOLDS,
        )

        logger.info("Starting model training.")
        model.fit(X_train, y_train)
        logger.info("Model training completed.")

        return model

    except TrainingExecutionError:
        raise

    except Exception as exc:
        raise TrainingExecutionError(
            f"Unexpected error during model training: {exc}"
        ) from exc


def run_cross_validation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    scoring: str = "f1",
) -> dict[str, Any]:
    """
    Run k-fold cross-validation.

    Parameters
    ----------
    model : Pipeline
        Model pipeline.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    cv_folds : int
        Number of cross-validation folds.

    scoring : str
        Scikit-learn scoring strategy.

    Returns
    -------
    dict[str, Any]
        Cross-validation scores and summary statistics.
    """
    try:
        validate_training_inputs(
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv_folds=cv_folds,
        )

        if not isinstance(scoring, str):
            raise TrainingExecutionError("scoring must be a string.")

        logger.info(
            "Starting cross-validation with cv_folds=%s and scoring='%s'.",
            cv_folds,
            scoring,
        )

        scores = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv_folds,
            scoring=scoring,
        )

        cv_results = {
            "cv_scores": scores.tolist(),
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "scoring": scoring,
            "cv_folds": cv_folds,
        }

        logger.info("Cross-validation completed.")
        logger.info("CV scores: %s", cv_results["cv_scores"])
        logger.info("CV mean: %.4f", cv_results["cv_mean"])
        logger.info("CV std: %.4f", cv_results["cv_std"])

        return cv_results

    except TrainingExecutionError:
        raise

    except Exception as exc:
        raise TrainingExecutionError(
            f"Unexpected error during cross-validation: {exc}"
        ) from exc
