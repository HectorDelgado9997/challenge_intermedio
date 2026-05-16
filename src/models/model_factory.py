from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config.settings import RANDOM_STATE, VALID_MODEL_NAMES
from src.utils.exceptions import ModelConfigurationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_model_name(model_name: str) -> None:
    """
    Validate that the requested model name is supported.
    """
    if not isinstance(model_name, str):
        raise ModelConfigurationError("model_name must be a string.")

    if model_name not in VALID_MODEL_NAMES:
        raise ModelConfigurationError(
            f"Unsupported model_name '{model_name}'. "
            f"Valid options are: {VALID_MODEL_NAMES}"
        )


def get_default_hyperparameters(model_name: str) -> dict[str, Any]:
    """
    Return default hyperparameters for each supported model.
    """
    validate_model_name(model_name)

    default_hyperparameters = {
        "logistic_regression": {
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
        "knn": {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski",
        },
        "decision_tree": {
            "criterion": "gini",
            "max_depth": 4,
            "random_state": RANDOM_STATE,
        },
    }

    return default_hyperparameters[model_name]


def build_model(
    model_name: str,
    hyperparameters: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline for the selected model.

    Logistic Regression and KNN use StandardScaler because both are sensitive
    to feature scale. Decision Tree does not require scaling.

    Parameters
    ----------
    model_name : str
        Name of the model to build.

    hyperparameters : dict[str, Any] | None
        Optional hyperparameters. If None, default values are used.

    Returns
    -------
    Pipeline
        scikit-learn pipeline containing preprocessing and estimator.
    """
    try:
        validate_model_name(model_name)

        params = get_default_hyperparameters(model_name)

        if hyperparameters is not None:
            if not isinstance(hyperparameters, dict):
                raise ModelConfigurationError(
                    "hyperparameters must be a dictionary or None."
                )
            params.update(hyperparameters)

        if model_name == "logistic_regression":
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(**params)),
                ]
            )

        elif model_name == "knn":
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(**params)),
                ]
            )

        elif model_name == "decision_tree":
            model = Pipeline(
                steps=[
                    ("classifier", DecisionTreeClassifier(**params)),
                ]
            )

        else:
            raise ModelConfigurationError(
                f"Unsupported model_name '{model_name}'."
            )

        logger.info("Model pipeline created for: %s", model_name)
        logger.info("Model hyperparameters: %s", params)

        return model

    except ModelConfigurationError:
        raise

    except Exception as exc:
        raise ModelConfigurationError(
            f"Unexpected error while building model '{model_name}': {exc}"
        ) from exc
