import pytest
from sklearn.pipeline import Pipeline

from src.models.model_factory import build_model, get_default_hyperparameters
from src.utils.exceptions import ModelConfigurationError


def test_build_logistic_regression_model():
    model = build_model("logistic_regression")

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


def test_build_knn_model():
    model = build_model("knn")

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


def test_build_decision_tree_model():
    model = build_model("decision_tree")

    assert isinstance(model, Pipeline)
    assert "classifier" in model.named_steps


def test_invalid_model_name_raises_error():
    with pytest.raises(ModelConfigurationError):
        build_model("random_forest")


def test_get_default_hyperparameters():
    params = get_default_hyperparameters("logistic_regression")

    assert "C" in params
    assert "solver" in params
    assert "max_iter" in params
