from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import HealthResponse, ModelTrainingRequest, PredictionRequest, PredictionResponse
from src.config.settings import OUTPUTS_DIR, VALID_MODEL_NAMES
from src.data.load_data import load_dataset
from src.data.validate_data import encode_target, validate_dataset
from src.mlops.mlflow_tracking import log_model_to_mlflow
from src.models.evaluate import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_roc_curve_plot,
)
from src.models.model_factory import build_model
from src.models.train import run_cross_validation, train_model
from src.preprocessing.preprocess import create_train_test_split, split_features_target
from src.utils.exceptions import (
    DataLoadingError,
    DataValidationError,
    MLflowLoggingError,
    ModelConfigurationError,
    TrainingExecutionError,
)
from src.utils.logger import get_logger


router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
def health_check() -> dict[str, str]:
    """
    Verify that the API is running.
    """
    return {
        "status": "ok",
        "project": "Cancer Detection",
    }


@router.get("/dataset/info")
def get_dataset_info(
    attribute: str = Query(..., min_length=1),
    domain: str = Query(..., min_length=1),
) -> dict[str, Any]:
    """
    Return structured information for a selected dataset attribute.
    """
    try:
        df = load_dataset()

        if attribute not in df.columns:
            raise HTTPException(
                status_code=404,
                detail=f"Attribute '{attribute}' was not found in the dataset.",
            )

        series = df[attribute]

        response = {
            "attribute": attribute,
            "domain": domain,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
        }

        if pd.api.types.is_numeric_dtype(series):
            response.update(
                {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                }
            )
        else:
            response.update(
                {
                    "unique_values": series.dropna().unique().tolist(),
                    "value_counts": series.value_counts(dropna=False).to_dict(),
                }
            )

        return response

    except HTTPException:
        raise

    except DataLoadingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:
        logger.error("Unexpected error in /dataset/info: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while retrieving dataset info: {exc}",
        ) from exc


@router.post("/preprocessing/run")
def run_preprocessing() -> dict[str, Any]:
    """
    Run dataset loading, validation, target encoding and train-test split.
    """
    try:
        df = load_dataset()
        target_column = validate_dataset(df)
        df = encode_target(df, target_column)

        X, y = split_features_target(
            df=df,
            target_column=target_column,
            columns_to_drop=["id"],
        )

        X_train, X_test, y_train, y_test = create_train_test_split(X, y)

        categorical_columns = [
            column for column in df.columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]

        value_counts = {
            column: df[column].value_counts(dropna=False).to_dict()
            for column in categorical_columns
        }

        null_values = {
            column: int(df[column].isna().sum())
            for column in df.columns
            if df[column].isna().sum() > 0
        }

        return {
            "target_column": target_column,
            "data_shape": df.shape,
            "data_info": {
                "columns": df.columns.tolist(),
                "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
                "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            },
            "data_describe": df.describe().to_dict(),
            "value_counts": value_counts,
            "null_values": null_values,
            "feature_matrix_shape": X.shape,
            "target_vector_shape": y.shape,
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_distribution": y_train.value_counts().to_dict(),
            "y_test_distribution": y_test.value_counts().to_dict(),
        }

    except (DataLoadingError, DataValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.error("Unexpected error in /preprocessing/run: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while running preprocessing: {exc}",
        ) from exc


@router.post("/models/train")
def train_selected_model(request: ModelTrainingRequest) -> dict[str, Any]:
    """
    Train, evaluate and log one selected model.
    """
    try:
        if request.model_name not in VALID_MODEL_NAMES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported model_name '{request.model_name}'. "
                    f"Valid options are: {VALID_MODEL_NAMES}"
                ),
            )

        df = load_dataset()
        target_column = validate_dataset(df)
        df = encode_target(df, target_column)

        X, y = split_features_target(
            df=df,
            target_column=target_column,
            columns_to_drop=["id"],
        )

        X_train, X_test, y_train, y_test = create_train_test_split(
            X=X,
            y=y,
            test_size=request.test_size,
            random_state=request.random_state,
        )

        model = build_model(
            model_name=request.model_name,
            hyperparameters=request.hyperparameters,
        )

        cv_results = run_cross_validation(
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv_folds=request.cv_folds,
            scoring="f1",
        )

        trained_model = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
        )

        test_metrics = evaluate_model(
            model=trained_model,
            X_test=X_test,
            y_test=y_test,
        )

        confusion_matrix_path = save_confusion_matrix_plot(
            model=trained_model,
            X_test=X_test,
            y_test=y_test,
            model_name=request.model_name,
        )

        roc_curve_path = save_roc_curve_plot(
            model=trained_model,
            X_test=X_test,
            y_test=y_test,
            model_name=request.model_name,
        )

        mlflow_run_id = log_model_to_mlflow(
            model_name=request.model_name,
            model=trained_model,
            X_train=X_train,
            metrics=test_metrics,
            cv_results=cv_results,
            confusion_matrix_path=confusion_matrix_path,
            roc_curve_path=roc_curve_path,
        )

        return {
            "model_name": request.model_name,
            "cv_mean_f1": cv_results["cv_mean"],
            "cv_std_f1": cv_results["cv_std"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "confusion_matrix_path": str(confusion_matrix_path),
            "roc_curve_path": str(roc_curve_path),
            "mlflow_run_id": mlflow_run_id,
        }

    except HTTPException:
        raise

    except (
        DataLoadingError,
        DataValidationError,
        ModelConfigurationError,
        TrainingExecutionError,
        MLflowLoggingError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.error("Unexpected error in /models/train: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while training model: {exc}",
        ) from exc


@router.get("/metrics/summary")
def get_metrics_summary() -> dict[str, Any]:
    """
    Return the generated metrics summary CSV.
    """
    try:
        metrics_path = OUTPUTS_DIR / "metrics_summary.csv"

        if not Path(metrics_path).exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "metrics_summary.csv was not found. "
                    "Run python run_pipeline.py first."
                ),
            )

        metrics_df = pd.read_csv(metrics_path)

        return {
            "metrics_summary_path": str(metrics_path),
            "records": metrics_df.to_dict(orient="records"),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.error("Unexpected error in /metrics/summary: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while retrieving metrics summary: {exc}",
        ) from exc


@router.post("/models/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, Any]:
    """
    Run a prediction using a previously trained model loaded from MLflow.
    """
    try:
        if request.model_name not in VALID_MODEL_NAMES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported model_name '{request.model_name}'. "
                    f"Valid options are: {VALID_MODEL_NAMES}"
                ),
            )

        import mlflow.sklearn
        model = mlflow.sklearn.load_model(f"models:/{request.model_name}/latest")

        input_df = pd.DataFrame([request.features])
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        return {
            "model_name": request.model_name,
            "prediction": prediction,
            "label": "Malignant" if prediction == 1 else "Benign",
            "probability_malignant": round(float(probabilities[1]), 4),
            "probability_benign": round(float(probabilities[0]), 4),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.error("Unexpected error in /models/predict: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while running prediction: {exc}",
        ) from exc
