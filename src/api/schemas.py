from typing import Any

from pydantic import BaseModel, Field


class DatasetInfoRequest(BaseModel):
    attribute: str = Field(..., min_length=1)
    domain: str = Field(..., min_length=1)


class ModelTrainingRequest(BaseModel):
    model_name: str = Field(..., min_length=1)
    cv_folds: int = Field(default=5, ge=2)
    test_size: float = Field(default=0.20, gt=0, lt=1)
    random_state: int = Field(default=42)
    hyperparameters: dict[str, Any] | None = None


class PredictionRequest(BaseModel):
    model_name: str = Field(..., min_length=1)
    features: dict[str, float] = Field(
        ...,
        description="Feature values keyed by column name, e.g. {'radius_mean': 17.99, ...}"
    )


class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    label: str
    probability_malignant: float
    probability_benign: float


class HealthResponse(BaseModel):
    status: str
    project: str


class MessageResponse(BaseModel):
    message: str
