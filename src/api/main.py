from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="Cancer Detection API",
    description=(
        "Local API for Breast Cancer Wisconsin binary classification using "
        "Logistic Regression, KNN and Decision Tree with MLflow tracking."
    ),
    version="1.0.0",
)

app.include_router(router)
