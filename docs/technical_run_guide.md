# Technical Run Guide

## Prerequisites

Make sure you have the following installed before running the project:

| Tool       | Version recommended |
|------------|---------------------|
| Python     | 3.9+                |
| Git        | Any recent version  |
| Git Bash   | (Windows users)     |

## 1. Clone the Repository

```bash
git clone https://github.com/HectorDelgado9997/challenge_intermedio.git
cd challenge_intermedio
```

## 2. Create and Activate Virtual Environment

```bash
# Create the environment
python -m venv .venv

# Activate — Git Bash / Linux / Mac
source .venv/Scripts/activate      # Windows Git Bash
source .venv/bin/activate          # Linux / Mac
```

> You should see `(.venv)` at the start of your terminal prompt.

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies installed:

| Package        | Purpose                        |
|----------------|--------------------------------|
| pandas         | Data loading and manipulation  |
| numpy          | Numerical operations           |
| scikit-learn   | Model training and evaluation  |
| matplotlib     | Plots and visualizations       |
| mlflow         | Experiment tracking            |
| fastapi        | API serving                    |
| uvicorn        | ASGI server for FastAPI        |
| pydantic       | Data validation                |
| pytest         | Unit testing                   |
| joblib         | Model serialization            |
| python-dotenv  | Environment variable loading   |

## 4. Verify the Dataset

Make sure the dataset file exists at:

```text
data/breast_cancer_wisconsin.csv
```

> If the file is missing, download it from the
> [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
> and place it in the `data/` folder.

## 5. Run the Full Pipeline

```bash
python run_pipeline.py
```

This single command executes the complete workflow: Load dataset → Validate → Encode target → Split data
→ For each model (Logistic Regression, KNN, Decision Tree):
→ Cross-validation (5-fold, F1)
→ Train on full training set
→ Evaluate on test set
→ Save plots (confusion matrix, ROC curve)
→ Log run to MLflow
→ Save metrics_summary.csv
## 6. Check the Outputs

After a successful run, the following files are generated:

```text
outputs/
├── metrics_summary.csv
├── confusion_matrix_logistic_regression.png
├── confusion_matrix_knn.png
├── confusion_matrix_decision_tree.png
├── roc_curve_logistic_regression.png
├── roc_curve_knn.png
└── roc_curve_decision_tree.png
```

## 7. Launch the MLflow UI

```bash
mlflow ui
```

Open your browser at `http://127.0.0.1:5000` to explore all tracked runs.

## 8. Run the Tests

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

Test configuration is defined in `pytest.ini` at the project root.

## Common Errors

| Error                              | Likely cause                        | Fix                                  |
|------------------------------------|-------------------------------------|--------------------------------------|
| `ModuleNotFoundError`              | Virtual env not activated           | Run `source .venv/Scripts/activate`  |
| `FileNotFoundError` on CSV        | Dataset missing from `data/`        | Add the CSV file to `data/`          |
| `mlflow.exceptions` on UI         | Port 5000 already in use            | Run `mlflow ui --port 5001`          |
| `pytest: no tests ran`            | Wrong directory                     | Run from project root                |
