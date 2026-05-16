from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

DATA_FILE_PATH = DATA_DIR / "breast_cancer_wisconsin.csv"

TARGET_COLUMN_CANDIDATES = ["diagnosis", "target"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Cancer Detection"

VALID_MODEL_NAMES = [
    "logistic_regression",
    "knn",
    "decision_tree",
]