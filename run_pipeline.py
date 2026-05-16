import pandas as pd

from src.config.settings import OUTPUTS_DIR, VALID_MODEL_NAMES
from src.data.load_data import load_dataset
from src.data.validate_data import encode_target, validate_dataset
from src.models.evaluate import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_roc_curve_plot,
)
from src.models.model_factory import build_model
from src.models.train import run_cross_validation, train_model
from src.preprocessing.preprocess import create_train_test_split, split_features_target
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    df = load_dataset()
    target_column = validate_dataset(df)
    df = encode_target(df, target_column)

    X, y = split_features_target(
        df=df,
        target_column=target_column,
        columns_to_drop=["id"],
    )

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    logger.info("Preprocessing stage completed.")
    logger.info("Final X_train shape: %s", X_train.shape)
    logger.info("Final X_test shape: %s", X_test.shape)
    logger.info("Final y_train shape: %s", y_train.shape)
    logger.info("Final y_test shape: %s", y_test.shape)

    metrics_summary = []

    for model_name in VALID_MODEL_NAMES:
        logger.info("Starting workflow for model: %s", model_name)

        model = build_model(model_name)

        cv_results = run_cross_validation(
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv_folds=5,
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
            model_name=model_name,
        )

        roc_curve_path = save_roc_curve_plot(
            model=trained_model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
        )

        model_summary = {
            "model_name": model_name,
            "cv_mean_f1": cv_results["cv_mean"],
            "cv_std_f1": cv_results["cv_std"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "confusion_matrix_path": str(confusion_matrix_path),
            "roc_curve_path": str(roc_curve_path),
        }

        metrics_summary.append(model_summary)

        logger.info(
            "Completed workflow for model: %s | "
            "cv_mean_f1=%.4f | test_f1=%.4f | test_auc=%.4f",
            model_name,
            cv_results["cv_mean"],
            test_metrics["f1_score"],
            test_metrics["roc_auc"],
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_summary_path = OUTPUTS_DIR / "metrics_summary.csv"
    pd.DataFrame(metrics_summary).to_csv(metrics_summary_path, index=False)

    logger.info("Training and evaluation completed for all models.")
    logger.info("Metrics summary saved to %s", metrics_summary_path)


if __name__ == "__main__":
    main()
