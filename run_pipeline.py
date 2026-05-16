from src.config.settings import VALID_MODEL_NAMES
from src.data.load_data import load_dataset
from src.data.validate_data import encode_target, validate_dataset
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

    trained_models = {}
    cv_results_by_model = {}

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

        trained_models[model_name] = trained_model
        cv_results_by_model[model_name] = cv_results

        logger.info(
            "Completed workflow for model: %s | cv_mean=%.4f | cv_std=%.4f",
            model_name,
            cv_results["cv_mean"],
            cv_results["cv_std"],
        )

    logger.info("Training stage completed for all models.")
    logger.info("Trained models: %s", list(trained_models.keys()))


if __name__ == "__main__":
    main()
