from src.config.settings import VALID_MODEL_NAMES
from src.data.load_data import load_dataset
from src.data.validate_data import encode_target, validate_dataset
from src.models.model_factory import build_model
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

    for model_name in VALID_MODEL_NAMES:
        model = build_model(model_name)
        logger.info("Created model: %s", model_name)
        logger.info("Pipeline structure: %s", model)


if __name__ == "__main__":
    main()
