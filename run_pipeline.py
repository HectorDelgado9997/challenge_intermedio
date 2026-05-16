from src.data.load_data import load_dataset
from src.data.validate_data import encode_target, validate_dataset
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    df = load_dataset()
    target_column = validate_dataset(df)
    df = encode_target(df, target_column)

    logger.info("Pipeline validation stage completed.")
    logger.info("Final dataset shape: %s", df.shape)
    logger.info("Target distribution:\n%s", df[target_column].value_counts())


if __name__ == "__main__":
    main()