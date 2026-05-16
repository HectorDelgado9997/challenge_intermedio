import pandas as pd

from src.config.settings import TARGET_COLUMN_CANDIDATES
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Detect the target column from known target candidates.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    str
        Detected target column name.

    Raises
    ------
    DataValidationError
        If no valid target column is found.
    """
    for column in TARGET_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise DataValidationError(
        f"No target column found. Expected one of: {TARGET_COLUMN_CANDIDATES}"
    )


def validate_dataset(df: pd.DataFrame) -> str:
    """
    Validate dataset structure and target values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    str
        Name of the detected target column.

    Raises
    ------
    DataValidationError
        If validation fails.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input object must be a pandas DataFrame.")

        if df.empty:
            raise DataValidationError("Dataset is empty.")

        target_column = detect_target_column(df)

        if df[target_column].isna().any():
            raise DataValidationError("Target column contains missing values.")

        target_values = set(df[target_column].dropna().unique())

        valid_string_target = target_values.issubset({"M", "B"})
        valid_numeric_target = target_values.issubset({0, 1})

        if not valid_string_target and not valid_numeric_target:
            raise DataValidationError(
                f"Invalid target values found: {target_values}. "
                "Expected {'M', 'B'} or {0, 1}."
            )

        feature_columns = [col for col in df.columns if col != target_column]

        if not feature_columns:
            raise DataValidationError("Dataset does not contain feature columns.")

        non_numeric_features = [
            col for col in feature_columns
            if not pd.api.types.is_numeric_dtype(df[col])
        ]

        if non_numeric_features:
            raise DataValidationError(
                f"Non-numeric feature columns found: {non_numeric_features}"
            )

        fully_null_columns = [
            col for col in df.columns if df[col].isna().all()
        ]

        if fully_null_columns:
            raise DataValidationError(
                f"Columns with all null values found: {fully_null_columns}"
            )

        logger.info("Dataset validation completed successfully.")
        logger.info("Detected target column: %s", target_column)

        return target_column

    except DataValidationError:
        raise

    except Exception as exc:
        raise DataValidationError(
            f"Unexpected error during dataset validation: {exc}"
        ) from exc


def encode_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Encode target column to binary format.

    M -> 1
    B -> 0

    If the target is already numeric with values {0, 1}, it is preserved.
    """
    df_encoded = df.copy()

    target_values = set(df_encoded[target_column].dropna().unique())

    if target_values.issubset({"M", "B"}):
        df_encoded[target_column] = df_encoded[target_column].map({"M": 1, "B": 0})

    df_encoded[target_column] = df_encoded[target_column].astype(int)

    return df_encoded
