from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import RANDOM_STATE, TEST_SIZE
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    columns_to_drop: list[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a validated dataset into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    target_column : str
        Name of the target column.

    columns_to_drop : list[str] | None
        Columns to exclude from the feature matrix.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame.")

        if target_column not in df.columns:
            raise DataValidationError(
                f"Target column '{target_column}' was not found in dataset."
            )

        columns_to_drop = columns_to_drop or []

        unavailable_columns = [
            column for column in columns_to_drop if column not in df.columns
        ]

        if unavailable_columns:
            logger.warning(
                "Columns requested for dropping were not found: %s",
                unavailable_columns,
            )

        valid_columns_to_drop = [
            column for column in columns_to_drop if column in df.columns
        ]

        feature_columns_to_drop = [target_column] + valid_columns_to_drop

        X = df.drop(columns=feature_columns_to_drop)
        y = df[target_column]

        if X.empty:
            raise DataValidationError("Feature matrix is empty after column removal.")

        logger.info("Feature-target split completed.")
        logger.info("Feature matrix shape: %s", X.shape)
        logger.info("Target vector shape: %s", y.shape)

        return X, y

    except DataValidationError:
        raise

    except Exception as exc:
        raise DataValidationError(
            f"Unexpected error during feature-target split: {exc}"
        ) from exc


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a stratified train-test split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target vector.

    test_size : float
        Proportion of the dataset assigned to the test set.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test.
    """
    try:
        if not 0 < test_size < 1:
            raise DataValidationError("test_size must be between 0 and 1.")

        if len(X) != len(y):
            raise DataValidationError(
                "Feature matrix and target vector must have the same length."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info("Train-test split completed.")
        logger.info("X_train shape: %s", X_train.shape)
        logger.info("X_test shape: %s", X_test.shape)
        logger.info("y_train distribution:\n%s", y_train.value_counts())
        logger.info("y_test distribution:\n%s", y_test.value_counts())

        return X_train, X_test, y_train, y_test

    except DataValidationError:
        raise

    except Exception as exc:
        raise DataValidationError(
            f"Unexpected error during train-test split: {exc}"
        ) from exc
