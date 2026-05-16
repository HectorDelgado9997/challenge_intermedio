from pathlib import Path

import pandas as pd

from src.config.settings import DATA_FILE_PATH
from src.utils.exceptions import DataLoadingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_dataset(file_path: Path = DATA_FILE_PATH) -> pd.DataFrame:
    """
    Load the Breast Cancer Wisconsin dataset from a local CSV file.

    This function also removes fully empty unnamed columns commonly generated
    by trailing delimiters in CSV files.

    Parameters
    ----------
    file_path : Path
        Path to the local CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    DataLoadingError
        If the file does not exist or cannot be loaded.
    """
    try:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        if not file_path.exists():
            raise DataLoadingError(f"Dataset file not found: {file_path}")

        logger.info("Loading dataset from %s", file_path)
        df = pd.read_csv(file_path)

        if df.empty:
            raise DataLoadingError("Dataset was loaded but is empty.")

        unnamed_columns = [
            column for column in df.columns
            if column.startswith("Unnamed")
        ]

        fully_empty_unnamed_columns = [
            column for column in unnamed_columns
            if df[column].isna().all()
        ]

        if fully_empty_unnamed_columns:
            logger.info(
                "Dropping fully empty unnamed columns: %s",
                fully_empty_unnamed_columns,
            )
            df = df.drop(columns=fully_empty_unnamed_columns)

        logger.info("Dataset loaded successfully with shape %s", df.shape)
        return df

    except DataLoadingError:
        raise

    except Exception as exc:
        raise DataLoadingError(
            f"Unexpected error while loading dataset: {exc}"
        ) from exc
