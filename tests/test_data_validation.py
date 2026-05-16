import pandas as pd
import pytest

from src.data.validate_data import detect_target_column, encode_target, validate_dataset
from src.utils.exceptions import DataValidationError


def test_detect_target_column_diagnosis():
    df = pd.DataFrame({
        "diagnosis": ["M", "B"],
        "radius_mean": [12.1, 10.2],
    })

    target_column = detect_target_column(df)

    assert target_column == "diagnosis"


def test_validate_dataset_valid_input():
    df = pd.DataFrame({
        "diagnosis": ["M", "B"],
        "radius_mean": [12.1, 10.2],
        "texture_mean": [15.3, 12.8],
    })

    target_column = validate_dataset(df)

    assert target_column == "diagnosis"


def test_validate_dataset_missing_target():
    df = pd.DataFrame({
        "radius_mean": [12.1, 10.2],
        "texture_mean": [15.3, 12.8],
    })

    with pytest.raises(DataValidationError):
        validate_dataset(df)


def test_encode_target():
    df = pd.DataFrame({
        "diagnosis": ["M", "B", "B", "M"],
        "radius_mean": [12.1, 10.2, 9.8, 14.5],
    })

    encoded_df = encode_target(df, "diagnosis")

    assert encoded_df["diagnosis"].tolist() == [1, 0, 0, 1]
