import pandas as pd

from src.preprocessing.preprocess import create_train_test_split, split_features_target


def test_split_features_target_drops_target_and_id():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "diagnosis": [1, 0, 1, 0],
        "radius_mean": [12.1, 10.2, 13.4, 9.8],
        "texture_mean": [15.3, 12.8, 18.1, 11.4],
    })

    X, y = split_features_target(
        df=df,
        target_column="diagnosis",
        columns_to_drop=["id"],
    )

    assert "diagnosis" not in X.columns
    assert "id" not in X.columns
    assert X.shape[1] == 2
    assert y.name == "diagnosis"


def test_create_train_test_split_shapes():
    X = pd.DataFrame({
        "radius_mean": [1, 2, 3, 4, 5, 6, 7, 8],
        "texture_mean": [8, 7, 6, 5, 4, 3, 2, 1],
    })

    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="diagnosis")

    X_train, X_test, y_train, y_test = create_train_test_split(
        X=X,
        y=y,
        test_size=0.25,
        random_state=42,
    )

    assert len(X_train) == 6
    assert len(X_test) == 2
    assert len(y_train) == 6
    assert len(y_test) == 2
