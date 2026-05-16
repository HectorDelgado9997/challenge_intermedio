# Dataset Extraction

## Project Context

This project uses the **Breast Cancer Wisconsin Diagnostic** dataset to build a binary classification system that predicts whether a tumor is malignant or benign based on cellular measurements.

The target variable is:
- `M`: Malignant
- `B`: Benign

During preprocessing, the target is encoded as:
- `M` → `1`
- `B` → `0`

## Dataset Location

The dataset is stored locally inside the repository:

```text
data/breast_cancer_wisconsin.csv
```

## Dataset Overview

| Property        | Value                              |
|-----------------|------------------------------------|
| Source          | UCI Machine Learning Repository    |
| Samples         | 569                                |
| Features        | 30 numeric + 1 ID + 1 target       |
| Target classes  | Malignant (M), Benign (B)          |
| Missing values  | None                               |
| File format     | CSV                                |

## Features Description

The dataset contains 30 real-valued features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. For each cell nucleus, the following measurements are captured:

| Group     | Features                                              |
|-----------|-------------------------------------------------------|
| Mean      | radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension |
| SE        | Same 10 features — standard error                    |
| Worst     | Same 10 features — worst (largest) value             |

## Columns Dropped During Preprocessing

The `id` column is dropped before training since it carries no predictive information:

```python
columns_to_drop = ["id"]
```

## Loading Process

The dataset is loaded via `src/data/load_data.py`:

```python
from src.data.load_data import load_dataset
df = load_dataset()
```

## Validation & Encoding

After loading, the dataset is validated and the target column is encoded via `src/data/validate_data.py`:

```python
from src.data.validate_data import validate_dataset, encode_target

target_column = validate_dataset(df)
df = encode_target(df, target_column)
```

## Class Distribution

| Class     | Label | Approximate count |
|-----------|-------|-------------------|
| Benign    | 0     | 357 (62.7%)       |
| Malignant | 1     | 212 (37.3%)       |

> The dataset is moderately imbalanced. F1-score is used as the primary evaluation metric to account for this.
