from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train.train import (
    get_data,
    lgbm_preprocessor_and_model,
    logreg_preprocessor_and_model,
    split_dataframes,
)
from src.train.utils import load_yaml_file


def test_returns_df_if_file_exists():
    # Given
    file_path = "data/Base.csv"
    # When
    df = get_data(file_path)
    # Then
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1_000_000


def test_returns_required_parameters_from_yaml_file():
    # Given
    file_path = "config.yaml"

    # When
    yaml_data = load_yaml_file(file_path)

    # Then
    assert yaml_data is not None, "Failed to load YAML file."

    # Given
    required_parameters = ["seed", "target", "data_path", "train_size", "test_size"]

    # When
    missing_parameters = [
        param for param in required_parameters if param not in yaml_data
    ]
    # Then
    assert (
        not missing_parameters
    ), f"Missing parameters: {', '.join(missing_parameters)}"
    assert isinstance(yaml_data["seed"], int), "Seed should be an integer."
    assert isinstance(
        yaml_data["target"], str
    ), "Target column name should be a string."
    assert (
        isinstance(yaml_data["train_size"], float) and 0 <= yaml_data["train_size"] < 1
    ), "Train size should be a float and between 0 and 1."
    assert (
        isinstance(yaml_data["test_size"], float) and 0 <= yaml_data["test_size"] < 1
    ), "Test size should be a float and between 0 and 1."


def test_returns_train_holdout_test_dataframes():
    # Given
    yaml_path = "config.yaml"
    data = {
        "A": np.random.randint(0, 100, 100),
        "B": np.random.rand(100),
        "C": np.random.choice(["X", "Y", "Z"], 100),
        "fraud_bool": np.random.choice([True, False], 100),
    }
    df = pd.DataFrame(data)

    # When
    X_train, X_holdout, X_test, y_train, y_holdout, y_test = split_dataframes(
        df, load_yaml_file(yaml_path)
    )
    full_df_length = len(df)

    # Then
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_holdout, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_holdout, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) + len(X_holdout) + len(X_test) == full_df_length


def test_returns_log_reg_model_and_test_metrics():
    # Given
    file_path = "data/Base.csv"
    yaml_path = "config.yaml"

    # When
    df = get_data(file_path).sample(1000)
    X_train, X_holdout, X_test, y_train, y_holdout, y_test = split_dataframes(
        df, load_yaml_file(yaml_path)
    )
    model, recall, roc_auc = logreg_preprocessor_and_model(
        X_train, X_test, y_train, y_test
    )

    # Then
    assert isinstance(model, Pipeline)
    assert 0 <= recall <= 1, "Recall should be between 0 and 1."
    assert 0 <= roc_auc <= 1, "ROC AUC should be between 0 and 1."


def test_returns_lgbm_preprocessor_model_and_test_metrics():
    # Given
    file_path = "data/Base.csv"
    yaml_path = "config.yaml"

    # When
    df = get_data(file_path).sample(1000)
    X_train, X_holdout, X_test, y_train, y_holdout, y_test = split_dataframes(
        df, load_yaml_file(yaml_path)
    )
    preprocessor, model, recall, roc_auc = lgbm_preprocessor_and_model(
        X_train, X_test, y_train, y_test
    )

    # Then
    assert isinstance(preprocessor, Pipeline)
    assert isinstance(model, lgb.Booster)
    assert 0 <= recall <= 1, "Recall should be between 0 and 1."
    assert 0 <= roc_auc <= 1, "ROC AUC should be between 0 and 1."
