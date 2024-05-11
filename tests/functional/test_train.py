from datetime import datetime
import pytest
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train.trainer import (
    lgbm_preprocessor_and_model,
    logreg_preprocessor_and_model,
    split_dataframes,
)
from src.train.utils import load_yaml_file, get_data


def _fake_get_data(n: int = 100):
    # Create a fake DataFrame
    data = {
        "fraud_bool": np.random.randint(0, 2, size=n),
        "income": np.random.uniform(20000, 150000, size=n),
        "name_email_similarity": np.random.uniform(0, 1, size=n),
        "prev_address_months_count": np.random.randint(1, 61, size=n),
        "current_address_months_count": np.random.randint(1, 61, size=n),
        "customer_age": np.random.randint(18, 100, size=n),
        "days_since_request": np.random.uniform(0, 365, size=n),
        "intended_balcon_amount": np.random.uniform(100, 10000, size=n),
        "payment_type": np.random.choice(["Credit Card", "Debit Card", "Cash"], size=n),
        "zip_count_4w": np.random.randint(1, 100, size=n),
        "velocity_6h": np.random.uniform(0, 100, size=n),
        "velocity_24h": np.random.uniform(0, 100, size=n),
        "velocity_4w": np.random.uniform(0, 100, size=n),
        "bank_branch_count_8w": np.random.randint(1, 11, size=n),
        "date_of_birth_distinct_emails_4w": np.random.randint(1, 11, size=n),
        "employment_status": np.random.choice(
            ["Employed", "Unemployed", "Self-Employed"], size=n
        ),
        "credit_risk_score": np.random.randint(300, 850, size=n),
        "email_is_free": np.random.randint(0, 2, size=n),
        "housing_status": np.random.choice(["Own", "Rent", "Mortgage"], size=n),
        "phone_home_valid": np.random.randint(0, 2, size=n),
        "phone_mobile_valid": np.random.randint(0, 2, size=n),
        "bank_months_count": np.random.randint(1, 121, size=n),
        "has_other_cards": np.random.randint(0, 2, size=n),
        "proposed_credit_limit": np.random.uniform(1000, 50000, size=n),
        "foreign_request": np.random.randint(0, 2, size=n),
        "source": np.random.choice(["Online", "In-Store", "Phone"], size=n),
        "session_length_in_minutes": np.random.uniform(1, 600, size=n),
        "device_os": np.random.choice(["iOS", "Android", "Windows"], size=n),
        "keep_alive_session": np.random.randint(0, 2, size=n),
        "device_distinct_emails_8w": np.random.randint(1, 11, size=n),
        "device_fraud_count": np.random.randint(0, 11, size=n),
        "month": np.random.randint(1, 13, size=n),
    }

    df = pd.DataFrame(data)
    return df


class TestTrainer:
    def test_returns_required_parameters_from_yaml_file(self):
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
            isinstance(yaml_data["train_size"], float)
            and 0 <= yaml_data["train_size"] < 1
        ), "Train size should be a float and between 0 and 1."
        assert (
            isinstance(yaml_data["test_size"], float)
            and 0 <= yaml_data["test_size"] < 1
        ), "Test size should be a float and between 0 and 1."

    def test_returns_train_holdout_test_dataframes(self):
        # Given
        yaml_path = "config.yaml"
        df = _fake_get_data(1000)

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

    def test_returns_log_reg_model_and_test_metrics(self):
        # Given
        yaml_path = "config.yaml"

        # When
        df = _fake_get_data(1000)
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

    def test_returns_lgbm_preprocessor_model_and_test_metrics(self):
        # Given
        yaml_path = "config.yaml"

        # When
        df = _fake_get_data()
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
