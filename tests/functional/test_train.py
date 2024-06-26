import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sqlalchemy import MetaData, Table, create_engine, delete, select

from src.api.database import CONNECTION_STRING
from src.db.db_utils import (
    SCHEMA,
    TABLE_LIST,
    crete_database_schemas_tables,
)
from src.train.trainer import (
    lgbm_preprocessor_and_model,
    logreg_preprocessor_and_model,
    split_dataframes,
    train_model,
)


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
    def test_returns_train_holdout_test_dataframes(self):
        # Given
        df = _fake_get_data(1000)

        # When
        X_train, X_test, y_train, y_test = split_dataframes(df)

        # Then
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert len(X_train) + len(X_test) == len(df)

    def test_returns_log_reg_model_and_test_metrics(self):
        # Given
        df = _fake_get_data(1000)

        # When
        X_train, X_test, y_train, y_test = split_dataframes(df)
        model, recall, roc_auc = logreg_preprocessor_and_model(
            X_train, X_test, y_train, y_test
        )

        # Then
        assert isinstance(model, Pipeline)
        assert 0 <= recall <= 1, "Recall should be between 0 and 1."
        assert 0 <= roc_auc <= 1, "ROC AUC should be between 0 and 1."

    def test_returns_lgbm_preprocessor_model_and_test_metrics(self):
        # Given
        df = _fake_get_data()

        # When
        X_train, X_test, y_train, y_test = split_dataframes(df)
        preprocessor, model, recall, roc_auc = lgbm_preprocessor_and_model(
            X_train, X_test, y_train, y_test
        )

        # Then
        assert isinstance(preprocessor, Pipeline)
        assert isinstance(model, lgb.Booster)
        assert 0 <= recall <= 1, "Recall should be between 0 and 1."
        assert 0 <= roc_auc <= 1, "ROC AUC should be between 0 and 1."

    def test_run_logreg_trainer_and_get_results(self):
        # Given
        model_type = "logreg"
        model_name = "foo_bar_0"
        schema_name = SCHEMA
        table_name = "models"
        df = _fake_get_data(10_000)
        X_train, X_test, y_train, y_test = split_dataframes(df)

        # When
        crete_database_schemas_tables(CONNECTION_STRING, SCHEMA, TABLE_LIST)
        _, path_to_delete = train_model(
            model_name=model_name,
            model_type=model_type,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        # Then
        engine = create_engine(CONNECTION_STRING)
        conn = engine.connect()
        with conn.begin():
            # Define the table
            metadata = MetaData()
            table = Table(
                table_name, metadata, autoload_with=engine, schema=schema_name
            )

            # Create a Select object
            stmt = select(table).where(table.c.model_name == model_name)

            # Execute the statement and fetch one row
            result = conn.execute(stmt).fetchone()
            # Assert that the inserted data is correct
            assert result[2] == model_name
            assert len(result) == 10

            # Delete the inserted row
            delete_stmt = delete(table).where(table.c.model_name == model_name)
            conn.execute(delete_stmt)

        if os.path.isfile(path_to_delete):
            os.remove(path_to_delete)

    def test_run_lightgbm_trainer_and_get_results(self):
        # Given
        model_type = "lightgbm"
        model_name = "foo_bar_1"
        schema_name = SCHEMA
        table_name = "models"
        df = _fake_get_data(10_000)
        X_train, X_test, y_train, y_test = split_dataframes(df)

        # When
        crete_database_schemas_tables(CONNECTION_STRING, SCHEMA, TABLE_LIST)
        path_to_delete_pr, path_to_delete_model = train_model(
            model_name=model_name,
            model_type=model_type,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        # Then
        engine = create_engine(CONNECTION_STRING)
        conn = engine.connect()
        with conn.begin():
            # Define the table
            metadata = MetaData()
            table = Table(
                table_name, metadata, autoload_with=engine, schema=schema_name
            )

            # Create a Select object
            stmt = select(table).where(table.c.model_name == model_name)

            # Execute the statement and fetch one row
            result = conn.execute(stmt).fetchone()
            # Assert that the inserted data is correct
            assert result[2] == model_name
            assert len(result) == 10

            # Delete the inserted row
            delete_stmt = delete(table).where(table.c.model_name == model_name)
            conn.execute(delete_stmt)

        if os.path.isfile(path_to_delete_pr):
            os.remove(path_to_delete_pr)
        if os.path.isfile(path_to_delete_model):
            os.remove(path_to_delete_model)
