"""
This module contains functions for training and evaluating machine learning models for fraud detection.

The module includes the following functions:
- split_dataframes: Split the data into training, holdout, and test sets.
- recall_at_5percent_fpr: Calculate recall at 5% false positive rate.
- logreg_preprocessor_and_model: Train a logistic regression model using grid search and custom scoring function.
- lgbm_preprocessor_and_model: Train a LightGBM model using custom scoring function.
- joblib_dump: Save an object to disk using joblib.

The module also includes a main block that parses command line arguments, loads the data, trains the specified model, and saves the trained model to disk.

Note: This code assumes the presence of other modules and functions imported from external files.
"""

import argparse
from datetime import datetime
import json
from uuid import uuid4

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline

from src.api.database import CONNECTION_STRING
from src.train.transform import (
    COLS_MISSING_NEG,
    COLS_MISSING_NEG1,
    COLS_TO_DROP,
    COLS_TO_FLAG,
    FILL_VALUE,
    OHE_COLS,
    CategoricalConverter,
    ColumnDropper,
    CustomOneHotEncoder,
    CustomScaler,
    IncomeRounder,
    Merger,
    MissingAsNan,
    MissingFlagger,
    MissingValueFiller,
)
from src.train.utils import get_data
from src.db.db_utils import insert_values_into_table, SCHEMA

PATH_TO_DATA = "data/Base.csv"
SEED = 42
TARGET = "fraud_bool"
DEFAULT_MODEL = "TestLogRegFraudModel"


def split_dataframes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and test sets.
    """
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=0.8, random_state=SEED
    )

    return X_train, X_test, y_train, y_test


# Recall @ 5% FPR
def recall_at_5percent_fpr(y_true: np.array, y_pred_proba: np.array) -> np.array:
    """
    Calculate recall at 5% false positive rate.
    Used in fraud detection as metrics.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    target_fpr = 0.05
    idx = (np.abs(fpr - target_fpr)).argmin()
    return tpr[idx]


def logreg_preprocessor_and_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict | None = None,
) -> tuple[Pipeline, float, float]:
    """
    Train a logistic regression model using
    grid search and custom scoring function.
    """
    # Create the preprocessor and model for logistic regression
    logreg_preprocessor = make_pipeline(
        ColumnDropper(COLS_TO_DROP),
        MissingAsNan(COLS_MISSING_NEG1, COLS_MISSING_NEG),
        MissingFlagger(COLS_TO_FLAG),
        MissingValueFiller(FILL_VALUE),
        IncomeRounder(),
        Merger(),
        CustomOneHotEncoder(OHE_COLS),
        CustomScaler(),
    )
    logreg_model = LogisticRegression(
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    # Create pipeline for logistic regression
    logreg_pipeline = Pipeline(
        [
            ("logreg_preprocessor", logreg_preprocessor),
            ("logisticregression", logreg_model),
        ]
    )
    if params is None:
        params = {"logisticregression__C": [1]}
    # Create StratifiedKFold object
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Create custom scorer using the custom scoring function
    custom_scorer = make_scorer(recall_at_5percent_fpr, response_method="predict")

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        logreg_pipeline, params, cv=stratified_cv, scoring=custom_scorer, n_jobs=2
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    # Make probability predictions on test
    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]

    recall_test = recall_at_5percent_fpr(y_test, y_pred_test_proba)
    roc_auc_test = roc_auc_score(y_test, y_pred_test_proba)

    return best_model, recall_test, roc_auc_test


def lgbm_preprocessor_and_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict | None = None,
) -> tuple[Pipeline, lgb.Booster, float, float]:
    """
    Train a LightGBM model using custom scoring function.
    """
    # Define the pipeline
    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_error",
            "num_leaves": 17,
            "learning_rate": 0.05,
            "verbose": -1,
            "early_stopping_rounds": 250,
        }
    lgbm_preprocessor = make_pipeline(
        ColumnDropper(COLS_TO_DROP),
        MissingAsNan(COLS_MISSING_NEG1, COLS_MISSING_NEG),
        MissingFlagger(COLS_TO_FLAG),
        MissingValueFiller(FILL_VALUE),
        IncomeRounder(),
        Merger(),
        CategoricalConverter(OHE_COLS),
    )

    # Preprocess the data
    X_train_processed = lgbm_preprocessor.fit_transform(X_train)
    X_test_processed = lgbm_preprocessor.transform(X_test)

    # Train/validation split with stratified sampling
    X_train_lgbm, X_val_lgbm, y_train_lgbm, y_val_lgbm = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    # Create dataset for LGBM
    lgb_train = lgb.Dataset(X_train_lgbm, y_train_lgbm)
    lgb_val = lgb.Dataset(X_val_lgbm, y_val_lgbm)

    # Train the model
    model = lgb.train(
        params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_val]
    )

    # Make probability predictions on test
    y_pred_test_proba = model.predict(
        X_test_processed, num_iteration=model.best_iteration
    )
    # Test performance
    recall_test = recall_at_5percent_fpr(y_test, y_pred_test_proba)
    roc_auc_test = roc_auc_score(y_test, y_pred_test_proba)

    return lgbm_preprocessor, model, recall_test, roc_auc_test


def train_model(
    model_type: str,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict | None = None,
):
    if model_type == "logreg":
        log_reg_model_id = uuid4()
        log_reg_path = f"./models/{log_reg_model_id}.pkl"
        logreg_model, logreg_recall, logreg_roc_auc = logreg_preprocessor_and_model(
            X_train, X_test, y_train, y_test, params
        )
        values_to_insert_logreg = {
            "model_id": log_reg_model_id,
            "train_date": datetime.now(),
            "model_name": model_name,
            "model_type": "Logistic Regression",
            "hyperparameters": params,
            "roc_auc_train": None,
            "recall_train": None,
            "roc_auc_test": logreg_roc_auc,
            "recall_test": logreg_recall,
            "model_path": log_reg_path,
        }

        # Insert values into the models table
        insert_values_into_table(
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA,
            table_name="models",
            values=values_to_insert_logreg,
        )
        # Save the model to local disk
        joblib.dump(logreg_model, log_reg_path)

        #  Results
        print("\nLogistic Regression model trained successfully!")
        print(f"\nTest Recall @ 5% FPR: {logreg_recall}")
        print(f"\nTest roc_auc_score: {logreg_roc_auc}")

        return None, log_reg_path

    elif model_type == "lightgbm":
        lgbm_model_id = uuid4()
        lgbm_preprocessor_path = f"./models/{lgbm_model_id}_preproc.pkl"
        lgbm_model_path = f"./models/{lgbm_model_id}_model.txt"

        # Train the LightGBM model
        lgbm_preprocessor, lgbm_model, lgbm_recall, lgbm_roc_auc = (
            lgbm_preprocessor_and_model(X_train, X_test, y_train, y_test, params)
        )
        values_to_insert_lightgbm = {
            "model_id": lgbm_model_id,
            "train_date": datetime.now(),
            "model_name": model_name,
            "model_type": "LightGBM",
            "hyperparameters": params,
            "roc_auc_train": None,
            "recall_train": None,
            "roc_auc_test": lgbm_roc_auc,
            "recall_test": lgbm_recall,
            "model_path": lgbm_model_path,
        }
        # Insert values into the models table
        insert_values_into_table(
            connection_string=CONNECTION_STRING,
            schema_name=SCHEMA,
            table_name="models",
            values=values_to_insert_lightgbm,
        )
        # Save the model to local disk
        joblib.dump(lgbm_preprocessor, lgbm_preprocessor_path)
        lgbm_model.save_model(lgbm_model_path)

        print("\nLightGBM model trained successfully!")
        print(f"\nTest Recall @ 5% FPR: {lgbm_recall}")
        print(f"\nTest roc_auc_score: {lgbm_roc_auc}")

        return lgbm_preprocessor_path, lgbm_model_path
    else:
        print("Please provide a valid model type.")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="logreg",
        choices=["logreg", "lightgbm"],
        help="Type of model to train.",
    )
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL, help="Model ID."
    )
    parser.add_argument(
        "--params",
        type=json.loads,
        default=None,
        help="Parameters for training the model.",
    )
    args = parser.parse_args()
    # Load data
    df = get_data(PATH_TO_DATA)
    X_train, X_test, y_train, y_test = split_dataframes(df)
    # Train model
    train_model(
        args.model_type, args.model_name, X_train, X_test, y_train, y_test, args.params
    )
