import os

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, recall_score, make_scorer, roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from src.train.transform import (
    ColumnDropper,
    MissingAsNan,
    MissingFlagger,
    MissingValueFiller,
    IncomeRounder,
    Merger,
    CustomOneHotEncoder,
    CustomScaler, CategoricalConverter,
)
from src.train.utils import load_yaml_file
import lightgbm as lgb

PATH_TO_DATA = load_yaml_file("config.yaml")["data_path"]
SEED = load_yaml_file("config.yaml")["seed"]

cols_to_drop = ["source", "device_fraud_count"]

cols_missing_neg1 = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
cols_missing_neg = ["intended_balcon_amount"]

cols_to_flag = [
    "prev_address_months_count",
    "intended_balcon_amount",
    "bank_months_count",
]

fill_value = -1

ohe_cols = [
    "payment_type",
    "employment_status",
    "housing_status",
    "device_os",
    "device_distinct_emails_8w",
]


def get_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"File not found: {path}")


def split_dataframes(
    df: pd.DataFrame, yaml_data: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the data into training, holdout, and test sets.
    """
    seed = yaml_data["seed"]
    train_size = yaml_data["train_size"]
    test_size = yaml_data["test_size"]
    target = yaml_data["target"]
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=train_size, random_state=seed
    )
    X_holdout, X_test, y_holdout, y_test = train_test_split(
        X_test, y_test, stratify=y_test, test_size=test_size, random_state=seed
    )

    return X_train, X_holdout, X_test, y_train, y_holdout, y_test


# Recall @ 5% FPR
def recall_at_5percent_fpr(y_true: np.array, y_pred_proba: np.array) -> np.array:
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    target_fpr = 0.05
    idx = (np.abs(fpr - target_fpr)).argmin()
    return tpr[idx]


def logreg_preprocessor_and_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict | None = {"logisticregression__C": [1]},
) -> tuple[Pipeline, float, float]:
    # Create the preprocessor and model for logistic regression
    logreg_preprocessor = make_pipeline(
        ColumnDropper(cols_to_drop),
        MissingAsNan(cols_missing_neg1, cols_missing_neg),
        MissingFlagger(cols_to_flag),
        MissingValueFiller(fill_value),
        IncomeRounder(),
        Merger(),
        CustomOneHotEncoder(ohe_cols),
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
    params: dict | None = {
                'objective': 'binary',
                'metric': 'binary_error',
                'num_leaves':17,
                'learning_rate': 0.05,
                'verbose': -1,
                'early_stopping_rounds': 250
            },
) -> tuple[Pipeline, lgb.Booster, float, float]:
    # Define the pipeline
    lgbm_preprocessor = make_pipeline(ColumnDropper(cols_to_drop),
                                      MissingAsNan(cols_missing_neg1, cols_missing_neg),
                                      MissingFlagger(cols_to_flag),
                                      MissingValueFiller(fill_value),
                                      IncomeRounder(),
                                      Merger(),
                                      CategoricalConverter(ohe_cols)
                                      )
    # Preprocess the data
    X_train_processed = lgbm_preprocessor.fit_transform(X_train)
    X_test_processed = lgbm_preprocessor.transform(X_test)
    # Train/validation split with stratified sampling
    X_train_lgbm, X_val_lgbm, y_train_lgbm, y_val_lgbm = train_test_split \
        (X_train_processed, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    # Create dataset for LGBM
    lgb_train = lgb.Dataset(X_train_lgbm, y_train_lgbm)
    lgb_val = lgb.Dataset(X_val_lgbm, y_val_lgbm)

    # Train the model
    model = lgb.train(params, lgb_train, num_boost_round=1000,
                      valid_sets=[lgb_train, lgb_val])

    # Make probability predictions on test
    y_pred_test_proba = model.predict \
        (X_test_processed, num_iteration=model.best_iteration)
    # Test performance
    recall_test = recall_at_5percent_fpr(y_test, y_pred_test_proba)
    roc_auc_test = roc_auc_score(y_test, y_pred_test_proba)

    return lgbm_preprocessor, model, recall_test, roc_auc_test
