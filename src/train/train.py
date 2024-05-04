import os
from datetime import datetime

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
from src.train.utils import load_yaml_file

PATH_TO_DATA = load_yaml_file("config.yaml")["data_path"]
SEED = load_yaml_file("config.yaml")["seed"]


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
    params: dict = {"logisticregression__C": [1]},
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


def joblib_dump(
    object_to_dump: Pipeline | lgb.Booster, location: str, filename: str
) -> None:
    extension = "pkl" if isinstance(object_to_dump, Pipeline) else "txt"
    joblib.dump(
        object_to_dump,
        f'{location}/{filename}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.{extension}',
    )


if __name__ == "__main__":
    df = get_data(PATH_TO_DATA)
    X_train, X_holdout, X_test, y_train, y_holdout, y_test = split_dataframes(
        df, load_yaml_file("config.yaml")
    )
    # LogReg
    logreg_model, logreg_recall, logreg_roc_auc = logreg_preprocessor_and_model(
        X_train, X_test, y_train, y_test
    )
    param_grid = {"logisticregression__C": [0.01, 0.1, 1, 10]}

    # LightGBM
    params = {
        "objective": "binary",
        "metric": "binary_error",
        "num_leaves": 17,
        "learning_rate": 0.05,
        "verbose": -1,
        "early_stopping_rounds": 250,
    }
    lgbm_preprocessor, lgbm_model, lgbm_recall, lgbm_roc_auc = (
        lgbm_preprocessor_and_model(X_train, X_test, y_train, y_test, params)
    )

    # Dump models
    loc = "./models"
    joblib_dump(logreg_model, loc, "logreg_pipeline")
    joblib_dump(lgbm_preprocessor, loc, "lgbm_preprocessor")
    joblib_dump(lgbm_model, loc, "lgbm_model")

    # Results
    print("\nLogistic Regression model trained successfully!")
    print(f"\nTest Recall @ 5% FPR: {logreg_recall}")
    print(f"\nTest roc_auc_score: {logreg_roc_auc}")

    print("\nLightGBM model trained successfully!")
    print(f"\nTest Recall @ 5% FPR: {lgbm_recall}")
    print(f"\nTest roc_auc_score: {lgbm_roc_auc}")
