import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import DMatrix, train

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(data_dir: Path):
    """Load preprocessed training and validation data."""
    logger.info("Loading preprocessed data...")
    
    # Load training data
    x_train = pd.read_csv(data_dir / "train_features.csv")
    y_train = pd.read_csv(data_dir / "train_target.csv").iloc[:, 0]  # Get first column as target
    
    # Load validation data
    x_val = pd.read_csv(data_dir / "val_features.csv")
    y_val = pd.read_csv(data_dir / "val_target.csv").iloc[:, 0]
    
    logger.info(f"Loaded training data with {len(x_train)} samples and {x_train.shape[1]} features")
    return x_train, y_train, x_val, y_val

def train_model(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series):
    """Train XGBoost classification model and evaluate using ROC-AUC."""
    logger.info("Training XGBoost classification model...")
    
    # Convert data to DMatrix format
    train_dmatrix = DMatrix(data=x_train, label=y_train)
    val_dmatrix = DMatrix(data=x_val, label=y_val)
    
    # Update parameters to use all CPU cores
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1  # Use all available CPU cores
    }
    
    # Train model with early stopping
    model = train(
        params=params,
        dtrain=train_dmatrix,
        num_boost_round=1000,
        evals=[(val_dmatrix, "validation")],
        early_stopping_rounds=50,
        verbose_eval=True
    )
    
    # Get validation predictions and calculate ROC-AUC
    y_val_pred = model.predict(val_dmatrix)
    roc_auc = roc_auc_score(y_val, y_val_pred)
    logger.info(f"Validation ROC-AUC: {roc_auc:.4f}")
    
    return model

def save_model(model, model_dir: Path):
    """Save the trained model."""
    # Save the trained model
    trained_model_dir = Path("data/trained_models")
    trained_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(trained_model_dir / "model.json")
    logger.info(f"Model saved to {trained_model_dir / 'model.json'}")

def main():
    # Setup directories
    data_dir = Path("data/processed")
    model_dir = Path("models")
    
    # Load data
    x_train, y_train, x_val, y_val = load_processed_data(data_dir)
    
    # Train model
    model = train_model(x_train, y_train, x_val, y_val)
    
    # Save model
    save_model(model, model_dir)

if __name__ == "__main__":
    main()
