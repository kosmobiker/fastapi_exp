import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

    def load_processed_data(self, save_path: str) -> dict[str, np.ndarray]:
        """
        Load data from the specified path.
        """
        X_train = np.load(f"{save_path}/X_train.npy")
        y_train = np.load(f"{save_path}/y_train.npy")
        X_val = np.load(f"{save_path}/X_val.npy")
        y_val = np.load(f"{save_path}/y_val.npy")
        X_test = np.load(f"{save_path}/X_test.npy")
        y_test = np.load(f"{save_path}/y_test.npy")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

    def train_model(self, data: dict[str, pd.DataFrame]) -> None:
        """
        Train the XGBoost model using the provided data.
        """
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "eta": 0.1,
            "max_depth": 6,
            "seed": 42,
        }
        evals = [(dtrain, "train"), (dval, "val")]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1_000,
            early_stopping_rounds=20,
            evals=evals,
            verbose_eval=10,
        )
        try:
            model.save_model(f"{self.model_path}/xgb_model.json")
        except Exception as e:
            raise IOError(f"Error saving model: {e}")

        dtest = xgb.DMatrix(X_test, label=y_test)
        y_prob = model.predict(dtest)
        y_pred = (y_prob > 0.5).astype(int)
        logger.info("Accuracy: %f", accuracy_score(y_test, y_pred))
        logger.info("Precision: %f", precision_score(y_test, y_pred))
        logger.info("Recall: %f", recall_score(y_test, y_pred))
        logger.info("F1 Score: %f", f1_score(y_test, y_pred))
        logger.info("ROC AUC: %f", roc_auc_score(y_test, y_prob))


if __name__ == "__main__":
    trainer = Trainer(
        model_path="data/model_artifacts", preprocessor_path="data/model_artifacts"
    )
    processed_data = trainer.load_processed_data(save_path="data/model_artifacts")
    logger.info("Processed data loaded successfully 😊")
    trainer.train_model(processed_data)
    logger.info("Model trained and saved successfully 🤖")
