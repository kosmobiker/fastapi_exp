from datetime import datetime
from io import StringIO
import pickle
import lightgbm as lgb
import joblib
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from src.api.models import TrainedModels
from src.train.trainer import DEFAULT_MODEL
from src.api.schemas import TransactionBase


def get_models(
    db: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 100,
):
    if start_date and end_date:
        models = (
            db.query(TrainedModels)
            .filter(
                and_(
                    TrainedModels.train_date >= start_date,
                    TrainedModels.train_date <= end_date,
                )
            )
            .limit(limit)
            .all()
        )
    elif start_date:
        models = (
            db.query(TrainedModels)
            .filter(TrainedModels.train_date >= start_date)
            .limit(limit)
            .all()
        )
    elif end_date:
        models = (
            db.query(TrainedModels)
            .filter(TrainedModels.train_date <= end_date)
            .limit(limit)
            .all()
        )
    else:
        models = db.query(TrainedModels).limit(limit).all()
    return models


def predict_fraud(
    transaction: TransactionBase, db: Session, model_to_use: str | None = None
):
    """
    Check model_type_to_use:
    - If None, use the default model
    Check type of model in database:
    - If it is logreg, then load the pipeline
    - If it is lightgbm, then load preprocessor and model
    Make prediction
    """
    if not model_to_use:
        model_to_use = DEFAULT_MODEL
    model = (
        db.query(TrainedModels)
        .filter(TrainedModels.model_name == model_to_use)
        .order_by(TrainedModels.roc_auc_test.desc())
        .first()
    )
    if model is None:
        response = {"error": "No model found"}
        return response

    model_id, model_type = model.model_id, model.model_type
    if model_type == "Logistic Regression":
        log_reg_path = f"./models/{model_id}.pkl"
        model = joblib.load(log_reg_path)
        # Make the prediction
        transaction_dict = transaction.model_dump()
        df = pd.DataFrame([transaction_dict])
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        # Prepare the response
        response = {
            "prediction_label": int(prediction[0]),
            "prediction_proba": prediction_proba[0].tolist(),
            "model_used": model_to_use,
        }
        return response

    elif model_type == "LightGBM":
        preprocessor_path = f"./models/{model_id}_preproc.pkl"
        model_path = f"./models/{model_id}_model.txt"
        preprocessor = joblib.load(preprocessor_path)
        model = lgb.Booster(model_file=model_path)
        # Make the prediction
        transaction_dict = transaction.model_dump()
        df = pd.DataFrame([transaction_dict])
        df_transformed = preprocessor.transform(df)
        prediction_proba = model.predict(df_transformed)
        prediction = [1 if prob > 0.5 else 0 for prob in prediction_proba]

        # Prepare the response
        response = {
            "prediction_label": int(prediction[0]),
            "prediction_proba": prediction_proba[0].tolist(),
            "model_used": model_to_use,
        }

        return response
    else:
        response = {"error": f"Model type {model_type} not supported"}
