from datetime import datetime
from io import StringIO
import pickle
from uuid import uuid4
import lightgbm as lgb
import joblib
import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from src.api.models import Prediction, TrainedModels
from src.train.trainer import DEFAULT_MODEL
from src.api.schemas import TransactionBase


def get_models(
    db: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 100,
):
    """
    Retrieve trained models from the database based on the specified criteria.

    Args:
        db (Session): The database session.
        start_date (datetime | None, optional): The start date for filtering the models. Defaults to None.
        end_date (datetime | None, optional): The end date for filtering the models. Defaults to None.
        limit (int, optional): The maximum number of models to retrieve. Defaults to 100.

    Returns:
        List[TrainedModels]: A list of trained models that match the specified criteria.
    """
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
    transactions: TransactionBase, db: Session, model_to_use: str | None = None
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
        return {"error": "No model found"}

    model_id, model_type = model.model_id, model.model_type
    if model_type == "Logistic Regression":
        log_reg_path = f"./models/{model_id}.pkl"
        model = joblib.load(log_reg_path)
        # Make the prediction
        transactions_dict = [t.model_dump() for t in transactions]
        df = pd.DataFrame(transactions_dict)
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        # Prepare the response
        response = {
            "prediction_label": prediction.tolist(),
            "prediction_proba": prediction_proba[:, 0].tolist(),
            "model_used": model_to_use,
        }
        prediction_record = Prediction(
            id=uuid4(),
            model_name=model_to_use,
            ts=datetime.now(),
            input_data=transactions_dict,
            prediction_label=prediction.tolist(),
            prediction_proba=prediction_proba[:, 0].tolist(),
        )

    elif model_type == "LightGBM":
        preprocessor_path = f"./models/{model_id}_preproc.pkl"
        model_path = f"./models/{model_id}_model.txt"
        preprocessor = joblib.load(preprocessor_path)
        model = lgb.Booster(model_file=model_path)
        # Make the prediction
        transactions_dict = [t.model_dump() for t in transactions]
        df = pd.DataFrame(transactions_dict)
        df_transformed = preprocessor.transform(df)
        prediction_proba = model.predict(df_transformed)
        prediction = [1 if prob > 0.5 else 0 for prob in prediction_proba]

        # Prepare the response
        response = {
            "prediction_label": prediction,
            "prediction_proba": prediction_proba.tolist(),
            "model_used": model_to_use,
        }
        prediction_record = Prediction(
            id=uuid4(),
            model_name=model_to_use,
            ts=datetime.now(),
            input_data=transactions_dict,
            prediction_label=prediction,
            prediction_proba=prediction_proba.tolist(),
        )
    else:
        response = {"error": f"Model type {model_type} not supported"}

    db.add(prediction_record)
    db.commit()
    return response


def get_history(
    db: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    model_to_use: str | None = None,
    limit: int = 100,
):
    """
    Retrieve prediction history from the database based on the specified criteria.

    Args:
        db (Session): The database session.
        start_date (datetime | None, optional): The start date for filtering the history. Defaults to None.
        end_date (datetime | None, optional): The end date for filtering the history. Defaults to None.
        model_to_use (str | None, optional): The model name for filtering the history. Defaults to None.
        limit (int, optional): The maximum number of records to retrieve. Defaults to 100.

    Returns:
        List[Prediction]: A list of prediction records that match the specified criteria.
    """
    if start_date and end_date and model_to_use:
        history = (
            db.query(Prediction)
            .filter(
                and_(
                    Prediction.ts >= start_date,
                    Prediction.ts <= end_date,
                    Prediction.model_name == model_to_use,
                )
            )
            .limit(limit)
            .all()
        )
    elif start_date and end_date:
        history = (
            db.query(Prediction)
            .filter(
                and_(
                    Prediction.ts >= start_date,
                    Prediction.ts <= end_date,
                )
            )
            .limit(limit)
            .all()
        )
    elif start_date and model_to_use:
        history = (
            db.query(Prediction)
            .filter(
                and_(
                    Prediction.ts >= start_date,
                    Prediction.model_name == model_to_use,
                )
            )
            .limit(limit)
            .all()
        )
    elif end_date and model_to_use:
        history = (
            db.query(Prediction)
            .filter(
                and_(
                    Prediction.ts <= end_date,
                    Prediction.model_name == model_to_use,
                )
            )
            .limit(limit)
            .all()
        )
    elif start_date:
        history = (
            db.query(Prediction).filter(Prediction.ts >= start_date).limit(limit).all()
        )
    elif end_date:
        history = (
            db.query(Prediction).filter(Prediction.ts <= end_date).limit(limit).all()
        )
    elif model_to_use:
        history = (
            db.query(Prediction)
            .filter(Prediction.model_name == model_to_use)
            .limit(limit)
            .all()
        )
    else:
        history = db.query(Prediction).limit(limit).all()
    return history
