from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from uuid import UUID
import joblib
import xgboost as xgb
import pandas as pd

from fastapi_exp.core.database import get_db
from fastapi_exp.models.models import FeatureStore

router = APIRouter()

# Load the trained model and preprocessors
try:
    model = xgb.Booster()
    model.load_model("data/processed/model.json")
    preprocessors = joblib.load("data/processed/preprocessors.pkl")
    feature_info = joblib.load("data/processed/feature_info.pkl")
    print("Model and preprocessors loaded successfully!")
except Exception as e:
    print(f"Error loading model or preprocessors: {e}")
    model = None
    preprocessors = None
    feature_info = None

class PredictionOutput(BaseModel):
    is_fraud: bool
    fraud_probability: float

@router.post("/predict/{user_id}", response_model=PredictionOutput)
def predict_fraud(user_id: UUID, db: Session = Depends(get_db)):
    if model is None or preprocessors is None or feature_info is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    feature_data_orm = db.query(FeatureStore).filter(FeatureStore.id == user_id).first()

    if not feature_data_orm:
        raise HTTPException(status_code=404, detail="User features not found")

    # Convert ORM object to dictionary, then to pandas DataFrame
    feature_dict = feature_data_orm.to_dict()
    # Remove non-feature columns like 'id', 'created_at', 'updated_at'
    for key in ['id', 'created_at', 'updated_at']:
        if key in feature_dict:
            del feature_dict[key]

    # Ensure the order of features matches the training data
    feature_df = pd.DataFrame([feature_dict])
    feature_df = feature_df[feature_info['feature_names']]

    # Apply preprocessors
    x_processed = feature_df.copy()

    # Numeric imputation
    if 'numeric_imputer' in preprocessors and feature_info['numeric_columns']:
        x_processed[feature_info['numeric_columns']] = preprocessors['numeric_imputer'].transform(x_processed[feature_info['numeric_columns']])

    # Categorical imputation
    if 'categorical_imputer' in preprocessors and feature_info['categorical_columns']:
        x_processed[feature_info['categorical_columns']] = preprocessors['categorical_imputer'].transform(x_processed[feature_info['categorical_columns']].astype(str))

    # Encode categorical variables
    if 'label_encoders' in preprocessors and feature_info['categorical_columns']:
        for col in feature_info['categorical_columns']:
            if col in preprocessors['label_encoders']:
                le = preprocessors['label_encoders'][col]
                x_processed[col] = le.transform(x_processed[col].astype(str))

    # Scale numeric features
    if 'scaler' in preprocessors and feature_info['numeric_columns']:
        x_processed[feature_info['numeric_columns']] = preprocessors['scaler'].transform(x_processed[feature_info['numeric_columns']])

    # Convert boolean columns to int
    if feature_info['boolean_columns']:
        for col in feature_info['boolean_columns']:
            x_processed[col] = x_processed[col].astype(int)

    # Make prediction
    dmatrix = xgb.DMatrix(x_processed)
    fraud_probability = model.predict(dmatrix)[0]
    is_fraud = fraud_probability > 0.5

    return PredictionOutput(is_fraud=is_fraud, fraud_probability=fraud_probability)
