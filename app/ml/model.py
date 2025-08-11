from pathlib import Path

import joblib
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model(Path("data/model_artifacts/xgb_model.json"))
preprocessor = joblib.load(Path("data/model_artifacts/preprocessor.pkl"))


def predict_proba(features: list) -> tuple[int, float, str]:
    X = preprocessor.transform([features])
    proba = model.predict_proba(X)[0][1]
    label = int(proba >= 0.5)

    # Risk classification
    if proba >= 0.8:
        risk = "high"
    elif proba >= 0.5:
        risk = "medium"
    else:
        risk = "low"

    return label, float(proba), risk
