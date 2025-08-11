from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models.feature_store import FeatureStore
from app.db.models.predictions import Prediction
from app.db.session import SessionLocal
from app.ml.model import predict_proba
from app.schemas.predict import PredictRequest, PredictResponse
from uuid import UUID

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/predict/sync", response_model=PredictResponse)
def sync_predict(request: PredictRequest, db: Session = Depends(get_db)):
    record = db.query(FeatureStore).filter(FeatureStore.id == request.user_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="User not found")

    features = record.features
    label, proba, risk = predict_proba(features)

    prediction = Prediction(
        user_id=request.user_id,
        prediction=label,
        probability=proba,
        risk_class=risk,
        model_version="v1",
    )
    db.add(prediction)
    db.commit()

    return PredictResponse(prediction=label, probability=proba, risk_class=risk)


@router.get("/predict/sync/{user_id}", response_model=PredictResponse)
def sync_predict_get(user_id: UUID, db: Session = Depends(get_db)):
    record = db.query(FeatureStore).filter(FeatureStore.id == user_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="User not found")
    features = record.features
    label, proba, risk = predict_proba(features)
    prediction = Prediction(
        user_id=user_id,
        prediction=label,
        probability=proba,
        risk_class=risk,
        model_version="v1",
    )
    db.add(prediction)
    db.commit()
    return PredictResponse(prediction=label, probability=proba, risk_class=risk)
