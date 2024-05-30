from datetime import datetime
from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from src.api import crud, database, models, schemas
from src.api.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/models/", response_model=list[schemas.TrainedModelBase])
def models(
    start_date: datetime = None,
    end_date: datetime = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    models = crud.get_models(db, start_date=start_date, end_date=end_date, limit=limit)
    return models


@app.post("/predict/")
def predict(
    transaction: list[schemas.TransactionBase],
    db: Session = Depends(get_db),
    model_to_use: str | None = None,
):
    predictions = crud.predict_fraud(transaction, db, model_to_use)
    return predictions


@app.get("/history/")
def history(
    start_date: datetime = None,
    end_date: datetime = None,
    model_to_use: str = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    history = crud.get_history(db)
    return history
