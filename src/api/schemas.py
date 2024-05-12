from datetime import datetime
from typing import Any
from uuid import UUID
from pydantic import BaseModel


class TrainedModelBase(BaseModel):
    model_id: UUID
    train_date: datetime
    model_name: str
    model_type: str
    hyperparameters: dict[str, Any] | None
    roc_auc_train: float | None
    recall_train: float | None
    roc_auc_test: float
    recall_test: float
    model_path: str

    class Config:
        orm_mode = True
