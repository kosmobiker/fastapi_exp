from uuid import UUID

from pydantic import BaseModel


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    risk_class: str


class PredictRequest(BaseModel):
    user_id: UUID
