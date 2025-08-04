from sqlalchemy import Column, ForeignKey, Integer, Float, String, TIMESTAMP, func
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("feature_store.id"), index=True)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_class = Column(String, nullable=False)
    true_label = Column(Integer, nullable=True)
    model_version = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
