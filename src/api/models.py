from sqlalchemy import (
    JSON,
    UUID,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from src.api.database import Base


class TrainedModels(Base):
    __table_args__ = {"schema": "ml_schema"}
    __tablename__ = "models"

    model_id = Column(UUID, primary_key=True)
    train_date = Column(DateTime)
    model_name = Column(String)
    model_type = Column(String)
    hyperparameters = Column(JSON)
    roc_auc_train = Column(Float)
    recall_train = Column(Float)
    roc_auc_test = Column(Float)
    recall_test = Column(Float)
    model_path = Column(String)
