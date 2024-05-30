from sqlalchemy import (
    ARRAY,
    JSON,
    UUID,
    Column,
    DateTime,
    Double,
    Float,
    Integer,
    PickleType,
    String,
)

from src.api.database import Base


class TrainedModels(Base):
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


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID, primary_key=True)
    model_name = Column(String)
    ts = Column(DateTime)
    input_data = Column(JSON)
    prediction_label = Column(ARRAY(Integer))
    prediction_proba = Column(ARRAY(Double))
