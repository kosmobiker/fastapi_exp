from datetime import datetime
from sqlalchemy import and_
from sqlalchemy.orm import Session

from src.api.models import TrainedModels


def get_models(
    db: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 100,
):
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
