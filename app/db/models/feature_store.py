import uuid

from sqlalchemy import JSON, TIMESTAMP, Column, String, func
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


class FeatureStore(Base):
    __tablename__ = "feature_store"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    features = Column(JSON, nullable=False)
    version = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
