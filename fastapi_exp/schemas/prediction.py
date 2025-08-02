from pydantic import BaseModel
import uuid
from datetime import datetime

class FeatureStoreBase(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int | None = None
    current_address_months_count: int | None = None
    customer_age: int
    days_since_request: int
    intended_balcon_amount: float | None = None
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    credit_risk_score: float
    bank_months_count: int | None = None
    proposed_credit_limit: float
    session_length_in_minutes: float | None = None
    device_distinct_emails: int | None = None
    device_fraud_count: int
    month: int
    payment_type: str
    employment_status: str
    housing_status: str
    source: str
    device_os: str
    email_is_free: bool
    phone_home_valid: bool
    phone_mobile_valid: bool
    has_other_cards: bool
    foreign_request: bool
    keep_alive_session: bool

class FeatureStoreCreate(FeatureStoreBase):
    pass

class FeatureStore(FeatureStoreBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
