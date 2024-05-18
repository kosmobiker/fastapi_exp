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


class TransactionBase(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: int
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: int
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int
    device_distinct_emails_8w: int
    device_fraud_count: int
    month: int

    class Config:
        orm_mode = True
