import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from fastapi_exp.core.database import Base


class FeatureStore(Base):
    __tablename__ = "feature_store"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True, comment="User's identifier")
    
    # Numeric features
    income = Column(Float, nullable=False, comment="Annual income in decile form [0.1, 0.9]")
    name_email_similarity = Column(Float, nullable=False, comment="Similarity between email and name [0, 1]")
    prev_address_months_count = Column(Integer, nullable=True, comment="Months in previous address [-1, 380], -1 = missing")
    current_address_months_count = Column(Integer, nullable=True, comment="Months in current address [-1, 429], -1 = missing")
    customer_age = Column(Integer, nullable=False, comment="Age in years, rounded to decade [10, 90]")
    days_since_request = Column(Integer, nullable=False, comment="Days since application [0, 79]")
    intended_balcon_amount = Column(Float, nullable=True, comment="Initial transfer amount [-16, 114], negatives = missing")
    zip_count_4w = Column(Integer, nullable=False, comment="Applications in same zip (4 weeks) [1, 6830]")
    velocity_6h = Column(Float, nullable=False, comment="Applications velocity (6 hours) [-175, 16818]")
    velocity_24h = Column(Float, nullable=False, comment="Applications velocity (24 hours) [1297, 9586]")
    velocity_4w = Column(Float, nullable=False, comment="Applications velocity (4 weeks) [2825, 7020]")
    bank_branch_count_8w = Column(Integer, nullable=False, comment="Branch applications (8 weeks) [0, 2404]")
    date_of_birth_distinct_emails_4w = Column(Integer, nullable=False, comment="Emails with same DOB (4 weeks) [0, 39]")
    credit_risk_score = Column(Float, nullable=False, comment="Internal risk score [-191, 389]")
    bank_months_count = Column(Integer, nullable=True, comment="Previous account age in months [-1, 32], -1 = missing")
    proposed_credit_limit = Column(Float, nullable=False, comment="Proposed credit limit [200, 2000]")
    session_length_in_minutes = Column(Float, nullable=True, comment="Session length [-1, 107], -1 = missing")
    device_distinct_emails = Column(Integer, nullable=True, comment="Distinct emails from device (8 weeks) [-1, 2], -1 = missing")
    device_fraud_count = Column(Integer, nullable=False, comment="Fraudulent applications from device [0, 1]")
    month = Column(Integer, nullable=False, comment="Application month [0, 7]")
    
    # Categorical features
    payment_type = Column(String(50), nullable=False, comment="Credit payment plan type (5 anonymized values)")
    employment_status = Column(String(50), nullable=False, comment="Employment status (7 anonymized values)")
    housing_status = Column(String(50), nullable=False, comment="Residential status (7 anonymized values)")
    source = Column(String(20), nullable=False, comment="Application source: INTERNET or TELEAPP")
    device_os = Column(String(20), nullable=False, comment="Device OS: Windows, macOS, Linux, X11, other")
    
    # Binary features
    email_is_free = Column(Boolean, nullable=False, comment="Email domain type (free or paid)")
    phone_home_valid = Column(Boolean, nullable=False, comment="Home phone validity")
    phone_mobile_valid = Column(Boolean, nullable=False, comment="Mobile phone validity")
    has_other_cards = Column(Boolean, nullable=False, comment="Has other cards from same bank")
    foreign_request = Column(Boolean, nullable=False, comment="Request from different country")
    keep_alive_session = Column(Boolean, nullable=False, comment="Session logout option")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<FeatureStore(id={self.id}, customer_age={self.customer_age}>"
    
    def to_dict(self):
        """Convert model to dictionary for API responses"""
        return {
            'id': str(self.id),
            'income': self.income,
            'name_email_similarity': self.name_email_similarity,
            'prev_address_months_count': self.prev_address_months_count,
            'current_address_months_count': self.current_address_months_count,
            'customer_age': self.customer_age,
            'days_since_request': self.days_since_request,
            'intended_balcon_amount': self.intended_balcon_amount,
            'payment_type': self.payment_type,
            'zip_count_4w': self.zip_count_4w,
            'velocity_6h': self.velocity_6h,
            'velocity_24h': self.velocity_24h,
            'velocity_4w': self.velocity_4w,
            'bank_branch_count_8w': self.bank_branch_count_8w,
            'date_of_birth_distinct_emails_4w': self.date_of_birth_distinct_emails_4w,
            'employment_status': self.employment_status,
            'credit_risk_score': self.credit_risk_score,
            'email_is_free': self.email_is_free,
            'housing_status': self.housing_status,
            'phone_home_valid': self.phone_home_valid,
            'phone_mobile_valid': self.phone_mobile_valid,
            'bank_months_count': self.bank_months_count,
            'has_other_cards': self.has_other_cards,
            'proposed_credit_limit': self.proposed_credit_limit,
            'foreign_request': self.foreign_request,
            'source': self.source,
            'session_length_in_minutes': self.session_length_in_minutes,
            'device_os': self.device_os,
            'keep_alive_session': self.keep_alive_session,
            'device_distinct_emails': self.device_distinct_emails,
            'device_fraud_count': self.device_fraud_count,
            'month': self.month,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Predictions(Base):
    __tablename__ = "predictions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Foreign key to feature store record
    feature_store_id = Column(UUID(as_uuid=True), ForeignKey('feature_store.id'), nullable=False, index=True)
    
    # Prediction results
    predicted_fraud = Column(Boolean, nullable=False, comment="Model prediction: fraud or not fraud")
    fraud_probability = Column(Float, nullable=False, comment="Probability score [0.0, 1.0]")
    confidence_score = Column(Float, nullable=True, comment="Model confidence in prediction [0.0, 1.0]")
    
    # Model information
    model_name = Column(String(100), nullable=False, comment="Name/identifier of the model used")
    model_version = Column(String(50), nullable=False, comment="Version of the model")
    model_type = Column(String(50), nullable=True, comment="Type of model (e.g., RandomForest, XGBoost, NN)")
    
    # Performance metrics (if available)
    prediction_time_ms = Column(Float, nullable=True, comment="Time taken for prediction in milliseconds")

    # Business context
    risk_category = Column(String(20), nullable=True, comment="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    action_recommended = Column(String(50), nullable=True, comment="Recommended action: APPROVE, REVIEW, REJECT")
    
    # Validation and feedback
    is_validated = Column(Boolean, default=False, comment="Has this prediction been validated?")
    actual_fraud = Column(Boolean, nullable=True, comment="Actual fraud outcome (for model feedback)")
    validation_date = Column(DateTime, nullable=True, comment="When validation was performed")
    validation_source = Column(String(50), nullable=True, comment="Source of validation (manual, automated, etc.)")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship
    feature_store = relationship("FeatureStore", backref="predictions")

    def __repr__(self):
        return f"<Predictions(id={self.id}, predicted_fraud={self.predicted_fraud}, probability={self.fraud_probability:.3f})>"
    
    def to_dict(self):
        """Convert model to dictionary for API responses"""
        return {
            'id': str(self.id),
            'feature_store_id': str(self.feature_store_id),
            'predicted_fraud': self.predicted_fraud,
            'fraud_probability': self.fraud_probability,
            'confidence_score': self.confidence_score,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_type': self.model_type,
            'prediction_time_ms': self.prediction_time_ms,
            'risk_category': self.risk_category,
            'action_recommended': self.action_recommended,
            'is_validated': self.is_validated,
            'actual_fraud': self.actual_fraud,
            'validation_date': self.validation_date.isoformat() if self.validation_date else None,
            'validation_source': self.validation_source,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_pipeline(cls, feature_store_id, prob, model_name, version, model_type, prediction_time_ms=None):
        predicted_fraud = prob > 0.5
        instance = cls(
            feature_store_id=feature_store_id,
            predicted_fraud=predicted_fraud,
            fraud_probability=prob,
            confidence_score=prob,
            model_name=model_name,
            model_version=version,
            model_type=model_type,
            prediction_time_ms=prediction_time_ms,
            risk_category=cls.estimate_risk(prob),
            action_recommended=cls.recommend_action(predicted_fraud, prob),
        )
        return instance

    @staticmethod
    def estimate_risk(prob):
        if prob >= 0.8:
            return "CRITICAL"
        elif prob >= 0.6:
            return "HIGH"
        elif prob >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def recommend_action(predicted_fraud, prob):
        if predicted_fraud and prob >= 0.7:
            return "REJECT"
        elif predicted_fraud and prob >= 0.4:
            return "REVIEW"
        return "APPROVE"