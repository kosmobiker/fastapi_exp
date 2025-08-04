import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from app.db.models.feature_store import FeatureStore
import uuid
import os
import json
import random
import uuid

TEST_DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://test_user:test_pass@localhost:5433/test_db")

engine = create_engine(TEST_DB_URL)
TestingSessionLocal = sessionmaker(bind=engine)

def generate_synthetic_user(user_id=None):
    return {
        "id": user_id or uuid.uuid4(),
        "features": {
            "income": round(random.uniform(0.0, 1.0), 2),
            "name_email_similarity": round(random.uniform(0.0, 1.0), 2),
            "prev_address_months_count": random.randint(0, 60),
            "current_address_months_count": random.randint(1, 120),
            "customer_age": random.randint(18, 70),
            "days_since_request": round(random.uniform(-1.0, 1.0), 2),
            "intended_balcon_amount": round(random.uniform(-1.0, 1.0), 2),
            "payment_type": random.choice(["AB", "AC", "AD"]),
            "zip_count_4w": random.randint(100, 5000),
            "velocity_6h": round(random.uniform(0, 10000), 2),
            "velocity_24h": round(random.uniform(0, 10000), 2),
            "velocity_4w": round(random.uniform(0, 10000), 2),
            "bank_branch_count_8w": random.randint(0, 10),
            "date_of_birth_distinct_emails_4w": random.randint(0, 5),
            "employment_status": random.choice(["CC", "EMPLOYED", "UNEMPLOYED"]),
            "credit_risk_score": random.randint(0, 150),
            "email_is_free": random.choice([0, 1]),
            "housing_status": random.choice(["BC", "OWN", "RENT"]),
            "phone_home_valid": random.choice([0, 1]),
            "phone_mobile_valid": random.choice([0, 1]),
            "bank_months_count": random.randint(0, 60),
            "has_other_cards": random.choice([0, 1]),
            "proposed_credit_limit": round(random.uniform(100.0, 2000.0), 2),
            "foreign_request": random.choice([0, 1]),
            "source": random.choice(["INTERNET", "BRANCH"]),
            "session_length_in_minutes": round(random.uniform(0.1, 10.0), 2),
            "device_os": random.choice(["macintosh", "windows", "linux"]),
            "keep_alive_session": random.choice([0, 1]),
            "device_distinct_emails_8w": random.randint(0, 5),
            "device_fraud_count": random.randint(0, 3),
            "month": random.randint(1, 12),
        },
        "version": "v1"
    }

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    for _ in range(10):
        user_data = generate_synthetic_user()
        fs = FeatureStore(
            id=user_data["id"],
            features=user_data["features"],
            version=user_data["version"]
        )
        db.add(fs)

    db.commit()
    db.close()
    yield

