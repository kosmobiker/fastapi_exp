import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.endpoints.predict import get_db
from app.db.models.feature_store import FeatureStore
from app.db.models.predictions import Prediction
from app.main import app

from .conftest import TestingSessionLocal, generate_synthetic_user


@pytest.fixture(scope="function")
def db_session():
    """
    Yield a new database session for each test function.
    Rollback any changes after the test is done.
    """
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def client_fixture(db_session):
    """
    Provide a test client with a patched database dependency.
    """

    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestSyncPredict:
    def test_predict_success_high_risk(self, client_fixture, db_session):
        # Given: A user with features in the database
        user_data = generate_synthetic_user()
        user_id = user_data["id"]
        feature_record = FeatureStore(
            id=user_id,
            features=user_data["features"],
            created_at=user_data["created_at"],
            version=user_data["version"],
        )
        db_session.add(feature_record)
        db_session.commit()

        # When: The /predict/sync endpoint is called with the user_id
        with patch("app.api.endpoints.predict.predict_proba") as mock_predict_proba:
            mock_predict_proba.return_value = (1, 0.85, "high")
            response = client_fixture.post(
                "/predict/sync", json={"user_id": str(user_id)}
            )

        # Then: The response should be successful and contain the prediction
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert data["probability"] == 0.85
        assert data["risk_class"] == "high"

        # And: The prediction should be saved to the database
        prediction_db = (
            db_session.query(Prediction).filter(Prediction.user_id == user_id).one()
        )
        assert prediction_db.prediction == 1
        assert prediction_db.probability == 0.85
        assert prediction_db.risk_class == "high"
        assert prediction_db.model_version == "v1"

    def test_predict_success_low_risk(self, client_fixture, db_session):
        # Given: A user with features in the database
        user_data = generate_synthetic_user()
        user_id = user_data["id"]
        feature_record = FeatureStore(
            id=user_id,
            features=user_data["features"],
            created_at=user_data["created_at"],
            version=user_data["version"],
        )
        db_session.add(feature_record)
        db_session.commit()

        # When: The /predict/sync endpoint is called with the user_id
        with patch("app.api.endpoints.predict.predict_proba") as mock_predict_proba:
            mock_predict_proba.return_value = (0, 0.25, "low")
            response = client_fixture.post(
                "/predict/sync", json={"user_id": str(user_id)}
            )

        # Then: The response should be successful and contain the prediction
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0
        assert data["probability"] == 0.25
        assert data["risk_class"] == "low"

        # And: The prediction should be saved to the database
        prediction_db = (
            db_session.query(Prediction).filter(Prediction.user_id == user_id).one()
        )
        assert prediction_db.prediction == 0
        assert prediction_db.probability == 0.25
        assert prediction_db.risk_class == "low"

    def test_predict_user_not_found(self, client_fixture):
        # Given: A user_id that does not exist in the database
        non_existent_user_id = str(uuid.uuid4())

        # When: The /predict/sync endpoint is called
        response = client_fixture.post(
            "/predict/sync", json={"user_id": non_existent_user_id}
        )

        # Then: The API should return a 404 Not Found error
        assert response.status_code == 404
        assert response.json()["detail"] == "User not found"

    def test_predict_invalid_request_body(self, client_fixture):
        # Given: An invalid request payload
        invalid_payload = {"not_a_user_id": "some_value"}

        # When: The /predict/sync endpoint is called
        response = client_fixture.post("/predict/sync", json=invalid_payload)

        # Then: The API should return a 422 Unprocessable Entity error
        assert response.status_code == 422

    def test_sync_predict_database_error_on_commit(self, client_fixture, db_session):
        # Given: A user with features in the database
        user_data = generate_synthetic_user()
        user_id = user_data["id"]
        feature_record = FeatureStore(
            id=user_id,
            features=user_data["features"],
            created_at=user_data["created_at"],
            version="v1",
        )
        db_session.add(feature_record)
        db_session.commit()

        # When: A database error occurs during the commit
        with patch("app.api.endpoints.predict.predict_proba") as mock_predict_proba:
            mock_predict_proba.return_value = (1, 0.85, "high")
            with patch.object(db_session, "commit") as mock_commit:
                mock_commit.side_effect = Exception("Database error")
                with pytest.raises(Exception) as excinfo:
                    client_fixture.post("/predict/sync", json={"user_id": str(user_id)})

        # Then: The transaction should be rolled back and a 500 error should not be returned
        assert "Database error" in str(excinfo.value)
        assert not db_session.new
        assert not db_session.dirty
