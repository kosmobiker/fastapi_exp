from datetime import datetime
import json
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, Table, create_engine, delete

from src.api.main import app
from src.db.db_utils import CONNECTION_STRING, SCHEMA, insert_values_into_table
from tests.functional.test_train import _fake_get_data

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_read_models():
    # Given
    engine = create_engine(CONNECTION_STRING)
    conn = engine.connect()
    fake_model_id = uuid4()
    fake_trained_model = {
        "model_id": str(fake_model_id),
        "train_date": datetime(2024, 1, 1, 1, 1, 1).isoformat(),
        "model_name": "foobar_model",
        "model_type": "FoobarForestClassifier",
        "hyperparameters": {"n_estimators": 100},
        "roc_auc_train": 0.9,
        "recall_train": 0.8,
        "roc_auc_test": 0.7,
        "recall_test": 0.6,
        "model_path": "models/test_model.pkl",
    }
    insert_values_into_table(CONNECTION_STRING, SCHEMA, "models", fake_trained_model)

    # When
    response = client.get("/models/?end_date=2024-01-02")

    # Then
    with conn.begin():
        # Define the table
        metadata = MetaData()
        table = Table("models", metadata, autoload_with=engine, schema=SCHEMA)
        assert response.status_code == 200
        assert response.json() == [fake_trained_model]

        delete_stmt = delete(table).where(table.c.model_id == fake_model_id)
        conn.execute(delete_stmt)

def test_predict_fraud_default_logreg_model_one_tx():
    # Given
    transaction_json = _fake_get_data(1).drop(["fraud_bool"], axis=1).to_dict(orient="records")[0]

    # When
    response = client.post("/predict/", json=transaction_json)

    # Then
    assert response.status_code == 200

    result = response.json()
    print("this is a transaction result")
    print(result)
    assert result["prediction_label"] == 0 or result["prediction_label"] == 1, "Label should be 0 or 1"
    assert 0 <= result["prediction_proba"][0] <= 1, "Prediction probability should be between 0 and 1"
    assert 0 <= result["prediction_proba"][1] <= 1, "Prediction probability should be between 0 and 1"
    assert result["prediction_proba"][0] + result["prediction_proba"][1] == 1, "Probabilities should sum to 1"
    assert isinstance(result["model_used"], str)
