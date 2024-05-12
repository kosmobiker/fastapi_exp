from datetime import datetime
from uuid import uuid4
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, Table, create_engine, delete

from src.api.main import app
from src.db.db_utils import CONNECTION_STRING, SCHEMA, insert_values_into_table

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
