from sqlalchemy import create_engine, inspect
from src.db.db_utils import crete_database_schemas_tables, insert_values_into_table
from sqlalchemy_utils.functions import database_exists
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table, select, delete
from datetime import datetime
from uuid import uuid4

CONNECTION_STING = "postgresql://myuser:mypassword@localhost:5432/mydatabase"
SCHEMA = "ml_schema"
TABLE_LIST = ["models"]


def tests_connection_db_schemas_tables(
    connection_string: str = CONNECTION_STING,
    schema_name: str = SCHEMA,
    table_list: list[str] = TABLE_LIST,
):
    # Given
    engine = create_engine(connection_string)
    conn = engine.connect()

    # When
    inspector = inspect(engine)
    crete_database_schemas_tables(connection_string, schema_name, table_list)

    # Then
    assert database_exists(connection_string) == True
    assert schema_name in inspector.get_schema_names()
    assert all(
        [table in inspector.get_table_names(schema=schema_name) for table in table_list]
    )
    conn.close()


def test_insert_values(
    connection_string: str = CONNECTION_STING,
    schema_name: str = SCHEMA,
):
    # Given
    engine = create_engine(connection_string)
    conn = engine.connect()
    table_name = "models"
    values = {
        "model_id": uuid4(),  # Generate a random UUID
        "train_date": datetime(2024, 1, 1, 1, 1),  # Use the current date and time
        "model_name": "test_model",
        "model_type": "test_type",
        "hyperparameters": {"param1": "foo", "param2": "bar"},  # Example JSON data
        "roc_auc_train": 0.9,
        "recall_train": 0.8,
        "roc_auc_test": 0.7,
        "recall_test": 0.6,
        "model_path": "path/to/model",
    }
    # Start a transaction
    with conn.begin():
        # When
        insert_values_into_table(connection_string, schema_name, table_name, values)

        # Then
        # Define the table
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine, schema=schema_name)

        # Create a Select object
        stmt = select(table).where(table.c.model_id == values["model_id"])

        # Execute the statement and fetch one row
        result = conn.execute(stmt).fetchone()
        keys = list(values)
        result_dict = dict(zip(keys, result))
        # Assert that the inserted data is correct
        assert result is not None, "No result found"
        for key, value in values.items():
            assert result_dict[key] == value, f"Value for {key} does not match"

        # Delete the inserted row
        delete_stmt = delete(table).where(table.c.model_id == values["model_id"])
        conn.execute(delete_stmt)
