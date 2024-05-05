from sqlalchemy import create_engine, inspect, text
from sqlalchemy_utils.functions import database_exists, create_database
from sqlalchemy import create_engine, inspect, text, Table, Column, MetaData
from sqlalchemy import String, DateTime, Float, JSON
from sqlalchemy.dialects.postgresql import UUID

CONNECTION_STING = "postgresql://myuser:mypassword@localhost:5432/mydatabase"
SCHEMA = "ml_schema"
TABLE_LIST = ["models"]


def _create_schema(engine, schema) -> None:
    stmt = text(f"CREATE SCHEMA {schema}")
    with engine.connect() as conn:
        conn.execute(stmt)
        conn.commit()


def crete_database_schemas_tables(
    connection_string: str, schema_name: str, table_list: list[str]
) -> None:
    engine = create_engine(connection_string)
    conn = engine.connect()
    # Create database if not exists
    if not database_exists(connection_string):
        create_database(connection_string)

    # Create schema if not exists
    inspector = inspect(engine)
    if schema_name not in inspector.get_schema_names():
        _create_schema(engine, schema_name)

    # Define metadata
    metadata = MetaData()
    # Define tables
    table_models = Table(
        "models",
        metadata,
        Column("model_id", UUID, primary_key=True),
        Column("train_date", DateTime),
        Column("model_name", String),
        Column("model_type", String),
        Column("hyperparameters", JSON),
        Column("roc_auc_train", Float),
        Column("recall_train", Float),
        Column("roc_auc_test", Float),
        Column("recall_test", Float),
        schema=schema_name,
    )

    # Create tables if not exists
    for table in table_list:
        if table not in inspector.get_table_names(schema=schema_name):
            metadata.create_all(engine)

    conn.close()
