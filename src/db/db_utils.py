"""
This module provides utility functions for working with databases using SQLAlchemy.

Functions:
- _create_schema(engine, schema): Creates a new schema in the database.
- crete_database_schemas_tables(connection_string, schema_name, table_list): Creates the database, schema, and tables if they do not exist.
- insert_values_into_table(connection_string, schema_name, table_name, values): Inserts values into a table in the database.
"""

from sqlalchemy import Double, Integer, create_engine, inspect, text, insert
from sqlalchemy_utils.functions import database_exists, create_database
from sqlalchemy import create_engine, inspect, text, Table, Column, MetaData
from sqlalchemy import String, DateTime, Float, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from typing import Any
from src.api.database import Base

SCHEMA = "public"
TABLE_LIST = ["models", "predictions"]


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

    # # Define metadata
    # metadata = MetaData()
    # # Define tables
    # table_models = Table(
    #     "models",
    #     metadata,
    #     Column("model_id", UUID, primary_key=True),
    #     Column("train_date", DateTime),
    #     Column("model_name", String),
    #     Column("model_type", String),
    #     Column("hyperparameters", JSON),
    #     Column("roc_auc_train", Float),
    #     Column("recall_train", Float),
    #     Column("roc_auc_test", Float),
    #     Column("recall_test", Float),
    #     Column("model_path", String),
    #     schema=schema_name,
    # )

    # table_predictions = Table(
    #     "predictions",
    #     metadata,
    #     Column("id", UUID, primary_key=True),
    #     Column("model_name", String),
    #     Column("ts", DateTime),
    #     Column("input_data", JSON),
    #     Column("prediction_label", Integer),
    #     Column("prediction_proba", Double),
    #     schema=schema_name,
    # )

    # # Create tables if not exists
    # for table in table_list:
    #     if table not in inspector.get_table_names(schema=schema_name):
    #         metadata.create_all(engine)

    # conn.close()
    Base.metadata.create_all(bind=engine)


def insert_values_into_table(
    connection_string: str, schema_name: str, table_name: str, values: dict[str, Any]
) -> None:
    engine = create_engine(connection_string)
    conn = engine.connect()

    # Define metadata
    metadata = MetaData()

    # Define the table
    table = Table(table_name, metadata, autoload_with=engine, schema=schema_name)

    # Create an Insert object
    stmt = insert(table).values(values)

    # Execute the statement
    with engine.connect() as connection:
        connection.execute(stmt)
        connection.commit()
