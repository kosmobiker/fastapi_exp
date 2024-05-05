from sqlalchemy import create_engine, inspect
from src.db.db_utils import crete_database_schemas_tables
from sqlalchemy_utils.functions import database_exists

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
