
import duckdb
import os
import polars as pl
from duckdb.duckdb import InvalidInputException
from sqlalchemy import create_engine, MetaData, text, Column, String, Integer, Table, Date
from sqlalchemy.engine import Engine, make_url

MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN")
SQL_ALCHEMY_TYPES = {"String": String, "Integer": Integer, "Datetime": Date}


class MotherDucking:

    def __init__(self, database_name: str, read_only: bool = False):
        self.database_name = database_name
        self.read_only = read_only
        self.duckdb_conn = self._motherduck_connection()
        self.duckdb_engine = self._motherduck_engine()

    def _motherduck_connection(self) -> duckdb.DuckDBPyConnection:
        """
        This function sets up a DuckDB connection to MotherDuck. This function should not be used for multithreading and
        concurrency.

        :param database_name: MotherDuck database name
        :param read_only: Specifies if the connection is read-only connection. Default is set to False
        :return: DuckDB Python connection to MotherDuck server
        """

        connection_url = f"md:{self.database_name}?motherduck_token={MOTHERDUCK_TOKEN}"

        # Add read-only mode to database URL if needed
        if self.read_only:
            connection_url = f"{connection_url}&access_mode=read_only"

        # Setup and test Motherduck connection
        try:
            duck_conn = duckdb.connect(connection_url)
            return duck_conn

        except InvalidInputException:
            raise Exception("Invalid MotherDuck connection.")

    def _motherduck_engine(self) -> Engine:
        """
        This function sets up SQLAlchemy connection engine to MotherDuck. This function can be used for multithreading,
        concurrency, and connection pooling.

        :param database_name: MotherDuck database name
        :param read_only: Specifies if the connection is read-only connection. Default is set to False
        :return: SQLAlchemy engine object connected to MotherDuck server
        """

        conn_string = f"duckdb:///md:{self.database_name}?motherduck_token={MOTHERDUCK_TOKEN}"

        # Add read-only mode to database URL if needed
        if self.read_only:
            conn_string = f"{conn_string}&access_mode=read_only"

        # Create and test SQLAlchemy engine
        conn_url = make_url(conn_string)
        duck_engine = create_engine(conn_url)

        try:
            with duck_engine.connect() as conn:
                duck_engine.dialect.do_ping(conn.connection)
            return duck_engine

        except:
            raise Exception("Invalid MotherDuck connection.")


def md_table_setup(
    duck_engine: Engine, schema_name: str, table_name: str,
    table_schema: list[dict], column_key: str, rebuild_table: bool = False,
) -> None:
    """
    This function sets up the table specified in the parameter of this function. The function will only run a CREATE
    script if the table is not already present on the database. Optionally, the function can perform a DROP script
    if specified.

    :param duck_engine: SQLAlchemy engine object connected to MotherDuck server
    :param schema_name: Target schema name in the MotherDuck database
    :param table_name: Target table name in the MotherDuck database
    :param table_schema: Specifies the table schema in the MotherDuck database
    :param rebuild_table: Set this to True to rebuild the table. Default is False
    :return: None
    """

    # Execute delete table string if rebuild table is required
    if rebuild_table:
        delete_string = f'DROP TABLE IF EXISTS "{schema_name}".{table_name}'
        with duck_engine.begin() as conn:
            conn.execute(text(delete_string))

    # Format configuration and create table in MotherDuck
    metadata = MetaData()
    columns = [
        Column(
            name=schema.get(column_key),
            type_=SQL_ALCHEMY_TYPES.get(schema["Column_Type"], String),
            primary_key=schema.get("Primary_Key", False),
            autoincrement=False
        )
        for schema in table_schema
    ]
    Table(table_name, metadata, *columns, schema=schema_name)
    metadata.create_all(duck_engine, checkfirst=True)


def md_read_table(
    duck_engine: Engine, md_schema: str, md_table: str, keep_columns: list[str] | None, custom_query: str = None
) -> pl.LazyFrame:
    """
    This function will query the data into Polars LazyFrame from MotherDuck database.
    It also performs column selection using the column names that user have provided.

    :param duck_engine: MotherDuck SQLAlchemy connection engine to MotherDuck server
    :param md_schema: MotherDuck schema name
    :param md_table: MotherDuck table name
    :param keep_columns: Columns to select from table in MotherDuck database
    :param custom_query: User can choose to provide custom query string to query MotherDuck table
    :return: A Polars LazyFrame object from MotherDuck database
    """

    if custom_query is None:
        query_string = f'SELECT * FROM "{md_schema}".{md_table}'
    else:
        query_string = custom_query

    with duck_engine.begin() as conn:
        if custom_query is None:
            df = (
                pl.read_database(text(query_string), connection=conn)
                .select(keep_columns)
                .lazy()
            )

        else:
            df = pl.read_database(text(query_string), connection=conn).lazy()

    return df
