
from pathlib import Path
import duckdb

SQL_SCRIPTS_LOC = Path(__file__).joinpath("..", "..", "..", "sql_scripts").resolve()


def execute_sql_scripts(duck_conn: duckdb.DuckDBPyConnection, sql_script: str) -> None:
    # Create the actual location of the script and read in the script
    sql_string = open(SQL_SCRIPTS_LOC.joinpath(sql_script).resolve(), mode="r").read()
    sql_parts = [stmt.strip() for stmt in sql_string.split(";") if stmt.strip()]

    # Execute SQL Script
    for sql_part in sql_parts:
        duck_conn.execute(sql_part)
