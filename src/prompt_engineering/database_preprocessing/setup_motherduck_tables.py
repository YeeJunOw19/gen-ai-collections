
import yaml
from pathlib import Path
from dagster import asset
from src.data_ingestion.mdutils import motherduck_setup

CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["Prompt_Engineering_Preparation"]


@asset
def prompt_engineering_preprocessing() -> None:
    # Create an instance of MotherDuck engine
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], False)
    md_schema = CONFIG["MotherDuck_Schema"]
    md_tables = CONFIG["MotherDuck_Tables"]

    # Loop through all the tables and make sure that MotherDuck has a copy of those tables
    for table in md_tables:
        table_name = table["Table_Name"]
        table_schema = table["Table_Schema"]

        motherduck_setup.md_table_setup(
            duck_engine=md.duckdb_engine,
            schema_name=md_schema,
            table_name=table_name,
            table_schema=table_schema,
            column_key="Column_Name"
        )
