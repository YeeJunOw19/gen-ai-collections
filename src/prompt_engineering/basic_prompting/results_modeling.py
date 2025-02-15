
import polars as pl
import yaml
from pathlib import Path
from dagster import asset, AssetIn
from src.data_ingestion.mdutils import motherduck_setup, motherduck_load
from src.prompt_engineering.utils import motherduck_utils
from src.prompt_engineering.basic_prompting.main_prompt import CONFIG

SCHEMA_LOCATION = Path(__file__).joinpath("..", "..", "database_preprocessing", "config.yaml").resolve()
TABLE_CONFIG = yaml.safe_load(open(SCHEMA_LOCATION, mode="r"))["Prompt_Engineering_Preparation"]


@asset(ins={"df": AssetIn(key="main_basic_scoring")})
def data_modeling(df: pl.DataFrame):
    # Create an instance of MotherDuck
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], True)

    # Get current table indexes from MotherDuck
    dim_idx = motherduck_utils.max_id_extraction(
        duck_engine=md.duckdb_engine,
        md_schema=CONFIG["MotherDuck_Schema"],
        md_table=CONFIG["MotherDuck_Dimension_Table"],
        column_name=CONFIG["MotherDuck_Dimension_ID"],
    )

    fact_idx = motherduck_utils.max_id_extraction(
        duck_engine=md.duckdb_engine,
        md_schema=CONFIG["MotherDuck_Schema"],
        md_table=CONFIG["MotherDuck_Fact_Table"],
        column_name=CONFIG["MotherDuck_Fact_ID"],
    )

    # Using the indexes, create both dim and fact tables from input results
    modeling_dict = motherduck_utils.model_runs_modeling(df, dim_idx, fact_idx)

    return modeling_dict


@asset(ins={"dfs": AssetIn(key="data_modeling")})
def dim_table_load(dfs: dict) -> None:
    # Get Dim table schema
    table_schema = TABLE_CONFIG["MotherDuck_Tables"][0]["Table_Schema"]

    # Create MotherDuck instance and Arrow Buffer instance
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], False)
    arrow_buffer = motherduck_load.ArrowTableLoadingBuffer(
        duck_conn=md.duckdb_conn,
        data_schema=table_schema,
        md_schema=CONFIG["MotherDuck_Schema"],
        md_table=CONFIG["MotherDuck_Dimension_Table"],
        delete_before_load=False,
        custom_delete_string=None
    )

    # Load Dim table
    arrow_buffer.motherduck_insert(dfs["df_dim"].lazy())


@asset(ins={"dfs": AssetIn(key="data_modeling")})
def fact_table_load(dfs: dict) -> None:
    # Get Fact table schema
    table_schema = TABLE_CONFIG["MotherDuck_Tables"][1]["Table_Schema"]

    # Create MotherDuck instance and Arrow Buffer instance
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], False)
    arrow_buffer = motherduck_load.ArrowTableLoadingBuffer(
        duck_conn=md.duckdb_conn,
        data_schema=table_schema,
        md_schema=CONFIG["MotherDuck_Schema"],
        md_table=CONFIG["MotherDuck_Fact_Table"],
        delete_before_load=False,
        custom_delete_string=None
    )

    # Load Fact table
    arrow_buffer.motherduck_insert(dfs["df_fact"].lazy())
