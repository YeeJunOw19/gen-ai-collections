
import polars as pl
import yaml
from pathlib import Path
from dagster import asset, AssetIn
from src.data_ingestion.mdutils import motherduck_setup, motherduck_load
from src.prompt_engineering.utils import motherduck_utils

SCHEMA_LOCATION = Path(__file__).joinpath("..", "..", "database_preprocessing", "config.yaml").resolve()
TABLE_CONFIG = yaml.safe_load(open(SCHEMA_LOCATION, mode="r"))["Prompt_Engineering_Preparation"]
CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["Data_Modeling_Configurations"]


@asset(
    ins={
        "basic_results": AssetIn(key="basic_scoring"),
        "role_based_results": AssetIn(key="role_scoring")
    }
)
def dim_fact_modeling(basic_results, role_based_results):
    """
    Dagster asset that performs the actual data modeling work and prepare data to be ingested into MotherDuck.

    :param df: Polars dataframe with scores from basic level prompt engineering
    :return: A dictionary containing Polars dataframes for dimensional table and fact table
    """

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

    # Append all the dataframes together into one dataframe
    all_results = [basic_results, role_based_results]
    df = pl.DataFrame()
    for result in all_results:
        df = df.vstack(result["df"])

    # Using the indexes, create both dim and fact tables from input results
    modeling_dict = motherduck_utils.model_runs_modeling(df, dim_idx, fact_idx)

    return modeling_dict
