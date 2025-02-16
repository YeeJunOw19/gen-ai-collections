
from dagster import asset, AssetIn
from src.data_ingestion.mdutils import motherduck_setup, motherduck_load
from src.prompt_engineering.data_modeling.dim_fact_modeling import CONFIG, TABLE_CONFIG


@asset(ins={"dfs": AssetIn(key="dim_fact_modeling")})
def dim_table_load(dfs: dict) -> None:
    """
    Dagster asset to load dimension table to MotherDuck

    :param dfs: A dictionary containing Polars dataframes for dimensional table and fact table
    :return: None
    """

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


@asset(ins={"dfs": AssetIn(key="dim_fact_modeling")})
def fact_table_load(dfs: dict) -> None:
    """
    Dagster asset to load fact table to MotherDuck

    :param dfs: A dictionary containing Polars dataframes for dimensional table and fact table
    :return: None
    """

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