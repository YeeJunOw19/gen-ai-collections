
import duckdb
import polars as pl
from sqlalchemy.engine import Engine
from src.data_ingestion.mdutils import motherduck_setup


def max_id_extraction(duck_engine: Engine, md_schema: str, md_table: str, column_name: str) -> int:
    """
    This function takes in a table and the index column from the table and find the maximum index. This is required
    because incremental indexing is not implemented in MotherDuck.

    :param duck_engine: DuckDB SQLAlchemy engine object
    :param md_schema: MotherDuck Schema
    :param md_table: Target table in MotherDuck to get the index
    :param column_name: The target primary key column with the index
    :return: Current maximum index number in the target MotherDuck table
    """

    query_string = f'SELECT MAX({column_name}) AS \'Id\' FROM "{md_schema}".{md_table}'

    # Extract the maximum Id number from MotherDuck table. If empty, return 0
    max_number = (
        motherduck_setup.md_read_table(
            duck_engine=duck_engine,
            md_schema=md_schema,
            md_table=md_table,
            keep_columns=None,
            custom_query=query_string
        )
        .collect()
        .to_series()
        .to_list()
    )

    if max_number[0] is None:
        return 0
    else:
        return max_number[0]


def model_runs_modeling(df: pl.DataFrame, dim_idx: int, fact_idx: int) -> dict:
    """
    This function takes in a Polars dataframe with the results from a given prompt engineering task, and calculates
    the accuracy of the model run. Then it further splits the dataframe into a dimension table and a fact table to be
    stored efficiently in MotherDuck.

    :param df: Polars dataframe with the results from prompt engineering task
    :param dim_idx: The current index count of the MotherDuck dimension table
    :param fact_idx: The current index count of the MotherDuck fact table
    :return: A dictionary containing two dataframes, one for dim table and one for fact table
    """

    # From the provided table, calculate accuracy and prep table for Dim table upload
    df_dim = (
        df
        .group_by(["ModelName", "ModelTemperature", "RunStyle", "RunDate"])
        .agg(
            pl.col("IndCorrect").sum().alias("CorrectCounts"),
            pl.len().alias("TotalCounts"),
        )
        .with_columns(((pl.col("CorrectCounts") / pl.col("TotalCounts")) * 100).alias("ModelAccuracy"))
        .with_row_index("ModelId", offset=dim_idx + 1)
        .select("ModelId", "ModelName", "ModelTemperature", "RunStyle", "ModelAccuracy", "RunDate")
    )

    df_fact = (
        df
        .with_columns(pl.lit(dim_idx + 1).alias("ModelId"))
        .with_row_index("Id", offset=fact_idx + 1)
        .select("Id", "ModelId", "RowId", "OpenAIAnswer", "ExtractedAnswer", "IndCorrect")
    )

    return {"df_dim": df_dim, "df_fact": df_fact}
