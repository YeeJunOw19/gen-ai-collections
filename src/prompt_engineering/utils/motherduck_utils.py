
import duckdb
import polars as pl
from sqlalchemy.engine import Engine
from src.data_ingestion.mdutils import motherduck_setup


def max_id_extraction(duck_engine: Engine, md_schema: str, md_table: str, column_name: str) -> int:
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
