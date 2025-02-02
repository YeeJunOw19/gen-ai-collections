
import polars as pl
import os
import yaml
from pathlib import Path
from dagster import asset

HUGGING_FACE_API = os.getenv("HUGGING_FACE_API")
CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))
POLARS_TYPE = {"String": pl.Utf8, "Integer": pl.Int64, "Datetime": pl.Datetime(time_unit="ms")}


@asset
def get_hugging_face_data() -> pl.LazyFrame:
    """
    This Dagster asset is used to query Hugging Face data using Polars native API. This asset does not load Hugging Face
    data into memory, instead it loads a Polars LazyFrame object.

    :return: Polars LazyFrame object of Hugging Face data.
    """

    run_config = CONFIG["News_Dataset_Config"]

    # Get the data schema and convert it to Polars schema
    table_schema = {}
    for schema in run_config["File_Format"]:
        if not schema.get("Source_Exclude", False):
            table_schema[schema["Column_Name"]] = POLARS_TYPE.get(schema["Column_Type"], pl.Utf8)

    # Get a dictionary of mapping to rename columns names
    col_names = {}
    for schema in run_config["File_Format"]:
        if not schema.get("Source_Exclude", False):
            col_names[schema["Column_Name"]] = schema["New_Name"]

    # Query data from Hugging Face
    source_url = run_config["Dataset_Path"]
    df = (
        pl.scan_parquet(
            source=f"hf://{source_url}",
            storage_options={"token": HUGGING_FACE_API},
            row_index_name="Id",
            row_index_offset=1,
            schema=table_schema,
        )
        .rename(col_names)
        .with_columns(pl.col("NewsDate").dt.date().alias("NewsDate"))
    )

    return df
