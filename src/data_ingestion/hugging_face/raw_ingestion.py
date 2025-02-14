
import polars as pl
import os
import yaml
from pathlib import Path
from dagster import asset
from datetime import datetime

HUGGING_FACE_API = os.getenv("HUGGING_FACE_API")
CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))
POLARS_TYPE = {"String": pl.Utf8, "Integer": pl.Int64, "Datetime": pl.Datetime(time_unit="ms")}


def _get_hugging_face_data(run_config: dict) -> pl.LazyFrame:
    """
    This Dagster asset is used to query Hugging Face data using Polars native API. This asset does not load Hugging Face
    data into memory, instead it loads a Polars LazyFrame object.

    :return: Polars LazyFrame object of Hugging Face data.
    """

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
    if run_config["MotherDuck_Table"] == "RawNewsCategory":
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
            .filter(pl.col("NewsDate") > datetime(2021, 9, 30))
        )

    else:
        df = (
            pl.scan_parquet(
                source=f"hf://{source_url}",
                storage_options={"token": HUGGING_FACE_API},
                row_index_name="Id",
                row_index_offset=1,
                schema=table_schema,
            )
            .rename(col_names)
        )

    return df


@asset
def get_news_dataset() -> pl.LazyFrame:
    # Get configuration and run workflow
    configs = CONFIG["News_Dataset_Config"]
    return _get_hugging_face_data(configs)


@asset
def get_qa_dataset() -> pl.LazyFrame:
    # Get configuration and run workflow
    configs = CONFIG["GSM8K_Dataset_Config"]
    return _get_hugging_face_data(configs)
