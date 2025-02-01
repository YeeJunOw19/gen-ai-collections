
from dagster import Definitions
from src.data_ingestion import raw_data_ingestion
from src.jobs import run_raw_data_save

all_assets = [*raw_data_ingestion]

defs = Definitions(
    assets=all_assets,
    jobs=[run_raw_data_save]
)
