
from dagster import AssetSelection, define_asset_job
from src.data_ingestion import DATA_INGESTION

run_raw_data_save = define_asset_job(
    name="run_raw_data_save",
    selection=AssetSelection.groups(DATA_INGESTION)
)
