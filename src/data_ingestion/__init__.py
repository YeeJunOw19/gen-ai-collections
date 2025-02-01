
from dagster import load_assets_from_package_module
from src.data_ingestion import hugging_face

# Group name for run pipelines
DATA_INGESTION = "data_ingestion"

# Load all assets from different modules
raw_data_ingestion = load_assets_from_package_module(package_module=hugging_face, group_name=DATA_INGESTION)
