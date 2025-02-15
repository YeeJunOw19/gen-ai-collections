
from dagster import AssetSelection, define_asset_job
from src.data_ingestion import DATA_INGESTION, VECTOR_EMBEDDING
from src.prompt_engineering import PROMPT_ENGINEERING

run_raw_data_save = define_asset_job(
    name="run_raw_data_save",
    selection=AssetSelection.groups(DATA_INGESTION)
)

run_vector_embedding = define_asset_job(
    name="run_vector_embedding",
    selection=AssetSelection.groups(VECTOR_EMBEDDING)
)

run_prompt_engineering = define_asset_job(
    name="run_prompt_engineering",
    selection=AssetSelection.groups(PROMPT_ENGINEERING)
)
