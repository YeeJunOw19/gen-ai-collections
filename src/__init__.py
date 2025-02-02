
from dagster import Definitions
from src.data_ingestion import raw_data_ingestion, vector_embedding_wf
from src.jobs import run_raw_data_save, run_vector_embedding

all_assets = [*raw_data_ingestion, *vector_embedding_wf]

defs = Definitions(
    assets=all_assets,
    jobs=[run_raw_data_save, run_vector_embedding]
)
