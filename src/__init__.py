
from dagster import Definitions
from src.data_ingestion import raw_data_ingestion, vector_embedding_wf
from src.prompt_engineering import tables_preprocessing, basic_prompting_wf
from src.jobs import run_raw_data_save, run_vector_embedding, run_prompt_engineering

all_assets = [*raw_data_ingestion, *vector_embedding_wf, *tables_preprocessing, *basic_prompting_wf]

defs = Definitions(
    assets=all_assets,
    jobs=[run_raw_data_save, run_vector_embedding, run_prompt_engineering]
)
