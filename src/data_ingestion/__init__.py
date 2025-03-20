
from dagster import load_assets_from_package_module
from src.data_ingestion import hugging_face, text_embedding, python_embedding

# Group name for run pipelines
DATA_INGESTION = "data_ingestion"
VECTOR_EMBEDDING = "vector_embedding"

# Load all assets from different modules
raw_data_ingestion = load_assets_from_package_module(package_module=hugging_face, group_name=DATA_INGESTION)
vector_embedding_wf = load_assets_from_package_module(package_module=text_embedding, group_name=VECTOR_EMBEDDING)
python_embedding_wf = load_assets_from_package_module(package_module=python_embedding, group_name=DATA_INGESTION)
