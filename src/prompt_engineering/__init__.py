
from dagster import load_assets_from_package_module
from src.prompt_engineering import database_preprocessing, main_workflows, data_modeling, opro_implementation

# Group name for run pipelines
PROMPT_ENGINEERING = "prompt_engineering"

# Load all assets from different modules
tables_preprocessing = load_assets_from_package_module(package_module=database_preprocessing, group_name=PROMPT_ENGINEERING)
prompting_wf = load_assets_from_package_module(package_module=main_workflows, group_name=PROMPT_ENGINEERING)
data_modeling_wf = load_assets_from_package_module(package_module=data_modeling, group_name=PROMPT_ENGINEERING)
opro_wf = load_assets_from_package_module(package_module=opro_implementation, group_name=PROMPT_ENGINEERING)
