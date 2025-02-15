
from dagster import load_assets_from_package_module
from src.prompt_engineering import database_preprocessing, basic_prompting

# Group name for run pipelines
PROMPT_ENGINEERING = "prompt_engineering"

# Load all assets from different modules
tables_preprocessing = load_assets_from_package_module(package_module=database_preprocessing, group_name=PROMPT_ENGINEERING)
basic_prompting_wf = load_assets_from_package_module(package_module=basic_prompting, group_name=PROMPT_ENGINEERING)
