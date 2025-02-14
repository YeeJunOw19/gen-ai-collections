
from dagster import asset
from src.data_ingestion.hugging_face.raw_ingestion import CONFIG
from src.data_ingestion.hugging_face import raw_to_motherduck as rm
from src.data_ingestion.mdutils import motherduck_dml as dm, motherduck_setup

ALL_SCRIPTS = CONFIG["MotherDuck_DML_Scripts"]


@asset(deps=[rm.load_news_dataset, rm.load_qa_dataset])
def execute_dml_scripts() -> None:
    # Run through the list to execute any scripts that need to be executed
    for script in ALL_SCRIPTS:
        md = motherduck_setup.MotherDucking(script["MotherDuck_Database"], False)
        dm.execute_sql_scripts(md.duckdb_conn, script["Execution_Scripts"])
