
from pathlib import Path
from src.data_ingestion.mdutils import motherduck_setup

MOTHERDUCK_DATABASE = "mlds-database"
MOTHERDUCK_SCHEMA = "gen-ai"
MOTHERDUCK_TABLE = "PythonCodeInstruct"
CACHED_DATA = "training_data.parquet"
TRAINING_SAMPLES = 300


def _get_training_data() -> None:
    # Create an instance of MotherDuck
    md = motherduck_setup.MotherDucking(MOTHERDUCK_DATABASE, True)

    # SQL script to query training data
    sql_string = (
        f"""
        SELECT setseed(0.3825);
        SELECT *
        FROM "{MOTHERDUCK_SCHEMA}".{MOTHERDUCK_TABLE}
        WHERE DataSplit = 'Training' AND QuestionInput = ''
        ORDER BY RANDOM()
        LIMIT {TRAINING_SAMPLES}; 
        """
    )

    # Query the data from MotherDuck and save a copy of them locally
    cache_location = Path(__file__).joinpath("..", "..", "data", CACHED_DATA).resolve()
    df = motherduck_setup.md_read_table(
        duck_engine=md.duckdb_engine, md_schema=MOTHERDUCK_SCHEMA, md_table=MOTHERDUCK_TABLE,
        keep_columns=None, custom_query=sql_string
    )
    df.collect().write_parquet(cache_location, compression="zstd", compression_level=22)


if __name__ == "__main__":
    _get_training_data()
