
import polars as pl
from dagster import asset, AssetIn
from src.data_ingestion.hugging_face.raw_ingestion import CONFIG
from src.data_ingestion.mdutils import motherduck_setup, motherduck_load

RUN_CONFIG = CONFIG["News_Dataset_Config"]


@asset
def prep_motherduck() -> None:
    # Configurate MotherDuck setup and run setup scripts to setup MotherDuck database
    table_schema = RUN_CONFIG["File_Format"]
    md = motherduck_setup.MotherDucking(RUN_CONFIG["MotherDuck_Database"])

    motherduck_setup.md_table_setup(
        duck_engine=md.duckdb_engine,
        schema_name = RUN_CONFIG["MotherDuck_Schema"],
        table_name = RUN_CONFIG["MotherDuck_Table"],
        table_schema = table_schema,
        rebuild_table=True
    )


@asset(
    deps=[prep_motherduck],
    ins={"df": AssetIn(key="get_hugging_face_data")},
)
def load_data_to_motherduck(df: pl.LazyFrame) -> None:
    # Configurate MotherDuck engines
    md = motherduck_setup.MotherDucking(RUN_CONFIG["MotherDuck_Database"])

    # Setup arrow buffer and load data into MotherDuck by chunk
    arrow_buffer = motherduck_load.ArrowTableLoadingBuffer(
        duck_conn=md.duckdb_conn,
        data_schema=RUN_CONFIG["File_Format"],
        md_schema=RUN_CONFIG["MotherDuck_Schema"],
        md_table=RUN_CONFIG["MotherDuck_Table"],
        chunk_size=5000
    )
    arrow_buffer.motherduck_insert(df)
