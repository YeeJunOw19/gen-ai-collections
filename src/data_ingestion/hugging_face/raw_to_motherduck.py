
import polars as pl
import random
from dagster import asset, AssetIn
from src.data_ingestion.hugging_face.raw_ingestion import CONFIG
from src.data_ingestion.mdutils import motherduck_setup, motherduck_load


def _prep_motherduck(run_config: dict) -> None:
    """
    This Dagster asset runs the MotherDuck setup scripts to prepare MotherDuck database for downstream data ingestion.

    :return: None
    """

    # Configurate MotherDuck setup and run setup scripts to setup MotherDuck database
    table_schema = run_config["File_Format"]
    md = motherduck_setup.MotherDucking(run_config["MotherDuck_Database"])

    motherduck_setup.md_table_setup(
        duck_engine=md.duckdb_engine,
        schema_name = run_config["MotherDuck_Schema"],
        table_name = run_config["MotherDuck_Table"],
        table_schema = table_schema,
        column_key="New_Name",
        rebuild_table=True
    )


def _load_data_to_motherduck(df: pl.LazyFrame, run_config: dict) -> None:
    """
    This Dagster asset takes in an upstream Polars LazyFrame object and sets up MotherDuck engine, and load the data
    into MotherDuck by chunk. This asset also depends on an upstream Dagster asset that prepares MotherDuck database.

    :param df: Polars LazyFrame object from upstream Dagster asset
    :param run_config: A dictionary of configurations to be used in this Dagster run
    :return: None
    """

    # Configurate MotherDuck engines
    md = motherduck_setup.MotherDucking(run_config["MotherDuck_Database"])

    # Setup arrow buffer and load data into MotherDuck by chunk
    arrow_buffer = motherduck_load.ArrowTableLoadingBuffer(
        duck_conn=md.duckdb_conn,
        data_schema=run_config["File_Format"],
        md_schema=run_config["MotherDuck_Schema"],
        md_table=run_config["MotherDuck_Table"],
        chunk_size=5000
    )
    arrow_buffer.motherduck_insert(df)


@asset
def prep_md_news_dataset() -> None:
    # Set configuration and run code to prep tables in MotherDuck
    config = CONFIG["News_Dataset_Config"]
    _prep_motherduck(config)


@asset
def prep_md_qa_dataset() -> None:
    # Set configuration and run code to prep tables in MotherDuck
    config = CONFIG["GSM8K_Dataset_Config"]
    _prep_motherduck(config)

@asset
def prep_python_dataset() -> None:
    # Set configuration and run code to prep tables in MotherDuck
    config = CONFIG["Python_Dataset_Modeling_Config"]
    _prep_motherduck(config)


@asset(
    deps=[prep_md_news_dataset],
    ins={"df": AssetIn(key="get_news_dataset")}
)
def load_news_dataset(df: pl.LazyFrame) -> None:
    # Set configuration and load data to MotherDuck
    config = CONFIG["News_Dataset_Config"]
    _load_data_to_motherduck(df, config)


@asset(
    deps=[prep_md_qa_dataset],
    ins={"df": AssetIn(key="get_qa_dataset")}
)
def load_qa_dataset(df: pl.LazyFrame) -> None:
    # Set configuration and load data to MotherDuck
    config = CONFIG["GSM8K_Dataset_Config"]
    _load_data_to_motherduck(df, config)


@asset(
    deps=[prep_python_dataset],
    ins={"df": AssetIn(key="get_python_dataset")}
)
def load_python_dataset(df: pl.LazyFrame) -> None:
    # Set configuration and load data to MotherDuck
    config = CONFIG["Python_Dataset_Modeling_Config"]

    # From the dataset, perform data modeling to fit the data into the proper format
    df = (
        df
        .collect()
        .select("Message", "MessageType", "QueryId")
        .pivot("MessageType", index="QueryId", values="Message")
        .with_columns((pl.col("QueryId") + 1).alias("QueryId"))
    )
    df.columns = ["Id", "QuestionAsked", "QuestionInput", "OutputAnswer"]

    # Randomly assign different IDs to training, validation, and testing sets
    ids = df.select("Id").to_series().to_list()
    random.shuffle(ids)

    n = len(ids)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)

    train = ids[:train_end]
    val = ids[train_end:val_end]
    test = ids[val_end:]

    # Create a new column for data splitting
    df = (
        df
        .with_columns(
            pl.when(pl.col("Id").is_in(train)).then(pl.lit("Training"))
            .when(pl.col("Id").is_in(val)).then(pl.lit("Validation"))
            .otherwise(pl.lit("Testing"))
            .alias("DataSplit")
        )
        .lazy()
    )

    # Pass the configuration and dataframe to load to MotherDuck
    _load_data_to_motherduck(df, config)
