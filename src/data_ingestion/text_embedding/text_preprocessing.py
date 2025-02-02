
import yaml
import polars as pl
import os
import pickle
import torch
from pathlib import Path
from dagster import asset
from sentence_transformers import SentenceTransformer
from src.data_ingestion.mdutils import motherduck_setup

CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))
DATA_DUMP = Path(__file__).joinpath("..", "..", "..", "..", "data_dump").resolve()


@asset
def get_news_data() -> pl.LazyFrame:
    run_config = CONFIG["News_Data_Embeddings"]

    # Create MotherDuck query engine and get all the data needed from MotherDuck
    md = motherduck_setup.MotherDucking(database_name=run_config["Source_Database"], read_only=True)
    df = (
        motherduck_setup.md_read_table(
            duck_engine=md.duckdb_engine,
            md_schema=run_config["Source_Schema"],
            md_table=run_config["Source_Table"],
            keep_columns=run_config["Source_Columns"]
        )
        .with_columns(
            pl.concat_str([pl.col("NewsHeadline"), pl.col("ShortDescription")], separator= " ").alias("NewsDetails")
        )
        .select(pl.exclude("NewsHeadline", "ShortDescription"))
        .lazy()
    )

    return df


@asset
def text_embedding(get_news_data) -> list[dict]:
    run_config = CONFIG["News_Data_Embeddings"]
    df = get_news_data.collect()[:-run_config["Holdout_Rows"]]

    # Split the data into different lists
    text_list = df.select("NewsDetails").to_series().to_list()
    ids = df.select("Id").to_series().to_list()
    news_dates = df.select("NewsDate").to_series().to_list()
    news_categories = df.select("NewsCategory").to_series().to_list()

    # Check if there's any cached vectors laying around. If there is one, use that instead of re-embedding
    cached_data = DATA_DUMP.joinpath(run_config["Cache_File"]).resolve()
    if os.path.exists(cached_data):
        with open(cached_data, "rb") as file:
            pinecone_data = pickle.load(file)

    else:
        # Perform text embedding
        model = SentenceTransformer(model_name_or_path=run_config["Embedding_Model"], device=torch.device("mps"))
        embeddings = model.encode(text_list, show_progress_bar=True, batch_size=run_config["Embedding_Batch"])

        # Generate object and metadata for Pinecone
        pinecone_data = []
        for idx, embedding in enumerate(embeddings):
            temp_dict = {
                "Id": str(ids[idx]),
                "vector": embedding,
                "metadata": {"news_date": str(news_dates[idx].strftime("%Y-%m-%d")), "news_category": str(news_categories[idx])}
            }
            pinecone_data.append(temp_dict)

        with open(cached_data, "wb") as file:
            pickle.dump(pinecone_data, file)

    return pinecone_data
