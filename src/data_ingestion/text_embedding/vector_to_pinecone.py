
from pinecone import Pinecone
from dagster import asset, AssetIn
from src.data_ingestion.pcutils import pinecone_setup, pinecone_load
from src.data_ingestion.text_embedding.text_preprocessing import CONFIG


@asset
def prep_pinecone() -> dict:
    """
    Dagster asset to set up and prepare Pinecone server and database for vector embeddings ingestion.

    :return: None
    """

    run_config = CONFIG["News_Data_Embeddings"]

    # Setup Pinecone engine and setup Pinecone database
    pc = pinecone_setup.PineconeInstance(
        index_name=run_config["Pinecone_Index"],
        dimension=run_config["Model_Dimension"],
        cloud_provider=run_config["Cloud_Provider"],
        cloud_region=run_config["Cloud_Region"],
        rebuild_index=True
    )

    return run_config


@asset(ins={"vector_data": AssetIn(key="text_embedding"), "run_config": AssetIn(key="prep_pinecone")})
def load_data_to_pinecone(vector_data: list[dict], run_config: dict) -> None:
    """
    Dagster asset to load vector embeddings in batches into Pinecone.

    :param vector_data: A list of dictionaries describing vector embeddings and their metadata
    :param run_config: YAML file configurations
    :return: None
    """

    # Prepare data for Pinecone ingestion
    pc = pinecone_setup.PineconeInstance(
        index_name=run_config["Pinecone_Index"],
        dimension=run_config["Model_Dimension"],
        cloud_provider=run_config["Cloud_Provider"],
        cloud_region=run_config["Cloud_Region"],
        rebuild_index=False
    )
    pc_index = pc.pinecone_setup()
    pc_buffer = pinecone_load.PineconeLoadingBuffer(pc_index=pc_index, embeddings=vector_data)
    pc_buffer.pinecone_upsert()
