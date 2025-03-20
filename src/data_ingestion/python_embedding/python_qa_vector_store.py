
from dagster import asset, AssetIn
from src.data_ingestion.pcutils import pinecone_setup, pinecone_load
from src.data_ingestion.python_embedding.python_qa_embedding import CONFIG


@asset(ins={"python_vectors": AssetIn(key="pythong_string_embedding")})
def python_vector_store(python_vectors: list[dict]) -> None:
    # Setup Pinecone engine and setup Pinecone database
    pc = pinecone_setup.PineconeInstance(
        index_name=CONFIG["Pinecone_Index"],
        dimension=CONFIG["Model_Dimension"],
        cloud_provider=CONFIG["Cloud_Provider"],
        cloud_region=CONFIG["Cloud_Region"],
        rebuild_index=True
    )

    # Load vectors into Pinecone
    pc_index = pc.pinecone_setup()
    pc_buffer = pinecone_load.PineconeLoadingBuffer(pc_index=pc_index, embeddings=python_vectors)
    pc_buffer.pinecone_upsert()
