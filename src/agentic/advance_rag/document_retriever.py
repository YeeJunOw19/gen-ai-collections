
import yaml
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
from src.data_ingestion.pcutils import pinecone_setup
from src.data_ingestion.mdutils import motherduck_setup

CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["Advanced_Agentic_Configurations"]
DEVICE = "cuda" if torch.cuda.is_available() else "mps"


def index_retriever(question: str) -> list[int]:
    # Create an instance of Pinecone
    pc = pinecone_setup.PineconeInstance(index_name=CONFIG["Pinecone_Index"])

    # Create an instance of encoder and query from Pinecone
    model = SentenceTransformer(CONFIG["Embedding_Model"], device=DEVICE)
    embeddings = model.encode(question, show_progress_bar=False).tolist()
    vectors = pc.query_pinecone(
        pinecone_index=pc.pinecone_setup(False),
        vector_embedding=embeddings,
        top_n=CONFIG["Top_N"]
    )

    # Get the IDs and return that
    ids = [x["id"] for x in vectors]
    return ids


def document_retrieve_rerank(indexes: list[str], question: str) -> str:
    # Create the query to used for Python Q&A retrieval
    query_string = (
        f"""
        SELECT CONCAT(QuestionAsked, ' ', OutputAnswer) AS 'PythonString'
        FROM "gen-ai".PythonCodeInstruct
        WHERE DataSplit = 'Training' AND Id IN ({(', '.join(indexes))})
        """
    )

    # Create a MotherDuck instance and retrieve the documents
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], True)
    docs = (
        motherduck_setup.md_read_table(
            md.duckdb_engine, md_schema=None, md_table=None, keep_columns=None, custom_query=query_string
        )
        .collect()
        .to_series()
        .to_list()
    )

    # Rerank the retrieved documents based on relevance
    rerank_model = CrossEncoder(CONFIG["Reranker_Model"], device=DEVICE)
    rerank_pairs = [(question, doc) for doc in docs]
    scores = rerank_model.predict(rerank_pairs).tolist()

    # Rerank all the documents and return the ranked documents for RAG
    sorted_candidates = [passage for _, passage in sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)]
    rag_context = "\n\n".join(sorted_candidates)

    return rag_context
