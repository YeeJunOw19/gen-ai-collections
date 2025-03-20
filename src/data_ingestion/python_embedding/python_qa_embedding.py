
import yaml
from dagster import asset
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.data_ingestion.mdutils import motherduck_setup
from src.data_ingestion.hugging_face import raw_to_motherduck

CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["Python_Question_Answer_Embedding"]


@asset(deps=[raw_to_motherduck.load_python_dataset])
def pythong_string_embedding() -> list[dict]:
    # Create an instance of MotherDuck engine
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], True)

    # Query the dataset from MotherDuck
    query_string = (
        """
        SELECT Id, CONCAT(QuestionAsked, ' ', OutputAnswer) AS 'PythonString'
        FROM "gen-ai".PythonCodeInstruct
        WHERE DataSplit = 'Training' AND Length(CONCAT(QuestionAsked, OutputAnswer)) < 300
        """
    )
    df = motherduck_setup.md_read_table(
        duck_engine=md.duckdb_engine, md_schema=None, md_table=None, keep_columns=None, custom_query=query_string
    ).collect()

    # Create embeddings for each Python question and answer pair
    python_pairs = df.select("PythonString").to_series().to_list()
    model = SentenceTransformer(model_name_or_path=CONFIG["Embedding_Model"], device="mps")
    embeddings = model.encode(python_pairs, show_progress_bar=True, batch_size=CONFIG["Embedding_Batch"])

    # Create metadata for vector database storage
    vectors = []
    python_ids = df.select("Id").to_series().to_list()
    for idx, value in enumerate(embeddings):
        temp_dict = {"Id": str(python_ids[idx]), "vector": value, "metadata": {"info": "Python Q&A"}}
        vectors.append(temp_dict)

    return vectors
