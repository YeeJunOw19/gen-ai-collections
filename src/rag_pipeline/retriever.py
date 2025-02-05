
from sentence_transformers import SentenceTransformer
from src.data_ingestion.mdutils import motherduck_setup


class TextEncoder:

    def __init__(self, model_name: str, batch_size: int = 10_000):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = self._model_instance()

    def _model_instance(self):
        try:
            model = SentenceTransformer(model_name_or_path=self.model_name, device="mps")
        except:
            raise Exception(f"Model {self.model_name} not found.")
        return model

    def encoding(self, sentences: str) -> list:
        embeddings = self.model.encode(sentences, show_progress_bar=True, batch_size=self.batch_size)
        return embeddings.tolist()


class ContextRetriever:

    def __init__(self, pinecone_instance, pinecone_index, encoder: TextEncoder):
        self.pinecone_instance = pinecone_instance
        self.pinecone_index = pinecone_index
        self.encoder = encoder

    def retrieve_sentences(
        self, question: str | None, md_database: str, md_shema: str, md_table: str,
        top_k: int = 5, embeddings: list | None = None
    ) -> str:
        if embeddings is None:
            # Get a list of top searches from Pinecone
            embeddings = self.encoder.encoding(question)

        results = self.pinecone_instance.query_pinecone(
            pinecone_index=self.pinecone_index,
            vector_embedding=embeddings,
            top_n=top_k
        )

        # Get Ids out of the results from above and query MotherDuck
        ids = [x["id"] for x in results]
        query_string = (
            f"""
            SELECT concat(NewsHeadline, ' ', ShortDescription) AS 'DocText'
            FROM "gen-ai".RawNewsCategory
            WHERE Id IN ({(', '.join(ids))})
            """
        )

        md = motherduck_setup.MotherDucking(md_database, True)
        query_df = motherduck_setup.md_read_table(
            duck_engine=md.duckdb_engine,
            md_schema=md_shema,
            md_table=md_table,
            keep_columns=None,
            custom_query=query_string
        )

        # If the dataset is empty, return an empty string
        if query_df.collect().is_empty():
            return ""

        else:
            contexts = query_df.collect().to_series().to_list()
            input_context = "\n\n".join(contexts)
            return input_context
