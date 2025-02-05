
import os
from langchain import hub
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from src.rag_pipeline import retriever
from src.data_ingestion.pcutils import pinecone_setup
from src.data_ingestion.mdutils import motherduck_setup

OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")


class RagInstance:

    def __init__(
        self, llm_model_name: str, llm_repo_name: str, reranker_model: str,
        pinecone_index_name: str, text_encoder_model: str,
        md_database_name: str, md_schema_name: str, md_table_name: str
    ):
        # Assign user provided parameters
        self.llm_model_name = llm_model_name
        self.llm_repo_name = llm_repo_name
        self.pinecone_index_name = pinecone_index_name
        self.text_encoder_model = text_encoder_model
        self.md_database_name = md_database_name
        self.md_schema_name = md_schema_name
        self.md_table_name = md_table_name
        self.reranker_model = reranker_model

        # Create instances of engines and APIs
        self.llm_model = ChatOpenAI(model=self.llm_model_name, api_key=OPEN_AI_KEY, seed=3825)
        self.pinecone_instance = pinecone_setup.PineconeInstance(self.pinecone_index_name)
        self.pinecone_index = self.pinecone_instance.pinecone_setup(index_setup=False)
        self.motherduck_instance = motherduck_setup.MotherDucking(self.md_database_name, read_only=True)

        # Create an instance for template prompt, retriever, and encoder
        self.template_prompt = hub.pull(self.llm_repo_name)
        self.text_encoder = retriever.TextEncoder(self.text_encoder_model)
        self.context_retriever = retriever.ContextRetriever(self.pinecone_instance, self.pinecone_index, self.text_encoder)

        # Create an instance of reranker for reranking vector database results
        self.reranker = CrossEncoder(self.reranker_model)

    def _generic_response(self, question: str, context: str) -> str:
        messages = self.template_prompt.invoke({"question": question, "context": context})
        response = self.llm_model.invoke(messages)
        return response.content

    def vanilla_llm(self, question: str) -> str:
        return self.llm_model.invoke(question).content

    def basic_rag(self, question: str) -> str:
        # Using the question, create a list of context to pass into the LLM
        rag_context = self.context_retriever.retrieve_sentences(
            question=question,
            md_database=self.md_database_name,
            md_shema=self.md_schema_name,
            md_table=self.md_table_name
        )

        # Using the question and context, plug them into a RAG chain and return an answer
        return self._generic_response(question, rag_context)

    def hyde_rag_implementation(self, question: str) -> str:
        # Using the question provided, get a hypothetical context from LLM
        question_prompt = f"Write me a short description that might answer: {question}"
        llm_context = self.vanilla_llm(question_prompt)

        # Embed the new context and perform similarity search through vector database
        rag_context = self.context_retriever.retrieve_sentences(
            question=llm_context,
            md_database=self.md_database_name,
            md_shema=self.md_schema_name,
            md_table=self.md_table_name
        )

        # Finally, using the new documents, plug them into a RAG chain and return an answer
        return self._generic_response(question, rag_context)

    def reranking_rag_implementation(self, question: str) -> str:
        # Using the question provided, query the top 10 responses from vector database
        context_results = self.context_retriever.retrieve_sentences(
            question=question,
            md_database=self.md_database_name,
            md_shema=self.md_schema_name,
            md_table=self.md_table_name,
            top_k=10
        ).split("\n\n")

        # Prepare pairs for reranking: each pair is (question, candidate passage)
        rerank_pairs = [(question, passage) for passage in context_results]
        scores = self.reranker.predict(rerank_pairs).tolist()
        sorted_candidates = [
            passage for _, passage in sorted(zip(scores, context_results), key=lambda pair: pair[0], reverse=True)
        ]

        # Create RAG context and plug them into a RAG chain and return an answer
        rag_context = "\n\n".join(sorted_candidates[:5])
        return self._generic_response(question, rag_context)
