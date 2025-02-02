
from pinecone import Pinecone, ServerlessSpec
import os

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


class PineconeInstance:

    def __init__(
        self, index_name: str, dimension: int, cloud_provider: str, cloud_region: str,
        metric: str = "cosine", rebuild_index: bool = False
    ):
        self.index_name = index_name
        self.dimension = dimension
        self.cloud_provider = cloud_provider
        self.cloud_region = cloud_region
        self.metric = metric
        self.rebuild_index = rebuild_index
        self.pinecone_engine = Pinecone(api_key=PINECONE_API_KEY)
        self.pinecone_index = self._pinecone_setup()

    def query_pinecone(self, vector_embedding, top_n: int = 5) -> dict:
        """
        This function takes in an embedded vector and use that vector to query Pinecone database. The query will return
        the top 5 closes vectors to the embedded vector, by default.

        :param vector_embedding: Embedded vector generated from an text encoder
        :param top_n: Specify the top n results to return. Default is 5
        :return: A dictionary with the results and metadata from Pinecone
        """
        # Query Pinecone database using the vector provided
        results = self.pinecone_index.query(vector=vector_embedding, top_n=top_n, include_metadata=True)
        if results["matches"]:
            return results["matches"]
        else:
            return {}

    def _pinecone_setup(self) -> Pinecone.Index:
        """
        This function creates a Pinecone index if it doesn't already exist. The user can also specify whether or not
        to rebuild the index. The function will return a Pinecone index if it exists.

        :return: Pinecone index
        """
        # Get a list of indexes from Pinecone. Delete existing index if specified
        indexes = self.pinecone_engine.list_indexes().names()
        if self.rebuild_index:
            if self.index_name in indexes:
                self.pinecone_engine.delete_index(self.index_name)

        else:
            if not self.index_name in indexes:
                self.pinecone_engine.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud_provider, region=self.cloud_region)
                )

        return self.pinecone_engine.Index(name=self.index_name)
