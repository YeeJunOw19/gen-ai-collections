
from pinecone import Pinecone

class PineconeLoadingBuffer:

    def __init__(
        self,
        pc_index: Pinecone.Index,
        embeddings: list[dict],
        chink_size: int = 1_000,
        delete_before_load: bool = True
    ):
        self.pc_index = pc_index
        self.embeddings = embeddings
        self.chunk_size = chink_size
        self.delete_before_load = delete_before_load
        self.loading_embeddings = self._prep_loading_list()

    def pinecone_upsert(self) -> None:
        """
        This function takes in a list of embeddings and metadata, and loads them into Pinecone. The default loading
        batch size is 1,000.

        :return: None
        """

        # Delete all the existing embeddings before loading
        if self.delete_before_load:
            self.pc_index.delete(delete_all=True)

        # Load embeddings in chunk
        for i in range(0, len(self.loading_embeddings), self.chunk_size):
            batch_data = self.loading_embeddings[i:i + self.chunk_size]
            self.pc_index.upsert(vectors=batch_data)

    def _prep_loading_list(self) -> list[dict]:
        """
        This function takes in a list of dictionaries containing embeddings and metadata, and formats them into
        Pinecone dictionaries.

        :return: A list of dictionaries formatted to Pinecone format
        """

        upsert_list = []
        for embedding in self.embeddings:
            upsert_list.append({
                "id": embedding["Id"],
                "values": embedding["vector"],
                "metadata": embedding["metadata"]
            })

        return upsert_list
