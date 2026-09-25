from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore


class SemanticRetriever:
    """
    Retrieve document chunks related to a user question.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        number_of_results: int = 5
    ) -> list[dict]:
        """
        Convert a question into an embedding and retrieve
        similar chunks from ChromaDB.
        """

        question = question.strip()

        if not question:
            raise ValueError(
                "Please enter a question."
            )

        # Generate one embedding for the question
        query_embedding = (
            self.embedding_model.embed_texts(
                [question]
            )[0]
        )

        # Search ChromaDB
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            number_of_results=number_of_results
        )

        retrieved_chunks = []

        documents = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]
        distances = search_results["distances"][0]

        for rank, (
            document,
            metadata,
            distance
        ) in enumerate(
            zip(documents, metadatas, distances),
            start=1
        ):
            retrieved_chunks.append(
                {
                    "rank": rank,
                    "text": document,
                    "source": metadata["source"],
                    "page_number": metadata["page_number"],
                    "chunk_id": metadata["chunk_id"],
                    "token_count": metadata["token_count"],
                    "distance": float(distance)
                }
            )

        return retrieved_chunks