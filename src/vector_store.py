import hashlib

import chromadb
import numpy as np


class VectorStore:
    """
    Store and retrieve document chunk embeddings using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "research_papers"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=None
            )
        )

    def _create_chunk_id(
        self,
        chunk: dict
    ) -> str:
        """
        Generate a repeatable unique ID for a chunk.
        """

        id_content = (
            f"{chunk['source']}:"
            f"{chunk['page_number']}:"
            f"{chunk['chunk_id']}:"
            f"{chunk['text']}"
        )

        return hashlib.sha256(
            id_content.encode("utf-8")
        ).hexdigest()

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: np.ndarray
    ) -> int:
        """
        Store chunks, embeddings and metadata in ChromaDB.
        """

        if not chunks:
            raise ValueError(
                "No chunks were supplied."
            )

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks and embeddings "
                "must be equal."
            )

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(
                self._create_chunk_id(chunk)
            )

            documents.append(
                chunk["text"]
            )

            metadatas.append(
                {
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "token_count": chunk["token_count"]
                }
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

        return len(chunks)

    def count(self) -> int:
        """
        Return the number of stored chunks.
        """

        return self.collection.count()

    def search(
        self,
        query_embedding: np.ndarray,
        number_of_results: int = 5
    ) -> dict:
        """
        Search for chunks similar to a query embedding.
        """

        stored_count = self.count()

        if stored_count == 0:
            raise ValueError(
                "The vector database is empty."
            )

        number_of_results = min(
            number_of_results,
            stored_count
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=number_of_results,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return results