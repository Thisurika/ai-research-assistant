import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Convert text into semantic embedding vectors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: list[str]
    ) -> np.ndarray:
        """
        Generate one embedding for every supplied text.
        """

        if not texts:
            return np.empty((0, 0))

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings

    def embed_chunks(
        self,
        chunks: list[dict]
    ) -> np.ndarray:
        """
        Generate one embedding for every document chunk.
        """

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        return self.embed_texts(chunk_texts)